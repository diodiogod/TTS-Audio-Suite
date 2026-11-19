"""Original four-head GDN, with independent request-local inference state.

TTS Audio Suite Patch: this file is rewritten. Upstream runs fla-core's 64-frame chunked kernels (the trainer's
arithmetic) and replays each request's unfinished chunk on every forward. Here every pass runs the per-token recurrence
of the reference implementation (``zonos3.reference.lm.Zonos3LM._gdn``) through one Triton kernel
(zonos3_sgl.kernel.triton.gated_delta), so a slot's persistent state is always at the request's last row: the causal
convolution's last three input rows and the float32 recurrent state. Accent adapters are not part of this copy.

The small mixer is replicated across TP ranks. All persistent model tensors use BaseOP's strict checkpoint loader.
Caches start with an underscore so they cannot masquerade as checkpoint weights.

CUDA graphs: decode graphs (``_forward_decode_static``) take one new row per padded request; prefill graphs
(zonos3_sgl.engine.prefill_graph) pad each fresh request to its complete 64-row chunks plus 64 rows with zero rows
after the real ones (``_forward_prefill_static``). In both, the requests' slots and real row counts are read from
tensors, so every shape depends on the padded layout only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from zonos3_sgl.core import get_global_ctx
from zonos3_sgl.kernel.triton.gated_delta import gated_delta
from zonos3_sgl.layers.base import BaseOP
from zonos3_sgl.utils.torch_utils import make_device_tensor

CHUNK = 64  # prefill graphs pad requests to multiples of this many rows
HEADS, HEAD_DIM, KERNEL = 4, 128, 4
INNER = HEADS * HEAD_DIM
CHANNELS = 3 * INNER


@dataclass(frozen=True)
class PrefillLayout:
    """Static GDN tables of a padded prefill: request i owns ``lengths[i] = CHUNK * (whole[i] + 1)`` rows of the batch
    from ``starts[i]``, its real rows first."""

    lengths: tuple[int, ...]
    starts: torch.Tensor  # [b] long

    @classmethod
    def build(cls, whole: list[int], device: torch.device) -> PrefillLayout:
        lengths = tuple(CHUNK * (w + 1) for w in whole)
        starts = [sum(lengths[:i]) for i in range(len(lengths))]
        return cls(lengths=lengths, starts=make_device_tensor(starts, torch.long, device))

    def tensors(self) -> list[torch.Tensor]:
        """The tensors a captured graph keeps pointers to."""
        return [self.starts]


@dataclass
class PrefillInputs:
    """A padded prefill's GDN inputs (TTSBatch.gdn_prefill): the layout and the graph's per-replay buffers."""

    layout: PrefillLayout
    slots: torch.Tensor  # [b] long: the requests' table slots
    lengths: torch.Tensor  # [b] long: the requests' real rows


class _Weight(BaseOP):
    def __init__(self, shape):
        self.weight = torch.empty(shape)


class OriginalGDN(BaseOP):
    def __init__(self, config, layer_id):
        self._layer_id = layer_id
        self._hidden = config.hidden_size
        self.in_proj = _Weight((4 * INNER + 2 * HEADS, config.hidden_size))
        self.conv = _Weight((CHANNELS, 1, KERNEL))
        self.out_weight = torch.empty(HEAD_DIM)
        self.A_log = torch.empty(HEADS, dtype=torch.float32)
        self.dt_bias = torch.empty(HEADS, dtype=torch.float32)
        self.wo = _Weight((config.hidden_size, INNER))
        self._capacity = 0
        # CUDA-graph decode support (see enable_graph_decode)
        self._static_mode = False
        self._graph_slots: torch.Tensor | None = None
        self._static_rows: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def allocate_cache(self, capacity, device, dtype):
        if self._capacity:
            raise RuntimeError("GDN cache already allocated")
        # Per slot: the convolution's last KERNEL - 1 input rows (oldest first) and the recurrent state [head, value, key].
        self._conv_state = torch.zeros((capacity, CHANNELS, KERNEL - 1), device=device, dtype=dtype)
        self._recurrent_state = torch.zeros((capacity, HEADS, HEAD_DIM, HEAD_DIM), device=device, dtype=torch.float32)
        self._owners = [None] * capacity
        self._lengths = [0] * capacity
        self._capacity = capacity

    # ------------------------------------------------------------------ CUDA graphs
    @staticmethod
    def static_tables(device: torch.device, bs_list) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Per batch size: (first row, row count) of each padded request's one new row. Built once by the engine and
        shared by every GDN layer; captured graphs keep pointers to them."""
        return {
            bs: (
                torch.arange(bs, device=device, dtype=torch.long),
                torch.ones(bs, device=device, dtype=torch.long),
            )
            for bs in bs_list
        }

    def enable_graph_decode(self, slots: torch.Tensor, tables: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> None:
        """``slots`` is the engine's static [max_bs] int64 buffer of table indices for the padded decode batch (the
        engine fills it before every capture / replay); ``tables`` comes from ``static_tables``."""
        self._graph_slots = slots
        self._static_rows.update(tables)

    def set_static_mode(self, enabled: bool) -> None:
        """While set, decode batches run the fixed-shape path (used for graph warm-up and capture)."""
        self._static_mode = bool(enabled)

    def note_graph_forward(self, reqs) -> None:
        """Host-side mirrors after a replayed decode step or prefill (the graph advanced the device state)."""
        for r in reqs:
            self._owners[r.table_idx] = (r.uid, r.is_cfg_uncond)
            self._lengths[r.table_idx] = r.device_len

    # ------------------------------------------------------------------ forward
    def forward(self, x):
        batch = get_global_ctx().batch
        if batch.is_decode and self._static_mode:
            return self._forward_decode_static(x, batch.padded_size)
        if batch.gdn_prefill is not None:
            return self._forward_prefill_static(x, batch.gdn_prefill)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("GDN ragged path cannot be captured; graphs use the static decode / prefill paths")
        reqs = batch.reqs
        if len(batch.padded_reqs) != len(reqs):
            raise RuntimeError("GDN eager path does not accept dummy graph-padding requests")
        slots = [r.table_idx for r in reqs]
        lengths = [r.extend_len for r in reqs]
        if (
            not self._capacity
            or len(set(slots)) != len(slots)
            or any(s < 0 or s >= self._capacity for s in slots)
            or any(n <= 0 for n in lengths)
            or sum(lengths) != x.shape[0]
        ):
            raise RuntimeError("Invalid GDN request slots or packed lengths")
        if batch.is_decode and any(n != 1 for n in lengths):
            raise RuntimeError("GDN decode expects one new token per request")
        owners = [(r.uid, r.is_cfg_uncond) for r in reqs]
        # Validate the entire batch before changing any cache. A global KV-only
        # prefix hit is deliberately rejected: it has no matching recurrent state.
        for r, owner in zip(reqs, owners):
            s = r.table_idx
            if r.cached_len and (self._owners[s] != owner or self._lengths[s] != r.cached_len):
                raise RuntimeError(
                    f"GDN layer {self._layer_id}: missing/stale state at slot {s}; "
                    f"expected {(owner, r.cached_len)}, "
                    f"found {(self._owners[s], self._lengths[s])}"
                )
        fresh = [r.table_idx for r in reqs if r.cached_len == 0]
        if fresh:
            index = make_device_tensor(fresh, torch.long, x.device)
            self._conv_state.index_fill_(0, index, 0)
            self._recurrent_state.index_fill_(0, index, 0)
        starts = [sum(lengths[:i]) for i in range(len(lengths))]
        packed = make_device_tensor(slots + starts + lengths, torch.long, x.device).view(3, len(reqs))
        qkv, z, a, b = self._project(x)
        mixed = []
        for i, (start, n) in enumerate(zip(starts, lengths)):
            sequence = torch.cat((self._conv_state[slots[i]], qkv[start : start + n].T), -1)
            mixed.append(self._convolve(sequence))
            self._conv_state[slots[i]] = sequence[:, -(KERNEL - 1) :]
        y = self._mix(torch.cat(mixed), z, a, b, packed[1], packed[2], packed[0])
        for r, owner in zip(reqs, owners):
            self._owners[r.table_idx] = owner
            self._lengths[r.table_idx] = r.device_len
        return y

    def _forward_decode_static(self, x, bs):
        """One new row per padded request; every shape depends on ``bs`` only."""
        if self._graph_slots is None or bs not in self._static_rows:
            raise RuntimeError(f"GDN static decode is not prepared for batch size {bs}")
        slots = self._graph_slots[:bs]
        starts, lengths = self._static_rows[bs]
        qkv, z, a, b = self._project(x)
        sequence = torch.cat((self._conv_state.index_select(0, slots), qkv.unsqueeze(-1)), -1)  # [bs, channels, 4]
        conv = F.silu(F.conv1d(sequence, self.conv.weight, groups=CHANNELS).squeeze(-1))
        self._conv_state.index_copy_(0, slots, sequence[:, :, 1:])
        return self._mix(conv, z, a, b, starts, lengths, slots)

    def _forward_prefill_static(self, x, inputs: PrefillInputs):
        """A padded prefill of fresh requests (PrefillLayout); the rows after a request's real rows are zero on the way
        in and get zero recurrence outputs. Every shape depends on the layout only."""
        layout, slots = inputs.layout, inputs.slots
        self._recurrent_state.index_fill_(0, slots, 0)
        qkv, z, a, b = self._project(x)
        history = qkv.new_zeros((CHANNELS, KERNEL - 1))
        last = torch.arange(KERNEL - 1, device=x.device)
        mixed, states, start = [], [], 0
        for i, n in enumerate(layout.lengths):
            sequence = torch.cat((history, qkv[start : start + n].T), -1)
            mixed.append(self._convolve(sequence))
            # the request's last three real input rows: sequence columns [real, real + 3)
            states.append(sequence.index_select(1, inputs.lengths[i] + last))
            start += n
        self._conv_state.index_copy_(0, slots, torch.stack(states))
        return self._mix(torch.cat(mixed), z, a, b, layout.starts, inputs.lengths, slots)

    def _project(self, x):
        return F.linear(x, self.in_proj.weight).split((CHANNELS, INNER, HEADS, HEADS), -1)

    def _convolve(self, sequence):
        """Depthwise causal cross-correlation of [channels, rows + 3] (weight[0] is oldest) -> [rows, channels]."""
        return F.silu(F.conv1d(sequence.unsqueeze(0), self.conv.weight, groups=CHANNELS).squeeze(0).T)

    def _mix(self, conv, z, a, b, starts, lengths, slots):
        """Rows after the convolution -> the layer output; advances the recurrent state of ``slots``."""
        rows = conv.shape[0]
        q, k, v = [part.reshape(rows, HEADS, HEAD_DIM).float() for part in conv.split(INNER, -1)]
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        decay = torch.exp(-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float()))
        beta = b.float().sigmoid()
        recurrent = gated_delta(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            decay.contiguous(),
            beta.contiguous(),
            self._recurrent_state,
            starts,
            lengths,
            slots,
            HEAD_DIM**-0.5,
        ).to(conv.dtype)
        # FLA's rms_norm_gated: fixed eps=1e-5 and a swish gate, per head.
        normalized = F.rms_norm(recurrent, (HEAD_DIM,), weight=self.out_weight, eps=1e-5)
        gated = (normalized.float() * F.silu(z.reshape(rows, HEADS, HEAD_DIM).float())).to(conv.dtype)
        return F.linear(gated.reshape(rows, INNER), self.wo.weight)
