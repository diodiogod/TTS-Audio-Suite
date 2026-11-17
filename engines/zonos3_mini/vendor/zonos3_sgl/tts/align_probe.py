"""Alignment-head probe for TTS decode: read (and optionally window-mask) text attention, and stop a request early.

Two kinds of observer:

- ``on_step(probe, step, pos, m)`` (default): the summed text distribution of ``probe.heads`` reduced on the GPU to its
  argmax ``pos`` and centroid ``m``. The readout runs batched for every probed request of a decode step, needs no host
  sync inside the forward and is captured into the decode CUDA graphs when the engine was built with
  ``align_probe_max_text > 0``. The scheduler copies the results to the host together with the sampled tokens and calls
  the observer while it processes that step; ``probe.stop`` / ``probe.mask`` set there take effect from the next
  scheduled step (with overlap scheduling the step already in flight still runs; its token is dropped on stop).
- ``on_attention(probe, step, text_attn, text_mass)`` (``all_heads=True``, head discovery): every head's text
  distribution and text mass, called synchronously inside the forward. Eager only.

A masked head (``probe.mask``) has its attention output recomputed over the request's keys with the text keys outside
its window at -inf (ACI hard mask; non-text keys are never masked); its readout is the window-renormalized
distribution. Masked requests run inside the graphs too, up to ``align_probe_max_masked`` per batch and
``align_probe_max_keys`` keys per request; beyond that (or with an all-heads probe) the batch runs eager.
``TTSLLM.generate`` refuses probes with speaker CFG or a non-FA backend.

Positions are indices into the text span: 0 is the BOS row, ``text_len - 1`` the EOS row (``locate_text_span``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch
from zonos3.tts.probe import Head, attention_layers, clip_window, probe_weight

if TYPE_CHECKING:
    from zonos3_sgl.attention.base import BaseAttnBackend
    from zonos3_sgl.core import TTSBatch

_OPEN = float(1 << 30)  # default window upper bound (float32-exact)


class AlignObserver(Protocol):
    def on_step(self, probe: "AlignProbe", step: int, pos: int, m: float) -> None: ...


class AttentionObserver(Protocol):
    def on_attention(
        self,
        probe: "AlignProbe",
        step: int,
        text_attn: dict[Head, torch.Tensor],  # fp32 [text_len], sums to 1 over the text span
        text_mass: dict[Head, torch.Tensor],  # scalar softmax mass on the text keys
    ) -> None: ...


@dataclass(eq=False)
class AlignProbe:
    heads: tuple[Head, ...]
    observer: AlignObserver | AttentionObserver
    all_heads: bool = False  # discovery: on_attention with every head of every attention layer (eager)
    mask: dict[Head, tuple[int, int]] | None = None  # per head inclusive text-index window; None = unmasked
    text_start: int = -1  # filled by the scheduler (prompt row of BOS)
    text_len: int = 0  # filled by the scheduler (BOS..EOS rows inclusive)
    steps: int = 0  # decode forwards observed
    stop: bool = False  # set by the observer; the request then finishes
    _weight: np.ndarray | None = field(default=None, repr=False)  # [n_layers, n_heads] head weights (cached)
    _attn: dict = field(default_factory=dict, repr=False)  # all_heads stash within one forward
    _mass: dict = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------------------------------- packed inputs
#
# Readout: one float32 row per decode row: [text_start, text_len, weight (L*H), lo (L*H), hi (L*H)], L = attention
# layers, H = q heads. Masked heads: one row per masked request: [batch row (-1 = unused), n_keys, text_start,
# text_len, flag (L*H), lo (L*H), hi (L*H)]. Defaults (length 1, zero weights / flags, open window) let a captured
# graph run on any batch.


def _width(lead: int, n_layers: int, n_heads: int) -> int:
    return lead + 3 * n_layers * n_heads


def _default_rows(n: int, lead: int, n_layers: int, n_heads: int) -> np.ndarray:
    nh = n_layers * n_heads
    rows = np.zeros((n, _width(lead, n_layers, n_heads)), dtype=np.float32)
    rows[:, lead - 1] = 1.0  # text_len (readout) / text_len (masked)
    if lead == 4:
        rows[:, 0] = -1.0  # unused masked slot
        rows[:, 1] = 1.0  # one key
    rows[:, lead + 2 * nh :] = _OPEN
    return rows


@dataclass(eq=False)
class ProbeInputs:
    """Per-forward probe inputs on the device; ``out`` [n, 2] receives (argmax, centroid) per row."""

    packed: torch.Tensor  # [n, readout width] float32
    lmax: int  # text keys read per row (static under graphs)
    out: torch.Tensor  # [n, 2] float32
    rows: list[int]  # decode rows carrying a reduced probe (empty during capture)
    masked: torch.Tensor | None = None  # [slots, masked width] float32
    kmax: int = 0  # keys read per masked request (static under graphs)
    # derived within one forward (recomputed at the first attention layer, so a captured graph recomputes them too)
    _d: dict = field(default_factory=dict, repr=False)


class ProbeStatic:
    """Static probe buffers for CUDA-graph decode (allocated by the graph runner when ``max_text > 0``)."""

    def __init__(
        self,
        max_bs: int,
        max_text: int,
        max_masked: int,
        max_keys: int,
        n_layers: int,
        n_heads: int,
        device: torch.device,
    ):
        self.max_text, self.max_masked, self.max_keys = int(max_text), int(max_masked), int(max_keys)
        self.packed = torch.from_numpy(_default_rows(max_bs, 2, n_layers, n_heads)).to(device)
        self.masked = torch.from_numpy(_default_rows(max(1, self.max_masked), 4, n_layers, n_heads)).to(device)
        self.out = torch.zeros((max_bs, 2), dtype=torch.float32, device=device)

    def inputs(self, n: int, rows: list[int], masked: bool) -> ProbeInputs:
        """Inputs of an n-row graph; ``masked`` selects the graph variant that also recomputes masked heads."""
        return ProbeInputs(
            self.packed[:n],
            self.max_text,
            self.out[:n],
            rows,
            self.masked if masked and self.max_masked > 0 else None,
            self.max_keys,
        )


def _probe_weight(probe: AlignProbe, layers: tuple[int, ...], n_heads: int) -> np.ndarray:
    if probe._weight is None:
        probe._weight = probe_weight(probe.heads, layers, n_heads)
    return probe._weight


def graph_compatible(batch: "TTSBatch", static: ProbeStatic | None) -> bool:
    """Whether the probes of a decode batch can run inside the captured graphs."""
    masked = 0
    for r in batch.reqs:
        p = r.align_probe
        if p is None:
            continue
        if static is None or p.all_heads or p.text_len > static.max_text:
            return False
        if p.mask is not None:
            masked += 1
            if masked > static.max_masked or r.device_len > static.max_keys:
                return False
    return True


def _windows(probe: AlignProbe, lo: np.ndarray, hi: np.ndarray, layers: tuple[int, ...], flag=None) -> None:
    for (layer, head), window in probe.mask.items():
        s = layers.index(layer)
        lo[s, head], hi[s, head] = clip_window(window, probe.text_len)
        if flag is not None:
            flag[s, head] = 1.0


def prepare(batch: "TTSBatch", static: ProbeStatic | None, n_heads: int, device: torch.device) -> None:
    """Set ``batch.probe`` (batched readout and masked heads) and ``batch.probe_legacy`` (all-heads rows, eager) for a
    decode batch. ``static`` = the graph buffers when this batch replays a graph (the packed rows are copied into
    them), else None (eager: fresh device tensors sized to the batch)."""
    batch.probe, batch.probe_legacy = None, None
    reduced, masked, legacy = [], [], []
    for i, r in enumerate(batch.reqs):
        p = r.align_probe
        if p is None:
            continue
        if p.text_len <= 0:
            raise RuntimeError("align probe has no text span")
        if p.all_heads:
            legacy.append(i)
            continue
        reduced.append(i)
        if p.mask is not None:
            masked.append(i)
    batch.probe_legacy = legacy or None
    if not reduced:
        return
    layers, nl = attention_layers(), len(attention_layers())
    nh = nl * n_heads
    n = batch.padded_size if static is not None else batch.size
    packed = _default_rows(n, 2, nl, n_heads)
    for i in reduced:
        p = batch.reqs[i].align_probe
        packed[i, 0], packed[i, 1] = p.text_start, p.text_len
        packed[i, 2 : 2 + nh] = _probe_weight(p, layers, n_heads).reshape(-1)
        if p.mask:
            _windows(
                p,
                packed[i, 2 + nh : 2 + 2 * nh].reshape(nl, n_heads),
                packed[i, 2 + 2 * nh :].reshape(nl, n_heads),
                layers,
            )
    mrows = None
    if masked:
        mrows = _default_rows(static.max_masked if static is not None else len(masked), 4, nl, n_heads)
        for slot, i in enumerate(masked):
            r = batch.reqs[i]
            p = r.align_probe
            mrows[slot, :4] = (i, r.device_len, p.text_start, p.text_len)
            row = mrows[slot, 4:]
            _windows(
                p,
                row[nh : 2 * nh].reshape(nl, n_heads),
                row[2 * nh :].reshape(nl, n_heads),
                layers,
                flag=row[:nh].reshape(nl, n_heads),
            )
    pin = torch.device(device).type == "cuda"
    host = torch.from_numpy(packed)
    hmask = None if mrows is None else torch.from_numpy(mrows)
    if pin:
        host = host.pin_memory()
        hmask = None if hmask is None else hmask.pin_memory()
    if static is not None:
        static.packed[:n].copy_(host, non_blocking=True)
        if masked:  # only the masked graph variant reads these slots
            static.masked.copy_(hmask, non_blocking=True)
        batch.probe = static.inputs(n, reduced, masked=bool(masked))
    else:
        lmax = max(batch.reqs[i].align_probe.text_len for i in reduced)
        kmax = max((batch.reqs[i].device_len for i in masked), default=0)
        batch.probe = ProbeInputs(
            host.to(device, non_blocking=True),
            lmax,
            torch.empty((n, 2), dtype=torch.float32, device=device),
            reduced,
            None if not masked else hmask.to(device, non_blocking=True),
            kmax,
        )


# ------------------------------------------------------------------------------------------------- batched kernels


def _readout(layer_id: int, q: torch.Tensor, batch: "TTSBatch", backend: "BaseAttnBackend") -> None:
    """Batched text-attention readout for one attention layer; capture-safe (no host syncs, static shapes)."""
    pb = batch.probe
    layers = attention_layers()
    slot = layers.index(layer_id)
    n, n_heads, head_dim = pb.packed.shape[0], q.shape[1], q.shape[2]
    k_pool = backend.kvcache.k_cache(layer_id)[:, 0]  # [pages, kv heads, head_dim]
    d = pb._d
    if slot == 0:
        nh = len(layers) * n_heads
        pk = pb.packed
        ar = torch.arange(pb.lmax, device=pk.device)
        start, length = pk[:, 0].long(), pk[:, 1].long()
        page_table = batch.attn_metadata.page_table[:n]
        idx = (start[:, None] + ar[None, :]).clamp(max=page_table.shape[1] - 1)
        d.clear()
        d.update(
            ar=ar,
            slots=page_table.gather(1, idx).long().clamp(0, k_pool.shape[0] - 1),
            valid=ar[None, :] < length[:, None],
            weight=pk[:, 2 : 2 + nh].view(n, len(layers), n_heads),
            lo=pk[:, 2 + nh : 2 + 2 * nh].long().view(n, len(layers), n_heads),
            hi=pk[:, 2 + 2 * nh :].long().view(n, len(layers), n_heads),
        )
    keys = k_pool[d["slots"]].float()  # [n, lmax, kv heads, head_dim]
    kv = keys.shape[2]
    logits = torch.einsum("nkgd,nlkd->nkgl", q[:n].float().view(n, kv, n_heads // kv, head_dim), keys).reshape(
        n, n_heads, -1
    ) * float(backend.scale)
    pos = d["ar"].view(1, 1, -1)
    allowed = d["valid"][:, None, :] & (pos >= d["lo"][:, slot, :, None]) & (pos <= d["hi"][:, slot, :, None])
    p = torch.softmax(logits.masked_fill(~allowed, float("-inf")), dim=-1)
    a = (p * d["weight"][:, slot, :, None]).sum(1)  # [n, lmax]
    d["acc"] = a if slot == 0 else d["acc"] + a
    if slot == len(layers) - 1:
        acc = d["acc"]
        m = (acc * d["ar"]).sum(-1) / acc.sum(-1).clamp_min(1e-30)
        pb.out.copy_(torch.stack((acc.argmax(-1).to(torch.float32), m), dim=-1))


def _masked(
    layer_id: int, q: torch.Tensor, o: torch.Tensor, batch: "TTSBatch", backend: "BaseAttnBackend"
) -> torch.Tensor:
    """Recompute the masked heads of the masked requests over all their keys (ACI hard mask on the text keys) and
    return ``o`` with those head outputs replaced; capture-safe."""
    pb = batch.probe
    layers = attention_layers()
    slot, nl = layers.index(layer_id), len(attention_layers())
    n, n_heads, head_dim = o.shape[0], q.shape[1], q.shape[2]
    k_pool = backend.kvcache.k_cache(layer_id)[:, 0]
    v_pool = backend.kvcache.v_cache(layer_id)[:, 0]
    d = pb._d
    if slot == 0:  # after _readout cleared the dict for this forward
        mp, nh = pb.masked, nl * n_heads
        n_slots = mp.shape[0]
        rows = mp[:, 0].long()
        safe = rows.clamp(0, n - 1)
        ar = torch.arange(pb.kmax, device=mp.device)
        page_table = batch.attn_metadata.page_table[:n]
        d.update(
            m_rows=torch.where(rows < 0, n, rows),  # unused slots write a scratch row
            m_safe=safe,
            m_slots=page_table[safe][:, ar.clamp(max=page_table.shape[1] - 1)].long().clamp(0, k_pool.shape[0] - 1),
            m_invalid=ar[None, :] >= mp[:, 1].long()[:, None],
            m_tidx=ar[None, :] - mp[:, 2].long()[:, None],
            m_tlen=mp[:, 3].long(),
            m_flag=mp[:, 4 : 4 + nh].view(n_slots, nl, n_heads),
            m_lo=mp[:, 4 + nh : 4 + 2 * nh].long().view(n_slots, nl, n_heads),
            m_hi=mp[:, 4 + 2 * nh :].long().view(n_slots, nl, n_heads),
        )
    keys = k_pool[d["m_slots"]].float()  # [slots, kmax, kv heads, head_dim]
    values = v_pool[d["m_slots"]].float()
    n_slots, kv = keys.shape[0], keys.shape[2]
    group = n_heads // kv
    logits = torch.einsum("mkgd,mlkd->mkgl", q[d["m_safe"]].float().view(n_slots, kv, group, head_dim), keys).reshape(
        n_slots, n_heads, -1
    ) * float(backend.scale)
    tidx = d["m_tidx"][:, None, :]
    in_text = (tidx >= 0) & (tidx < d["m_tlen"][:, None, None])
    outside = in_text & ((tidx < d["m_lo"][:, slot, :, None]) | (tidx > d["m_hi"][:, slot, :, None]))
    p = torch.softmax(logits.masked_fill(outside | d["m_invalid"][:, None, :], float("-inf")), dim=-1)
    out = torch.einsum("mkgl,mlkd->mkgd", p.view(n_slots, kv, group, -1), values).reshape(n_slots, n_heads, head_dim)
    new = torch.where(d["m_flag"][:, slot, :, None] > 0, out.to(o.dtype), o[d["m_safe"]])
    padded = torch.cat((o, o.new_zeros((1, n_heads, head_dim))))
    padded.index_copy_(0, d["m_rows"], new)
    return padded[:n]


def _legacy(layer_id: int, q: torch.Tensor, batch: "TTSBatch", backend: "BaseAttnBackend") -> None:
    """Eager per-request all-heads distributions (head discovery)."""
    n_heads = q.shape[1]
    scale = float(backend.scale)
    k_pool = backend.kvcache.k_cache(layer_id)[:, 0]
    kv_idx = torch.arange(n_heads, device=q.device) // (n_heads // k_pool.shape[1])
    last = layer_id == attention_layers()[-1]
    for i in batch.probe_legacy:
        req = batch.reqs[i]
        probe = req.align_probe
        t0, tl = probe.text_start, probe.text_len
        slots = batch.attn_metadata.page_table[i, : req.device_len].long()
        logits = torch.einsum("hd,nhd->hn", q[i].float(), k_pool[slots].float()[:, kv_idx]) * scale
        p_text = torch.softmax(logits, dim=-1)[:, t0 : t0 + tl]
        mass = p_text.sum(-1)
        for h in range(n_heads):
            probe._attn[(layer_id, h)] = p_text[h] / mass[h].clamp_min(1e-30)
            probe._mass[(layer_id, h)] = mass[h]
        if last:
            probe.steps += 1
            attn, mass_d = probe._attn, probe._mass
            probe._attn, probe._mass = {}, {}
            probe.observer.on_attention(probe, probe.steps, attn, mass_d)
            if probe.stop:
                req.max_device_len = min(req.max_device_len, req.device_len + 1)


def apply(
    layer_id: int, q: torch.Tensor, o: torch.Tensor, batch: "TTSBatch", backend: "BaseAttnBackend"
) -> torch.Tensor:
    """Hook called by ``Attention.forward`` after the backend attention of a decode batch. Returns ``o`` (masked heads
    replaced)."""
    if batch.probe is not None:
        _readout(layer_id, q, batch, backend)
        if batch.probe.masked is not None:
            o = _masked(layer_id, q, o, batch, backend)
    if batch.probe_legacy and not torch.cuda.is_current_stream_capturing():
        _legacy(layer_id, q, batch, backend)
    return o


def deliver(batch: "TTSBatch", values: torch.Tensor, finished) -> None:
    """Hand one decode step's host readout ([batch rows, 2]: argmax, centroid) to the observers. A stop makes the
    observed step the request's last: the step in flight (overlap scheduling) is dropped like any post-finish row."""
    vals = values.tolist()
    for i in batch.probe.rows:
        req = batch.reqs[i]
        if req in finished:
            continue
        probe = req.align_probe
        probe.steps += 1
        if not probe.stop:
            probe.observer.on_step(probe, probe.steps, int(vals[i][0]), float(vals[i][1]))
        if probe.stop:
            req.max_device_len = min(req.max_device_len, req.device_len)


def laci_probe(session) -> AlignProbe:
    """The alignment probe of a LACI session's next attempt (``zonos3.tts.laci.LaciSession``): its heads, the session as
    observer and its ACI windows."""
    return AlignProbe(session.heads.heads, session, mask=session.probe_mask())
