"""Gated delta rule recurrence, in Triton.

TTS Audio Suite addition (not an upstream file): replaces fla-core's chunked kernels
(``fla.ops.gated_delta_rule.chunk``) for the ZONOS3 GDN layers with the per-token recurrence the reference
implementation runs (``zonos3.reference.lm.Zonos3LM._gdn``):

    state = state * decay
    state = state + (beta * (v - state @ k)) outer k
    out   = state @ (q * scale)

One program per (value block, sequence, head) walks its sequence's rows and keeps its [BV, K] block of the float32
state in registers. A sequence's first row, row count and state slot are read from tensors, so a captured CUDA graph
replays the kernel correctly as they change; a sequence of zero rows leaves its state untouched.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_V = 16


@triton.jit
def _gated_delta_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    decay_ptr,
    beta_ptr,
    out_ptr,
    state_ptr,
    starts_ptr,
    lengths_ptr,
    slots_ptr,
    scale,
    HEADS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    block, pid = tl.program_id(0), tl.program_id(1)
    sequence = pid // HEADS
    head = pid % HEADS
    first = tl.load(starts_ptr + sequence).to(tl.int64)
    rows = tl.load(lengths_ptr + sequence).to(tl.int64)
    slot = tl.load(slots_ptr + sequence).to(tl.int64)
    keys = tl.arange(0, K)
    values = block * BV + tl.arange(0, BV)

    state_at = state_ptr + (slot * HEADS + head) * V * K + values[:, None] * K + keys[None, :]
    state = tl.load(state_at)  # [value, key], float32
    for t in range(0, rows):
        row = (first + t) * HEADS + head
        k = tl.load(k_ptr + row * K + keys)
        q = tl.load(q_ptr + row * K + keys) * scale
        v = tl.load(v_ptr + row * V + values)
        state = state * tl.load(decay_ptr + row)
        delta = tl.load(beta_ptr + row) * (v - tl.sum(state * k[None, :], axis=1))
        state = state + delta[:, None] * k[None, :]
        tl.store(out_ptr + row * V + values, tl.sum(state * q[None, :], axis=1))
    tl.store(state_at, state)


def gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    slots: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Outputs [rows, heads, V] float32 of packed sequences; ``state`` [slots, heads, V, K] float32 advances in place.

    ``q`` / ``k`` [rows, heads, K], ``v`` [rows, heads, V], ``decay`` / ``beta`` [rows, heads]: contiguous float32.
    Sequence i covers rows ``starts[i] .. starts[i] + lengths[i]`` and owns ``state[slots[i]]`` (int64 tensors). Rows
    outside every sequence get zeros."""
    rows, heads, dim_k = q.shape
    dim_v = v.shape[-1]
    assert dim_v % BLOCK_V == 0 and state.dtype == torch.float32 and state.is_contiguous()
    for tensor in (q, k, v, decay, beta):
        assert tensor.dtype == torch.float32 and tensor.is_contiguous()
    out = torch.zeros((rows, heads, dim_v), dtype=torch.float32, device=q.device)
    _gated_delta_kernel[(dim_v // BLOCK_V, starts.numel() * heads)](
        q, k, v, decay, beta, out, state, starts, lengths, slots, scale, HEADS=heads, K=dim_k, V=dim_v, BV=BLOCK_V
    )
    return out
