"""Causal attention over a paged KV cache, in Triton.

TTS Audio Suite addition (not an upstream file): replaces ``sgl_kernel.flash_attn.flash_attn_with_kvcache``
(FlashAttention-3, a compiled CUDA extension) for the ZONOS3 global-attention layers.

One program per (query row, head). It walks the row's visible keys in blocks, reading each key's page from the page
table, and accumulates an online softmax in float32. Every per-row quantity (owning request, last visible position) is
read from a tensor, so a captured CUDA graph replays the kernel correctly as those tensors change.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    page_table_ptr,
    owner_ptr,
    last_ptr,
    out_ptr,
    scale,
    page_table_stride,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // HEADS
    head = pid % HEADS
    kv_head = head // (HEADS // KV_HEADS)
    dims = tl.arange(0, DIM)

    q = tl.load(q_ptr + (row * HEADS + head) * DIM + dims).to(tl.float32) * scale
    owner = tl.load(owner_ptr + row).to(tl.int64)
    last = tl.load(last_ptr + row).to(tl.int64)  # the row's own position: keys 0..last are visible

    running_max = float("-inf")
    total = 0.0
    acc = tl.zeros([DIM], dtype=tl.float32)
    for start in range(0, last + 1, BLOCK):
        columns = start + tl.arange(0, BLOCK)
        visible = columns <= last
        pages = tl.load(page_table_ptr + owner * page_table_stride + columns, mask=visible, other=0).to(tl.int64)
        offsets = (pages * KV_HEADS + kv_head) * DIM
        k = tl.load(k_ptr + offsets[:, None] + dims[None, :], mask=visible[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1)
        scores = tl.where(visible, scores, float("-inf"))
        new_max = tl.maximum(running_max, tl.max(scores, axis=0))
        rescale = tl.exp(running_max - new_max)
        weights = tl.exp(scores - new_max)
        v = tl.load(v_ptr + offsets[:, None] + dims[None, :], mask=visible[:, None], other=0.0).to(tl.float32)
        total = total * rescale + tl.sum(weights, axis=0)
        acc = acc * rescale + tl.sum(weights[:, None] * v, axis=0)
        running_max = new_max

    tl.store(out_ptr + (row * HEADS + head) * DIM + dims, (acc / total).to(out_ptr.dtype.element_ty))


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    owner: torch.Tensor,
    last: torch.Tensor,
    scale: float,
    block: int = 64,
) -> torch.Tensor:
    """``q`` [rows, heads, dim]; ``k_cache`` / ``v_cache`` [pages, 1, kv heads, dim] (page size 1); ``page_table``
    [requests, columns] int32; ``owner`` [rows] the request of each query row; ``last`` [rows] the last key position
    the row sees (its own). Returns [rows, heads, dim] in ``q``'s dtype."""
    rows, heads, dim = q.shape
    kv_heads = k_cache.shape[-2]
    assert q.is_contiguous() and k_cache.is_contiguous() and v_cache.is_contiguous() and page_table.is_contiguous()
    assert page_table.dtype == torch.int32 and owner.dtype == torch.int32 and last.dtype == torch.int32
    assert dim & (dim - 1) == 0 and heads % kv_heads == 0
    out = torch.empty_like(q)
    _paged_attention_kernel[(rows * heads,)](
        q,
        k_cache,
        v_cache,
        page_table,
        owner,
        last,
        out,
        scale,
        page_table.stride(0),
        HEADS=heads,
        KV_HEADS=kv_heads,
        DIM=dim,
        BLOCK=block,
    )
    return out
