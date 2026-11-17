"""Fused, reference-checked Block Attention Residual routing.

The kernels are adapted from the MIT-licensed fused routing implementation in
wdlctc/multi-head-attention-residuals (H=1 recovers single-head AttnRes).  The
state management differs because Block AttnRes retains only completed block
sums and treats the current partial block as a temporary final source.

The eager reference below is the equation from the official Attention
Residuals paper.  It remains the fallback on CPU and is used as the numerical
oracle in tests.
"""

from __future__ import annotations

import math
from functools import lru_cache
from importlib.util import find_spec
from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - exercised in CPU-only environments
    HAS_TRITON = False


def block_attnres_reference(
    completed_blocks: Optional[torch.Tensor],
    partial_block: torch.Tensor,
    query: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct paper equation; deliberately simple and independently testable."""
    partial_source = partial_block.unsqueeze(0)
    sources = partial_source if completed_blocks is None else torch.cat((completed_blocks, partial_source), dim=0)
    keys = F.rms_norm(
        sources,
        (sources.shape[-1],),
        weight=None,
        eps=norm_eps,
    )
    logits = torch.sum(keys * query, dim=-1, dtype=torch.float32)
    weights = torch.softmax(logits, dim=0).to(dtype=sources.dtype)
    mixed = torch.sum(sources * weights.unsqueeze(-1), dim=0)
    return mixed, sources


def flash_attnres_available() -> bool:
    """Whether the pinned, independently maintained two-phase kernels exist."""
    return find_spec("flash_attn_res") is not None


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    """Expose an FSDP DTensor's local replica to raw Triton with autograd intact."""
    return value.to_local() if isinstance(value, DTensor) else value


_EXACT_B4_PHASE1_FORWARD_CONFIGS = {
    (2, 4): (2, 4),
    (3, 4): (4, 3),
    (4, 4): (2, 3),
    (5, 4): (1, 4),
    (6, 2): (2, 4),
    (7, 1): (2, 1),
}

_EXACT_B4_PHASE1_BACKWARD_DLSE_CONFIGS = {
    (2, 4): (1, 2),
    (3, 4): (1, 4),
    (4, 4): (2, 4),
    (5, 4): (2, 2),
    (6, 2): (4, 4),
    (7, 1): (4, 3),
}

_EXACT_B4_PHASE1_BACKWARD_NO_DLSE_CONFIGS = {
    (2, 4): (1, 2),
    (3, 4): (4, 2),
    (4, 4): (2, 4),
    (5, 4): (2, 1),
    (6, 2): (2, 1),
    (7, 1): (2, 3),
}


def _phase1_forward_launch_config(
    exact_b4_shape: bool,
    completed: int,
    query_count: int,
) -> tuple[int, int]:
    """Select measured metadata only for an exact production schedule row."""
    if exact_b4_shape:
        measured = _EXACT_B4_PHASE1_FORWARD_CONFIGS.get((completed, query_count))
        if measured is not None:
            return measured
    if query_count >= 4:
        return 1, 4
    if query_count == 2:
        return 1, 2
    return 4, 3


def _phase1_backward_launch_config(
    exact_b4_shape: bool,
    completed: int,
    query_count: int,
    has_dlses: bool,
) -> tuple[int, int]:
    """Select measured metadata only for an exact production schedule row."""
    if exact_b4_shape:
        measured = (
            _EXACT_B4_PHASE1_BACKWARD_DLSE_CONFIGS if has_dlses else _EXACT_B4_PHASE1_BACKWARD_NO_DLSE_CONFIGS
        ).get((completed, query_count))
        if measured is not None:
            return measured
    if query_count == 4 and completed == 3:
        return 2, 4
    if query_count >= 4 and completed <= 3:
        return 1, 2
    if query_count >= 4 and completed == 4:
        return 2, 4
    if query_count >= 4:
        return 4, 4
    if query_count == 2:
        return 4, 4
    return 2, 3


@lru_cache(maxsize=1)
def _load_flash_attnres_ops():
    from flash_attn_res.ops.phase_1 import phase_1_batched_attention_triton_op
    from flash_attn_res.ops.phase_2 import phase_2_online_softmax_merge_triton_op

    return (
        phase_1_batched_attention_triton_op,
        phase_2_online_softmax_merge_triton_op,
    )


@lru_cache(maxsize=1)
def _load_flash_attnres_internal_ops():
    """Load the released kernels used by the shared-bank autograd wrapper."""
    from flash_attn_res.ops import phase_1

    return phase_1


def flash_phase1(
    completed_blocks: torch.Tensor,
    queries: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch multiple pseudo-queries over one immutable inter-block bank."""
    phase1, _ = _load_flash_attnres_ops()
    token_shape = completed_blocks.shape[1:-1]
    dim = completed_blocks.shape[-1]
    values = completed_blocks.reshape(completed_blocks.shape[0], 1, -1, dim)
    outputs, lses = phase1(values, queries, float(norm_eps))
    return (
        outputs.reshape(outputs.shape[0], *token_shape, dim),
        lses.reshape(lses.shape[0], *token_shape),
    )


def flash_phase2(
    partial_block: torch.Tensor,
    query: torch.Tensor,
    interblock_output: torch.Tensor,
    interblock_lse: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """Exactly merge the current partial source with a phase-1 softmax."""
    _, phase2 = _load_flash_attnres_ops()
    token_shape = partial_block.shape[:-1]
    dim = partial_block.shape[-1]
    output = phase2(
        partial_block.reshape(1, -1, dim),
        query,
        interblock_output.reshape(1, -1, dim),
        interblock_lse.reshape(1, -1),
        float(norm_eps),
    )
    return output.reshape(*token_shape, dim)


if HAS_TRITON:

    @triton.jit
    def _phase2_norm_fwd_kernel(
        PARTIAL,
        QUERY,
        INTER,
        INTER_LSE,
        NORM_WEIGHT,
        OUT,
        INTRA_LOGIT,
        INTRA_INV_RMS,
        OUTPUT_INV_RMS,
        D: tl.constexpr,
        EPS: tl.constexpr,
        INV_D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        base = token * D + offs
        x = tl.load(PARTIAL + base, mask=mask, other=0.0).to(tl.float32)
        q = tl.load(QUERY + offs, mask=mask, other=0.0).to(tl.float32)
        inter = tl.load(INTER + base, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(NORM_WEIGHT + offs, mask=mask, other=0.0).to(tl.float32)
        inter_lse = tl.load(INTER_LSE + token).to(tl.float32)

        x_inv = tl.rsqrt(tl.sum(x * x, axis=0) * INV_D + EPS)
        x_logit = tl.sum(x * q, axis=0) * x_inv
        probability = tl.sigmoid(x_logit - inter_lse)
        mixed = (inter + probability * (x - inter)).to(PARTIAL.dtype.element_ty).to(tl.float32)
        mixed_inv = tl.rsqrt(tl.sum(mixed * mixed, axis=0) * INV_D + EPS)

        tl.store(OUT + base, mixed * mixed_inv * weight, mask=mask)
        tl.store(INTRA_LOGIT + token, x_logit)
        tl.store(INTRA_INV_RMS + token, x_inv)
        tl.store(OUTPUT_INV_RMS + token, mixed_inv)

    @triton.jit
    def _phase2_norm_bwd_main_kernel(
        PARTIAL,
        QUERY,
        INTER,
        INTER_LSE,
        NORM_WEIGHT,
        DOUT,
        INTRA_LOGIT,
        INTRA_INV_RMS,
        OUTPUT_INV_RMS,
        DPARTIAL,
        DINTER,
        DINTER_LSE,
        DINTRA_LOGIT,
        D: tl.constexpr,
        INV_D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        base = token * D + offs
        x = tl.load(PARTIAL + base, mask=mask, other=0.0).to(tl.float32)
        q = tl.load(QUERY + offs, mask=mask, other=0.0).to(tl.float32)
        inter = tl.load(INTER + base, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(NORM_WEIGHT + offs, mask=mask, other=0.0).to(tl.float32)
        dout = tl.load(DOUT + base, mask=mask, other=0.0).to(tl.float32)
        inter_lse = tl.load(INTER_LSE + token).to(tl.float32)
        x_logit = tl.load(INTRA_LOGIT + token).to(tl.float32)
        x_inv = tl.load(INTRA_INV_RMS + token).to(tl.float32)
        mixed_inv = tl.load(OUTPUT_INV_RMS + token).to(tl.float32)

        probability = tl.sigmoid(x_logit - inter_lse)
        inter_probability = 1.0 - probability
        mixed = (inter + probability * (x - inter)).to(PARTIAL.dtype.element_ty).to(tl.float32)
        weighted_dout = weight * dout
        norm_coupling = tl.sum(weighted_dout * mixed, axis=0)
        dmixed = (
            (mixed_inv * weighted_dout - (mixed * (mixed_inv * mixed_inv * mixed_inv * INV_D * norm_coupling)))
            .to(PARTIAL.dtype.element_ty)
            .to(tl.float32)
        )

        merge_coupling = tl.sum(dmixed * (inter - x), axis=0)
        probability_product = probability * inter_probability
        dinter_lse = probability_product * merge_coupling
        dx_logit = -dinter_lse
        dx = probability * dmixed + dx_logit * x_inv * (q - x_logit * x_inv * x * INV_D)
        dinter = inter_probability * dmixed

        tl.store(DPARTIAL + base, dx, mask=mask)
        tl.store(DINTER + base, dinter, mask=mask)
        tl.store(DINTER_LSE + token, dinter_lse)
        tl.store(DINTRA_LOGIT + token, dx_logit)

    @triton.jit
    def _phase2_norm_bwd_reduce_kernel(
        PARTIAL,
        INTER,
        INTER_LSE,
        DOUT,
        INTRA_LOGIT,
        INTRA_INV_RMS,
        OUTPUT_INV_RMS,
        DINTRA_LOGIT,
        DQUERY,
        DNORM_WEIGHT,
        BT: tl.constexpr,
        D: tl.constexpr,
        BLOCK_BT: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token_offsets = tl.program_id(0) * BLOCK_BT + tl.arange(0, BLOCK_BT)
        hidden_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        token_mask = token_offsets < BT
        hidden_mask = hidden_offsets < D
        mask = token_mask[:, None] & hidden_mask[None, :]
        offsets = token_offsets[:, None] * D + hidden_offsets[None, :]

        x = tl.load(PARTIAL + offsets, mask=mask, other=0.0).to(tl.float32)
        inter = tl.load(INTER + offsets, mask=mask, other=0.0).to(tl.float32)
        dout = tl.load(DOUT + offsets, mask=mask, other=0.0).to(tl.float32)
        x_logit = tl.load(INTRA_LOGIT + token_offsets, mask=token_mask, other=0.0)
        inter_lse = tl.load(INTER_LSE + token_offsets, mask=token_mask, other=0.0)
        x_inv = tl.load(INTRA_INV_RMS + token_offsets, mask=token_mask, other=0.0)
        mixed_inv = tl.load(OUTPUT_INV_RMS + token_offsets, mask=token_mask, other=0.0)
        dx_logit = tl.load(DINTRA_LOGIT + token_offsets, mask=token_mask, other=0.0)
        probability = tl.sigmoid(x_logit - inter_lse)
        mixed = (inter + probability[:, None] * (x - inter)).to(PARTIAL.dtype.element_ty).to(tl.float32)

        dq = tl.sum(dx_logit[:, None] * x_inv[:, None] * x, axis=0)
        dweight = tl.sum(dout * mixed * mixed_inv[:, None], axis=0)
        tl.atomic_add(DQUERY + hidden_offsets, dq, mask=hidden_mask, sem="relaxed")
        tl.atomic_add(
            DNORM_WEIGHT + hidden_offsets,
            dweight,
            mask=hidden_mask,
            sem="relaxed",
        )

    @triton.jit
    def _phase2_norm_bwd_fused_kernel(
        PARTIAL,
        QUERY,
        INTER,
        INTER_LSE,
        NORM_WEIGHT,
        DOUT,
        INTRA_LOGIT,
        INTRA_INV_RMS,
        OUTPUT_INV_RMS,
        DPARTIAL,
        DINTER,
        DCARRY,
        DINTER_LSE,
        DQUERY,
        DNORM_WEIGHT,
        BT: tl.constexpr,
        D: tl.constexpr,
        INV_D: tl.constexpr,
        BLOCK_BT: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_CARRY: tl.constexpr,
    ):
        """Fuse phase-2, affine RMSNorm, and both parameter reductions."""
        token_offsets = tl.program_id(0) * BLOCK_BT + tl.arange(0, BLOCK_BT)
        hidden_offsets = tl.arange(0, BLOCK_D)
        token_mask = token_offsets < BT
        hidden_mask = hidden_offsets < D
        mask = token_mask[:, None] & hidden_mask[None, :]
        offsets = token_offsets[:, None] * D + hidden_offsets[None, :]

        x = tl.load(PARTIAL + offsets, mask=mask, other=0.0).to(tl.float32)
        inter = tl.load(INTER + offsets, mask=mask, other=0.0).to(tl.float32)
        query = tl.load(QUERY + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        weight = tl.load(NORM_WEIGHT + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        dout = tl.load(DOUT + offsets, mask=mask, other=0.0).to(tl.float32)
        inter_lse = tl.load(INTER_LSE + token_offsets, mask=token_mask, other=0.0).to(tl.float32)
        x_logit = tl.load(INTRA_LOGIT + token_offsets, mask=token_mask, other=0.0).to(tl.float32)
        x_inv = tl.load(INTRA_INV_RMS + token_offsets, mask=token_mask, other=0.0).to(tl.float32)
        mixed_inv = tl.load(OUTPUT_INV_RMS + token_offsets, mask=token_mask, other=0.0).to(tl.float32)

        probability = tl.sigmoid(x_logit - inter_lse)
        inter_probability = 1.0 - probability
        mixed = (inter + probability[:, None] * (x - inter)).to(PARTIAL.dtype.element_ty).to(tl.float32)
        weighted_dout = weight[None, :] * dout
        norm_coupling = tl.sum(weighted_dout * mixed, axis=1)
        dmixed = (
            (
                mixed_inv[:, None] * weighted_dout
                - mixed * (mixed_inv * mixed_inv * mixed_inv * INV_D * norm_coupling)[:, None]
            )
            .to(PARTIAL.dtype.element_ty)
            .to(tl.float32)
        )

        merge_coupling = tl.sum(dmixed * (inter - x), axis=1)
        dinter_lse = probability * inter_probability * merge_coupling
        dx_logit = -dinter_lse
        dx = probability[:, None] * dmixed + dx_logit[:, None] * x_inv[:, None] * (
            query[None, :] - x_logit[:, None] * x_inv[:, None] * x * INV_D
        )
        dinter = inter_probability[:, None] * dmixed
        if HAS_CARRY:
            dinter += tl.load(DCARRY + offsets, mask=mask, other=0.0).to(tl.float32)

        tl.store(DPARTIAL + offsets, dx, mask=mask)
        tl.store(DINTER + offsets, dinter, mask=mask)
        tl.store(DINTER_LSE + token_offsets, dinter_lse, mask=token_mask)
        dquery = tl.sum(dx_logit[:, None] * x_inv[:, None] * x, axis=0)
        dweight = tl.sum(dout * mixed * mixed_inv[:, None], axis=0)
        tl.atomic_add(DQUERY + hidden_offsets, dquery, mask=hidden_mask, sem="relaxed")
        tl.atomic_add(
            DNORM_WEIGHT + hidden_offsets,
            dweight,
            mask=hidden_mask,
            sem="relaxed",
        )

    @triton.jit
    def _phase1_norm_fwd_kernel(
        BLOCKS,
        OWNER,
        QUERIES,
        NORM_WEIGHTS,
        OUT,
        LSES,
        SOURCE_INV_RMS_CACHE,
        PROBABILITIES,
        OUTPUT_INV_RMS,
        OWNER_ROW: tl.constexpr,
        N: tl.constexpr,
        BT: tl.constexpr,
        D: tl.constexpr,
        Q: tl.constexpr,
        MAX_SOURCES: tl.constexpr,
        EPS: tl.constexpr,
        INV_D: tl.constexpr,
        PAD_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        APPLY_NORM: tl.constexpr,
        HAS_OWNER: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        source_offsets = tl.arange(0, PAD_N)
        hidden_offsets = tl.arange(0, BLOCK_D)
        source_mask = source_offsets < N
        hidden_mask = hidden_offsets < D
        mask = source_mask[:, None] & hidden_mask[None, :]
        if HAS_OWNER:
            owner_source_mask = source_offsets == OWNER_ROW
            bank_mask = mask & ~owner_source_mask[:, None]
            source_inv_mask = source_mask & ~owner_source_mask
        else:
            bank_mask = mask
            source_inv_mask = source_mask
        block_offsets = source_offsets[:, None] * (BT * D) + token * D + hidden_offsets[None, :]
        sources = tl.load(BLOCKS + block_offsets, mask=bank_mask, other=0.0).to(tl.float32)
        source_inv = tl.load(
            SOURCE_INV_RMS_CACHE + token * MAX_SOURCES + source_offsets,
            mask=source_inv_mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_OWNER:
            owner_offsets = token * D + hidden_offsets
            owner = tl.load(
                OWNER + owner_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            owner_inv = tl.rsqrt(tl.sum(owner * owner, axis=0) * INV_D + EPS)
            sources = tl.where(
                owner_source_mask[:, None],
                owner[None, :],
                sources,
            )
            source_inv = tl.where(owner_source_mask, owner_inv, source_inv)
            tl.store(
                BLOCKS + OWNER_ROW * BT * D + owner_offsets,
                owner,
                mask=hidden_mask,
            )
            tl.store(
                SOURCE_INV_RMS_CACHE + token * MAX_SOURCES + OWNER_ROW,
                owner_inv,
            )

        for query_index in tl.static_range(Q):
            query = tl.load(
                QUERIES + query_index * D + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            logits = tl.sum(sources * query[None, :], axis=1) * source_inv
            logits = tl.where(source_mask, logits, -float("inf"))
            maximum = tl.max(logits, axis=0)
            exponentials = tl.exp(logits - maximum)
            denominator = tl.sum(exponentials, axis=0)
            probabilities = exponentials / denominator
            mixed = tl.sum(probabilities[:, None] * sources, axis=0).to(BLOCKS.dtype.element_ty).to(tl.float32)
            output_base = query_index * BT * D + token * D + hidden_offsets
            if query_index == 0 and APPLY_NORM:
                norm_weight = tl.load(
                    NORM_WEIGHTS + hidden_offsets,
                    mask=hidden_mask,
                    other=0.0,
                ).to(tl.float32)
                output_inv = tl.rsqrt(tl.sum(mixed * mixed, axis=0) * INV_D + EPS)
                output = mixed * output_inv * norm_weight
                tl.store(OUTPUT_INV_RMS + token, output_inv)
            else:
                output = mixed
            tl.store(OUT + output_base, output, mask=hidden_mask)
            tl.store(LSES + query_index * BT + token, maximum + tl.log(denominator))
            tl.store(
                PROBABILITIES + query_index * BT * N + token * N + source_offsets,
                probabilities,
                mask=source_mask,
            )

    @triton.jit
    def _phase1_single_source_norm_fwd_kernel(
        OWNER,
        BLOCKS,
        SOURCE_INV_RMS_CACHE,
        NORM_WEIGHT,
        NORM_OUT,
        SOURCE_INV_RMS,
        NORMALIZED_OWNER,
        BT: tl.constexpr,
        D: tl.constexpr,
        MAX_SOURCES: tl.constexpr,
        EPS: tl.constexpr,
        INV_D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        PERSIST_SOURCE: tl.constexpr,
    ):
        """Exact N=1 phase 1 without materializing Q identical value rows."""
        token = tl.program_id(0).to(tl.int64)
        hidden_offsets = tl.arange(0, BLOCK_D)
        hidden_mask = hidden_offsets < D
        owner_offsets = token * D + hidden_offsets
        owner = tl.load(OWNER + owner_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        source_inv = tl.rsqrt(tl.sum(owner * owner, axis=0) * INV_D + EPS)
        norm_weight = tl.load(NORM_WEIGHT + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)

        tl.store(NORM_OUT + owner_offsets, owner * source_inv * norm_weight, mask=hidden_mask)
        tl.store(
            NORMALIZED_OWNER + owner_offsets,
            owner * source_inv,
            mask=hidden_mask,
        )
        if PERSIST_SOURCE:
            tl.store(BLOCKS + owner_offsets, owner, mask=hidden_mask)
            tl.store(
                SOURCE_INV_RMS_CACHE + token * MAX_SOURCES,
                source_inv,
            )
        tl.store(SOURCE_INV_RMS + token, source_inv)

    @triton.jit
    def _phase1_single_source_lse_kernel(
        OWNER,
        QUERIES,
        SOURCE_INV_RMS,
        LSES,
        BT: tl.constexpr,
        D: tl.constexpr,
        Q: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Compute a small query chunk without spilling a Q=22 program."""
        token = tl.program_id(0).to(tl.int64)
        hidden_offsets = tl.arange(0, BLOCK_D)
        hidden_mask = hidden_offsets < D
        owner = tl.load(
            OWNER + token * D + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        source_inv = tl.load(SOURCE_INV_RMS + token).to(tl.float32)
        for query_index in tl.static_range(Q):
            query = tl.load(
                QUERIES + query_index * D + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            logit = tl.sum(owner * query, axis=0) * source_inv
            tl.store(LSES + query_index * BT + token, logit)

    @triton.jit
    def _phase1_single_source_norm_bwd_kernel(
        OWNER,
        QUERIES,
        NORM_WEIGHT,
        SOURCE_INV_RMS,
        DNORM_OUT,
        DLSES,
        DBANK_OWNER,
        DOWNER,
        DQUERY_PARTIAL,
        DNORM_WEIGHT_PARTIAL,
        BT: tl.constexpr,
        D: tl.constexpr,
        Q: tl.constexpr,
        INV_D: tl.constexpr,
        HAS_DLSES: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Backward for the exact one-source identity/normalization shortcut."""
        token = tl.program_id(0).to(tl.int64)
        hidden_offsets = tl.arange(0, BLOCK_D)
        hidden_mask = hidden_offsets < D
        owner_offsets = token * D + hidden_offsets
        owner = tl.load(OWNER + owner_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        norm_weight = tl.load(NORM_WEIGHT + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        source_inv = tl.load(SOURCE_INV_RMS + token).to(tl.float32)
        dnorm_out = tl.load(DNORM_OUT + owner_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        weighted_dout = norm_weight * dnorm_out
        norm_coupling = tl.sum(weighted_dout * owner, axis=0)
        downer = (
            (source_inv * weighted_dout - owner * source_inv * source_inv * source_inv * INV_D * norm_coupling)
            .to(OWNER.dtype.element_ty)
            .to(tl.float32)
        )
        tl.store(
            DNORM_WEIGHT_PARTIAL + owner_offsets,
            dnorm_out * owner * source_inv,
            mask=hidden_mask,
        )

        for query_index in tl.static_range(Q):
            query = tl.load(
                QUERIES + query_index * D + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            dlse = tl.load(DLSES + query_index * BT + token).to(tl.float32) if HAS_DLSES else 0.0
            logit = tl.sum(owner * query, axis=0) * source_inv
            downer += dlse * (source_inv * query - logit * source_inv * source_inv * owner * INV_D)
            tl.store(
                DQUERY_PARTIAL + query_index * BT * D + owner_offsets,
                dlse * source_inv * owner,
                mask=hidden_mask,
            )

        dbank = tl.load(DBANK_OWNER + owner_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        tl.store(DOWNER + owner_offsets, dbank + downer, mask=hidden_mask)

    @triton.jit
    def _phase1_single_source_norm_bwd_finalize_kernel(
        OWNER,
        NORM_WEIGHT,
        SOURCE_INV_RMS,
        DNORM_OUT,
        DNORMALIZED_OWNER,
        DBANK_OWNER,
        DOWNER,
        DNORM_WEIGHT,
        BT: tl.constexpr,
        D: tl.constexpr,
        INV_D: tl.constexpr,
        BLOCK_BT: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_DBANK: tl.constexpr,
    ):
        """Finish N=1 gradients after the two skinny GEMMs over query sites."""
        token_offsets = tl.program_id(0) * BLOCK_BT + tl.arange(0, BLOCK_BT)
        hidden_offsets = tl.arange(0, BLOCK_D)
        token_mask = token_offsets < BT
        hidden_mask = hidden_offsets < D
        mask = token_mask[:, None] & hidden_mask[None, :]
        offsets = token_offsets[:, None] * D + hidden_offsets[None, :]
        owner = tl.load(OWNER + offsets, mask=mask, other=0.0).to(tl.float32)
        norm_weight = tl.load(NORM_WEIGHT + hidden_offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        source_inv = tl.load(SOURCE_INV_RMS + token_offsets, mask=token_mask, other=0.0).to(tl.float32)
        dnorm_out = tl.load(DNORM_OUT + offsets, mask=mask, other=0.0).to(tl.float32)
        dnormalized = tl.load(DNORMALIZED_OWNER + offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_DBANK:
            dbank = tl.load(DBANK_OWNER + offsets, mask=mask, other=0.0).to(tl.float32)
        else:
            dbank = 0.0

        weighted_dout = norm_weight[None, :] * dnorm_out
        norm_coupling = tl.sum(weighted_dout * owner, axis=1)
        downer_norm = (
            (
                source_inv[:, None] * weighted_dout
                - owner * (source_inv * source_inv * source_inv * INV_D * norm_coupling)[:, None]
            )
            .to(OWNER.dtype.element_ty)
            .to(tl.float32)
        )
        lse_coupling = tl.sum(dnormalized * owner, axis=1)
        downer_lse = (
            source_inv[:, None] * dnormalized
            - owner * (source_inv * source_inv * source_inv * INV_D * lse_coupling)[:, None]
        )
        tl.store(
            DOWNER + offsets,
            dbank + downer_norm + downer_lse,
            mask=mask,
        )
        dnorm_weight = tl.sum(dnorm_out * owner * source_inv[:, None], axis=0)
        tl.atomic_add(
            DNORM_WEIGHT + hidden_offsets,
            dnorm_weight,
            mask=hidden_mask,
            sem="relaxed",
        )

    @triton.jit
    def _phase1_norm_bwd_main_kernel(
        BLOCKS,
        QUERIES,
        NORM_WEIGHTS,
        SOURCE_INV_RMS_CACHE,
        PROBABILITIES,
        OUTPUT_INV_RMS,
        DOUT0,
        DOUT1,
        DOUT2,
        DOUT3,
        DLSE0,
        DLSE1,
        DLSE2,
        DLSE3,
        DBLOCKS,
        DLOGITS,
        QUERY_COEFFICIENTS,
        DQUERY_PARTIAL,
        DNORM_WEIGHT_PARTIAL,
        N: tl.constexpr,
        BT: tl.constexpr,
        D: tl.constexpr,
        Q: tl.constexpr,
        MAX_SOURCES: tl.constexpr,
        INV_D: tl.constexpr,
        SPLIT_GRADS: tl.constexpr,
        HAS_DOUT0: tl.constexpr,
        HAS_DOUT1: tl.constexpr,
        HAS_DOUT2: tl.constexpr,
        HAS_DOUT3: tl.constexpr,
        HAS_DLSE0: tl.constexpr,
        HAS_DLSE1: tl.constexpr,
        HAS_DLSE2: tl.constexpr,
        HAS_DLSE3: tl.constexpr,
        PAD_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        APPLY_NORM: tl.constexpr,
        COMPUTE_SOURCE_GRADS: tl.constexpr,
        WRITE_DLOGITS: tl.constexpr,
        WRITE_NORM_PARTIAL: tl.constexpr,
        WRITE_QUERY_COEFFICIENTS: tl.constexpr,
        WRITE_QUERY_PARTIAL: tl.constexpr,
        ACCUMULATE_SOURCE_GRADS: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        source_offsets = tl.arange(0, PAD_N)
        hidden_offsets = tl.arange(0, BLOCK_D)
        source_mask = source_offsets < N
        hidden_mask = hidden_offsets < D
        mask = source_mask[:, None] & hidden_mask[None, :]
        block_offsets = source_offsets[:, None] * (BT * D) + token * D + hidden_offsets[None, :]
        sources = tl.load(BLOCKS + block_offsets, mask=mask, other=0.0).to(tl.float32)
        source_inv = tl.load(
            SOURCE_INV_RMS_CACHE + token * MAX_SOURCES + source_offsets,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
        if COMPUTE_SOURCE_GRADS:
            grad_sources = tl.zeros((PAD_N, BLOCK_D), tl.float32)

        for query_index in tl.static_range(Q):
            query = tl.load(
                QUERIES + query_index * D + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            grad_output_offsets = token * D + hidden_offsets
            if SPLIT_GRADS:
                if query_index == 0:
                    dout = (
                        tl.load(
                            DOUT0 + grad_output_offsets,
                            mask=hidden_mask,
                            other=0.0,
                        ).to(tl.float32)
                        if HAS_DOUT0
                        else tl.zeros((BLOCK_D,), tl.float32)
                    )
                elif query_index == 1:
                    dout = (
                        tl.load(
                            DOUT1 + grad_output_offsets,
                            mask=hidden_mask,
                            other=0.0,
                        ).to(tl.float32)
                        if HAS_DOUT1
                        else tl.zeros((BLOCK_D,), tl.float32)
                    )
                elif query_index == 2:
                    dout = (
                        tl.load(
                            DOUT2 + grad_output_offsets,
                            mask=hidden_mask,
                            other=0.0,
                        ).to(tl.float32)
                        if HAS_DOUT2
                        else tl.zeros((BLOCK_D,), tl.float32)
                    )
                else:
                    dout = (
                        tl.load(
                            DOUT3 + grad_output_offsets,
                            mask=hidden_mask,
                            other=0.0,
                        ).to(tl.float32)
                        if HAS_DOUT3
                        else tl.zeros((BLOCK_D,), tl.float32)
                    )
            else:
                dout = (
                    tl.load(
                        DOUT0 + query_index * BT * D + grad_output_offsets,
                        mask=hidden_mask,
                        other=0.0,
                    ).to(tl.float32)
                    if HAS_DOUT0
                    else tl.zeros((BLOCK_D,), tl.float32)
                )
            probabilities = tl.load(
                PROBABILITIES + query_index * BT * N + token * N + source_offsets,
                mask=source_mask,
                other=0.0,
            ).to(tl.float32)
            if query_index == 0 and APPLY_NORM:
                mixed = tl.sum(probabilities[:, None] * sources, axis=0).to(BLOCKS.dtype.element_ty).to(tl.float32)
                norm_weight = tl.load(
                    NORM_WEIGHTS + hidden_offsets,
                    mask=hidden_mask,
                    other=0.0,
                ).to(tl.float32)
                output_inv = tl.load(OUTPUT_INV_RMS + token).to(tl.float32)
                weighted_dout = norm_weight * dout
                norm_coupling = tl.sum(weighted_dout * mixed, axis=0)
                dmixed = (
                    (
                        output_inv * weighted_dout
                        - (mixed * output_inv * output_inv * output_inv * INV_D * norm_coupling)
                    )
                    .to(BLOCKS.dtype.element_ty)
                    .to(tl.float32)
                )
            else:
                dmixed = dout

            grad_value_dots = tl.sum(sources * dmixed[None, :], axis=1)
            expected_grad_value_dot = tl.sum(probabilities * grad_value_dots, axis=0)
            if SPLIT_GRADS:
                if query_index == 0:
                    grad_lse = tl.load(DLSE0 + token).to(tl.float32) if HAS_DLSE0 else 0.0
                elif query_index == 1:
                    grad_lse = tl.load(DLSE1 + token).to(tl.float32) if HAS_DLSE1 else 0.0
                elif query_index == 2:
                    grad_lse = tl.load(DLSE2 + token).to(tl.float32) if HAS_DLSE2 else 0.0
                else:
                    grad_lse = tl.load(DLSE3 + token).to(tl.float32) if HAS_DLSE3 else 0.0
            else:
                grad_lse = tl.load(DLSE0 + query_index * BT + token).to(tl.float32) if HAS_DLSE0 else 0.0
            dlogits = probabilities * (grad_lse + grad_value_dots - expected_grad_value_dot)
            if COMPUTE_SOURCE_GRADS:
                logits = tl.sum(sources * query[None, :], axis=1) * source_inv
                grad_sources += probabilities[:, None] * dmixed[None, :] + (
                    dlogits[:, None]
                    * (
                        source_inv[:, None] * query[None, :]
                        - logits[:, None] * source_inv[:, None] * source_inv[:, None] * sources * INV_D
                    )
                )
            if WRITE_DLOGITS:
                tl.store(
                    DLOGITS + query_index * BT * N + token * N + source_offsets,
                    dlogits,
                    mask=source_mask,
                )
            if WRITE_QUERY_COEFFICIENTS:
                tl.store(
                    QUERY_COEFFICIENTS + source_offsets * BT * Q + token * Q + query_index,
                    dlogits * source_inv,
                    mask=source_mask,
                )
            if WRITE_QUERY_PARTIAL:
                dquery = tl.sum(
                    dlogits[:, None] * source_inv[:, None] * sources,
                    axis=0,
                )
                tl.store(
                    DQUERY_PARTIAL + query_index * BT * D + token * D + hidden_offsets,
                    dquery,
                    mask=hidden_mask,
                )
            if WRITE_NORM_PARTIAL and query_index == 0 and APPLY_NORM:
                tl.store(
                    DNORM_WEIGHT_PARTIAL + token * D + hidden_offsets,
                    dout * mixed * output_inv,
                    mask=hidden_mask,
                )
        if COMPUTE_SOURCE_GRADS:
            if ACCUMULATE_SOURCE_GRADS:
                grad_sources += tl.load(DBLOCKS + block_offsets, mask=mask, other=0.0).to(tl.float32)
            tl.store(DBLOCKS + block_offsets, grad_sources, mask=mask)

    @triton.jit
    def _phase1_source_bwd_kernel(
        BLOCKS,
        QUERIES,
        SOURCE_INV_RMS_CACHE,
        PROBABILITIES,
        DOUT0,
        DOUT1,
        DOUT2,
        DOUT3,
        DLOGITS,
        DBLOCKS,
        N: tl.constexpr,
        BT: tl.constexpr,
        D: tl.constexpr,
        Q: tl.constexpr,
        MAX_SOURCES: tl.constexpr,
        INV_D: tl.constexpr,
        SPLIT_GRADS: tl.constexpr,
        HAS_DOUT0: tl.constexpr,
        HAS_DOUT1: tl.constexpr,
        HAS_DOUT2: tl.constexpr,
        HAS_DOUT3: tl.constexpr,
        PAD_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        ACCUMULATE_SOURCE_GRADS: tl.constexpr,
    ):
        """Accumulate only immutable-source gradients for an async future route."""
        token = tl.program_id(0).to(tl.int64)
        source_offsets = tl.arange(0, PAD_N)
        hidden_offsets = tl.arange(0, BLOCK_D)
        source_mask = source_offsets < N
        hidden_mask = hidden_offsets < D
        mask = source_mask[:, None] & hidden_mask[None, :]
        block_offsets = source_offsets[:, None] * (BT * D) + token * D + hidden_offsets[None, :]
        sources = tl.load(BLOCKS + block_offsets, mask=mask, other=0.0).to(tl.float32)
        source_inv = tl.load(
            SOURCE_INV_RMS_CACHE + token * MAX_SOURCES + source_offsets,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
        grad_sources = tl.zeros((PAD_N, BLOCK_D), tl.float32)

        for query_index in tl.static_range(Q):
            query = tl.load(
                QUERIES + query_index * D + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            grad_output_offsets = token * D + hidden_offsets
            if SPLIT_GRADS:
                if query_index == 0:
                    dout = (
                        tl.load(
                            DOUT0 + grad_output_offsets,
                            mask=hidden_mask,
                            other=0.0,
                        ).to(tl.float32)
                        if HAS_DOUT0
                        else tl.zeros((BLOCK_D,), tl.float32)
                    )
                elif query_index == 1:
                    dout = (
                        tl.load(
                            DOUT1 + grad_output_offsets,
                            mask=hidden_mask,
                            other=0.0,
                        ).to(tl.float32)
                        if HAS_DOUT1
                        else tl.zeros((BLOCK_D,), tl.float32)
                    )
                elif query_index == 2:
                    dout = (
                        tl.load(
                            DOUT2 + grad_output_offsets,
                            mask=hidden_mask,
                            other=0.0,
                        ).to(tl.float32)
                        if HAS_DOUT2
                        else tl.zeros((BLOCK_D,), tl.float32)
                    )
                else:
                    dout = (
                        tl.load(
                            DOUT3 + grad_output_offsets,
                            mask=hidden_mask,
                            other=0.0,
                        ).to(tl.float32)
                        if HAS_DOUT3
                        else tl.zeros((BLOCK_D,), tl.float32)
                    )
            else:
                dout = (
                    tl.load(
                        DOUT0 + query_index * BT * D + grad_output_offsets,
                        mask=hidden_mask,
                        other=0.0,
                    ).to(tl.float32)
                    if HAS_DOUT0
                    else tl.zeros((BLOCK_D,), tl.float32)
                )
            probabilities = tl.load(
                PROBABILITIES + query_index * BT * N + token * N + source_offsets,
                mask=source_mask,
                other=0.0,
            ).to(tl.float32)
            dlogits = tl.load(
                DLOGITS + query_index * BT * N + token * N + source_offsets,
                mask=source_mask,
                other=0.0,
            ).to(tl.float32)
            logits = tl.sum(sources * query[None, :], axis=1) * source_inv
            grad_sources += probabilities[:, None] * dout[None, :] + (
                dlogits[:, None]
                * (
                    source_inv[:, None] * query[None, :]
                    - logits[:, None] * source_inv[:, None] * source_inv[:, None] * sources * INV_D
                )
            )

        if ACCUMULATE_SOURCE_GRADS:
            grad_sources += tl.load(DBLOCKS + block_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(DBLOCKS + block_offsets, grad_sources, mask=mask)

    @triton.jit
    def _phase1_norm_bwd_grouped_reduce_kernel(
        BLOCKS,
        SOURCE_INV_RMS_CACHE,
        PROBABILITIES,
        DLOGITS,
        OUTPUT_INV_RMS,
        DOUT,
        DQUERIES,
        DNORM_WEIGHT,
        N: tl.constexpr,
        Q: tl.constexpr,
        BT: tl.constexpr,
        D: tl.constexpr,
        MAX_SOURCES: tl.constexpr,
        PAD_N: tl.constexpr,
        BLOCK_BT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        APPLY_NORM: tl.constexpr,
        HAS_DOUT0: tl.constexpr,
    ):
        token_offsets = tl.program_id(0) * BLOCK_BT + tl.arange(0, BLOCK_BT)
        source_offsets = tl.arange(0, PAD_N)
        hidden_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        token_mask = token_offsets < BT
        source_mask = source_offsets < N
        hidden_mask = hidden_offsets < D
        source_values = tl.load(
            BLOCKS
            + source_offsets[None, :, None] * (BT * D)
            + token_offsets[:, None, None] * D
            + hidden_offsets[None, None, :],
            mask=(token_mask[:, None, None] & source_mask[None, :, None] & hidden_mask[None, None, :]),
            other=0.0,
        ).to(tl.float32)
        source_inv = tl.load(
            SOURCE_INV_RMS_CACHE + token_offsets[:, None] * MAX_SOURCES + source_offsets[None, :],
            mask=token_mask[:, None] & source_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        for query_index in tl.static_range(Q):
            dlogits = tl.load(
                DLOGITS + query_index * BT * N + token_offsets[:, None] * N + source_offsets[None, :],
                mask=token_mask[:, None] & source_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            dquery_rows = tl.sum(
                dlogits[:, :, None] * source_inv[:, :, None] * source_values,
                axis=1,
            )
            dquery = tl.sum(dquery_rows, axis=0)
            tl.atomic_add(
                DQUERIES + query_index * D + hidden_offsets,
                dquery,
                mask=hidden_mask,
                sem="relaxed",
            )
            if query_index == 0 and APPLY_NORM:
                probabilities = tl.load(
                    PROBABILITIES + token_offsets[:, None] * N + source_offsets[None, :],
                    mask=token_mask[:, None] & source_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                mixed = (
                    tl.sum(
                        probabilities[:, :, None] * source_values,
                        axis=1,
                    )
                    .to(BLOCKS.dtype.element_ty)
                    .to(tl.float32)
                )
                dout = (
                    tl.load(
                        DOUT + token_offsets[:, None] * D + hidden_offsets[None, :],
                        mask=token_mask[:, None] & hidden_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    if HAS_DOUT0
                    else tl.zeros((BLOCK_BT, BLOCK_H), tl.float32)
                )
                output_inv = tl.load(
                    OUTPUT_INV_RMS + token_offsets,
                    mask=token_mask,
                    other=0.0,
                ).to(tl.float32)
                dnorm_weight = tl.sum(dout * mixed * output_inv[:, None], axis=0)
                tl.atomic_add(
                    DNORM_WEIGHT + hidden_offsets,
                    dnorm_weight,
                    mask=hidden_mask,
                    sem="relaxed",
                )

    @triton.jit
    def _reduce_query_weight_partials_kernel(
        DQUERY_PARTIAL,
        DNORM_WEIGHT_PARTIAL,
        DQUERIES,
        DNORM_WEIGHT,
        BT: tl.constexpr,
        D: tl.constexpr,
        BLOCK_BT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        APPLY_NORM: tl.constexpr,
    ):
        token_offsets = tl.program_id(0) * BLOCK_BT + tl.arange(0, BLOCK_BT)
        query_index = tl.program_id(1)
        hidden_offsets = tl.program_id(2) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = (token_offsets[:, None] < BT) & (hidden_offsets[None, :] < D)
        offsets = token_offsets[:, None] * D + hidden_offsets[None, :]
        dquery = tl.load(
            DQUERY_PARTIAL + query_index * BT * D + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dquery = tl.sum(dquery, axis=0)
        tl.atomic_add(
            DQUERIES + query_index * D + hidden_offsets,
            dquery,
            mask=hidden_offsets < D,
            sem="relaxed",
        )
        if query_index == 0 and APPLY_NORM:
            dnorm_weight = tl.load(
                DNORM_WEIGHT_PARTIAL + offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            dnorm_weight = tl.sum(dnorm_weight, axis=0)
            tl.atomic_add(
                DNORM_WEIGHT + hidden_offsets,
                dnorm_weight,
                mask=hidden_offsets < D,
                sem="relaxed",
            )

    @triton.jit
    def _reduce_norm_weight_partial_kernel(
        DNORM_WEIGHT_PARTIAL,
        DNORM_WEIGHT,
        BT: tl.constexpr,
        D: tl.constexpr,
        BLOCK_BT: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token_offsets = tl.program_id(0) * BLOCK_BT + tl.arange(0, BLOCK_BT)
        hidden_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = (token_offsets[:, None] < BT) & (hidden_offsets[None, :] < D)
        partial = tl.load(
            DNORM_WEIGHT_PARTIAL + token_offsets[:, None] * D + hidden_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.atomic_add(
            DNORM_WEIGHT + hidden_offsets,
            tl.sum(partial, axis=0),
            mask=hidden_offsets < D,
            sem="relaxed",
        )

    @triton.jit
    def _phase1_norm_bwd_reduce_kernel(
        BLOCKS,
        SOURCE_INV_RMS_CACHE,
        PROBABILITIES,
        DLOGITS,
        OUTPUT_INV_RMS,
        DOUT,
        DQUERIES,
        DNORM_WEIGHTS,
        N: tl.constexpr,
        BT: tl.constexpr,
        D: tl.constexpr,
        MAX_SOURCES: tl.constexpr,
        BLOCK_BT: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        query_index = tl.program_id(0)
        token_offsets = tl.program_id(1) * BLOCK_BT + tl.arange(0, BLOCK_BT)
        hidden_offsets = tl.program_id(2) * BLOCK_H + tl.arange(0, BLOCK_H)
        token_mask = token_offsets < BT
        hidden_mask = hidden_offsets < D
        mask = token_mask[:, None] & hidden_mask[None, :]
        offsets = token_offsets[:, None] * D + hidden_offsets[None, :]
        mixed = tl.zeros((BLOCK_BT, BLOCK_H), tl.float32)
        dquery_rows = tl.zeros((BLOCK_BT, BLOCK_H), tl.float32)

        for source_index in tl.static_range(N):
            source = tl.load(
                BLOCKS + source_index * BT * D + offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            source_inv = tl.load(
                SOURCE_INV_RMS_CACHE + token_offsets * MAX_SOURCES + source_index,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            probability = tl.load(
                PROBABILITIES + query_index * BT * N + token_offsets * N + source_index,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            dlogit = tl.load(
                DLOGITS + query_index * BT * N + token_offsets * N + source_index,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            mixed += probability[:, None] * source
            dquery_rows += dlogit[:, None] * source_inv[:, None] * source

        mixed = mixed.to(BLOCKS.dtype.element_ty).to(tl.float32)
        dout = tl.load(
            DOUT + query_index * BT * D + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dquery = tl.sum(dquery_rows, axis=0)
        tl.atomic_add(
            DQUERIES + query_index * D + hidden_offsets,
            dquery,
            mask=hidden_mask,
            sem="relaxed",
        )
        if query_index == 0:
            output_inv = tl.load(
                OUTPUT_INV_RMS + token_offsets,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            dnorm_weight = tl.sum(dout * mixed * output_inv[:, None], axis=0)
            tl.atomic_add(
                DNORM_WEIGHTS + hidden_offsets,
                dnorm_weight,
                mask=hidden_mask,
                sem="relaxed",
            )

    @triton.jit
    def _block_attnres_fwd_kernel(
        BLOCKS,
        PARTIAL,
        QUERY,
        OUT,
        WEIGHTS,
        C,
        # BT (tokens in the forward) and STRIDE_BLOCK (= BT * D) only enter address arithmetic; as runtime arguments a
        # new prefill length reuses the compiled kernel instead of JIT-compiling one per length (numerics unchanged).
        BT,
        D: tl.constexpr,
        STRIDE_BLOCK,
        EPS: tl.constexpr,
        INV_D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per token; fuse norm, depth scores, softmax, and mixing."""
        token = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        q = tl.load(QUERY + offs, mask=mask, other=0.0).to(tl.float32)
        block_base = BLOCKS + token * D
        weight_base = WEIGHTS + token

        maximum = tl.full((), -float("inf"), tl.float32)
        denominator = tl.zeros((), tl.float32)

        # First pass computes and stores logits. The explicit input-dtype casts
        # mirror the eager BF16 norm/product boundaries rather than silently
        # changing the routing equation while fusing it.
        for source in range(C):
            source64 = source.to(tl.int64)
            value = tl.load(
                block_base + source64 * STRIDE_BLOCK + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            inv_rms = tl.rsqrt(tl.sum(value * value, axis=0) * INV_D + EPS)
            key = (value * inv_rms).to(BLOCKS.dtype.element_ty).to(tl.float32)
            product = (key * q).to(BLOCKS.dtype.element_ty).to(tl.float32)
            logit = tl.sum(product, axis=0)
            tl.store(weight_base + source64 * BT, logit)
            new_maximum = tl.maximum(maximum, logit)
            denominator = denominator * tl.exp(maximum - new_maximum) + tl.exp(logit - new_maximum)
            maximum = new_maximum

        partial = tl.load(PARTIAL + token * D + offs, mask=mask, other=0.0).to(tl.float32)
        partial_inv_rms = tl.rsqrt(tl.sum(partial * partial, axis=0) * INV_D + EPS)
        partial_key = (partial * partial_inv_rms).to(PARTIAL.dtype.element_ty).to(tl.float32)
        partial_product = (partial_key * q).to(PARTIAL.dtype.element_ty).to(tl.float32)
        partial_logit = tl.sum(partial_product, axis=0)
        tl.store(weight_base + C * BT, partial_logit)
        new_maximum = tl.maximum(maximum, partial_logit)
        denominator = denominator * tl.exp(maximum - new_maximum) + tl.exp(partial_logit - new_maximum)
        maximum = new_maximum

        # A second source pass is intentional: it avoids materializing normalized
        # keys while preserving the eager path's BF16 softmax-weight boundary.
        mixed = tl.zeros((BLOCK_D,), tl.float32)
        for source in range(C):
            source64 = source.to(tl.int64)
            value = tl.load(
                block_base + source64 * STRIDE_BLOCK + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            logit = tl.load(weight_base + source64 * BT)
            weight = (tl.exp(logit - maximum) / denominator).to(BLOCKS.dtype.element_ty)
            tl.store(weight_base + source64 * BT, weight.to(tl.float32))
            term = (value * weight.to(tl.float32)).to(BLOCKS.dtype.element_ty)
            mixed += term.to(tl.float32)

        partial_weight = (tl.exp(partial_logit - maximum) / denominator).to(PARTIAL.dtype.element_ty)
        tl.store(
            weight_base + C * BT,
            partial_weight.to(tl.float32),
        )
        partial_term = (partial * partial_weight.to(tl.float32)).to(PARTIAL.dtype.element_ty)
        mixed += partial_term.to(tl.float32)
        tl.store(OUT + token * D + offs, mixed, mask=mask)

    @triton.jit
    def _block_attnres_bwd_kernel(
        BLOCKS,
        PARTIAL,
        QUERY,
        WEIGHTS,
        DOUT,
        DBLOCKS,
        DPARTIAL,
        DQUERY_PARTIAL,
        C,
        OWNER_ROW,
        BT: tl.constexpr,
        D: tl.constexpr,
        STRIDE_BLOCK: tl.constexpr,
        EPS: tl.constexpr,
        INV_D: tl.constexpr,
        ACCUMULATE: tl.constexpr,
        HAS_OWNER: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Analytical backward, recomputing only RMS statistics from sources."""
        token = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        q = tl.load(QUERY + offs, mask=mask, other=0.0).to(tl.float32)
        dout = tl.load(DOUT + token * D + offs, mask=mask, other=0.0).to(tl.float32)
        block_base = BLOCKS + token * D
        dblock_base = DBLOCKS + token * D
        weight_base = WEIGHTS + token

        coupling = tl.zeros((), tl.float32)
        for source in range(C):
            source64 = source.to(tl.int64)
            value = tl.load(
                block_base + source64 * STRIDE_BLOCK + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(weight_base + source64 * BT)
            coupling += weight * tl.sum(dout * value, axis=0)

        partial = tl.load(PARTIAL + token * D + offs, mask=mask, other=0.0).to(tl.float32)
        partial_weight = tl.load(weight_base + C * BT)
        coupling += partial_weight * tl.sum(dout * partial, axis=0)

        dq = tl.zeros((BLOCK_D,), tl.float32)
        for source in range(C):
            source64 = source.to(tl.int64)
            value = tl.load(
                block_base + source64 * STRIDE_BLOCK + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(weight_base + source64 * BT)
            dweight = tl.sum(dout * value, axis=0)
            dlogit = weight * (dweight - coupling)
            inv_rms = tl.rsqrt(tl.sum(value * value, axis=0) * INV_D + EPS)
            key = (value * inv_rms).to(BLOCKS.dtype.element_ty).to(tl.float32)
            dq += dlogit * key
            key_dot_query = tl.sum(key * q, axis=0)
            dvalue = weight * dout + dlogit * inv_rms * (q - key * key_dot_query * INV_D)
            destination = dblock_base + source64 * STRIDE_BLOCK + offs
            if ACCUMULATE:
                dvalue += tl.load(destination, mask=mask, other=0.0)
            tl.store(destination, dvalue, mask=mask)

        partial_dweight = tl.sum(dout * partial, axis=0)
        partial_dlogit = partial_weight * (partial_dweight - coupling)
        partial_inv_rms = tl.rsqrt(tl.sum(partial * partial, axis=0) * INV_D + EPS)
        partial_key = (partial * partial_inv_rms).to(PARTIAL.dtype.element_ty).to(tl.float32)
        # Keep this after partial_key is defined; written explicitly instead of
        # sharing source-loop state so the current partial is never copied.
        dq += partial_dlogit * partial_key
        partial_key_dot_query = tl.sum(partial_key * q, axis=0)
        dpartial = partial_weight * dout + partial_dlogit * partial_inv_rms * (
            q - partial_key * partial_key_dot_query * INV_D
        )
        if HAS_OWNER:
            dpartial += tl.load(
                dblock_base + OWNER_ROW * STRIDE_BLOCK + offs,
                mask=mask,
                other=0.0,
            )
        tl.store(DPARTIAL + token * D + offs, dpartial, mask=mask)
        tl.store(DQUERY_PARTIAL + token * D + offs, dq, mask=mask)


class FusedBlockAttnResContext:
    """One forward's immutable completed-block bank and backward accumulator."""

    def __init__(
        self,
        max_completed_blocks: int,
        reference: torch.Tensor,
        gradient_dtype: torch.dtype = torch.float32,
        *,
        zero_initialize_gradient_bank: bool = False,
    ):
        if not HAS_TRITON:
            raise RuntimeError("Triton is unavailable")
        if reference.device.type != "cuda":
            raise RuntimeError("fused Block AttnRes requires CUDA")
        self.token_shape = tuple(reference.shape[:-1])
        self.dim = int(reference.shape[-1])
        self.bt = int(math.prod(self.token_shape))
        self.max_completed_blocks = int(max_completed_blocks)
        self.gradient_dtype = gradient_dtype
        # The first reverse route covers every source row that can receive a
        # later gradient, so it can initialize the bank directly.  The opt-in
        # zero mode remains available for a reference/test arm.
        self.zero_initialize_gradient_bank = bool(zero_initialize_gradient_bank)
        self.blocks = torch.empty(
            (self.max_completed_blocks, *reference.shape),
            device=reference.device,
            dtype=reference.dtype,
        )
        self.source_inv_rms_cache = torch.empty(
            (self.bt, self.max_completed_blocks),
            device=reference.device,
            dtype=torch.float32,
        )
        self._source_norm_eps: Optional[float] = None
        self._routing_mode: Optional[str] = None
        self.exact_b4_production_shape = (
            self.bt == 110_592
            and self.dim == 1024
            and reference.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(reference.device) == (9, 0)
        )
        self.completed_count = 0
        self.call_count = 0
        self.dblocks: Optional[torch.Tensor] = None
        self._next_backward_call: Optional[int] = None
        self.flash_precomputed: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self.future_stream = torch.cuda.Stream(device=reference.device)
        self.future_forward_event = torch.cuda.Event()
        self.source_backward_event = torch.cuda.Event()
        self._future_forward_pending = False
        self._source_backward_pending = False

    def route_metadata(
        self,
        partial: torch.Tensor,
        append_current: bool,
        *,
        copy_current: bool = True,
        routing_mode: str,
    ) -> tuple[int, int, int]:
        self.validate_owner(partial)
        self.claim_routing_mode(routing_mode)
        completed = self.completed_count
        owner_row = -1
        if append_current:
            if completed >= self.max_completed_blocks:
                raise RuntimeError("Block AttnRes completed-block buffer overflow")
            owner_row = completed
            if copy_current:
                with torch.no_grad():
                    self.blocks[owner_row].copy_(partial)
            self.completed_count += 1
        call_index = self.call_count
        self.call_count += 1
        return completed, owner_row, call_index

    def validate_owner(self, partial: torch.Tensor) -> None:
        if tuple(partial.shape[:-1]) != self.token_shape or partial.shape[-1] != self.dim:
            raise RuntimeError(
                "Block AttnRes context shape changed within one model forward: "
                f"expected {(*self.token_shape, self.dim)}, got {tuple(partial.shape)}"
            )
        if partial.device != self.blocks.device:
            raise RuntimeError(
                "Block AttnRes context device changed within one model forward: "
                f"expected {self.blocks.device}, got {partial.device}"
            )
        if partial.dtype != self.blocks.dtype:
            raise RuntimeError(
                "Block AttnRes context dtype changed within one model forward: "
                f"expected {self.blocks.dtype}, got {partial.dtype}"
            )

    def validate_cached_route_inputs(
        self,
        owner: torch.Tensor,
        queries: torch.Tensor,
        norm_weights: torch.Tensor,
    ) -> None:
        self.validate_owner(owner)
        if queries.ndim != 2 or queries.shape[0] == 0 or queries.shape[1] != self.dim:
            raise RuntimeError(
                f"cached Block AttnRes queries must have shape (Q, {self.dim}) with Q > 0, got {tuple(queries.shape)}"
            )
        if tuple(norm_weights.shape) != (self.dim,):
            raise RuntimeError(
                f"cached Block AttnRes norm weights must have shape ({self.dim},), got {tuple(norm_weights.shape)}"
            )
        for label, tensor in (
            ("queries", queries),
            ("norm weights", norm_weights),
        ):
            if tensor.device != self.blocks.device:
                raise RuntimeError(f"cached Block AttnRes {label} must be on {self.blocks.device}, got {tensor.device}")
            if tensor.dtype != self.blocks.dtype:
                raise RuntimeError(
                    f"cached Block AttnRes {label} must have dtype {self.blocks.dtype}, got {tensor.dtype}"
                )

    def claim_routing_mode(self, routing_mode: str) -> None:
        if routing_mode not in {"cached_norm", "uncached"}:
            raise ValueError(f"unknown Block AttnRes routing mode: {routing_mode}")
        if self._routing_mode is None:
            self._routing_mode = routing_mode
        elif self._routing_mode != routing_mode:
            raise RuntimeError(
                "Block AttnRes context cannot mix cached-normalized and uncached "
                f"routing APIs (started in {self._routing_mode!r}, got "
                f"{routing_mode!r})"
            )

    def check_source_norm_eps(self, norm_eps: float) -> None:
        norm_eps = float(norm_eps)
        if self._source_norm_eps is None:
            self._source_norm_eps = norm_eps
        elif self._source_norm_eps != norm_eps:
            raise RuntimeError(
                "Block AttnRes source norm epsilon changed within one forward: "
                f"expected {self._source_norm_eps}, got {norm_eps}"
            )

    def begin_backward(self, call_index: int, completed: int) -> bool:
        """Prepare reverse routing; return whether this call must overwrite rows."""
        first = self._next_backward_call is None
        if first:
            if call_index != self.call_count - 1:
                raise RuntimeError("Block AttnRes fused backward must begin at the final routing call")
            if self.dblocks is None:
                factory = torch.zeros if self.zero_initialize_gradient_bank else torch.empty
                self.dblocks = factory(
                    (self.completed_count, *self.token_shape, self.dim),
                    device=self.blocks.device,
                    dtype=self.gradient_dtype,
                )
            self._next_backward_call = call_index
        if call_index != self._next_backward_call:
            raise RuntimeError(
                "Block AttnRes fused backward order changed; use the reference "
                "backend with checkpointing, retained graphs, or partial backward"
            )
        self._next_backward_call = call_index - 1
        if completed > self.completed_count:
            raise RuntimeError("Block AttnRes fused context is inconsistent")
        return first and not self.zero_initialize_gradient_bank

    def completed_view(self) -> torch.Tensor:
        return self.blocks[: self.completed_count]

    def mark_future_forward(self) -> None:
        self.future_forward_event.record(self.future_stream)
        self._future_forward_pending = True

    def wait_for_future_forward(self) -> None:
        if self._future_forward_pending:
            torch.cuda.current_stream().wait_event(self.future_forward_event)
            self._future_forward_pending = False

    def mark_source_backward(self) -> None:
        self.source_backward_event.record(self.future_stream)
        self._source_backward_pending = True

    def wait_for_source_backward(self) -> None:
        if self._source_backward_pending:
            torch.cuda.current_stream().wait_event(self.source_backward_event)
            self._source_backward_pending = False


class _SharedFlashPhase1(torch.autograd.Function):
    """Phase-1 routing over one shared bank, with exact owner gradients.

    The public flash-attn-res autograd wrapper is mathematically correct, but
    repeatedly concatenating the growing source bank both copies activations
    and keeps every copied prefix alive for backward.  This wrapper retains one
    immutable bank for the complete model forward and returns each source
    row's accumulated gradient to the tensor that originally owned that row.
    The numerical kernels are the pinned reference implementation's kernels.
    """

    @staticmethod
    def forward(
        ctx,
        owner: torch.Tensor,
        queries: torch.Tensor,
        state: FusedBlockAttnResContext,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        owner = owner.contiguous()
        previous_count, owner_row, call_index = state.route_metadata(
            owner,
            True,
            routing_mode="uncached",
        )
        if owner_row != previous_count:
            raise RuntimeError("shared Block AttnRes source-bank ordering changed")

        phase_1 = _load_flash_attnres_internal_ops()
        blocks = state.completed_view().reshape(state.completed_count, 1, state.bt, state.dim)
        outputs, lses, inverse_rms_norms, attention_logits = (
            phase_1._phase_1_batched_attention_forward_with_aux_triton_op(
                blocks,
                queries,
                float(norm_eps),
            )
        )
        ctx.save_for_backward(
            queries,
            lses,
            inverse_rms_norms,
            attention_logits,
        )
        ctx.state = state
        ctx.meta = (
            state.completed_count,
            owner_row,
            call_index,
            float(norm_eps),
            owner.dtype,
        )
        return (
            outputs.reshape(outputs.shape[0], *state.token_shape, state.dim),
            lses.reshape(lses.shape[0], *state.token_shape),
        )

    @staticmethod
    def backward(ctx, grad_outputs, grad_lses):
        queries, lses, inverse_rms_norms, attention_logits = ctx.saved_tensors
        state = ctx.state
        completed, owner_row, call_index, norm_eps, owner_dtype = ctx.meta
        overwrite_source_grads = state.begin_backward(call_index, completed)

        query_count = queries.shape[0]
        output_shape = (query_count, 1, state.bt, state.dim)
        if grad_outputs is None:
            grad_outputs_f32 = torch.zeros(
                output_shape,
                device=queries.device,
                dtype=torch.float32,
            )
        else:
            grad_outputs_f32 = grad_outputs.reshape(output_shape).float().contiguous()

        grad_queries = torch.zeros_like(queries, dtype=torch.float32)
        phase_1 = _load_flash_attnres_internal_ops()
        # The reference backward uses this tensor only after consuming the
        # corresponding output-gradient row, so the FP32 output-gradient buffer
        # safely doubles as its query-gradient reduction scratch.
        phase_1._batched_attention_backward_accumulate(
            state.completed_view()[:completed].reshape(completed, 1, state.bt, state.dim),
            queries,
            lses.reshape(query_count, 1, state.bt),
            grad_outputs_f32,
            (None if grad_lses is None else grad_lses.reshape(query_count, 1, state.bt).contiguous()),
            state.dblocks[:completed].reshape(completed, 1, state.bt, state.dim),
            grad_queries,
            grad_outputs_f32,
            eps=norm_eps,
            accumulate_grad_blocks=not overwrite_source_grads,
            inverse_rms_norms=inverse_rms_norms,
            attention_logits=attention_logits,
        )
        owner_grad = state.dblocks[owner_row].to(owner_dtype)
        return owner_grad, grad_queries.to(queries.dtype), None, None


def shared_flash_phase1(
    state: FusedBlockAttnResContext,
    owner: torch.Tensor,
    queries: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append ``owner`` once and route batched queries over the shared bank."""
    return _SharedFlashPhase1.apply(
        owner,
        queries.contiguous(),
        state,
        float(norm_eps),
    )


class _SharedFlashPhase1Norm(torch.autograd.Function):
    """Shared-bank phase 1, optionally returning query rows independently.

    The split return mode makes each preallocated output/LSE row a direct
    autograd output.  PyTorch can consequently deliver the row gradients to
    ``backward`` independently instead of running ``UnbindBackward`` and
    concatenating all rows into two large packed gradient tensors.
    """

    @staticmethod
    def forward(
        ctx,
        owner: torch.Tensor,
        queries: torch.Tensor,
        norm_weights: torch.Tensor,
        state: FusedBlockAttnResContext,
        norm_eps: float,
        append_owner: bool,
        apply_norm: bool,
        async_future: bool,
        split_outputs: bool,
    ) -> tuple[torch.Tensor, ...]:
        owner = owner.contiguous()
        state.validate_cached_route_inputs(owner, queries, norm_weights)
        query_count = queries.shape[0]
        if split_outputs and query_count > 4:
            raise RuntimeError("split phase 1 supports at most four query rows")
        state.check_source_norm_eps(norm_eps)
        previous_count, owner_row, call_index = state.route_metadata(
            owner,
            bool(append_owner),
            copy_current=False,
            routing_mode="cached_norm",
        )
        if append_owner and owner_row != previous_count:
            raise RuntimeError("shared Block AttnRes source-bank ordering changed")
        completed = state.completed_count
        block_d = triton.next_power_of_2(state.dim)
        pad_n = triton.next_power_of_2(completed)
        outputs = torch.empty(
            (query_count, *state.token_shape, state.dim),
            device=owner.device,
            dtype=owner.dtype,
        )
        lses = torch.empty(
            (query_count, *state.token_shape),
            device=owner.device,
            dtype=torch.float32,
        )
        probabilities = torch.empty(
            (query_count, state.bt, completed),
            device=owner.device,
            dtype=torch.float32,
        )
        output_inv_rms = torch.empty(
            (state.bt,),
            device=owner.device,
            dtype=torch.float32,
        )
        forward_warps, forward_stages = _phase1_forward_launch_config(
            state.exact_b4_production_shape,
            completed,
            query_count,
        )
        launch_stream = torch.cuda.current_stream(device=owner.device)
        if async_future:
            state.future_stream.wait_stream(launch_stream)
        with torch.cuda.stream(state.future_stream) if async_future else torch.cuda.stream(launch_stream):
            _phase1_norm_fwd_kernel[(state.bt,)](
                state.blocks,
                owner,
                queries,
                norm_weights,
                outputs,
                lses,
                state.source_inv_rms_cache,
                probabilities,
                output_inv_rms,
                OWNER_ROW=owner_row,
                N=completed,
                BT=state.bt,
                D=state.dim,
                Q=query_count,
                MAX_SOURCES=state.max_completed_blocks,
                EPS=float(norm_eps),
                INV_D=1.0 / state.dim,
                PAD_N=pad_n,
                BLOCK_D=block_d,
                APPLY_NORM=bool(apply_norm),
                HAS_OWNER=bool(append_owner),
                num_warps=forward_warps,
                num_stages=forward_stages,
            )
            if async_future:
                state.mark_future_forward()
        if async_future:
            for tensor in (
                state.blocks,
                owner,
                queries,
                norm_weights,
                outputs,
                lses,
                state.source_inv_rms_cache,
                probabilities,
                output_inv_rms,
            ):
                tensor.record_stream(state.future_stream)
        ctx.save_for_backward(
            queries,
            norm_weights,
            probabilities,
            output_inv_rms,
        )
        ctx.set_materialize_grads(False)
        ctx.state = state
        ctx.meta = (
            completed,
            owner_row,
            call_index,
            query_count,
            block_d,
            pad_n,
            owner.dtype,
            bool(append_owner),
            bool(apply_norm),
            bool(async_future),
            bool(split_outputs),
        )
        if split_outputs:
            return (*outputs.unbind(0), *lses.unbind(0))
        return outputs, lses

    @staticmethod
    def backward(ctx, *grad_results):
        (
            queries,
            norm_weights,
            probabilities,
            output_inv_rms,
        ) = ctx.saved_tensors
        state = ctx.state
        (
            completed,
            owner_row,
            call_index,
            query_count,
            block_d,
            pad_n,
            owner_dtype,
            append_owner,
            apply_norm,
            async_future,
            split_outputs,
        ) = ctx.meta
        if split_outputs:
            if len(grad_results) != 2 * query_count:
                raise RuntimeError("split phase-1 backward received an inconsistent gradient count")
            grad_output_rows = tuple(grad_results[:query_count])
            grad_lse_rows = tuple(grad_results[query_count:])
        else:
            if len(grad_results) != 2:
                raise RuntimeError("packed phase-1 backward received an inconsistent gradient count")
            grad_outputs, grad_lses = grad_results
            grad_output_rows = (grad_outputs,)
            grad_lse_rows = (grad_lses,)
        overwrite_source_grads = state.begin_backward(call_index, completed)
        if append_owner:
            state.wait_for_source_backward()
        grad_output_rows = tuple(None if grad is None else grad.contiguous() for grad in grad_output_rows)
        grad_lse_rows = tuple(None if grad is None else grad.contiguous() for grad in grad_lse_rows)
        output_grad_flags = tuple(grad is not None for grad in grad_output_rows)
        lse_grad_flags = tuple(grad is not None for grad in grad_lse_rows)
        grad_output_ptrs = tuple(grad if grad is not None else queries for grad in grad_output_rows)
        grad_lse_ptrs = tuple(grad if grad is not None else probabilities for grad in grad_lse_rows)
        grad_output_ptrs += (queries,) * (4 - len(grad_output_ptrs))
        grad_lse_ptrs += (probabilities,) * (4 - len(grad_lse_ptrs))
        output_grad_flags += (False,) * (4 - len(output_grad_flags))
        lse_grad_flags += (False,) * (4 - len(lse_grad_flags))
        # The partial-buffer reduction is faster in the full training schedule
        # even where the grouped atomic reducer wins in isolation.
        use_grouped_reduction = False
        use_query_gemm = (
            not async_future
            and state.exact_b4_production_shape
            and owner_dtype == torch.bfloat16
            and state.blocks.dtype == torch.bfloat16
            and state.dim == 1024
            and query_count == 4
            and 2 <= completed <= 5
            and apply_norm
        )
        write_dlogits = use_grouped_reduction or async_future
        dlogits = torch.empty_like(probabilities) if write_dlogits else probabilities
        query_coefficients = (
            torch.empty(
                (completed, state.bt, query_count),
                device=queries.device,
                dtype=owner_dtype,
            )
            if use_query_gemm
            else queries
        )
        if use_grouped_reduction or use_query_gemm:
            dquery_partial = queries
            dnorm_weight_partial = (
                torch.empty(
                    (*state.token_shape, state.dim),
                    device=queries.device,
                    dtype=owner_dtype,
                )
                if use_query_gemm and apply_norm
                else norm_weights
            )
        else:
            dquery_partial = torch.empty(
                (query_count, *state.token_shape, state.dim),
                device=queries.device,
                dtype=owner_dtype,
            )
            dnorm_weight_partial = torch.empty(
                (*state.token_shape, state.dim),
                device=queries.device,
                dtype=owner_dtype,
            )
        has_dlses = any(lse_grad_flags)
        backward_warps, backward_stages = _phase1_backward_launch_config(
            state.exact_b4_production_shape,
            completed,
            query_count,
            has_dlses,
        )
        _phase1_norm_bwd_main_kernel[(state.bt,)](
            state.blocks,
            queries,
            norm_weights,
            state.source_inv_rms_cache,
            probabilities,
            output_inv_rms,
            *grad_output_ptrs,
            *grad_lse_ptrs,
            state.dblocks,
            dlogits,
            query_coefficients,
            dquery_partial,
            dnorm_weight_partial,
            N=completed,
            BT=state.bt,
            D=state.dim,
            Q=query_count,
            MAX_SOURCES=state.max_completed_blocks,
            INV_D=1.0 / state.dim,
            SPLIT_GRADS=split_outputs,
            HAS_DOUT0=output_grad_flags[0],
            HAS_DOUT1=output_grad_flags[1],
            HAS_DOUT2=output_grad_flags[2],
            HAS_DOUT3=output_grad_flags[3],
            HAS_DLSE0=lse_grad_flags[0],
            HAS_DLSE1=lse_grad_flags[1],
            HAS_DLSE2=lse_grad_flags[2],
            HAS_DLSE3=lse_grad_flags[3],
            PAD_N=pad_n,
            BLOCK_D=block_d,
            APPLY_NORM=apply_norm,
            COMPUTE_SOURCE_GRADS=not async_future,
            WRITE_DLOGITS=write_dlogits,
            WRITE_NORM_PARTIAL=not use_grouped_reduction,
            WRITE_QUERY_COEFFICIENTS=use_query_gemm,
            WRITE_QUERY_PARTIAL=not use_grouped_reduction and not use_query_gemm,
            ACCUMULATE_SOURCE_GRADS=not overwrite_source_grads,
            num_warps=backward_warps,
            num_stages=backward_stages,
        )
        dqueries = torch.zeros_like(queries, dtype=torch.float32)
        dnorm_weights = torch.zeros_like(norm_weights, dtype=torch.float32)
        if use_query_gemm:
            dqueries = torch.mm(
                query_coefficients.reshape(completed * state.bt, query_count).transpose(0, 1),
                state.blocks[:completed].reshape(completed * state.bt, state.dim),
                out_dtype=torch.float32,
            )
            if apply_norm:
                block_bt = 256
                block_h = 64
                _reduce_norm_weight_partial_kernel[
                    (
                        triton.cdiv(state.bt, block_bt),
                        triton.cdiv(state.dim, block_h),
                    )
                ](
                    dnorm_weight_partial,
                    dnorm_weights,
                    BT=state.bt,
                    D=state.dim,
                    BLOCK_BT=block_bt,
                    BLOCK_H=block_h,
                    num_warps=4,
                )
        elif use_grouped_reduction:
            block_bt = 32
            block_h = 128 if completed == 1 else 64
            _phase1_norm_bwd_grouped_reduce_kernel[
                (
                    triton.cdiv(state.bt, block_bt),
                    triton.cdiv(state.dim, block_h),
                )
            ](
                state.blocks,
                state.source_inv_rms_cache,
                probabilities,
                dlogits,
                output_inv_rms,
                grad_output_ptrs[0],
                dqueries,
                dnorm_weights,
                N=completed,
                Q=query_count,
                BT=state.bt,
                D=state.dim,
                MAX_SOURCES=state.max_completed_blocks,
                PAD_N=pad_n,
                BLOCK_BT=block_bt,
                BLOCK_H=block_h,
                APPLY_NORM=apply_norm,
                HAS_DOUT0=output_grad_flags[0],
                num_warps=1,
            )
        else:
            block_bt = 256
            block_h = 64
            _reduce_query_weight_partials_kernel[
                (
                    triton.cdiv(state.bt, block_bt),
                    query_count,
                    triton.cdiv(state.dim, block_h),
                )
            ](
                dquery_partial,
                dnorm_weight_partial,
                dqueries,
                dnorm_weights,
                BT=state.bt,
                D=state.dim,
                BLOCK_BT=block_bt,
                BLOCK_H=block_h,
                APPLY_NORM=apply_norm,
                num_warps=4,
            )
        if async_future:
            current_stream = torch.cuda.current_stream(device=queries.device)
            state.future_stream.wait_stream(current_stream)
            with torch.cuda.stream(state.future_stream):
                _phase1_source_bwd_kernel[(state.bt,)](
                    state.blocks,
                    queries,
                    state.source_inv_rms_cache,
                    probabilities,
                    *grad_output_ptrs,
                    dlogits,
                    state.dblocks,
                    N=completed,
                    BT=state.bt,
                    D=state.dim,
                    Q=query_count,
                    MAX_SOURCES=state.max_completed_blocks,
                    INV_D=1.0 / state.dim,
                    SPLIT_GRADS=split_outputs,
                    HAS_DOUT0=output_grad_flags[0],
                    HAS_DOUT1=output_grad_flags[1],
                    HAS_DOUT2=output_grad_flags[2],
                    HAS_DOUT3=output_grad_flags[3],
                    PAD_N=pad_n,
                    BLOCK_D=block_d,
                    ACCUMULATE_SOURCE_GRADS=not overwrite_source_grads,
                    num_warps=backward_warps,
                    num_stages=backward_stages,
                )
                state.mark_source_backward()
            for tensor in (
                state.blocks,
                queries,
                state.source_inv_rms_cache,
                probabilities,
                dlogits,
                state.dblocks,
                *(grad for grad in grad_output_rows if grad is not None),
            ):
                tensor.record_stream(state.future_stream)
        owner_grad = state.dblocks[owner_row].to(owner_dtype) if append_owner else None
        return (
            owner_grad,
            dqueries.to(queries.dtype),
            dnorm_weights.to(norm_weights.dtype) if apply_norm else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _SharedFlashSingleSourcePhase1Norm(torch.autograd.Function):
    """Exact N=1 route that returns only the non-identity phase-1 values."""

    @staticmethod
    def forward(
        ctx,
        owner: torch.Tensor,
        queries: torch.Tensor,
        norm_weights: torch.Tensor,
        state: FusedBlockAttnResContext,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        owner = owner.contiguous()
        state.validate_cached_route_inputs(owner, queries, norm_weights)
        state.check_source_norm_eps(norm_eps)
        previous_count, owner_row, call_index = state.route_metadata(
            owner,
            True,
            copy_current=False,
            routing_mode="cached_norm",
        )
        if previous_count != 0 or owner_row != 0 or state.completed_count != 1:
            raise RuntimeError("single-source phase 1 must initialize an empty bank")
        query_count = queries.shape[0]
        block_d = triton.next_power_of_2(state.dim)
        norm_output = torch.empty_like(owner)
        normalized_owner = torch.empty(
            (*state.token_shape, state.dim),
            device=owner.device,
            dtype=owner.dtype,
        )
        lses = torch.empty(
            (query_count, *state.token_shape),
            device=owner.device,
            dtype=torch.float32,
        )
        source_inv_rms = torch.empty((state.bt,), device=owner.device, dtype=torch.float32)
        forward_warps = 1 if query_count >= 4 else (2 if query_count == 2 else 4)
        _phase1_single_source_norm_fwd_kernel[(state.bt,)](
            owner,
            state.blocks,
            state.source_inv_rms_cache,
            norm_weights,
            norm_output,
            source_inv_rms,
            normalized_owner,
            BT=state.bt,
            D=state.dim,
            MAX_SOURCES=state.max_completed_blocks,
            EPS=float(norm_eps),
            INV_D=1.0 / state.dim,
            BLOCK_D=block_d,
            PERSIST_SOURCE=True,
            num_warps=forward_warps,
            num_stages=2,
        )
        for query_start in range(0, query_count, 4):
            query_end = min(query_start + 4, query_count)
            _phase1_single_source_lse_kernel[(state.bt,)](
                state.blocks,
                queries[query_start:query_end],
                source_inv_rms,
                lses[query_start:query_end],
                BT=state.bt,
                D=state.dim,
                Q=query_end - query_start,
                BLOCK_D=block_d,
                num_warps=1,
                num_stages=2,
            )
        ctx.save_for_backward(queries, norm_weights, source_inv_rms, normalized_owner)
        ctx.state = state
        ctx.meta = (
            owner_row,
            call_index,
            query_count,
            block_d,
            owner.dtype,
        )
        return norm_output, lses

    @staticmethod
    def backward(ctx, grad_norm_output, grad_lses):
        queries, norm_weights, source_inv_rms, normalized_owner = ctx.saved_tensors
        state = ctx.state
        owner_row, call_index, query_count, block_d, owner_dtype = ctx.meta
        overwrite_source_grads = state.begin_backward(call_index, 1)
        if grad_norm_output is None:
            grad_norm_output = torch.zeros(
                (*state.token_shape, state.dim),
                device=queries.device,
                dtype=owner_dtype,
            )
        else:
            grad_norm_output = grad_norm_output.contiguous()
        grad_lses = None if grad_lses is None else grad_lses.contiguous()
        owner_grad = torch.empty(
            (*state.token_shape, state.dim),
            device=queries.device,
            dtype=owner_dtype,
        )
        if grad_lses is None:
            dqueries = torch.zeros_like(queries, dtype=torch.float32)
            dnormalized_owner = torch.zeros(
                (state.bt, state.dim),
                device=queries.device,
                dtype=torch.float32,
            )
        else:
            grad_lses_2d = grad_lses.reshape(query_count, state.bt)
            # N=1 turns the phase-1 query/source gradients into two skinny
            # matrix products. This preserves the exact equations and avoids
            # writing and rereading Q full activation-sized partial buffers.
            grad_lses_bf16 = grad_lses_2d.to(owner_dtype)
            dqueries = torch.mm(
                grad_lses_bf16,
                normalized_owner.reshape(state.bt, state.dim),
                out_dtype=torch.float32,
            )
            dnormalized_owner = torch.mm(
                grad_lses_bf16.transpose(0, 1),
                queries,
                out_dtype=torch.float32,
            )
        dnorm_weights = torch.zeros_like(norm_weights, dtype=torch.float32)
        block_bt = 4
        _phase1_single_source_norm_bwd_finalize_kernel[(triton.cdiv(state.bt, block_bt),)](
            state.blocks[owner_row],
            norm_weights,
            source_inv_rms,
            grad_norm_output,
            dnormalized_owner,
            state.dblocks[owner_row],
            owner_grad,
            dnorm_weights,
            BT=state.bt,
            D=state.dim,
            INV_D=1.0 / state.dim,
            BLOCK_BT=block_bt,
            BLOCK_D=block_d,
            HAS_DBANK=not overwrite_source_grads,
            num_warps=4,
            num_stages=1,
        )
        return (
            owner_grad,
            dqueries.to(queries.dtype),
            dnorm_weights.to(norm_weights.dtype),
            None,
            None,
        )


class _StatelessSingleSourcePhase1Norm(torch.autograd.Function):
    """N=1 phase 1 without a mutable source-bank context or activation copy."""

    @staticmethod
    def forward(
        ctx,
        owner: torch.Tensor,
        queries: torch.Tensor,
        norm_weights: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        owner = owner.contiguous()
        token_shape = tuple(owner.shape[:-1])
        dim = int(owner.shape[-1])
        bt = int(math.prod(token_shape))
        query_count = int(queries.shape[0])
        block_d = triton.next_power_of_2(dim)
        norm_output = torch.empty_like(owner)
        normalized_owner = torch.empty_like(owner)
        lses = torch.empty(
            (query_count, *token_shape),
            device=owner.device,
            dtype=torch.float32,
        )
        source_inv_rms = torch.empty((bt,), device=owner.device, dtype=torch.float32)
        forward_warps = 1 if query_count >= 4 else (2 if query_count == 2 else 4)
        _phase1_single_source_norm_fwd_kernel[(bt,)](
            owner,
            owner,
            owner,
            norm_weights,
            norm_output,
            source_inv_rms,
            normalized_owner,
            BT=bt,
            D=dim,
            MAX_SOURCES=1,
            EPS=float(norm_eps),
            INV_D=1.0 / dim,
            BLOCK_D=block_d,
            PERSIST_SOURCE=False,
            num_warps=forward_warps,
            num_stages=2,
        )
        for query_start in range(0, query_count, 4):
            query_end = min(query_start + 4, query_count)
            _phase1_single_source_lse_kernel[(bt,)](
                owner,
                queries[query_start:query_end],
                source_inv_rms,
                lses[query_start:query_end],
                BT=bt,
                D=dim,
                Q=query_end - query_start,
                BLOCK_D=block_d,
                num_warps=1,
                num_stages=2,
            )
        ctx.save_for_backward(owner, queries, norm_weights, source_inv_rms, normalized_owner)
        ctx.meta = (token_shape, dim, bt, query_count, block_d)
        return norm_output, lses

    @staticmethod
    def backward(ctx, grad_norm_output, grad_lses):
        owner, queries, norm_weights, source_inv_rms, normalized_owner = ctx.saved_tensors
        token_shape, dim, bt, query_count, block_d = ctx.meta
        if grad_norm_output is None:
            grad_norm_output = torch.zeros_like(owner)
        else:
            grad_norm_output = grad_norm_output.contiguous()
        grad_lses = None if grad_lses is None else grad_lses.contiguous()
        if grad_lses is None:
            dqueries = torch.zeros_like(queries, dtype=torch.float32)
            dnormalized_owner = torch.zeros((bt, dim), device=owner.device, dtype=torch.float32)
        else:
            grad_lses_bf16 = grad_lses.reshape(query_count, bt).to(owner.dtype)
            dqueries = torch.mm(
                grad_lses_bf16,
                normalized_owner.reshape(bt, dim),
                out_dtype=torch.float32,
            )
            dnormalized_owner = torch.mm(
                grad_lses_bf16.transpose(0, 1),
                queries,
                out_dtype=torch.float32,
            )
        owner_grad = torch.empty_like(owner)
        dnorm_weights = torch.zeros_like(norm_weights, dtype=torch.float32)
        block_bt = 4
        _phase1_single_source_norm_bwd_finalize_kernel[(triton.cdiv(bt, block_bt),)](
            owner,
            norm_weights,
            source_inv_rms,
            grad_norm_output,
            dnormalized_owner,
            owner,
            owner_grad,
            dnorm_weights,
            BT=bt,
            D=dim,
            INV_D=1.0 / dim,
            BLOCK_BT=block_bt,
            BLOCK_D=block_d,
            HAS_DBANK=False,
            num_warps=4,
            num_stages=1,
        )
        return (
            owner_grad,
            dqueries.to(queries.dtype),
            dnorm_weights.to(norm_weights.dtype),
            None,
        )


@torch.compiler.disable
def single_source_phase1_split_norm(
    owner: torch.Tensor,
    queries: torch.Tensor,
    norm_weights: torch.Tensor,
    norm_eps: float,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Precompute all routes for one immutable source without shared state."""
    owner = owner.contiguous()
    queries = _local_tensor(queries).contiguous()
    norm_weights = _local_tensor(norm_weights).contiguous()
    norm_output, lses = _StatelessSingleSourcePhase1Norm.apply(
        owner,
        queries,
        norm_weights,
        float(norm_eps),
    )
    output_rows = (norm_output,) + (owner,) * (queries.shape[0] - 1)
    return output_rows, tuple(lses.unbind(0))


@torch.compiler.disable
def shared_flash_phase1_norm(
    state: FusedBlockAttnResContext,
    owner: torch.Tensor,
    queries: torch.Tensor,
    norm_weights: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append a source; normalize only the route consumed before phase 2.

    Output zero is immediately consumed by a model sublayer and is therefore
    RMS-normalized here. Later outputs remain raw inter-block softmax values so
    the exact phase-2 online merge can incorporate the evolving partial block.
    """
    return _SharedFlashPhase1Norm.apply(
        owner,
        _local_tensor(queries).contiguous(),
        _local_tensor(norm_weights).contiguous(),
        state,
        float(norm_eps),
        True,
        True,
        False,
        False,
    )


@torch.compiler.disable
def shared_flash_phase1_split_norm(
    state: FusedBlockAttnResContext,
    owner: torch.Tensor,
    queries: torch.Tensor,
    norm_weights: torch.Tensor,
    norm_eps: float,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Append once and cap fused query groups to avoid D=1024 register spills."""
    owner = owner.contiguous()
    queries = _local_tensor(queries).contiguous()
    norm_weights = _local_tensor(norm_weights).contiguous()
    if state.completed_count == 0:
        norm_output, lses = _SharedFlashSingleSourcePhase1Norm.apply(
            owner,
            queries,
            norm_weights,
            state,
            float(norm_eps),
        )
        # With one source, softmax is exactly one.  Only the first consumer
        # needs an affine RMSNorm; all later value rows are identity aliases.
        output_rows = (norm_output,) + (owner,) * (queries.shape[0] - 1)
        return output_rows, tuple(lses.unbind(0))
    output_rows = []
    lse_rows = []
    for query_start in range(0, queries.shape[0], 4):
        query_end = min(query_start + 4, queries.shape[0])
        flat_results = _SharedFlashPhase1Norm.apply(
            owner,
            queries[query_start:query_end],
            norm_weights,
            state,
            float(norm_eps),
            query_start == 0,
            query_start == 0,
            False,
            True,
        )
        chunk_query_count = query_end - query_start
        output_rows.extend(flat_results[:chunk_query_count])
        lse_rows.extend(flat_results[chunk_query_count:])
    return tuple(output_rows), tuple(lse_rows)


class _FusedBlockAttnRes(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        partial: torch.Tensor,
        query: torch.Tensor,
        state: FusedBlockAttnResContext,
        norm_eps: float,
        append_current: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        partial_contiguous = partial.contiguous()
        # Validate and claim the bank before launching: this path delays the
        # actual append until after routing the current partial.
        state.validate_owner(partial_contiguous)
        state.claim_routing_mode("uncached")
        completed = state.completed_count
        out = torch.empty_like(partial_contiguous)
        weights = torch.empty(
            (completed + 1, state.bt),
            device=partial.device,
            dtype=torch.float32,
        )
        block_d = triton.next_power_of_2(state.dim)
        if block_d > 8192:
            raise RuntimeError("fused Block AttnRes supports hidden dimensions <= 8192")
        num_warps = 8 if block_d >= 2048 else (4 if block_d >= 512 else 2)
        _block_attnres_fwd_kernel[(state.bt,)](
            state.blocks,
            partial_contiguous,
            query,
            out,
            weights,
            completed,
            BT=state.bt,
            D=state.dim,
            STRIDE_BLOCK=state.bt * state.dim,
            EPS=float(norm_eps),
            INV_D=1.0 / state.dim,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
        completed, owner_row, call_index = state.route_metadata(
            partial_contiguous,
            bool(append_current),
            routing_mode="uncached",
        )
        ctx.save_for_backward(partial_contiguous, query, weights)
        ctx.state = state
        ctx.meta = (completed, owner_row, call_index, float(norm_eps), block_d, num_warps)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        partial, query, weights = ctx.saved_tensors
        state = ctx.state
        completed, owner_row, call_index, norm_eps, block_d, num_warps = ctx.meta
        overwrite_source_grads = state.begin_backward(call_index, completed)
        dout = dout.contiguous()
        dpartial = torch.empty_like(partial)
        dquery_partial = torch.empty(
            (*state.token_shape, state.dim),
            device=dout.device,
            dtype=torch.float32,
        )
        _block_attnres_bwd_kernel[(state.bt,)](
            state.blocks,
            partial,
            query,
            weights,
            dout,
            state.dblocks,
            dpartial,
            dquery_partial,
            completed,
            owner_row,
            BT=state.bt,
            D=state.dim,
            STRIDE_BLOCK=state.bt * state.dim,
            EPS=norm_eps,
            INV_D=1.0 / state.dim,
            ACCUMULATE=not overwrite_source_grads,
            HAS_OWNER=owner_row >= 0 and not overwrite_source_grads,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
        dquery = dquery_partial.reshape(-1, state.dim).sum(dim=0).to(query.dtype)
        return dpartial, dquery, None, None, None


class _FusedPhase2Norm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        partial: torch.Tensor,
        query: torch.Tensor,
        interblock_output: torch.Tensor,
        interblock_lse: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not HAS_TRITON or partial.device.type != "cuda":
            raise RuntimeError("fused phase-2 Block AttnRes norm requires CUDA Triton")
        # The stateless wrapper intentionally drops the identity carrier.  Do
        # not materialize a full-size zero gradient for that unused output.
        ctx.set_materialize_grads(False)
        partial = partial.contiguous()
        interblock_output = interblock_output.contiguous()
        interblock_lse = interblock_lse.contiguous()
        token_shape = partial.shape[:-1]
        dim = partial.shape[-1]
        bt = math.prod(token_shape)
        block_d = triton.next_power_of_2(dim)
        out = torch.empty_like(partial)
        intra_logit = torch.empty(bt, device=partial.device, dtype=torch.float32)
        intra_inv_rms = torch.empty_like(intra_logit)
        output_inv_rms = torch.empty_like(intra_logit)
        _phase2_norm_fwd_kernel[(bt,)](
            partial,
            query,
            interblock_output,
            interblock_lse,
            norm_weight,
            out,
            intra_logit,
            intra_inv_rms,
            output_inv_rms,
            D=dim,
            EPS=float(norm_eps),
            INV_D=1.0 / dim,
            BLOCK_D=block_d,
            num_warps=4,
        )
        ctx.save_for_backward(
            partial,
            query,
            interblock_output,
            interblock_lse,
            norm_weight,
            intra_logit,
            intra_inv_rms,
            output_inv_rms,
        )
        ctx.meta = (token_shape, dim, bt, block_d)
        return out, interblock_output

    @staticmethod
    def backward(ctx, dout, dcarry):
        (
            partial,
            query,
            interblock_output,
            interblock_lse,
            norm_weight,
            intra_logit,
            intra_inv_rms,
            output_inv_rms,
        ) = ctx.saved_tensors
        token_shape, dim, bt, block_d = ctx.meta
        dout = dout.contiguous()
        dcarry = None if dcarry is None else dcarry.contiguous()
        dpartial = torch.empty_like(partial)
        dinter = torch.empty_like(interblock_output)
        dinter_lse = torch.empty_like(interblock_lse)
        parameter_grads = torch.zeros((2, dim), device=dout.device, dtype=torch.float32)
        dquery, dnorm_weight = parameter_grads.unbind(0)
        block_bt = 4
        _phase2_norm_bwd_fused_kernel[(triton.cdiv(bt, block_bt),)](
            partial,
            query,
            interblock_output,
            interblock_lse,
            norm_weight,
            dout,
            intra_logit,
            intra_inv_rms,
            output_inv_rms,
            dpartial,
            dinter,
            dout if dcarry is None else dcarry,
            dinter_lse,
            dquery,
            dnorm_weight,
            BT=bt,
            D=dim,
            INV_D=1.0 / dim,
            BLOCK_BT=block_bt,
            BLOCK_D=block_d,
            HAS_CARRY=dcarry is not None,
            num_warps=4,
            num_stages=1,
        )
        return (
            dpartial,
            dquery.to(query.dtype),
            dinter,
            dinter_lse,
            dnorm_weight.to(norm_weight.dtype),
            None,
        )


def fused_block_attnres(
    state: FusedBlockAttnResContext,
    partial: torch.Tensor,
    query: torch.Tensor,
    norm_eps: float,
    append_current: bool,
) -> torch.Tensor:
    return _FusedBlockAttnRes.apply(
        partial,
        query,
        state,
        float(norm_eps),
        bool(append_current),
    )


class _FusedFullAttnRes(torch.autograd.Function):
    """Full AttnRes route over the shared bank.

    Uses the same routing kernels as ``_FusedBlockAttnRes``. An appended source
    is read back from its bank row during backward, so the graph does not
    retain a second copy of every sublayer output.
    """

    @staticmethod
    def forward(
        ctx,
        source: torch.Tensor,
        query: torch.Tensor,
        state: FusedBlockAttnResContext,
        norm_eps: float,
        append_current: bool,
    ) -> torch.Tensor:
        source_contiguous = source.contiguous()
        state.validate_owner(source_contiguous)
        state.claim_routing_mode("uncached")
        completed = state.completed_count
        out = torch.empty_like(source_contiguous)
        weights = torch.empty(
            (completed + 1, state.bt),
            device=source.device,
            dtype=torch.float32,
        )
        block_d = triton.next_power_of_2(state.dim)
        if block_d > 8192:
            raise RuntimeError("fused Full AttnRes supports hidden dimensions <= 8192")
        # With Triton 3.5.1 these routing kernels corrupt a fraction of tokens
        # when launched with more than one warp and two or more bank rows; a
        # single warp matches the reference at every tested shape.
        num_warps = 1
        _block_attnres_fwd_kernel[(state.bt,)](
            state.blocks,
            source_contiguous,
            query,
            out,
            weights,
            completed,
            BT=state.bt,
            D=state.dim,
            STRIDE_BLOCK=state.bt * state.dim,
            EPS=float(norm_eps),
            INV_D=1.0 / state.dim,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
        completed, owner_row, call_index = state.route_metadata(
            source_contiguous,
            bool(append_current),
            routing_mode="uncached",
        )
        if owner_row >= 0:
            ctx.save_for_backward(query, weights)
        else:
            ctx.save_for_backward(query, weights, source_contiguous)
        ctx.state = state
        ctx.meta = (completed, owner_row, call_index, float(norm_eps), block_d, num_warps)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        saved = ctx.saved_tensors
        query, weights = saved[0], saved[1]
        state = ctx.state
        completed, owner_row, call_index, norm_eps, block_d, num_warps = ctx.meta
        source = state.blocks[owner_row] if owner_row >= 0 else saved[2]
        overwrite_source_grads = state.begin_backward(call_index, completed)
        dout = dout.contiguous()
        dsource = torch.empty_like(source)
        dquery_partial = torch.empty(
            (*state.token_shape, state.dim),
            device=dout.device,
            dtype=torch.float32,
        )
        _block_attnres_bwd_kernel[(state.bt,)](
            state.blocks,
            source,
            query,
            weights,
            dout,
            state.dblocks,
            dsource,
            dquery_partial,
            completed,
            owner_row,
            BT=state.bt,
            D=state.dim,
            STRIDE_BLOCK=state.bt * state.dim,
            EPS=norm_eps,
            INV_D=1.0 / state.dim,
            ACCUMULATE=not overwrite_source_grads,
            # The shared kernel forms OWNER_ROW * STRIDE_BLOCK in 32 bits, which
            # overflows for a full-depth bank at production size (row 19 of
            # 110592 x 1024). Add the owner row's accumulated gradient here.
            HAS_OWNER=False,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
        if owner_row >= 0 and not overwrite_source_grads:
            dsource += state.dblocks[owner_row]
        dquery = dquery_partial.reshape(-1, state.dim).sum(dim=0).to(query.dtype)
        return dsource, dquery, None, None, None


def fused_full_attnres(
    state: FusedBlockAttnResContext,
    source: torch.Tensor,
    query: torch.Tensor,
    norm_eps: float,
    append_current: bool,
) -> torch.Tensor:
    return _FusedFullAttnRes.apply(
        source,
        query,
        state,
        float(norm_eps),
        bool(append_current),
    )


def fused_block_attnres_available() -> bool:
    return HAS_TRITON


@torch.compiler.disable
def fused_phase2_norm(
    partial_block: torch.Tensor,
    query: torch.Tensor,
    interblock_output: torch.Tensor,
    interblock_lse: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """Exact phase-2 online merge immediately fused with affine RMSNorm."""
    output, _ = _FusedPhase2Norm.apply(
        partial_block,
        _local_tensor(query),
        interblock_output,
        interblock_lse,
        _local_tensor(norm_weight),
        float(norm_eps),
    )
    return output


@torch.compiler.disable
def fused_phase2_norm_carry(
    partial_block: torch.Tensor,
    query: torch.Tensor,
    source_carrier: torch.Tensor,
    interblock_lse: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase 2 with an identity source-gradient carrier for an N=1 block."""
    return _FusedPhase2Norm.apply(
        partial_block,
        _local_tensor(query),
        source_carrier,
        interblock_lse,
        _local_tensor(norm_weight),
        float(norm_eps),
    )
