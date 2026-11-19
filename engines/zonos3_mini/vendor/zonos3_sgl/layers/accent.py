"""Per-token accent adapter shifts applied inside ops.

See zonos3_sgl.models.accent_adapters: an adapted op carries ``_accent`` (down
[K, in], up [K, out]) and a router ``_accent_bias`` [slots, experts]; the active
batch carries the per-token slots, strengths and the [tokens, K] mask. An op
whose side path is not batch-invariant on this GPU also carries
``_accent_block_rows`` (see accent_adapters.ensure_batch_invariance). MoE
experts with per-expert factors carry ``_accent_experts`` (an
accent_adapters.ExpertAccents) and run sonic_experts_with_accents.
"""

from __future__ import annotations

import torch

from zonos3_sgl.core import get_global_ctx


def accent_mask() -> torch.Tensor | None:
    """The active batch's [tokens, K] adapter mask; None outside a forward or without adapters."""
    batch = get_global_ctx()._batch
    return None if batch is None else batch.accent_mask


def accent_shift(op, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``((x @ down.T) * mask) @ up`` for an adapted op.

    With ``_accent_block_rows`` set, the rows run as fixed-shape [block, ...]
    pieces: cuBLAS then picks the same kernels whatever the row count, so a row's
    bits do not depend on the rest of its batch.
    """
    down, up = op._accent
    block = op.__dict__.get("_accent_block_rows")
    if not block:
        return ((x @ down.T) * mask) @ up
    rows = x.shape[0]
    padded = -(-rows // block) * block
    if padded != rows:
        x = torch.cat((x, x.new_zeros(padded - rows, x.shape[1])))
        mask = torch.cat((mask, mask.new_zeros(padded - rows, mask.shape[1])))
    out = x.new_empty(padded, up.shape[1])
    for start in range(0, padded, block):
        piece = slice(start, start + block)
        torch.mm((x[piece] @ down.T) * mask[piece], up, out=out[piece])
    return out[:rows]


def apply_accent(op, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """``y`` plus the op's per-row adapter shift, if it has one.

    ``mask`` defaults to the batch's per-token mask; ops whose rows are not the
    batch tokens (the GDN mixer's replayed windows) pass their own [rows, K].
    Rows of requests without an accent have an all-zero mask row and get an
    exact zero shift.
    """
    if op.__dict__.get("_accent") is None:
        return y
    if mask is None:
        mask = accent_mask()
        if mask is None:
            return y
    return y + accent_shift(op, x, mask)


def accent_router_bias(router, bias: torch.Tensor) -> torch.Tensor:
    """Balancing biases per token: base plus each request's scaled shift."""
    table = router.__dict__.get("_accent_bias")
    if table is None:
        return bias
    batch = get_global_ctx()._batch
    if batch is None or batch.accent_slots is None:
        return bias
    return bias + batch.accent_scales.float().unsqueeze(1) * table[batch.accent_slots.long()]


def expert_pair_shift(
    accents,
    weight: str,
    rows: torch.Tensor,
    tokens: torch.Tensor,
    experts: torch.Tensor,
    slots: torch.Tensor,
    scales: torch.Tensor,
    *,
    gather_rows: bool = False,
) -> torch.Tensor:
    """``alpha * up[a, e] @ (down[a, e] @ row)`` per routed pair, for expert weight ``weight``.

    Pair p is token ``tokens[p]`` on expert ``experts[p]`` (int32); its adapter
    slot and strength are ``slots[t]`` / ``scales[t]`` of that token, its input
    row ``rows[tokens[p]]`` (``gather_rows``) or ``rows[p]``. Returns [pairs, out]
    in ``rows``' dtype; pairs whose slot has no expert factors (no accent, /1
    adapters) get exact zeros. One Triton launch per rank group
    (zonos3_sgl.kernel.triton.expert_lora): per-pair float32 sums in a fixed
    order, so a pair's bits do not depend on its batch.
    """
    from zonos3_sgl.kernel.triton.expert_lora import expert_lora_shift

    out = rows.new_empty(tokens.shape[0], accents.groups[0].factors[weight][1].shape[-1])
    for index, group in enumerate(accents.groups):
        down, up = group.factors[weight]
        expert_lora_shift(
            rows,
            tokens,
            experts,
            slots,
            scales,
            group.first,
            down,
            up,
            out,
            gather_rows=gather_rows,
            zero_others=index == 0,
        )
    return out


def expert_accent_inputs() -> tuple[torch.Tensor, torch.Tensor] | None:
    """The active batch's per-token (slots, strengths); None outside a forward or without adapters."""
    batch = get_global_ctx()._batch
    if batch is None or batch.accent_slots is None:
        return None
    return batch.accent_slots, batch.accent_scales


def sonic_experts_with_accents(
    op,
    x: torch.Tensor,
    scores: torch.Tensor,
    topk_ids: torch.Tensor,
    slots: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """SonicMoE ``moe_general_routing_inputs`` (inference) with per-pair expert adapter terms.

    The same routing metadata, QuACK kernels and top-k combine as the exact
    path, except that both grouped GEMMs take an addend C (``D = A @ B + C``,
    added to the float32 accumulator before the epilogue):

    - up projection: C1[p] = alpha * up13[a, e] @ (down13[a, e] @ x[t]) on the
      gate/up pre-activation of pair p = (token t, expert e), before SwiGLU;
    - down projection: C2[p] = alpha * up2[a, e] @ (down2[a, e] @ h[p]) on the
      pair's expert output (h: the adapted activation), before the weighted
      top-k sum.

    Pairs of requests without an expert adapter have zero rows in C, and the
    kernels with a zero addend return the plain call's bits.
    """
    from quack.gemm_interface import gemm_add, gemm_gated
    from sonicmoe.functional.forward import _router_forward
    from sonicmoe.functional.triton_kernels import general_routing_router_metadata_triton

    w13, w2 = op._sonic_w13, op._sonic_w2  # [E, 2I, H] (gate/up interleaved), [E, H, I]
    num_experts = w13.shape[0]
    tokens, top_k = topk_ids.shape
    pairs = tokens * top_k
    device = x.device
    x = x.contiguous()
    token_indices = torch.arange(tokens, dtype=torch.int32, device=device).repeat_interleave(top_k)
    expert_indices = topk_ids.reshape(-1).to(torch.int32).contiguous()
    s_scatter_idx = torch.empty(pairs, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(pairs, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(num_experts, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(pairs, dtype=torch.int32, device=device)
    token_offset = torch.empty(tokens + 1, dtype=torch.int32, device=device)
    general_routing_router_metadata_triton(
        token_indices,
        expert_indices,
        tokens,
        num_experts,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        token_offset,
    )

    # Kernel rows are pairs sorted by expert: row p is token x_gather_idx[p] and
    # original pair s_scatter_idx[p].
    accents = op._accent_experts
    pair_experts = expert_indices.index_select(0, s_scatter_idx.long())
    slots, scales = slots.contiguous(), scales.float().contiguous()
    up_shift = expert_pair_shift(accents, "w13", x, x_gather_idx, pair_experts, slots, scales, gather_rows=True)
    hidden = x.new_empty(pairs, w13.shape[1] // 2)
    gemm_gated(
        x,
        w13.mT,
        C=up_shift,
        activation="swiglu",
        postact_out=hidden,
        cu_seqlens_m=expert_frequency_offset,
        A_idx=x_gather_idx,
        store_preact=False,
    )
    down_shift = expert_pair_shift(accents, "w2", hidden, x_gather_idx, pair_experts, slots, scales)
    y = gemm_add(hidden, w2.mT, down_shift, cu_seqlens_m=expert_frequency_offset)

    out = x.new_empty(tokens, w2.shape[1])
    _router_forward(
        y=y,
        o=out,
        topk_scores=scores.reshape(-1).float().contiguous(),
        s_reverse_scatter_idx=s_reverse_scatter_idx,
        num_activated_expert_per_token_offset=token_offset,
        varlen_K_max=num_experts,
        H=w2.shape[1],
        is_varlen_K=True,
    )
    return out
