from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

EOA_TOKEN = 1024  # End of audio token


@dataclass(frozen=True)
class ClampedSampling:
    """One request's sampler settings as every sampler applies them (:func:`clamp_sampling`)."""

    temperature: float
    topk: int
    top_p: float
    min_p: float
    repetition_window: int
    repetition_penalty: float
    repetition_codebooks: int


def clamp_sampling(p, *, n_codebooks: int, codebook_size: int) -> ClampedSampling:
    """Request sampling fields (``temperature``, ``topk``, ``top_p``, ``min_p``, ``repetition_window``,
    ``repetition_penalty``, ``repetition_codebooks``) clamped to their usable ranges: topk < 1 = the whole codebook,
    repetition_codebooks < 0 = every codebook."""
    codebooks = int(p.repetition_codebooks)
    return ClampedSampling(
        temperature=max(float(p.temperature), 1e-6),
        topk=int(p.topk) if int(p.topk) >= 1 else int(codebook_size),
        top_p=min(max(float(p.top_p), 0.0), 1.0),
        min_p=max(float(p.min_p), 0.0),
        repetition_window=max(int(p.repetition_window), 0),
        repetition_penalty=max(float(p.repetition_penalty), 1.0),
        repetition_codebooks=n_codebooks if codebooks < 0 else min(max(codebooks, 0), n_codebooks),
    )


def active_repetition_window(c: ClampedSampling, available: int) -> int:
    """The repetition-penalty window over the ``available`` history rows (0 = no penalty)."""
    if c.repetition_window > 0 and c.repetition_penalty > 1.0 and c.repetition_codebooks > 0:
        return min(c.repetition_window, available)
    return 0


def sample_logits_in_fp32(model_config) -> bool:
    """GDN-family checkpoints (attention_pattern != "global") decode eagerly and are sampled from float32 logits,
    like the training generator; ZONOS3_SAMPLE_FP32_LOGITS=1 forces it for every model."""
    import os

    pattern = str(getattr(model_config, "attention_pattern", "global"))
    return pattern != "global" or os.environ.get("ZONOS3_SAMPLE_FP32_LOGITS") == "1"


def apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Apply nucleus (top-p) sampling to probabilities."""
    if p <= 0.0 or p >= 1.0:
        return probs
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort = probs_sort.masked_fill(mask, 0.0)
    probs = probs.scatter(-1, probs_idx, probs_sort)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return probs


def apply_min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    """Apply min-p filtering to probabilities."""
    if min_p <= 0.0:
        return probs
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    tokens_to_remove = probs < (min_p * top_probs)
    probs = probs.masked_fill(tokens_to_remove, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return probs


def apply_repetition_penalty(
    logits: torch.Tensor,
    repetition_token_ids: torch.Tensor | None,
    repetition_penalties: torch.Tensor | None,
) -> torch.Tensor:
    """Apply a per-sequence repetition penalty to per-codebook logits.

    repetition_token_ids is shaped (B, n_codebooks, window). A token id in one
    codebook only penalizes the matching codebook's logits.
    """
    if repetition_token_ids is None or repetition_penalties is None:
        return logits
    if repetition_token_ids.numel() == 0:
        return logits

    B, C, V = logits.shape
    safe_token_ids = repetition_token_ids.clamp(min=0, max=V - 1).long()
    valid = (repetition_token_ids >= 0) & (repetition_token_ids < V)

    counts = torch.zeros((B, C, V), dtype=torch.int32, device=logits.device)
    counts.scatter_add_(-1, safe_token_ids, valid.to(torch.int32))
    repeated = counts > 0

    penalties = repetition_penalties.view(B, 1, 1).clamp(min=1.0)
    adjusted_logits = torch.where(logits > 0, logits / penalties, logits * penalties)
    return torch.where(repeated, adjusted_logits, logits)


def _on_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """``t`` on ``device`` without blocking the host: a CPU tensor bound for CUDA goes through pinned memory (a
    pageable source may make the driver synchronize with the stream)."""
    if t.device == device:
        return t
    if device.type == "cuda" and t.device.type == "cpu":
        t = t if t.is_pinned() else t.pin_memory()
        return t.to(device, non_blocking=True)
    return t.to(device)


_debug_sample_count = 0


def _compute_logit_stats_debug(logits: torch.Tensor) -> dict:
    """Compute per-codebook stats matching reference format.

    Args:
        logits: (B, n_codebooks, vocab) - uses first batch element

    Returns:
        Dict with per-codebook stats: entropy, top1_prob, margin, top5_ids
    """
    lg = logits[0].float()  # (n_codebooks, vocab)

    probs = F.softmax(lg, dim=-1)
    log_probs = F.log_softmax(lg, dim=-1)

    # Entropy: -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)

    # Top-k probs and ids
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    top1_prob = sorted_probs[:, 0]
    top5_ids = sorted_indices[:, :5]

    # Logit margin (top1 - top2)
    sorted_logits, _ = lg.sort(dim=-1, descending=True)
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]

    return {
        "entropy": entropy.tolist(),
        "top1_prob": top1_prob.tolist(),
        "margin": margin.tolist(),
        "top5_ids": top5_ids.tolist(),
    }


def sample_tts(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    *,
    codebook_size: int,
    audio_pad_id: int,
    stop_token_ids: torch.Tensor,
    repetition_token_ids: torch.Tensor | None = None,
    repetition_penalties: torch.Tensor | None = None,
    codebook_sizes: tuple[int, ...] | list[int] | None = None,
    text_vocab: int = 0,
    generators: list[torch.Generator | None] | None = None,
    stop_blocked: torch.Tensor | None = None,
    stop_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Functional interface for TTS sampling.

    Args:
        logits: Shape (B, n_codebooks, vocab_size)
        temperatures: Per-sequence temperatures (B,)
        top_ks: Per-sequence top-k values (B,)
        top_ps: Per-sequence top-p values (B,)
        min_ps: Per-sequence min-p values (B,)
        codebook_size: Shared ordinary-code limit; ids below it are codec codes,
            ids at or above it are special (EOA, padding, optional span-end).
        audio_pad_id: Audio padding id; never sampled.
        stop_token_ids: Per-sequence stop id (B,), the only special id each row
            may sample (EOA for utterances, span-end for edit spans).
        repetition_token_ids: Recent token ids per sequence/codebook (B, C, W)
        repetition_penalties: Per-sequence repetition penalties (B,), 1.0 disables
        codebook_sizes: Valid ordinary-code range for each codebook.
        text_vocab: Text vocabulary size (appended as placeholder)
        stop_blocked: Optional (B, C) bool mask of the codebooks whose stop token is
            not allowed yet (a request's min_frames); None leaves every stop id allowed.
        stop_bias: Optional (B,) float added to each row's stop-id logit in every
            codebook before the filters (LACI's stop boost once the text is read); None adds nothing.

    The values of temperatures, top_ks, top_ps, min_ps and stop_token_ids pick host-side branches (greedy, the
    batch's top-k / top-p / min-p, each row's vocabulary width): pass them as CPU tensors and sampling never waits on
    the GPU (a server can then queue the next step behind it); on the logits' device each read blocks on a copy.

    Returns:
        (B, n_codebooks + 1) int32 on the logits' device: each row's codes [cb0, ..., cb8] and the text placeholder
    """
    global _debug_sample_count
    B, C, V = logits.shape
    device = logits.device
    codebook_size = int(codebook_size)
    audio_pad_id = int(audio_pad_id)
    if not 0 < codebook_size < V or not codebook_size <= audio_pad_id < V:
        raise ValueError(f"Invalid codebook_size={codebook_size} / audio_pad_id={audio_pad_id} for vocabulary {V}")
    if tuple(stop_token_ids.shape) != (B,):
        raise ValueError(f"Expected stop_token_ids of shape ({B},), received {tuple(stop_token_ids.shape)}")
    host_stop_ids = stop_token_ids.tolist()

    # Padding targets are ignored during training, so their logits are
    # unconstrained and padding must never be sampled as an audio code. Of the
    # end tokens only the row's own stop id stays available: it is a supervised
    # target in every delayed codebook during the autoregressive flush, while
    # the other end token (EOA vs span-end) would end the wrong kind of segment.
    token_ids = torch.arange(V, device=device).view(1, 1, V)
    stop_ids = _on_device(stop_token_ids, device).to(token_ids.dtype).view(B, 1, 1)
    invalid = (token_ids == audio_pad_id) | ((token_ids >= codebook_size) & (token_ids != stop_ids))

    if codebook_sizes is not None:
        sizes = tuple(int(item) for item in codebook_sizes)
        if len(sizes) != C:
            raise ValueError(f"Expected {C} per-codebook sizes, received {len(sizes)}")
        if any(size <= 0 or size > codebook_size for size in sizes):
            raise ValueError(f"Invalid per-codebook sizes {sizes!r} for codebook size {codebook_size}")
        limits = _on_device(torch.tensor(sizes), device).view(1, C, 1)
        invalid = invalid | ((token_ids >= limits) & (token_ids < codebook_size))
    if stop_blocked is not None:
        invalid = invalid | ((token_ids == stop_ids) & _on_device(stop_blocked, device).view(B, C, 1))
    logits = logits.masked_fill(invalid, float("-inf"))
    if stop_bias is not None:
        logits = logits + (token_ids == stop_ids) * _on_device(stop_bias, device).to(logits.dtype).view(B, 1, 1)

    # Debug: print detailed stats matching reference format for first 10 samples
    # Guard with is_current_stream_capturing to avoid CUDA graph capture errors
    if (
        logger.isEnabledFor(logging.DEBUG)
        and _debug_sample_count < 10
        and EOA_TOKEN < V
        and not torch.cuda.is_current_stream_capturing()
    ):
        _debug_sample_count += 1
        step = _debug_sample_count

        stats = _compute_logit_stats_debug(logits)

        logger.debug("Step %d per-codebook stats:", step)
        for cb in range(C):
            top5 = stats["top5_ids"][cb]
            logger.debug(
                "  CB%d: entropy=%.4f, top1_prob=%.4f, top1_id=%d, margin=%.2f, top5=%s",
                cb,
                stats["entropy"][cb],
                stats["top1_prob"][cb],
                top5[0],
                stats["margin"][cb],
                top5,
            )

        probs_raw = F.softmax(logits[0], dim=-1)
        eoa_probs = probs_raw[:, EOA_TOKEN].tolist()
        eoa_logits = logits[0, :, EOA_TOKEN].tolist()
        logger.debug(
            "Step %d EOA(1024) logits: [%s]",
            step,
            ", ".join(f"{x:.2f}" for x in eoa_logits),
        )
        logger.debug(
            "Step %d EOA(1024) probs:  [%s]",
            step,
            ", ".join(f"{x:.4f}" for x in eoa_probs),
        )

    logits = apply_repetition_penalty(
        logits,
        repetition_token_ids=repetition_token_ids,
        repetition_penalties=repetition_penalties,
    )

    # Check if all greedy (temperature <= 0)
    if bool((temperatures <= 0).all()):
        next_ids = torch.argmax(logits, dim=-1)
    else:
        # Per-sequence temperature scaling: (B,) -> (B, 1, 1) for broadcasting
        temp_broadcast = _on_device(temperatures, device).view(B, 1, 1).clamp(min=1e-8)
        logits = logits / temp_broadcast
        logits_flat = logits.view(B * C, V)

        # Use min top_k across batch for efficient single topk call
        top_k = int(top_ks.min().item())
        if 0 < top_k < V:
            values, _ = torch.topk(logits_flat, top_k, dim=-1)
            kth = values[..., -1].unsqueeze(-1)
            mask = logits_flat < kth
            logits_flat = logits_flat.masked_fill(mask, float("-inf"))

        probs = F.softmax(logits_flat, dim=-1)

        # Apply top_p (use max across batch - more restrictive)
        top_p = float(top_ps.max().item())
        if 0.0 < top_p < 1.0:
            probs = apply_top_p(probs, top_p)

        # Apply min_p (use max across batch - more restrictive)
        min_p = float(min_ps.max().item())
        if min_p > 0.0:
            probs = apply_min_p(probs, min_p)

        # An aggressive min_p (>1) or top_p can remove every token in a row,
        # leaving an all-zero distribution. torch.multinomial would then raise a
        # CUDA device-side assert that aborts the scheduler, so fall back to
        # greedy (argmax) for any fully-masked row. Applied unconditionally (a
        # no-op select on valid rows) so the host never reads the mask.
        invalid_rows = probs.sum(dim=-1) <= 0
        greedy = logits_flat.argmax(dim=-1)
        fallback = torch.zeros_like(probs)
        fallback.scatter_(-1, greedy.unsqueeze(-1), 1.0)
        probs = torch.where(invalid_rows.unsqueeze(-1), fallback, probs)

        # Draw each row over the narrowest vocabulary prefix that holds its allowed ids:
        # codes, EOA and pad (the base vocabulary), plus the trailing span-end column only
        # for span rows. torch.multinomial draws one exponential per column, so an
        # utterance on an edit checkpoint then consumes exactly the random numbers of the
        # same checkpoint without the span-end row: same seed, same audio.
        widths = [max(int(stop), audio_pad_id) + 1 for stop in host_stop_ids]
        # Use per-request generators for deterministic sampling when seeds are set
        has_generators = generators and any(g is not None for g in generators)
        if has_generators and B == 1 and generators[0] is not None:
            # Fast path: single request with a generator
            next_ids = torch.multinomial(
                probs[:, : widths[0]].contiguous(), num_samples=1, generator=generators[0]
            ).view(B, C)
        elif has_generators:
            # Batched with mixed generators: sample per-row
            # probs is (B*C, V), each request owns C consecutive rows
            rows = []
            for i in range(B):
                row_probs = probs[i * C : (i + 1) * C, : widths[i]].contiguous()  # (C, width)
                gen = generators[i] if i < len(generators) else None
                rows.append(torch.multinomial(row_probs, num_samples=1, generator=gen))
            next_ids = torch.cat(rows, dim=0).view(B, C)
        else:
            next_ids = torch.multinomial(probs[:, : max(widths)].contiguous(), num_samples=1).view(B, C)

    # Debug: print sampled tokens for first 10 samples
    if (
        logger.isEnabledFor(logging.DEBUG)
        and _debug_sample_count <= 10
        and not torch.cuda.is_current_stream_capturing()
    ):
        step = _debug_sample_count
        next_ids_list = next_ids.tolist()
        logger.debug("Step %d sampled: %s", step, next_ids_list[0] if next_ids_list else [])
        for i, row in enumerate(next_ids_list):
            eoa_positions = [j for j, t in enumerate(row) if t == EOA_TOKEN]
            if eoa_positions:
                logger.debug(
                    "Step %d *** EOA SAMPLED at CB %s for batch %d ***",
                    step,
                    eoa_positions,
                    i,
                )

    # Unpacked rows: the audio codes, then the text placeholder
    placeholder = torch.full((B, 1), int(text_vocab), dtype=torch.int32, device=device)
    return torch.cat((next_ids.to(torch.int32), placeholder), dim=1)
