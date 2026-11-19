"""Delay pattern of the codebook rows."""

from __future__ import annotations

import torch


def shear_up(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Remove delay: column j shifted up by j rows.

    This is the inverse of shear() - it removes the delay pattern applied
    during generation to align all codebook outputs for decoding.

    Args:
        x: Input tensor of shape (..., H, W) where H is frames and W is codebooks
        pad_id: Padding value to fill empty positions

    Returns:
        De-sheared tensor of same shape
    """
    H, W = x.shape[-2:]
    out = x.new_full(x.shape, pad_id)
    for j in range(W):
        if H > j:
            out[..., : H - j, j] = x[..., j:, j]
    return out


def undelay_window(buffer, start: int, end: int, n_codebooks: int, pad_id: int, *, device=None) -> torch.Tensor:
    """Undelayed output frames [start, end) of the delayed rows ``buffer`` (row r holds codebook j of frame r - j):
    the rows after ``end`` (up to n_codebooks - 1 of them) complete the last frames; int64 [end - start, n_codebooks]."""
    raw_end = min(end + n_codebooks - 1, len(buffer))
    return shear_up(torch.as_tensor(buffer[start:raw_end], dtype=torch.int64, device=device), pad_id)[: end - start]


def apply_cold_start_delay(row, total_generated: int, *, n_codebooks: int, pad_id: int, has_audio_prefix: bool) -> None:
    """Force the unsupervised initial delay triangle of generated row ``total_generated`` to audio padding, in place.

    In training, codebook ``c`` is shifted by ``c`` frames and targets in the leading triangle are ignored. Sampling
    those logits can create an immediate EOA or arbitrary residual codes. Codebooks whose delayed stream has not
    started are padded, while prompts that already contain an audio prefix continue as usual."""
    if not has_audio_prefix and total_generated < n_codebooks - 1:
        row[total_generated + 1 : n_codebooks] = pad_id
