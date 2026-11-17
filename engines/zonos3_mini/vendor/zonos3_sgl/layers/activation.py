from __future__ import annotations

import torch
import torch.nn.functional as F

# TTS Audio Suite Patch: flashinfer.silu_and_mul replaced by PyTorch.


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return (F.silu(x[..., :half].float()) * x[..., half:].float()).to(x.dtype)


__all__ = ["silu_and_mul"]
