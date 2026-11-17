import os
from typing import Tuple

import torch
import torch.nn.functional as F

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self._size = size
        self._use_training_rmsnorm = os.environ.get("ZONOS3_USE_TRAINING_RMSNORM", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.weight = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TTS Audio Suite Patch: always the training arithmetic (F.rms_norm); the flashinfer kernel is not used.
        return F.rms_norm(x, (self._size,), self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        x.copy_(F.rms_norm(x, (self._size,), self.weight, self.eps))


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float, elementwise_affine: bool = True) -> None:
        self.eps = eps
        self._use_training_rmsnorm = os.environ.get("ZONOS3_USE_TRAINING_RMSNORM", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.elementwise_affine = elementwise_affine
        self._size = size

        if elementwise_affine:
            self.weight = torch.empty(size)
        # When elementwise_affine=False, we use a ones buffer created lazily
        # to ensure correct device/dtype

        self._ones_buffer: torch.Tensor | None = None

    def _get_weight(self, x: torch.Tensor) -> torch.Tensor:
        if self.elementwise_affine:
            return self.weight
        # Use cached ones buffer, recreate if needed for device/dtype match
        if self._ones_buffer is None or self._ones_buffer.device != x.device:
            self._ones_buffer = torch.ones(self._size, device=x.device, dtype=x.dtype)
        return self._ones_buffer

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = self._get_weight(x)
        # TTS Audio Suite Patch: always the training arithmetic (F.rms_norm); the flashinfer kernels are not used.
        combined = x if residual is None else x + residual
        return (
            F.rms_norm(combined, (self._size,), weight, self.eps),
            combined,
        )
