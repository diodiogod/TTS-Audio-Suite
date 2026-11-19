from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        self._use_training_rope = os.environ.get("ZONOS3_USE_TRAINING_ROPE") == "1"
        self._freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        # The backbone ablations use 32-wide heads, while the FlashInfer 0.5.3
        # RoPE kernel bundled with this environment only supports 64 and up.
        # Keep its optimized path for supported sizes and use the equivalent
        # PyTorch implementation for D=32.
        if self.head_size not in (32, 64, 128, 256, 512):
            raise ValueError(f"Unsupported RoPE head size: {self.head_size}")

        # TTS Audio Suite Patch: the flashinfer RoPE kernel is not used; every head size takes the PyTorch path.

    def _apply_torch_rope_inplace(
        self,
        positions: torch.Tensor,
        tensor: torch.Tensor,
        *,
        is_neox: bool,
    ) -> None:
        token_count = tensor.shape[0]
        heads = tensor.reshape(token_count, -1, self.head_size)
        cos_sin = self._cos_sin_cache[positions.to(torch.long)]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        source = heads.float()
        if is_neox:
            first, second = source.chunk(2, dim=-1)
            rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
        else:
            pairs = source.reshape(token_count, -1, self.head_size // 2, 2)
            even, odd = pairs.unbind(dim=-1)
            rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
        heads.copy_(rotated.to(dtype=heads.dtype))

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        is_neox: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._use_training_rope:
            if is_neox:
                raise ValueError("Training RoPE requires interleaved pairs")
            freqs = self._freqs_cis[positions.to(torch.long), None, :]

            def rotate(tensor):
                heads = tensor.reshape(tensor.shape[0], -1, self.head_size)
                pairs = torch.view_as_complex(heads.float().unflatten(-1, (-1, 2)))
                result = torch.view_as_real(pairs * freqs).flatten(-2)
                return result.to(dtype=tensor.dtype).reshape_as(tensor)

            return rotate(query), rotate(key)
        self._apply_torch_rope_inplace(positions, query, is_neox=is_neox)
        self._apply_torch_rope_inplace(positions, key, is_neox=is_neox)
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    raise ValueError(f"Unsupported rotary scaling in TTS release: {rope_scaling!r}")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@lru_cache()
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError("We cannot use meta device for rope. Please call set_rope_device() first.")
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
