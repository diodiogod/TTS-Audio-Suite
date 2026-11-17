from __future__ import annotations

from typing import Tuple

import torch

# TTS Audio Suite Patch: the JIT-compiled CUDA gather (tvm_ffi, kernel/csrc/jit/index.cu) is replaced by torch.index_select, which
# copies the same rows. Tensor parallelism (vocab_range) is not supported in this copy.


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if vocab_range is not None:
        raise NotImplementedError("vocabulary-parallel embedding is not supported in the TTS Audio Suite copy")
    return torch.index_select(weights, 0, indices.long(), out=output)
