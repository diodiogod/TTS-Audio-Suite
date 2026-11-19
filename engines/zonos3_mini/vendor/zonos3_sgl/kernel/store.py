from __future__ import annotations

import torch

# TTS Audio Suite Patch: the JIT-compiled CUDA scatter (tvm_ffi, kernel/csrc/jit/store.cu) is replaced by index_copy_, which
# writes the same rows.


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    num_tokens = k_cache.shape[0]
    indices = indices.long()
    k_cache.view(num_tokens, -1).index_copy_(0, indices, k.reshape(indices.shape[0], -1))
    v_cache.view(num_tokens, -1).index_copy_(0, indices, v.reshape(indices.shape[0], -1))
