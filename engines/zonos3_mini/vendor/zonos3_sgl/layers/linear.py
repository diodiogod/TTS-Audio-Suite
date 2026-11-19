from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from zonos3_sgl.distributed import DistributedCommunicator, get_tp_info
from zonos3_sgl.utils.misc import divide_even

from .accent import apply_accent
from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [divide_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        GQA_ratio = divide_even(num_qo_heads, num_kv_heads)
        local_num_kv = divide_even(num_kv_heads, tp_info.size)
        full_isize = hidden_size
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = divide_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = divide_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = apply_accent(self, x, F.linear(x, self.weight, self.bias))
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class ChunkedLinear(BaseOP):
    """Linear layer with 3D weight [divisor, out_per_chunk, in_features].

    Used for fused projections (e.g., wkv for K/V, w_in for gate/up).
    The weight is stored as 3D but reshaped to 2D for the linear operation.
    """

    def __init__(self, in_features: int, out_features: int, divisor: int, has_bias: bool = False):
        if out_features % divisor != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by divisor ({divisor}).")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.divisor = int(divisor)
        self.out_per_chunk = int(out_features // divisor)
        # 3D weight: [divisor, out_per_chunk, in_features]
        self.weight = torch.empty(self.divisor, self.out_per_chunk, self.in_features)
        self.bias = torch.empty(self.out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape 3D weight to 2D for standard linear operation
        w2d = self.weight.view(self.out_features, self.in_features)
        return apply_accent(self, x, F.linear(x, w2d, self.bias))
