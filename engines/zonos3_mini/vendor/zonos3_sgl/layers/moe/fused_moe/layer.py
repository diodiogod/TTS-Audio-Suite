from typing import List, Optional

import torch
import torch.nn as nn
from zonos3_sgl.distributed import DistributedCommunicator, get_tp_info
from zonos3_sgl.layers.base import BaseOP
from zonos3_sgl.layers.moe.fused_moe.fused_moe_impl import fused_experts, fused_moe
from zonos3_sgl.utils.misc import divide_even


class FusedMoEMethod(nn.Module):
    def __init__(self):
        super().__init__()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        tp_size: int = 1,
        tp_rank: int = 0,
        **extra_weight_attrs,
    ):
        intermediate_size_per_partition = divide_even(intermediate_size, tp_size)

        gate_up_proj = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        setattr(layer, "gate_up_proj", gate_up_proj)

        down_proj = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        setattr(layer, "down_proj", down_proj)

    def forward(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        block_shape: Optional[List[int]] = None,
        inplace: bool = True,
        no_combine: bool = False,
    ) -> torch.Tensor:
        return fused_moe(
            hidden_states=x,
            w1=layer.gate_up_proj,
            w2=layer.down_proj,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            inplace=inplace,
            activation=activation,
            no_combine=no_combine,
        )


class FusedMoE(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: Optional[int] = None,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        tp_size: Optional[int] = None,
        prefix: str = "",
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.params_dtype = params_dtype
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_size = tp_info.size if tp_size is None else tp_size
        self.tp_rank = tp_info.rank if tp_size is None else 0

        assert self.intermediate_size % self.tp_size == 0, (
            f"Intermediate size ({self.intermediate_size}) must be divisible by tp_size ({self.tp_size})"
        )

        self.intermediate_size_per_partition = self.intermediate_size // self.tp_size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.inplace = inplace
        self.no_combine = no_combine
        self.layer_id = layer_id
        self.prefix = prefix

        self.fused_moe_method: Optional[FusedMoEMethod] = FusedMoEMethod()
        self.fused_moe_method.create_weights(
            self,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        *,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
    ):
        assert self.fused_moe_method is not None

        if topk_weights is None and topk_ids is None:
            if router_logits is None:
                raise ValueError("router_logits must be provided when top-k routing is not precomputed.")
            final_hidden_states = self.fused_moe_method.forward(
                layer=self,
                x=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                inplace=self.inplace,
                no_combine=self.no_combine,
            )
        else:
            if topk_weights is None or topk_ids is None:
                raise ValueError("topk_weights and topk_ids must be provided together.")
            topk_weights = topk_weights.to(dtype=torch.float32)
            if self.renormalize:
                topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
            final_hidden_states = fused_experts(
                hidden_states=hidden_states,
                w1=self.gate_up_proj,
                w2=self.down_proj,
                topk_weights=topk_weights.contiguous(),
                topk_ids=topk_ids.to(dtype=torch.int32).contiguous(),
                inplace=self.inplace,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                no_combine=self.no_combine,
            )

        if self.tp_size > 1:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)

        return final_hidden_states
