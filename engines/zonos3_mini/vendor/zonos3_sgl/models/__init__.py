from __future__ import annotations

from zonos3.models.config import ModelConfig

from .weight import load_checkpoint_weight
from .zonos3 import Zonos3ForCausalLM


def create_model(config: ModelConfig) -> Zonos3ForCausalLM:
    return Zonos3ForCausalLM(config.model_config, config.moe_backend)


__all__ = ["Zonos3ForCausalLM", "load_checkpoint_weight", "create_model"]
