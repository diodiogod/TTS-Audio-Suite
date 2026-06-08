import glob
import json
import os
from contextlib import contextmanager

from torch import nn

from transformers.modeling_utils import PreTrainedModel
from transformers import AutoConfig

from .configuration_higgs_audio import HiggsAudioConfig


_HIGGS_CHECKPOINT_COMPLETENESS_CACHE = {}


class HiggsAudioPreTrainedModel(PreTrainedModel):
    config_class = HiggsAudioConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.init_std if hasattr(self.config, "init_std") else self.config.audio_encoder_config.init_std

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


def _get_local_checkpoint_keys(checkpoint_dir: str) -> set[str]:
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        return set(index_data.get("weight_map", {}).keys())

    key_set: set[str] = set()
    try:
        from safetensors import safe_open
    except ImportError:
        return key_set

    for shard_path in sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))):
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            key_set.update(shard.keys())
    return key_set


def _is_complete_local_higgs_checkpoint(checkpoint_dir: str) -> bool:
    cached = _HIGGS_CHECKPOINT_COMPLETENESS_CACHE.get(checkpoint_dir)
    if cached is not None:
        return cached

    checkpoint_keys = _get_local_checkpoint_keys(checkpoint_dir)
    if not checkpoint_keys:
        _HIGGS_CHECKPOINT_COMPLETENESS_CACHE[checkpoint_dir] = False
        return False

    original_init_weights = HiggsAudioPreTrainedModel._init_weights
    try:
        # TTS Audio Suite patch: build the key map without paying the 3B random-init cost.
        HiggsAudioPreTrainedModel._init_weights = lambda self, module: None
        config = AutoConfig.from_pretrained(checkpoint_dir)
        from .modeling_higgs_audio import HiggsAudioModel

        model = HiggsAudioModel(config)
        model_keys = set(model.state_dict().keys())
        is_complete = model_keys == checkpoint_keys
        _HIGGS_CHECKPOINT_COMPLETENESS_CACHE[checkpoint_dir] = is_complete
        return is_complete
    finally:
        HiggsAudioPreTrainedModel._init_weights = original_init_weights


@contextmanager
def skip_hf_missing_key_initialization_if_checkpoint_complete(checkpoint_dir: str):
    """
    TTS Audio Suite patch:
    Skip HF's expensive `_initialize_missing_keys()` pass when the local Higgs checkpoint
    already contains the full model key set. On transformers 5 this unnecessary pass can
    spend several minutes random-initializing a 3B model despite a complete checkpoint.
    """
    if not _is_complete_local_higgs_checkpoint(checkpoint_dir):
        yield
        return

    original_initialize_missing_keys = HiggsAudioPreTrainedModel._initialize_missing_keys
    try:
        HiggsAudioPreTrainedModel._initialize_missing_keys = lambda self, is_quantized: None
        yield
    finally:
        HiggsAudioPreTrainedModel._initialize_missing_keys = original_initialize_missing_keys
