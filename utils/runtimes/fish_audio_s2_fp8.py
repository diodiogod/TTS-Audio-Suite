"""Weight-only FP8 loader for the community Fish Audio S2 checkpoint."""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class FP8Linear(nn.Module):
    """Linear layer with FP8 row-scaled weight storage and BF16 compute."""

    def __init__(self, in_features, out_features, bias=False, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        kwargs = {"device": device} if device is not None else {}
        self.register_buffer(
            "qweight",
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, **kwargs),
        )
        self.register_buffer(
            "scale",
            torch.empty(out_features, 1, dtype=torch.float32, **kwargs),
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16, **kwargs))
            if bias else None
        )

    @property
    def weight(self):
        return self.qweight.to(torch.bfloat16) * self.scale.to(torch.bfloat16)

    def forward(self, inputs):
        weight = self.qweight.to(inputs.dtype) * self.scale.to(inputs.dtype)
        return F.linear(inputs, weight, self.bias)


def _replace_linear_layers(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            replacement = FP8Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device="meta",
            )
            if child.bias is not None:
                replacement.bias = child.bias
            setattr(module, name, replacement)
        else:
            _replace_linear_layers(child)


def load_fp8_model(checkpoint_path, context_length):
    """Load drbaph/s2-pro-fp8 without torchao or BF16 weight expansion."""
    from fish_speech.models.text2semantic.llama import (
        BaseModelArgs,
        DualARTransformer,
        _remap_fish_qwen3_omni_keys,
        precompute_freqs_cis,
    )
    from fish_speech.tokenizer import FishTokenizer
    from safetensors.torch import load_file

    checkpoint_path = Path(checkpoint_path)
    config = BaseModelArgs.from_pretrained(str(checkpoint_path))
    config.max_seq_len = int(context_length)
    tokenizer = FishTokenizer.from_pretrained(checkpoint_path)
    config.semantic_begin_id = tokenizer.semantic_begin_id
    config.semantic_end_id = tokenizer.semantic_end_id

    with torch.device("meta"):
        model = DualARTransformer(config)

    model.freqs_cis = precompute_freqs_cis(
        config.max_seq_len, config.head_dim, config.rope_base
    )
    model.causal_mask = torch.tril(
        torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
    )
    model.fast_freqs_cis = precompute_freqs_cis(
        config.num_codebooks, config.fast_head_dim, config.rope_base
    )

    weights = OrderedDict(
        load_file(str(checkpoint_path / "model.safetensors"), device="cpu")
    )
    weights = _remap_fish_qwen3_omni_keys(weights)
    fp8_data = {}
    fp8_scales = {}
    normal = {}
    for key, value in weights.items():
        if key.startswith("_buf."):
            continue
        if key.endswith(".scale") and key[:-6] in weights:
            fp8_scales[key[:-6]] = value
        elif value.dtype == torch.float8_e4m3fn:
            fp8_data[key] = value
        else:
            normal[key] = value

    model.load_state_dict(normal, strict=False, assign=True)
    _replace_linear_layers(model)

    restored = 0
    fp8_layer_count = 0
    for name, layer in model.named_modules():
        if not isinstance(layer, FP8Linear):
            continue
        fp8_layer_count += 1
        weight_name = f"{name}.weight"
        if weight_name not in fp8_data or weight_name not in fp8_scales:
            raise RuntimeError(f"FP8 checkpoint is missing weight or scale for '{name}'")
        layer.qweight = fp8_data[weight_name]
        scale = fp8_scales[weight_name]
        layer.scale = scale[:, None] if scale.ndim == 1 else scale
        restored += 1

    if restored != fp8_layer_count:
        raise RuntimeError(
            f"Restored {restored} FP8 layers but model contains {fp8_layer_count}"
        )

    model.tokenizer = tokenizer
    return model
