"""Optional BitsAndBytes quantization for Fish Audio S2."""

import torch
import torch.nn as nn


def _bitsandbytes():
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise ImportError(
            "Fish S2 BNB quantization requires optional 'bitsandbytes'. "
            "Install it into the existing ComfyUI environment without replacing Torch."
        ) from exc
    return bnb


def quantize_linear_layers(model, mode, device):
    """Replace every Fish linear projection with INT8 or NF4 storage."""
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("Fish S2 BNB quantization requires an NVIDIA CUDA GPU")
    if mode not in {"int8", "nf4"}:
        raise ValueError(f"Unsupported Fish S2 BNB mode: {mode}")

    bnb = _bitsandbytes()
    replaced = 0
    for module_name, child in list(model.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        parent = model
        parts = module_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)

        if mode == "int8":
            replacement = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
            )
            replacement.weight = bnb.nn.Int8Params(
                child.weight.detach().contiguous(),
                requires_grad=False,
                has_fp16_weights=False,
            ).to(device)
        else:
            replacement = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type="nf4",
            )
            replacement.weight = bnb.nn.Params4bit(
                child.weight.detach().contiguous(),
                requires_grad=False,
                compress_statistics=True,
                quant_type="nf4",
            ).to(device)

        if child.bias is not None:
            replacement.bias = nn.Parameter(
                child.bias.detach().clone(), requires_grad=False
            )
        setattr(parent, parts[-1], replacement)
        replaced += 1

    if replaced == 0:
        raise RuntimeError("Fish S2 BNB quantization found no linear layers")
    return model, replaced
