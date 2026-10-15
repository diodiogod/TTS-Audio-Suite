"""BitsAndBytes load-time quantization for the MOSS language backbone."""

import importlib.util

import torch


QUANTIZATION_MODES = ("none", "bnb_int8", "bnb_nf4")


def normalize_quantization(value: str = "none") -> str:
    mode = str(value or "none").strip().lower()
    if mode not in QUANTIZATION_MODES:
        raise ValueError(f"Unsupported MOSS quantization: {value!r}. Choose {', '.join(QUANTIZATION_MODES)}.")
    return mode


def quantized_load_kwargs(mode: str, device: str, dtype: torch.dtype) -> dict:
    """Load directly onto the selected CUDA device without quantizing audio heads."""
    mode = normalize_quantization(mode)
    if mode == "none":
        return {}
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MOSS BitsAndBytes quantization requires an NVIDIA CUDA GPU. Select CUDA/auto or quantization=none.")
    for package in ("bitsandbytes", "accelerate"):
        if importlib.util.find_spec(package) is None:
            raise ImportError(f"MOSS {mode} requires '{package}' in the ComfyUI Python environment. Install it and restart ComfyUI.")

    from transformers import BitsAndBytesConfig

    # Keep the prediction heads and Local-Transformer audio components at the
    # selected floating dtype; only the language transformer is quantized.
    options = {
        "llm_int8_skip_modules": [
            "lm_heads",
            "local_transformer",
            "speech_embedding_to_local_mlp",
            "local_to_speech_embedding_mlps",
        ],
    }
    if mode == "bnb_int8":
        options["load_in_8bit"] = True
    else:
        options.update(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    return {
        "quantization_config": BitsAndBytesConfig(**options),
        "device_map": {"": str(device)},
    }
