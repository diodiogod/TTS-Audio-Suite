from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch

from zonos3_sgl.distributed import get_tp_info
from zonos3_sgl.utils.misc import divide_up


def _shard_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Shard Zonos3 checkpoint tensors for tensor parallelism."""
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size

    split_dim_0 = [
        ".wq",
        ".gater",
    ]
    split_dim_1 = [
        ".wo",
        ".w_out",
    ]

    gdn_prefixes = tuple(
        key.removesuffix("in_proj.weight") for key in state_dict if key.endswith(".attention.in_proj.weight")
    )
    for key, value in state_dict.items():
        if gdn_prefixes and key.startswith(gdn_prefixes):
            shard_state_dict[key] = value  # Entire original GDN replicated, including wo.
        elif ".attention.temp" in key:
            shard_state_dict[key] = value.chunk(n, dim=1)[r]
        elif ".wkv." in key:
            if value.dim() == 3:
                shard_state_dict[key] = value.chunk(n, dim=1)[r]
            else:
                shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif ".w_in." in key:
            if value.dim() == 3:
                shard_state_dict[key] = value.chunk(n, dim=1)[r]
            else:
                shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif ".experts." in key:
            if "gate_up_proj" in key or "w1" in key or "w3" in key:
                if value.dim() == 3:
                    shard_state_dict[key] = value.chunk(n, dim=1)[r]
                else:
                    shard_state_dict[key] = value.chunk(n, dim=0)[r]
            elif "down_proj" in key or "w2" in key:
                if value.dim() == 3:
                    shard_state_dict[key] = value.chunk(n, dim=2)[r]
                else:
                    shard_state_dict[key] = value.chunk(n, dim=1)[r]
            else:
                shard_state_dict[key] = value
        elif any(key.count(sub) for sub in split_dim_0):
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif any(key.count(sub) for sub in split_dim_1):
            shard_state_dict[key] = value.chunk(n, dim=1)[r]
        elif key.count("embedders") or key.count("multi_output"):
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = divide_up(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        else:
            shard_state_dict[key] = value
    return shard_state_dict


def _normalize_zonos3_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Normalize training checkpoint keys for inference."""
    from zonos3_sgl.utils.logger import init_logger

    logger = init_logger(__name__)

    keys_to_remap = []
    keys_to_remove = []
    for key in list(state_dict.keys()):
        if ".parametrizations." in key and ".original" in key:
            new_key = key.replace(".parametrizations.", ".").replace(".original", "")
            keys_to_remap.append((key, new_key))
        elif ".router.ent_denom" in key or ".router.normalized_entropy" in key:
            keys_to_remove.append(key)

    for old_key, new_key in keys_to_remap:
        logger.info("Remapping parametrized key: %s -> %s", old_key, new_key)
        value = state_dict.pop(old_key)
        if new_key.endswith(".router.balancing_biases"):
            value = value.float()
            value = value - value.mean()
        state_dict[new_key] = value

    for key in keys_to_remove:
        logger.info("Removing training-only key: %s", key)
        state_dict.pop(key)

    return state_dict


def _single_file_checkpoint(path: str) -> Path | None:
    """Return the consolidated single-file checkpoint inside a directory, if any."""
    checkpoint_dir = Path(path)
    if not checkpoint_dir.is_dir():
        return None
    for name in ("model.pth", "model.pt", "consolidated/consolidated.pth"):
        candidate = checkpoint_dir / name
        if candidate.is_file():
            return candidate
    return None


def _load_torch_weight(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in state:
        return state["model"]
    return state


def load_checkpoint_weight(model_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load a Zonos3 TTS checkpoint (consolidated model.pth + params.json).

    Accepts a local path or a Hugging Face repo id (e.g. ``hf://Zyphra/ZONOS3-Mini/lm/zonos3_mini``).
    """
    from zonos3.utils.hf import resolve_model_path

    from zonos3_sgl.utils.logger import init_logger

    logger = init_logger(__name__)
    model_path = resolve_model_path(model_path)
    single_file = _single_file_checkpoint(model_path)
    if single_file is not None:
        logger.info("Loading consolidated checkpoint from %s", single_file)
        state_dict = _load_torch_weight(str(single_file))
    elif os.path.isfile(model_path) and model_path.endswith((".pt", ".pth")):
        state_dict = _load_torch_weight(model_path)
    else:
        raise ValueError(
            f"Unsupported checkpoint at {model_path!r}. Expected a directory "
            "containing model.pth (with params.json) or a direct .pt/.pth file."
        )

    if get_tp_info().size > 1:
        state_dict = _shard_state_dict(state_dict)

    state_dict = _normalize_zonos3_state_dict(state_dict)
    return {k: v.to(device) for k, v in state_dict.items()}
