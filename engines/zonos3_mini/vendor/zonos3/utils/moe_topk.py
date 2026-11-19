from collections.abc import Mapping
from typing import Any


def normalize_special_topk_layers(
    special_topk_layers: Mapping[Any, Any] | None,
) -> dict[int, int] | None:
    if special_topk_layers is None:
        return None

    normalized: dict[int, int] = {}
    for layer_idx, topk in special_topk_layers.items():
        layer_idx = int(layer_idx)
        topk = int(topk)
        if topk < 1:
            raise ValueError(f"special_topk_layers[{layer_idx}] must be >= 1, got {topk}")
        normalized[layer_idx] = topk
    return normalized


def resolve_layer_topk(
    default_topk: int,
    special_topk_layers: Mapping[Any, Any] | None,
    layer_idx: int,
) -> int:
    topk = default_topk
    if special_topk_layers is not None:
        topk = special_topk_layers.get(layer_idx, special_topk_layers.get(str(layer_idx), default_topk))

    topk = int(topk)
    if topk < 1:
        raise ValueError(f"top-k for layer {layer_idx} must be >= 1, got {topk}")
    return topk
