from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml


_AUX_MODELS_YAML = Path(__file__).resolve().parents[2] / "docs/Dev reports/tts_audio_suite_aux_models.yaml"


@lru_cache(maxsize=1)
def load_aux_model_data() -> Dict[str, Any]:
    with open(_AUX_MODELS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_category(category_id: str) -> Dict[str, Any]:
    data = load_aux_model_data()
    for category in data.get("categories", []):
        if category.get("id") == category_id:
            return category
    raise KeyError(f"Unknown auxiliary model category: {category_id}")


def get_models(category_id: str) -> List[Dict[str, Any]]:
    return list(get_category(category_id).get("models", []))


def get_punctuation_models() -> List[Dict[str, Any]]:
    return get_models("punctuation_truecase")


def get_punctuation_model_by_label(label: str) -> Dict[str, Any]:
    for model in get_punctuation_models():
        if model.get("dropdown_label") == label:
            return model
    raise KeyError(f"Unknown punctuation model: {label}")


def get_punctuation_dropdown_labels() -> List[str]:
    return [model["dropdown_label"] for model in get_punctuation_models()]
