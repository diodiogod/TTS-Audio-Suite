"""Inline IndexTTS-2 emotion controls and persistent preset storage."""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any, Dict, List, Tuple

from utils.text.segment_parameters import INDEX_TTS_EMOTIONS

MAX_EMOTION_VALUE = 1.2
PRESET_FILENAME = "emotion_presets.json"


def get_emotion_preset_path() -> str:
    import folder_paths
    directory = os.path.join(folder_paths.models_dir, "TTS", "IndexTTS")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, PRESET_FILENAME)


def load_emotion_presets() -> Dict[str, Dict[str, Any]]:
    path = get_emotion_preset_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("IndexTTS emotion presets must be a JSON object")
    return data


def save_emotion_presets(presets: Dict[str, Dict[str, Any]]) -> str:
    if not isinstance(presets, dict):
        raise ValueError("presets must be an object")
    for name, preset in presets.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("preset names may contain only letters, numbers, underscores, and hyphens")
        if not isinstance(preset, dict) or preset.get("type") not in {"text", "vector"}:
            raise ValueError(f"preset '{name}' must be a text or vector preset")
        if preset["type"] == "text" and not str(preset.get("description", "")).strip():
            raise ValueError(f"text preset '{name}' requires a description")
        if preset["type"] == "vector":
            values = preset.get("values")
            if not isinstance(values, list) or len(values) != len(INDEX_TTS_EMOTIONS):
                raise ValueError(f"vector preset '{name}' requires exactly 8 values")
            preset["values"] = [round(_clamp(float(value)), 2) for value in values]
    path = get_emotion_preset_path()
    directory = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(prefix="emotion_presets_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(presets, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _clamp(value: float) -> float:
    return max(0.0, min(MAX_EMOTION_VALUE, value))


def _base_vector(config: Dict[str, Any]) -> List[float]:
    vector = config.get("emotion_vector") or [0.0] * len(INDEX_TTS_EMOTIONS)
    if len(vector) != len(INDEX_TTS_EMOTIONS):
        return [0.0] * len(INDEX_TTS_EMOTIONS)
    return [_clamp(float(value)) for value in vector]


def _parse_full_vector(raw: str, base: List[float]) -> List[float]:
    values = [part.strip() for part in raw.split(",")]
    if len(values) != len(INDEX_TTS_EMOTIONS):
        raise ValueError(
            f"vector requires exactly 8 values in this order: {', '.join(INDEX_TTS_EMOTIONS)}"
        )
    signed = [value.startswith(("+", "-")) for value in values]
    if any(signed) and not all(signed):
        raise ValueError("relative vector values must all include an explicit + or - sign")
    numbers = [float(value) for value in values]
    if all(signed):
        return [_clamp(current + delta) for current, delta in zip(base, numbers)]
    return [_clamp(value) for value in numbers]


def _resolve_text_control(raw: str) -> Tuple[str, bool]:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        text = value[1:-1].strip()
        if not text:
            raise ValueError("quoted emotion text cannot be empty")
        return text, "{seg}" in text

    preset = load_emotion_presets().get(value)
    if not preset:
        raise ValueError(f"unknown IndexTTS emotion preset: {value}")
    if preset.get("type") != "text":
        raise ValueError(f"emotion preset '{value}' is not a text preset")
    text = str(preset.get("description", "")).strip()
    if not text:
        raise ValueError(f"emotion preset '{value}' has no description")
    return text, "{seg}" in text


def resolve_inline_emotion(config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Resolve parsed inline controls, returning (new_config, has_override)."""
    resolved = dict(config)
    vector_raw = resolved.pop("emotion_vector_inline", None)
    text_raw = resolved.pop("emotion_text_inline", None)
    named = {
        name: resolved.pop(f"emotion_{name}_inline")
        for name in INDEX_TTS_EMOTIONS
        if f"emotion_{name}_inline" in resolved
    }
    if text_raw is not None and (vector_raw is not None or named):
        raise ValueError("emotion text/presets cannot be combined with vector controls in one tag")
    if vector_raw is not None and named:
        raise ValueError("full vector and named emotion values cannot be combined in one tag")

    if text_raw is not None:
        text, dynamic = _resolve_text_control(text_raw)
        resolved.update(
            emotion_vector=None,
            use_emotion_text=True,
            emotion_text=text,
            is_dynamic_template=dynamic,
            inline_emotion_override=True,
        )
        return resolved, True

    if vector_raw is None and not named:
        return resolved, False

    vector = _base_vector(resolved)
    if vector_raw is not None:
        vector = _parse_full_vector(vector_raw, vector)
    else:
        for name, raw in named.items():
            raw = str(raw).strip()
            index = INDEX_TTS_EMOTIONS.index(name)
            number = float(raw)
            vector[index] = _clamp(vector[index] + number) if raw.startswith(("+", "-")) else _clamp(number)

    resolved.update(
        emotion_vector=vector,
        use_emotion_text=False,
        emotion_text=None,
        is_dynamic_template=False,
        inline_emotion_override=True,
    )
    return resolved, True
