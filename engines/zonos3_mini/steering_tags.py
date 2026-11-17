"""ZONOS3-Mini steering as per-segment tags: ``[Alice|happy:2.5|pace:-1|british_accent:1|full_strength:2]``.

The tags are the configuration nodes' own controls: every Voice Characteristics / Emotion & Delivery slider (same
ranges and direction mappings), every additional effect by its native id (``emotions.proud:2``), and the
effect-strength envelope. ``utils/text/segment_parameters.py`` registers them from ``TAG_PARAMETERS``; the processor
folds the tagged values into a segment's ``controls`` / ``steer_*`` with ``apply_steering_tags``. The engine node's
configuration is the baseline: a tag replaces that one control for its segment only.
"""

import math
from typing import Any, Dict, Tuple

from .steering_controls import DELIVERY, VOICE_DESIGN
from .steering_ui import EFFECT_DESCRIPTIONS, validate_steering

PREFIX = "zonos3_"
EFFECT_PREFIX = PREFIX + "effect_"
EFFECT_RANGE = (-50.0, 50.0)

_SLIDERS = {control[0]: control for control in VOICE_DESIGN + DELIVERY}
_ACCENT_SETS = ("accents_en", "accents_l1", "mandarin")
# tag -> (short alias, config key, min, max, tag value -> native value, description)
_ENVELOPE = {
    "full_strength_seconds": ("full_strength", "steer_onset_seconds", 0.0, 600.0, float, "seconds an effect stays at full strength"),
    "reduction_seconds": ("reduction", "steer_ramp_seconds", 0.0, 600.0, float, "seconds an effect takes to fade down"),
    "remaining_strength_percent": ("remaining_strength", "steer_hold", 0.0, 100.0, lambda v: float(v) / 100, "percent of an effect kept after fading down"),
    "control_guidance": ("control_guidance", "controls_cfg_scale", 0.0, 5.0, float, "extra guidance for delivery effects"),
}


def _slider_tag(control) -> str:
    """A slider's tag: its widget name; accents get ``_accent`` (their bare names are language tags)."""
    widget, low, _high, mapping, _tooltip = control
    is_accent = next(iter(mapping(1.0))).split(".")[0] in _ACCENT_SETS
    return f"{widget}_accent" if is_accent else widget


def _build() -> Dict[str, Tuple[str, float, float, str]]:
    tags: Dict[str, Tuple[str, float, float, str]] = {}
    for widget, low, high, _mapping, tooltip in _SLIDERS.values():
        tags[_slider_tag(_SLIDERS[widget])] = (PREFIX + widget, low, high, f"ZONOS3-Mini {widget}: {tooltip}")
    for direction, description in EFFECT_DESCRIPTIONS.items():
        tags[direction] = (EFFECT_PREFIX + direction, *EFFECT_RANGE, f"ZONOS3-Mini effect {direction}: {description}")
    for tag, (alias, _key, low, high, _convert, description) in _ENVELOPE.items():
        tags[tag] = tags[alias] = (PREFIX + tag, low, high, f"ZONOS3-Mini {description}")
    return tags


# tag name a user writes -> (canonical parameter name, min, max, description)
TAG_PARAMETERS = _build()


def _number(tag: str, raw: Any, low: float, high: float):
    """A tag value as a float inside the control's range; None (with a warning) when it is not."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value) or not low <= value <= high:
        print(f"⚠️ ZONOS3-Mini: tag '{tag}' must be a number between {low:g} and {high:g}, got '{raw}'; ignoring it")
        return None
    return value


def apply_steering_tags(config: Dict[str, Any]) -> Dict[str, Any]:
    """A segment config after ``apply_segment_parameters`` -> a new config with its steering tags folded into
    ``controls`` and the envelope settings, checked against the node's mode restrictions. Untagged configs come back
    as they are."""
    def shared(widget):  # happy / angry / sad / calm reuse the suite's existing emotion tags
        return f"emotion_{widget}_inline"

    if not any(key.startswith(PREFIX) or key in map(shared, _SLIDERS) for key in config):
        return config
    config = dict(config)
    controls = {str(k): float(v) for k, v in (config.get("controls") or {}).items()}
    for widget, (_w, low, high, mapping, _tooltip) in _SLIDERS.items():
        raw, raw_shared = config.pop(PREFIX + widget, None), config.pop(shared(widget), None)
        raw = raw_shared if raw is None else raw
        value = None if raw is None else _number(widget, raw, low, high)
        if value is None:
            continue
        for direction in {**mapping(low), **mapping(high)}:
            controls.pop(direction, None)
        if value:
            controls.update({direction: float(dose) for direction, dose in mapping(value).items()})
    for direction in EFFECT_DESCRIPTIONS:
        raw = config.pop(EFFECT_PREFIX + direction, None)
        value = None if raw is None else _number(direction, raw, *EFFECT_RANGE)
        if value is None:
            continue
        controls.pop(direction, None)
        if value:
            controls[direction] = value
    for tag, (_alias, key, low, high, convert, _description) in _ENVELOPE.items():
        raw = config.pop(PREFIX + tag, None)
        value = None if raw is None else _number(tag, raw, low, high)
        if value is not None:
            config[key] = convert(value)
    config["controls"] = controls
    validate_steering(
        str(config.get("voice_mode", "embedding") or "embedding"),
        controls,
        float(config.get("controls_cfg_scale", 1.0)),
        bool(config.get("laci", False)),
        float(config.get("speaker_cfg_scale", 1.0)),
    )
    return config
