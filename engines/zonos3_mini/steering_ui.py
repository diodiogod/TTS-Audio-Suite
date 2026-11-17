"""UI metadata derived from the bundled native directions, without loading tensors."""

import json
import math
from functools import lru_cache
from pathlib import Path

from .steering_controls import DELIVERY, VOICE_DESIGN

VOICE_NAME = "🌈 ZONOS3-Mini Voice Characteristics"
DELIVERY_NAME = "🌈 ZONOS3-Mini Emotion & Delivery"

GROUPS = {
    "identity": "Identity", "pitch": "Pitch & intonation", "texture": "Voice texture",
    "room": "Recording character", "accents_en": "English accents", "accents_l1": "Other accents",
    "mandarin": "Other accents", "mandarin_hidden": "Accent delivery", "delivery": "Delivery",
    "expressive": "Delivery", "disfluency": "Delivery", "mood": "Mood",
    "emotions": "Emotions", "emphasis_nv_nodur": "Emphasis",
}
LABELS = {
    "pitch_range": "Intonation range", "authority": "Authority", "room": "Recording character",
    "projection": "Whisper ↔ projected", "pace": "Slower ↔ faster", "chatty": "Read ↔ conversational",
    "hesitant": "Fluent ↔ hesitant", "lively": "Flat ↔ lively", "valence": "Unpleasant ↔ pleasant",
    "arousal": "Low ↔ high energy", "gender": "Masculine ↔ feminine", "age": "Younger ↔ older",
}
EFFECT_DESCRIPTIONS = {
    "delivery.pitch": "Lowers or raises the pitch during speech delivery.",
    "mood.dominance": "Steers the voice toward a more dominant or assertive delivery.",
    "mood.loudness": "Steers how loudly the voice speaks during generation.",
    "mood.tempo": "Steers the speaking pace during generation.",
    "mood.clarity": "Steers the voice toward clearer speech.",
    "mood.articulation": "Steers how distinctly the voice articulates speech sounds.",
    "emotions.fearful": "Adds a fearful tone to the delivery.",
    "emotions.disgusted": "Adds a tone of disgust to the delivery.",
    "emotions.surprised": "Adds a surprised tone to the delivery.",
    "emotions.adoring": "Adds an affectionate, adoring tone to the delivery.",
    "emotions.content": "Adds a contented, satisfied tone to the delivery.",
    "emotions.cute": "Adds a cute tone to the delivery.",
    "emotions.desirous": "Adds a longing, desirous tone to the delivery.",
    "emotions.disappointed": "Adds a disappointed tone to the delivery.",
    "emotions.distressed": "Adds a distressed tone to the delivery.",
    "emotions.embarrassed": "Adds an embarrassed tone to the delivery.",
    "emotions.ecstatic": "Adds an ecstatic, intensely joyful tone to the delivery.",
    "emotions.guilty": "Adds a guilty tone to the delivery.",
    "emotions.interested": "Adds an interested, engaged tone to the delivery.",
    "emotions.pained": "Adds a pained tone to the delivery.",
    "emotions.proud": "Adds a proud tone to the delivery.",
    "emotions.realizing": "Adds a tone of realization, as if something has just become clear.",
    "emotions.relieved": "Adds a relieved tone to the delivery.",
    "emphasis_nv_nodur.emphasis": "Adds emphasis to the speech delivery.",
}


def slider_metadata(control):
    widget, low, _high, mapping, _tooltip = control
    group = GROUPS[next(iter(mapping(1.0))).split(".")[0]]
    return {"label": LABELS.get(widget, widget.replace("_", " ").title()), "group": group}


@lru_cache(maxsize=1)
def control_catalog():
    root = Path(__file__).parent / "assets" / "steering"
    bundle = json.loads((root / "studio_bundle_mect60k_x10.json").read_text(encoding="utf-8"))
    covered = {}
    for control in VOICE_DESIGN + DELIVERY:
        widget, low, high, mapping, tooltip = control
        for value in (low, high):
            for direction in mapping(value):
                covered[direction] = widget
    catalog = []
    for set_name, folder in bundle["sets"].items():
        manifest = json.loads((root / folder / "manifest.json").read_text(encoding="utf-8"))
        for name, spec in manifest["directions"].items():
            direction = f"{set_name}.{name}"
            poles = spec.get("poles", {})
            label = name.replace("_", " ").title()
            hint = EFFECT_DESCRIPTIONS.get(direction, f"Steers the voice's {label.lower()} characteristic.")
            if poles:
                hint += f" Negative: {poles.get('-', 'less')}. Positive: {poles.get('+', 'more')}."
            catalog.append({
                "id": direction, "label": label,
                "group": GROUPS[set_name], "embedding_only": manifest["space"] == "raw",
                "slider": covered.get(direction), "step": 0.1,
                "hint": hint,
            })
    return {"effects": catalog, "release": bundle["steer_release"]}


def parse_additional_effects(text):
    """Validate the effect editor's structured values; slider effects cannot be overridden."""
    try:
        values = json.loads(text)
    except (ValueError, TypeError):
        raise ValueError("ZONOS3-Mini additional effects must be a JSON object of effect strengths.") from None
    if not isinstance(values, dict):
        raise ValueError("ZONOS3-Mini additional effects must be a JSON object of effect strengths.")
    available = {effect["id"] for effect in control_catalog()["effects"] if not effect["slider"] and not effect["embedding_only"]}
    controls = {}
    for direction, value in values.items():
        if direction not in available:
            raise ValueError(f"ZONOS3-Mini additional effect '{direction}' is unavailable or already has a slider.")
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"ZONOS3-Mini effect '{direction}' needs a finite numeric strength.")
        if value != 0:
            controls[direction] = float(value)
    return controls
