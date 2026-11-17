"""ZONOS3-Mini steering widgets: slider tables and their mapping to the request's ``controls`` ({direction: dose}).

Ranges and mappings follow the ``ui`` table of upstream's assets/steering/studio_bundle_mect60k_x10.json (that table
is not part of the trimmed copy bundled here).
Raw sets (identity, pitch, texture, room, accents_en, accents_l1, mandarin) move the speaker embedding and need
speaker_mode=embedding; hidden sets (delivery, disfluency, expressive, mood, emotions, mandarin_hidden) work in both
speaker modes.
"""

from typing import Callable, Dict, List, Tuple

Mapping = Callable[[float], Dict[str, float]]
Control = Tuple[str, float, float, Mapping, str]  # widget, min, max, mapping, tooltip


def _direct(direction: str) -> Mapping:
    return lambda value: {direction: value}


def _projection(value: float) -> Dict[str, float]:
    return {"delivery.whisper": -value} if value < 0 else {"delivery.loud": value}


def _mandarin(value: float) -> Dict[str, float]:
    return {"mandarin.cmn": value, "mandarin_hidden.cmn": value / 3}


VOICE_DESIGN: List[Control] = [
    ("gender", -3.0, 3.0, _direct("identity.gender"), "Negative = more masculine, positive = more feminine. About ±2 flips the perceived gender; similarity to the source voice drops as the dose grows."),
    ("age", -3.0, 3.0, _direct("identity.age"), "Negative = younger, positive = older."),
    ("pitch", -3.0, 3.0, _direct("pitch.pitch"), "Lower or raise the voice's pitch."),
    ("pitch_range", -3.0, 3.0, _direct("pitch.pitch_range"), "Negative = flatter, more monotone; positive = wider pitch movement."),
    ("deep", 0.0, 6.0, _direct("texture.deep"), "Deeper, chestier voice."),
    ("bright", 0.0, 6.0, _direct("texture.shrill"), "Brighter, sharper voice."),
    ("husky", 0.0, 6.0, _direct("texture.husky"), "Husky, breathy texture."),
    ("raspy", 0.0, 6.0, _direct("texture.raspy"), "Rough, raspy texture."),
    ("soft", 0.0, 6.0, _direct("texture.soft"), "Softer, gentler voice."),
    ("authority", 0.0, 6.0, _direct("texture.authoritative"), "More authoritative, commanding voice."),
    ("room", -4.0, 6.0, _direct("room.clean"), "Recording character: positive = cleaner studio sound, negative = more room/noise."),
] + [
    (name, 0.0, 2.0, _direct(f"accents_en.{name}"), f"English accent direction: {name.title()}. Around 1-1.5 is audible; higher doses move further from the source voice.")
    for name in ("american", "british", "scottish", "irish", "welsh", "australian", "indian")
] + [
    (name, 0.0, 2.0, _direct(f"accents_l1.{code}"), f"Non-native accent direction: speech coloured by a {name.title()} first language.")
    for name, code in (
        ("hindi", "hin"), ("spanish", "spa"), ("french", "fra"), ("portuguese", "por"),
        ("german", "deu"), ("vietnamese", "vie"), ("polish", "pol"),
    )
] + [
    ("mandarin", 0.0, 1.5, _mandarin, "Mandarin accent direction (applies the speaker direction plus one third of the dose on the hidden direction, as the official studio does)."),
]

DELIVERY: List[Control] = [
    ("projection", -3.0, 4.0, _projection, "Negative = whisper, positive = loud, projected speech."),
    ("pace", -3.0, 3.0, _direct("delivery.tempo"), "Negative = slower delivery, positive = faster."),
    ("chatty", -5.0, 5.0, _direct("delivery.conversational"), "Negative = read/formal, positive = casual conversational delivery."),
    ("hesitant", -8.0, 20.0, _direct("disfluency.disfluent"), "Positive adds hesitations and disfluencies; negative makes delivery more fluent."),
    ("lively", -16.0, 16.0, _direct("expressive.expressive"), "Negative = flatter, positive = more expressive delivery."),
    ("valence", -6.0, 6.0, _direct("mood.valence"), "Negative = unpleasant mood, positive = pleasant mood."),
    ("arousal", -6.0, 6.0, _direct("mood.arousal"), "Negative = low energy, positive = high energy."),
    ("angry", 0.0, 6.0, _direct("emotions.angry"), "Angry. Recommended dose on the public model: 2-3."),
    ("happy", 0.0, 6.0, _direct("emotions.happy"), "Happy. Recommended dose on the public model: 2-3."),
    ("sad", 0.0, 8.0, _direct("emotions.sad"), "Sad. Recommended dose on the public model: 4-6."),
    ("calm", 0.0, 8.0, _direct("emotions.serene"), "Calm, serene. Recommended dose on the public model: 4."),
    ("scared", 0.0, 14.0, _direct("emotions.scared"), "Scared. Recommended dose on the public model: 8."),
    ("confused", 0.0, 4.0, _direct("emotions.confused"), "Confused. Recommended dose on the public model: 2."),
]


def build_controls(table: List[Control], values: Dict[str, float]) -> Dict[str, float]:
    """Widget values -> ``{direction: dose}``; widgets at zero contribute nothing."""
    controls: Dict[str, float] = {}
    for widget, _low, _high, mapping, _tooltip in table:
        value = float(values.get(widget, 0.0) or 0.0)
        if value == 0.0:
            continue
        for direction, dose in mapping(value).items():
            controls[direction] = controls.get(direction, 0.0) + float(dose)
    return controls


def parse_extra_controls(text: str) -> Dict[str, float]:
    """``set.direction=dose`` lines -> ``{direction: dose}``; blank lines are skipped."""
    controls: Dict[str, float] = {}
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        name, sep, dose = line.partition("=")
        name = name.strip()
        set_name, dot, direction = name.partition(".")
        try:
            if not (sep and dot and set_name and direction) or " " in name:
                raise ValueError(line)
            controls[name] = float(dose)
        except ValueError:
            raise ValueError(
                f"ZONOS3-Mini extra_controls line '{line}' must be 'set.direction=number'"
            ) from None
    return controls
