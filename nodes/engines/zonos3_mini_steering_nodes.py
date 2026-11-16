"""
ZONOS3-Mini steering option nodes.

Voice Design (speaker-embedding directions) and Delivery & Mood (hidden-state directions) sliders that
connect to the ZONOS3-Mini Engine node. Values are the native steering doses of the public bundle.
"""

import os
import sys

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.zonos3_mini.steering_controls import DELIVERY, VOICE_DESIGN, build_controls, parse_extra_controls


def _sliders(table):
    return {
        widget: ("FLOAT", {
            "default": 0.0, "min": low, "max": high, "step": 0.1,
            "display": "slider",
            "tooltip": f"{tooltip} 0 = off.",
        })
        for widget, low, high, _mapping, tooltip in table
    }


class Zonos3MiniVoiceDesignNode:
    """Speaker-embedding steering: who is speaking."""

    @classmethod
    def NAME(cls):
        return "🎚️ ZONOS3-Mini Voice Design"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": _sliders(VOICE_DESIGN)}

    RETURN_TYPES = ("ZONOS3_STEERING",)
    RETURN_NAMES = ("voice_design",)
    FUNCTION = "create_steering"
    CATEGORY = "TTS Audio Suite/⚙️ Engines/ZONOS3-Mini"
    DESCRIPTION = (
        "Reshape the voice itself with ZONOS3-Mini's native speaker directions: gender, age, pitch, texture, "
        "room and accent. Connect to the voice_design input of the ZONOS3-Mini Engine node. These directions move "
        "the speaker embedding, so the engine's voice_mode must be 'embedding'."
    )

    def create_steering(self, **values):
        controls = build_controls(VOICE_DESIGN, values)
        print(f"🎚️ ZONOS3-Mini Voice Design: {controls if controls else 'neutral'}")
        return ({"controls": controls},)


class Zonos3MiniDeliveryNode:
    """Hidden-state steering: how the line is delivered."""

    @classmethod
    def NAME(cls):
        return "🎭 ZONOS3-Mini Delivery & Mood"

    @classmethod
    def INPUT_TYPES(cls):
        optional = _sliders(DELIVERY)
        optional.update({
            "extra_controls": ("STRING", {
                "default": "",
                "multiline": True,
                "tooltip": "Additional native steering directions, one 'set.direction=dose' per line, for directions without a slider. Examples: emotions.proud=3, emotions.surprised=3, emotions.disgusted=3, mood.dominance=2. Speaker sets (identity, pitch, texture, room, accents_en, accents_l1, mandarin) need voice_mode 'embedding'.",
            }),
            "steer_hold": ("FLOAT", {
                "default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                "tooltip": "Fraction of the dose that is kept after the steering onset has passed (steer-then-hold). -1 uses the bundle default (0.35). Pace and whisper are always held fully.",
            }),
            "steer_onset_seconds": ("FLOAT", {
                "default": -1.0, "min": -1.0, "max": 60.0, "step": 0.1,
                "tooltip": "Seconds of audio generated at full dose before the dose is released toward steer_hold. -1 uses the bundle default (4 s).",
            }),
            "steer_ramp_seconds": ("FLOAT", {
                "default": -1.0, "min": -1.0, "max": 60.0, "step": 0.1,
                "tooltip": "Seconds over which the dose ramps from full to steer_hold. -1 uses the bundle default (1 s).",
            }),
            "controls_cfg_scale": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                "tooltip": "Classifier-free guidance on the steering controls. 1 = off. Other values run the model twice per frame (about half speed) and cannot be combined with LACI.",
            }),
        })
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("ZONOS3_STEERING",)
    RETURN_NAMES = ("delivery",)
    FUNCTION = "create_steering"
    CATEGORY = "TTS Audio Suite/⚙️ Engines/ZONOS3-Mini"
    DESCRIPTION = (
        "Steer how a line is delivered with ZONOS3-Mini's native hidden-state directions: projection, pace, "
        "liveliness, mood and emotions. Works with both voice modes. Zyphra recommends moderate doses on the public "
        "model: angry/happy 2-3, sad 4-6, calm 4, scared 8, confused 2. Connect to the delivery input of the "
        "ZONOS3-Mini Engine node."
    )

    def create_steering(
        self,
        extra_controls="",
        steer_hold=-1.0,
        steer_onset_seconds=-1.0,
        steer_ramp_seconds=-1.0,
        controls_cfg_scale=1.0,
        **values,
    ):
        controls = build_controls(DELIVERY, values)
        controls.update(parse_extra_controls(extra_controls))
        print(f"🎭 ZONOS3-Mini Delivery & Mood: {controls if controls else 'neutral'}")
        return ({
            "controls": controls,
            "steer_hold": float(steer_hold),
            "steer_onset_seconds": float(steer_onset_seconds),
            "steer_ramp_seconds": float(steer_ramp_seconds),
            "controls_cfg_scale": float(controls_cfg_scale),
        },)


NODE_CLASS_MAPPINGS = {
    "Zonos3MiniVoiceDesignNode": Zonos3MiniVoiceDesignNode,
    "Zonos3MiniDeliveryNode": Zonos3MiniDeliveryNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Zonos3MiniVoiceDesignNode": "🎚️ ZONOS3-Mini Voice Design",
    "Zonos3MiniDeliveryNode": "🎭 ZONOS3-Mini Delivery & Mood",
}
