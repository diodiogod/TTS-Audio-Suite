"""
ZONOS3-Mini steering option nodes.

Voice Characteristics (speaker-embedding directions) and Emotion & Delivery (hidden-state directions) sliders that
connect to the ZONOS3-Mini Engine node. Values are the native steering doses of the public bundle.
"""

import os
import sys

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.zonos3_mini.steering_controls import DELIVERY, VOICE_DESIGN, build_controls
from engines.zonos3_mini.steering_ui import VOICE_NAME, DELIVERY_NAME, slider_metadata, control_catalog, parse_additional_effects


def _sliders(table):
    sliders = {}
    for control in table:
        widget, low, high, _mapping, tooltip = control
        sliders[widget] = ("FLOAT", {
            "default": 0.0, "min": low, "max": high, "step": 0.1,
            "display": "slider",
            "tooltip": f"{tooltip}\n\n0 = no change.",
            "zonos3": slider_metadata(control),
        })
    return sliders


class Zonos3MiniVoiceCharacteristicsNode:
    """Speaker-embedding steering: who is speaking."""

    @classmethod
    def NAME(cls):
        return VOICE_NAME

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": _sliders(VOICE_DESIGN)}

    RETURN_TYPES = ("ZONOS3_VOICE_CHARACTERISTICS",)
    RETURN_NAMES = ("voice_characteristics",)
    FUNCTION = "create_steering"
    CATEGORY = "TTS Audio Suite/⚙️ Engines/ZONOS3-Mini"
    DESCRIPTION = (
        "Change the voice's gender, age, pitch, texture, recording character and accent.\n\n"
        "Connect to voice_characteristics on the ZONOS3-Mini Engine.\n"
        "Requires embedding voice mode."
    )

    def create_steering(self, **values):
        controls = build_controls(VOICE_DESIGN, values)
        print(f"🌈 ZONOS3-Mini Voice Characteristics: {controls if controls else 'neutral'}")
        return ({"kind": "voice_characteristics", "controls": controls},)


class Zonos3MiniDeliveryNode:
    """Hidden-state steering: how the line is delivered."""

    @classmethod
    def NAME(cls):
        return DELIVERY_NAME

    @classmethod
    def INPUT_TYPES(cls):
        optional = _sliders(DELIVERY)
        optional.update({
            "additional_effects": ("STRING", {
                "default": "{}",
                "tooltip": "Choose more voice effects in the Add effect panel and set their strengths.",
                "zonos3_catalog": control_catalog(),
            }),
            "use_model_defaults": ("BOOLEAN", {
                "default": True,
                "tooltip": "Use the model's standard timing:\n\nFull strength for 4 seconds.\nReduce over 1 second.\nKeep 35% strength afterwards.",
            }),
            "full_strength_seconds": ("FLOAT", {
                "default": 4.0, "min": 0.0, "max": 600.0, "step": 0.1,
                "tooltip": "How many seconds the effect stays at full strength before fading down.\n\nMeasured from the start of each generated chunk or subtitle.",
            }),
            "reduction_seconds": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 600.0, "step": 0.1,
                "tooltip": "How long the effect takes to fade down to the remaining strength.\n\n0: change immediately.",
            }),
            "remaining_strength_percent": ("FLOAT", {
                "default": 35.0, "min": 0.0, "max": 100.0, "step": 1.0,
                "tooltip": "How much of the original effect remains after fading down.\n\n0%: remove the effect.\n100%: keep full strength.\n\nPace and whisper stay constant. This percentage controls the effect setting, not perceived emotion.",
            }),
            "control_guidance": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                "tooltip": "Extra guidance for the selected delivery effects.\n\n1: normal guidance; effects remain active.\nOther values: adjust guidance, with extra processing time.\n\nRequires embedding voice mode, LACI off and speaker guidance at 1.",
            }),
        })
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("ZONOS3_DELIVERY",)
    RETURN_NAMES = ("delivery",)
    FUNCTION = "create_steering"
    CATEGORY = "TTS Audio Suite/⚙️ Engines/ZONOS3-Mini"
    DESCRIPTION = (
        "Change how the voice speaks: emotion, energy, pace and delivery.\n\n"
        "Connect to delivery on the ZONOS3-Mini Engine.\n"
        "Works with both voice modes. Each emotion's tooltip includes the upstream suggested strength."
    )

    def create_steering(
        self,
        additional_effects="{}",
        use_model_defaults=True,
        full_strength_seconds=4.0,
        reduction_seconds=1.0,
        remaining_strength_percent=35.0,
        control_guidance=1.0,
        **values,
    ):
        controls = build_controls(DELIVERY, values)
        controls.update(parse_additional_effects(additional_effects))
        print(f"🌈 ZONOS3-Mini Emotion & Delivery: {controls if controls else 'neutral'}")
        return ({
            "kind": "delivery",
            "controls": controls,
            "steer_hold": -1.0 if use_model_defaults else float(remaining_strength_percent) / 100,
            "steer_onset_seconds": -1.0 if use_model_defaults else float(full_strength_seconds),
            "steer_ramp_seconds": -1.0 if use_model_defaults else float(reduction_seconds),
            "controls_cfg_scale": float(control_guidance),
        },)


NODE_CLASS_MAPPINGS = {
    "Zonos3MiniVoiceCharacteristicsNode": Zonos3MiniVoiceCharacteristicsNode,
    "Zonos3MiniDeliveryNode": Zonos3MiniDeliveryNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Zonos3MiniVoiceCharacteristicsNode": VOICE_NAME,
    "Zonos3MiniDeliveryNode": DELIVERY_NAME,
}
