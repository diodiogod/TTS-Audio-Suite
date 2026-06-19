"""
OmniVoice Instruction Builder helper node.

Builds a valid OmniVoice voice-design instruct string with canonical category
ordering while leaving the main engine's raw instruct field available for
manual edits.
"""

from typing import Dict, Tuple


EN_TO_ZH: Dict[str, str] = {
    "male": "男",
    "female": "女",
    "child": "儿童",
    "teenager": "少年",
    "young adult": "青年",
    "middle-aged": "中年",
    "elderly": "老年",
    "very low pitch": "极低音调",
    "low pitch": "低音调",
    "moderate pitch": "中音调",
    "high pitch": "高音调",
    "very high pitch": "极高音调",
    "whisper": "耳语",
}


def _normalize_value(value: str) -> str:
    text = str(value or "").strip()
    return "" if not text or text == "None" else text


class OmniVoiceInstructionBuilderNode:
    """Structured OmniVoice instruction helper with canonical output order."""

    @classmethod
    def NAME(cls):
        return "📐 OmniVoice Instruction Builder"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gender": (["None", "male", "female"], {
                    "default": "None",
                    "tooltip": "OmniVoice gender attribute. Hidden by the custom builder UI."
                }),
                "age": (["None", "child", "teenager", "young adult", "middle-aged", "elderly"], {
                    "default": "None",
                    "tooltip": "OmniVoice age attribute. Hidden by the custom builder UI."
                }),
                "pitch": (["None", "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"], {
                    "default": "None",
                    "tooltip": "OmniVoice pitch attribute. Hidden by the custom builder UI."
                }),
                "style": (["None", "whisper"], {
                    "default": "None",
                    "tooltip": "OmniVoice style attribute. Hidden by the custom builder UI."
                }),
                "accent": ([
                    "None",
                    "american accent",
                    "british accent",
                    "australian accent",
                    "canadian accent",
                    "indian accent",
                    "chinese accent",
                    "korean accent",
                    "japanese accent",
                    "portuguese accent",
                    "russian accent",
                ], {
                    "default": "None",
                    "tooltip": "OmniVoice English accent attribute. Hidden by the custom builder UI."
                }),
                "dialect": ([
                    "None",
                    "河南话",
                    "陕西话",
                    "四川话",
                    "贵州话",
                    "云南话",
                    "桂林话",
                    "济南话",
                    "石家庄话",
                    "甘肃话",
                    "宁夏话",
                    "青岛话",
                    "东北话",
                ], {
                    "default": "None",
                    "tooltip": "OmniVoice Chinese dialect attribute. Hidden by the custom builder UI."
                }),
                "output_language": (["English", "Chinese"], {
                    "default": "English",
                    "tooltip": "Output language mode for the builder UI and emitted instruct string. Hidden by the custom builder UI."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("instruct",)
    FUNCTION = "build_instruction"
    CATEGORY = "TTS Audio Suite/📐 OmniVoice"
    DESCRIPTION = (
        "Visual helper for OmniVoice voice-design instructions. Outputs a "
        "canonical instruct string that can be connected into the OmniVoice "
        "engine's instruct input."
    )

    def build_instruction(
        self,
        gender: str,
        age: str,
        pitch: str,
        style: str,
        accent: str,
        dialect: str,
        output_language: str,
    ) -> Tuple[str]:
        gender = _normalize_value(gender)
        age = _normalize_value(age)
        pitch = _normalize_value(pitch)
        style = _normalize_value(style)
        accent = _normalize_value(accent)
        dialect = _normalize_value(dialect)
        output_language = str(output_language or "English").strip()

        # The builder UI enforces mutual exclusion; this keeps corrupted
        # workflow state from emitting an invalid mixed instruct string.
        if accent and dialect:
            dialect = ""

        parts = [gender, age, pitch, style, accent or dialect]
        parts = [part for part in parts if part]

        if not parts:
            return ("",)

        if output_language == "Chinese":
            translated = [EN_TO_ZH.get(part, part) for part in parts]
            return ("，".join(translated),)

        return (", ".join(parts),)


NODE_CLASS_MAPPINGS = {
    "OmniVoiceInstructionBuilderNode": OmniVoiceInstructionBuilderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniVoiceInstructionBuilderNode": "📐 OmniVoice Instruction Builder",
}
