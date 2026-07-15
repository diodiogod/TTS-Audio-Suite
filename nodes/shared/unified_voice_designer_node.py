"""Unified reference-free voice-design operation."""

from utils.voice.designers import design_voice


class UnifiedVoiceDesignerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TTS_engine": ("TTS_ENGINE", {
                    "tooltip": (
                        "Configured Qwen3-TTS, MOSS-TTS, or OmniVoice engine. The engine owns the "
                        "voice instruction, language, and model. Qwen and MOSS require their voice-design "
                        "model to be selected in the engine node."
                    )
                }),
                "reference_text": ("STRING", {
                    "default": "Welcome. This sample creates a reusable voice reference for your character.",
                    "multiline": True,
                    "tooltip": (
                        "Plain text spoken to create the reusable voice reference. The exact transcript "
                        "is stored inside opt_narrator for Character Voices and Save Character Voice."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Generation seed. 0 keeps the provider's normal random behavior.",
                }),
            },
        }

    RETURN_TYPES = ("NARRATOR_VOICE", "AUDIO", "STRING")
    RETURN_NAMES = ("opt_narrator", "preview_audio", "voice_info")
    FUNCTION = "design"
    CATEGORY = "TTS Audio Suite/🎭 Voice & Character"

    def design(self, TTS_engine, reference_text, seed):
        result = design_voice(TTS_engine, reference_text, seed)
        return (
            result.opt_narrator,
            result.preview_audio,
            result.voice_info,
        )
