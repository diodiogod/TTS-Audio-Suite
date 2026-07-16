"""Unified reference-free voice-design operation."""

from utils.voice.designers import design_voice


class UnifiedVoiceDesignerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TTS_engine": ("TTS_ENGINE", {
                    "tooltip": (
                        "Configured Qwen3-TTS, MOSS-TTS, or OmniVoice engine. This node owns the voice-design "
                        "instruction; the engine owns its model, language, and generation settings. Qwen and "
                        "MOSS require their voice-design model, while OmniVoice requires Voice Design mode."
                    )
                }),
                "reference_text": ("STRING", {
                    "default": (
                        "Welcome to the TTS Audio Suite. This advanced text-to-speech system brings "
                        "your stories to life with natural, expressive voices. Whether you're creating "
                        "audiobooks, videos, or interactive experiences, our technology delivers "
                        "exceptional quality and versatility. What will you create today?"
                    ),
                    "multiline": True,
                    "tooltip": (
                        "Plain text spoken to create the reusable voice reference. The exact transcript "
                        "is stored inside opt_narrator for Character Voices and Save Character Voice. "
                        "Use roughly 10 seconds or more with varied intonation, questions, and representative "
                        "sounds to evaluate and clone the designed voice reliably."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": (
                        "Generation seed. 0 keeps the provider's random behavior. A fixed nonzero seed also "
                        "lets Save Character Voice recognize an identical generation safely."
                    ),
                }),
                "voice_instruction": ("STRING", {
                    "default": "A warm, confident adult voice with natural pacing, clear articulation, and a subtle expressive smile.",
                    "multiline": True,
                    "tooltip": (
                        "Describe the voice identity and delivery to create: age, gender, pitch, texture, accent, "
                        "pace, emotion, and speaking style. This overrides any stored engine instruction for this "
                        "voice-design generation."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("NARRATOR_VOICE", "AUDIO", "STRING")
    RETURN_NAMES = ("opt_narrator", "preview_audio", "voice_info")
    FUNCTION = "design"
    CATEGORY = "TTS Audio Suite/🎭 Voice & Character"

    def design(self, TTS_engine, reference_text, seed, voice_instruction):
        result = design_voice(TTS_engine, voice_instruction, reference_text, seed)
        return (
            result.opt_narrator,
            result.preview_audio,
            result.voice_info,
        )
