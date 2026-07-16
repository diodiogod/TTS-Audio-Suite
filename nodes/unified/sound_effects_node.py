"""Unified engine-agnostic text-to-sound generation node."""

from utils.audio.sound_effects import generate_sound_effect


class UnifiedSoundEffectsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TTS_engine": ("TTS_ENGINE", {
                    "tooltip": (
                        "Connect an engine configured for sound-effect generation. "
                        "MOSS-SoundEffect v1 uses the MOSS-TTS Engine; v2 uses the "
                        "MOSS SoundEffect v2 Engine. Speech-only engines stop with guidance."
                    ),
                }),
                "description": ("STRING", {
                    "multiline": True,
                    "default": "Heavy rain on a metal rooftop with distant thunder and occasional wind gusts.",
                    "tooltip": (
                        "Describe the sound, environment, actions, texture, distance, and timing you want. "
                        "This text is a sound description, not words to be spoken."
                    ),
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.5,
                    "max": 300.0,
                    "step": 0.5,
                    "tooltip": (
                        "Requested output duration in seconds. Engine limits still apply. "
                        "MOSS SoundEffect v2 supports up to 30 seconds; v1 converts this to its native audio-token hint."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Controls the generated variation. Reuse a positive seed and the same settings for repeatable output; set 0 for a random seed.",
                }),
            },
            "optional": {
                "enable_audio_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reuse audio when the engine, description, duration, seed, and generation settings are identical.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")
    FUNCTION = "generate"
    CATEGORY = "TTS Audio Suite/🌩️ Sound Effects"
    DESCRIPTION = "Generate non-speech sound effects from a text description using any compatible engine."

    def generate(self, TTS_engine, description, duration_seconds, seed, enable_audio_cache=True):
        result = generate_sound_effect(
            engine_data=TTS_engine,
            description=description,
            duration_seconds=duration_seconds,
            seed=seed,
            enable_audio_cache=enable_audio_cache,
        )
        return result.audio, result.generation_info


NODE_CLASS_MAPPINGS = {"UnifiedSoundEffectsNode": UnifiedSoundEffectsNode}
NODE_DISPLAY_NAME_MAPPINGS = {"UnifiedSoundEffectsNode": "🌩️ Sound Effects"}
