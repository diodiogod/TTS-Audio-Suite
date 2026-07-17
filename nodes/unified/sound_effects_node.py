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
                        "This is a sound description, not text to be spoken. "
                        "Inline parameter and pause tags are supported."
                    ),
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.5,
                    "max": 300.0,
                    "step": 0.5,
                    "tooltip": (
                        "Duration of each described segment. A single MOSS v2 segment longer than 30 seconds "
                        "is generated in overlapping chunks and trimmed to this exact duration. Inline [seconds:X] overrides it."
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
                "crossfade_seconds": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": (
                        "Overlap between adjacent generated segments and automatic long-duration chunks. "
                        "Use 0 for a hard join. [pause:X], [wait:X], and [stop:X] insert exact silence and disable the crossfade across that boundary."
                    ),
                }),
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

    def generate(self, TTS_engine, description, duration_seconds, seed, crossfade_seconds=1.0, enable_audio_cache=True):
        result = generate_sound_effect(
            engine_data=TTS_engine,
            description=description,
            duration_seconds=duration_seconds,
            seed=seed,
            enable_audio_cache=enable_audio_cache,
            crossfade_seconds=crossfade_seconds,
        )
        return result.audio, result.generation_info


NODE_CLASS_MAPPINGS = {"UnifiedSoundEffectsNode": UnifiedSoundEffectsNode}
NODE_DISPLAY_NAME_MAPPINGS = {"UnifiedSoundEffectsNode": "🌩️ Sound Effects"}
