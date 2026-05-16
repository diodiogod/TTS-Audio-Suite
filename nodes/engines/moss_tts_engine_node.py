"""
MOSS-TTS Engine Configuration Node.

Configures the official OpenMOSS MOSS-TTS models for the unified TTS nodes.
"""

import os
import sys
import importlib.util
from typing import Dict, List

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

BaseTTSNode = base_module.BaseTTSNode

from engines.moss_tts.moss_tts import MossTTSEngine
from utils.models.extra_paths import get_all_tts_model_paths


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")


class MossTTSEngineNode(BaseTTSNode):
    """MOSS-TTS engine configuration node."""

    STANDARD_MODEL_OPTIONS = [
        "Small 1.7B (Local)",
        "8B (Delay)",
    ]
    NATIVE_MODEL_OPTION = "Native 8B Dialogue (MOSS-TTSD-v1.0)"
    UI_MODEL_VARIANT_MAP = {
        "Small 1.7B (Local)": "MOSS-TTS-Local-Transformer",
        "8B (Delay)": "MOSS-TTS",
        "Native 8B Dialogue (MOSS-TTSD-v1.0)": "MOSS-TTSD-v1.0",
    }

    MODEL_DEFAULTS: Dict[str, Dict[str, float]] = {
        name: {
            "temperature": values["audio_temperature"],
            "top_p": values["audio_top_p"],
            "top_k": values["audio_top_k"],
            "repetition_penalty": values["audio_repetition_penalty"],
            "max_new_tokens": values["max_new_tokens"],
        }
        for name, values in MossTTSEngine.MODEL_VARIANTS.items()
    }

    @classmethod
    def NAME(cls):
        return "⚙️ MOSS-TTS Engine"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (cls.STANDARD_MODEL_OPTIONS, {
                    "default": "Small 1.7B (Local)",
                    "tooltip": (
                        "Standard MOSS-TTS model for normal/custom character switching.\n"
                        "Small 1.7B (Local): smaller official local-transformer model.\n"
                        "8B (Delay): larger official delay model.\n"
                        "Native Multi-Speaker Dialogue mode ignores this selector and uses MOSS-TTSD-v1.0 automatically."
                    )
                }),
                "multi_speaker_mode": (["Custom Character Switching", "Native Multi-Speaker Dialogue"], {
                    "default": "Custom Character Switching",
                    "tooltip": (
                        "Custom Character Switching uses normal MOSS-TTS per character block.\n"
                        "Native Multi-Speaker Dialogue uses MOSS-TTSD-v1.0 with [S1]...[S5] speaker mapping in one dialogue context."
                    )
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device for MOSS-TTS. CUDA is strongly recommended; CPU is very slow."
                }),
                "language": ([
                    "Auto", "Chinese", "English", "German", "Spanish", "French",
                    "Japanese", "Italian", "Hungarian", "Korean", "Russian",
                    "Persian", "Arabic", "Polish", "Portuguese", "Czech",
                    "Danish", "Swedish", "Greek", "Turkish"
                ], {
                    "default": "Auto",
                    "tooltip": "Optional official language hint. Auto leaves language unset for MOSS-TTS."
                }),
                "sampler_preset": (["Model default", "Custom"], {
                    "default": "Model default",
                    "tooltip": "Model default uses OpenMOSS recommended sampling parameters for the selected variant."
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 2.5, "step": 0.05,
                    "tooltip": "Official audio_temperature. Ignored when sampler_preset is Model default."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Official audio_top_p. Ignored when sampler_preset is Model default."
                }),
                "top_k": ("INT", {
                    "default": 50, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Official audio_top_k. Ignored when sampler_preset is Model default."
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.1, "min": 0.5, "max": 3.0, "step": 0.05,
                    "tooltip": "Official audio_repetition_penalty. Ignored when sampler_preset is Model default."
                }),
                "max_new_tokens": ("INT", {
                    "default": 4096, "min": 64, "max": 16384, "step": 64,
                    "tooltip": "Maximum generated tokens. Higher values allow longer outputs and use more VRAM/time."
                }),
            },
            "optional": {
                "instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "How the whole segment should be spoken.\n"
                        "Best used as an engine default, or per segment with [instruction:...].\n"
                        "Use short natural instructions.\n"
                        "Examples:\n"
                        "• Speak softly and calmly\n"
                        "• Read like a documentary narrator\n"
                        "• Sound excited and energetic\n"
                        "This affects the full segment, not one exact word.\n"
                        "Do not use <instruction:...> for MOSS. <> should stay reserved for real inline post-processing tags."
                    )
                }),
                "quality": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Overall recording or presentation quality for the whole segment.\n"
                        "Best used as an engine default, or per segment with [quality:...].\n"
                        "Use short descriptive phrases.\n"
                        "Examples:\n"
                        "• Studio recording\n"
                        "• Clean close-mic voice\n"
                        "• Telephone call quality\n"
                        "• Distant PA system\n"
                        "This applies to the full segment.\n"
                        "Do not use <quality:...> for MOSS. <> should stay reserved for real inline post-processing tags."
                    )
                }),
                "sound_event": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Main sound effect or vocal event for the whole segment.\n"
                        "Best used as an engine default, or per segment with [sound_event:...].\n"
                        "Use short event names.\n"
                        "Examples:\n"
                        "• Laughter\n"
                        "• Sigh\n"
                        "• Breathing\n"
                        "• Crying\n"
                        "Important: this is not exact placement.\n"
                        "Use this only for whole-segment conditioning. Do not use <> for MOSS sound events."
                    )
                }),
                "ambient_sound": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Background ambience for the whole segment.\n"
                        "Best used as an engine default, or per segment with [ambient_sound:...].\n"
                        "Use short environment descriptions.\n"
                        "Examples:\n"
                        "• Rain\n"
                        "• Crowd\n"
                        "• Forest ambience\n"
                        "• Office room tone\n"
                        "This affects the full segment, not an exact point in the sentence.\n"
                        "Do not use <ambient_sound:...> for MOSS. <> should stay reserved for real inline post-processing tags."
                    )
                }),
                "duration_tokens": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1,
                    "tooltip": (
                        "Target output length hint. This is audio tokens, not text tokens.\n"
                        "It can indirectly affect pacing:\n"
                        "• Lower values usually make speech shorter and tighter\n"
                        "• Higher values usually make speech longer and slower-feeling\n"
                        "This is not a true speed control.\n"
                        "\n"
                        "Rough guide:\n"
                        "• 12-13 = about 1 second\n"
                        "• 25 = about 2 seconds\n"
                        "• 50 = about 4 seconds\n"
                        "• 125 = about 10 seconds\n"
                        "Use lower values for shorter delivery, higher values for longer delivery.\n"
                        "Set 0 to disable.\n"
                        "If audio is getting cut off, raise max_new_tokens too."
                    )
                }),
                "n_vq_for_inference": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Local Transformer only: number of RVQ layers/codebooks for inference. 0 uses model default."
                }),
                "dtype": (["auto", "bfloat16", "float16", "float32"], {
                    "default": "auto",
                    "tooltip": "Model precision. Auto uses bf16 on capable CUDA GPUs, fp16 on older CUDA GPUs, fp32 on CPU."
                }),
                "attn_implementation": (["auto", "flash_attention_2", "sdpa", "eager"], {
                    "default": "auto",
                    "tooltip": "Attention backend. Auto prefers FlashAttention 2 when installed and supported, otherwise SDPA/eager."
                }),
                "codec_model": (["MOSS-Audio-Tokenizer"], {
                    "default": "MOSS-Audio-Tokenizer",
                    "tooltip": "Official shared MOSS audio tokenizer required by MOSS-TTS."
                }),
                "speaker2_voice": (any_typ, {
                    "tooltip": "Voice for S2 in Native Multi-Speaker Dialogue. Connect Character Voices opt_narrator output."
                }),
                "speaker3_voice": (any_typ, {
                    "tooltip": "Voice for S3 in Native Multi-Speaker Dialogue. Connect Character Voices opt_narrator output."
                }),
                "speaker4_voice": (any_typ, {
                    "tooltip": "Voice for S4 in Native Multi-Speaker Dialogue. Connect Character Voices opt_narrator output."
                }),
                "speaker5_voice": (any_typ, {
                    "tooltip": "Voice for S5 in Native Multi-Speaker Dialogue. Connect Character Voices opt_narrator output."
                }),
            }
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_adapter"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    @classmethod
    def _get_model_variants(cls) -> List[str]:
        variants = list(MossTTSEngine.MODEL_VARIANTS.keys())
        try:
            for base_path in get_all_tts_model_paths("TTS"):
                moss_dir = os.path.join(base_path, "moss_tts")
                if not os.path.isdir(moss_dir):
                    continue
                for name in sorted(os.listdir(moss_dir)):
                    candidate = os.path.join(moss_dir, name)
                    if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
                        local_name = f"local:{name}"
                        if local_name not in variants:
                            variants.insert(0, local_name)
        except Exception:
            pass
        return variants

    @classmethod
    def _resolve_model_variant(cls, model_variant: str, multi_speaker_mode: str) -> str:
        if multi_speaker_mode == "Native Multi-Speaker Dialogue":
            return "MOSS-TTSD-v1.0"

        if model_variant in cls.UI_MODEL_VARIANT_MAP:
            return cls.UI_MODEL_VARIANT_MAP[model_variant]

        if model_variant == "MOSS-TTSD-v1.0":
            return "MOSS-TTS-Local-Transformer"

        return model_variant

    def create_engine_adapter(
        self,
        model_variant: str,
        multi_speaker_mode: str,
        device: str,
        language: str,
        sampler_preset: str,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        max_new_tokens: int,
        instruction: str = "",
        quality: str = "",
        sound_event: str = "",
        ambient_sound: str = "",
        duration_tokens: int = 0,
        n_vq_for_inference: int = 0,
        dtype: str = "auto",
        attn_implementation: str = "auto",
        codec_model: str = "MOSS-Audio-Tokenizer",
        speaker2_voice=None,
        speaker3_voice=None,
        speaker4_voice=None,
        speaker5_voice=None,
    ):
        resolved_model_variant = self._resolve_model_variant(model_variant, multi_speaker_mode)
        if resolved_model_variant != model_variant:
            print(f"🔄 MOSS-TTS: Resolved model selection '{model_variant}' -> '{resolved_model_variant}'")

        resolved_variant = (
            resolved_model_variant.replace("local:", "")
            if resolved_model_variant.startswith("local:")
            else resolved_model_variant
        )
        defaults = self.MODEL_DEFAULTS.get(resolved_variant, self.MODEL_DEFAULTS["MOSS-TTS-Local-Transformer"])

        if sampler_preset == "Model default":
            temperature = defaults["temperature"]
            top_p = defaults["top_p"]
            top_k = int(defaults["top_k"])
            repetition_penalty = defaults["repetition_penalty"]

        config = {
            "engine_type": "moss_tts",
            "model_variant": resolved_model_variant,
            "multi_speaker_mode": multi_speaker_mode,
            "device": device,
            "language": language,
            "sampler_preset": sampler_preset,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "max_new_tokens": int(max_new_tokens),
            "instruction": str(instruction or "").strip() or None,
            "quality": str(quality or "").strip() or None,
            "sound_event": str(sound_event or "").strip() or None,
            "ambient_sound": str(ambient_sound or "").strip() or None,
            "duration_tokens": int(duration_tokens) if duration_tokens else None,
            "n_vq_for_inference": int(n_vq_for_inference) if n_vq_for_inference else None,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "codec_model": codec_model,
            "speaker2_voice": speaker2_voice,
            "speaker3_voice": speaker3_voice,
            "speaker4_voice": speaker4_voice,
            "speaker5_voice": speaker5_voice,
        }

        print(f"⚙️ MOSS-TTS: Configured {resolved_model_variant} on {device}")
        print(f"   Mode: {multi_speaker_mode}")
        print(
            "   Language={language}, temp={temperature}, top_p={top_p}, top_k={top_k}, rep_penalty={rep}, max_new_tokens={max_new_tokens}".format(
                language=language,
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                rep=config["repetition_penalty"],
                max_new_tokens=config["max_new_tokens"],
            )
        )
        if config["duration_tokens"]:
            print(f"   Duration hint: {config['duration_tokens']} audio tokens")
        prompt_fields = [
            f"{field_name}={config[field_name]}"
            for field_name in ("instruction", "quality", "sound_event", "ambient_sound")
            if config.get(field_name)
        ]
        if prompt_fields:
            print(f"   Official prompt fields: {', '.join(prompt_fields)}")

        return ({
            "engine_type": "moss_tts",
            "config": config,
            "adapter_class": "MossTTSEngineAdapter",
        },)
