"""
MOSS-TTS Engine Configuration Node.

Configures the official OpenMOSS MOSS-TTS models for the unified TTS nodes.
"""

import os
import sys
import importlib.util
from typing import Dict, List
import re

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
from utils.downloads.unified_downloader import UnifiedDownloader


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
    LEGACY_LOCAL_MODEL_OPTIONS = [
        "local:MOSS-TTS-Local-Transformer",
        "local:MOSS-TTS",
        "local:MOSS-TTSD-v1.0",
    ]
    FRIENDLY_MODEL_ALIASES = [
        "Small 1.7B (Local)",
        "8B (Delay)",
        "Native 8B Dialogue (MOSS-TTSD-v1.0)",
    ]
    UI_MODEL_VARIANT_MAP = {
        "Small 1.7B (Local)": "MOSS-TTS-Local-Transformer",
        "8B (Delay)": "MOSS-TTS",
        "Native 8B Dialogue (MOSS-TTSD-v1.0)": "MOSS-TTSD-v1.0",
    }
    NO_LORA_OPTION = "None"

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
                "model_variant": (cls._get_ui_model_options(), {
                    "default": cls._get_ui_standard_model_options()[0],
                    "tooltip": (
                        "MOSS-TTS model selection.\n"
                        "\n"
                        "If a local model folder is detected, the dropdown shows local:ModelName.\n"
                        "Otherwise it shows friendly built-in choices.\n"
                        "\n"
                        "Small 1.7B (Local): smaller official local-transformer model.\n"
                        "8B (Delay): larger official delay model.\n"
                        "\n"
                        "Native Multi-Speaker Dialogue mode ignores this selector and uses MOSS-TTSD-v1.0 automatically."
                    )
                }),
                "multi_speaker_mode": (["Custom Character Switching", "Native Multi-Speaker Dialogue"], {
                    "default": "Custom Character Switching",
                    "tooltip": (
                        "Custom Character Switching uses normal MOSS-TTS per character block.\n"
                        "\n"
                        "Native Multi-Speaker Dialogue uses MOSS-TTSD-v1.0 with [S1]...[S5] speaker mapping in one dialogue context.\n"
                        "\n"
                        "Native mode hard-fails (no auto-fallback) if incompatible controls are detected:\n"
                        "• pause tags\n"
                        "• inline edit tags\n"
                        "• per-segment parameter changes\n"
                        "• more than 5 speakers\n"
                        "Switch to Custom Character Switching and choose a standard MOSS model for those cases."
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
                    "tooltip": (
                        "Optional language hint for MOSS-TTS.\n"
                        "Auto does not run a separate language detector.\n"
                        "It simply sends no language hint, so MOSS must infer the language from the text itself.\n"
                        "Use an explicit language when the text is short, ambiguous, mixed-language, or when Auto guesses wrong."
                    )
                }),
                "duration_tokens": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1,
                    "tooltip": (
                        "Target output length hint. This is audio tokens, not text tokens.\n"
                        "This is one of the most useful MOSS controls for pacing.\n"
                        "It can indirectly affect pacing:\n"
                        "• Lower values usually make speech shorter and tighter\n"
                        "• Higher values usually make speech longer and slower-feeling\n"
                        "This is not a true speed control.\n"
                        "Very low values can behave oddly, so avoid extreme settings.\n"
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
                "chunk_minutes": ("INT", {
                    "default": 0, "min": 0, "max": 90, "step": 1,
                    "tooltip": (
                        "Time-based chunking override for MOSS (like VibeVoice).\n"
                        "0: disable chunking entirely and ignore Unified chunk controls.\n"
                        ">0: force chunking using approx 750 chars/min and ignore Unified chunk controls.\n"
                        "Applies to both Custom Character Switching and Native Multi-Speaker Dialogue."
                    )
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
                        "Status: experimental on base MOSS-TTS in this suite.\n"
                        "It is a real official field, but audible effect may be weak or inconsistent.\n"
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
                        "Status: experimental on base MOSS-TTS in this suite.\n"
                        "It is a real official field, but audible effect may be weak or inconsistent.\n"
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
                        "Status: experimental on base MOSS-TTS in this suite.\n"
                        "OpenMOSS publicly documents ambient-sound prompting mainly for MOSS-SoundEffect, not as a proven short-utterance TTS control.\n"
                        "So results may be weak, absent, or unstable.\n"
                        "This affects the full segment, not an exact point in the sentence.\n"
                        "Important: ambience can make MOSS keep generating until it reaches the requested duration budget.\n"
                        "If it runs too long, lower max_new_tokens or set duration_tokens.\n"
                        "Do not use <ambient_sound:...> for MOSS. <> should stay reserved for real inline post-processing tags."
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
                "local_lora_adapter": (cls._get_ui_lora_options(), {
                    "default": cls.NO_LORA_OPTION,
                    "tooltip": (
                        "Optional local MOSS LoRA adapter discovered under models/TTS/moss_tts/loras.\n"
                        "\n"
                        "This expects a PEFT adapter folder, not just a raw weight file.\n"
                        "Training should save a full adapter directory with adapter_config.json."
                    )
                }),
                "lora_adapter_override": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Advanced override for MOSS LoRA adapter inference.\n"
                        "\n"
                        "Accepts a local adapter folder path or Hugging Face repo id.\n"
                        "Example: ToSee-Norway/MOSS-TTS-Norwegian-LoRA\n"
                        "\n"
                        "If you enter a Hugging Face repo id, it will be installed into models/TTS/moss_tts/loras and then loaded locally.\n"
                        "If this field is filled, it overrides the local LoRA dropdown."
                    )
                }),
            },
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
    def _find_local_variant(cls, model_name: str) -> str:
        try:
            for base_path in get_all_tts_model_paths("TTS"):
                candidate = os.path.join(base_path, "moss_tts", model_name)
                if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
                    return f"local:{model_name}"
        except Exception:
            pass
        return model_name

    @classmethod
    def _get_ui_standard_model_options(cls) -> List[str]:
        small = cls._find_local_variant("MOSS-TTS-Local-Transformer")
        delay = cls._find_local_variant("MOSS-TTS")
        return [
            small if small.startswith("local:") else "Small 1.7B (Local)",
            delay if delay.startswith("local:") else "8B (Delay)",
        ]

    @classmethod
    def _get_ui_native_model_option(cls) -> str:
        native = cls._find_local_variant("MOSS-TTSD-v1.0")
        return native if native.startswith("local:") else cls.NATIVE_MODEL_OPTION

    @classmethod
    def _get_ui_model_options(cls) -> List[str]:
        values = cls._get_ui_standard_model_options() + [cls._get_ui_native_model_option()]
        for option in cls.FRIENDLY_MODEL_ALIASES + cls.LEGACY_LOCAL_MODEL_OPTIONS:
            if option not in values:
                values.append(option)
        return values

    @classmethod
    def _discover_local_lora_adapters(cls) -> List[str]:
        discovered: List[str] = []
        seen = set()
        try:
            for base_path in get_all_tts_model_paths("TTS"):
                lora_root = os.path.join(base_path, "moss_tts", "loras")
                if not os.path.isdir(lora_root):
                    continue
                for name in sorted(os.listdir(lora_root)):
                    candidate = os.path.join(lora_root, name)
                    if not os.path.isdir(candidate):
                        continue
                    if not os.path.exists(os.path.join(candidate, "adapter_config.json")):
                        continue
                    label = f"local:{name}"
                    if label not in seen:
                        seen.add(label)
                        discovered.append(label)
        except Exception:
            pass
        return discovered

    @classmethod
    def _get_ui_lora_options(cls) -> List[str]:
        return [cls.NO_LORA_OPTION] + cls._discover_local_lora_adapters()

    @classmethod
    def _get_moss_lora_root(cls) -> str:
        from utils.models.extra_paths import get_preferred_download_path

        moss_root = get_preferred_download_path("TTS", "moss_tts")
        lora_root = os.path.join(moss_root, "loras")
        os.makedirs(lora_root, exist_ok=True)
        return lora_root

    @classmethod
    def _repo_id_to_local_lora_name(cls, repo_id: str) -> str:
        safe = str(repo_id or "").strip().replace("\\", "/").strip("/")
        safe = safe.replace("/", "__")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", safe)
        return safe or "moss_lora"

    @classmethod
    def _looks_like_hf_repo_id(cls, value: str) -> bool:
        text = str(value or "").strip()
        if not text or os.path.isabs(text) or os.path.exists(text):
            return False
        if "://" in text or text.startswith(".") or text.startswith("~"):
            return False
        parts = text.split("/")
        return len(parts) == 2 and all(parts)

    @classmethod
    def _install_hf_lora_adapter(cls, repo_id: str) -> str:
        local_name = cls._repo_id_to_local_lora_name(repo_id)
        target_dir = os.path.join(cls._get_moss_lora_root(), local_name)
        config_path = os.path.join(target_dir, "adapter_config.json")
        has_weights = False
        if os.path.isdir(target_dir):
            for filename in os.listdir(target_dir):
                if filename.endswith(".safetensors") or filename.endswith(".bin"):
                    has_weights = True
                    break
        if os.path.exists(config_path) and has_weights:
            return target_dir

        downloader = UnifiedDownloader()
        downloader.download_huggingface_snapshot(
            repo_id=repo_id,
            target_dir=target_dir,
            allow_patterns=[
                "adapter_config.json",
                "*.safetensors",
                "*.bin",
                "*.json",
                "*.txt",
                "*.model",
            ],
            required_files=["adapter_config.json"],
            description=f"MOSS LoRA {repo_id}",
        )
        return target_dir

    @classmethod
    def _resolve_lora_adapter(
        cls,
        local_lora_adapter: str,
        lora_adapter_override: str,
        legacy_lora_adapter: str,
    ) -> str:
        manual = str(lora_adapter_override or legacy_lora_adapter or "").strip()
        if manual:
            if cls._looks_like_hf_repo_id(manual):
                return cls._install_hf_lora_adapter(manual)
            return manual

        local_value = str(local_lora_adapter or "").strip()
        if not local_value or local_value == cls.NO_LORA_OPTION:
            return ""

        if local_value.startswith("local:"):
            adapter_name = local_value.split(":", 1)[1]
            for base_path in get_all_tts_model_paths("TTS"):
                candidate = os.path.join(base_path, "moss_tts", "loras", adapter_name)
                if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "adapter_config.json")):
                    return candidate

        if cls._looks_like_hf_repo_id(local_value):
            return cls._install_hf_lora_adapter(local_value)

        return local_value

    @classmethod
    def _resolve_model_variant(cls, model_variant: str, multi_speaker_mode: str) -> str:
        if multi_speaker_mode == "Native Multi-Speaker Dialogue":
            return cls._find_local_variant("MOSS-TTSD-v1.0")

        if model_variant.startswith("local:") or model_variant in MossTTSEngine.MODEL_VARIANTS:
            return model_variant

        if model_variant in cls.UI_MODEL_VARIANT_MAP:
            return cls._find_local_variant(cls.UI_MODEL_VARIANT_MAP[model_variant])

        if model_variant == "MOSS-TTSD-v1.0":
            return cls._find_local_variant("MOSS-TTS-Local-Transformer")

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
        chunk_minutes: int,
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
        local_lora_adapter: str = "None",
        lora_adapter_override: str = "",
    ):
        resolved_model_variant = self._resolve_model_variant(model_variant, multi_speaker_mode)
        if resolved_model_variant != model_variant:
            print(f"🔄 MOSS-TTS: Resolved model selection '{model_variant}' -> '{resolved_model_variant}'")
        resolved_lora_adapter = self._resolve_lora_adapter(local_lora_adapter, lora_adapter_override, "")

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

        chunk_minutes_value = int(chunk_minutes) if chunk_minutes else 0
        chunk_chars = chunk_minutes_value * 750 if chunk_minutes_value > 0 else 0

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
            "chunk_minutes": chunk_minutes_value,
            "chunk_chars": int(chunk_chars),
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
            "lora_adapter": resolved_lora_adapter or None,
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
        if chunk_minutes_value > 0:
            print(f"   Chunking override: every {chunk_minutes_value} min (~{chunk_chars} chars)")
        else:
            print("   Chunking override: disabled (ignores Unified chunk settings)")
        prompt_fields = [
            f"{field_name}={config[field_name]}"
            for field_name in ("instruction", "quality", "sound_event", "ambient_sound")
            if config.get(field_name)
        ]
        if prompt_fields:
            print(f"   Official prompt fields: {', '.join(prompt_fields)}")
        if config.get("lora_adapter"):
            print(f"   LoRA adapter: {config['lora_adapter']}")

        return ({
            "engine_type": "moss_tts",
            "config": config,
            "adapter_class": "MossTTSEngineAdapter",
        },)
