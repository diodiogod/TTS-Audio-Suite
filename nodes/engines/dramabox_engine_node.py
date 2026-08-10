"""DramaBox engine configuration node."""

import importlib.util
import os
import sys
from typing import List

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

from engines.dramabox.dramabox_downloader import DramaBoxDownloader
from utils.models.extra_paths import get_all_tts_model_paths


class DramaBoxEngineNode(BaseTTSNode):
    """Configure official DramaBox expressive English TTS."""

    @classmethod
    def NAME(cls):
        return "⚙️ DramaBox Engine"

    @classmethod
    def INPUT_TYPES(cls):
        models = DramaBoxDownloader().get_available_models()
        return {
            "required": {
                "model_name": (models, {
                    "default": "DramaBox",
                    "tooltip": (
                        "Official Resemble AI DramaBox 3.3B checkpoint.\n"
                        "Downloads about 16.4 GB. Fast mode targets roughly 24 GB VRAM; "
                        "experimental staged modes can use less.\n"
                        "LTX-2 Community License: entities with at least USD 10M "
                        "annual revenue need a paid commercial license."
                    ),
                }),
                "device": (["auto", "cuda"], {
                    "default": "auto",
                    "tooltip": "DramaBox requires an NVIDIA CUDA GPU; CPU inference is unsupported.",
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 2.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Official CFG scale. Lower is more natural; higher is more text-faithful.",
                }),
                "negative_prompt": ("STRING", {
                    "default": (
                        "worst quality, inconsistent, robotic, distorted, noise, static, "
                        "muffled, unclear, unnatural, monotone"
                    ),
                    "multiline": True,
                    "tooltip": "CFG negative prompt: sounds or qualities to discourage.",
                }),
                "stg_scale": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Official skip-token guidance scale.",
                }),
                "duration_multiplier": ("FLOAT", {
                    "default": 1.1,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.05,
                    "tooltip": (
                        "Scales DramaBox's native estimated output duration. "
                        "Increase for more breathing room; decrease for tighter speech."
                    ),
                }),
                "gen_duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 60.0,
                    "step": 0.5,
                    "tooltip": (
                        "Explicit duration for each generated segment. 0 uses automatic "
                        "prompt-based estimation and duration_multiplier. Keep 0 for normal "
                        "SRT generation until native duration targeting is integrated."
                    ),
                }),
                "ref_duration": ("FLOAT", {
                    "default": 10.0,
                    "min": 3.0,
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": (
                        "Seconds used from the beginning of the voice reference. "
                        "Audio after this point is ignored."
                    ),
                }),
                "rescale_scale": ("STRING", {
                    "default": "auto",
                    "multiline": False,
                    "tooltip": (
                        "CFG latent rescaling: auto, or a fixed value from 0 to 1. "
                        "Auto adjusts rescaling from cfg_scale."
                    ),
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Apply the official imperceptible Perth watermark. Disabled by "
                        "default to keep watermarking optional; requires Perth."
                    ),
                }),
                "prompt_template": ("STRING", {
                    "default": '"{seg}"',
                    "multiline": False,
                    "tooltip": (
                        "Template applied to each plain segment. {seg} is replaced by the "
                        "segment text. Default: \"{seg}\" (literal spoken dialogue). Example "
                        "with a delivery description: A man speaks warmly, \"{seg}\". Clear "
                        "the field to send text unchanged. Non-empty templates should contain "
                        "{seg}; if omitted, a quoted {seg} is appended automatically with a "
                        "console warning. Complete scene prompts are preserved."
                    ),
                }),
            },
            "optional": {
                "precision": (["auto", "bfloat16", "float16"], {
                    "default": "auto",
                    "tooltip": "LTX inference precision. Auto prefers bfloat16 on Ampere-or-newer GPUs.",
                }),
                "memory_mode": (["fast", "staged", "sequential"], {
                    "default": "fast",
                    "tooltip": (
                        "Fast keeps every component on CUDA. Staged and Sequential are "
                        "experimental strategies for lowering peak VRAM. Staged releases "
                        "temporary components; Sequential runs one major GPU stage at a time."
                    ),
                }),
                "transformer_quantization": (["none", "fp8_cast"], {
                    "default": "none",
                    "tooltip": (
                        "Official LTX FP8 weight-storage policy for the diffusion transformer. "
                        "fp8_cast lowers VRAM but upcasts each linear layer during inference. "
                        "DramaBox LoRAs remain as an unmerged BF16 branch over the FP8 base."
                    ),
                }),
                "compile_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Compile the diffusion transformer blocks. First generation is much "
                        "slower and may reserve more VRAM; later denoising can be faster."
                    ),
                }),
                "local_lora_adapter": (cls._get_ui_lora_options(), {
                    "default": "None",
                    "tooltip": (
                        "Optional DramaBox audio LoRA discovered under models/TTS/dramabox/loras. "
                        "Training outputs are copied there when a run completes."
                    ),
                }),
                "lora_adapter_override": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Advanced local path to a DramaBox LoRA file or adapter folder. "
                        "If filled, this overrides the local adapter dropdown."
                    ),
                }),
                "lora_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Scale applied to the trained DramaBox LoRA. 1.0 uses the adapter's trained strength; 0 disables it.",
                }),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine_config(
        self,
        model_name: str,
        device: str,
        cfg_scale: float,
        negative_prompt: str,
        stg_scale: float,
        duration_multiplier: float,
        gen_duration: float,
        ref_duration: float,
        rescale_scale: str,
        watermark: bool,
        prompt_template: str,
        precision: str = "auto",
        memory_mode: str = "fast",
        transformer_quantization: str = "none",
        compile_model: bool = False,
        local_lora_adapter: str = "None",
        lora_adapter_override: str = "",
        lora_strength: float = 1.0,
    ) -> tuple:
        lora_path = self._resolve_lora_adapter(local_lora_adapter, lora_adapter_override)
        config = {
            "engine_type": "dramabox",
            "model_name": model_name,
            "device": device,
            "precision": precision,
            "cfg_scale": float(cfg_scale),
            "stg_scale": float(stg_scale),
            "duration_multiplier": float(duration_multiplier),
            "gen_duration": float(gen_duration),
            "ref_duration": float(ref_duration),
            "rescale_scale": self._validate_rescale_scale(rescale_scale),
            "watermark": bool(watermark),
            "prompt_template": str(prompt_template or ""),
            "negative_prompt": str(negative_prompt or ""),
            "memory_mode": str(memory_mode),
            "transformer_quantization": str(transformer_quantization),
            "compile_model": bool(compile_model),
            "lora_path": lora_path,
            "lora_strength": float(lora_strength),
        }
        print(f"⚙️ DramaBox: {model_name} on {device} ({precision})")
        print(
            f"   Settings: cfg_scale={cfg_scale}, stg_scale={stg_scale}, "
            f"duration_multiplier={duration_multiplier}, gen_duration={gen_duration}, "
            f"ref_duration={ref_duration}, rescale_scale={rescale_scale}, "
            f"watermark={watermark}, memory_mode={memory_mode}, "
            f"transformer_quantization={transformer_quantization}, compile={compile_model}"
        )
        if lora_path:
            print(f"   LoRA: {lora_path} (strength={float(lora_strength):.2f})")
        print("   Prompt: dialogue in quotes; expressive stage directions outside quotes")
        return ({
            "engine_type": "dramabox",
            "config": config,
            "capabilities": ["tts"],
        },)

    @classmethod
    def _discover_local_loras(cls) -> List[str]:
        discovered: List[str] = []
        seen = set()
        try:
            for base_path in get_all_tts_model_paths("TTS"):
                root = os.path.join(base_path, "dramabox", "loras")
                if not os.path.isdir(root):
                    continue
                for name in sorted(os.listdir(root)):
                    candidate = os.path.join(root, name)
                    if os.path.isdir(candidate):
                        has_weights = any(
                            filename.endswith(".safetensors")
                            for filename in os.listdir(candidate)
                        )
                    else:
                        has_weights = os.path.isfile(candidate) and candidate.endswith(".safetensors")
                    if has_weights and f"local:{name}" not in seen:
                        seen.add(f"local:{name}")
                        discovered.append(f"local:{name}")
        except Exception:
            pass
        return discovered

    @classmethod
    def _get_ui_lora_options(cls) -> List[str]:
        return ["None"] + cls._discover_local_loras()

    @classmethod
    def _resolve_lora_adapter(cls, local_value: str, override: str) -> str:
        manual = str(override or "").strip()
        if manual:
            return os.path.abspath(os.path.expanduser(manual))
        selected = str(local_value or "").strip()
        if not selected or selected == "None":
            return ""
        if selected.startswith("local:"):
            name = selected.split(":", 1)[1]
            for base_path in get_all_tts_model_paths("TTS"):
                candidate = os.path.join(base_path, "dramabox", "loras", name)
                if os.path.exists(candidate):
                    return candidate
        return os.path.abspath(os.path.expanduser(selected))

    @staticmethod
    def _validate_rescale_scale(value: str):
        text = str(value).strip().lower()
        if text == "auto":
            return "auto"
        try:
            numeric = float(text)
        except ValueError as exc:
            raise ValueError("rescale_scale must be 'auto' or a number from 0 to 1") from exc
        if not 0.0 <= numeric <= 1.0:
            raise ValueError("rescale_scale must be between 0 and 1")
        return numeric
