"""TADA engine configuration node."""

import importlib.util
import os
import sys

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

from engines.tada.languages import TADA_LANGUAGE_OPTIONS, validate_tada_model_language
from engines.tada.tada_downloader import TadaDownloader
from utils.models.factory_config import (
    RUNTIME_MODE_SHARED,
    normalize_runtime_mode,
)


RUNTIME_MODE_MAIN_LABEL = "Main Environment"
RUNTIME_MODE_SHARED_LABEL = "⚠️ Shared Runtime"


class TadaEngineNode(BaseTTSNode):
    """Configure official Hume TADA zero-shot voice cloning."""

    RUNTIME_PROFILE = "vibevoice_transformers4_shared"

    @classmethod
    def NAME(cls):
        return "⚙️ TADA Engine"

    @classmethod
    def INPUT_TYPES(cls):
        try:
            model_options = TadaDownloader().get_available_models()
        except Exception:
            model_options = ["TADA-1B", "TADA-3B-ML"]

        return {
            "required": {
                "model_variant": (model_options, {
                    "default": "TADA-1B",
                    "tooltip": "Choose the TADA model.\n"
                    "• TADA-1B: faster, English only (about 3.9 GB).\n"
                    "• TADA-3B-ML: multilingual (about 8.9 GB).\n"
                    "Non-English characters automatically use 3B-ML. Missing files download on first use.",
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Choose where TADA runs.\n"
                    "• Auto: recommended; uses an NVIDIA GPU when available.\n"
                    "• CUDA: force GPU use.\n"
                    "• CPU: works without a GPU, but is extremely slow.",
                }),
                "language": (list(TADA_LANGUAGE_OPTIONS), {
                    "default": "English",
                    "tooltip": "Language spoken in the reference clip and generated text.\n"
                    "This improves pronunciation and voice alignment.\n"
                    "Character language tags can change it automatically for each speaker.",
                }),
                "acoustic_cfg_scale": ("FLOAT", {
                    "default": 1.6,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "How strongly TADA follows the voice conditioning.\n"
                    "1.6 is recommended. Higher can sound stronger but less natural.\n"
                    "1.0 disables guidance and is faster.",
                }),
                "duration_cfg_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "How strongly guidance affects pauses and sound duration.\n"
                    "Keep 1.0 for natural automatic timing.\n"
                    "Higher values can produce unusual pacing.",
                }),
                "num_flow_matching_steps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of audio refinement steps.\n"
                    "10 is recommended. Fewer is faster but may reduce quality.\n"
                    "More is slower and may not improve the result.",
                }),
                "noise_temperature": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Controls variation in the generated voice.\n"
                    "0.9 is recommended. Lower is more predictable.\n"
                    "Higher adds variation but may create unstable sounds.",
                }),
            },
            "optional": {
                "dtype": (["auto", "bfloat16", "float16", "float32"], {
                    "default": "auto",
                    "tooltip": "Controls model precision and memory use.\n"
                    "Auto is recommended. Float16 and bfloat16 use less GPU memory.\n"
                    "Float32 uses much more memory and is mainly for troubleshooting.",
                }),
                "cfg_schedule": (["cosine", "linear", "constant"], {
                    "default": "cosine",
                    "tooltip": "How guidance changes during audio refinement.\n"
                    "• Cosine: smooth reduction; recommended.\n"
                    "• Linear: even reduction.\n"
                    "• Constant: full guidance throughout.",
                }),
                "time_schedule": (["logsnr", "cosine", "uniform"], {
                    "default": "logsnr",
                    "tooltip": "Controls where refinement steps are concentrated.\n"
                    "• LogSNR: recommended.\n"
                    "• Cosine: more work near the beginning and end.\n"
                    "• Uniform: steps are evenly spaced.",
                }),
                "negative_condition_source": (["negative_step_output", "prompt", "zero"], {
                    "default": "negative_step_output",
                    "tooltip": "Baseline used by voice guidance.\n"
                    "• negative_step_output: recommended; stronger but uses more compute.\n"
                    "• prompt: uses the reference prompt.\n"
                    "• zero: simplest and fastest baseline.",
                }),
                "speed_up_factor": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Changes speaking speed using TADA's native method.\n"
                    "0 disables it; values above 1 speak faster.\n"
                    "Enabling it performs two passes and takes about twice as long.",
                }),
                "num_transition_steps": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Smooths the change from the reference clip to generated speech.\n"
                    "5 is recommended. Lower is more abrupt.\n"
                    "Higher is smoother but adds work.",
                }),
                "use_torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Compile TADA's audio refinement stage for faster repeated generation.\n"
                    "The first run is slower while compiling; later runs reuse a persistent cache.\n"
                    "Uses extra disk space and may use slightly more GPU memory.",
                }),
                "runtime_mode": ([RUNTIME_MODE_SHARED_LABEL, RUNTIME_MODE_MAIN_LABEL], {
                    "default": RUNTIME_MODE_SHARED_LABEL,
                    "tooltip": "Choose the Python environment used by TADA.\n"
                    "• Shared Runtime: recommended; uses the compatible Transformers 4 environment.\n"
                    "• Main Environment: only works if ComfyUI already has compatible TADA dependencies.",
                }),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine_config(
        self,
        model_variant: str,
        device: str,
        language: str,
        acoustic_cfg_scale: float,
        duration_cfg_scale: float,
        num_flow_matching_steps: int,
        noise_temperature: float,
        dtype: str = "auto",
        cfg_schedule: str = "cosine",
        time_schedule: str = "logsnr",
        negative_condition_source: str = "negative_step_output",
        speed_up_factor: float = 0.0,
        num_transition_steps: int = 5,
        use_torch_compile: bool = False,
        runtime_mode: str = RUNTIME_MODE_SHARED_LABEL,
    ) -> tuple:
        validate_tada_model_language(model_variant, language)
        runtime_mode = normalize_runtime_mode(runtime_mode)
        runtime_profile = self.RUNTIME_PROFILE if runtime_mode == RUNTIME_MODE_SHARED else None
        config = {
            "engine_type": "tada",
            "model_variant": model_variant,
            "model_name": model_variant,
            "model_path": model_variant,
            "device": device,
            "dtype": dtype,
            "use_torch_compile": bool(use_torch_compile),
            "attn_implementation": "sdpa",
            "language": language,
            "acoustic_cfg_scale": float(acoustic_cfg_scale),
            "duration_cfg_scale": float(duration_cfg_scale),
            "num_flow_matching_steps": int(num_flow_matching_steps),
            "noise_temperature": float(noise_temperature),
            "cfg_schedule": cfg_schedule,
            "time_schedule": time_schedule,
            "negative_condition_source": negative_condition_source,
            "speed_up_factor": float(speed_up_factor),
            "num_transition_steps": int(num_transition_steps),
            "runtime_mode": runtime_mode,
            "runtime_profile": runtime_profile,
        }

        print(f"⚙️ TADA: Configured {model_variant} on {device} ({language})")
        runtime_label = "Shared Transformers-4" if runtime_mode == RUNTIME_MODE_SHARED else "Main Environment"
        print(
            f"   Runtime: {runtime_label} | dtype={dtype} | "
            f"torch.compile={'on' if use_torch_compile else 'off'}"
        )
        print(
            "   Settings: "
            f"acoustic_cfg={acoustic_cfg_scale}, duration_cfg={duration_cfg_scale}, "
            f"flow_steps={num_flow_matching_steps}, noise={noise_temperature}"
        )
        if speed_up_factor > 0:
            print(f"   Native two-pass speed factor: {speed_up_factor}")

        return ({
            "engine_type": "tada",
            "config": config,
            "capabilities": ["tts"],
        },)


__all__ = ["TadaEngineNode"]
