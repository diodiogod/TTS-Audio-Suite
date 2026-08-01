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
from utils.models.factory_config import RUNTIME_MODE_SHARED


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
                    "tooltip": "Official Hume TADA checkpoint.\n"
                    "• TADA-1B: English only, about 3.9 GB\n"
                    "• TADA-3B-ML: multilingual, about 8.9 GB\n"
                    "Both auto-download with an exact ungated redistribution of Meta's Llama 3.2 "
                    "tokenizer. Manual files are also supported under models/TTS/tada/.",
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "TADA is designed for CUDA. CPU inference is supported by the wrapper but extremely slow.",
                }),
                "language": (list(TADA_LANGUAGE_OPTIONS), {
                    "default": "English",
                    "tooltip": "Target and reference language. TADA-1B supports English only. "
                    "TADA-3B-ML supports English, Arabic, Chinese, German, Spanish, French, "
                    "Italian, Japanese, Polish, and Portuguese. A matching language aligner is downloaded on demand.",
                }),
                "acoustic_cfg_scale": ("FLOAT", {
                    "default": 1.6,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Official acoustic classifier-free guidance scale. Higher is not automatically better.",
                }),
                "duration_cfg_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Official duration classifier-free guidance scale.",
                }),
                "num_flow_matching_steps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Acoustic flow-matching steps. More steps cost more time; 10 is the official default.",
                }),
                "noise_temperature": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Noise injected into acoustic generation. Lower values are more deterministic.",
                }),
            },
            "optional": {
                "dtype": (["auto", "bfloat16", "float16", "float32"], {
                    "default": "auto",
                    "tooltip": "Model precision inside the isolated Transformers-4 runtime.",
                }),
                "cfg_schedule": (["cosine", "linear", "constant"], {
                    "default": "cosine",
                    "tooltip": "How acoustic and duration CFG vary over flow-matching steps.",
                }),
                "time_schedule": (["logsnr", "cosine", "uniform"], {
                    "default": "logsnr",
                    "tooltip": "Official ODE timestep schedule.",
                }),
                "negative_condition_source": (["negative_step_output", "prompt", "zero"], {
                    "default": "negative_step_output",
                    "tooltip": "Official CFG negative condition source. The default runs a parallel negative batch and uses more compute.",
                }),
                "speed_up_factor": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Native two-pass duration scaling. 0 disables it; values above 1 speak faster. "
                    "Enabling this performs a second full generation pass.",
                }),
                "num_transition_steps": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Number of prompt-to-target acoustic transition steps.",
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
    ) -> tuple:
        validate_tada_model_language(model_variant, language)
        config = {
            "engine_type": "tada",
            "model_variant": model_variant,
            "model_name": model_variant,
            "model_path": model_variant,
            "device": device,
            "dtype": dtype,
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
            "runtime_mode": RUNTIME_MODE_SHARED,
            "runtime_profile": self.RUNTIME_PROFILE,
        }

        print(f"⚙️ TADA: Configured {model_variant} on {device} ({language})")
        print(f"   Runtime: Shared Transformers-4 | dtype={dtype}")
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
