"""
Echo-TTS Engine Configuration Node

Provides Echo-TTS configuration for TTS Audio Suite.
Uses upstream parameter names and example defaults.
"""

import os
import sys
import importlib.util
from typing import Dict, Any

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load base_node module directly
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseTTSNode = base_module.BaseTTSNode


class EchoTTSEngineNode(BaseTTSNode):
    """
    Echo-TTS engine configuration node.
    """

    SAMPLER_PRESETS = {
        "Independent-High-Speaker-CFG": {
            "num_steps": 40,
            "cfg_scale_text": 3.0,
            "cfg_scale_speaker": 8.0,
            "cfg_min_t": 0.5,
            "cfg_max_t": 1.0,
            "truncation_factor": 1.0,
            "rescale_k": 1.0,
            "rescale_sigma": 3.0,
        },
        "Independent-High-Speaker-CFG-Flat": {
            "num_steps": 40,
            "cfg_scale_text": 3.0,
            "cfg_scale_speaker": 8.0,
            "cfg_min_t": 0.5,
            "cfg_max_t": 1.0,
            "truncation_factor": 0.8,
            "rescale_k": 1.2,
            "rescale_sigma": 3.0,
        },
        "Independent-High-CFG": {
            "num_steps": 40,
            "cfg_scale_text": 8.0,
            "cfg_scale_speaker": 8.0,
            "cfg_min_t": 0.5,
            "cfg_max_t": 1.0,
            "truncation_factor": 1.0,
            "rescale_k": 1.0,
            "rescale_sigma": 3.0,
        },
        "Independent-High-CFG-Flat": {
            "num_steps": 40,
            "cfg_scale_text": 8.0,
            "cfg_scale_speaker": 8.0,
            "cfg_min_t": 0.5,
            "cfg_max_t": 1.0,
            "truncation_factor": 0.8,
            "rescale_k": 1.2,
            "rescale_sigma": 3.0,
        },
        "Independent-Low-CFG": {
            "num_steps": 40,
            "cfg_scale_text": 3.0,
            "cfg_scale_speaker": 3.0,
            "cfg_min_t": 0.5,
            "cfg_max_t": 1.0,
            "truncation_factor": 1.0,
            "rescale_k": 1.0,
            "rescale_sigma": 3.0,
        },
        "Independent-Low-CFG-Flat": {
            "num_steps": 40,
            "cfg_scale_text": 3.0,
            "cfg_scale_speaker": 3.0,
            "cfg_min_t": 0.5,
            "cfg_max_t": 1.0,
            "truncation_factor": 0.8,
            "rescale_k": 1.2,
            "rescale_sigma": 3.0,
        },
    }

    @classmethod
    def NAME(cls):
        return "Echo-TTS Engine"

    @classmethod
    def INPUT_TYPES(cls):
        preset_choices = ["Custom"] + list(cls.SAMPLER_PRESETS.keys())
        return {
            "required": {
                "preset": (preset_choices, {
                    "default": "Custom",
                    "tooltip": "Sampler preset from Echo-TTS. If not Custom, preset values override num_steps, CFG scales, CFG t-range, truncation, and rescale settings."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run Echo-TTS on:\n- auto: Use CUDA if available, else CPU\n- cuda: Force NVIDIA GPU (falls back to CPU if unavailable)\n- cpu: CPU-only (very slow)\nNote: CUDA recommended; Echo-TTS is non-commercial (CC-BY-NC-SA)."
                }),
                "num_steps": ("INT", {
                    "default": 40, "min": 1, "max": 200,
                    "tooltip": "Number of sampling steps. Higher can improve quality but increases time and VRAM."
                }),
                "cfg_scale_text": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Text guidance scale. Higher pushes closer to the prompt; too high can sound harsh."
                }),
                "cfg_scale_speaker": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Speaker guidance scale. Higher pushes closer to reference voice; too high can degrade quality."
                }),
                "cfg_min_t": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "CFG lower bound (0–1). Guidance is active when t >= cfg_min_t."
                }),
                "cfg_max_t": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "CFG upper bound (0–1). Guidance is active when t <= cfg_max_t."
                }),
                "truncation_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Truncation factor for initial noise. Lower can reduce diversity; higher can increase variation."
                }),
                "rescale_k": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Temporal score rescale k (see Echo-TTS rescaling). Only used if both rescale_k and rescale_sigma are set."
                }),
                "rescale_sigma": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Temporal score rescale sigma (see Echo-TTS rescaling). Only used if both rescale_k and rescale_sigma are set."
                }),
                "force_speaker_kv": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable speaker KV scaling to more strongly match the reference voice. Higher values may reduce quality. ⚠️ Not compatible with pause tags ([pause:...])."
                }),
                "speaker_kv_scale": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "KV Scale (>1 = stronger speaker influence). Used only when Force Speaker is enabled."
                }),
                "speaker_kv_max_layers": ("INT", {
                    "default": 24, "min": 0, "max": 24,
                    "tooltip": "Max layers to apply KV scaling (0–24). Used only when Force Speaker is enabled."
                }),
                "speaker_kv_min_t": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "KV Min t (0–1). Scaling is active until t drops below this value."
                }),
                "sequence_length": ("INT", {
                    "default": 640, "min": 64, "max": 2048,
                    "tooltip": "Sample latent length. 640 is ~30s (max seen during training). Best results at ≤30s; longer text uses unified chunking (best-effort)."
                }),
            }
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_adapter"
    CATEGORY = "TTS Audio Suite/Engines"

    def create_engine_adapter(self, preset: str, device: str, num_steps: int,
                              cfg_scale_text: float, cfg_scale_speaker: float,
                              cfg_min_t: float, cfg_max_t: float,
                              truncation_factor: float, rescale_k: float, rescale_sigma: float,
                              force_speaker_kv: bool, speaker_kv_scale: float, speaker_kv_max_layers: int, speaker_kv_min_t: float,
                              sequence_length: int):
        """Create Echo-TTS engine configuration."""
        try:
            # Import adapter to ensure dependency is available
            from engines.adapters.echo_tts_adapter import EchoTTSEngineAdapter
            _ = EchoTTSEngineAdapter  # silence unused

            config = {
                "engine_type": "echo_tts",
                "preset": preset,
                "model": "jordand/echo-tts-base",
                "device": device,
                "num_steps": int(num_steps),
                "cfg_scale_text": float(cfg_scale_text),
                "cfg_scale_speaker": float(cfg_scale_speaker),
                "cfg_min_t": float(cfg_min_t),
                "cfg_max_t": float(cfg_max_t),
                "truncation_factor": float(truncation_factor),
                "rescale_k": float(rescale_k),
                "rescale_sigma": float(rescale_sigma),
                "speaker_kv_scale": float(speaker_kv_scale) if force_speaker_kv else None,
                "speaker_kv_max_layers": int(speaker_kv_max_layers) if force_speaker_kv else None,
                "speaker_kv_min_t": float(speaker_kv_min_t) if force_speaker_kv else None,
                "sequence_length": int(sequence_length),
            }

            if preset and preset != "Custom":
                preset_values = self.SAMPLER_PRESETS.get(preset, {})
                for key, value in preset_values.items():
                    config[key] = value

            print(f"Echo-TTS Engine: Configured model '{config['model']}' on {device}")
            if preset and preset != "Custom":
                print(f"   Preset={preset} (overrides sampler fields)")
            print(
                "   Steps={steps}, cfg_text={cfg_text}, cfg_speaker={cfg_speaker}, seq_len={seq_len}".format(
                    steps=config["num_steps"],
                    cfg_text=config["cfg_scale_text"],
                    cfg_speaker=config["cfg_scale_speaker"],
                    seq_len=config["sequence_length"],
                )
            )

            engine_data = {
                "engine_type": "echo_tts",
                "config": config,
                "adapter_class": "EchoTTSEngineAdapter"
            }

            return (engine_data,)

        except Exception as e:
            print(f"ERROR: Echo-TTS Engine error: {e}")
            import traceback
            traceback.print_exc()

            # Return error config for graceful fallback
            error_config = {
                "engine_type": "echo_tts",
                "config": {
                    "model": model,
                    "device": "cpu",
                    "error": str(e)
                },
                "adapter_class": "EchoTTSEngineAdapter"
            }
            return (error_config,)


# Register the node class
NODE_CLASS_MAPPINGS = {
    "EchoTTSEngineNode": EchoTTSEngineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EchoTTSEngineNode": "Echo-TTS Engine"
}
