"""
Dots TTS engine configuration node.
"""

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

from engines.dots_tts.languages import DOTS_LANGUAGE_OPTIONS


class DotsTTSEngineNode(BaseTTSNode):
    """Dots TTS engine configuration node."""

    MODEL_VARIANTS = [
        "dots.tts-soar",
        "dots.tts-mf",
        "dots.tts-base",
    ]
    LANGUAGE_OPTIONS = DOTS_LANGUAGE_OPTIONS
    TEMPLATE_MODE_OPTIONS = [
        "TTS",
        "Instruction TTS",
    ]
    TEMPLATE_NAME_BY_MODE = {
        "TTS": "tts",
        "Instruction TTS": "instruction_tts",
    }

    @classmethod
    def NAME(cls):
        return "⚙️ Dots TTS Engine"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (cls.MODEL_VARIANTS, {
                    "default": "dots.tts-soar",
                    "tooltip": "Official Dots checkpoint.\n• dots.tts-soar: best zero-shot voice cloning\n• dots.tts-mf: fastest distilled checkpoint\n• dots.tts-base: pretrained baseline"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run Dots TTS on.\n• auto: use the best available device\n• cuda: NVIDIA GPU\n• cpu: CPU-only (very slow for 2B model)"
                }),
                "language": (cls.LANGUAGE_OPTIONS, {
                    "default": "Auto",
                    "tooltip": "Official Dots language tag.\n• Auto: let Dots detect language from text\n• None: disable explicit language tagging\n• Full language names: force the model-side language tag"
                }),
                "num_steps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Flow-matching sampling steps.\nOfficial recommendation: 10-32 for base/soar, 4 for mf."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Official CFG guidance scale.\nHigher values can increase energy and instability."
                }),
                "speaker_scale": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Official speaker embedding scale for reference audio conditioning."
                }),
                "max_generate_length": ("INT", {
                    "default": 500,
                    "min": 32,
                    "max": 1024,
                    "step": 32,
                    "tooltip": "Official maximum audio patch budget.\n500 is roughly 160 seconds of output budget."
                }),
            },
            "optional": {
                "template_mode": (cls.TEMPLATE_MODE_OPTIONS, {
                    "default": "TTS",
                    "tooltip": "Official non-standard Dots template mode from upstream.\n• TTS: standard Dots speech synthesis template\n• Instruction TTS: uses the same text field as normal TTS\nUpstream does not clearly document what behavior difference this mode is meant to produce.\nIt may yield different results, or little noticeable difference, versus standard TTS. Needs testing."
                }),
                "precision": (["auto", "bfloat16", "float16", "float32"], {
                    "default": "auto",
                    "tooltip": "Runtime precision.\n• auto: bfloat16 on newer CUDA GPUs, else float16, cpu -> float32"
                }),
                "normalize_text": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use Dots native text normalization before inference."
                }),
                "optimize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable official Dots runtime optimization / warmup.\nFirst load is slower; steady-state inference is faster."
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
        num_steps: int,
        guidance_scale: float,
        speaker_scale: float,
        max_generate_length: int,
        template_mode: str = "TTS",
        precision: str = "auto",
        normalize_text: bool = False,
        optimize: bool = False,
    ) -> tuple:
        template_name = self.TEMPLATE_NAME_BY_MODE.get(template_mode, "tts")
        config = {
            "engine_type": "dots_tts",
            "model_variant": model_variant,
            "device": device,
            "precision": precision,
            "language": language,
            "num_steps": int(num_steps),
            "guidance_scale": float(guidance_scale),
            "speaker_scale": float(speaker_scale),
            "max_generate_length": int(max_generate_length),
            "normalize_text": bool(normalize_text),
            "optimize": bool(optimize),
            "template_name": template_name,
        }

        print(f"⚙️ Dots TTS: Configured on {device}")
        print(f"   Model: {model_variant} | Language: {language} | Mode: {template_mode} | Precision: {precision}")
        print(
            f"   Settings: steps={num_steps}, guidance_scale={guidance_scale}, "
            f"speaker_scale={speaker_scale}, max_generate_length={max_generate_length}"
        )
        if normalize_text or optimize:
            print(f"   Advanced: normalize_text={normalize_text}, optimize={optimize}")

        engine_data = {
            "engine_type": "dots_tts",
            "config": config,
            "capabilities": ["tts"],
        }
        return (engine_data,)
