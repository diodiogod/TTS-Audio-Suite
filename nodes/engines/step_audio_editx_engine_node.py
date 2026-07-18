"""
Step Audio EditX Engine Configuration Node

Provides configuration interface for Step Audio EditX TTS engine with all parameters
for voice cloning and audio generation.
"""

import os
import sys
import importlib.util
from typing import Dict, Any, List

from utils.models.factory_config import (
    RUNTIME_MODE_DEDICATED,
    RUNTIME_MODE_MAIN,
    RUNTIME_MODE_SHARED,
    normalize_runtime_mode,
)

RUNTIME_MODE_MAIN_LABEL = "Main Environment"
RUNTIME_MODE_SHARED_LABEL = "⚠️ Shared Runtime"
RUNTIME_MODE_DEDICATED_LABEL = "⚠️ Dedicated Runtime"

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

import folder_paths
from utils.models.extra_paths import get_all_tts_model_paths


class StepAudioEditXEngineNode(BaseTTSNode):
    """
    Step Audio EditX TTS Engine configuration node.
    Provides configuration for zero-shot voice cloning with emotion/style/speed editing support.
    """

    @classmethod
    def NAME(cls):
        return "⚙️ Step Audio EditX Engine"

    @classmethod
    def INPUT_TYPES(cls):
        # Get available model paths
        model_paths = cls._get_model_paths()

        return {
            "required": {
                # Model Configuration
                "model_path": (model_paths, {
                    "default": model_paths[0] if model_paths else "Step-Audio-EditX",
                    "tooltip": "Step Audio EditX model selection:\n• local:ModelName: Use locally installed model (respects extra_model_paths.yaml)\n• ModelName: Auto-download model if not found locally\n• Downloads respect extra_model_paths.yaml configuration"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run Step Audio EditX on:\n• auto: Select the best available device\n• cuda: NVIDIA GPU; this is a heavy model and 12GB+ VRAM is recommended\n• cpu: Extremely slow and intended only as a fallback"
                }),
                "torch_dtype": (["bfloat16", "float16", "float32", "auto"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision:\n• bfloat16: Recommended on supported NVIDIA GPUs\n• float16: Compatibility alternative with similar memory use\n• float32: Much higher memory use; generally not recommended\n• auto: Selects bfloat16 when supported, otherwise float16"
                }),
                "quantization": (["none", "int4", "int8"], {
                    "default": "none",
                    "tooltip": "LLM weight quantization:\n• none: Best quality and compatibility\n• int8: Reduces LLM weight memory\n• int4: Stronger reduction with a larger quality/compatibility tradeoff\nThe audio tokenizer and vocoder are not quantized, so total VRAM does not shrink proportionally."
                }),
                # Generation parameters shared by clone and edit modes.
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "Sampling temperature for generation.\n• Lower (0.3-0.5): More consistent, predictable output\n• Default (0.7): Balanced creativity and stability\n• Higher (1.0+): More varied, potentially unstable"
                }),
                "do_sample": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable sampling for generation.\n• True: Uses temperature for varied output (recommended)\n• False: Greedy decoding can fail to emit an end token and is not recommended"
                }),
                "max_new_tokens": ("INT", {
                    "default": 1024, "min": 128, "max": 8192, "step": 128,
                    "tooltip": "Maximum generated audio tokens (safety ceiling):\n• 512: roughly 12s\n• 1024: roughly 25s (recommended)\n• Higher values allow longer output but increase runtime and KV-cache memory."
                }),
                # Keep new widgets last so existing serialized workflows retain
                # the positional values of every older Step EditX setting.
                "runtime_mode": ([
                    RUNTIME_MODE_SHARED_LABEL,
                    RUNTIME_MODE_DEDICATED_LABEL,
                    RUNTIME_MODE_MAIN_LABEL,
                ], {
                    "default": RUNTIME_MODE_SHARED_LABEL,
                    "tooltip": "Python runtime used by Step Audio EditX:\n"
                    "• Shared Runtime: Recommended. Uses the shared Transformers 4 runtime and was verified with Step EditX.\n"
                    "• Dedicated Runtime: Uses a separate Step EditX Transformers 4 environment. Choose this only to isolate dependency conflicts.\n"
                    "• Main Environment: Uses ComfyUI's packages. Transformers 5 currently produces invalid Step audio tokens and is not recommended."
                }),
            }
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_adapter"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    @classmethod
    def _is_valid_step_audio_model(cls, model_dir: str) -> bool:
        """
        Check if directory is a valid Step Audio EditX model.
        Valid models must contain the CosyVoice vocoder (required component).
        This filters out internal components like FunASR-Paraformer.
        """
        # A valid Step Audio EditX model has CosyVoice vocoder
        cosyvoice_path = os.path.join(model_dir, "CosyVoice-300M-25Hz")
        return os.path.isdir(cosyvoice_path)

    @classmethod
    def _get_model_paths(cls) -> List[str]:
        """Get available Step Audio EditX model paths."""
        paths = ["Step-Audio-EditX"]  # Auto-download option

        try:
            # Check all configured TTS model paths
            all_tts_paths = get_all_tts_model_paths('TTS')

            for base_path in all_tts_paths:
                # Check direct path (models/TTS/Step-Audio-EditX)
                direct_path = os.path.join(base_path, "Step-Audio-EditX")
                if cls._is_valid_step_audio_model(direct_path):
                    local_model = "local:Step-Audio-EditX"
                    if local_model not in paths:
                        paths.insert(0, local_model)  # Insert at beginning

                # Check organized path (models/TTS/step_audio_editx/Step-Audio-EditX)
                organized_base = os.path.join(base_path, "step_audio_editx")
                if os.path.exists(organized_base):
                    for item in os.listdir(organized_base):
                        model_dir = os.path.join(organized_base, item)
                        # Only include valid Step Audio EditX models (skip FunASR, etc.)
                        if os.path.isdir(model_dir) and cls._is_valid_step_audio_model(model_dir):
                            local_model = f"local:{item}"
                            if local_model not in paths:
                                paths.insert(-1, local_model)  # Insert before auto-download

        except Exception:
            # Fallback to original behavior if extra_paths fails
            base_dir = os.path.join(folder_paths.models_dir, "TTS", "step_audio_editx")
            if os.path.exists(base_dir):
                for item in os.listdir(base_dir):
                    model_dir = os.path.join(base_dir, item)
                    # Only include valid Step Audio EditX models (skip FunASR, etc.)
                    if os.path.isdir(model_dir) and cls._is_valid_step_audio_model(model_dir):
                        local_model = f"local:{item}"
                        if local_model not in paths:
                            paths.insert(-1, local_model)

        return paths

    def create_engine_adapter(
        self,
        model_path: str,
        device: str,
        torch_dtype: str,
        quantization: str,
        temperature: float,
        do_sample: bool,
        max_new_tokens: int,
        runtime_mode: str,
    ):
        """
        Create Step Audio EditX engine adapter with configuration.

        Returns:
            Tuple containing Step Audio EditX engine configuration data
        """
        try:
            normalized_runtime_mode = normalize_runtime_mode(runtime_mode)
            runtime_profiles = {
                RUNTIME_MODE_SHARED: "vibevoice_transformers4_shared",
                RUNTIME_MODE_DEDICATED: "step_audio_editx_transformers4",
                RUNTIME_MODE_MAIN: None,
            }

            # Create configuration dictionary
            config = {
                "model_path": model_path,
                "device": device,
                "torch_dtype": torch_dtype,
                "quantization": quantization if quantization != "none" else None,
                "temperature": temperature,
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens,
                "engine_type": "step_audio_editx",
                "runtime_mode": normalized_runtime_mode,
                "runtime_profile": runtime_profiles[normalized_runtime_mode],
            }

            print(f"⚙️ Step Audio EditX: Configured on {device}")
            print(f"   Model: {model_path}")
            print(f"   Precision: {torch_dtype}")
            print(f"   Runtime: {runtime_mode}")
            if quantization != "none":
                print(f"   Quantization: {quantization}")
            print(f"   Generation: temp={temperature}, do_sample={do_sample}, max_tokens={max_new_tokens}")

            # Return engine data for consumption by unified TTS nodes
            engine_data = {
                "engine_type": "step_audio_editx",
                "config": config,
                "adapter_class": "StepAudioEditXAdapter"
            }

            return (engine_data,)

        except Exception as e:
            print(f"❌ Step Audio EditX Engine error: {e}")
            import traceback
            traceback.print_exc()

            # Return error config
            error_config = {
                "engine_type": "step_audio_editx",
                "config": {
                    "model_path": model_path,
                    "device": "cpu",  # Fallback to CPU
                    "torch_dtype": "float32",
                    "quantization": None,
                    "temperature": 0.7,
                    "do_sample": True,
                    "max_new_tokens": 1024,
                    "runtime_mode": RUNTIME_MODE_SHARED,
                    "runtime_profile": "vibevoice_transformers4_shared",
                    "error": str(e)
                },
                "adapter_class": "StepAudioEditXAdapter"
            }
            return (error_config,)


# Register the node
NODE_CLASS_MAPPINGS = {
    "Step Audio EditX Engine": StepAudioEditXEngineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Step Audio EditX Engine": "⚙️ Step Audio EditX Engine"
}
