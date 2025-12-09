"""
Step Audio EditX Engine Configuration Node

Provides configuration interface for Step Audio EditX TTS engine with all parameters
for voice cloning and audio generation.
"""

import os
import sys
import importlib.util
from typing import Dict, Any, List

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
                    "tooltip": "Device to run Step Audio EditX model on:\n• auto: Automatically select best available (CUDA on NVIDIA, CPU fallback)\n• cuda: NVIDIA GPU (requires CUDA-capable GPU, ~8GB VRAM)\n• cpu: CPU-only processing (very slow, not recommended)"
                }),
                "torch_dtype": (["bfloat16", "float16", "float32", "auto"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision:\n• bfloat16: Best quality, stable, 8GB VRAM (recommended)\n• float16: Good quality, 6GB VRAM\n• float32: Maximum quality, 16GB VRAM\n• auto: Selects best for your GPU (bfloat16 if supported, else float16)"
                }),
                "quantization": (["none", "int4", "int8"], {
                    "default": "none",
                    "tooltip": "VRAM reduction via quantization:\n• none: Best quality, 8GB VRAM (recommended)\n• int8: Good quality, 4GB VRAM\n• int4: Acceptable quality, 3GB VRAM\nUse if low on VRAM. Quality degrades with stronger quantization."
                }),

                # Generation Parameters (used by clone mode, edit mode has hardcoded values)
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "Sampling temperature for generation.\n• Lower (0.3-0.5): More consistent, predictable output\n• Default (0.7): Balanced creativity and stability\n• Higher (1.0+): More varied, potentially unstable\nNote: Audio Editor uses hardcoded 0.7."
                }),
                "do_sample": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable sampling for generation.\n• True: Uses temperature for varied output (recommended)\n• False: Greedy decoding, deterministic but may be repetitive\nNote: Audio Editor uses hardcoded True."
                }),
                "max_new_tokens": ("INT", {
                    "default": 8192, "min": 256, "max": 16384, "step": 256,
                    "tooltip": "Maximum audio tokens to generate:\n• 2048: ~10s audio\n• 4096: ~20s audio\n• 8192: ~40s audio (default)\nHigher = more VRAM + longer generation time."
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
    ):
        """
        Create Step Audio EditX engine adapter with configuration.

        Returns:
            Tuple containing Step Audio EditX engine configuration data
        """
        try:
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
            }

            print(f"⚙️ Step Audio EditX: Configured on {device}")
            print(f"   Model: {model_path}")
            print(f"   Precision: {torch_dtype}")
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
                    "max_new_tokens": 8192,
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
