"""
CosyVoice3 Engine Configuration Node

Provides comprehensive configuration interface for CosyVoice3 TTS engine with
9-language zero-shot voice cloning, instruct mode for emotions/dialects, and
cross-lingual synthesis.
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

# AnyType for flexible input types (accepts any data type)
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


class CosyVoiceEngineNode(BaseTTSNode):
    """
    CosyVoice3 TTS Engine configuration node.
    Provides CosyVoice3 parameters and creates engine adapter for unified nodes.
    """
    
    @classmethod
    def NAME(cls):
        return "⚙️ CosyVoice3 Engine"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available model paths
        model_paths = cls._get_model_paths()
        
        return {
            "required": {
                # Model Configuration
                "model_path": (model_paths, {
                    "default": model_paths[0] if model_paths else "Fun-CosyVoice3-0.5B",
                    "tooltip": "CosyVoice3 model selection:\n• local:ModelName: Use locally installed model\n• ModelName: Auto-download model if not found locally"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run CosyVoice3 model on:\n• auto: Best available (CUDA > CPU)\n• cuda: NVIDIA GPU\n• cpu: CPU-only processing (slower)"
                }),
                
                # Generation Mode
                "mode": (["zero_shot", "instruct", "cross_lingual"], {
                    "default": "zero_shot",
                    "tooltip": "Generation mode:\n• zero_shot: Clone voice from reference (requires prompt_text)\n• instruct: Control emotions, dialects, speed via instructions\n• cross_lingual: Fine-grained control with [breath] tags"
                }),
                
                # Speed Control
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1,
                    "tooltip": "Speech speed multiplier (0.5=slow, 1.0=normal, 2.0=fast)"
                }),
                
                # Model Options
                "use_fp16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use FP16 for faster inference. Disable if you encounter numerical issues."
                }),
            },
            "optional": {
                # Instruct Mode Text
                "instruct_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": """Instructions for instruct mode. Examples:
• 请用广东话表达。 (Use Cantonese)
• 请用四川话说。 (Use Sichuan dialect)
• 请用尽可能快地语速说一句话。 (Speak as fast as possible)
• 请用温柔的语气说。 (Use gentle tone)
• 请用生气的语气说。 (Use angry tone)

Will be formatted as: You are a helpful assistant. [instruction]<|endofprompt|>"""
                }),
                
                # Reference Text (for zero_shot mode)
                "reference_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Transcript of the voice reference audio. REQUIRED for zero_shot mode.\nThis helps the model understand the voice characteristics better."
                }),
                
                # Advanced optimizations (usually not needed)
                "load_trt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Load TensorRT engine for faster inference (requires TensorRT setup)"
                }),
                "load_vllm": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Load vLLM engine for faster inference (requires vLLM setup)"
                }),
            }
        }
    
    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_adapter"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"
    
    @classmethod
    def _get_model_paths(cls) -> List[str]:
        """Get available CosyVoice3 model paths."""
        paths = ["Fun-CosyVoice3-0.5B"]  # Auto-download option

        try:
            # Check all configured TTS model paths
            all_tts_paths = get_all_tts_model_paths('TTS')

            for base_path in all_tts_paths:
                # Check direct path (models/TTS/Fun-CosyVoice3-0.5B)
                cosy_direct = os.path.join(base_path, "Fun-CosyVoice3-0.5B")
                if os.path.exists(os.path.join(cosy_direct, "cosyvoice3.yaml")):
                    local_model = "local:Fun-CosyVoice3-0.5B"
                    if local_model not in paths:
                        paths.insert(0, local_model)

                # Check organized path (models/TTS/CosyVoice/Fun-CosyVoice3-0.5B)
                cosy_organized = os.path.join(base_path, "CosyVoice")
                if os.path.exists(cosy_organized):
                    for item in os.listdir(cosy_organized):
                        model_dir = os.path.join(cosy_organized, item)
                        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "cosyvoice3.yaml")):
                            local_model = f"local:{item}"
                            if local_model not in paths:
                                paths.insert(-1, local_model)
        except Exception:
            # Fallback to default
            base_dir = os.path.join(folder_paths.models_dir, "TTS", "CosyVoice")
            if os.path.exists(base_dir):
                for item in os.listdir(base_dir):
                    model_dir = os.path.join(base_dir, item)
                    if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "cosyvoice3.yaml")):
                        local_model = f"local:{item}"
                        if local_model not in paths:
                            paths.insert(-1, local_model)

        return paths
    
    def create_engine_adapter(
        self,
        model_path: str,
        device: str,
        mode: str,
        speed: float,
        use_fp16: bool,
        instruct_text: str = "",
        reference_text: str = "",
        load_trt: bool = False,
        load_vllm: bool = False,
    ):
        """
        Create CosyVoice3 engine adapter with configuration.
        
        Returns:
            Tuple containing CosyVoice3 engine configuration data
        """
        try:
            # Validate mode and reference_text
            if mode == "zero_shot" and not reference_text.strip():
                print("⚠️ CosyVoice3: reference_text is REQUIRED for zero_shot mode")
                print("   → Provide a transcript of your voice reference audio")
                print("   → Or switch to 'instruct' mode which doesn't require transcript")
            
            # Create configuration dictionary
            config = {
                "model_path": model_path,
                "device": device,
                "mode": mode,
                "speed": speed,
                "use_fp16": use_fp16,
                "instruct_text": instruct_text.strip() if instruct_text else None,
                "reference_text": reference_text.strip() if reference_text else None,
                "load_trt": load_trt,
                "load_vllm": load_vllm,
                "engine_type": "cosyvoice",
            }
            
            print(f"⚙️ CosyVoice3: Configured on {device}")
            print(f"   Model: {model_path}")
            print(f"   Mode: {mode}")
            print(f"   Speed: {speed}x")
            if mode == "instruct" and instruct_text:
                print(f"   Instruction: {instruct_text[:50]}...")
            elif mode == "zero_shot" and reference_text:
                print(f"   Reference text: {reference_text[:50]}...")
            
            # Return engine data for consumption by unified TTS nodes
            engine_data = {
                "engine_type": "cosyvoice",
                "config": config,
                "adapter_class": "CosyVoiceAdapter"
            }
            
            return (engine_data,)
            
        except Exception as e:
            print(f"❌ CosyVoice3 Engine error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return error config
            error_config = {
                "engine_type": "cosyvoice",
                "config": {
                    "model_path": model_path,
                    "device": "cpu",  # Fallback to CPU
                    "mode": "zero_shot",
                    "speed": 1.0,
                    "error": str(e)
                },
                "adapter_class": "CosyVoiceAdapter"
            }
            return (error_config,)


# Register the node
NODE_CLASS_MAPPINGS = {
    "CosyVoice Engine": CosyVoiceEngineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CosyVoice Engine": "⚙️ CosyVoice3 Engine"
}
