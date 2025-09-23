"""
F5 TTS Engine Node - F5-TTS-specific configuration for TTS Audio Suite
Provides F5-TTS engine adapter with all F5-TTS-specific parameters
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

# Load f5tts_base_node module directly for model access
f5tts_base_node_path = os.path.join(nodes_dir, "base", "f5tts_base_node.py")
f5tts_base_spec = importlib.util.spec_from_file_location("f5tts_base_node_module", f5tts_base_node_path)
f5tts_base_module = importlib.util.module_from_spec(f5tts_base_spec)
sys.modules["f5tts_base_node_module"] = f5tts_base_module
f5tts_base_spec.loader.exec_module(f5tts_base_module)

# Import the base class
BaseF5TTSNode = f5tts_base_module.BaseF5TTSNode


class F5TTSEngineNode(BaseF5TTSNode):
    """
    F5-TTS Engine configuration node.
    Provides F5-TTS-specific parameters and creates engine adapter for unified nodes.
    """
    
    @classmethod
    def NAME(cls):
        return "⚙️ F5 TTS Engine"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "language": (cls.get_available_models_for_dropdown(), {
                    "default": "F5TTS_v1_Base",
                    "tooltip": "F5-TTS model variant to use. F5TTS_Base is the standard model, F5TTS_v1_Base is improved version, E2TTS_Base is enhanced variant. Note: This was previously called 'model' in individual nodes."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run F5-TTS model on. 'auto' selects best available (GPU if available, otherwise CPU)."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "Controls randomness in F5-TTS generation. Higher values = more creative/varied speech, lower values = more consistent/predictable speech."
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1,
                    "tooltip": "F5-TTS native speech speed control. 1.0 = normal speed, 0.5 = half speed (slower), 2.0 = double speed (faster)."
                }),
                "target_rms": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Target audio volume level (Root Mean Square). Controls output loudness normalization. Higher values = louder audio output."
                }),
                "cross_fade_duration": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Duration in seconds for smooth audio transitions between F5-TTS segments. Prevents audio clicks/pops by blending segment boundaries."
                }),
                "nfe_step": ("INT", {
                    "default": 32, "min": 1, "max": 71,
                    "tooltip": "Neural Function Evaluation steps for F5-TTS inference. Higher values = better quality but slower generation. 32 is a good balance. Values above 71 may cause ODE solver issues."
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Speech generation control. Lower values (1.0-1.5) = more natural, conversational delivery. Higher values (3.0-5.0) = crisper, more articulated speech with stronger emphasis. Default 2.0 balances naturalness and clarity."
                }),
            }
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_adapter"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine_adapter(self, language: str, device: str, temperature: float,
                            speed: float, target_rms: float, cross_fade_duration: float,
                            nfe_step: int, cfg_strength: float):
        """
        Create F5-TTS engine adapter with configuration.
        
        Args:
            language: F5-TTS model name (was previously called 'model')
            device: Device to run model on
            temperature: Generation randomness
            speed: Speech speed control
            target_rms: Target audio volume level
            cross_fade_duration: Crossfade duration for segments
            nfe_step: Neural Function Evaluation steps
            cfg_strength: Classifier-Free Guidance strength
            auto_phonemization: Enable automatic phonemization for multilingual text
            
        Returns:
            Tuple containing F5-TTS engine adapter
        """
        try:
            # Import the adapter class
            from engines.adapters.f5tts_adapter import F5TTSEngineAdapter
            
            # Normalize model name for backward compatibility (case-insensitive matching)
            # Convert V1, V2, etc. to v1, v2 for consistency
            import re
            language = re.sub(r'_V(\d+)_', r'_v\1_', language)
            
            # Validate and clamp nfe_step to prevent ODE solver issues
            safe_nfe_step = max(1, min(nfe_step, 71))
            if safe_nfe_step != nfe_step:
                print(f"⚠️ F5-TTS Engine: Clamped nfe_step from {nfe_step} to {safe_nfe_step} to prevent ODE solver issues")
            
            # Create configuration dictionary
            config = {
                "model": language,  # Keep as 'model' for backward compatibility with existing adapter
                "language": language,  # Also provide as 'language' for consistency
                "device": device,
                "temperature": temperature,
                "speed": speed,
                "target_rms": target_rms,
                "cross_fade_duration": cross_fade_duration,
                "nfe_step": safe_nfe_step,
                "cfg_strength": cfg_strength,
                "auto_phonemization": False,  # Disabled - use 📝 Phoneme Text Normalizer node instead
                "engine_type": "f5tts"
            }
            
            print(f"⚙️ F5-TTS Engine: Configured for {language} on {device}")
            print(f"   Settings: temperature={temperature}, speed={speed}, nfe_step={safe_nfe_step}, auto_phonemization=False")
            
            # Return engine data structure
            engine_data = {
                "engine_type": "f5tts", 
                "config": config,
                "adapter_class": "F5TTSEngineAdapter"
            }
            
            return (engine_data,)
            
        except Exception as e:
            print(f"❌ F5-TTS Engine error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a default config that indicates error state
            error_config = {
                "engine_type": "f5tts",
                "config": {
                    "model": language,
                    "language": language,
                    "device": "cpu",  # Fallback to CPU
                    "temperature": 0.8,
                    "speed": 1.0,
                    "target_rms": 0.1,
                    "cross_fade_duration": 0.15,
                    "nfe_step": 32,
                    "cfg_strength": 2.0,
                    "error": str(e)
                },
                "adapter_class": "F5TTSEngineAdapter"
            }
            return (error_config,)


# Register the node class
NODE_CLASS_MAPPINGS = {
    "F5TTSEngineNode": F5TTSEngineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "F5TTSEngineNode": "⚙️ F5 TTS Engine"
}