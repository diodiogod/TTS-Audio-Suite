"""
ChatterBox TTS Engine Node - ChatterBox-specific configuration for TTS Audio Suite
Provides ChatterBox engine adapter with all ChatterBox-specific parameters
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


class ChatterBoxEngineNode(BaseTTSNode):
    """
    ChatterBox TTS Engine configuration node.
    Provides ChatterBox-specific parameters and creates engine adapter for unified nodes.
    """
    
    @classmethod
    def NAME(cls):
        return "⚙️ ChatterBox TTS Engine"
    
    @classmethod
    def _get_available_chatterbox_languages(cls) -> list:
        """Get available ChatterBox languages without importing heavy modules.
        Reconstructs the discovery logic using file reading only."""
        # Static language definitions (not heavy)
        CHATTERBOX_MODELS = {
            "English": {"repo": "ResembleAI/chatterbox", "format": "pt"},
            "German": {"repo": "stlohrey/chatterbox_de", "format": "safetensors"},
            "German (SebastianBodza)": {"repo": "SebastianBodza/Kartoffelbox-v0.1", "format": "mixed"},
            "German (havok2)": {"repo": "havok2/Kartoffelbox-v0.1_0.65h2", "format": "safetensors"},
            "Norwegian": {"repo": "akhbar/chatterbox-tts-norwegian", "format": "safetensors"},
            "French": {"repo": "Thomcles/Chatterbox-TTS-French", "format": "safetensors"},
            "Russian": {"repo": "niobures/Chatterbox-TTS", "format": "safetensors"},
            "Armenian": {"repo": "niobures/Chatterbox-TTS", "format": "safetensors"},
            "Georgian": {"repo": "niobures/Chatterbox-TTS", "format": "safetensors"},
            "Japanese": {"repo": "niobures/Chatterbox-TTS", "format": "safetensors"},
            "Korean": {"repo": "niobures/Chatterbox-TTS", "format": "safetensors"},
            "Italian": {"repo": "niobures/Chatterbox-TTS", "format": "pt"},
        }

        languages = list(CHATTERBOX_MODELS.keys())
        found_local_models = set()

        try:
            import folder_paths

            # Check hardcoded paths (like F5-TTS pattern)
            search_paths = [
                os.path.join(folder_paths.models_dir, "TTS", "chatterbox"),
                os.path.join(folder_paths.models_dir, "chatterbox")
            ]

            for models_dir in search_paths:
                if not os.path.exists(models_dir):
                    continue

                try:
                    for item in os.listdir(models_dir):
                        item_path = os.path.join(models_dir, item)
                        if not os.path.isdir(item_path):
                            continue

                        # Check if it contains ChatterBox model files
                        # Must have files starting with: ve., t3_cfg., s3gen., tokenizer.json
                        required_file_starts = ["ve.", "t3_cfg.", "s3gen.", "tokenizer.json"]
                        has_model = False

                        try:
                            for file in os.listdir(item_path):
                                for required_start in required_file_starts:
                                    if file.startswith(required_start) and file.endswith((".pt", ".safetensors")):
                                        has_model = True
                                        break
                                if has_model:
                                    break
                        except OSError:
                            continue

                        if has_model:
                            local_model = f"local:{item}"
                            if local_model not in found_local_models:
                                found_local_models.add(local_model)

                except OSError:
                    continue
        except Exception:
            pass

        # Add found local models to the beginning
        for local_model in sorted(found_local_models):
            if local_model not in languages:
                languages.insert(0, local_model)

        return languages if languages else ["English"]

    @classmethod
    def INPUT_TYPES(cls):
        available_languages = cls._get_available_chatterbox_languages()
        
        return {
            "required": {
                "language": (available_languages, {
                    "default": "English",
                    "tooltip": "ChatterBox language model to use for text-to-speech generation. Local models are preferred over remote downloads."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run ChatterBox model on. 'auto' selects best available (GPU if available, otherwise CPU)."
                }),
                "exaggeration": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.25, 
                    "max": 2.0, 
                    "step": 0.05,
                    "tooltip": "Speech exaggeration level for ChatterBox. Higher values create more dramatic and expressive speech."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.05, 
                    "max": 5.0, 
                    "step": 0.05,
                    "tooltip": "Controls randomness in ChatterBox generation. Higher values = more creative/varied speech, lower values = more consistent speech."
                }),
                "cfg_weight": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "Classifier-Free Guidance weight for ChatterBox. Controls how strongly the model follows the text prompt."
                }),
                "crash_protection_template": ("STRING", {
                    "default": "hmm ,, {seg} hmm ,,",
                    "tooltip": "Custom padding template for short text segments to prevent ChatterBox crashes. ChatterBox has a bug where text shorter than ~21 characters causes CUDA tensor errors. Use {seg} as placeholder for the original text. Examples: '...ummmmm {seg}' (default hesitation), '{seg}... yes... {seg}' (repetition), 'Well, {seg}' (natural prefix), or empty string to disable padding."
                }),
            }
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_adapter"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine_adapter(self, language: str, device: str, exaggeration: float, 
                            temperature: float, cfg_weight: float, crash_protection_template: str):
        """
        Create ChatterBox engine adapter with configuration.
        
        Args:
            language: ChatterBox language model
            device: Device to run model on
            exaggeration: Speech exaggeration level
            temperature: Generation randomness
            cfg_weight: Classifier-Free Guidance weight
            crash_protection_template: Template for padding short text segments
            
        Returns:
            Tuple containing ChatterBox engine adapter
        """
        try:
            # Import the adapter class
            from engines.adapters.chatterbox_adapter import ChatterBoxEngineAdapter
            
            # Create configuration dictionary
            config = {
                "language": language,
                "device": device,
                "exaggeration": exaggeration,
                "temperature": temperature,
                "cfg_weight": cfg_weight,
                "crash_protection_template": crash_protection_template,
                "engine_type": "chatterbox"
            }
            
            print(f"⚙️ ChatterBox Engine: Configured for {language} on {device}")
            print(f"   Settings: exaggeration={exaggeration}, temperature={temperature}, cfg_weight={cfg_weight}")
            print(f"   Crash protection: {crash_protection_template or 'disabled'}")
            
            # For now, return the config dict. The actual adapter creation will happen 
            # in the consumer nodes when they have access to the node instance
            engine_data = {
                "engine_type": "chatterbox", 
                "config": config,
                "adapter_class": "ChatterBoxEngineAdapter"
            }
            
            return (engine_data,)
            
        except Exception as e:
            print(f"❌ ChatterBox Engine error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a default config that indicates error state
            error_config = {
                "engine_type": "chatterbox",
                "config": {
                    "language": language,
                    "device": "cpu",  # Fallback to CPU
                    "exaggeration": 0.5,
                    "temperature": 0.8,
                    "cfg_weight": 0.5,
                    "crash_protection_template": crash_protection_template,
                    "error": str(e)
                },
                "adapter_class": "ChatterBoxEngineAdapter"
            }
            return (error_config,)


# Register the node class
NODE_CLASS_MAPPINGS = {
    "ChatterBoxEngineNode": ChatterBoxEngineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterBoxEngineNode": "⚙️ ChatterBox TTS Engine"
}