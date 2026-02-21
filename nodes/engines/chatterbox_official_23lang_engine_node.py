"""
ChatterBox Official 23-Lang Engine Node - Official multilingual ChatterBox configuration
Provides ChatterBox Official 23-Lang engine adapter with multilingual parameters
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


class ChatterBoxOfficial23LangEngineNode(BaseTTSNode):
    """
    ChatterBox Official 23-Lang TTS Engine configuration node.
    Provides multilingual ChatterBox parameters and creates engine adapter for unified nodes.
    """
    
    @classmethod
    def NAME(cls):
        return "⚙️ ChatterBox Official 23-Lang Engine"
    
    @classmethod
    def _get_available_chatterbox_23lang_models(cls) -> list:
        """Get available ChatterBox Official 23-Lang models without importing heavy modules.
        Reconstructs the discovery logic using file reading only."""
        # Static model definitions (not heavy)
        OFFICIAL_23LANG_MODELS = {
            "ChatterBox Official 23-Lang": {
                "repo": "ResembleAI/chatterbox",
                "format": "mixed",
            },
            "Vietnamese (Viterbox)": {
                "repo": "dolly-vn/viterbox",
                "format": "mixed",
            },
            "Egyptian Arabic (oddadmix)": {
                "repo": "oddadmix/chatterbox-egyptian-v0",
                "format": "as-is",
            }
        }

        models = list(OFFICIAL_23LANG_MODELS.keys())
        found_local_models = set()

        try:
            import folder_paths

            # Check hardcoded paths (like F5-TTS and ChatterBox pattern)
            search_paths = [
                os.path.join(folder_paths.models_dir, "TTS", "chatterbox_official_23lang"),
                os.path.join(folder_paths.models_dir, "chatterbox_official_23lang")
            ]

            for models_dir in search_paths:
                if not os.path.exists(models_dir):
                    continue

                try:
                    for item in os.listdir(models_dir):
                        item_path = os.path.join(models_dir, item)
                        if not os.path.isdir(item_path):
                            continue

                        # Check if it contains official 23-lang model files with flexible detection
                        # Support different T3 filenames and tokenizers for community finetunes
                        has_model = False
                        item_files = os.listdir(item_path)

                        # Check for essential components
                        has_s3gen = "s3gen.pt" in item_files
                        has_ve = "ve.pt" in item_files
                        has_tokenizer = any(f in item_files for f in ["mtl_tokenizer.json", "tokenizer_vi_expanded.json"])

                        # Check for T3 model file with flexible pattern matching
                        # Supports: t3_23lang.safetensors, t3_mtl23ls_v2.safetensors, t3_ml24ls_v2.safetensors, etc.
                        has_t3 = any(f.startswith("t3_") and f.endswith(".safetensors") for f in item_files)

                        has_model = has_s3gen and has_ve and has_tokenizer and has_t3

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
            if local_model not in models:
                models.insert(0, local_model)

        return models if models else ["ChatterBox Official 23-Lang"]

    @classmethod
    def INPUT_TYPES(cls):
        available_models = cls._get_available_chatterbox_23lang_models()

        # Static language names (not heavy)
        SUPPORTED_LANGUAGES = {
            "ar": "Arabic",
            "da": "Danish",
            "de": "German",
            "el": "Greek",
            "en": "English",
            "es": "Spanish",
            "fi": "Finnish",
            "fr": "French",
            "he": "Hebrew",
            "hi": "Hindi",
            "it": "Italian",
            "ja": "Japanese",
            "ko": "Korean",
            "ms": "Malay",
            "nl": "Dutch",
            "no": "Norwegian",
            "pl": "Polish",
            "pt": "Portuguese",
            "ru": "Russian",
            "sv": "Swedish",
            "sw": "Swahili",
            "tr": "Turkish",
            "zh": "Chinese",
            "vi": "Vietnamese (Viterbox only)",
        }
        available_languages = list(SUPPORTED_LANGUAGES.values())

        return {
            "required": {
                "model_version": (["v1", "v2", "Vietnamese (Viterbox)", "Egyptian Arabic (oddadmix)"], {
                    "default": "v2",
                    "tooltip": "ChatterBox model version:\n• v1: Original 23-language model\n• v2: Enhanced with special tokens for emotions ([giggle], [laughter], [sigh]), sounds ([cough], [sneeze]), vocal styles ([singing], [whisper]), and improved Russian support\n• Vietnamese (Viterbox): Community finetune optimized for Vietnamese (3000+ hours training data), supports all 24 languages with Vietnamese language support\n• Egyptian Arabic (oddadmix): Community finetune optimized for Egyptian Arabic (requires 'Arabic' language selection)"
                }),
                "language": (available_languages, {
                    "default": "English",
                    "tooltip": "ChatterBox language model to use for text-to-speech generation. Local models are preferred over remote downloads."
                }),
                "device": (["auto", "cuda", "xpu", "cpu", "mps"], {
                    "default": "auto",
                    "tooltip": "Device to run ChatterBox model on:\n• auto: Automatically select best available (MPS on Apple Silicon, CUDA on NVIDIA, XPU on Intel, CPU fallback)\n• cuda: NVIDIA GPU (requires CUDA-capable GPU)\n• xpu: Intel GPU (requires Intel PyTorch XPU)\n• cpu: CPU-only processing (slower)\n• mps: Apple Metal Performance Shaders (Apple Silicon Macs only)"
                }),
                "exaggeration": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Emotion exaggeration control. WARNING: This parameter has minimal effect in the multilingual models (v1 and v2) due to model training issues. Values are internally scaled by 50000x. Even at extreme values (100000+), changes are barely noticeable. This appears to be a fundamental model limitation, not an implementation issue. Classic ChatterBox works as expected."
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
                "repetition_penalty": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Penalty for repeated tokens. Higher values reduce repetition in generated speech."
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum probability threshold for token selection. Lower values allow more diverse tokens."
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Nucleus sampling threshold. Controls the probability mass of tokens to consider."
                }),
            }
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_adapter"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine_adapter(self, model_version: str, language: str, device: str, exaggeration: float,
                            temperature: float, cfg_weight: float, repetition_penalty: float,
                            min_p: float, top_p: float):
        """
        Create ChatterBox Official 23-Lang engine adapter with configuration.

        Args:
            model_version: Model version (v1 or v2)
            language: Language for multilingual generation
            device: Device to run model on
            exaggeration: Speech exaggeration level
            temperature: Generation randomness
            cfg_weight: Classifier-Free Guidance weight
            repetition_penalty: Penalty for repeated tokens
            min_p: Minimum probability threshold
            top_p: Nucleus sampling threshold

        Returns:
            Tuple containing ChatterBox Official 23-Lang engine adapter
        """
        try:
            # Egyptian (oddadmix) model is an Arabic finetune, so it should be used with the Arabic language option
            if model_version == "Egyptian Arabic (oddadmix)" and language != "Arabic":
                raise ValueError("The 'Egyptian Arabic (oddadmix)' model is an Arabic finetune. Please set the language to 'Arabic'.")

            # Validate Vietnamese language is only used with Viterbox model
            if "Vietnamese" in language and model_version not in ["Vietnamese (Viterbox)"]:
                raise ValueError(
                    f"Vietnamese language is only supported with 'Vietnamese (Viterbox)' model version.\n"
                    f"Current model_version: '{model_version}'\n\n"
                    f"Please either:\n"
                    f"  • Change model_version to 'Vietnamese (Viterbox)', OR\n"
                    f"  • Select a different language from the 23 languages supported by v1/v2"
                )

            # Import the adapter class
            from engines.adapters.chatterbox_official_23lang_adapter import ChatterBoxOfficial23LangEngineAdapter

            # Scale exaggeration by 50000x for multilingual models due to training issues
            # Multilingual models have extremely small emotion_adv_fc weights requiring massive values
            scaled_exaggeration = exaggeration * 50000.0

            # Create configuration dictionary
            config = {
                "model_version": model_version,
                "language": language,
                "device": device,
                "exaggeration": scaled_exaggeration,
                "temperature": temperature,
                "cfg_weight": cfg_weight,
                "repetition_penalty": repetition_penalty,
                "min_p": min_p,
                "top_p": top_p,
                "engine_type": "chatterbox_official_23lang"
            }
            
            print(f"⚙️ ChatterBox Official 23-Lang {model_version}: Configured for {language} on {device}")
            print(f"   Settings: exaggeration={exaggeration}, temperature={temperature}, cfg_weight={cfg_weight}")
            print(f"   Advanced: repetition_penalty={repetition_penalty}, min_p={min_p}, top_p={top_p}")
            
            # For now, return the config dict. The actual adapter creation will happen 
            # in the consumer nodes when they have access to the node instance
            engine_data = {
                "engine_type": "chatterbox_official_23lang", 
                "config": config,
                "adapter_class": "ChatterBoxOfficial23LangEngineAdapter"
            }
            
            return (engine_data,)
            
        except Exception as e:
            print(f"❌ ChatterBox Engine error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a default config that indicates error state
            error_config = {
                "engine_type": "chatterbox_official_23lang",
                "config": {
                    "language": language,
                    "device": "cpu",  # Fallback to CPU
                    "exaggeration": 0.5,
                    "temperature": 0.8,
                    "cfg_weight": 0.5,
                    "repetition_penalty": 2.0,
                    "min_p": 0.05,
                    "top_p": 1.0,
                    "error": str(e)
                },
                "adapter_class": "ChatterBoxOfficial23LangEngineAdapter"
            }
            return (error_config,)


# Register the node class
NODE_CLASS_MAPPINGS = {
    "ChatterBoxOfficial23LangEngineNode": ChatterBoxOfficial23LangEngineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterBoxOfficial23LangEngineNode": "⚙️ ChatterBox Official 23-Lang Engine"
}