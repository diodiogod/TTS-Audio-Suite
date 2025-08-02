"""
ChatterBox Language Model Registry
Manages multilanguage ChatterBox models following F5-TTS pattern
"""

import os
import folder_paths
from typing import Dict, List, Tuple, Optional

# ChatterBox model configurations
CHATTERBOX_MODELS = {
    "English": {
        "repo": "ResembleAI/chatterbox", 
        "format": "pt",
        "description": "Original English ChatterBox model"
    },
    "German": {
        "repo": "stlohrey/chatterbox_de", 
        "format": "safetensors",
        "description": "German ChatterBox model with high quality"
    },
    "Norwegian": {
        "repo": "akhbar/chatterbox-tts-norwegian", 
        "format": "safetensors",
        "description": "Norwegian ChatterBox model (Bokmål and Nynorsk dialects) - 532M parameters"
    },
}

def get_chatterbox_models() -> List[str]:
    """
    Get list of available ChatterBox language models.
    Checks local models first, then includes predefined models.
    """
    models = list(CHATTERBOX_MODELS.keys())
    
    # Check for local models in ComfyUI models directory
    try:
        models_dir = os.path.join(folder_paths.models_dir, "chatterbox")
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path):
                    # Check if it contains ChatterBox model files
                    required_files = ["ve.", "t3_cfg.", "s3gen.", "tokenizer.json"]
                    has_model = False
                    
                    for file in os.listdir(item_path):
                        for required in required_files:
                            if file.startswith(required) and (file.endswith(".pt") or file.endswith(".safetensors")):
                                has_model = True
                                break
                        if has_model:
                            break
                    
                    if has_model:
                        local_model = f"local:{item}"
                        if local_model not in models:
                            models.append(local_model)
    except Exception:
        pass  # Ignore errors in model discovery
    
    return models

def get_model_config(language: str) -> Optional[Dict]:
    """Get configuration for a specific language model"""
    if language.startswith("local:"):
        # Local model
        local_name = language[6:]  # Remove "local:" prefix
        return {
            "repo": None,
            "format": "auto",  # Auto-detect format
            "local_path": os.path.join(folder_paths.models_dir, "chatterbox", local_name),
            "description": f"Local ChatterBox model: {local_name}"
        }
    
    return CHATTERBOX_MODELS.get(language)

def get_model_files_for_language(language: str) -> Tuple[str, str]:
    """
    Get the expected file format and repo for a language.
    Returns (format, repo_id) tuple.
    """
    config = get_model_config(language)
    if not config:
        # Default to English if language not found
        config = CHATTERBOX_MODELS["English"]
    
    return config.get("format", "pt"), config.get("repo")

def find_local_model_path(language: str) -> Optional[str]:
    """Find local model path for a given language"""
    if language.startswith("local:"):
        local_name = language[6:]
        model_path = os.path.join(folder_paths.models_dir, "chatterbox", local_name)
        if os.path.exists(model_path):
            return model_path
    else:
        # Check if we have a local version of a predefined language
        model_path = os.path.join(folder_paths.models_dir, "chatterbox", language)
        if os.path.exists(model_path):
            return model_path
    
    return None

def detect_model_format(model_path: str) -> str:
    """
    Auto-detect the format of models in a directory.
    Returns 'safetensors', 'pt', or 'mixed'
    """
    if not os.path.exists(model_path):
        return "pt"  # Default format
    
    has_safetensors = False
    has_pt = False
    
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            has_safetensors = True
        elif file.endswith(".pt"):
            has_pt = True
    
    if has_safetensors and has_pt:
        return "mixed"
    elif has_safetensors:
        return "safetensors"
    else:
        return "pt"

def get_available_languages() -> List[str]:
    """Get list of available language names for display"""
    models = get_chatterbox_models()
    # Clean up display names
    clean_models = []
    for model in models:
        if model.startswith("local:"):
            clean_models.append(model)  # Keep local: prefix for clarity
        else:
            clean_models.append(model)
    
    return clean_models