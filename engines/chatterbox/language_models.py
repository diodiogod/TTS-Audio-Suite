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
    "German (SebastianBodza)": {
        "repo": "SebastianBodza/Kartoffelbox-v0.1",
        "format": "mixed",
        "description": "German model with emotion control features (<haha>, <wow> tags) - 600K samples"
    },
    "German (havok2)": {
        "repo": "havok2/Kartoffelbox-v0.1_0.65h2",
        "format": "safetensors",
        "subdirectory": "merged_model_v2",
        "description": "Hybrid German model (65% multi-speaker + 35% Kartoffelbox) - User-rated best quality"
    },
    "Norwegian": {
        "repo": "akhbar/chatterbox-tts-norwegian", 
        "format": "safetensors",
        "description": "Norwegian ChatterBox model (Bokmål and Nynorsk dialects) - 532M parameters"
    },
    "French": {
        "repo": "Thomcles/Chatterbox-TTS-French",
        "format": "safetensors",
        "description": "French model trained on 1,400 hours of Emilia dataset - t3_cfg only, uses shared English components",
        "incomplete": True
    },
    "Russian": {
        "repo": "niobures/Chatterbox-TTS",
        "format": "safetensors",
        "subdirectory": "ru",
        "description": "Russian model with training artifacts - Complete fine-tuned model"
    },
    "Armenian": {
        "repo": "niobures/Chatterbox-TTS",
        "format": "safetensors", 
        "subdirectory": "hy",
        "description": "Armenian model with training artifacts - Complete fine-tuned model"
    },
    "Georgian": {
        "repo": "niobures/Chatterbox-TTS",
        "format": "safetensors",
        "subdirectory": "ka", 
        "description": "Georgian model - Complete fine-tuned model with unique architecture"
    },
    "Japanese": {
        "repo": "niobures/Chatterbox-TTS",
        "format": "safetensors",
        "subdirectory": "ja",
        "description": "Japanese model - t3_cfg only, uses shared English components",
        "incomplete": True
    },
    "Korean": {
        "repo": "niobures/Chatterbox-TTS", 
        "format": "safetensors",
        "subdirectory": "ko",
        "description": "Korean model - t3_cfg only, uses shared English components",
        "incomplete": True
    },
    "Italian": {
        "repo": "niobures/Chatterbox-TTS",
        "format": "pt",
        "subdirectory": "it,en",
        "description": "Bilingual Italian/English model - complete model in single file, extended vocabulary (1500 tokens)",
        "special": "unified_model"
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
        # Check both new TTS organization and legacy path
        tts_models_dir = os.path.join(folder_paths.models_dir, "TTS", "chatterbox")
        legacy_models_dir = os.path.join(folder_paths.models_dir, "chatterbox")
        
        # Try TTS path first, then legacy
        for models_dir in [tts_models_dir, legacy_models_dir]:
            if not os.path.exists(models_dir):
                continue
                
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
            "local_path": os.path.join(folder_paths.models_dir, "TTS", "chatterbox", local_name),
            "description": f"Local ChatterBox model: {local_name}"
        }
    
    config = CHATTERBOX_MODELS.get(language)
    if config and config.get("subdirectory"):
        # Create a copy with the subdirectory information for download handling
        config_copy = config.copy()
        config_copy["download_path"] = f"{config['repo']}/{config['subdirectory']}"
        return config_copy
    
    return config

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
        # Try TTS path first, then legacy
        for base_dir in ["TTS", ""]:
            if base_dir:
                model_path = os.path.join(folder_paths.models_dir, base_dir, "chatterbox", local_name)
            else:
                model_path = os.path.join(folder_paths.models_dir, "chatterbox", local_name)
            if os.path.exists(model_path):
                return model_path
    else:
        # Check if we have a local version of a predefined language
        # Try TTS path first, then legacy
        for base_dir in ["TTS", ""]:
            if base_dir:
                model_path = os.path.join(folder_paths.models_dir, base_dir, "chatterbox", language)
            else:
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

def is_model_incomplete(language: str) -> bool:
    """Check if a model is marked as incomplete (missing components)"""
    config = CHATTERBOX_MODELS.get(language)
    return config.get("incomplete", False) if config else False

def is_unified_model(language: str) -> bool:
    """Check if a model uses unified/special architecture (like Italian)"""
    config = CHATTERBOX_MODELS.get(language)
    return config.get("special") == "unified_model" if config else False

def get_tokenizer_filename(language: str) -> str:
    """Get the correct tokenizer filename for a language model"""
    if language == "Japanese":
        return "tokenizer_jp.json"
    elif language == "Korean":
        return "tokenizer_en_ko.json"
    else:
        return "tokenizer.json"

def get_model_requirements(language: str) -> List[str]:
    """Get list of required files for a ChatterBox model"""
    
    # Handle special Italian unified model
    if language == "Italian":
        return ["chatterbox_italian_final.pt", "config.json"]
    
    # Handle special cases for incomplete models
    if language == "French":
        # French model only has t3_cfg.safetensors, no tokenizer in repo
        base_requirements = ["t3_cfg.safetensors"]
    else:
        # Other models have tokenizers
        tokenizer_file = get_tokenizer_filename(language)
        base_requirements = ["t3_cfg.safetensors", tokenizer_file]
    
    # Complete models need all components
    if not is_model_incomplete(language):
        base_requirements.extend(["s3gen.safetensors", "ve.safetensors", "conds.pt"])
    
    return base_requirements

def validate_model_completeness(model_path: str, language: str) -> Tuple[bool, List[str]]:
    """
    Validate if a model has all required components.
    Returns (is_complete, missing_files)
    """
    if not os.path.exists(model_path):
        return False, ["Model directory not found"]
    
    required_files = get_model_requirements(language)
    missing_files = []
    
    existing_files = os.listdir(model_path)
    for required_file in required_files:
        # Check for file with correct extension
        file_found = False
        for existing_file in existing_files:
            if existing_file == required_file or existing_file.startswith(required_file.split('.')[0] + '.'):
                file_found = True
                break
        
        if not file_found:
            missing_files.append(required_file)
    
    return len(missing_files) == 0, missing_files

def supports_voice_conversion(language: str) -> bool:
    """
    Check if a language model supports voice conversion (has s3gen component).
    
    Args:
        language: Language model name
        
    Returns:
        True if VC is supported, False otherwise
    """
    # Check if model is marked as incomplete
    if is_model_incomplete(language):
        return False
    
    # Check if model requirements include s3gen (required for VC)
    requirements = get_model_requirements(language)
    has_s3gen = any(req.startswith("s3gen") for req in requirements)
    
    # For local models, check if s3gen file actually exists
    if language.startswith("local:"):
        local_path = find_local_model_path(language)
        if local_path and os.path.exists(local_path):
            # Check if s3gen file exists in the directory
            for ext in [".safetensors", ".pt"]:
                if os.path.exists(os.path.join(local_path, f"s3gen{ext}")):
                    return True
            return False
    
    return has_s3gen

def get_vc_supported_languages() -> List[str]:
    """Get list of languages that support voice conversion"""
    all_languages = get_chatterbox_models()
    return [lang for lang in all_languages if supports_voice_conversion(lang)]

def get_vc_unsupported_languages() -> List[str]:
    """Get list of languages that do NOT support voice conversion"""
    all_languages = get_chatterbox_models()
    return [lang for lang in all_languages if not supports_voice_conversion(lang)]

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