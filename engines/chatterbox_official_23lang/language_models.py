"""
ChatterBox Official 23-Lang Language Model Registry
Manages the official ResembleAI multilingual ChatterBox model supporting 23 languages
"""

import os
import folder_paths
from typing import Dict, List, Tuple, Optional

# Official 23-language ChatterBox model configuration
# Based on ResembleAI's official multilingual implementation
OFFICIAL_23LANG_MODELS = {
    # The single unified multilingual model supporting all 23 languages
    "ChatterBox Official 23-Lang": {
        "repo": "ResembleAI/chatterbox",
        "format": "mixed",  # Uses both .pt and .safetensors
        "description": "Official ResembleAI multilingual ChatterBox model supporting 23 languages",
        "languages": [
            "ar",  # Arabic
            "da",  # Danish
            "de",  # German
            "el",  # Greek
            "en",  # English
            "es",  # Spanish
            "fi",  # Finnish
            "fr",  # French
            "he",  # Hebrew
            "hi",  # Hindi
            "it",  # Italian
            "ja",  # Japanese
            "ko",  # Korean
            "ms",  # Malay
            "nl",  # Dutch
            "no",  # Norwegian
            "pl",  # Polish
            "pt",  # Portuguese
            "ru",  # Russian
            "sv",  # Swedish
            "sw",  # Swahili
            "tr",  # Turkish
            "zh",  # Chinese
        ],
        "required_files": {
            "v1": [
                "t3_23lang.safetensors",  # Multilingual T3 model v1
                "s3gen.pt",               # S3Gen model (same as English)
                "ve.pt",                  # Voice encoder (same as English)
                "mtl_tokenizer.json",     # Multilingual tokenizer
                "conds.pt"                # Conditioning (optional)
            ],
            "v2": [
                "t3_mtl23ls_v2.safetensors",      # Multilingual T3 model v2 with enhanced tokenization
                "s3gen.pt",                        # S3Gen model (same as English)
                "ve.pt",                           # Voice encoder (same as English)
                "grapheme_mtl_merged_expanded_v1.json",  # Enhanced grapheme/phoneme mappings with special tokens
                "mtl_tokenizer.json",              # Multilingual tokenizer (may be updated for v2)
                "conds.pt"                         # Conditioning (optional)
            ]
        },
        "multilingual": True
    }
}

# Language code to display name mapping  
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
}

def get_official_23lang_models() -> List[str]:
    """
    Get list of available official 23-lang models.
    This implementation only supports the unified multilingual model.
    """
    models = list(OFFICIAL_23LANG_MODELS.keys())
    
    # Check for local models in ComfyUI models directory
    try:
        # Check both new TTS organization and legacy path
        tts_models_dir = os.path.join(folder_paths.models_dir, "TTS", "chatterbox_official_23lang")
        legacy_models_dir = os.path.join(folder_paths.models_dir, "chatterbox_official_23lang")
        
        # Try TTS path first, then legacy
        for models_dir in [tts_models_dir, legacy_models_dir]:
            if not os.path.exists(models_dir):
                continue
                
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path):
                    # Check if it contains official 23-lang model files
                    required_files = ["t3_23lang.safetensors", "s3gen.pt", "ve.pt", "mtl_tokenizer.json"]
                    has_model = True
                    
                    for required_file in required_files:
                        file_path = os.path.join(item_path, required_file)
                        if not os.path.exists(file_path):
                            has_model = False
                            break
                    
                    if has_model:
                        local_model = f"local:{item}"
                        if local_model not in models:
                            models.append(local_model)
    except Exception:
        pass  # Ignore errors in model discovery
    
    return models

def get_supported_languages() -> List[str]:
    """Get list of supported language codes."""
    return list(SUPPORTED_LANGUAGES.keys())

def get_supported_language_names() -> List[str]:
    """Get list of supported language display names."""
    return list(SUPPORTED_LANGUAGES.values())

def get_language_name(language_code: str) -> str:
    """Get display name for a language code."""
    return SUPPORTED_LANGUAGES.get(language_code, language_code)

def get_model_config(model_name: str) -> Optional[Dict]:
    """Get configuration for a specific model"""
    if model_name.startswith("local:"):
        # Local model
        local_name = model_name[6:]  # Remove "local:" prefix
        return {
            "repo": None,
            "format": "mixed",  # Auto-detect format
            "local_path": os.path.join(folder_paths.models_dir, "TTS", "chatterbox_official_23lang", local_name),
            "description": f"Local ChatterBox Official 23-Lang model: {local_name}",
            "multilingual": True,
            "languages": list(SUPPORTED_LANGUAGES.keys())
        }
    
    return OFFICIAL_23LANG_MODELS.get(model_name)

def get_model_files_for_model(model_name: str) -> Tuple[str, str]:
    """
    Get the expected file format and repo for a model.
    Returns (format, repo_id) tuple.
    """
    config = get_model_config(model_name)
    if not config:
        # Default to the official model
        config = OFFICIAL_23LANG_MODELS["ChatterBox Official 23-Lang"]
    
    return config.get("format", "mixed"), config.get("repo")

def find_local_model_path(model_name: str) -> Optional[str]:
    """Find local model path for a given model"""
    if model_name.startswith("local:"):
        local_name = model_name[6:]
        # Try TTS path first, then legacy
        for base_dir in ["TTS", ""]:
            if base_dir:
                model_path = os.path.join(folder_paths.models_dir, base_dir, "chatterbox_official_23lang", local_name)
            else:
                model_path = os.path.join(folder_paths.models_dir, "chatterbox_official_23lang", local_name)
            if os.path.exists(model_path):
                return model_path
    else:
        # Check if we have a local version of a predefined model
        # Try TTS path first, then legacy
        for base_dir in ["TTS", ""]:
            if base_dir:
                model_path = os.path.join(folder_paths.models_dir, base_dir, "chatterbox_official_23lang", model_name)
            else:
                model_path = os.path.join(folder_paths.models_dir, "chatterbox_official_23lang", model_name)
            if os.path.exists(model_path):
                return model_path
    
    return None

def detect_model_format(model_path: str) -> str:
    """
    Auto-detect the format of models in a directory.
    Returns 'safetensors', 'pt', or 'mixed'
    """
    if not os.path.exists(model_path):
        return "mixed"  # Default format for official model
    
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

def get_model_requirements(model_name: str, model_version: str = "v1") -> List[str]:
    """Get list of required files for the official 23-lang model based on version"""
    config = get_model_config(model_name)
    if config and "required_files" in config:
        required_files = config["required_files"]

        # Handle version-specific requirements
        if isinstance(required_files, dict):
            return required_files.get(model_version, required_files.get("v1", []))
        else:
            # Legacy format - return as-is
            return required_files

    # Default requirements for official model v1
    return [
        "t3_23lang.safetensors",
        "s3gen.pt",
        "ve.pt",
        "mtl_tokenizer.json",
        "conds.pt"  # Optional but expected
    ]

def validate_model_completeness(model_path: str, model_name: str, model_version: str = "v1") -> Tuple[bool, List[str]]:
    """
    Validate if a model has all required components.
    Returns (is_complete, missing_files)
    """
    if not os.path.exists(model_path):
        return False, ["Model directory not found"]

    required_files = get_model_requirements(model_name, model_version)
    missing_files = []
    optional_files = ["conds.pt", "grapheme_mtl_merged_expanded_v1.json"]  # These are optional

    existing_files = os.listdir(model_path)
    for required_file in required_files:
        # Check for exact file name
        if required_file not in existing_files:
            # Only consider it missing if it's not optional
            if required_file not in optional_files:
                missing_files.append(required_file)

    return len(missing_files) == 0, missing_files

def supports_voice_conversion(model_name: str) -> bool:
    """
    Check if a model supports voice conversion.
    Official 23-lang model supports VC via S3Gen.
    """
    return True  # Official model supports VC

def get_available_models() -> List[str]:
    """Get list of available model names for display"""
    return get_official_23lang_models()

# Legacy compatibility functions (renamed from original chatterbox functions)
def get_chatterbox_models():
    """Legacy compatibility - redirect to official models"""
    return get_official_23lang_models()

def get_available_languages():
    """Legacy compatibility - redirect to supported languages"""  
    return get_supported_language_names()