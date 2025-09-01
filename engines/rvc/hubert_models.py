"""
HuBERT Model Registry for RVC
Manages available HuBERT models with auto-download support and detailed descriptions
"""

from typing import Dict, Optional, List
import os

# HuBERT model configurations with download URLs and descriptions
HUBERT_MODELS = {
    "auto": {
        "description": "Automatically select best available model",
        "tooltip": "Auto-select the best HuBERT model based on availability and input language",
        "url": None,
        "size": None,
        "filename": None
    },
    
    "content-vec-best": {
        "description": "Content Vec 768 (Recommended)",
        "tooltip": """Content Vec 768 - Best for RVC voice conversion
• Optimized specifically for RVC applications
• Better than standard HuBERT for voice similarity
• Excellent balance of speed and quality
• Works perfectly with all languages
• Native safetensors format (fastest loading)
• Size: 378MB
• Most reliable choice for RVC""",
        "url": "https://huggingface.co/SayanoAI/RVC-models/resolve/main/content-vec-best.safetensors",
        "fallback_url": "https://huggingface.co/lengyue233/content-vec-best/resolve/main/pytorch_model.bin",
        "size": "378MB",
        "filename": "content-vec-best.safetensors"
    },
    
    "hubert-base-japanese": {
        "description": "HuBERT Japanese",
        "tooltip": """HuBERT Base Japanese - Optimized for Japanese
• Fine-tuned on Japanese speech data
• Better phoneme recognition for Japanese
• Improved pitch extraction for tonal patterns
• Size: 378MB
• Recommended for Japanese voices""",
        "url": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/model.safetensors",
        "size": "378MB", 
        "filename": "hubert_base_jp.safetensors"
    },
    
    "hubert-base-korean": {
        "description": "HuBERT Korean",
        "tooltip": """HuBERT Base Korean - Optimized for Korean
• Trained on Korean speech corpus
• Better handling of Korean phonetics
• Improved consonant clustering recognition
• Size: 1.26GB
• Recommended for Korean voices""",
        "url": "https://huggingface.co/team-lucid/hubert-base-korean/resolve/main/model.safetensors",
        "size": "1.26GB",
        "filename": "hubert_base_kr.safetensors"
    },
    
    "chinese-hubert-base": {
        "description": "Chinese HuBERT Base",
        "tooltip": """Chinese HuBERT - Optimized for Mandarin
• Trained on Mandarin Chinese data
• Better tonal pattern recognition
• Improved for Chinese phonemes
• Size: ~190MB
• Best for Mandarin Chinese voices""",
        "url": "https://huggingface.co/TencentGameMate/chinese-hubert-base/resolve/main/pytorch_model.bin",
        "size": "190MB",
        "filename": "chinese-hubert-base.pt"
    },
    
    
    "hubert-large": {
        "description": "HuBERT Large (Highest Quality)",
        "tooltip": """HuBERT Large - Maximum quality model
• Highest quality feature extraction
• 1024-dimensional representations
• Best voice cloning accuracy
• Size: ~1.2GB
• Slower but highest quality results""",
        "url": "https://huggingface.co/facebook/hubert-large-ls960-ft/resolve/main/pytorch_model.bin",
        "size": "1.2GB",
        "filename": "hubert_large.pt"
    }
}

def get_available_hubert_models() -> List[str]:
    """Get list of available HuBERT model names."""
    return list(HUBERT_MODELS.keys())

def get_hubert_model_descriptions() -> List[str]:
    """Get list of HuBERT models with descriptions for dropdown."""
    return [f"{key}: {info['description']}" for key, info in HUBERT_MODELS.items()]

def get_hubert_model_info(model_key: str) -> Optional[Dict]:
    """Get detailed information about a specific HuBERT model."""
    # Handle description format "key: description"
    if ": " in model_key:
        model_key = model_key.split(": ")[0]
    return HUBERT_MODELS.get(model_key)

def get_hubert_tooltip(model_key: str) -> str:
    """Get the tooltip for a specific HuBERT model."""
    if ": " in model_key:
        model_key = model_key.split(": ")[0]
    info = HUBERT_MODELS.get(model_key, {})
    return info.get("tooltip", "No description available")

def get_best_hubert_for_language(language_code: str) -> str:
    """
    Get the recommended HuBERT model for a specific language.
    
    Args:
        language_code: Language code (e.g., 'en', 'ja', 'ko', 'zh')
        
    Returns:
        Recommended HuBERT model key
    """
    language_map = {
        'en': 'content-vec-best',  # Use RVC compatible version
        'ja': 'hubert-base-japanese',
        'jp': 'hubert-base-japanese',
        'ko': 'hubert-base-korean',
        'kr': 'hubert-base-korean',
        'zh': 'chinese-hubert-base',
        'cn': 'chinese-hubert-base',
        'cmn': 'chinese-hubert-base',
        # Default to content-vec for other languages
    }
    
    return language_map.get(language_code.lower(), 'content-vec-best')

def should_download_hubert(model_key: str, models_dir: str) -> bool:
    """
    Check if a HuBERT model needs to be downloaded.
    
    Args:
        model_key: HuBERT model key
        models_dir: Directory where models are stored
        
    Returns:
        True if model needs to be downloaded
    """
    if model_key == "auto":
        return False
        
    info = get_hubert_model_info(model_key)
    if not info or not info.get("filename"):
        return False
        
    # Check both TTS and legacy paths
    tts_path = os.path.join(models_dir, "TTS", "hubert", info["filename"])
    legacy_path = os.path.join(models_dir, "hubert", info["filename"])
    direct_path = os.path.join(models_dir, info["filename"])  # Some models might be directly in models/
    
    return not (os.path.exists(tts_path) or os.path.exists(legacy_path) or os.path.exists(direct_path))

def get_hubert_download_url(model_key: str) -> Optional[str]:
    """Get the download URL for a HuBERT model."""
    info = get_hubert_model_info(model_key)
    return info.get("url") if info else None

def get_hubert_filename(model_key: str) -> Optional[str]:
    """Get the local filename for a HuBERT model."""
    info = get_hubert_model_info(model_key)
    return info.get("filename") if info else None