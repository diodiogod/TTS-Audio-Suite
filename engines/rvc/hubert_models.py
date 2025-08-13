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
        "tooltip": """Content Vec 768 - Best overall quality (recommended)
• Best voice similarity and quality
• 768-dimensional features for rich voice representation
• Trained on diverse multilingual data
• Size: ~200MB
• Works well with all languages""",
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "size": "200MB",
        "filename": "content-vec-best.pt"
    },
    
    "hubert-base": {
        "description": "HuBERT Base (English)",
        "tooltip": """HuBERT Base - Standard English model
• Original Facebook HuBERT base model
• Optimized for English speakers
• Good balance of speed and quality
• Size: ~190MB
• Best for English content""",
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "size": "190MB",
        "filename": "hubert_base.pt"
    },
    
    "hubert-base-japanese": {
        "description": "HuBERT Japanese",
        "tooltip": """HuBERT Base Japanese - Optimized for Japanese
• Fine-tuned on Japanese speech data
• Better phoneme recognition for Japanese
• Improved pitch extraction for tonal patterns
• Size: ~190MB
• Recommended for Japanese voices""",
        "url": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/pytorch_model.bin",
        "size": "190MB", 
        "filename": "hubert_base_jp.pt"
    },
    
    "hubert-base-korean": {
        "description": "HuBERT Korean",
        "tooltip": """HuBERT Base Korean - Optimized for Korean
• Trained on Korean speech corpus
• Better handling of Korean phonetics
• Improved consonant clustering recognition
• Size: ~190MB
• Recommended for Korean voices""",
        "url": "https://huggingface.co/team-lucid/hubert-base-korean/resolve/main/pytorch_model.bin",
        "size": "190MB",
        "filename": "hubert_base_kr.pt"
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
    
    "hubert-soft": {
        "description": "HuBERT Soft (Speech Enhancement)",
        "tooltip": """HuBERT Soft - Enhanced speech features
• Soft speech unit extraction
• Better for whisper and soft voices
• Enhanced emotional preservation
• Size: ~190MB
• Good for gentle/soft voice styles""",
        "url": "https://huggingface.co/microsoft/hubert-base-ls960/resolve/main/pytorch_model.bin",
        "size": "190MB",
        "filename": "hubert-soft.pt"
    },
    
    "wav2vec2-xlsr-53": {
        "description": "Wav2Vec2 XLSR (Multilingual)",
        "tooltip": """Wav2Vec2 XLSR-53 - Universal multilingual model
• Trained on 53 languages
• Good for less common languages
• Broader language coverage
• Size: ~315MB
• Use when language-specific model unavailable""",
        "url": "https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english/resolve/main/pytorch_model.bin",
        "size": "315MB",
        "filename": "wav2vec2-xlsr-53.pt"
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
        'en': 'hubert-base',
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
        
    model_path = os.path.join(models_dir, "hubert", info["filename"])
    return not os.path.exists(model_path)

def get_hubert_download_url(model_key: str) -> Optional[str]:
    """Get the download URL for a HuBERT model."""
    info = get_hubert_model_info(model_key)
    return info.get("url") if info else None

def get_hubert_filename(model_key: str) -> Optional[str]:
    """Get the local filename for a HuBERT model."""
    info = get_hubert_model_info(model_key)
    return info.get("filename") if info else None