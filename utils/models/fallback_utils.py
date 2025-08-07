"""
Model Fallback Utils - Generic utilities for TTS engine model fallback handling
Provides simple local-first fallback behavior for ANY TTS engine
"""

import os
from typing import Optional, Callable, Any, List
from pathlib import Path


def try_local_first(
    search_paths: List[str],
    local_loader: Callable[[str], Any],
    fallback_loader: Callable[[], Any],
    fallback_name: str = "fallback model",
    original_request: str = "requested model"
) -> Any:
    """
    Generic local-first model loading with HuggingFace fallback.
    Works for ANY TTS engine - no hardcoded engine knowledge.
    
    Args:
        search_paths: List of directory paths to check for local models
        local_loader: Function that loads a local model given a path: (path) -> model  
        fallback_loader: Function that downloads/loads fallback model: () -> model
        fallback_name: Name for logging (e.g., "English", "F5TTS_Base")
        original_request: Original request for logging (e.g., "French", "F5-DE")
        
    Returns:
        Loaded model instance
        
    Example:
        # ChatterBox usage:
        return try_local_first(
            search_paths=["models/chatterbox"],
            local_loader=lambda path: ChatterboxTTS.from_local(path, device),
            fallback_loader=lambda: ChatterboxTTS.from_pretrained(device, "English"),
            fallback_name="English",
            original_request="French"
        )
    """
    print(f"🔄 {original_request} not found, checking local {fallback_name} before download...")
    
    # Try each search path
    for search_path in search_paths:
        path = Path(search_path)
        if path.exists() and path.is_dir():
            try:
                # Check if directory has any files (basic validation)
                if any(path.iterdir()):
                    print(f"📁 Found local {fallback_name} at {search_path}, using instead of download")
                    return local_loader(str(path))
            except Exception as e:
                print(f"⚠️ Local {fallback_name} at {search_path} failed to load: {e}")
                continue
    
    # No local model found or all failed - use fallback
    print(f"📦 Loading {fallback_name} from HuggingFace as final fallback")
    return fallback_loader()


def get_models_dir() -> Optional[str]:
    """Get ComfyUI models directory if available."""
    try:
        import folder_paths
        return folder_paths.models_dir
    except ImportError:
        return None