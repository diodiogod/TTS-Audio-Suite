"""
HuBERT Model Downloader for RVC
Handles automatic downloading of HuBERT models from Hugging Face
"""

import os
import requests
from pathlib import Path
from typing import Optional
import hashlib
from tqdm import tqdm

def download_hubert_model(model_key: str, models_dir: str, progress_callback=None) -> Optional[str]:
    """
    Download a HuBERT model if not already present.
    
    Args:
        model_key: Key of the HuBERT model to download
        models_dir: Base directory for models
        progress_callback: Optional callback for progress updates
        
    Returns:
        Path to the downloaded model file, or None if failed
    """
    from .hubert_models import get_hubert_model_info, get_hubert_filename, get_hubert_download_url

def _find_hubert_in_cache(model_key: str) -> Optional[str]:
    """
    Find the specific HuBERT model the user selected in HuggingFace cache
    
    Args:
        model_key: Specific HuBERT model key (e.g., "japanese-hubert-base")
        
    Returns:
        Path to the exact cached model if found, None otherwise
    """
    try:
        from .hubert_models import get_hubert_model_info
        
        # Get the model info to find the correct repo ID
        model_info = get_hubert_model_info(model_key)
        if not model_info or "url" not in model_info:
            return None
        
        # Extract repo ID from URL for known HuggingFace models
        url = model_info["url"]
        repo_id = None
        
        if "huggingface.co" in url:
            # Parse HuggingFace URL: https://huggingface.co/owner/repo/resolve/main/file.ext
            parts = url.split("/")
            if len(parts) >= 5:
                owner = parts[3]  # e.g., "rinna" 
                repo = parts[4]   # e.g., "japanese-hubert-base"
                repo_id = f"{owner}/{repo}"
        else:
            # Not a HuggingFace model, no cache to check
            return None
        
        if not repo_id:
            return None
        
        # Get HuggingFace cache directory  
        cache_home = os.environ.get('HUGGINGFACE_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
        
        # Convert repo ID to cache format
        cache_folder_name = f"models--{repo_id.replace('/', '--')}"
        cache_path = os.path.join(cache_home, cache_folder_name)
        
        if os.path.exists(cache_path):
            # Look for the snapshots directory
            snapshots_dir = os.path.join(cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                try:
                    snapshots = os.listdir(snapshots_dir)
                    if snapshots:
                        # Use the first available snapshot (typically latest)
                        latest_snapshot = os.path.join(snapshots_dir, snapshots[0])
                        if os.path.exists(latest_snapshot):
                            # Check for the exact model file we need
                            expected_filename = model_info.get("filename", "")
                            if expected_filename:
                                # Try to find the exact file first
                                expected_path = os.path.join(latest_snapshot, expected_filename)
                                if os.path.exists(expected_path):
                                    return expected_path
                            
                            # Fallback: look for common HuBERT model files
                            for model_file in ["pytorch_model.bin", "model.safetensors", "model.bin"]:
                                model_path = os.path.join(latest_snapshot, model_file)
                                if os.path.exists(model_path):
                                    return model_path
                except OSError:
                    pass
        
        return None
    except Exception as e:
        print(f"⚠️ Error checking HuggingFace cache for HuBERT model '{model_key}': {e}")
        return None
    
    # Get model information
    info = get_hubert_model_info(model_key)
    if not info:
        print(f"❌ Unknown HuBERT model: {model_key}")
        return None
    
    filename = get_hubert_filename(model_key)
    url = get_hubert_download_url(model_key)
    
    if not filename or not url:
        print(f"❌ No download information for {model_key}")
        return None
    
    # Create TTS/hubert directory if needed (new organization)
    hubert_dir = os.path.join(models_dir, "TTS", "hubert")
    os.makedirs(hubert_dir, exist_ok=True)
    
    # Full path for the model (new TTS organization)
    model_path = os.path.join(hubert_dir, filename)
    
    # Check if already exists in new location
    if os.path.exists(model_path):
        print(f"✅ HuBERT model already exists: {filename}")
        return model_path
        
    # Check if exists in legacy locations
    legacy_paths = [
        os.path.join(models_dir, "hubert", filename),
        os.path.join(models_dir, filename)  # Direct in models/
    ]
    
    for legacy_path in legacy_paths:
        if os.path.exists(legacy_path):
            print(f"✅ HuBERT model found in legacy location: {legacy_path}")
            return legacy_path
    
    # Check HuggingFace cache for the specific model
    cache_path = _find_hubert_in_cache(model_key)
    if cache_path:
        print(f"💾 Using HuggingFace cache for HuBERT model '{model_key}': {cache_path}")
        return cache_path
    
    # Download the model
    print(f"📥 Downloading HuBERT model: {info['description']}")
    print(f"   URL: {url}")
    print(f"   Size: {info.get('size', 'Unknown')}")
    
    temp_path = None
    try:
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Use temporary file during download
        temp_path = model_path + ".downloading"
        
        with open(temp_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        if progress_callback:
                            progress_callback(pbar.n, total_size)
        
        # Move temp file to final location
        os.rename(temp_path, model_path)
        
        print(f"✅ Successfully downloaded: {filename}")
        print(f"📁 Downloaded to: {model_path}")
        return model_path
        
    except requests.RequestException as e:
        print(f"❌ Failed to download {filename}: {e}")
        # Clean up partial download
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return None
    except Exception as e:
        print(f"❌ Unexpected error downloading {filename}: {e}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return None

def find_or_download_hubert(model_key: str, models_dir: str) -> Optional[str]:
    """
    Find a HuBERT model locally or download if needed.
    
    Args:
        model_key: HuBERT model key or "auto"
        models_dir: Base models directory
        
    Returns:
        Path to the HuBERT model file
    """
    from .hubert_models import (
        get_hubert_filename, 
        get_available_hubert_models,
        should_download_hubert
    )
    
    # Handle "auto" selection
    if model_key == "auto":
        return find_best_available_hubert(models_dir)
    
    # Check if download needed
    if should_download_hubert(model_key, models_dir):
        downloaded_path = download_hubert_model(model_key, models_dir)
        if downloaded_path:
            return downloaded_path
    else:
        # Model should already exist
        filename = get_hubert_filename(model_key)
        if filename:
            model_path = os.path.join(models_dir, "hubert", filename)
            if os.path.exists(model_path):
                return model_path
    
    # Fallback to finding any available model
    print(f"⚠️ Could not get {model_key}, falling back to auto-detection")
    return find_best_available_hubert(models_dir)

def find_best_available_hubert(models_dir: str) -> Optional[str]:
    """
    Find the best available HuBERT model in order of preference.
    
    Args:
        models_dir: Base models directory
        
    Returns:
        Path to the best available HuBERT model
    """
    from .hubert_models import HUBERT_MODELS
    
    hubert_dir = os.path.join(models_dir, "hubert")
    
    # Priority order for auto-selection
    priority_order = [
        'hubert-base-rvc',
        'chinese-hubert-base',
        'hubert-base-japanese',
        'hubert-base-korean',
        'hubert-large'
    ]
    
    # First check in priority order
    for model_key in priority_order:
        if model_key in HUBERT_MODELS:
            info = HUBERT_MODELS[model_key]
            if info.get('filename'):
                model_path = os.path.join(hubert_dir, info['filename'])
                if os.path.exists(model_path):
                    print(f"✅ Auto-selected HuBERT model: {info['description']}")
                    return model_path
    
    # Check for any .pt or .safetensors files in hubert directory
    if os.path.exists(hubert_dir):
        for file in os.listdir(hubert_dir):
            if file.endswith(('.pt', '.safetensors', '.bin')):
                model_path = os.path.join(hubert_dir, file)
                print(f"✅ Found HuBERT model: {file}")
                return model_path
    
    # Try to download hubert-base-rvc as fallback
    print("📥 No HuBERT model found, downloading recommended model...")
    return download_hubert_model('hubert-base-rvc', models_dir)

def ensure_hubert_model(model_key: str = "auto") -> Optional[str]:
    """
    Ensure a HuBERT model is available, downloading if necessary.
    
    Args:
        model_key: HuBERT model key or "auto"
        
    Returns:
        Path to the HuBERT model file
    """
    try:
        import folder_paths
        models_dir = folder_paths.models_dir
    except ImportError:
        # Fallback to common paths
        models_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(models_dir):
            models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    
    return find_or_download_hubert(model_key, models_dir)