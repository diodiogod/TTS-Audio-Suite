"""
Model Downloader - Auto-download utility for TTS Suite models
Adapted from reference RVC implementation for TTS Suite integration
"""

import os
import requests
import shutil
from zipfile import ZipFile
from typing import Tuple, List, Optional
import hashlib

# Import ComfyUI folder paths
try:
    import folder_paths
    MODELS_DIR = folder_paths.models_dir
except ImportError:
    MODELS_DIR = os.path.expanduser("~/ComfyUI/models")

def _find_legacy_model_in_cache(model_filename: str, download_url: str) -> Optional[str]:
    """
    Find legacy RVC/UVR model in HuggingFace cache based on the download URL
    
    Args:
        model_filename: Name of the model file
        download_url: The download URL to extract repo information from
        
    Returns:
        Path to the exact cached model if found, None otherwise
    """
    try:
        # Only check HuggingFace URLs
        if "huggingface.co" not in download_url:
            return None
        
        # Extract repo ID from URL
        # Format: https://huggingface.co/owner/repo/resolve/main/path/file.ext
        parts = download_url.split("/")
        if len(parts) < 5:
            return None
        
        owner = parts[3]
        repo = parts[4] 
        repo_id = f"{owner}/{repo}"
        
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
                        # Check each snapshot for the specific model file
                        for snapshot in snapshots:
                            snapshot_path = os.path.join(snapshots_dir, snapshot)
                            
                            # Try direct file path first
                            model_path = os.path.join(snapshot_path, model_filename)
                            if os.path.exists(model_path):
                                return model_path
                            
                            # Try with subdirectory path (for models like RVC/model.pth)
                            if "/" in model_filename:
                                model_path = os.path.join(snapshot_path, model_filename)
                                if os.path.exists(model_path):
                                    return model_path
                except OSError:
                    pass
        
        return None
    except Exception as e:
        print(f"⚠️ Error checking HuggingFace cache for legacy model '{model_filename}': {e}")
        return None

# Download sources - Updated to use official RVC sources
# RVC character models - using community models as reference implementation uses them
RVC_DOWNLOAD_BASE = 'https://huggingface.co/datasets/SayanoAI/RVC-Studio/resolve/main/'
# Base model URLs - using official sources where available
BASE_MODEL_URLS = {
    'content-vec-best.safetensors': 'https://huggingface.co/lengyue233/content-vec-best/resolve/main/content-vec-best.safetensors',
    'rmvpe.pt': 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt'
}
# UVR models - these should be downloaded via UVR's official download center
# For compatibility, we keep the current source but recommend official UVR download
UVR_DOWNLOAD_BASE = RVC_DOWNLOAD_BASE  # Temporary - should use official UVR sources

# Available models for auto-download
AVAILABLE_RVC_MODELS = [
    "RVC/Claire.pth",
    "RVC/Sayano.pth", 
    "RVC/Mae_v2.pth",
    "RVC/Fuji.pth",
    "RVC/Monika.pth"
]

AVAILABLE_RVC_INDEXES = [
    "RVC/.index/added_IVF1063_Flat_nprobe_1_Sayano_v2.index",
    "RVC/.index/added_IVF985_Flat_nprobe_1_Fuji_v2.index", 
    "RVC/.index/Monika_v2_40k.index",
    "RVC/.index/Sayano_v2_40k.index"
]

AVAILABLE_BASE_MODELS = [
    "content-vec-best.safetensors",
    "rmvpe.pt"
]

AVAILABLE_UVR_MODELS = [
    "UVR/HP5-vocals+instrumentals.pth",
    "UVR/UVR-DeEcho-DeReverb.pth",
    "UVR/5_HP-Karaoke-UVR.pth", 
    "UVR/6_HP-Karaoke-UVR.pth",
    "UVR/UVR-MDX-NET-vocal_FT.onnx",
    "UVR/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "UVR/UVR-BVE-4B_SN-44100-1.pth",
    "UVR/UVR-DeNoise.pth"
]


def download_model_from_url(model_name: str, target_path: str, download_url: str) -> Optional[str]:
    """
    Download a model file from a specific URL.
    
    Args:
        model_name: Name/path of model to download (for display)
        target_path: Local path where model should be saved
        download_url: Full URL to download from
        
    Returns:
        Path to downloaded model or None if failed
    """
    if os.path.exists(target_path):
        print(f"📁 Model already exists: {os.path.basename(target_path)}")
        return target_path
        
    try:
        print(f"📥 Downloading {model_name} from official source: {download_url}")
        
        # Create directory structure
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Download with progress
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r📥 Downloading {os.path.basename(target_path)}: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✅ Successfully downloaded: {os.path.basename(target_path)}")
        return target_path
        
    except Exception as e:
        print(f"\n❌ Download failed for {model_name}: {e}")
        # Clean up partial download
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except:
                pass
        return None


def download_model(model_name: str, target_path: str, base_url: str = RVC_DOWNLOAD_BASE) -> str:
    """
    Download a model file from remote source or find in cache.
    
    Args:
        model_name: Name/path of model to download (e.g., "RVC/Claire.pth")
        target_path: Local path where model should be saved (used if downloading)
        base_url: Base URL for downloads
        
    Returns:
        Path to model (target_path if downloaded, cache_path if found in cache, None if failed)
    """
    if os.path.exists(target_path):
        print(f"📁 Model already exists: {os.path.basename(target_path)}")
        return target_path
    
    # Check HuggingFace cache before downloading
    download_url = f"{base_url}{model_name}"
    cache_path = _find_legacy_model_in_cache(model_name, download_url)
    if cache_path:
        print(f"💾 Using HuggingFace cache for '{model_name}': {cache_path}")
        return cache_path  # Return cache path directly - no copying
    
    try:
        print(f"📥 Downloading {model_name} from {download_url}")
        
        # Create directory structure
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Download with progress
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r📥 Downloading {os.path.basename(target_path)}: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✅ Successfully downloaded: {os.path.basename(target_path)}")
        return target_path
        
    except Exception as e:
        print(f"\n❌ Download failed for {model_name}: {e}")
        # Clean up partial download
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except:
                pass
        return None


def download_rvc_model(model_name: str) -> Optional[str]:
    """
    Download RVC model if available.
    
    Args:
        model_name: Name of RVC model (e.g., "Claire.pth")
        
    Returns:
        Path to downloaded model or None if failed
    """
    # Ensure .pth extension
    if not model_name.endswith('.pth'):
        model_name = f"{model_name}.pth"
    
    # Check if it's in our available models
    rvc_path = f"RVC/{model_name}"
    if rvc_path not in AVAILABLE_RVC_MODELS:
        print(f"⚠️  Model {model_name} not available for auto-download")
        return None
    
    target_path = os.path.join(MODELS_DIR, "TTS", "RVC", model_name)
    
    downloaded_path = download_model(rvc_path, target_path)
    return downloaded_path  # Could be target_path (downloaded) or cache_path (cached) or None (failed)


def download_rvc_index(index_name: str) -> Optional[str]:
    """
    Download RVC index file if available.
    
    Args:
        index_name: Name of index file (e.g., "Claire.index")
        
    Returns:
        Path to downloaded index or None if failed
    """
    # Ensure .index extension
    if not index_name.endswith('.index'):
        index_name = f"{index_name}.index"
    
    # Check if it's in our available indexes
    index_path = f"RVC/.index/{index_name}"
    if index_path not in AVAILABLE_RVC_INDEXES:
        print(f"⚠️  Index {index_name} not available for auto-download")
        return None
    
    target_path = os.path.join(MODELS_DIR, "TTS", "RVC", ".index", index_name)
    
    downloaded_path = download_model(index_path, target_path)
    return downloaded_path  # Could be target_path (downloaded) or cache_path (cached) or None (failed)


def download_base_model(model_name: str) -> Optional[str]:
    """
    Download base model (Hubert, RMVPE, etc.) from official sources.
    
    Args:
        model_name: Name of base model
        
    Returns:
        Path to downloaded model or None if failed
    """
    if model_name not in AVAILABLE_BASE_MODELS:
        print(f"⚠️  Base model {model_name} not available for auto-download")
        return None
    
    target_path = os.path.join(MODELS_DIR, "TTS", "RVC", model_name)
    
    # Use official sources for base models when available
    if model_name in BASE_MODEL_URLS:
        official_url = BASE_MODEL_URLS[model_name]
        print(f"📥 Using official source for {model_name}: {official_url}")
        return download_model_from_url(model_name, target_path, official_url)
    else:
        # Fallback to default base URL
        downloaded_path = download_model(model_name, target_path)
        return downloaded_path  # Could be target_path (downloaded) or cache_path (cached) or None (failed)
    return None


def download_uvr_model(model_name: str) -> Optional[str]:
    """
    Download UVR model if available.
    
    Args:
        model_name: Name of UVR model
        
    Returns:
        Path to downloaded model or None if failed
    """
    # Handle different UVR path formats
    if not model_name.startswith("UVR/"):
        uvr_path = f"UVR/{model_name}"
    else:
        uvr_path = model_name
    
    if uvr_path not in AVAILABLE_UVR_MODELS:
        print(f"⚠️  UVR model {model_name} not available for auto-download")
        return None
    
    # Save to TTS organization: models/TTS/UVR/ instead of models/UVR/
    tts_uvr_path = uvr_path.replace("UVR/", "TTS/UVR/")
    target_path = os.path.join(MODELS_DIR, tts_uvr_path)
    
    downloaded_path = download_model(uvr_path, target_path)
    return downloaded_path  # Could be target_path (downloaded) or cache_path (cached) or None (failed)


def extract_zip_flat(zip_path: str, extract_to: str, cleanup: bool = False) -> List[str]:
    """
    Extract ZIP file without preserving directory structure.
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to
        cleanup: Whether to delete ZIP after extraction
        
    Returns:
        List of extracted file names
    """
    os.makedirs(extract_to, exist_ok=True)
    extracted_files = []
    
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                # Get filename without directory structure
                filename = os.path.basename(member)
                if filename:  # Skip directories
                    file_path = os.path.join(extract_to, filename)
                    
                    # Extract file
                    with zip_ref.open(member) as source, open(file_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    
                    extracted_files.append(filename)
        
        if cleanup and os.path.exists(zip_path):
            os.remove(zip_path)
        
        print(f"✅ Extracted {len(extracted_files)} files to {extract_to}")
        return extracted_files
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return []


def download_rmvpe_for_reference() -> Optional[str]:
    """
    Download RMVPE model to standard ComfyUI models directory
    
    Returns:
        Path to downloaded model or None if failed
    """
    try:
        rmvpe_url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
        
        # Use TTS organization for consistency
        rvc_models_dir = os.path.join(MODELS_DIR, "TTS", "RVC")
        
        # Ensure directory exists
        os.makedirs(rvc_models_dir, exist_ok=True)
        
        rmvpe_path = os.path.join(rvc_models_dir, "rmvpe.pt")
        
        if not os.path.exists(rmvpe_path):
            print(f"📥 Downloading RMVPE model to standard models directory...")
            try:
                response = requests.get(rmvpe_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(rmvpe_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                print(f"\r📥 Downloading RMVPE: {progress:.1f}%", end='', flush=True)
                
                print(f"\n✅ Downloaded RMVPE model to: {rmvpe_path}")
                return rmvpe_path
                
            except Exception as e:
                print(f"❌ Failed to download RMVPE model: {e}")
                return None
        else:
            print(f"✅ RMVPE model already exists: {rmvpe_path}")
            return rmvpe_path
            
    except Exception as e:
        print(f"❌ Error downloading RMVPE for reference: {e}")
        return None


def get_model_hash(file_path: str, hash_size: int = 1024*1024) -> str:
    """
    Get hash of model file for verification.
    
    Args:
        file_path: Path to model file
        hash_size: Number of bytes to hash (default 1MB)
        
    Returns:
        MD5 hash string
    """
    if not os.path.exists(file_path):
        return ""
    
    try:
        with open(file_path, 'rb') as f:
            data = f.read(hash_size)
            return hashlib.md5(data).hexdigest()
    except:
        return ""


def verify_model_integrity(file_path: str, min_size: int = 1024) -> bool:
    """
    Basic verification that model file is valid.
    
    Args:
        file_path: Path to model file
        min_size: Minimum expected file size
        
    Returns:
        True if file appears valid
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        file_size = os.path.getsize(file_path)
        return file_size >= min_size
    except:
        return False


# Convenience function for backward compatibility
def model_downloader(model_name: str) -> Optional[str]:
    """
    Auto-detect model type and download.
    
    Args:
        model_name: Name of model to download
        
    Returns:
        Path to downloaded model or None
    """
    if model_name.endswith('.pth'):
        return download_rvc_model(model_name)
    elif model_name.endswith('.index'):
        return download_rvc_index(model_name) 
    elif model_name.endswith('.safetensors') or model_name == 'rmvpe.pt':
        return download_base_model(model_name)
    elif 'UVR' in model_name or model_name.endswith('.onnx'):
        return download_uvr_model(model_name)
    else:
        print(f"⚠️  Unknown model type: {model_name}")
        return None