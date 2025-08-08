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

# Download sources
RVC_DOWNLOAD_BASE = 'https://huggingface.co/datasets/SayanoAI/RVC-Studio/resolve/main/'

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


def download_model(model_name: str, target_path: str, base_url: str = RVC_DOWNLOAD_BASE) -> bool:
    """
    Download a model file from remote source.
    
    Args:
        model_name: Name/path of model to download (e.g., "RVC/Claire.pth")
        target_path: Local path where model should be saved
        base_url: Base URL for downloads
        
    Returns:
        True if download successful, False otherwise
    """
    if os.path.exists(target_path):
        print(f"📁 Model already exists: {os.path.basename(target_path)}")
        return True
        
    try:
        download_url = f"{base_url}{model_name}"
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
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed for {model_name}: {e}")
        # Clean up partial download
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except:
                pass
        return False


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
    
    target_path = os.path.join(MODELS_DIR, "RVC", model_name)
    
    if download_model(rvc_path, target_path):
        return target_path
    return None


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
    
    target_path = os.path.join(MODELS_DIR, "RVC", ".index", index_name)
    
    if download_model(index_path, target_path):
        return target_path
    return None


def download_base_model(model_name: str) -> Optional[str]:
    """
    Download base model (Hubert, RMVPE, etc.) if available.
    
    Args:
        model_name: Name of base model
        
    Returns:
        Path to downloaded model or None if failed
    """
    if model_name not in AVAILABLE_BASE_MODELS:
        print(f"⚠️  Base model {model_name} not available for auto-download")
        return None
    
    target_path = os.path.join(MODELS_DIR, model_name)
    
    if download_model(model_name, target_path):
        return target_path
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
    
    target_path = os.path.join(MODELS_DIR, uvr_path)
    
    if download_model(uvr_path, target_path):
        return target_path
    return None


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
    Download RMVPE model specifically for reference implementation
    
    Returns:
        Path to downloaded model or None if failed
    """
    try:
        rmvpe_url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
        
        # Get reference models directory
        current_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(current_dir))
        reference_models_dir = os.path.join(project_root, "docs", "RVC", "Comfy-RVC-For-Reference", "models")
        
        # Ensure directory exists
        os.makedirs(reference_models_dir, exist_ok=True)
        
        rmvpe_path = os.path.join(reference_models_dir, "rmvpe.pt")
        
        if not os.path.exists(rmvpe_path):
            print(f"📥 Downloading RMVPE model for reference implementation...")
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