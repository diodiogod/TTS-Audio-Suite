"""
Unified Download System for TTS Audio Suite
Centralized downloading for all models (F5-TTS, ChatterBox, RVC, etc.) without cache duplication
"""

import os
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path
import folder_paths

class UnifiedDownloader:
    """
    Centralized downloader that handles all model downloads directly to organized TTS/ folder structure
    without using HuggingFace cache to avoid duplication.
    """
    
    def __init__(self):
        self.models_dir = folder_paths.models_dir
        self.tts_dir = os.path.join(self.models_dir, "TTS")
    
    def download_file(self, url: str, target_path: str, description: str = None) -> bool:
        """
        Download a file directly to target path with progress display.
        
        Args:
            url: Direct download URL
            target_path: Full local path where file should be saved
            description: Optional description for progress display
            
        Returns:
            True if successful, False otherwise
        """
        if os.path.exists(target_path):
            print(f"📁 File already exists: {os.path.basename(target_path)}")
            return True
            
        try:
            # Create directory structure
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Download with progress
            desc = description or os.path.basename(target_path)
            print(f"📥 Downloading {desc} directly (no cache)")
            
            response = requests.get(url, stream=True)
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
            
            print(f"\n✅ Downloaded: {target_path}")
            return True
            
        except Exception as e:
            print(f"\n❌ Download failed for {desc}: {e}")
            # Clean up partial download
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except:
                    pass
            return False
    
    def download_huggingface_model(self, repo_id: str, model_name: str, files: List[Dict[str, str]], 
                                 engine_type: str) -> Optional[str]:
        """
        Download HuggingFace model files to organized TTS/ structure.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "SWivid/F5-TTS")
            model_name: Model name for folder organization (e.g., "F5TTS_v1_Base")
            files: List of files to download, each dict with 'remote' and 'local' keys
            engine_type: Engine type for organization ("F5-TTS", "chatterbox", etc.)
            
        Returns:
            Path to model directory if successful, None otherwise
        """
        # Create organized path
        model_dir = os.path.join(self.tts_dir, engine_type, model_name)
        
        success = True
        for file_info in files:
            remote_path = file_info['remote']  # e.g., "F5TTS_v1_Base/model_1250000.safetensors"
            local_filename = file_info['local']  # e.g., "model_1250000.safetensors"
            
            url = f"https://huggingface.co/{repo_id}/resolve/main/{remote_path}"
            target_path = os.path.join(model_dir, local_filename)
            
            if not self.download_file(url, target_path, f"{model_name}/{local_filename}"):
                success = False
                break
        
        return model_dir if success else None
    
    def download_direct_url_model(self, base_url: str, model_path: str, target_dir: str) -> bool:
        """
        Download model from direct URL to target directory.
        
        Args:
            base_url: Base URL for downloads
            model_path: Relative path of model (e.g., "RVC/Claire.pth")
            target_dir: Target directory in TTS/ structure
            
        Returns:
            True if successful, False otherwise
        """
        url = f"{base_url}{model_path}"
        filename = os.path.basename(model_path)
        target_path = os.path.join(self.tts_dir, target_dir, filename)
        
        return self.download_file(url, target_path, f"{target_dir}/{filename}")
    
    def get_organized_path(self, engine_type: str, model_name: str = None) -> str:
        """
        Get the organized path for a model.
        
        Args:
            engine_type: Engine type ("F5-TTS", "chatterbox", "RVC", "UVR")
            model_name: Optional model name for subfolder
            
        Returns:
            Full path to organized model directory
        """
        if model_name:
            return os.path.join(self.tts_dir, engine_type, model_name)
        else:
            return os.path.join(self.tts_dir, engine_type)
    
    def check_legacy_location(self, engine_type: str, model_name: str = None) -> Optional[str]:
        """
        Check if model exists in legacy location for backward compatibility.
        
        Args:
            engine_type: Engine type
            model_name: Optional model name
            
        Returns:
            Path if found in legacy location, None otherwise
        """
        legacy_paths = []
        
        if engine_type == "F5-TTS":
            legacy_paths = [
                os.path.join(self.models_dir, "F5-TTS", model_name or ""),
                os.path.join(self.models_dir, "Checkpoints", "F5-TTS", model_name or "")
            ]
        elif engine_type == "chatterbox":
            legacy_paths = [
                os.path.join(self.models_dir, "chatterbox", model_name or "")
            ]
        elif engine_type == "RVC":
            legacy_paths = [
                os.path.join(self.models_dir, "RVC", model_name or "")
            ]
        elif engine_type == "UVR":
            legacy_paths = [
                os.path.join(self.models_dir, "UVR", model_name or "")
            ]
        
        for path in legacy_paths:
            if os.path.exists(path):
                return path
        
        return None

# Global instance for easy access
unified_downloader = UnifiedDownloader()