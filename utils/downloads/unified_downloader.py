"""
Unified Download System for TTS Audio Suite
Centralized downloading for all models (F5-TTS, ChatterBox, RVC, etc.) without cache duplication
"""

import os
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path
import folder_paths

# Import extra paths support
from utils.models.extra_paths import get_preferred_download_path, find_model_in_paths

class UnifiedDownloader:
    """
    Centralized downloader that handles all model downloads directly to organized TTS/ folder structure
    without using HuggingFace cache to avoid duplication.
    """
    
    def __init__(self):
        # Use extra_model_paths.yaml aware directory resolution
        self.tts_dir = get_preferred_download_path('TTS')
        # Keep backward compatibility
        self.models_dir = folder_paths.models_dir
        # SSL verification bypass for environments with certificate issues
        self.verify_ssl = os.environ.get('CURL_CA_BUNDLE') is not None or os.environ.get('REQUESTS_CA_BUNDLE') is not None
        if not self.verify_ssl:
            # Check if user explicitly disabled SSL verification
            self.verify_ssl = os.environ.get('TTS_DISABLE_SSL_VERIFY', '0') != '1'

        if not self.verify_ssl:
            print("⚠️ SSL certificate verification is DISABLED")
            print("   Set environment variable TTS_DISABLE_SSL_VERIFY=0 to re-enable")
    
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

            # Apply SSL verification setting
            response = requests.get(url, stream=True, verify=self.verify_ssl)
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
            
            # Re-raise authentication errors so they can be handled upstream
            error_str = str(e)
            if "401" in error_str or "Unauthorized" in error_str:
                raise RuntimeError(f"401 Unauthorized: Authentication required for {desc} - {e}")
            
            return False
    
    def download_huggingface_model(self, repo_id: str, model_name: str, files: List[Dict[str, str]], 
                                 engine_type: str, subfolder: str = None) -> Optional[str]:
        """
        Download HuggingFace model files to organized TTS/ structure.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "SWivid/F5-TTS")
            model_name: Model name for folder organization (e.g., "F5TTS_v1_Base")
            files: List of files to download, each dict with 'remote' and 'local' keys
            engine_type: Engine type for organization ("F5-TTS", "chatterbox", etc.)
            subfolder: Optional subfolder for organization (e.g., "vocos")
            
        Returns:
            Path to model directory if successful, None otherwise
        """
        # Create organized path with optional subfolder
        if subfolder:
            model_dir = os.path.join(self.tts_dir, engine_type, model_name, subfolder)
        else:
            model_dir = os.path.join(self.tts_dir, engine_type, model_name)
        
        success = True
        critical_files = ['config.json']  # These files are absolutely required
        failed_files = []
        
        for file_info in files:
            remote_path = file_info['remote']  # e.g., "F5TTS_v1_Base/model_1250000.safetensors"
            local_filename = file_info['local']  # e.g., "model_1250000.safetensors"
            
            target_path = os.path.join(model_dir, local_filename)
            
            # Skip if already exists in TTS folder
            if os.path.exists(target_path):
                continue
            
            # Note: Don't copy from cache - just download to TTS folder if missing
            
            # Download from HuggingFace if not in cache
            url = f"https://huggingface.co/{repo_id}/resolve/main/{remote_path}"
            if not self.download_file(url, target_path, f"{model_name}/{local_filename}"):
                failed_files.append(local_filename)
                # Only fail completely if critical files are missing
                if local_filename in critical_files:
                    success = False
                    break
        
        if failed_files:
            print(f"⚠️ Some files failed to download: {failed_files}")
            
        # For sharded models, we need either all shards OR none (to use cache fallback)
        if any('model-' in f and '.safetensors' in f for f in failed_files):
            print("❌ Sharded model files incomplete, using cache fallback")
            success = False
        
        return model_dir if success else None
    
    def download_chatterbox_model(self, repo_id: str, model_name: str, subdirectory: str = None, 
                                files: List[str] = None) -> Optional[str]:
        """
        Download ChatterBox model with optional subdirectory support.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "niobures/Chatterbox-TTS")
            model_name: Model name for folder organization (e.g., "Russian")
            subdirectory: Optional subdirectory in repo (e.g., "ru")
            files: Optional list of specific files to download
            
        Returns:
            Path to model directory if successful, None otherwise
        """
        # Default ChatterBox files
        if files is None:
            files = [
                "t3_cfg.safetensors",
                "s3gen.safetensors", 
                "ve.safetensors",
                "conds.pt",
                "tokenizer.json"
            ]
        
        # Create target directory - use different paths for different ChatterBox variants
        if "Official 23-Lang" in model_name:
            model_dir = os.path.join(self.tts_dir, "chatterbox_official_23lang", model_name)
        else:
            model_dir = os.path.join(self.tts_dir, "chatterbox", model_name)
        
        success = True
        # Different critical files for Official 23-Lang model
        if "Official 23-Lang" in model_name:
            # Check if downloading v2 model (t3_mtl23ls_v2.safetensors)
            is_v2 = "t3_mtl23ls_v2.safetensors" in files
            if is_v2:
                critical_files = ["t3_mtl23ls_v2.safetensors", "mtl_tokenizer.json"]  # v2 requirements
            else:
                critical_files = ["t3_23lang.safetensors", "mtl_tokenizer.json"]  # v1 requirements
        else:
            critical_files = ["t3_cfg.safetensors", "tokenizer.json"]  # Standard ChatterBox requirements
        failed_files = []
        
        for filename in files:
            # Build URL with optional subdirectory
            if subdirectory:
                url = f"https://huggingface.co/{repo_id}/resolve/main/{subdirectory}/{filename}"
            else:
                url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            
            target_path = os.path.join(model_dir, filename)
            
            if not self.download_file(url, target_path, f"{model_name}/{filename}"):
                failed_files.append(filename)
                # Only fail completely if critical files are missing
                if filename in critical_files:
                    success = False
                    break
        
        if failed_files:
            print(f"⚠️ Some files failed to download for {model_name}: {failed_files}")
            # For incomplete models (Japanese, Korean), missing optional files is okay
            if not any(f in critical_files for f in failed_files):
                print(f"ℹ️ {model_name} model will use shared English components for missing files")
        
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
        Get the organized path for a model, respecting extra_model_paths.yaml.
        
        Args:
            engine_type: Engine type ("F5-TTS", "chatterbox", "RVC", "UVR")
            model_name: Optional model name for subfolder
            
        Returns:
            Full path to organized model directory
        """
        # Use extra_model_paths.yaml aware directory resolution
        base_path = get_preferred_download_path('TTS', engine_type.lower())
        
        if model_name:
            return os.path.join(base_path, model_name)
        else:
            return base_path
    
    def check_existing_model(self, engine_type: str, model_name: str = None) -> Optional[str]:
        """
        Check if model exists in any configured path (extra_model_paths.yaml aware).
        
        Args:
            engine_type: Engine type
            model_name: Optional model name
            
        Returns:
            Path if found in any configured location, None otherwise
        """
        if not model_name:
            return None
        
        # First, try to find in any configured TTS paths using extra_paths system
        search_subdirs = [engine_type.lower(), engine_type]
        existing_path = find_model_in_paths(model_name, 'TTS', search_subdirs)
        if existing_path:
            return existing_path
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
        elif engine_type == "chatterbox_official_23lang":
            legacy_paths = [
                os.path.join(self.models_dir, "chatterbox_official_23lang", model_name or "")
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
    
    def download_vocos_model(self) -> Optional[str]:
        """
        Download Vocos vocoder model to organized TTS/ structure.
        
        Returns:
            Path to vocos directory if successful, None otherwise
        """
        # Check if vocos already exists in organized location
        vocos_dir = os.path.join(self.tts_dir, "F5-TTS", "vocos")
        config_path = os.path.join(vocos_dir, "config.yaml")
        model_path = os.path.join(vocos_dir, "pytorch_model.bin")
        
        if os.path.exists(config_path) and os.path.exists(model_path):
            # Vocos already available locally
            return vocos_dir
        
        # Download Vocos files
        vocos_files = [
            {'remote': 'config.yaml', 'local': 'config.yaml'},
            {'remote': 'pytorch_model.bin', 'local': 'pytorch_model.bin'}
        ]
        
        print("📥 Downloading Vocos vocoder to organized directory (no cache)")
        downloaded_dir = self.download_huggingface_model(
            repo_id="charactr/vocos-mel-24khz",
            model_name="vocos",
            files=vocos_files,
            engine_type="F5-TTS"
        )
        
        return downloaded_dir
    
    def _check_huggingface_cache(self, repo_id: str, file_path: str) -> Optional[str]:
        """
        Check if file exists in HuggingFace cache.
        
        Args:
            repo_id: Repository ID (e.g., "facebook/w2v-bert-2.0")
            file_path: File path in repo (e.g., "model.safetensors")
            
        Returns:
            Path to cached file if exists, None otherwise
        """
        import os
        import glob
        
        # Get HuggingFace cache directory
        cache_home = os.environ.get('HUGGINGFACE_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
        
        # Convert repo ID to cache format
        cache_folder_name = f"models--{repo_id.replace('/', '--')}"
        cache_repo_dir = os.path.join(cache_home, cache_folder_name)
        
        if not os.path.exists(cache_repo_dir):
            return None
        
        # Find snapshot directories (they have commit hashes as names)
        snapshot_pattern = os.path.join(cache_repo_dir, "snapshots", "*", file_path)
        matching_files = glob.glob(snapshot_pattern)
        
        if matching_files:
            # Return the most recent one (by modification time)
            return max(matching_files, key=os.path.getmtime)
        
        return None

# Global instance for easy access
unified_downloader = UnifiedDownloader()