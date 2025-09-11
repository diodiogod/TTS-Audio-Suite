"""
IndexTTS-2 Model Downloader

Handles automatic download and setup of IndexTTS-2 models using the unified download system.
Downloads models to organized TTS/IndexTTS/ structure.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.downloads.unified_downloader import unified_downloader
import folder_paths


class IndexTTSDownloader:
    """Downloader for IndexTTS-2 models using unified download system."""
    
    MODELS = {
        "IndexTTS-2": {
            "repo_id": "IndexTeam/IndexTTS-2",
            "files": [
                "config.yaml",
                "emo_matrix.pt", 
                "spk_matrix.pt",
                "gpt_checkpoint.pt",
                "s2mel_checkpoint.pt",
                "bpe_model.model",
                "w2v_stat.pt",
                "qwen_emo/**"  # QwenEmotion model directory
            ],
            "description": "IndexTTS-2 main model with emotion control"
        }
    }
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize downloader.
        
        Args:
            base_path: Base directory for IndexTTS-2 models (auto-detected if None)
        """
        if base_path is None:
            self.base_path = os.path.join(folder_paths.models_dir, "TTS", "IndexTTS")
        else:
            self.base_path = base_path
        
        self.downloader = unified_downloader
        
    def download_model(self, 
                      model_name: str = "IndexTTS-2",
                      force_download: bool = False,
                      **kwargs) -> str:
        """
        Download IndexTTS-2 model.
        
        Args:
            model_name: Model to download ("IndexTTS-2")
            force_download: Force re-download even if exists
            **kwargs: Additional download options
            
        Returns:
            Path to downloaded model directory
            
        Raises:
            ValueError: If model_name is not supported
            RuntimeError: If download fails
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
        
        model_info = self.MODELS[model_name]
        model_path = os.path.join(self.base_path, model_name)
        
        print(f"📥 Downloading IndexTTS-2 model: {model_name}")
        print(f"📁 Target directory: {model_path}")
        
        try:
            # Download model files
            self.downloader.download_from_hf(
                repo_id=model_info["repo_id"],
                local_dir=model_path,
                engine_name="index_tts",
                files=model_info["files"],
                force_download=force_download,
                **kwargs
            )
            
            # Download IndexTTS-2 code if not present
            code_path = os.path.join(model_path, "index-tts")
            if not os.path.exists(code_path) or force_download:
                print(f"📥 Downloading IndexTTS-2 source code...")
                self.downloader.download_from_git(
                    repo_url="https://github.com/index-tts/index-tts.git",
                    local_dir=code_path,
                    branch="main",
                    force_download=force_download
                )
                
            # Verify essential files
            self._verify_model(model_path)
            
            print(f"✅ IndexTTS-2 model downloaded successfully")
            print(f"📁 Model path: {model_path}")
            
            return model_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to download IndexTTS-2 model: {e}")
    
    def _verify_model(self, model_path: str) -> None:
        """
        Verify downloaded model has essential files.
        
        Args:
            model_path: Path to model directory
            
        Raises:
            RuntimeError: If verification fails
        """
        required_files = [
            "config.yaml",
            "gpt_checkpoint.pt",
            "s2mel_checkpoint.pt", 
            "bpe_model.model"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(file)
                
        if missing_files:
            raise RuntimeError(
                f"IndexTTS-2 model verification failed. Missing files: {missing_files}"
            )
            
        # Verify code directory
        code_path = os.path.join(model_path, "index-tts")
        indextts_path = os.path.join(code_path, "indextts")
        if not os.path.exists(code_path) or not os.path.exists(indextts_path):
            raise RuntimeError(
                f"IndexTTS-2 source code not found at {code_path}"
            )
            
        print(f"✅ Model verification passed")
    
    def is_model_available(self, model_name: str = "IndexTTS-2") -> bool:
        """
        Check if model is already downloaded.
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if model is available locally
        """
        if model_name not in self.MODELS:
            return False
            
        model_path = os.path.join(self.base_path, model_name)
        
        try:
            self._verify_model(model_path)
            return True
        except RuntimeError:
            return False
    
    def get_model_info(self, model_name: str = "IndexTTS-2") -> Optional[Dict[str, Any]]:
        """
        Get information about a model.
        
        Args:
            model_name: Model name
            
        Returns:
            Model information dictionary or None if not found
        """
        return self.MODELS.get(model_name)
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of all available models."""
        return self.MODELS.copy()
    
    def get_model_path(self, model_name: str = "IndexTTS-2") -> str:
        """
        Get the local path for a model.
        
        Args:
            model_name: Model name
            
        Returns:
            Local model path
        """
        return os.path.join(self.base_path, model_name)


# Global downloader instance
index_tts_downloader = IndexTTSDownloader()


def download_index_tts_model(model_name: str = "IndexTTS-2", 
                            force_download: bool = False,
                            **kwargs) -> str:
    """
    Convenience function to download IndexTTS-2 model.
    
    Args:
        model_name: Model to download
        force_download: Force re-download
        **kwargs: Additional options
        
    Returns:
        Path to downloaded model
    """
    return index_tts_downloader.download_model(
        model_name=model_name,
        force_download=force_download, 
        **kwargs
    )


def is_index_tts_available(model_name: str = "IndexTTS-2") -> bool:
    """Check if IndexTTS-2 model is available locally."""
    return index_tts_downloader.is_model_available(model_name)