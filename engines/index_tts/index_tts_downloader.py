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
from utils.models.extra_paths import get_preferred_download_path
import folder_paths


class IndexTTSDownloader:
    """Downloader for IndexTTS-2 models using unified download system."""
    
    MODELS = {
        "w2v-bert-2.0": {
            "repo_id": "facebook/w2v-bert-2.0",
            "files": [
                "config.json",
                "model.safetensors",
                "preprocessor_config.json"
            ],
            "description": "W2V-BERT semantic feature extractor for IndexTTS-2 (2GB)"
        },
        "MaskGCT": {
            "repo_id": "amphion/MaskGCT",
            "files": [
                "semantic_codec/model.safetensors"
            ],
            "description": "MaskGCT semantic codec for IndexTTS-2"
        },
        "bigvgan_v2_22khz_80band_256x": {
            "repo_id": "nvidia/bigvgan_v2_22khz_80band_256x",
            "files": [
                "config.json",
                "bigvgan_generator.pt"
            ],
            "description": "BigVGAN vocoder for IndexTTS-2"
        },
        "campplus": {
            "repo_id": "funasr/campplus",
            "files": [
                "campplus_cn_common.bin"
            ],
            "description": "CampPlus speaker embedding model for IndexTTS-2"
        },
        "IndexTTS-2": {
            "repo_id": "IndexTeam/IndexTTS-2",
            "files": [
                "config.yaml",
                "feat1.pt",
                "feat2.pt",
                "gpt.pth",
                "s2mel.pth",
                "bpe.model",
                "wav2vec2bert_stats.pt",
                # QwenEmotion model files - individual files instead of wildcard
                "qwen0.6bemo4-merge/Modelfile",
                "qwen0.6bemo4-merge/added_tokens.json",
                "qwen0.6bemo4-merge/chat_template.jinja",
                "qwen0.6bemo4-merge/config.json",
                "qwen0.6bemo4-merge/generation_config.json",
                "qwen0.6bemo4-merge/merges.txt",
                "qwen0.6bemo4-merge/model.safetensors",
                "qwen0.6bemo4-merge/special_tokens_map.json",
                "qwen0.6bemo4-merge/tokenizer.json",
                "qwen0.6bemo4-merge/tokenizer_config.json",
                "qwen0.6bemo4-merge/vocab.json"
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
            # Use extra_model_paths configuration for downloads
            try:
                self.base_path = get_preferred_download_path(model_type='TTS', engine_name='IndexTTS')
            except Exception:
                # Fallback to default if extra_paths fails
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

        # Check if model already exists
        if not force_download and os.path.exists(model_path):
            try:
                self._verify_model(model_path, model_name, verbose=False)
                # Model already exists and is valid, skip download
                return model_path
            except RuntimeError:
                # Model exists but is invalid, proceed with download
                pass

        print(f"📥 Downloading IndexTTS-2 model: {model_name}")
        print(f"📁 Target directory: {model_path}")

        try:
            # Prepare file list for unified downloader
            file_list = []
            for file_pattern in model_info["files"]:
                # All files are now explicit paths
                file_list.append({
                    'remote': file_pattern,
                    'local': file_pattern
                })
            
            # Download model files using unified downloader
            result_path = self.downloader.download_huggingface_model(
                repo_id=model_info["repo_id"],
                model_name=model_name,
                files=file_list,
                engine_type="IndexTTS",
                **kwargs
            )
            
            if not result_path:
                raise RuntimeError("HuggingFace download failed")
            
            # Use the path returned by the unified downloader
            model_path = result_path
            
            # Verify essential files
            self._verify_model(model_path, model_name)
            
            print(f"✅ {model_name} model downloaded successfully")
            print(f"📁 Model path: {model_path}")
            
            return model_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to download IndexTTS-2 model: {e}")
    
    def _verify_model(self, model_path: str, model_name: str = "IndexTTS-2", verbose: bool = True) -> None:
        """
        Verify downloaded model has all required files.

        Args:
            model_path: Path to model directory
            model_name: Model name to get file list
            verbose: Print verification messages (default True)

        Raises:
            RuntimeError: If verification fails
        """
        if model_name not in self.MODELS:
            raise RuntimeError(f"Unknown model: {model_name}")

        model_info = self.MODELS[model_name]
        missing_files = []

        for file in model_info["files"]:
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(file)

        if missing_files:
            raise RuntimeError(
                f"IndexTTS-2 model verification failed. Missing files: {missing_files}"
            )

        if verbose:
            # Check for QwenEmotion model if emotion text is supported
            qwen_dir = os.path.join(model_path, "qwen0.6bemo4-merge")
            if os.path.exists(qwen_dir) and os.listdir(qwen_dir):
                print(f"✅ QwenEmotion model found - text emotion support available")
            else:
                print(f"ℹ️ QwenEmotion model not found - audio emotion only")

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
            self._verify_model(model_path, model_name)
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