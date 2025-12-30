"""
CosyVoice3 Model Downloader

Handles automatic download and setup of Fun-CosyVoice3-0.5B models using the unified download system.
Downloads models to organized TTS/CosyVoice/ structure.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.downloads.unified_downloader import unified_downloader
from utils.models.extra_paths import get_preferred_download_path
import folder_paths


class CosyVoiceDownloader:
    """Downloader for CosyVoice3 models using unified download system."""
    
    # Shared files across all variants (downloaded once)
    SHARED_FILES = [
        "cosyvoice3.yaml",
        "campplus.onnx",
        "flow.pt",
        "hift.pt",
        "speech_tokenizer_v3.onnx",
        # CosyVoice-BlankEN subfolder (Qwen model for text processing)
        "CosyVoice-BlankEN/config.json",
        "CosyVoice-BlankEN/generation_config.json",
        "CosyVoice-BlankEN/merges.txt",
        "CosyVoice-BlankEN/model.safetensors",
        "CosyVoice-BlankEN/tokenizer_config.json",
        "CosyVoice-BlankEN/vocab.json",
    ]

    MODELS = {
        "Fun-CosyVoice3-0.5B": {
            "repo_id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            "llm_file": "llm.pt",  # Standard LLM model (2GB)
            "description": "Fun-CosyVoice3 0.5B base model (~5.4GB total)"
        },
        "Fun-CosyVoice3-0.5B-RL": {
            "repo_id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            "llm_file": "llm.rl.pt",  # RL-enhanced LLM model (2GB, better quality)
            "description": "Fun-CosyVoice3 0.5B RL-enhanced model (better quality) (~5.4GB total)"
        }
    }
    
    # Supported languages
    SUPPORTED_LANGUAGES = [
        "chinese", "english", "japanese", "korean", 
        "german", "spanish", "french", "italian", "russian"
    ]
    
    # Supported Chinese dialects
    SUPPORTED_DIALECTS = [
        "guangdong", "minnan", "sichuan", "dongbei", "shan3xi", "shan1xi",
        "shanghai", "tianjin", "shandong", "ningxia", "gansu"
    ]
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize downloader.

        Args:
            base_path: Base directory for CosyVoice models (auto-detected if None)
        """
        if base_path is None:
            # Use extra_model_paths configuration for downloads
            try:
                self.base_path = get_preferred_download_path(model_type='TTS', engine_name='CosyVoice')
            except Exception:
                # Fallback to default if extra_paths fails
                self.base_path = os.path.join(folder_paths.models_dir, "TTS", "CosyVoice")
        else:
            self.base_path = base_path

        self.downloader = unified_downloader
        
    def download_model(self,
                      model_name: str = "Fun-CosyVoice3-0.5B",
                      force_download: bool = False,
                      **kwargs) -> str:
        """
        Download CosyVoice3 model.

        Args:
            model_name: Model to download ("Fun-CosyVoice3-0.5B" or "Fun-CosyVoice3-0.5B-RL")
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

        # Both variants use the same folder (like ChatterBox 23-Lang v1/v2)
        # They share all files except the LLM file (llm.pt vs llm.rl.pt)
        folder_name = "Fun-CosyVoice3-0.5B"
        model_path = os.path.join(self.base_path, folder_name)

        # Check if model already exists
        if not force_download and os.path.exists(model_path):
            try:
                self._verify_model(model_path, model_name, verbose=False)
                # Model already exists and is valid, skip download
                return model_path
            except RuntimeError:
                # Model exists but is invalid, proceed with download
                pass

        print(f"📥 Downloading CosyVoice3 model: {model_name}")
        print(f"📁 Target directory: {model_path}")

        try:
            # Prepare file list: shared files + variant-specific LLM
            file_list = []

            # Add shared files
            for file_pattern in self.SHARED_FILES:
                file_list.append({
                    'remote': file_pattern,
                    'local': file_pattern
                })

            # Add variant-specific LLM file
            llm_file = model_info["llm_file"]
            file_list.append({
                'remote': llm_file,
                'local': llm_file  # Keep original name (llm.pt or llm.rl.pt)
            })

            # Download model files using unified downloader
            result_path = self.downloader.download_huggingface_model(
                repo_id=model_info["repo_id"],
                model_name=folder_name,
                files=file_list,
                engine_type="CosyVoice",
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
            raise RuntimeError(f"Failed to download CosyVoice3 model: {e}")
    
    def _verify_model(self, model_path: str, model_name: str = "Fun-CosyVoice3-0.5B", verbose: bool = True) -> None:
        """
        Verify downloaded model has all required files.

        Args:
            model_path: Path to model directory
            model_name: Model name to get LLM file requirement
            verbose: Print verification messages (default True)

        Raises:
            RuntimeError: If verification fails
        """
        # Essential shared files (common to all variants)
        missing_files = []
        for file in self.SHARED_FILES:
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(file)

        # Check for variant-specific LLM file
        if model_name in self.MODELS:
            required_llm = self.MODELS[model_name]["llm_file"]
            llm_path = os.path.join(model_path, required_llm)
            if not os.path.exists(llm_path):
                missing_files.append(required_llm)

        if missing_files:
            raise RuntimeError(
                f"CosyVoice3 model verification failed. Missing files: {missing_files}"
            )

        if verbose:
            print(f"✅ Model verification passed")
    
    def is_model_available(self, model_name: str = "Fun-CosyVoice3-0.5B-RL") -> bool:
        """
        Check if model is already downloaded.

        Args:
            model_name: Model name to check

        Returns:
            True if model is available locally
        """
        # Both variants use the same folder
        folder_name = "Fun-CosyVoice3-0.5B"
        model_path = os.path.join(self.base_path, folder_name)
        
        try:
            self._verify_model(model_path, model_name, verbose=False)
            return True
        except RuntimeError:
            return False
    
    def get_model_info(self, model_name: str = "Fun-CosyVoice3-0.5B") -> Optional[Dict[str, Any]]:
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
    
    def get_model_path(self, model_name: str = "Fun-CosyVoice3-0.5B-RL") -> str:
        """
        Get the local path for a model.

        Args:
            model_name: Model name (variant doesn't matter, both use same folder)

        Returns:
            Local model path
        """
        # Both variants use the same folder
        folder_name = "Fun-CosyVoice3-0.5B"
        return os.path.join(self.base_path, folder_name)


# Global downloader instance
cosyvoice_downloader = CosyVoiceDownloader()


def download_cosyvoice_model(model_name: str = "Fun-CosyVoice3-0.5B-RL",
                            force_download: bool = False,
                            **kwargs) -> str:
    """
    Convenience function to download CosyVoice3 model.
    
    Args:
        model_name: Model to download
        force_download: Force re-download
        **kwargs: Additional options
        
    Returns:
        Path to downloaded model
    """
    return cosyvoice_downloader.download_model(
        model_name=model_name,
        force_download=force_download, 
        **kwargs
    )


def is_cosyvoice_available(model_name: str = "Fun-CosyVoice3-0.5B-RL") -> bool:
    """Check if CosyVoice3 model is available locally."""
    return cosyvoice_downloader.is_model_available(model_name)
