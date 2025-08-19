"""
Higgs Audio 2 Model Downloader - Handles model downloads using unified downloader system
Downloads Higgs Audio models to organized TTS/HiggsAudio/ structure
"""

import os
import sys
from typing import Optional, Dict, List, Tuple
from pathlib import Path

# Add parent directory for imports
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.downloads.unified_downloader import unified_downloader
import folder_paths

# Higgs Audio model configurations
HIGGS_AUDIO_MODELS = {
    "higgs-audio-v2-3B": {
        "generation_repo": "bosonai/higgs-audio-v2-generation-3B-base",
        "tokenizer_repo": "bosonai/higgs-audio-v2-tokenizer",
        "description": "Higgs Audio v2 3B parameter model with audio tokenizer",
        "generation_files": [
            {"remote": "config.json", "local": "config.json"},
            {"remote": "model.safetensors", "local": "model.safetensors"},
            {"remote": "generation_config.json", "local": "generation_config.json"},
        ],
        "tokenizer_files": [
            {"remote": "config.json", "local": "config.json"},
            {"remote": "model.pth", "local": "model.pth"},
        ]
    }
}


class HiggsAudioDownloader:
    """
    Higgs Audio model downloader using unified downloader system
    Downloads models to organized TTS/HiggsAudio/ structure
    """
    
    def __init__(self):
        """Initialize Higgs Audio downloader"""
        self.downloader = unified_downloader
        self.base_path = os.path.join(folder_paths.models_dir, "TTS", "HiggsAudio")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Higgs Audio models
        
        Returns:
            List of model names
        """
        models = list(HIGGS_AUDIO_MODELS.keys())
        
        # Check for local models in organized directory
        try:
            if os.path.exists(self.base_path):
                for item in os.listdir(self.base_path):
                    item_path = os.path.join(self.base_path, item)
                    if os.path.isdir(item_path):
                        # Check if it has both generation and tokenizer subdirs
                        gen_path = os.path.join(item_path, "generation")
                        tok_path = os.path.join(item_path, "tokenizer")
                        if os.path.exists(gen_path) and os.path.exists(tok_path):
                            local_model = f"local:{item}"
                            if local_model not in models:
                                models.append(local_model)
        except Exception:
            pass  # Ignore errors in model discovery
        
        return models
    
    def download_model(self, model_name_or_path: str) -> str:
        """
        Download or locate Higgs Audio generation model
        
        Args:
            model_name_or_path: Model name or HuggingFace repo ID or local path
            
        Returns:
            Path to model directory or original path if already local
        """
        # If it's already a local path, return as is
        if os.path.exists(model_name_or_path):
            print(f"📁 Using local generation model: {model_name_or_path}")
            return model_name_or_path
        
        # Handle local: prefix
        if model_name_or_path.startswith("local:"):
            local_name = model_name_or_path[6:]  # Remove "local:" prefix
            local_path = os.path.join(self.base_path, local_name, "generation")
            if os.path.exists(local_path):
                print(f"📁 Using local generation model: {local_path}")
                return local_path
            else:
                raise FileNotFoundError(f"Local model not found: {local_path}")
        
        # Handle predefined models
        if model_name_or_path in HIGGS_AUDIO_MODELS:
            model_config = HIGGS_AUDIO_MODELS[model_name_or_path]
            repo_id = model_config["generation_repo"]
            files = model_config["generation_files"]
            
            # Check if already downloaded
            model_dir = os.path.join(self.base_path, model_name_or_path, "generation")
            if self._check_model_files_exist(model_dir, [f["local"] for f in files]):
                print(f"📁 Generation model already exists: {model_dir}")
                return model_dir
            
            # Download model
            print(f"📥 Downloading Higgs Audio generation model: {model_name_or_path}")
            downloaded_dir = self.downloader.download_huggingface_model(
                repo_id=repo_id,
                model_name=model_name_or_path,
                files=files,
                engine_type="HiggsAudio",
                subfolder="generation"
            )
            
            if downloaded_dir:
                print(f"✅ Generation model downloaded: {downloaded_dir}")
                return downloaded_dir
            else:
                raise RuntimeError(f"Failed to download generation model: {model_name_or_path}")
        
        # Handle direct HuggingFace repo IDs
        print(f"📥 Using HuggingFace model directly: {model_name_or_path}")
        return model_name_or_path
    
    def download_tokenizer(self, tokenizer_name_or_path: str) -> str:
        """
        Download or locate Higgs Audio tokenizer model
        
        Args:
            tokenizer_name_or_path: Tokenizer name or HuggingFace repo ID or local path
            
        Returns:
            Path to tokenizer directory or original path if already local
        """
        # If it's already a local path, return as is
        if os.path.exists(tokenizer_name_or_path):
            print(f"📁 Using local tokenizer model: {tokenizer_name_or_path}")
            return tokenizer_name_or_path
        
        # Handle local: prefix
        if tokenizer_name_or_path.startswith("local:"):
            local_name = tokenizer_name_or_path[6:]  # Remove "local:" prefix
            local_path = os.path.join(self.base_path, local_name, "tokenizer")
            if os.path.exists(local_path):
                print(f"📁 Using local tokenizer model: {local_path}")
                return local_path
            else:
                raise FileNotFoundError(f"Local tokenizer not found: {local_path}")
        
        # Handle predefined models
        model_name = None
        for name, config in HIGGS_AUDIO_MODELS.items():
            if config["tokenizer_repo"] == tokenizer_name_or_path:
                model_name = name
                break
        
        if model_name and model_name in HIGGS_AUDIO_MODELS:
            model_config = HIGGS_AUDIO_MODELS[model_name]
            repo_id = model_config["tokenizer_repo"]
            files = model_config["tokenizer_files"]
            
            # Check if already downloaded
            tokenizer_dir = os.path.join(self.base_path, model_name, "tokenizer")
            if self._check_model_files_exist(tokenizer_dir, [f["local"] for f in files]):
                print(f"📁 Tokenizer model already exists: {tokenizer_dir}")
                return tokenizer_dir
            
            # Download tokenizer
            print(f"📥 Downloading Higgs Audio tokenizer model: {model_name}")
            downloaded_dir = self.downloader.download_huggingface_model(
                repo_id=repo_id,
                model_name=model_name,
                files=files,
                engine_type="HiggsAudio",
                subfolder="tokenizer"
            )
            
            if downloaded_dir:
                print(f"✅ Tokenizer model downloaded: {downloaded_dir}")
                return downloaded_dir
            else:
                raise RuntimeError(f"Failed to download tokenizer model: {model_name}")
        
        # Handle direct HuggingFace repo IDs
        print(f"📥 Using HuggingFace tokenizer directly: {tokenizer_name_or_path}")
        return tokenizer_name_or_path
    
    def download_model_pair(self, model_name: str) -> Tuple[str, str]:
        """
        Download both generation and tokenizer models for a predefined model
        
        Args:
            model_name: Name of predefined model (e.g., "higgs-audio-v2-3B")
            
        Returns:
            Tuple of (generation_path, tokenizer_path)
        """
        if model_name not in HIGGS_AUDIO_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(HIGGS_AUDIO_MODELS.keys())}")
        
        model_config = HIGGS_AUDIO_MODELS[model_name]
        
        # Download generation model
        generation_path = self.download_model(model_config["generation_repo"])
        
        # Download tokenizer model
        tokenizer_path = self.download_tokenizer(model_config["tokenizer_repo"])
        
        return generation_path, tokenizer_path
    
    def download_voice_presets(self) -> bool:
        """
        Download voice preset files if they don't exist
        
        Returns:
            True if successful or already exist, False otherwise
        """
        voices_dir = os.path.join(project_root, "voices_examples", "higgs_audio")
        config_path = os.path.join(voices_dir, "config.json")
        
        # Check if voice presets already exist
        if os.path.exists(config_path):
            print(f"📁 Voice presets already exist: {voices_dir}")
            return True
        
        # Voice presets should have been copied during installation
        # This is mainly a fallback check
        print(f"⚠️ Voice presets not found at {voices_dir}")
        print("Voice presets should be included with the extension installation")
        return False
    
    def _check_model_files_exist(self, model_dir: str, required_files: List[str]) -> bool:
        """
        Check if all required model files exist in directory
        
        Args:
            model_dir: Directory to check
            required_files: List of required filenames
            
        Returns:
            True if all files exist, False otherwise
        """
        if not os.path.exists(model_dir):
            return False
        
        try:
            existing_files = os.listdir(model_dir)
            for required_file in required_files:
                if required_file not in existing_files:
                    return False
            return True
        except Exception:
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get information about a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dict or None if not found
        """
        if model_name in HIGGS_AUDIO_MODELS:
            return HIGGS_AUDIO_MODELS[model_name].copy()
        
        # Check if it's a local model
        if model_name.startswith("local:"):
            local_name = model_name[6:]
            local_path = os.path.join(self.base_path, local_name)
            if os.path.exists(local_path):
                return {
                    "description": f"Local Higgs Audio model: {local_name}",
                    "generation_repo": os.path.join(local_path, "generation"),
                    "tokenizer_repo": os.path.join(local_path, "tokenizer"),
                    "local": True
                }
        
        return None
    
    def cleanup_downloads(self, model_name: str) -> bool:
        """
        Clean up downloaded model files
        
        Args:
            model_name: Name of model to clean up
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_dir = os.path.join(self.base_path, model_name)
            if os.path.exists(model_dir):
                import shutil
                shutil.rmtree(model_dir)
                print(f"🗑️ Cleaned up model: {model_dir}")
                return True
        except Exception as e:
            print(f"❌ Failed to cleanup model {model_name}: {e}")
        
        return False