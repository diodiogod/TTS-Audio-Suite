"""
VibeVoice Model Downloader - Handles model downloads for VibeVoice TTS
Uses unified downloader to avoid HuggingFace cache duplication
"""

import os
import sys
from typing import Dict, List, Optional
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.downloads.unified_downloader import unified_downloader
import folder_paths

# VibeVoice model configurations
VIBEVOICE_MODELS = {
    "vibevoice-1.5B": {
        "repo": "microsoft/VibeVoice-1.5B",
        "description": "Microsoft VibeVoice 1.5B - Official model (actually 2.7B params)",
        "size": "5.4GB",
        "files": [
            # Required model files
            {"remote": "model-00001-of-00003.safetensors", "local": "model-00001-of-00003.safetensors"},
            {"remote": "model-00002-of-00003.safetensors", "local": "model-00002-of-00003.safetensors"},
            {"remote": "model-00003-of-00003.safetensors", "local": "model-00003-of-00003.safetensors"},
            {"remote": "model.safetensors.index.json", "local": "model.safetensors.index.json"},
            {"remote": "config.json", "local": "config.json"},
            {"remote": "preprocessor_config.json", "local": "preprocessor_config.json"}
            # Note: Tokenizer files not included - VibeVoice uses Qwen tokenizer fallback by design
        ]
    },
    "vibevoice-7B": {
        "repo": "WestZhang/VibeVoice-Large-pt",
        "description": "VibeVoice 7B Preview - Microsoft-endorsed community model (actually 9.3B params)",
        "size": "9.3GB",
        "files": [
            # Required model files
            {"remote": "model-00001-of-00004.safetensors", "local": "model-00001-of-00004.safetensors"},
            {"remote": "model-00002-of-00004.safetensors", "local": "model-00002-of-00004.safetensors"},
            {"remote": "model-00003-of-00004.safetensors", "local": "model-00003-of-00004.safetensors"},
            {"remote": "model-00004-of-00004.safetensors", "local": "model-00004-of-00004.safetensors"},
            {"remote": "model.safetensors.index.json", "local": "model.safetensors.index.json"},
            {"remote": "config.json", "local": "config.json"},
            {"remote": "preprocessor_config.json", "local": "preprocessor_config.json"}
            # Note: Tokenizer files not included - VibeVoice uses Qwen tokenizer fallback by design
        ]
    }
}


class VibeVoiceDownloader:
    """Handles VibeVoice model downloads using unified downloader"""
    
    def __init__(self):
        """Initialize VibeVoice downloader"""
        self.downloader = unified_downloader
        self.models_dir = folder_paths.models_dir
        self.tts_dir = os.path.join(self.models_dir, "TTS")
        self.vibevoice_dir = os.path.join(self.tts_dir, "vibevoice")
        
        # Create directories
        os.makedirs(self.vibevoice_dir, exist_ok=True)
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available VibeVoice models.
        
        Returns:
            List of model names
        """
        available = []
        
        # Check for downloaded models
        for model_key, model_info in VIBEVOICE_MODELS.items():
            model_dir = os.path.join(self.vibevoice_dir, model_key)
            config_path = os.path.join(model_dir, "config.json")
            
            if os.path.exists(config_path):
                available.append(model_key)
        
        # Always include model names for dropdown even if not downloaded
        for model_key in VIBEVOICE_MODELS.keys():
            if model_key not in available:
                available.append(model_key)
        
        return available
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        Get local path for a VibeVoice model, downloading if necessary.
        
        Args:
            model_name: Name of the model (e.g., "vibevoice-1.5B")
            
        Returns:
            Path to model directory or None if download failed
        """
        if model_name not in VIBEVOICE_MODELS:
            print(f"❌ Unknown VibeVoice model: {model_name}")
            return None
        
        model_dir = os.path.join(self.vibevoice_dir, model_name)
        config_path = os.path.join(model_dir, "config.json")
        
        # Check if already downloaded
        if os.path.exists(config_path):
            print(f"✅ VibeVoice model '{model_name}' already downloaded")
            return model_dir
        
        # Download model
        print(f"📥 Downloading VibeVoice model '{model_name}'...")
        model_info = VIBEVOICE_MODELS[model_name]
        
        result = self.downloader.download_huggingface_model(
            repo_id=model_info["repo"],
            model_name=model_name,
            files=model_info["files"],
            engine_type="vibevoice",
            subfolder=None
        )
        
        if result:
            print(f"✅ VibeVoice model '{model_name}' downloaded successfully")
            return result
        else:
            print(f"❌ Failed to download VibeVoice model '{model_name}'")
            return None
    
    def ensure_vibevoice_package(self) -> bool:
        """
        Ensure VibeVoice package is installed.
        
        Returns:
            True if package is available, False otherwise
        """
        try:
            import vibevoice
            print(f"✅ VibeVoice base package found: {vibevoice.__file__}")
            
            # Test specific modules we need
            from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            print("✅ VibeVoiceForConditionalGenerationInference imported successfully")
            
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
            print("✅ VibeVoiceProcessor imported successfully")
            
            return True
        except ImportError as e:
            print(f"❌ VibeVoice package import failed: {e}")
            print("🔄 This should have been installed via the install script")
            print("📦 If the issue persists, try reinstalling the node via ComfyUI Manager")
            print("   or manually: pip install git+https://github.com/microsoft/VibeVoice.git")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get information about a VibeVoice model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model info dict or None
        """
        return VIBEVOICE_MODELS.get(model_name)