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
        "size": "18GB",
        "files": [
            # Required model files (10 shards)
            {"remote": "model-00001-of-00010.safetensors", "local": "model-00001-of-00010.safetensors"},
            {"remote": "model-00002-of-00010.safetensors", "local": "model-00002-of-00010.safetensors"},
            {"remote": "model-00003-of-00010.safetensors", "local": "model-00003-of-00010.safetensors"},
            {"remote": "model-00004-of-00010.safetensors", "local": "model-00004-of-00010.safetensors"},
            {"remote": "model-00005-of-00010.safetensors", "local": "model-00005-of-00010.safetensors"},
            {"remote": "model-00006-of-00010.safetensors", "local": "model-00006-of-00010.safetensors"},
            {"remote": "model-00007-of-00010.safetensors", "local": "model-00007-of-00010.safetensors"},
            {"remote": "model-00008-of-00010.safetensors", "local": "model-00008-of-00010.safetensors"},
            {"remote": "model-00009-of-00010.safetensors", "local": "model-00009-of-00010.safetensors"},
            {"remote": "model-00010-of-00010.safetensors", "local": "model-00010-of-00010.safetensors"},
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
        Get local path for a VibeVoice model, checking multiple sources.
        Priority: local > legacy > cache > download
        
        Args:
            model_name: Name of the model (e.g., "vibevoice-1.5B")
            
        Returns:
            Path to model directory or None if download failed
        """
        if model_name not in VIBEVOICE_MODELS:
            print(f"❌ Unknown VibeVoice model: {model_name}")
            return None
        
        model_info = VIBEVOICE_MODELS[model_name]
        repo_id = model_info["repo"]
        
        # 1. Check current local path first
        model_dir = os.path.join(self.vibevoice_dir, model_name)
        config_path = os.path.join(model_dir, "config.json")
        
        if os.path.exists(config_path):
            # Verify all required model files exist
            all_files_exist = True
            for file_info in model_info["files"]:
                file_path = os.path.join(model_dir, file_info["local"])
                if not os.path.exists(file_path):
                    print(f"⚠️ Missing local file: {file_info['local']}")
                    all_files_exist = False
                    break
            
            if all_files_exist:
                print(f"📁 Using local VibeVoice model: {model_dir}")
                return model_dir
            else:
                print(f"🔄 Local model incomplete, will re-download: {model_dir}")
        
        # 2. Check legacy VibeVoice-ComfyUI path
        legacy_vibevoice_dir = os.path.join(self.models_dir, "vibevoice")
        legacy_model_dir = os.path.join(legacy_vibevoice_dir, f"models--{repo_id.replace('/', '--')}")
        legacy_config_path = os.path.join(legacy_model_dir, "config.json")
        
        if os.path.exists(legacy_config_path):
            print(f"📁 Using legacy VibeVoice model: {legacy_model_dir}")
            return legacy_model_dir
        
        # 3. Check HuggingFace cache
        try:
            from huggingface_hub import hf_hub_download
            # Try to find config.json in cache (local_files_only=True means cache only)
            cached_config = hf_hub_download(repo_id=repo_id, filename="config.json", local_files_only=True)
            cached_model_dir = os.path.dirname(cached_config)
            print(f"📁 Using cached VibeVoice model: {cached_model_dir}")
            return cached_model_dir
        except Exception as cache_error:
            print(f"📋 Cache check for {model_name}: {str(cache_error)[:100]}... - will download")
        
        # 4. Download model to local directory
        print(f"📥 Downloading VibeVoice model '{model_name}'...")
        
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