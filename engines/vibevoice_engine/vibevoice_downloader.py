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
        "type": "standard",
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
        "repo": "microsoft/VibeVoice-Large",
        "description": "Microsoft VibeVoice Large - Official 7B model (actually 9.3B params)",
        "size": "18GB",
        "type": "standard",
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
    },
    # DevParker's quantized models - drop-in replacements with VRAM savings
    "vibevoice-7B-4bit": {
        "repo": "DevParker/VibeVoice7b-low-vram",
        "description": "VibeVoice 7B (4-bit quantized) - 8GB VRAM instead of 19GB",
        "size": "~6GB",
        "type": "quantized",
        "subfolder": "4bit",
        "files": [
            {"remote": "4bit/config.json", "local": "config.json"},
            {"remote": "4bit/preprocessor_config.json", "local": "preprocessor_config.json"},
            {"remote": "4bit/model-00001-of-00002.safetensors", "local": "model-00001-of-00002.safetensors"},
            {"remote": "4bit/model-00002-of-00002.safetensors", "local": "model-00002-of-00002.safetensors"},
            {"remote": "4bit/model.safetensors.index.json", "local": "model.safetensors.index.json"},
            {"remote": "4bit/quantization_config.json", "local": "quantization_config.json"},
            {"remote": "4bit/generation_config.json", "local": "generation_config.json"}
        ]
    },
    # Note: 8bit model removed - DevParker only has 4bit version
    "vibevoice-7B-8bit-DISABLED": {
        "repo": "DevParker/VibeVoice7b-low-vram",
        "description": "NOT AVAILABLE - Use 4bit version instead",
        "size": "N/A",
        "type": "quantized", 
        "files": []
    },
    # GGUF models - experimental but worth testing
    "vibevoice-7B-Q8-gguf": {
        "repo": "wsbagnsv1/VibeVoice-Large-pt-gguf",
        "description": "VibeVoice 7B (GGUF Q8_0) - 10GB, experimental format",
        "size": "10.1GB",
        "type": "gguf",
        "files": [
            {"remote": "VibeVoice-Large-pt-q8_0.gguf", "local": "model.gguf"},
            # We'll need to copy config from original model
        ]
    },
    "vibevoice-7B-BF16-gguf": {
        "repo": "wsbagnsv1/VibeVoice-Large-pt-gguf",
        "description": "VibeVoice 7B (GGUF BF16) - 18.7GB, experimental format",
        "size": "18.7GB",
        "type": "gguf",
        "files": [
            {"remote": "VibeVoice-Large-pt-bf16.gguf", "local": "model.gguf"},
            # We'll need to copy config from original model
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
        Handles standard, quantized, and GGUF models.
        
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
        model_type = model_info.get("type", "standard")
        subfolder = model_info.get("subfolder")
        
        # 1. Check current local path first
        model_dir = os.path.join(self.vibevoice_dir, model_name)
        
        # Different validation based on model type
        if model_type == "gguf":
            gguf_path = os.path.join(model_dir, "model.gguf")
            if os.path.exists(gguf_path):
                print(f"📁 Using local GGUF VibeVoice model: {model_dir}")
                return model_dir
        else:
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
        
        # 2. Check legacy VibeVoice-ComfyUI path (only for standard models)
        if model_type == "standard":
            legacy_vibevoice_dir = os.path.join(self.models_dir, "vibevoice")
            legacy_model_dir = os.path.join(legacy_vibevoice_dir, f"models--{repo_id.replace('/', '--')}")
            legacy_config_path = os.path.join(legacy_model_dir, "config.json")
            
            if os.path.exists(legacy_config_path):
                print(f"📁 Using legacy VibeVoice model: {legacy_model_dir}")
                return legacy_model_dir
        
        # 3. Check HuggingFace cache (only for standard models)
        if model_type == "standard":
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
        print(f"📥 Downloading VibeVoice model '{model_name}' ({model_type})...")
        
        result = self.downloader.download_huggingface_model(
            repo_id=model_info["repo"],
            model_name=model_name,
            files=model_info["files"],
            engine_type="vibevoice",
            subfolder=subfolder
        )
        
        if result:
            print(f"✅ VibeVoice model '{model_name}' downloaded successfully")
            
            # Special handling for GGUF models - we may need to copy config from original model
            if model_type == "gguf":
                self._ensure_gguf_config(result, model_name)
            
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
    
    def _ensure_gguf_config(self, model_dir: str, model_name: str) -> None:
        """
        Ensure GGUF models have the necessary config files.
        GGUF files don't include config.json, so we copy from the original model.
        
        Args:
            model_dir: Path to the GGUF model directory
            model_name: Name of the GGUF model
        """
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            return  # Config already exists
        
        # Try to get config from the original 7B model
        original_model = "vibevoice-7B"  # GGUF models are based on 7B
        if original_model in VIBEVOICE_MODELS:
            original_path = self.get_model_path(original_model)
            if original_path:
                original_config = os.path.join(original_path, "config.json")
                original_preprocessor = os.path.join(original_path, "preprocessor_config.json")
                
                try:
                    # Copy config files
                    if os.path.exists(original_config):
                        import shutil
                        shutil.copy2(original_config, config_path)
                        print(f"📋 Copied config.json to GGUF model: {config_path}")
                    
                    if os.path.exists(original_preprocessor):
                        import shutil
                        preprocessor_path = os.path.join(model_dir, "preprocessor_config.json")
                        shutil.copy2(original_preprocessor, preprocessor_path)
                        print(f"📋 Copied preprocessor_config.json to GGUF model: {preprocessor_path}")
                        
                except Exception as e:
                    print(f"⚠️ Could not copy config files to GGUF model: {e}")
                    print(f"💡 GGUF model may still work, but config will be inferred")
    
    def is_gguf_model(self, model_name: str) -> bool:
        """
        Check if a model is a GGUF model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if it's a GGUF model
        """
        if model_name not in VIBEVOICE_MODELS:
            return False
        return VIBEVOICE_MODELS[model_name].get("type") == "gguf"
    
    def is_quantized_model(self, model_name: str) -> bool:
        """
        Check if a model is a pre-quantized model (not GGUF).
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if it's a pre-quantized model
        """
        if model_name not in VIBEVOICE_MODELS:
            return False
        return VIBEVOICE_MODELS[model_name].get("type") == "quantized"
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get information about a VibeVoice model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model info dict or None
        """
        return VIBEVOICE_MODELS.get(model_name)