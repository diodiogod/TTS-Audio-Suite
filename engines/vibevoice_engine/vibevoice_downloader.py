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

# VibeVoice model configurations with fallback sources
# Microsoft removed some official repos, so we include community mirrors
VIBEVOICE_MODELS = {
    "vibevoice-1.5B": {
        "repo": "microsoft/VibeVoice-1.5B",  # Primary: Microsoft official (still available)
        "fallback_repos": [  # Fallback sources if primary fails
            "Shinku/VibeVoice-1.5B",
            "callgg/vibevoice-bf16"
        ],
        "description": "Microsoft VibeVoice 1.5B - Official model (actually 2.7B params)",
        "size": "5.4GB",
        "tokenizer_repo": "Qwen/Qwen2.5-1.5B",  # Tokenizer source for this model
        "files": [
            # Required model files
            {"remote": "model-00001-of-00003.safetensors", "local": "model-00001-of-00003.safetensors"},
            {"remote": "model-00002-of-00003.safetensors", "local": "model-00002-of-00003.safetensors"},
            {"remote": "model-00003-of-00003.safetensors", "local": "model-00003-of-00003.safetensors"},
            {"remote": "model.safetensors.index.json", "local": "model.safetensors.index.json"},
            {"remote": "config.json", "local": "config.json"},
            {"remote": "preprocessor_config.json", "local": "preprocessor_config.json"},
            # Tokenizer files from Qwen2.5-1.5B (required to prevent crashes)
            {"remote": "tokenizer.json", "local": "tokenizer.json", "alt_repo": "Qwen/Qwen2.5-1.5B"}
        ]
    },
    "vibevoice-7B": {
        "repo": "aoi-ot/VibeVoice-Large",  # Primary: Community mirror (Microsoft's is blocked)
        "fallback_repos": [  # Additional fallback sources
            "aoi-ot/VibeVoice-7B",
            "DevParker/VibeVoice7b-low-vram",
            "SassyDiffusion/VibeVoice-Large-pt"
        ],
        "description": "VibeVoice Large - 7B model (actually 9.3B params) from community mirror",
        "size": "18GB",
        "tokenizer_repo": "Qwen/Qwen2.5-7B",  # Tokenizer source for this model
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
            {"remote": "preprocessor_config.json", "local": "preprocessor_config.json"},
            # Tokenizer files from Qwen2.5-7B (required to prevent crashes)
            {"remote": "tokenizer.json", "local": "tokenizer.json", "alt_repo": "Qwen/Qwen2.5-7B"}
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
        # Start with ALL official models (like ChatterBox does)
        available = list(VIBEVOICE_MODELS.keys())

        # Then add "local:" versions for downloaded models
        for model_key, model_info in VIBEVOICE_MODELS.items():
            model_dir = os.path.join(self.vibevoice_dir, model_key)
            config_path = os.path.join(model_dir, "config.json")

            if os.path.exists(config_path):
                # Add "local:" prefix for downloaded official models (consistent with F5/ChatterBox)
                local_model_name = f"local:{model_key}"
                available.append(local_model_name)

        # Check for standalone .safetensors files
        import folder_paths
        search_dirs = [self.vibevoice_dir]

        # Also check checkpoints folder if available
        if hasattr(folder_paths, 'get_folder_paths'):
            checkpoint_dirs = folder_paths.get_folder_paths("checkpoints")
            if checkpoint_dirs:
                search_dirs.extend(checkpoint_dirs)

        for model_path_dir in search_dirs:
            if not model_path_dir or not os.path.exists(model_path_dir):
                continue

            for item in os.listdir(model_path_dir):
                item_path = os.path.join(model_path_dir, item)

                # Look for standalone .safetensors files
                if os.path.isfile(item_path) and item.endswith('.safetensors'):
                    model_name = os.path.splitext(item)[0]
                    # Avoid duplicates and skip if it's part of a regular model
                    if model_name not in available and model_name not in VIBEVOICE_MODELS:
                        # Add "local:" prefix for consistency with ChatterBox/F5-TTS
                        local_model_name = f"local:{model_name}"
                        available.append(local_model_name)

        # Official models already added at the start

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
        # First check if this might be a standalone model
        # Strip "local:" prefix if present
        clean_model_name = model_name.replace("local:", "") if model_name.startswith("local:") else model_name
        standalone_path = self._find_standalone_model(clean_model_name)
        if standalone_path:
            print(f"📁 Using standalone VibeVoice model: {standalone_path}")
            return standalone_path

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
                # Ensure tokenizer.json exists (download if missing)
                self._ensure_tokenizer(model_name, model_dir)
                return model_dir
            else:
                print(f"🔄 Local model incomplete, will re-download: {model_dir}")
        
        # 2. Check legacy VibeVoice-ComfyUI path
        legacy_vibevoice_dir = os.path.join(self.models_dir, "vibevoice")
        legacy_model_dir = os.path.join(legacy_vibevoice_dir, f"models--{repo_id.replace('/', '--')}")
        legacy_config_path = os.path.join(legacy_model_dir, "config.json")
        
        if os.path.exists(legacy_config_path):
            print(f"📁 Using legacy VibeVoice model: {legacy_model_dir}")
            # Ensure tokenizer.json exists (download if missing)
            self._ensure_tokenizer(model_name, legacy_model_dir)
            return legacy_model_dir
        
        # 3. Check HuggingFace cache
        try:
            from huggingface_hub import hf_hub_download
            # Try to find config.json in cache (local_files_only=True means cache only)
            cached_config = hf_hub_download(repo_id=repo_id, filename="config.json", local_files_only=True)
            cached_model_dir = os.path.dirname(cached_config)
            print(f"📁 Using cached VibeVoice model: {cached_model_dir}")
            # Ensure tokenizer.json exists (download if missing)
            self._ensure_tokenizer(model_name, cached_model_dir)
            return cached_model_dir
        except Exception as cache_error:
            print(f"📋 Cache check for {model_name}: {str(cache_error)[:100]}... - will download")
        
        # 4. Download model to local directory with fallback repos
        print(f"📥 Downloading VibeVoice model '{model_name}'...")
        
        # Try primary repo first
        repos_to_try = [model_info["repo"]] + model_info.get("fallback_repos", [])
        
        for repo_id in repos_to_try:
            print(f"🔄 Trying repository: {repo_id}")
            
            result = self.downloader.download_huggingface_model(
                repo_id=repo_id,
                model_name=model_name,
                files=model_info["files"],
                engine_type="vibevoice",
                subfolder=None
            )
            
            if result:
                print(f"✅ VibeVoice model '{model_name}' downloaded successfully from {repo_id}")
                # Ensure tokenizer.json exists (download if missing)
                self._ensure_tokenizer(model_name, result)
                return result
            else:
                print(f"⚠️ Failed to download from {repo_id}, trying next...")
        
        print(f"❌ Failed to download VibeVoice model '{model_name}' from all sources")
        print(f"   Tried repos: {', '.join(repos_to_try)}")
        return None
    
    def ensure_vibevoice_package(self) -> bool:
        """
        Ensure VibeVoice package is installed.
        
        Returns:
            True if package is available, False otherwise
        """
        try:
            import vibevoice
            # print(f"✅ VibeVoice base package found: {vibevoice.__file__}")  # Verbose logging
            
            # Test specific modules we need
            from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            # print("✅ VibeVoiceForConditionalGenerationInference imported successfully")  # Verbose logging
            
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
            # print("✅ VibeVoiceProcessor imported successfully")  # Verbose logging
            
            return True
        except ImportError as e:
            print(f"❌ VibeVoice package import failed: {e}")
            print("🔄 This should have been installed via the install script")
            print("📦 If the issue persists, try reinstalling the node via ComfyUI Manager")
            print("   or manually: pip install git+https://github.com/microsoft/VibeVoice.git")
            return False
    
    def _ensure_tokenizer(self, model_name: str, model_path: str) -> None:
        """
        Ensure tokenizer.json exists in model folder, download if missing.
        
        Args:
            model_name: Name of the model
            model_path: Path to model directory
        """
        tokenizer_path = os.path.join(model_path, "tokenizer.json")
        
        if not os.path.exists(tokenizer_path):
            model_info = VIBEVOICE_MODELS.get(model_name)
            if not model_info:
                return
            
            tokenizer_repo = model_info.get("tokenizer_repo")
            if not tokenizer_repo:
                return
            
            print(f"📥 Downloading tokenizer.json from {tokenizer_repo}...")
            try:
                from huggingface_hub import hf_hub_download
                
                # Download directly to model folder
                hf_hub_download(
                    repo_id=tokenizer_repo,
                    filename="tokenizer.json",
                    local_dir=model_path,
                    local_dir_use_symlinks=False
                )
                print(f"✅ Tokenizer downloaded to {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to download tokenizer.json: {e}")
                print(f"   You can manually download it from {tokenizer_repo}")
                print(f"   and place it in {model_path}")

    def _find_standalone_model(self, model_name: str) -> Optional[str]:
        """
        Find standalone .safetensors file for a model name.

        Args:
            model_name: Name to search for

        Returns:
            Path to standalone .safetensors file or None if not found
        """
        import folder_paths

        search_dirs = [self.vibevoice_dir]

        # Also check checkpoints folder if available
        if hasattr(folder_paths, 'get_folder_paths'):
            checkpoint_dirs = folder_paths.get_folder_paths("checkpoints")
            if checkpoint_dirs:
                search_dirs.extend(checkpoint_dirs)

        for model_path_dir in search_dirs:
            if not model_path_dir or not os.path.exists(model_path_dir):
                continue

            # Look for exact match
            safetensors_path = os.path.join(model_path_dir, f"{model_name}.safetensors")
            if os.path.isfile(safetensors_path):
                return safetensors_path

        return None

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get information about a VibeVoice model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model info dict or None
        """
        return VIBEVOICE_MODELS.get(model_name)