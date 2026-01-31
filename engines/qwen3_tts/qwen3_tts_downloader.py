"""
Qwen3-TTS Model Downloader

Handles automatic download and setup of Qwen3-TTS models using the unified download system.
Downloads models to organized TTS/qwen3_tts/ structure.
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
from utils.models.extra_paths import get_preferred_download_path, get_all_tts_model_paths
import folder_paths


class Qwen3TTSDownloader:
    """Downloader for Qwen3-TTS models using unified download system."""

    MODELS = {
        "Qwen3-TTS-Tokenizer-12Hz": {
            "repo_id": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            "files": "all",  # Download all files
            "description": "Qwen3-TTS Tokenizer (12Hz) - Required by all models"
        },
        "Qwen3-TTS-12Hz-0.6B-CustomVoice": {
            "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "files": "all",
            "description": "Qwen3-TTS 0.6B CustomVoice - 9 preset speakers (low VRAM ~6GB)"
        },
        "Qwen3-TTS-12Hz-0.6B-Base": {
            "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "files": "all",
            "description": "Qwen3-TTS 0.6B Base - Zero-shot voice cloning (low VRAM ~6GB)"
        },
        "Qwen3-TTS-12Hz-1.7B-CustomVoice": {
            "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "files": "all",
            "description": "Qwen3-TTS 1.7B CustomVoice - 9 preset speakers with instruction support (~12GB)"
        },
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
            "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "files": "all",
            "description": "Qwen3-TTS 1.7B VoiceDesign - Text-to-voice design UNIQUE FEATURE (~12GB)"
        },
        "Qwen3-TTS-12Hz-1.7B-Base": {
            "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "files": "all",
            "description": "Qwen3-TTS 1.7B Base - Zero-shot voice cloning high quality (~12GB)"
        }
    }

    # Strict required file lists based on HF repos (2026-01-31).
    # Keep this tight to catch partial downloads that crash with KeyError 'default'.
    REQUIRED_FILES = {
        "Qwen3-TTS-Tokenizer-12Hz": [
            "config.json",
            "configuration.json",
            "model.safetensors",
            "preprocessor_config.json",
        ],
        "Qwen3-TTS-12Hz-0.6B-CustomVoice": [
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "speech_tokenizer/config.json",
            "speech_tokenizer/configuration.json",
            "speech_tokenizer/model.safetensors",
            "speech_tokenizer/preprocessor_config.json",
        ],
        "Qwen3-TTS-12Hz-0.6B-Base": [
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "speech_tokenizer/config.json",
            "speech_tokenizer/configuration.json",
            "speech_tokenizer/model.safetensors",
            "speech_tokenizer/preprocessor_config.json",
        ],
        "Qwen3-TTS-12Hz-1.7B-CustomVoice": [
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "speech_tokenizer/config.json",
            "speech_tokenizer/configuration.json",
            "speech_tokenizer/model.safetensors",
            "speech_tokenizer/preprocessor_config.json",
        ],
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign": [
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "speech_tokenizer/config.json",
            "speech_tokenizer/configuration.json",
            "speech_tokenizer/model.safetensors",
            "speech_tokenizer/preprocessor_config.json",
        ],
        "Qwen3-TTS-12Hz-1.7B-Base": [
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "speech_tokenizer/config.json",
            "speech_tokenizer/configuration.json",
            "speech_tokenizer/model.safetensors",
            "speech_tokenizer/preprocessor_config.json",
        ],
    }

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize downloader.

        Args:
            base_path: Base directory for Qwen3-TTS models (auto-detected if None)
        """
        if base_path is None:
            # Use extra_model_paths configuration for downloads (respects YAML paths)
            try:
                self.base_path = get_preferred_download_path(model_type='TTS', engine_name='qwen3_tts')
            except Exception as e:
                # Fallback to default if extra_paths fails
                self.base_path = os.path.join(folder_paths.models_dir, "TTS", "qwen3_tts")
        else:
            self.base_path = base_path

        # Ensure base directory exists
        os.makedirs(self.base_path, exist_ok=True)

    def download_model(self, model_name: str, force: bool = False) -> str:
        """
        Download a Qwen3-TTS model.

        Args:
            model_name: Model name (e.g., "Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen3-TTS-Tokenizer-12Hz")
            force: Force re-download even if files exist

        Returns:
            Path to downloaded model directory
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")

        model_info = self.MODELS[model_name]
        repo_id = model_info["repo_id"]
        files = model_info["files"]
        description = model_info["description"]

        # Target directory
        model_dir = os.path.join(self.base_path, model_name)

        print(f"\n{'='*60}")
        print(f"📦 Qwen3-TTS Model Download")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Description: {description}")
        print(f"Repository: {repo_id}")
        print(f"Target: {model_dir}")
        print(f"{'='*60}\n")

        # Check if already downloaded (basic check - directory exists with files)
        if not force and os.path.exists(model_dir):
            # Basic completeness check - if directory has files, assume complete
            if os.listdir(model_dir):
                print(f"✅ Model already downloaded: {model_dir}")
                return model_dir

        # Download using HuggingFace snapshot_download (download entire repo)
        try:
            print(f"📥 Downloading {model_name} from HuggingFace...")

            from huggingface_hub import snapshot_download

            # Download entire model repository to target directory
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )

            if os.path.exists(model_dir) and os.listdir(model_dir):
                print(f"\n✅ Download complete: {model_dir}")
                return model_dir
            else:
                raise RuntimeError("Download completed but model directory is empty")

        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            raise

    def download_all(self, force: bool = False):
        """
        Download all Qwen3-TTS models (tokenizer + 5 model variants).

        Args:
            force: Force re-download even if files exist
        """
        print("\n🚀 Downloading all Qwen3-TTS models (this will take a while)...")
        print(f"   Total models: {len(self.MODELS)}")

        for model_name in self.MODELS.keys():
            try:
                self.download_model(model_name, force=force)
            except Exception as e:
                print(f"⚠️ Failed to download {model_name}: {e}")
                continue

        print("\n✅ All Qwen3-TTS models downloaded!")

    def download_essential(self, force: bool = False):
        """
        Download essential models for quick start (tokenizer + 1.7B-CustomVoice).

        This gives users the preset voice functionality immediately.
        """
        print("\n📥 Downloading essential Qwen3-TTS models for quick start...")
        print("   - Tokenizer (required)")
        print("   - 1.7B CustomVoice (9 preset speakers)")

        essential = ["Qwen3-TTS-Tokenizer-12Hz", "Qwen3-TTS-12Hz-1.7B-CustomVoice"]

        for model_name in essential:
            try:
                self.download_model(model_name, force=force)
            except Exception as e:
                print(f"⚠️ Failed to download {model_name}: {e}")
                continue

        print(f"\n✅ Essential models ready!")

    def resolve_model_path(self, model_identifier: str) -> str:
        """
        Resolve model path handling "local:" prefix and auto-download.

        Args:
            model_identifier: Model identifier ("local:ModelName" or "ModelName")

        Returns:
            Resolved filesystem path to model directory

        Raises:
            FileNotFoundError: If local model not found
            RuntimeError: If auto-download fails
        """
        # Handle None or empty model identifier
        if not model_identifier:
            model_identifier = "Qwen3-TTS-12Hz-1.7B-CustomVoice"  # Default

        # Handle local: prefix - search in all configured TTS paths
        if model_identifier.startswith("local:"):
            local_name = model_identifier[6:]  # Remove "local:" prefix

            # Search in all configured TTS paths
            for base_tts_path in get_all_tts_model_paths('TTS'):
                for folder_name in ["qwen3_tts", "Qwen3-TTS"]:
                    local_path = os.path.join(base_tts_path, folder_name, local_name)
                    if os.path.exists(local_path):
                        print(f"📁 Using local Qwen3-TTS model: {local_path}")
                        return local_path

            # If not found, raise error
            raise FileNotFoundError(f"Local Qwen3-TTS model not found: {local_name}")

        # Handle predefined models or auto-download
        return self.get_model_path(model_identifier)

    def get_model_path(self, model_name: str) -> str:
        """
        Get the path to a model (auto-downloads if missing).

        Args:
            model_name: Model name or already-resolved path

        Returns:
            Path to model directory
        """
        # If already a full path, return it as-is (handles cached resolved paths)
        if os.path.isabs(model_name) and os.path.exists(model_name):
            return model_name

        model_dir = os.path.join(self.base_path, model_name)

        # Auto-download if missing
        if model_name in self.MODELS:
            if not os.path.exists(model_dir) or not os.listdir(model_dir):
                print(f"📥 Model not found, downloading {model_name}...")
                return self.download_model(model_name)
            if not self._is_model_complete(model_name, model_dir):
                print(f"⚠️ Incomplete Qwen3-TTS model detected: {model_dir}")
                print(f"📥 Re-downloading {model_name} to fix missing files...")
                return self.download_model(model_name, force=True)
        elif not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model not found: {model_name}")

        return model_dir

    def _is_model_complete(self, model_name: str, model_dir: str) -> bool:
        """
        Strict completeness check for Qwen3-TTS model folders.

        Uses HF repo file lists captured in REQUIRED_FILES.
        """
        if not os.path.isdir(model_dir):
            return False
        required = self.REQUIRED_FILES.get(model_name)
        if not required:
            # Fall back to basic check if we don't have a strict list
            return os.path.exists(os.path.join(model_dir, "config.json"))

        missing = []
        for rel_path in required:
            if not os.path.exists(os.path.join(model_dir, rel_path)):
                missing.append(rel_path)

        if missing:
            print(f"❌ Qwen3-TTS model incomplete. Missing {len(missing)} file(s):")
            for path in missing[:20]:
                print(f"   - {path}")
            if len(missing) > 20:
                print(f"   ... and {len(missing) - 20} more")
            return False

        return True


def main():
    """CLI interface for Qwen3-TTS downloader."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Qwen3-TTS models")
    parser.add_argument(
        "model",
        nargs="?",
        default="essential",
        choices=["all", "essential"] + list(Qwen3TTSDownloader.MODELS.keys()),
        help="Model to download (default: essential)"
    )
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--path", type=str, help="Custom download path")

    args = parser.parse_args()

    downloader = Qwen3TTSDownloader(base_path=args.path)

    if args.model == "all":
        downloader.download_all(force=args.force)
    elif args.model == "essential":
        downloader.download_essential(force=args.force)
    else:
        downloader.download_model(args.model, force=args.force)


if __name__ == "__main__":
    main()
