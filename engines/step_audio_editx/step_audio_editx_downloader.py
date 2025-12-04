"""
Step Audio EditX Model Downloader

Handles automatic download and setup of Step Audio EditX models using the unified download system.
Downloads models to organized TTS/step_audio_editx/ structure.
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


class StepAudioEditXDownloader:
    """Downloader for Step Audio EditX models using unified download system."""

    MODELS = {
        "Step-Audio-EditX": {
            "files": [
                # Main config and model files
                {"remote": "config.json", "local": "config.json"},
                {"remote": "configuration_step1.py", "local": "configuration_step1.py"},
                {"remote": "modeling_step1.py", "local": "modeling_step1.py"},

                # LLM model files (3B parameter - single shard)
                {"remote": "model-00001.safetensors", "local": "model-00001.safetensors"},
                {"remote": "model.safetensors.index.json", "local": "model.safetensors.index.json"},

                # Tokenizer files
                {"remote": "tokenizer.model", "local": "tokenizer.model"},
                {"remote": "tokenizer_config.json", "local": "tokenizer_config.json"},

                # CosyVoice vocoder (24kHz output)
                {"remote": "CosyVoice-300M-25Hz/FLOW_VERSION", "local": "CosyVoice-300M-25Hz/FLOW_VERSION"},
                {"remote": "CosyVoice-300M-25Hz/campplus.onnx", "local": "CosyVoice-300M-25Hz/campplus.onnx"},
                {"remote": "CosyVoice-300M-25Hz/cosyvoice.yaml", "local": "CosyVoice-300M-25Hz/cosyvoice.yaml"},
                {"remote": "CosyVoice-300M-25Hz/flow.pt", "local": "CosyVoice-300M-25Hz/flow.pt"},
                {"remote": "CosyVoice-300M-25Hz/hift.pt", "local": "CosyVoice-300M-25Hz/hift.pt"},
                {"remote": "CosyVoice-300M-25Hz/speech_tokenizer_v1.onnx", "local": "CosyVoice-300M-25Hz/speech_tokenizer_v1.onnx"}
            ],
            "additional_downloads": [
                # Download tokenizer files from Step-Audio-Tokenizer repo into main directory
                {
                    "repo_id": "stepfun-ai/Step-Audio-Tokenizer",
                    "files": [
                        {"remote": "linguistic_tokenizer.npy", "local": "linguistic_tokenizer.npy"},
                        {"remote": "speech_tokenizer_v1.onnx", "local": "speech_tokenizer_v1.onnx"}
                    ]
                }
            ],
            "repo_id": "stepfun-ai/Step-Audio-EditX",
            "description": "Step Audio EditX - 3B LLM-based TTS with emotion/style editing (7GB)"
        },
        "Step-Audio-Tokenizer": {
            "repo_id": "stepfun-ai/Step-Audio-Tokenizer",
            "files": [
                "speech_tokenizer_v1.onnx",
                "linguistic_tokenizer.npy"
            ],
            "description": "Step Audio dual-codebook tokenizer (VQ02 + VQ06) - included in Step-Audio-EditX download"
        },
        "FunASR-Paraformer": {
            "repo_id": "Diogodiogod/FunASR-Paraformer-Cantonese",
            "files": [
                "model.pb",
                "config.yaml",
                "am.mvn",
                "tokens.txt",
                "seg_dict"
            ],
            "description": "FunASR Paraformer model for VQ02 encoding (881MB)"
        }
    }

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize downloader.

        Args:
            base_path: Base directory for Step Audio EditX models (auto-detected if None)
        """
        if base_path is None:
            # Use extra_model_paths configuration for downloads (respects YAML paths)
            try:
                self.base_path = get_preferred_download_path(model_type='TTS', engine_name='step_audio_editx')
            except Exception as e:
                # Fallback to default if extra_paths fails
                self.base_path = os.path.join(folder_paths.models_dir, "TTS", "step_audio_editx")
        else:
            self.base_path = base_path

        # Ensure base directory exists
        os.makedirs(self.base_path, exist_ok=True)

    def download_model(self, model_name: str = "Step-Audio-EditX", force: bool = False) -> str:
        """
        Download a Step Audio EditX model.

        Args:
            model_name: Model name ("Step-Audio-EditX" or "Step-Audio-Tokenizer")
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
        print(f"📦 Step Audio EditX Model Download")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Description: {description}")
        print(f"Repository: {repo_id}")
        print(f"Target: {model_dir}")
        print(f"Files: {len(files)} files")
        print(f"{'='*60}\n")

        # Build complete file list including additional downloads
        if files and isinstance(files[0], dict):
            check_files = [f["local"] for f in files]
        else:
            check_files = list(files)

        # Add additional downloads to completeness check
        additional_downloads = model_info.get("additional_downloads", [])
        for additional in additional_downloads:
            for file_dict in additional["files"]:
                check_files.append(file_dict["local"])

        # Check if already downloaded
        if not force and self._is_model_complete(model_dir, check_files):
            print(f"✅ Model already downloaded and complete: {model_dir}")
            return model_dir

        # Download using appropriate method based on repo_id prefix
        try:
            if repo_id.startswith("modelscope:"):
                # ModelScope downloads not supported - models should be mirrored to HuggingFace
                raise NotImplementedError(
                    f"ModelScope download not supported for {model_name}. "
                    f"This model should be available on HuggingFace instead."
                )
            else:
                # HuggingFace download via unified_downloader
                print(f"📥 Downloading {model_name} from HuggingFace...")

                # Convert files list to the format expected by unified downloader
                # Handle both string format and dict format
                if files and isinstance(files[0], dict):
                    files_dicts = files
                else:
                    files_dicts = [{"remote": f, "local": f} for f in files]

                unified_downloader.download_huggingface_model(
                    repo_id=repo_id,
                    model_name=model_name,
                    files=files_dicts,
                    engine_type="step_audio_editx"
                )

                # Download additional files from other repos if specified
                additional_downloads = model_info.get("additional_downloads", [])
                for additional in additional_downloads:
                    additional_repo = additional["repo_id"]
                    additional_files = additional["files"]
                    print(f"📥 Downloading additional files from {additional_repo}...")

                    unified_downloader.download_huggingface_model(
                        repo_id=additional_repo,
                        model_name=model_name,  # Same target directory
                        files=additional_files,
                        engine_type="step_audio_editx"
                    )

            # Verify download (check main files only, not additional)
            if files and isinstance(files[0], dict):
                check_files = [f["local"] for f in files]
            else:
                check_files = files

            if self._is_model_complete(model_dir, check_files):
                print(f"\n✅ Download complete: {model_dir}")
                return model_dir
            else:
                raise RuntimeError("Download completed but model verification failed")

        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            raise

    def download_all(self, force: bool = False):
        """
        Download all Step Audio EditX models.

        Args:
            force: Force re-download even if files exist
        """
        print("\n🚀 Downloading all Step Audio EditX models...")

        for model_name in self.MODELS.keys():
            try:
                self.download_model(model_name, force=force)
            except Exception as e:
                print(f"⚠️ Failed to download {model_name}: {e}")
                continue

        print("\n✅ All Step Audio EditX models downloaded!")

    def _is_model_complete(self, model_dir: str, required_files: list) -> bool:
        """
        Check if model is completely downloaded.

        Args:
            model_dir: Model directory
            required_files: List of required files (can be strings or dicts with 'local' key)

        Returns:
            True if all required files exist
        """
        if not os.path.exists(model_dir):
            return False

        for file in required_files:
            # Handle both string format and dict format
            if isinstance(file, dict):
                file_name = file["local"]
            else:
                file_name = file

            file_path = os.path.join(model_dir, file_name)
            if not os.path.exists(file_path):
                return False

        return True

    def _verify_model(self, model_dir: str, model_name: str):
        """
        Verify model completeness (raises exception if incomplete).

        Args:
            model_dir: Model directory
            model_name: Model name

        Raises:
            FileNotFoundError: If model is incomplete
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        required_files = self.MODELS[model_name]["files"]

        if not self._is_model_complete(model_dir, required_files):
            missing_files = []
            for file in required_files:
                file_path = os.path.join(model_dir, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)

            raise FileNotFoundError(
                f"Step Audio EditX model incomplete at {model_dir}. "
                f"Missing {len(missing_files)} files: {missing_files[:5]}..."
            )

    def get_model_path(self, model_name: str = "Step-Audio-EditX") -> str:
        """
        Get the path to a model (auto-downloads if missing).

        Args:
            model_name: Model name

        Returns:
            Path to model directory
        """
        model_dir = os.path.join(self.base_path, model_name)

        # Auto-download if missing
        if not self._is_model_complete(model_dir, self.MODELS[model_name]["files"]):
            print(f"📥 Model not found, downloading {model_name}...")
            return self.download_model(model_name)

        return model_dir


def main():
    """CLI interface for Step Audio EditX downloader."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Step Audio EditX models")
    parser.add_argument(
        "model",
        nargs="?",
        default="all",
        choices=["all"] + list(StepAudioEditXDownloader.MODELS.keys()),
        help="Model to download (default: all)"
    )
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--path", type=str, help="Custom download path")

    args = parser.parse_args()

    downloader = StepAudioEditXDownloader(base_path=args.path)

    if args.model == "all":
        downloader.download_all(force=args.force)
    else:
        downloader.download_model(args.model, force=args.force)


if __name__ == "__main__":
    main()
