"""
Qwen3-ASR Model Downloader

Handles automatic download and setup of Qwen3-ASR models.
Downloads models to organized ASR/qwen3_asr/ structure.
"""

import os
import sys
from typing import Optional

# Add project root to path
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.models.extra_paths import get_preferred_download_path
import folder_paths


class Qwen3ASRDownloader:
    """Downloader for Qwen3-ASR models using Hugging Face snapshot downloads."""

    MODELS = {
        "Qwen3-ASR-1.7B": {
            "repo_id": "Qwen/Qwen3-ASR-1.7B",
            "files": "all",
            "description": "Qwen3-ASR 1.7B - high quality ASR model"
        },
        "Qwen3-ASR-0.6B": {
            "repo_id": "Qwen/Qwen3-ASR-0.6B",
            "files": "all",
            "description": "Qwen3-ASR 0.6B - low VRAM ASR model"
        },
        "Qwen3-ForcedAligner-0.6B": {
            "repo_id": "Qwen/Qwen3-ForcedAligner-0.6B",
            "files": "all",
            "description": "Qwen3 forced aligner for timestamps"
        },
    }

    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            try:
                # Keep Qwen3 family unified under the same TTS engine folder
                tts_base = get_preferred_download_path(model_type="TTS", engine_name="qwen3_tts")
                self.base_path = os.path.join(tts_base, "asr")
            except Exception:
                self.base_path = os.path.join(folder_paths.models_dir, "TTS", "qwen3_tts", "asr")
        else:
            self.base_path = base_path

        os.makedirs(self.base_path, exist_ok=True)

    def download_model(self, model_name: str, force: bool = False) -> str:
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")

        model_info = self.MODELS[model_name]
        repo_id = model_info["repo_id"]
        description = model_info["description"]

        model_dir = os.path.join(self.base_path, model_name)

        print(f"\n{'='*60}")
        print("📦 Qwen3-ASR Model Download")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Description: {description}")
        print(f"Repository: {repo_id}")
        print(f"Target: {model_dir}")
        print(f"{'='*60}\n")

        if not force and os.path.exists(model_dir) and os.listdir(model_dir):
            print(f"✅ Model already downloaded: {model_dir}")
            return model_dir

        try:
            print(f"📥 Downloading {model_name} from Hugging Face...")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )

            if os.path.exists(model_dir) and os.listdir(model_dir):
                print(f"\n✅ Download complete: {model_dir}")
                return model_dir
            raise RuntimeError("Download completed but model directory is empty")
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            raise

    def resolve_model_path(self, model_identifier: str) -> str:
        if not model_identifier:
            model_identifier = "Qwen3-ASR-1.7B"

        if model_identifier.startswith("local:"):
            model_name = model_identifier[6:]
            local_path = os.path.join(self.base_path, model_name)
            if os.path.exists(local_path):
                return local_path
            raise FileNotFoundError(f"Local Qwen3-ASR model '{model_name}' not found in {self.base_path}")

        model_dir = os.path.join(self.base_path, model_identifier)
        if os.path.exists(model_dir) and os.listdir(model_dir):
            return model_dir

        # Legacy fallback: reference node used diffusion_models/Qwen3-ASR
        try:
            path = folder_paths.get_full_path("diffusion_models", f"Qwen3-ASR/{model_identifier}")
            if path:
                return os.path.dirname(path) if os.path.isfile(path) else path
        except Exception:
            pass

        if model_identifier in self.MODELS:
            return self.download_model(model_identifier)

        return model_identifier
