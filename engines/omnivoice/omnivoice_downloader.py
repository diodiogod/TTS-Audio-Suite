"""
OmniVoice model downloader.

Downloads the official OmniVoice assets into ComfyUI/models/TTS/omnivoice/.
"""

import os
from typing import Dict, Optional

import folder_paths
from huggingface_hub import snapshot_download

from utils.hf_download_logging import quiet_hf_download_logs
from utils.models.extra_paths import get_all_tts_model_paths, get_preferred_download_path


class OmniVoiceDownloader:
    """Resolve and download official OmniVoice model folders."""

    MODELS: Dict[str, Dict[str, object]] = {
        "OmniVoice": {
            "repo_id": "k2-fsa/OmniVoice",
            "description": "Official OmniVoice multilingual zero-shot TTS checkpoint",
            "required_files": [
                "chat_template.jinja",
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "audio_tokenizer/config.json",
                "audio_tokenizer/model.safetensors",
                "audio_tokenizer/preprocessor_config.json",
            ],
        },
    }

    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            try:
                self.base_path = get_preferred_download_path(model_type="TTS", engine_name="omnivoice")
            except Exception:
                self.base_path = os.path.join(folder_paths.models_dir, "TTS", "omnivoice")
        else:
            self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def resolve_model_path(self, model_identifier: str) -> str:
        """Resolve absolute paths, local: names, or known official identifiers."""
        if not model_identifier:
            model_identifier = "OmniVoice"

        if os.path.isabs(model_identifier) and os.path.exists(model_identifier):
            return model_identifier

        if os.path.isdir(model_identifier):
            return model_identifier

        if model_identifier.startswith("local:"):
            local_name = model_identifier[6:]
            for base_path in get_all_tts_model_paths("TTS"):
                for folder_name in ("omnivoice", "OmniVoice", ""):
                    candidate = (
                        os.path.join(base_path, folder_name, local_name)
                        if folder_name else os.path.join(base_path, local_name)
                    )
                    if os.path.exists(candidate):
                        print(f"📁 Using local OmniVoice model: {candidate}")
                        return candidate
            raise FileNotFoundError(f"Local OmniVoice model not found: {local_name}")

        if model_identifier in self.MODELS:
            return self.get_model_path(model_identifier)

        raise FileNotFoundError(f"Unknown OmniVoice model identifier: {model_identifier}")

    def get_model_path(self, model_name: str) -> str:
        model_dir = os.path.join(self.base_path, model_name)
        if not os.path.exists(model_dir) or not self._is_omnivoice_model_complete(model_dir):
            return self.download_model(model_name, force=os.path.exists(model_dir))
        return model_dir

    def download_model(self, model_name: str, force: bool = False) -> str:
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown OmniVoice model: {model_name}")

        model_info = self.MODELS[model_name]
        repo_id = str(model_info["repo_id"])
        model_dir = os.path.join(self.base_path, model_name)

        print(f"\n{'=' * 60}")
        print("📦 OmniVoice Model Download")
        print(f"{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"Description: {model_info['description']}")
        print(f"Repository: {repo_id}")
        print(f"Target: {model_dir}")
        print(f"{'=' * 60}\n")

        try:
            with quiet_hf_download_logs():
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    force_download=force,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to download {model_name} from {repo_id}: {e}") from e

        if not self._is_omnivoice_model_complete(model_dir):
            raise RuntimeError(f"Downloaded OmniVoice model is incomplete: {model_dir}")

        print(f"✅ OmniVoice model ready: {model_dir}")
        return model_dir

    def _is_omnivoice_model_complete(self, model_dir: str) -> bool:
        if not os.path.isdir(model_dir):
            return False

        required_files = self.MODELS["OmniVoice"]["required_files"]
        missing = [
            rel_path
            for rel_path in required_files
            if not os.path.exists(os.path.join(model_dir, rel_path))
        ]
        if missing:
            print(f"❌ OmniVoice model incomplete. Missing {len(missing)} file(s):")
            for rel_path in missing:
                print(f"   - {rel_path}")
            return False
        return True
