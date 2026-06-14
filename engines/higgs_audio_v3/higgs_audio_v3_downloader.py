"""
Higgs Audio v3 model downloader.

Downloads Boson Higgs Audio v3 into ComfyUI/models/TTS/higgs_audio_v3/.
"""

import os
from typing import Dict, Optional

import folder_paths
from utils.models.extra_paths import get_all_tts_model_paths, get_preferred_download_path


class HiggsAudioV3Downloader:
    """Resolve and download official Higgs Audio v3 model folders."""

    MODEL_NAME = "higgs-audio-v3-tts-4b"
    REPO_ID = "bosonai/higgs-audio-v3-tts-4b"

    MODELS: Dict[str, Dict[str, str]] = {
        MODEL_NAME: {
            "repo_id": REPO_ID,
            "description": "Higgs Audio v3 TTS 4B - multilingual controllable TTS with inline tags",
        }
    }

    REQUIRED_FILES = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "chat_template.jinja",
        "LICENSE",
    ]

    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            try:
                self.base_path = get_preferred_download_path(model_type="TTS", engine_name="higgs_audio_v3")
            except Exception:
                self.base_path = os.path.join(folder_paths.models_dir, "TTS", "higgs_audio_v3")
        else:
            self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def resolve_model_path(self, model_identifier: str) -> str:
        """Resolve local paths, local: names, or known Higgs v3 model identifiers."""
        model_identifier = model_identifier or self.MODEL_NAME

        if os.path.isabs(model_identifier) and os.path.exists(model_identifier):
            return model_identifier

        if model_identifier.startswith("local:"):
            local_name = model_identifier[6:]
            for base_path in get_all_tts_model_paths("TTS"):
                for folder_name in ("HiggsAudioV3", "higgs_audio_v3", "higgsaudiov3"):
                    candidate = os.path.join(base_path, folder_name, local_name)
                    if os.path.exists(candidate):
                        print(f"📁 Using local Higgs Audio v3 model: {candidate}")
                        return candidate
            raise FileNotFoundError(f"Local Higgs Audio v3 model not found: {local_name}")

        return self.get_model_path(model_identifier)

    def get_model_path(self, model_name: str) -> str:
        if os.path.isabs(model_name) and os.path.exists(model_name):
            return model_name

        if model_name not in self.MODELS:
            raise ValueError(f"Unknown Higgs Audio v3 model: {model_name}")

        model_dir = os.path.join(self.base_path, model_name)
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            return self.download_model(model_name)
        if not self._is_model_complete(model_dir):
            print(f"⚠️ Incomplete Higgs Audio v3 model detected: {model_dir}")
            return self.download_model(model_name, force=True)
        return model_dir

    def download_model(self, model_name: str, force: bool = False) -> str:
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown Higgs Audio v3 model: {model_name}")

        repo_id = self.MODELS[model_name]["repo_id"]
        model_dir = os.path.join(self.base_path, model_name)

        print(f"\n{'=' * 60}")
        print("📦 Higgs Audio v3 Model Download")
        print(f"{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"Repository: {repo_id}")
        print(f"Target: {model_dir}")
        print("License: Boson Higgs Audio v3 Research and Non-Commercial License")
        print(f"{'=' * 60}\n")

        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                force_download=force,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download Higgs Audio v3 from {repo_id}: {e}") from e

        if not self._is_model_complete(model_dir):
            raise RuntimeError(f"Downloaded Higgs Audio v3 model is incomplete: {model_dir}")

        print(f"✅ Higgs Audio v3 model ready: {model_dir}")
        return model_dir

    def _is_model_complete(self, model_dir: str) -> bool:
        if not os.path.isdir(model_dir):
            return False

        missing = [
            rel_path
            for rel_path in self.REQUIRED_FILES
            if not os.path.exists(os.path.join(model_dir, rel_path))
        ]
        if missing:
            print(f"❌ Higgs Audio v3 model incomplete. Missing {len(missing)} file(s):")
            for rel_path in missing[:20]:
                print(f"   - {rel_path}")
            return False
        return True

    def get_available_models(self) -> list:
        available = [self.MODEL_NAME]
        for base_path in get_all_tts_model_paths("TTS"):
            for folder_name in ("HiggsAudioV3", "higgs_audio_v3", "higgsaudiov3"):
                root = os.path.join(base_path, folder_name)
                if not os.path.isdir(root):
                    continue
                try:
                    for item in sorted(os.listdir(root)):
                        path = os.path.join(root, item)
                        if os.path.isdir(path) and self._is_model_complete(path):
                            name = f"local:{item}"
                            if name not in available:
                                available.insert(0, name)
                except OSError:
                    continue
        return available
