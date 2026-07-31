"""Organized model download and discovery for official Audio8 TTS."""

from __future__ import annotations

import os
from typing import List, Optional

import folder_paths

from utils.downloads.unified_downloader import unified_downloader
from utils.models.extra_paths import (
    get_all_tts_model_paths,
    get_preferred_download_path,
)


class Audio8TTSDownloader:
    """Resolve Audio8 checkpoints without using the Hugging Face model cache."""

    MODEL_NAME = "Audio8-TTS-Preview-0.6b"
    REPO_ID = "Audio8/Audio8-TTS-Preview-0.6b"
    REVISION = "1b17c91db5f4dccb6914aa4aa5cb0e56661a6c17"

    # Complete runtime snapshot at the pinned revision. Non-runtime model-card
    # assets are deliberately excluded.
    REQUIRED_FILES = [
        "codec.pth",
        "config.json",
        "configuration_arktts.py",
        "generation_config.json",
        "model.safetensors",
        "modeling_arktts.py",
        "modeling_arktts_codec.py",
        "preprocessor_config.json",
        "processing_arktts.py",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            try:
                self.base_path = get_preferred_download_path(
                    model_type="TTS",
                    engine_name="audio8_tts",
                )
            except Exception:
                self.base_path = os.path.join(
                    folder_paths.models_dir,
                    "TTS",
                    "audio8_tts",
                )
        else:
            self.base_path = os.path.abspath(os.fspath(base_path))
        os.makedirs(self.base_path, exist_ok=True)

    def get_available_models(self) -> List[str]:
        """Return the canonical model plus complete local installations."""
        models = [self.MODEL_NAME]
        canonical_path = os.path.normcase(
            os.path.abspath(os.path.join(self.base_path, self.MODEL_NAME))
        )
        for base_path in get_all_tts_model_paths("TTS"):
            for folder_name in ("audio8_tts", "Audio8_TTS", ""):
                root = (
                    os.path.join(base_path, folder_name) if folder_name else base_path
                )
                if not os.path.isdir(root):
                    continue
                for item in sorted(os.listdir(root)):
                    candidate = os.path.join(root, item)
                    if os.path.normcase(os.path.abspath(candidate)) == canonical_path:
                        continue
                    local_name = f"local:{item}"
                    if local_name not in models and self._is_model_complete(candidate):
                        models.append(local_name)
        return models

    def resolve_model_path(
        self,
        model_identifier: str = MODEL_NAME,
    ) -> str:
        """Resolve an absolute path, ``local:`` name, or canonical model name."""
        model_identifier = str(model_identifier or self.MODEL_NAME).strip()

        if os.path.isabs(model_identifier) or os.path.isdir(model_identifier):
            candidate = os.path.abspath(model_identifier)
            if self._is_model_complete(candidate, verbose=True):
                return candidate
            raise FileNotFoundError(
                f"Audio8 TTS model path is missing required files: {candidate}"
            )

        if model_identifier.startswith("local:"):
            local_name = model_identifier[6:].strip()
            if not local_name:
                raise ValueError("Audio8 TTS local model name must not be empty")
            for base_path in get_all_tts_model_paths("TTS"):
                for folder_name in ("audio8_tts", "Audio8_TTS", ""):
                    candidate = (
                        os.path.join(base_path, folder_name, local_name)
                        if folder_name
                        else os.path.join(base_path, local_name)
                    )
                    if self._is_model_complete(candidate):
                        print(f"📁 Using local Audio8 TTS model: {candidate}")
                        return candidate
            raise FileNotFoundError(
                f"Local Audio8 TTS model not found or incomplete: {local_name}"
            )

        if model_identifier != self.MODEL_NAME:
            raise ValueError(f"Unknown Audio8 TTS model: {model_identifier}")
        return self.get_model_path()

    def get_model_path(self, model_name: str = MODEL_NAME) -> str:
        """Return the organized canonical model path, downloading if needed."""
        if model_name != self.MODEL_NAME:
            raise ValueError(f"Unknown Audio8 TTS model: {model_name}")
        model_dir = os.path.join(self.base_path, self.MODEL_NAME)
        if not self._is_model_complete(model_dir):
            return self.download_model(model_dir)
        return model_dir

    def download_model(
        self,
        model_dir: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """Download the pinned official snapshot into the organized model folder."""
        model_dir = model_dir or os.path.join(self.base_path, self.MODEL_NAME)

        print("\n" + "=" * 60)
        print("📦 Audio8 TTS Model Download")
        print("=" * 60)
        print(f"Repository: {self.REPO_ID}")
        print(f"Revision: {self.REVISION}")
        print(f"Target: {model_dir}")
        print("=" * 60 + "\n")

        unified_downloader.download_huggingface_snapshot(
            repo_id=self.REPO_ID,
            target_dir=model_dir,
            revision=self.REVISION,
            allow_patterns=self.REQUIRED_FILES,
            required_files=self.REQUIRED_FILES,
            force_download=bool(force),
            description=self.MODEL_NAME,
        )
        if not self._is_model_complete(model_dir, verbose=True):
            raise RuntimeError(
                f"Downloaded Audio8 TTS model is incomplete: {model_dir}"
            )
        print(f"✅ Audio8 TTS model ready: {model_dir}")
        return model_dir

    def _is_model_complete(
        self,
        model_dir: str,
        *,
        verbose: bool = False,
    ) -> bool:
        if not os.path.isdir(model_dir):
            return False
        missing = [
            rel_path
            for rel_path in self.REQUIRED_FILES
            if not os.path.isfile(os.path.join(model_dir, rel_path))
            or os.path.getsize(os.path.join(model_dir, rel_path)) <= 0
        ]
        if missing and verbose:
            print(
                f"❌ Audio8 TTS model incomplete. Missing or empty "
                f"{len(missing)} file(s):"
            )
            for rel_path in missing:
                print(f"   - {rel_path}")
        return not missing
