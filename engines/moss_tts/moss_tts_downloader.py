"""
MOSS-TTS model downloader.

Downloads official OpenMOSS Hugging Face repositories into
ComfyUI/models/TTS/moss_tts/ instead of the default HF cache.
"""

import os
from typing import Dict, Optional

import folder_paths
from utils.models.extra_paths import get_all_tts_model_paths, get_preferred_download_path


class MossTTSDownloader:
    """Resolve and download official MOSS-TTS model folders."""

    MODELS: Dict[str, Dict[str, str]] = {
        "MOSS-TTS": {
            "repo_id": "OpenMOSS-Team/MOSS-TTS",
            "description": "MOSS-TTS Delay 8B - flagship long-form model",
        },
        "MOSS-TTS-Local-Transformer": {
            "repo_id": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
            "description": "MOSS-TTS Local 1.7B - smaller/faster model",
        },
        "MOSS-Audio-Tokenizer": {
            "repo_id": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
            "description": "MOSS audio tokenizer required by MOSS-TTS",
        },
    }

    REQUIRED_FILES = {
        "MOSS-TTS": [
            "config.json",
            "processor_config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            "modeling_moss_tts.py",
            "processing_moss_tts.py",
            "configuration_moss_tts.py",
        ],
        "MOSS-TTS-Local-Transformer": [
            "config.json",
            "generation_config.json",
            "processor_config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "modeling_moss_tts.py",
            "processing_moss_tts.py",
            "configuration_moss_tts.py",
        ],
        "MOSS-Audio-Tokenizer": [
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "modeling_moss_audio_tokenizer.py",
            "configuration_moss_audio_tokenizer.py",
        ],
    }

    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            try:
                self.base_path = get_preferred_download_path(model_type="TTS", engine_name="moss_tts")
            except Exception:
                self.base_path = os.path.join(folder_paths.models_dir, "TTS", "moss_tts")
        else:
            self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def resolve_model_path(self, model_identifier: str) -> str:
        """Resolve local: names, absolute paths, or known MOSS model identifiers."""
        if not model_identifier:
            model_identifier = "MOSS-TTS-Local-Transformer"

        if os.path.isabs(model_identifier) and os.path.exists(model_identifier):
            return model_identifier

        if model_identifier.startswith("local:"):
            local_name = model_identifier[6:]
            for base_path in get_all_tts_model_paths("TTS"):
                for folder_name in ("moss_tts", "MOSS-TTS"):
                    candidate = os.path.join(base_path, folder_name, local_name)
                    if os.path.exists(candidate):
                        print(f"📁 Using local MOSS-TTS model: {candidate}")
                        return candidate
            raise FileNotFoundError(f"Local MOSS-TTS model not found: {local_name}")

        return self.get_model_path(model_identifier)

    def get_model_path(self, model_name: str) -> str:
        if os.path.isabs(model_name) and os.path.exists(model_name):
            return model_name

        model_dir = os.path.join(self.base_path, model_name)
        if model_name in self.MODELS:
            if not os.path.exists(model_dir) or not os.listdir(model_dir):
                return self.download_model(model_name)
            if not self._is_model_complete(model_name, model_dir):
                print(f"⚠️ Incomplete MOSS-TTS model detected: {model_dir}")
                return self.download_model(model_name, force=True)
        elif not os.path.exists(model_dir):
            raise FileNotFoundError(f"MOSS-TTS model not found: {model_name}")

        return model_dir

    def download_model(self, model_name: str, force: bool = False) -> str:
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown MOSS-TTS model: {model_name}")

        model_info = self.MODELS[model_name]
        repo_id = model_info["repo_id"]
        model_dir = os.path.join(self.base_path, model_name)

        print(f"\n{'=' * 60}")
        print("📦 MOSS-TTS Model Download")
        print(f"{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"Description: {model_info['description']}")
        print(f"Repository: {repo_id}")
        print(f"Target: {model_dir}")
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
            raise RuntimeError(f"Failed to download {model_name} from {repo_id}: {e}") from e

        if not self._is_model_complete(model_name, model_dir):
            raise RuntimeError(f"Downloaded MOSS-TTS model is incomplete: {model_dir}")

        print(f"✅ MOSS-TTS model ready: {model_dir}")
        return model_dir

    def _is_model_complete(self, model_name: str, model_dir: str) -> bool:
        if not os.path.isdir(model_dir):
            return False

        missing = [
            rel_path
            for rel_path in self.REQUIRED_FILES.get(model_name, ["config.json"])
            if not os.path.exists(os.path.join(model_dir, rel_path))
        ]
        if missing:
            print(f"❌ MOSS-TTS model incomplete. Missing {len(missing)} file(s):")
            for rel_path in missing[:20]:
                print(f"   - {rel_path}")
            if len(missing) > 20:
                print(f"   ... and {len(missing) - 20} more")
            return False
        return True
