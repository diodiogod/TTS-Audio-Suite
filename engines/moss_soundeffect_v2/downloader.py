"""Download MOSS-SoundEffect v2 into the suite's organized model folder."""

import os
from typing import Optional

import folder_paths

from utils.hf_download_logging import quiet_hf_download_logs
from utils.models.extra_paths import get_all_tts_model_paths, get_preferred_download_path


class MossSoundEffectV2Downloader:
    MODEL_NAME = "MOSS-SoundEffect-v2.0"
    REPO_ID = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
    REQUIRED_FILES = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/merges.txt",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/vae_128d_48k.pth",
    ]

    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            try:
                base_path = get_preferred_download_path("TTS", "moss_soundeffect_v2")
            except Exception:
                base_path = os.path.join(folder_paths.models_dir, "TTS", "moss_soundeffect_v2")
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    @classmethod
    def is_complete(cls, model_dir: str) -> bool:
        return os.path.isdir(model_dir) and all(
            os.path.isfile(os.path.join(model_dir, rel_path)) for rel_path in cls.REQUIRED_FILES
        )

    def resolve_model_path(self, identifier: str = MODEL_NAME) -> str:
        identifier = identifier or self.MODEL_NAME
        if os.path.isabs(identifier):
            if self.is_complete(identifier):
                return identifier
            raise FileNotFoundError(f"Incomplete MOSS-SoundEffect v2 model folder: {identifier}")

        local_name = identifier.removeprefix("local:")
        for base_path in get_all_tts_model_paths("TTS"):
            candidate = os.path.join(base_path, "moss_soundeffect_v2", local_name)
            if self.is_complete(candidate):
                print(f"📁 Using local MOSS-SoundEffect v2 model: {candidate}")
                return candidate

        if identifier.startswith("local:"):
            raise FileNotFoundError(f"Local MOSS-SoundEffect v2 model not found: {local_name}")
        return self.download_model()

    def download_model(self, force: bool = False) -> str:
        model_dir = os.path.join(self.base_path, self.MODEL_NAME)
        print(f"\n{'=' * 60}")
        print("📦 MOSS-SoundEffect v2 Model Download")
        print(f"{'=' * 60}")
        print(f"Model: {self.MODEL_NAME}")
        print("Description: MOSS v2 text-to-audio diffusion model (48 kHz, up to 30 seconds)")
        print(f"Repository: {self.REPO_ID}")
        print(f"Target: {model_dir}")
        print(f"{'=' * 60}\n")
        try:
            from huggingface_hub import snapshot_download

            with quiet_hf_download_logs():
                snapshot_download(
                    repo_id=self.REPO_ID,
                    local_dir=model_dir,
                    force_download=force,
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to download {self.REPO_ID}: {exc}") from exc
        if not self.is_complete(model_dir):
            raise RuntimeError(f"Downloaded MOSS-SoundEffect v2 model is incomplete: {model_dir}")
        print(f"\n✅ MOSS-SoundEffect v2 download complete: {model_dir}")
        return model_dir
