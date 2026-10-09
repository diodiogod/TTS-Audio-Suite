"""Organized model download and discovery for official DramaBox."""

import os
from typing import Dict, List, Optional

import folder_paths

from utils.downloads.unified_downloader import unified_downloader
from utils.models.extra_paths import get_all_tts_model_paths, get_preferred_download_path


from utils.security.path_access import validate_model_paths, allowed_path

class DramaBoxDownloader:
    """Resolve DramaBox checkpoints without using Hugging Face cache storage."""

    MODEL_NAME = "DramaBox"
    DRAMABOX_REPO = "ResembleAI/Dramabox"
    GEMMA_REPO = "unsloth/gemma-3-12b-it-bnb-4bit"

    DRAMABOX_FILES = [
        "dramabox-dit-v1.safetensors",
        "dramabox-audio-components.safetensors",
        "assets/silence_latent_frame.pt",
    ]
    GEMMA_FILES = [
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "chat_template.json",
    ]

    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            try:
                self.base_path = get_preferred_download_path(
                    model_type="TTS", engine_name="dramabox"
                )
            except Exception:
                self.base_path = os.path.join(folder_paths.models_dir, "TTS", "dramabox")
        else:
            self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def get_available_models(self) -> List[str]:
        models = [self.MODEL_NAME]
        for base_path in get_all_tts_model_paths("TTS"):
            for folder_name in ("dramabox", "DramaBox"):
                root = os.path.join(base_path, folder_name)
                if not os.path.isdir(root):
                    continue
                for item in sorted(os.listdir(root)):
                    candidate = os.path.join(root, item)
                    if os.path.isdir(candidate) and self._is_model_complete(candidate):
                        local_name = f"local:{item}"
                        if local_name not in models:
                            models.insert(0, local_name)
        return models

    @validate_model_paths
    def resolve_model_path(self, model_identifier: str = MODEL_NAME) -> Dict[str, str]:
        model_identifier = model_identifier or self.MODEL_NAME
        if os.path.isabs(model_identifier) and os.path.isdir(model_identifier):
            return self._paths_for(model_identifier)

        if model_identifier.startswith("local:"):
            local_name = model_identifier[6:]
            for base_path in get_all_tts_model_paths("TTS"):
                for folder_name in ("dramabox", "DramaBox"):
                    candidate = os.path.join(base_path, folder_name, local_name)
                    if self._is_model_complete(candidate):
                        print(f"📁 Using local DramaBox model: {candidate}")
                        return self._paths_for(candidate)
            raise FileNotFoundError(f"Local DramaBox model not found or incomplete: {local_name}")

        if model_identifier != self.MODEL_NAME:
            raise ValueError(f"Unknown DramaBox model: {model_identifier}")

        model_dir = os.path.join(self.base_path, self.MODEL_NAME)
        if not self._is_model_complete(model_dir):
            self.download_model(model_dir)
        return self._paths_for(model_dir)

    def download_model(self, model_dir: Optional[str] = None) -> str:
        model_dir = model_dir or os.path.join(self.base_path, self.MODEL_NAME)
        gemma_dir = os.path.join(model_dir, "gemma-3-12b-it-bnb-4bit")

        print("\n" + "=" * 60)
        print("📦 DramaBox Model Download")
        print("=" * 60)
        print(f"DramaBox: {self.DRAMABOX_REPO}")
        print(f"Gemma encoder: {self.GEMMA_REPO}")
        print(f"Target: {model_dir}")
        print("License: LTX-2 Community License (commercial threshold applies)")
        print("=" * 60 + "\n")

        base_files = [
            {"remote": rel_path, "local": rel_path}
            for rel_path in self.DRAMABOX_FILES
        ]
        result = unified_downloader.download_huggingface_model(
            repo_id=self.DRAMABOX_REPO,
            model_name=self.MODEL_NAME,
            files=base_files,
            engine_type="dramabox",
            target_dir=model_dir,
        )
        if not result:
            raise RuntimeError("Failed to download official DramaBox weights")

        unified_downloader.download_huggingface_snapshot(
            repo_id=self.GEMMA_REPO,
            target_dir=gemma_dir,
            allow_patterns=self.GEMMA_FILES,
            required_files=self.GEMMA_FILES,
            description="DramaBox Gemma 3 12B 4-bit encoder",
        )
        if not self._is_model_complete(model_dir):
            raise RuntimeError(f"Downloaded DramaBox model is incomplete: {model_dir}")
        print(f"✅ DramaBox model ready: {model_dir}")
        return model_dir

    def _paths_for(self, model_dir: str) -> Dict[str, str]:
        return {
            "model_dir": model_dir,
            "transformer": os.path.join(model_dir, "dramabox-dit-v1.safetensors"),
            "audio_components": os.path.join(
                model_dir, "dramabox-audio-components.safetensors"
            ),
            "silence_latent": os.path.join(
                model_dir, "assets", "silence_latent_frame.pt"
            ),
            "gemma_root": os.path.join(model_dir, "gemma-3-12b-it-bnb-4bit"),
        }

    def _is_model_complete(self, model_dir: str) -> bool:
        if not os.path.isdir(model_dir):
            return False
        required = self.DRAMABOX_FILES + [
            os.path.join("gemma-3-12b-it-bnb-4bit", rel_path)
            for rel_path in self.GEMMA_FILES
        ]
        return all(os.path.isfile(os.path.join(model_dir, path)) for path in required)
