"""
Granite ASR Model Downloader

Handles automatic download and setup of Granite ASR models using the unified download system.
"""

import os
import sys
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


class GraniteASRDownloader:
    """Downloader for Granite ASR models using unified download system."""

    MODELS = {
        "granite-4.0-1b-speech": {
            "repo_id": "ibm-granite/granite-4.0-1b-speech",
            "files": [
                "config.json",
                "processor_config.json",
                "preprocessor_config.json",
                "chat_template.jinja",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "model.safetensors.index.json",
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
            ],
            "description": "IBM Granite 4.0 1B multilingual ASR/AST model"
        }
    }

    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            try:
                self.base_path = get_preferred_download_path(model_type="TTS", engine_name="granite_asr")
            except Exception:
                self.base_path = os.path.join(folder_paths.models_dir, "TTS", "granite_asr")
        else:
            self.base_path = base_path

        self.downloader = unified_downloader

    def download_model(
        self,
        model_name: str = "granite-4.0-1b-speech",
        force_download: bool = False,
        **kwargs
    ) -> str:
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown Granite model: {model_name}. Available: {list(self.MODELS.keys())}")

        model_info = self.MODELS[model_name]
        model_path = os.path.join(self.base_path, model_name)

        if not force_download and os.path.exists(model_path):
            try:
                self._verify_model(model_path, model_name, verbose=False)
                return model_path
            except RuntimeError:
                pass

        print(f"📥 Downloading Granite ASR model: {model_name}")
        print(f"📁 Target directory: {model_path}")

        file_list = [{"remote": file_name, "local": file_name} for file_name in model_info["files"]]
        result_path = self.downloader.download_huggingface_model(
            repo_id=model_info["repo_id"],
            model_name=model_name,
            files=file_list,
            engine_type="granite_asr",
            **kwargs,
        )

        if not result_path:
            raise RuntimeError("Granite HuggingFace download failed")

        self._verify_model(result_path, model_name)
        print(f"✅ Granite model downloaded successfully")
        print(f"📁 Model path: {result_path}")
        return result_path

    def _verify_model(self, model_path: str, model_name: str, verbose: bool = True) -> None:
        if model_name not in self.MODELS:
            raise RuntimeError(f"Unknown Granite model: {model_name}")

        missing_files = []
        for file_name in self.MODELS[model_name]["files"]:
            if not os.path.exists(os.path.join(model_path, file_name)):
                missing_files.append(file_name)

        if missing_files:
            raise RuntimeError(f"Granite model verification failed. Missing files: {missing_files}")

        if verbose:
            print("✅ Granite model verification passed")

    def resolve_model_path(self, model_identifier: str) -> str:
        if not model_identifier:
            model_identifier = "granite-4.0-1b-speech"

        if model_identifier.startswith("local:"):
            model_name = model_identifier[6:]
            local_path = os.path.join(self.base_path, model_name)
            if os.path.exists(local_path):
                return local_path
            raise FileNotFoundError(f"Local Granite model '{model_name}' not found in {self.base_path}")

        model_dir = os.path.join(self.base_path, model_identifier)
        if os.path.exists(model_dir):
            try:
                self._verify_model(model_dir, model_identifier, verbose=False)
                return model_dir
            except RuntimeError:
                pass

        if model_identifier in self.MODELS:
            return self.download_model(model_identifier)

        return model_identifier

    def get_model_info(self, model_name: str = "granite-4.0-1b-speech") -> Optional[Dict[str, Any]]:
        return self.MODELS.get(model_name)
