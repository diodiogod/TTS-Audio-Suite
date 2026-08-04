"""Strict, organized downloader for the official Hume TADA assets."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

import folder_paths
from huggingface_hub import snapshot_download

from utils.hf_download_logging import quiet_hf_download_logs
from utils.models.extra_paths import get_all_tts_model_paths, get_preferred_download_path

from .languages import normalize_tada_language, validate_tada_model_language


class TadaDownloader:
    """Download only the files used by TTS Audio Suite's local-only TADA loader."""

    MODELS = {
        "TADA-1B": {
            "repo_id": "HumeAI/tada-1b",
            "files": [
                "config.json",
                "generation_config.json",
                "model.safetensors",
            ],
        },
        "TADA-3B-ML": {
            "repo_id": "HumeAI/tada-3b-ml",
            "files": [
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        },
    }

    # Ungated redistribution whose three tokenizer files are byte-identical to
    # Meta's official Llama 3.2 1B repository. Keep the official repo as the
    # provenance reference and verify Git blob hashes after every install.
    TOKENIZER_REPO = "onnx-community/Llama-3.2-1B"
    TOKENIZER_REFERENCE_REPO = "meta-llama/Llama-3.2-1B"
    TOKENIZER_DIRECTORY = "llama-3.2-1b-tokenizer"
    TOKENIZER_FILES = [
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    TOKENIZER_DISTRIBUTION_FILES = TOKENIZER_FILES + ["LICENSE.txt", "USE_POLICY.md"]
    TOKENIZER_GIT_BLOB_SHA1 = {
        "special_tokens_map.json": "cfabacc2620186cd3dd4b1dde9a37e057208636e",
        "tokenizer.json": "5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
        "tokenizer_config.json": "cb9ec25536e44d86778b10509d3e5bdca459a5cf",
    }

    CODEC_REPO = "HumeAI/tada-codec"
    CODEC_DIRECTORY = "tada-codec"
    CODEC_BASE_FILES = [
        "encoder/config.json",
        "encoder/model.safetensors",
        "decoder/config.json",
        "decoder/model.safetensors",
    ]
    WAV2VEC_REPO = "facebook/wav2vec2-large"
    WAV2VEC_CONFIG_LOCAL = "wav2vec2-large/config.json"

    _ALIASES = {
        "tada-1b": "TADA-1B",
        "1b": "TADA-1B",
        "humeai/tada-1b": "TADA-1B",
        "tada-3b-ml": "TADA-3B-ML",
        "tada-3b": "TADA-3B-ML",
        "3b": "TADA-3B-ML",
        "3b-ml": "TADA-3B-ML",
        "humeai/tada-3b-ml": "TADA-3B-ML",
    }

    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            try:
                base_path = get_preferred_download_path(model_type="TTS", engine_name="tada")
            except Exception:
                base_path = os.path.join(folder_paths.models_dir, "TTS", "tada")
        self.base_path = os.path.abspath(str(base_path))
        os.makedirs(self.base_path, exist_ok=True)

    @classmethod
    def canonical_model_name(cls, model_identifier: str) -> Optional[str]:
        value = str(model_identifier or "TADA-1B").strip()
        if value in cls.MODELS:
            return value
        return cls._ALIASES.get(value.lower().replace("_", "-"))

    @staticmethod
    def _files_ready(target_dir: str, files: List[str]) -> bool:
        return all(
            os.path.isfile(os.path.join(target_dir, name))
            and os.path.getsize(os.path.join(target_dir, name)) > 0
            for name in files
        )

    @staticmethod
    def _git_blob_sha1(path: str) -> str:
        size = os.path.getsize(path)
        digest = hashlib.sha1()
        digest.update(f"blob {size}\0".encode("ascii"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _validate_tokenizer_files(cls, target_dir: str) -> None:
        mismatches = []
        for filename, expected in cls.TOKENIZER_GIT_BLOB_SHA1.items():
            path = os.path.join(target_dir, filename)
            actual = cls._git_blob_sha1(path)
            if actual != expected:
                mismatches.append(f"{filename} ({actual})")
        if mismatches:
            raise RuntimeError(
                "The installed TADA tokenizer does not match Meta's official Llama 3.2 1B "
                "tokenizer. Remove the tokenizer directory and retry. Mismatched files: "
                + ", ".join(mismatches)
            )

    @staticmethod
    def _download_files(
        repo_id: str,
        target_dir: str,
        files: List[str],
        label: str,
        force: bool,
    ) -> str:
        if not force and TadaDownloader._files_ready(target_dir, files):
            return target_dir
        os.makedirs(target_dir, exist_ok=True)
        print(f"\n{'=' * 60}")
        print("📦 TADA Asset Download")
        print(f"{'=' * 60}")
        print(f"Asset: {label}")
        print(f"Repository: {repo_id}")
        print(f"Target: {target_dir}")
        print(f"Files: {len(files)}")
        print(f"{'=' * 60}\n")
        try:
            with quiet_hf_download_logs():
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_dir,
                    allow_patterns=files,
                    force_download=force,
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to download {label} from {repo_id}: {exc}") from exc

        if not TadaDownloader._files_ready(target_dir, files):
            missing = [name for name in files if not os.path.isfile(os.path.join(target_dir, name))]
            raise RuntimeError(
                f"TADA asset download for {label} is incomplete. Missing files: {missing}"
            )
        print(f"✅ TADA asset ready: {target_dir}")
        return target_dir

    def _search_roots(self) -> List[str]:
        roots = [self.base_path]
        try:
            roots.extend(os.path.join(path, "tada") for path in get_all_tts_model_paths("TTS"))
        except Exception:
            pass
        seen = set()
        return [path for path in roots if not (path in seen or seen.add(path))]

    def _resolve_local_model(self, identifier: str) -> str:
        value = str(identifier).removeprefix("local:")
        if os.path.isdir(value):
            return os.path.abspath(value)
        for root in self._search_roots():
            candidate = os.path.join(root, value)
            if os.path.isdir(candidate):
                return candidate
        raise FileNotFoundError(
            f"Local TADA checkpoint '{value}' was not found under the configured TTS/tada paths."
        )

    @classmethod
    def is_model_complete(cls, model_path: str) -> bool:
        if not os.path.isfile(os.path.join(model_path, "config.json")):
            return False
        try:
            return any(
                name.endswith(".safetensors")
                and os.path.getsize(os.path.join(model_path, name)) > 0
                for name in os.listdir(model_path)
            )
        except OSError:
            return False

    def get_available_models(self) -> List[str]:
        models = list(self.MODELS)
        local_names = set()
        for root in self._search_roots():
            if not os.path.isdir(root):
                continue
            try:
                entries = os.listdir(root)
            except OSError:
                continue
            for entry in entries:
                candidate = os.path.join(root, entry)
                if entry not in self.MODELS and self.is_model_complete(candidate):
                    local_names.add(f"local:{entry}")
        return models + sorted(local_names)

    def ensure_model(self, model_identifier: str, force: bool = False) -> tuple[str, str]:
        canonical = self.canonical_model_name(model_identifier)
        if canonical is None:
            model_path = self._resolve_local_model(model_identifier)
            if not self.is_model_complete(model_path):
                raise RuntimeError(f"Local TADA checkpoint is incomplete: {model_path}")
            return os.path.basename(os.path.normpath(model_path)), model_path

        details = self.MODELS[canonical]
        target = os.path.join(self.base_path, canonical)
        try:
            self._download_files(
                details["repo_id"], target, details["files"], canonical, force
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to download the public {canonical} checkpoint. Check the network, "
                f"or manually place the required checkpoint files in "
                f"ComfyUI/models/TTS/tada/{canonical}/, "
                f"then retry. Download error: {exc}"
            ) from exc
        return canonical, target

    def ensure_tokenizer(self, force: bool = False) -> str:
        target = os.path.join(self.base_path, self.TOKENIZER_DIRECTORY)
        if not force and self._files_ready(target, self.TOKENIZER_FILES):
            self._validate_tokenizer_files(target)
            return target
        try:
            result = self._download_files(
                self.TOKENIZER_REPO,
                target,
                self.TOKENIZER_DISTRIBUTION_FILES,
                "Llama-3.2-1B tokenizer",
                force,
            )
            self._validate_tokenizer_files(result)
            return result
        except Exception as exc:
            raise RuntimeError(
                "Unable to download the TADA tokenizer automatically from the exact ungated "
                f"redistribution https://huggingface.co/{self.TOKENIZER_REPO}.\n\n"
                "Manual alternative: copy these three exact Meta Llama 3.2 1B files:\n"
                "ComfyUI/models/TTS/tada/\n"
                "└── llama-3.2-1b-tokenizer/\n"
                "    ├── tokenizer.json\n"
                "    ├── tokenizer_config.json\n"
                "    └── special_tokens_map.json\n\n"
                f"Download error: {exc}"
            ) from exc

    def ensure_codec_assets(
        self,
        language: object = None,
        force: bool = False,
        codec_path: Optional[str] = None,
    ) -> str:
        code = normalize_tada_language(language)
        target = os.path.abspath(codec_path or os.path.join(self.base_path, self.CODEC_DIRECTORY))
        aligner_dir = "aligner" if code is None else f"aligner-{code}"
        codec_files = self.CODEC_BASE_FILES + [
            f"{aligner_dir}/config.json",
            f"{aligner_dir}/model.safetensors",
        ]
        self._download_files(self.CODEC_REPO, target, codec_files, "TADA codec", force)

        wav2vec_file = os.path.join(target, self.WAV2VEC_CONFIG_LOCAL)
        if force or not os.path.isfile(wav2vec_file) or os.path.getsize(wav2vec_file) == 0:
            self._download_files(
                self.WAV2VEC_REPO,
                os.path.dirname(wav2vec_file),
                ["config.json"],
                "TADA Wav2Vec aligner configuration",
                force,
            )
        return target

    def resolve_model_assets(
        self,
        model_identifier: str,
        language: object = None,
        force: bool = False,
    ) -> Dict[str, str]:
        canonical = self.canonical_model_name(model_identifier)
        validation_name = canonical or os.path.basename(str(model_identifier))
        validate_tada_model_language(validation_name, language)
        # Validate the small tokenizer first so source/hash failures happen
        # before several gigabytes of TADA assets are downloaded.
        tokenizer_path = self.ensure_tokenizer(force=force)
        model_name, model_path = self.ensure_model(model_identifier, force=force)
        codec_path = self.ensure_codec_assets(language=language, force=force)
        return {
            "model_name": model_name,
            "model_path": model_path,
            "codec_path": codec_path,
            "tokenizer_path": tokenizer_path,
        }


tada_downloader = TadaDownloader()


__all__ = ["TadaDownloader", "tada_downloader"]
