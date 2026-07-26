"""Download and inspect official VoxCPM checkpoints."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional

import folder_paths

from utils.downloads.unified_downloader import unified_downloader
from utils.hf_download_logging import quiet_hf_download_logs
from utils.models.extra_paths import (
    get_all_tts_model_paths,
    get_preferred_download_path,
)


_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "VoxCPM2": {
        "repo_id": "openbmb/VoxCPM2",
        "folder_name": "VoxCPM2",
        "architecture": "voxcpm2",
        "sample_rate": 48000,
        "legacy": False,
        "required_files": (
            "audiovae.pth",
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenization_voxcpm2.py",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    },
    "VoxCPM1.5": {
        "repo_id": "openbmb/VoxCPM1.5",
        "folder_name": "VoxCPM1.5",
        "architecture": "voxcpm",
        "sample_rate": 44100,
        "legacy": False,
        "required_files": (
            "audiovae.pth",
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    },
    "VoxCPM-0.5B": {
        "repo_id": "openbmb/VoxCPM-0.5B",
        "folder_name": "VoxCPM-0.5B",
        "architecture": "voxcpm",
        "sample_rate": 16000,
        "legacy": True,
        "required_files": (
            "audiovae.pth",
            "config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    },
}


class VoxCPMDownloader:
    """Resolve official and local VoxCPM model folders."""

    DEFAULT_MODEL = "VoxCPM2"
    MODELS = _MODEL_SPECS
    REQUIRED_FILES = {
        name: tuple(spec["required_files"])
        for name, spec in _MODEL_SPECS.items()
    }

    _COMMON_LOCAL_FILES = (
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    _AUDIO_VAE_FILES = ("audiovae.safetensors", "audiovae.pth")
    _MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            try:
                base_path = get_preferred_download_path(
                    model_type="TTS",
                    engine_name="voxcpm",
                )
            except Exception:
                base_path = os.path.join(
                    folder_paths.models_dir,
                    "TTS",
                    "voxcpm",
                )
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)

    @classmethod
    def _canonical_name(cls, selection: Optional[str]) -> Optional[str]:
        value = str(selection or cls.DEFAULT_MODEL).strip()
        for canonical, spec in cls.MODELS.items():
            if value == canonical or value == spec["repo_id"]:
                return canonical
        return None

    @staticmethod
    def _file_is_present(path: str) -> bool:
        try:
            return os.path.isfile(path) and os.path.getsize(path) > 0
        except OSError:
            return False

    @classmethod
    def _read_local_metadata(cls, model_dir: str) -> Dict[str, Any]:
        model_dir = os.path.abspath(model_dir)
        config_path = os.path.join(model_dir, "config.json")
        if not cls._file_is_present(config_path):
            raise ValueError(f"VoxCPM config is missing or empty: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid VoxCPM config: {config_path}") from exc

        architecture = str(config.get("architecture", "")).lower()
        if architecture not in {"voxcpm", "voxcpm2"}:
            raise ValueError(
                f"Unsupported or missing VoxCPM architecture in {config_path}: "
                f"{architecture or '(missing)'}"
            )

        audio_vae_config = config.get("audio_vae_config")
        if not isinstance(audio_vae_config, dict):
            audio_vae_config = {}

        if architecture == "voxcpm2":
            # AudioVAEV2 uses 48 kHz when the optional embedded VAE config is
            # omitted, so this is deterministic rather than a heuristic.
            sample_rate = audio_vae_config.get("out_sample_rate", 48000)
        else:
            # The original 0.5B runtime uses AudioVAE's 16 kHz default when
            # audio_vae_config is absent.
            sample_rate = audio_vae_config.get("sample_rate", 16000)

        try:
            sample_rate = int(sample_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid VoxCPM output sample rate in {config_path}: {sample_rate}"
            ) from exc
        if sample_rate <= 0:
            raise ValueError(
                f"Invalid VoxCPM output sample rate in {config_path}: {sample_rate}"
            )

        if architecture == "voxcpm2" and sample_rate == 48000:
            canonical = "VoxCPM2"
        elif architecture == "voxcpm" and sample_rate == 44100:
            canonical = "VoxCPM1.5"
        elif architecture == "voxcpm" and sample_rate == 16000:
            canonical = "VoxCPM-0.5B"
        else:
            canonical = f"local:{os.path.basename(model_dir)}"

        return {
            "canonical": canonical,
            "repo_id": None,
            "folder_name": os.path.basename(model_dir),
            "architecture": architecture,
            "sample_rate": sample_rate,
            "legacy": architecture == "voxcpm" and sample_rate == 16000,
        }

    @classmethod
    def _missing_local_files(cls, model_dir: str) -> List[str]:
        missing = [
            rel_path
            for rel_path in cls._COMMON_LOCAL_FILES
            if not cls._file_is_present(os.path.join(model_dir, rel_path))
        ]
        if not any(
            cls._file_is_present(os.path.join(model_dir, rel_path))
            for rel_path in cls._AUDIO_VAE_FILES
        ):
            missing.append("audiovae.safetensors | audiovae.pth")
        if not any(
            cls._file_is_present(os.path.join(model_dir, rel_path))
            for rel_path in cls._MODEL_WEIGHT_FILES
        ):
            missing.append("model.safetensors | pytorch_model.bin")
        return missing

    def get_model_spec(
        self,
        selection: Optional[str],
        resolved_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return architecture and output metadata for a model selection."""
        selection = str(selection or self.DEFAULT_MODEL).strip()
        canonical = self._canonical_name(selection)
        if canonical is not None:
            spec = dict(self.MODELS[canonical])
            spec["canonical"] = canonical
            spec["required_files"] = list(spec["required_files"])
            if resolved_path is not None:
                local_spec = self._read_local_metadata(resolved_path)
                if local_spec["architecture"] != spec["architecture"]:
                    raise ValueError(
                        f"{canonical} expects architecture {spec['architecture']}, "
                        f"but {resolved_path} declares "
                        f"{local_spec['architecture']}"
                    )
                if local_spec["sample_rate"] != spec["sample_rate"]:
                    raise ValueError(
                        f"{canonical} expects {spec['sample_rate']} Hz output, "
                        f"but {resolved_path} declares "
                        f"{local_spec['sample_rate']} Hz"
                    )
            return spec

        model_dir = resolved_path
        if model_dir is None and selection.startswith("local:"):
            model_dir = self._find_local_path(selection[6:])
        if model_dir is None and os.path.isdir(selection):
            model_dir = selection
        if model_dir is None:
            raise ValueError(f"Unknown VoxCPM model selection: {selection}")

        model_dir = os.path.abspath(model_dir)
        spec = self._read_local_metadata(model_dir)
        required_files = list(self._COMMON_LOCAL_FILES)
        required_files.extend(
            rel_path
            for rel_path in self._AUDIO_VAE_FILES
            if self._file_is_present(os.path.join(model_dir, rel_path))
        )
        required_files.extend(
            rel_path
            for rel_path in self._MODEL_WEIGHT_FILES
            if self._file_is_present(os.path.join(model_dir, rel_path))
        )
        spec["required_files"] = required_files
        return spec

    def is_model_complete(
        self,
        model_dir: str,
        spec: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Return whether a folder has every file needed by the official runtime."""
        if not os.path.isdir(model_dir):
            return False

        try:
            local_spec = self._read_local_metadata(model_dir)
        except ValueError:
            return False

        missing = self._missing_local_files(model_dir)
        if spec is not None:
            required_files = list(spec.get("required_files", ()))
            missing.extend(
                rel_path
                for rel_path in required_files
                if (
                    rel_path not in missing
                    and not self._file_is_present(
                        os.path.join(model_dir, rel_path)
                    )
                )
            )
            expected_architecture = spec.get("architecture")
            expected_sample_rate = spec.get("sample_rate")
            if (
                expected_architecture
                and local_spec["architecture"] != expected_architecture
            ):
                missing.append(
                    "config architecture "
                    f"{expected_architecture} (found {local_spec['architecture']})"
                )
            if (
                expected_sample_rate
                and local_spec["sample_rate"] != int(expected_sample_rate)
            ):
                missing.append(
                    "config output sample rate "
                    f"{expected_sample_rate} (found {local_spec['sample_rate']})"
                )
        if missing:
            print(
                f"❌ VoxCPM model incomplete or incompatible: {model_dir}"
            )
            for requirement in missing:
                print(f"   - {requirement}")
            return False
        return True

    def _search_roots(self) -> List[str]:
        roots = [self.base_path]
        try:
            configured_paths = get_all_tts_model_paths("TTS")
        except Exception:
            configured_paths = []

        for base_path in configured_paths:
            normalized = os.path.abspath(base_path)
            if os.path.basename(normalized).lower() == "voxcpm":
                roots.append(normalized)
            else:
                roots.extend(
                    (
                        os.path.join(normalized, "voxcpm"),
                        os.path.join(normalized, "VoxCPM"),
                    )
                )

        result: List[str] = []
        seen = set()
        for root in roots:
            key = os.path.normcase(os.path.normpath(root))
            if key not in seen:
                seen.add(key)
                result.append(root)
        return result

    def _find_local_path(self, local_name: str) -> str:
        local_name = str(local_name or "").strip()
        if not local_name:
            raise FileNotFoundError("A local VoxCPM model name is required")

        if os.path.isabs(local_name) and os.path.isdir(local_name):
            if self.is_model_complete(local_name):
                return os.path.abspath(local_name)
            raise FileNotFoundError(
                f"Local VoxCPM model is incomplete: {local_name}"
            )

        incomplete: List[str] = []
        for root in self._search_roots():
            candidate = os.path.join(root, local_name)
            if not os.path.isdir(candidate):
                continue
            if self.is_model_complete(candidate):
                return os.path.abspath(candidate)
            incomplete.append(candidate)

        suffix = (
            f" Incomplete candidates: {incomplete}"
            if incomplete
            else ""
        )
        raise FileNotFoundError(
            f"Local VoxCPM model not found or incomplete: {local_name}.{suffix}"
        )

    def get_available_models(self) -> List[str]:
        """Return official choices plus complete local checkpoints."""
        available = list(self.MODELS)
        official_folders = {
            str(spec["folder_name"]).casefold()
            for spec in self.MODELS.values()
        }
        local_choices: List[str] = []
        for root in self._search_roots():
            if not os.path.isdir(root):
                continue
            try:
                entries = sorted(os.listdir(root))
            except OSError:
                continue
            for entry in entries:
                if entry.casefold() in official_folders:
                    continue
                candidate = os.path.join(root, entry)
                choice = f"local:{entry}"
                if (
                    choice not in local_choices
                    and self.is_model_complete(candidate)
                ):
                    local_choices.append(choice)
        return local_choices + available

    def get_model_path(self, model_name: str) -> str:
        """Find or download an official model."""
        canonical = self._canonical_name(model_name)
        if canonical is None:
            raise ValueError(f"Unknown VoxCPM model: {model_name}")
        spec = self.get_model_spec(canonical)

        for root in self._search_roots():
            candidate = os.path.join(root, spec["folder_name"])
            if self.is_model_complete(candidate, spec):
                return os.path.abspath(candidate)

        return self.download_model(canonical)

    def resolve_model_path(self, selection: Optional[str] = None) -> str:
        """Resolve a canonical choice, exact repo ID, path, or ``local:`` name."""
        selection = str(selection or self.DEFAULT_MODEL).strip()
        canonical = self._canonical_name(selection)
        if canonical is not None:
            return self.get_model_path(canonical)

        if selection.startswith("local:"):
            return self._find_local_path(selection[6:])

        if os.path.isdir(selection):
            model_dir = os.path.abspath(selection)
            if self.is_model_complete(model_dir):
                return model_dir
            raise FileNotFoundError(
                f"VoxCPM model folder is incomplete: {model_dir}"
            )

        raise ValueError(f"Unknown VoxCPM model selection: {selection}")

    def download_model(self, model_name: str, force: bool = False) -> str:
        """Download one official checkpoint into the organized TTS folder."""
        canonical = self._canonical_name(model_name)
        if canonical is None:
            raise ValueError(f"Unknown VoxCPM model: {model_name}")
        spec = self.get_model_spec(canonical)
        model_dir = os.path.join(self.base_path, spec["folder_name"])

        if not force and self.is_model_complete(model_dir, spec):
            return os.path.abspath(model_dir)

        files = []
        for rel_path in spec["required_files"]:
            files.append(
                {
                    "remote": rel_path,
                    "local": rel_path,
                    "force_download": bool(
                        force
                        or not self._file_is_present(
                            os.path.join(model_dir, rel_path)
                        )
                    ),
                }
            )

        print("\n" + "=" * 60)
        print("📦 VoxCPM Model Download")
        print("=" * 60)
        print(f"Model: {canonical}")
        print(f"Repository: {spec['repo_id']}")
        print(f"Target: {model_dir}")
        print("=" * 60 + "\n")

        try:
            with quiet_hf_download_logs():
                result = unified_downloader.download_huggingface_model(
                    repo_id=spec["repo_id"],
                    model_name=spec["folder_name"],
                    files=files,
                    engine_type="voxcpm",
                    force_download=force,
                    target_dir=model_dir,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {canonical} from {spec['repo_id']}: {exc}"
            ) from exc

        if not result or not self.is_model_complete(model_dir, spec):
            raise RuntimeError(
                f"Downloaded VoxCPM model is incomplete: {model_dir}"
            )

        print(f"✅ VoxCPM model ready: {model_dir}")
        return os.path.abspath(model_dir)
