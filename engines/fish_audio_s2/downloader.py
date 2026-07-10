"""Organized Hugging Face downloader for the gated Fish Audio S2 Pro model."""

import os
import shutil

import folder_paths

from utils.downloads.unified_downloader import unified_downloader


class FishAudioS2Downloader:
    VARIANTS = {
        "s2-pro": {
            "repo_id": "fishaudio/s2-pro",
            "directory": "fish_audio_s2_pro",
            "files": [
                ".gitattributes", "LICENSE.md", "README.md", "chat_template.jinja", "codec.pth",
                "config.json", "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors", "model.safetensors.index.json",
                "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
            ],
        },
        "s2-pro-fp8": {
            "repo_id": "drbaph/s2-pro-fp8",
            "directory": "fish_audio_s2_pro_fp8",
            "files": [
                ".gitattributes", "LICENSE.md", "README.md", "chat_template.jinja", "codec.pth",
                "config.json", "model.safetensors", "quantization_info.json",
                "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
            ],
        },
    }

    @classmethod
    def is_model_complete(cls, model_path: str) -> bool:
        """Recognize a local Fish S2 checkpoint without loading model code."""
        if not os.path.isdir(model_path):
            return False
        required = ("config.json", "codec.pth", "tokenizer.json")
        if not all(os.path.isfile(os.path.join(model_path, name)) for name in required):
            return False
        try:
            return any(
                name.endswith(".safetensors")
                and os.path.getsize(os.path.join(model_path, name)) > 0
                for name in os.listdir(model_path)
            )
        except OSError:
            return False

    @classmethod
    def local_model_paths(cls) -> list[str]:
        """Return complete local Fish checkpoints from all configured TTS paths."""
        try:
            from utils.models.extra_paths import get_all_tts_model_paths
            search_paths = get_all_tts_model_paths("TTS")
        except Exception:
            search_paths = [os.path.join(folder_paths.models_dir, "TTS")]

        found = {}
        for base_path in search_paths:
            if not os.path.isdir(base_path):
                continue
            try:
                entries = os.listdir(base_path)
            except OSError:
                continue
            for entry in entries:
                candidate = os.path.join(base_path, entry)
                if cls.is_model_complete(candidate):
                    found.setdefault(entry, candidate)
        return [found[name] for name in sorted(found)]

    @classmethod
    def resolve_local_model_path(cls, selection: str) -> str:
        """Resolve a `local:<folder>` UI selection to its filesystem path."""
        name = str(selection).removeprefix("local:")
        for path in cls.local_model_paths():
            if os.path.basename(os.path.normpath(path)) == name:
                return path
        raise FileNotFoundError(f"Local Fish Audio S2 model not found: {selection}")

    @classmethod
    def resolve_model_variant(cls, selection: str, model_path: str | None = None) -> str:
        """Map a UI selection/path to the runtime's known variant behavior."""
        if selection in cls.VARIANTS:
            return selection
        path = model_path or selection
        if os.path.isfile(os.path.join(path, "quantization_info.json")) or "fp8" in os.path.basename(path).lower():
            return "s2-pro-fp8"
        return "s2-pro"

    @classmethod
    def resolve_model_path(cls, selection: str) -> str:
        """Resolve local selections or download the selected official variant."""
        if str(selection).startswith("local:"):
            return cls.resolve_local_model_path(selection)
        return cls.ensure_model(selection)

    @classmethod
    def model_dir(cls, variant: str = "s2-pro") -> str:
        if variant not in cls.VARIANTS:
            raise ValueError(f"Unknown Fish Audio S2 model variant: {variant}")
        return os.path.join(folder_paths.models_dir, "TTS", cls.VARIANTS[variant]["directory"])

    @classmethod
    def ensure_model(cls, variant: str = "s2-pro") -> str:
        details = cls.VARIANTS.get(variant)
        if details is None:
            raise ValueError(f"Unknown Fish Audio S2 model variant: {variant}")
        target = cls.model_dir(variant)
        if variant == "s2-pro-fp8":
            target_codec = os.path.join(target, "codec.pth")
            official_codec = os.path.join(cls.model_dir("s2-pro"), "codec.pth")
            if not os.path.isfile(target_codec) and os.path.isfile(official_codec):
                os.makedirs(target, exist_ok=True)
                try:
                    os.link(official_codec, target_codec)
                except OSError:
                    shutil.copy2(official_codec, target_codec)
        files = details["files"]
        if all(os.path.isfile(os.path.join(target, name)) and os.path.getsize(os.path.join(target, name)) > 0
               for name in files):
            return target
        os.makedirs(target, exist_ok=True)
        download_files = [{"remote": name, "local": name} for name in files]
        try:
            result = unified_downloader.download_huggingface_model(
                repo_id=details["repo_id"],
                model_name=variant,
                files=download_files,
                engine_type="fish_audio_s2",
                target_dir=target,
            )
            if not result:
                raise RuntimeError("one or more required model files failed to download")
        except Exception as exc:
            raise RuntimeError(
                f"Fish Audio S2 variant '{variant}' is gated. Accept its non-commercial license at "
                f"https://huggingface.co/{details['repo_id']} and authenticate Hugging Face "
                f"before retrying. Download error: {exc}"
            ) from exc
        return target
