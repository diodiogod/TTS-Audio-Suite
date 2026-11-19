"""
ZONOS3-Mini model downloader.

Downloads the native Zyphra/ZONOS3-Mini files into ComfyUI/models/TTS/zonos3_mini/ZONOS3-Mini/,
keeping the repository layout that the vendored reference implementation expects.
"""

import os
from typing import Dict, Iterable, List, Tuple

from utils.models.extra_paths import get_all_tts_model_paths, get_preferred_download_path
from utils.security.path_access import allowed_path, validate_model_paths

REPO_ID = "Zyphra/ZONOS3-Mini"
REVISION = "d2d142a86b1e6aeb4252679673c86ecd86177bf8"
MODEL_NAME = "ZONOS3-Mini"
ENGINE_FOLDER = "zonos3_mini"

# feature -> [(path inside the repo, exact size in bytes)]
FILES: Dict[str, List[Tuple[str, int]]] = {
    "base": [
        ("lm/zonos3_mini/model.pth", 1772391535),
        ("lm/zonos3_mini/params.json", 2500),
        ("lm/zonos3_mini/tokenizer/superbpe8192.json", 559287),
        ("lm/zonos3_mini/text_rows/pause_tokens_s5000.pt", 589771),
        ("autoencoder/generator.pt", 7960647220),
        ("speaker_encoders/ownmect_s2/ckpt_0037.pth", 50277003),
    ],
    "keep2_plain": [
        ("autoencoder/heads/keep2_plain/config_full_unfreezeall_mr_drop30_4x10.json", 10304),
        ("autoencoder/heads/keep2_plain/step_0050000.pt", 518498361),
    ],
    "full_mr": [
        ("autoencoder/heads/full_mr/config_full_unfreezeall_mr.json", 11828),
        ("autoencoder/heads/full_mr/step_0050000.pt", 4012134189),
    ],
    "teacher": [
        ("autoencoder/teacher/config.json", 1863),
        ("autoencoder/teacher/model.safetensors", 774162368),
    ],
}


class Zonos3MiniDownloader:
    """Resolve and download the ZONOS3-Mini model folder, one feature at a time."""

    @validate_model_paths
    def model_root(self, model_identifier: str = MODEL_NAME) -> str:
        model_identifier = model_identifier or MODEL_NAME
        if model_identifier.startswith("local:"):
            name = model_identifier[6:]
            for base in get_all_tts_model_paths("TTS"):
                candidate = os.path.join(base, ENGINE_FOLDER, name)
                if os.path.isdir(candidate):
                    return candidate
            return os.path.join(get_preferred_download_path(model_type="TTS", engine_name=ENGINE_FOLDER), name)
        if model_identifier != MODEL_NAME:
            raise ValueError(f"Unknown ZONOS3-Mini model identifier: {model_identifier}")
        for base in get_all_tts_model_paths("TTS"):
            candidate = os.path.join(base, ENGINE_FOLDER, MODEL_NAME)
            if os.path.isdir(candidate):
                return candidate
        return os.path.join(get_preferred_download_path(model_type="TTS", engine_name=ENGINE_FOLDER), MODEL_NAME)

    @staticmethod
    def missing(root: str, features: Iterable[str]) -> List[str]:
        """Repo paths of the requested features that are absent or have the wrong size under ``root``."""
        absent = []
        for feature in features:
            for rel_path, size in FILES[feature]:
                path = os.path.join(root, rel_path)
                if not os.path.isfile(path) or os.path.getsize(path) != size:
                    absent.append(rel_path)
        return absent

    def _download(self, root: str, rel_paths: List[str]) -> None:
        from utils.downloads.unified_downloader import UnifiedDownloader

        total = sum(size for files in FILES.values() for path, size in files if path in rel_paths)
        print(f"📦 ZONOS3-Mini: downloading {len(rel_paths)} file(s), {total / 1e9:.2f} GB, from {REPO_ID}")
        print(f"   Target: {root}")
        UnifiedDownloader().download_huggingface_model(
            repo_id=REPO_ID,
            model_name=MODEL_NAME,
            # a file that exists with the wrong size is a broken or partial download: fetch it again
            files=[
                {"remote": p, "local": p, "force_download": os.path.exists(os.path.join(root, p))}
                for p in rel_paths
            ],
            engine_type=ENGINE_FOLDER,
            revision=REVISION,
            target_dir=allowed_path(root),
        )

    def ensure(self, model_identifier: str, features: Iterable[str]) -> str:
        """The model root with every file of ``features`` present; downloads only for the canonical model name."""
        features = list(features)
        model_identifier = model_identifier or MODEL_NAME
        root = self.model_root(model_identifier)
        absent = self.missing(root, features)
        if not absent:
            return root
        if model_identifier.startswith("local:"):
            raise FileNotFoundError(
                f"ZONOS3-Mini local model '{model_identifier[6:]}' is missing: {', '.join(absent)}"
            )
        self._download(root, absent)
        absent = self.missing(root, features)
        if absent:
            raise RuntimeError(f"ZONOS3-Mini download incomplete, missing: {', '.join(absent)}")
        return root

    def get_available_models(self) -> List[str]:
        models = [MODEL_NAME]
        for base in get_all_tts_model_paths("TTS"):
            engine_dir = os.path.join(base, ENGINE_FOLDER)
            if not os.path.isdir(engine_dir):
                continue
            for name in sorted(os.listdir(engine_dir)):
                entry = f"local:{name}"
                if name == MODEL_NAME or entry in models:
                    continue
                if os.path.isfile(os.path.join(engine_dir, name, "lm", "zonos3_mini", "params.json")):
                    models.append(entry)
        return models
