"""
Punctuation / Truecase runtime helpers for standalone transcript cleanup nodes.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from utils.aux_models.registry import get_punctuation_model_by_label


_MODEL_CACHE: Dict[str, object] = {}


class PunctuationModelDownloader:
    """Lazy downloader for punctuation/truecase helper models."""

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from utils.models.extra_paths import get_preferred_download_path
                self.base_path = get_preferred_download_path(model_type="TTS", engine_name="punctuation")
            except Exception:
                try:
                    import folder_paths
                    models_dir = folder_paths.models_dir
                except Exception:
                    models_dir = str(Path(__file__).resolve().parents[2] / "models")
                self.base_path = os.path.join(models_dir, "TTS", "punctuation")
        else:
            self.base_path = base_path

    def resolve_model_path(self, model_label: str) -> Tuple[Dict[str, object], str]:
        model_info = get_punctuation_model_by_label(model_label)
        model_dir = os.path.join(self.base_path, model_info["model_dir"])

        if self._is_model_ready(model_dir):
            return model_info, model_dir

        self._download_model(model_info, model_dir)
        return model_info, model_dir

    def _download_model(self, model_info: Dict[str, object], model_dir: str) -> None:
        print(f"📥 Downloading punctuation model: {model_info['name']}")
        print(f"📁 Target directory: {model_dir}")

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for punctuation model downloads. "
                "Please reinstall the suite dependencies."
            ) from exc

        snapshot_download(
            repo_id=model_info["repo_id"],
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )

        if not self._is_model_ready(model_dir):
            raise RuntimeError(f"Punctuation model download incomplete: {model_info['name']}")

        print(f"✅ Punctuation model downloaded successfully")

    @staticmethod
    def _is_model_ready(model_dir: str) -> bool:
        path = Path(model_dir)
        if not path.exists():
            return False
        return any(path.glob("*.onnx")) and any(path.glob("*.model")) and (path / "config.yaml").exists()


def _load_model(model_label: str):
    model_info, model_dir = PunctuationModelDownloader().resolve_model_path(model_label)
    cache_key = os.path.abspath(model_dir)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], model_info, model_dir

    try:
        from punctuators.models.punc_cap_seg_model import PunctCapSegConfigONNX, PunctCapSegModelONNX
    except ImportError as exc:
        raise RuntimeError(
            "The punctuation node requires the optional 'punctuators' package. "
            "Run the suite installer again or install 'punctuators' in the same Python environment."
        ) from exc

    cfg = PunctCapSegConfigONNX(
        directory=model_dir,
        spe_filename=model_info.get("spe_filename", "sp.model"),
        model_filename=model_info.get("model_filename", "model.onnx"),
        config_filename=model_info.get("config_filename", "config.yaml"),
    )
    model = PunctCapSegModelONNX(cfg=cfg)
    _MODEL_CACHE[cache_key] = model
    return model, model_info, model_dir


def _normalize_infer_output(output, sentence_mode: bool) -> str:
    if isinstance(output, list):
        if sentence_mode:
            flattened: List[str] = []
            for item in output:
                if isinstance(item, list):
                    flattened.extend(str(x).strip() for x in item if str(x).strip())
                elif str(item).strip():
                    flattened.append(str(item).strip())
            return "\n".join(flattened)

        joined = " ".join(str(item).strip() for item in output if str(item).strip())
        return " ".join(joined.split())

    return str(output).strip()


def _looks_already_punctuated(text: str) -> bool:
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return False

    words = re.findall(r"\b\w+\b", cleaned, flags=re.UNICODE)
    if len(words) < 6:
        return False

    sentence_endings = re.findall(r"[.!?…]+(?:\s+|$)", cleaned)
    commas = re.findall(r",", cleaned)
    capitalized_sentence_starts = re.findall(r"(?:(?<=^)|(?<=[.!?…]\s))[A-Z]", cleaned)

    # Strong signal: prose-like sentence punctuation already exists.
    if len(sentence_endings) >= 2 and len(capitalized_sentence_starts) >= 1:
        return True

    # Medium signal: enough punctuation density for non-trivial text.
    if len(words) >= 12 and (len(sentence_endings) >= 2 or (len(sentence_endings) >= 1 and len(commas) >= 2)):
        return True

    # Another medium signal: multiple comma-separated clauses plus a sentence stop.
    if len(commas) >= 3 and len(sentence_endings) >= 1:
        return True

    return False


def _split_text(text: str, processing_scope: str) -> Tuple[List[str], str]:
    if processing_scope == "Per Paragraph":
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]
        return chunks or [text.strip()], "\n\n"
    return [text.strip()], ""


def restore_punctuation(
    text: str,
    model_label: str,
    processing_scope: str = "Whole Text",
    output_mode: str = "Restored Paragraphs",
    lowercase_input_first: bool = True,
) -> Tuple[str, str]:
    if not text or not text.strip():
        return "", "status=empty"

    model, model_info, model_dir = _load_model(model_label)
    chunks, joiner = _split_text(text, processing_scope)
    sentence_mode = output_mode == "One Sentence Per Line"
    processed_chunks: List[str] = []
    skipped_chunks = 0

    for chunk in chunks:
        prepared = " ".join(chunk.split())
        if _looks_already_punctuated(prepared):
            processed_chunks.append(chunk.strip())
            skipped_chunks += 1
            continue

        if lowercase_input_first:
            prepared = prepared.lower()

        inference = model.infer(texts=[prepared], apply_sbd=sentence_mode)
        first = inference[0] if isinstance(inference, list) and inference else inference
        processed_chunks.append(_normalize_infer_output(first, sentence_mode))

    restored_text = joiner.join(chunk for chunk in processed_chunks if chunk).strip()
    info = (
        f"model={model_info['name']} | "
        f"scope={processing_scope.lower().replace(' ', '_')} | "
        f"output={output_mode.lower().replace(' ', '_')} | "
        f"lowercase_input={str(lowercase_input_first).lower()} | "
        f"skipped_punctuated_chunks={skipped_chunks}/{len(chunks)} | "
        f"path={model_dir}"
    )
    return restored_text, info
