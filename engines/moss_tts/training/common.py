"""
Shared helpers for MOSS-TTS training.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import folder_paths

from engines.moss_tts.moss_tts_downloader import MossTTSDownloader
from engines.moss_tts.model_specs import MOSS_MODEL_SPECS
from utils.models.extra_paths import get_all_tts_model_paths


FRIENDLY_VARIANT_MAP = {
    "Small 1.7B (Local)": "MOSS-TTS-Local-Transformer",
    "8B (Delay)": "MOSS-TTS",
    "Recommended 8B v1.5 (Delay)": "MOSS-TTS-v1.5",
    "Legacy 8B v1.0 (Delay)": "MOSS-TTS",
    "Native 8B Dialogue (MOSS-TTSD-v1.0)": "MOSS-TTSD-v1.0",
}

SUPPORTED_DELAY_TRAINING_VARIANTS = {"MOSS-TTS", "MOSS-TTS-v1.5"}


def slugify(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value).strip())
    safe = safe.strip("_")
    return safe or "moss"


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_manifest_path(dataset_source: str) -> str:
    raw = os.path.expanduser(str(dataset_source or "").strip())
    if not raw:
        raise ValueError("dataset_source is required")

    candidates = [
        raw,
        os.path.join(folder_paths.get_input_directory(), raw),
        os.path.join(folder_paths.get_input_directory(), "datasets", raw),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(f"MOSS training manifest not found: {dataset_source}")


def fingerprint_paths(paths: Sequence[str]) -> str:
    digest = hashlib.md5()
    for path in paths:
        stat = os.stat(path)
        digest.update(f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()


def split_train_val(
    records: List[Dict[str, Any]],
    validation_split: float,
    split_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(records) < 2:
        raise ValueError("MOSS training requires at least 2 records when no validation manifest is provided.")

    indexed = list(records)
    rng = random.Random(int(split_seed))
    rng.shuffle(indexed)

    ratio = max(0.0, min(0.5, float(validation_split)))
    val_count = max(1, int(round(len(indexed) * ratio)))
    val_count = min(val_count, len(indexed) - 1)
    val_records = indexed[:val_count]
    train_records = indexed[val_count:]
    if not train_records or not val_records:
        raise ValueError("Train/validation split produced an empty subset.")
    return train_records, val_records


def resolve_variant_name(model_variant: str) -> str:
    value = str(model_variant or "").strip()
    if not value:
        return "MOSS-TTS"
    if value.startswith("local:"):
        value = value.split(":", 1)[1]
    value = FRIENDLY_VARIANT_MAP.get(value, value)
    return value


def resolve_delay_training_variant(config: Dict[str, Any]) -> str:
    variant = resolve_variant_name(config.get("model_variant", "MOSS-TTS"))
    if variant not in SUPPORTED_DELAY_TRAINING_VARIANTS:
        raise RuntimeError(
            "MOSS training supports the Delay 8B v1.0 and v1.5 models only. "
            f"Selected variant '{variant}' is not supported yet."
        )
    return variant


def resolve_model_path(model_variant: str) -> str:
    downloader = MossTTSDownloader()
    return downloader.resolve_model_path(model_variant)


def resolve_model_repo_id(model_variant: str) -> str:
    variant = resolve_variant_name(model_variant)
    spec = MOSS_MODEL_SPECS.get(variant)
    if not spec:
        raise ValueError(f"Unknown MOSS model variant: {variant}")
    return str(spec["repo_id"])


def resolve_codec_path(codec_model: str = "MOSS-Audio-Tokenizer") -> str:
    downloader = MossTTSDownloader()
    return downloader.resolve_model_path(codec_model or "MOSS-Audio-Tokenizer")


def get_moss_training_root() -> str:
    return os.path.join(folder_paths.get_output_directory(), "tts_audio_suite_training", "moss_tts")


def get_managed_lora_root() -> str:
    for base_path in get_all_tts_model_paths("TTS"):
        candidate = os.path.join(base_path, "moss_tts", "loras")
        os.makedirs(candidate, exist_ok=True)
        return candidate
    fallback = os.path.join(folder_paths.models_dir, "TTS", "moss_tts", "loras")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def next_available_adapter_dir(base_name: str, overwrite: bool = False) -> str:
    root = get_managed_lora_root()
    target = os.path.join(root, slugify(base_name))
    if overwrite or not os.path.exists(target):
        return target

    counter = 2
    while True:
        candidate = f"{target}_{counter}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def resolve_continue_from_adapter_path(continue_from: Any) -> str:
    if continue_from is None:
        return ""

    if isinstance(continue_from, str):
        value = continue_from.strip()
        if not value:
            return ""
        if os.path.isdir(value):
            return value
        raise FileNotFoundError(f"MOSS continue_from adapter path not found: {value}")

    if isinstance(continue_from, dict):
        data_type = str(continue_from.get("type", "") or "").strip().lower()
        if data_type == "training_artifacts":
            if str(continue_from.get("engine_type", "") or "").strip().lower() != "moss_tts":
                raise ValueError("continue_from TRAINING_ARTIFACTS must come from a MOSS training run")
            adapter_path = str(continue_from.get("model_path", "") or "").strip()
            if adapter_path and os.path.isdir(adapter_path):
                return adapter_path
            lora_info = continue_from.get("lora_adapter")
            if isinstance(lora_info, dict):
                adapter_path = str(lora_info.get("adapter_path", "") or "").strip()
                if adapter_path and os.path.isdir(adapter_path):
                    return adapter_path
            raise FileNotFoundError("continue_from TRAINING_ARTIFACTS does not contain a valid MOSS adapter path")

    raise ValueError(
        "Unsupported continue_from input for MOSS training. Use a direct adapter folder path or MOSS TRAINING_ARTIFACTS."
    )


def summarize_lora_mode(training_config: Dict[str, Any]) -> str:
    return (
        f"LoRA r={int(training_config.get('lora_r', 16))}, "
        f"alpha={int(training_config.get('lora_alpha', 32))}, "
        f"modules={training_config.get('trainable_lora_modules', 'mlp')}"
    )
