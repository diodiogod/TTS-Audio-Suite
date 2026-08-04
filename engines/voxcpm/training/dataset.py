"""Manifest validation and deterministic splitting for VoxCPM2 training."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from .common import get_training_root, slugify, write_jsonl


def _load_manifest(path: str) -> List[Dict[str, Any]]:
    manifest = os.path.abspath(os.path.expanduser(str(path or "").strip()))
    if not os.path.isfile(manifest):
        raise FileNotFoundError(f"VoxCPM training manifest not found: {path}")
    base_dir = os.path.dirname(manifest)
    records = []
    with open(manifest, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {manifest}: {exc}"
                ) from exc
            audio = str(raw.get("audio") or "").strip()
            text = str(raw.get("text") or "").strip()
            if not audio or not text:
                raise ValueError(
                    f"VoxCPM manifest line {line_number} requires non-empty audio and text"
                )
            audio_path = (
                audio if os.path.isabs(audio) else os.path.join(base_dir, audio)
            )
            audio_path = os.path.abspath(audio_path)
            if not os.path.isfile(audio_path):
                raise FileNotFoundError(
                    f"VoxCPM audio on manifest line {line_number} was not found: {audio_path}"
                )
            record = {"audio": audio_path, "text": text}
            for key in ("duration", "dataset_id", "ref_audio", "ref_duration", "is_prompt"):
                if key in raw:
                    record[key] = raw[key]
            if record.get("ref_audio"):
                ref_path = str(record["ref_audio"])
                if not os.path.isabs(ref_path):
                    ref_path = os.path.join(base_dir, ref_path)
                ref_path = os.path.abspath(ref_path)
                if not os.path.isfile(ref_path):
                    raise FileNotFoundError(
                        f"VoxCPM ref_audio on line {line_number} was not found: {ref_path}"
                    )
                record["ref_audio"] = ref_path
            records.append(record)
    if not records:
        raise ValueError(f"VoxCPM training manifest is empty: {manifest}")
    return records


def prepare_voxcpm_dataset(
    shared_settings: Dict[str, Any],
    *,
    dataset_source: str,
    model_name: str,
    validation_source: str = "",
    validation_split: float = 0.05,
    split_seed: int = 42,
    reuse_existing: bool = True,
    **_: Any,
) -> Dict[str, Any]:
    train_records = _load_manifest(dataset_source)
    val_records = _load_manifest(validation_source) if str(validation_source).strip() else []

    if not val_records and float(validation_split) > 0:
        if len(train_records) < 2:
            raise ValueError(
                "Automatic VoxCPM validation splitting needs at least two clips. "
                "Set validation_split to 0 for a one-clip smoke test."
            )
        shuffled = list(train_records)
        random.Random(int(split_seed)).shuffle(shuffled)
        val_count = min(
            len(shuffled) - 1,
            max(1, int(round(len(shuffled) * min(float(validation_split), 0.5)))),
        )
        val_records, train_records = shuffled[:val_count], shuffled[val_count:]

    fingerprint = hashlib.sha256()
    for record in train_records + val_records:
        stat = os.stat(record["audio"])
        fingerprint.update(
            f"{record['audio']}|{stat.st_size}|{stat.st_mtime_ns}|{record['text']}".encode()
        )
    dataset_dir = os.path.join(
        get_training_root(),
        "datasets",
        f"{slugify(model_name)}_{fingerprint.hexdigest()[:10]}",
    )
    train_path = os.path.join(dataset_dir, "train.jsonl")
    val_path = os.path.join(dataset_dir, "validation.jsonl")
    if not (
        reuse_existing
        and os.path.isfile(train_path)
        and (not val_records or os.path.isfile(val_path))
    ):
        write_jsonl(train_records, train_path)
        if val_records:
            write_jsonl(val_records, val_path)

    return {
        "type": "training_dataset",
        "engine_type": "voxcpm",
        "training_mode": "lora_adapter",
        "model_variant": shared_settings["model_variant"],
        "model_name": str(model_name or "VoxCPM2LoRA"),
        "dataset_dir": dataset_dir,
        "train_manifest": train_path,
        "val_manifest": val_path if val_records else "",
        "train_records": len(train_records),
        "val_records": len(val_records),
        "sample_rate": 16000,
        "output_sample_rate": shared_settings["sample_rate"],
        "source_summary": Path(dataset_source).name
        + (f" + {Path(validation_source).name}" if validation_source else ""),
    }
