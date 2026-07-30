"""Shared paths and validation for VoxCPM2 LoRA training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import folder_paths

from engines.voxcpm.voxcpm_downloader import VoxCPMDownloader
from utils.models.extra_paths import get_all_tts_model_paths


def slugify(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in str(value or "").strip()
    ).strip("_")
    return safe or "voxcpm2_lora"


def get_training_root() -> str:
    return os.path.join(
        folder_paths.get_output_directory(),
        "tts_audio_suite_training",
        "voxcpm",
    )


def get_managed_lora_root() -> str:
    for base_path in get_all_tts_model_paths("TTS"):
        target = os.path.join(base_path, "voxcpm", "loras")
        os.makedirs(target, exist_ok=True)
        return target
    target = os.path.join(folder_paths.models_dir, "TTS", "voxcpm", "loras")
    os.makedirs(target, exist_ok=True)
    return target


def discover_lora_adapters() -> List[str]:
    root = get_managed_lora_root()
    results = []
    for entry in sorted(os.scandir(root), key=lambda item: item.name.lower()):
        if entry.is_dir() and is_lora_adapter(entry.path):
            results.append(entry.path)
    return results


def is_lora_adapter(path: str) -> bool:
    folder = os.path.abspath(os.path.expanduser(str(path or "").strip()))
    return (
        os.path.isdir(folder)
        and os.path.isfile(os.path.join(folder, "lora_config.json"))
        and os.path.isfile(os.path.join(folder, "lora_weights.safetensors"))
    )


def read_lora_info(path: str) -> Dict[str, Any]:
    folder = os.path.abspath(os.path.expanduser(str(path or "").strip()))
    if not is_lora_adapter(folder):
        raise FileNotFoundError(
            "VoxCPM LoRA folder must contain lora_config.json and "
            f"lora_weights.safetensors: {folder}"
        )
    with open(os.path.join(folder, "lora_config.json"), "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("lora_config"), dict):
        raise ValueError(f"Invalid VoxCPM LoRA configuration: {folder}")
    return data


def resolve_voxcpm2_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    selection = str(
        config.get("model_variant") or config.get("model_name") or "VoxCPM2"
    )
    downloader = VoxCPMDownloader()
    model_path = downloader.resolve_model_path(selection)
    spec = downloader.get_model_spec(selection, resolved_path=model_path)
    if str(spec.get("architecture", "")).lower() != "voxcpm2":
        raise RuntimeError(
            "VoxCPM LoRA training currently supports VoxCPM2 only. "
            f"Selected checkpoint is {spec.get('canonical', selection)}."
        )
    return {
        "model_variant": str(spec.get("canonical") or selection),
        "model_path": os.path.abspath(model_path),
        "repo_id": str(spec.get("repo_id") or "openbmb/VoxCPM2"),
        "architecture": "voxcpm2",
        "sample_rate": int(spec.get("sample_rate") or 48000),
        "device": str(config.get("device") or "auto"),
    }


def next_adapter_dir(name: str, overwrite: bool) -> str:
    target = os.path.join(get_managed_lora_root(), slugify(name))
    if overwrite or not os.path.exists(target):
        return target
    suffix = 2
    while os.path.exists(f"{target}_{suffix}"):
        suffix += 1
    return f"{target}_{suffix}"


def resolve_continue_from(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        path = value
    elif isinstance(value, dict):
        if str(value.get("engine_type", "")).lower() != "voxcpm":
            raise ValueError("continue_from artifacts must come from VoxCPM training")
        info = value.get("lora_adapter")
        path = (
            str(info.get("adapter_path", ""))
            if isinstance(info, dict)
            else str(value.get("model_path", ""))
        )
    else:
        raise ValueError(
            "VoxCPM continue_from accepts a LoRA folder or VoxCPM TRAINING_ARTIFACTS"
        )
    path = os.path.abspath(os.path.expanduser(path))
    read_lora_info(path)
    return path


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
