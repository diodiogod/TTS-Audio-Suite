"""Official-compatible VoxCPM LoRA checkpoint persistence.

Adapted from ``OpenBMB/VoxCPM/scripts/train_voxcpm_finetune.py`` (Apache-2.0).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict


def load_checkpoint(model, optimizer, scheduler, save_dir: Path) -> int:
    latest = save_dir / "latest"
    if not latest.is_dir():
        return 0
    import torch
    from safetensors.torch import load_file

    weights = latest / "lora_weights.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Resumable VoxCPM LoRA weights are missing: {weights}")
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.load_state_dict(load_file(str(weights)), strict=False)

    optimizer_path = latest / "optimizer.pth"
    scheduler_path = latest / "scheduler.pth"
    state_path = latest / "training_state.json"
    if optimizer_path.is_file():
        optimizer.load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=True)
        )
    if scheduler_path.is_file():
        scheduler.load_state_dict(
            torch.load(scheduler_path, map_location="cpu", weights_only=True)
        )
    if not state_path.is_file():
        raise FileNotFoundError(f"VoxCPM training state is missing: {state_path}")
    with open(state_path, "r", encoding="utf-8") as handle:
        return int(json.load(handle).get("step", 0))


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    save_dir: Path,
    step: int,
    base_model_path: str,
    metadata: Dict[str, Any],
) -> Path:
    """Save the official VoxCPM LoRA checkpoint contract plus resume state."""
    import torch
    from safetensors.torch import save_file

    # TTS Audio Suite adaptation of OpenBMB's official save_checkpoint:
    # retain tensor-only LoRA export while adding suite compatibility metadata.
    unwrapped = model.module if hasattr(model, "module") else model
    lora_config = unwrapped.lora_config
    folder = save_dir / f"step_{int(step):07d}"
    folder.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in unwrapped.state_dict().items()
        if "lora_" in key
    }
    save_file(state, str(folder / "lora_weights.safetensors"))
    config_payload = {
        "base_model": os.path.abspath(base_model_path),
        "lora_config": (
            lora_config.model_dump()
            if hasattr(lora_config, "model_dump")
            else vars(lora_config)
        ),
        "tts_audio_suite": metadata,
    }
    with open(folder / "lora_config.json", "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2, ensure_ascii=False)
    torch.save(optimizer.state_dict(), folder / "optimizer.pth")
    torch.save(scheduler.state_dict(), folder / "scheduler.pth")
    with open(folder / "training_state.json", "w", encoding="utf-8") as handle:
        json.dump({"step": int(step)}, handle)

    latest = save_dir / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(folder, latest)
    return folder
