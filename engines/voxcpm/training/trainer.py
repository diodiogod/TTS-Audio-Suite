"""In-process VoxCPM2 LoRA trainer based on OpenBMB's official launcher.

Adapted from ``OpenBMB/VoxCPM/scripts/train_voxcpm_finetune.py`` (Apache-2.0).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from engines.training.progress_io import write_json_progress_file
from engines.training.progress_registry import (
    finalize_training_job,
    register_training_job,
    update_training_job,
)

from .common import (
    next_adapter_dir,
    read_lora_info,
    resolve_continue_from,
    slugify,
)
from .checkpoint import (
    load_checkpoint as _load_checkpoint,
    save_checkpoint as _save_checkpoint,
)


def _interrupted() -> bool:
    try:
        import comfy.model_management as model_management

        return bool(model_management.processing_interrupted())
    except Exception:
        return False


def _write_progress(path: str, *, status: str, phase: str, **updates: Any) -> None:
    payload: Dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload.update(json.load(handle))
        except Exception:
            payload = {}
    payload.update(updates)
    payload.update(
        {
            "status": status,
            "phase": phase,
            "updated_at": datetime.now().isoformat(),
        }
    )
    write_json_progress_file(path, payload, default=str)


def _build_job_dirs(
    dataset_info: Dict[str, Any],
    output_name: str,
    *,
    resume: bool,
    overwrite: bool,
) -> Tuple[str, str, str]:
    if resume and overwrite:
        raise ValueError("VoxCPM resume and overwrite cannot be enabled together")
    name = slugify(output_name or dataset_info.get("model_name") or "voxcpm2_lora")
    root = os.path.join(
        os.path.dirname(os.path.dirname(str(dataset_info["dataset_dir"]))),
        "jobs",
    )
    fingerprint = hashlib.sha256(
        f"{name}|{os.path.abspath(dataset_info['dataset_dir'])}".encode()
    ).hexdigest()[:10]
    canonical = os.path.join(root, f"{name}_{fingerprint}")
    if resume:
        if not os.path.isdir(os.path.join(canonical, "checkpoints", "latest")):
            raise FileNotFoundError(
                f"No resumable VoxCPM checkpoint exists for '{name}': {canonical}"
            )
        job_dir = canonical
    elif overwrite:
        job_dir = canonical
        if os.path.isdir(job_dir):
            shutil.rmtree(job_dir)
    elif os.path.exists(canonical):
        job_dir = f"{canonical}_{int(time.time())}"
    else:
        job_dir = canonical
    os.makedirs(job_dir, exist_ok=True)
    return job_dir, next_adapter_dir(name, overwrite), name


def _validate(model, loader, batch_processor, accelerator, lambdas) -> Dict[str, float]:
    if loader is None:
        return {}
    import torch

    model.eval()
    values: Dict[str, list] = {}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= 10:
                break
            processed = batch_processor(batch)
            with accelerator.autocast(dtype=torch.bfloat16):
                outputs = model(
                    processed["text_tokens"],
                    processed["text_mask"],
                    processed["audio_feats"],
                    processed["audio_mask"],
                    processed["loss_mask"],
                    processed["position_ids"],
                    processed["labels"],
                    progress=0.0,
                    sample_generate=False,
                )
            for key, value in outputs.items():
                if key.startswith("loss/"):
                    values.setdefault(key, []).append(
                        float((value * float(lambdas.get(key, 1.0))).detach().item())
                    )
    model.train()
    return {
        f"val/{key}": sum(items) / len(items)
        for key, items in values.items()
        if items
    }


def run_voxcpm_training_job(
    shared_settings: Dict[str, Any],
    dataset_info: Dict[str, Any],
    training_config: Dict[str, Any],
    output_name: str = "",
    resume: bool = False,
    overwrite: bool = False,
    continue_from: Any = None,
    node_id: str = "",
) -> Dict[str, Any]:
    if dataset_info.get("model_variant") != shared_settings["model_variant"]:
        raise RuntimeError(
            "VoxCPM dataset/base mismatch. Prepare the dataset again with the selected engine."
        )
    if str(training_config.get("training_mode", "")).lower() != "lora_adapter":
        raise RuntimeError("VoxCPM training currently supports LoRA adapters only")
    if resume and continue_from not in (None, ""):
        raise ValueError("Use either exact resume or continue_from, not both")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "VoxCPM2 LoRA training requires CUDA. CPU/MPS training is not supported "
            "by the official BF16 training path."
        )
    try:
        from torch.optim import AdamW
        from transformers import get_cosine_schedule_with_warmup
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "VoxCPM training requires torch, transformers, datasets, and safetensors"
        ) from exc

    from engines.voxcpm.voxcpm_engine import VoxCPMEngine
    from utils.models.unified_model_interface import unified_model_interface

    unified_model_interface.clear_engine_models("voxcpm")
    import gc

    gc.collect()
    for device_index in range(torch.cuda.device_count()):
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()

    requested_device = int(training_config.get("cuda_device_index", -1))
    device_count = torch.cuda.device_count()
    if requested_device >= device_count:
        raise ValueError(
            f"VoxCPM cuda_device_index={requested_device} is invalid; "
            f"this process sees {device_count} CUDA device(s)"
        )
    if requested_device < 0:
        free_memory = []
        for index in range(device_count):
            with torch.cuda.device(index):
                free_bytes, _ = torch.cuda.mem_get_info()
            free_memory.append(int(free_bytes))
        requested_device = max(range(device_count), key=free_memory.__getitem__)
    torch.cuda.set_device(requested_device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("VoxCPM2 LoRA training requires a CUDA GPU with BF16 support")

    # Import through the engine's collision-safe official-package loader.
    VoxCPMEngine(
        model_name=shared_settings["model_variant"],
        model_dir=shared_settings["model_path"],
    )._import_official_package()
    from voxcpm.model import VoxCPM2Model
    from voxcpm.model.voxcpm2 import LoRAConfig
    from voxcpm.training import (
        Accelerator,
        BatchProcessor,
        build_dataloader,
        load_audio_text_datasets,
    )

    job_dir, final_adapter_dir, resolved_name = _build_job_dirs(
        dataset_info,
        output_name,
        resume=resume,
        overwrite=overwrite,
    )
    checkpoint_dir = Path(job_dir) / "checkpoints"
    progress_file = os.path.join(job_dir, "progress.json")
    continue_path = resolve_continue_from(continue_from)
    total_steps = int(training_config.get("max_train_steps", 1000))
    if total_steps < 1:
        raise ValueError("VoxCPM max_train_steps must be at least 1")

    lora_values = {
        "enable_lm": bool(training_config.get("lora_enable_lm", True)),
        "enable_dit": bool(training_config.get("lora_enable_dit", True)),
        "enable_proj": bool(training_config.get("lora_enable_proj", False)),
        "r": int(training_config.get("lora_r", 32)),
        "alpha": int(training_config.get("lora_alpha", 32)),
        "dropout": float(training_config.get("lora_dropout", 0.0)),
    }
    resolved_config = {
        "dataset": dataset_info,
        "training_config": training_config,
        "shared_settings": shared_settings,
        "lora_config": lora_values,
        "continue_from": continue_path,
    }
    resolved_config_path = os.path.join(job_dir, "resolved_training_config.json")
    if resume and os.path.isfile(resolved_config_path):
        with open(resolved_config_path, "r", encoding="utf-8") as handle:
            previous_run = json.load(handle)
        previous_lora = previous_run.get("lora_config") or {}
        if any(previous_lora.get(key) != value for key, value in lora_values.items()):
            raise ValueError(
                "Exact VoxCPM resume requires the same LoRA module groups, rank, "
                "alpha, and dropout as the saved job"
            )
        previous_model_value = str(
            (previous_run.get("shared_settings") or {}).get("model_path") or ""
        ).strip()
        previous_model = (
            os.path.abspath(previous_model_value) if previous_model_value else ""
        )
        if previous_model and previous_model != os.path.abspath(shared_settings["model_path"]):
            raise ValueError("Exact VoxCPM resume requires the same base model path")
    with open(resolved_config_path, "w", encoding="utf-8") as handle:
        json.dump(resolved_config, handle, indent=2, default=str)

    register_training_job(
        node_id,
        engine_type="voxcpm",
        progress_file=progress_file,
        job_dir=job_dir,
        model_name=resolved_name,
        sample_rate="16k input / 48k output",
        total_epochs=0,
    )
    _write_progress(
        progress_file,
        status="starting",
        phase="loading",
        node_id=str(node_id or ""),
        step=0,
        total_steps=total_steps,
        summary=f"VoxCPM2 LoRA r={lora_values['r']}, alpha={lora_values['alpha']}",
    )

    try:
        accelerator = Accelerator(amp=True)
        # The official single-process accelerator defaults to LOCAL_RANK=0.
        # TTS Audio Suite patch: respect the training node's selected CUDA GPU.
        accelerator.local_rank = requested_device
        accelerator.device_ctx = torch.cuda.device(requested_device)
        base_model = VoxCPM2Model.from_local(
            shared_settings["model_path"],
            optimize=False,
            training=True,
            lora_config=LoRAConfig(**lora_values),
        )
        if int(base_model.audio_vae.sample_rate) != 16000:
            raise RuntimeError(
                f"Unexpected VoxCPM2 AudioVAE input rate: {base_model.audio_vae.sample_rate}"
            )
        tokenizer = base_model.text_tokenizer
        train_ds, val_ds = load_audio_text_datasets(
            train_manifest=dataset_info["train_manifest"],
            val_manifest=dataset_info.get("val_manifest", ""),
            sample_rate=16000,
        )

        def tokenize(batch):
            return {"text_ids": [tokenizer(text) for text in batch["text"]]}

        train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
        if val_ds is not None:
            val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

        max_batch_tokens = int(training_config.get("max_batch_tokens", 8192))
        batch_size = int(training_config.get("batch_size", 1))
        if max_batch_tokens > 0:
            from voxcpm.training.data import compute_sample_lengths

            lengths = compute_sample_lengths(
                train_ds,
                audio_vae_fps=base_model.audio_vae.sample_rate
                / base_model.audio_vae.hop_length,
                patch_size=base_model.config.patch_size,
            )
            max_sample = max_batch_tokens // max(batch_size, 1)
            train_ds = train_ds.select(
                [index for index, length in enumerate(lengths) if length <= max_sample]
            )
        if len(train_ds) < batch_size:
            raise RuntimeError(
                "No complete VoxCPM training batch remains. Reduce batch_size, "
                "increase max_batch_tokens, or shorten the clips."
            )

        workers = int(training_config.get("num_workers", 0))
        train_loader = build_dataloader(
            train_ds,
            accelerator=accelerator,
            batch_size=batch_size,
            num_workers=workers,
            drop_last=True,
        )
        val_loader = (
            build_dataloader(
                val_ds,
                accelerator=accelerator,
                batch_size=batch_size,
                num_workers=workers,
                drop_last=False,
            )
            if val_ds is not None
            else None
        )
        dataset_count = (
            int(max(train_ds["dataset_id"])) + 1
            if "dataset_id" in train_ds.column_names
            else 1
        )
        batch_processor = BatchProcessor(
            config=base_model.config,
            audio_vae=base_model.audio_vae,
            dataset_cnt=dataset_count,
            device=accelerator.device,
        )
        del base_model.audio_vae

        model = accelerator.prepare_model(base_model)
        unwrapped = accelerator.unwrap(model)
        if continue_path:
            previous = read_lora_info(continue_path)
            previous_config = previous["lora_config"]
            if any(
                previous_config.get(key) != value
                for key, value in lora_values.items()
            ):
                raise ValueError(
                    "continue_from LoRA settings differ from the new training configuration"
                )
            unwrapped.load_state_dict(
                load_file(os.path.join(continue_path, "lora_weights.safetensors")),
                strict=False,
            )
        model.train()
        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(training_config.get("learning_rate", 1e-4)),
            weight_decay=float(training_config.get("weight_decay", 0.01)),
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(training_config.get("warmup_steps", 100)),
            num_training_steps=total_steps,
        )
        start_step = _load_checkpoint(model, optimizer, scheduler, checkpoint_dir) if resume else 0
        if start_step >= total_steps:
            raise ValueError(
                f"VoxCPM resume checkpoint is already at step {start_step}; "
                f"set max_train_steps above {start_step}"
            )
        grad_accum = max(
            1, int(training_config.get("gradient_accumulation_steps", 8))
        )
        log_steps = max(1, int(training_config.get("log_steps", 10)))
        save_steps = int(training_config.get("save_steps", 500))
        eval_steps = int(training_config.get("eval_steps", 500))
        max_grad_norm = float(training_config.get("max_grad_norm", 1.0))
        lambdas = {"loss/diff": 1.0, "loss/stop": 1.0}
        train_iterator = iter(train_loader)
        recent_loss = []

        def next_batch():
            nonlocal train_iterator
            try:
                return next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                return next(train_iterator)

        for step in range(start_step, total_steps):
            if _interrupted():
                _save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    checkpoint_dir,
                    step,
                    shared_settings["model_path"],
                    shared_settings,
                )
                raise InterruptedError("VoxCPM training interrupted by user")
            optimizer.zero_grad(set_to_none=True)
            losses = {}
            for micro_step in range(grad_accum):
                processed = batch_processor(next_batch())
                sync = (
                    contextlib.nullcontext()
                    if micro_step == grad_accum - 1
                    else accelerator.no_sync()
                )
                with sync, accelerator.autocast(dtype=torch.bfloat16):
                    outputs = model(
                        processed["text_tokens"],
                        processed["text_mask"],
                        processed["audio_feats"],
                        processed["audio_mask"],
                        processed["loss_mask"],
                        processed["position_ids"],
                        processed["labels"],
                        progress=step / max(total_steps, 1),
                    )
                    total_loss = sum(
                        value * lambdas.get(key, 1.0) / grad_accum
                        for key, value in outputs.items()
                        if key.startswith("loss/")
                    )
                accelerator.backward(total_loss)
                losses = {
                    key: float(value.detach().item())
                    for key, value in outputs.items()
                    if key.startswith("loss/")
                }

            accelerator.scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                unwrapped.parameters(), max_grad_norm if max_grad_norm > 0 else 1e9
            )
            accelerator.step(optimizer)
            accelerator.update()
            scheduler.step()
            completed = step + 1

            if completed % log_steps == 0 or completed == total_steps:
                metrics = {
                    **losses,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "grad_norm": float(grad_norm),
                }
                recent_loss = (recent_loss + [{"step": completed, **metrics}])[-100:]
                progress = completed / total_steps
                update_training_job(
                    node_id,
                    status="running",
                    phase="training",
                    current_step=completed,
                    total_steps=total_steps,
                )
                _write_progress(
                    progress_file,
                    status="running",
                    phase="training",
                    step=completed,
                    current_step=completed,
                    total_steps=total_steps,
                    completed_total_steps=completed,
                    overall_progress=progress,
                    current_metrics=metrics,
                    recent_loss_trace=recent_loss,
                )

            if val_loader is not None and eval_steps > 0 and (
                completed % eval_steps == 0 or completed == total_steps
            ):
                validation = _validate(
                    model, val_loader, batch_processor, accelerator, lambdas
                )
                _write_progress(
                    progress_file,
                    status="running",
                    phase="validation",
                    step=completed,
                    total_steps=total_steps,
                    current_metrics={**losses, **validation},
                    overall_progress=completed / total_steps,
                )

            if save_steps > 0 and (
                completed % save_steps == 0 or completed == total_steps
            ):
                _save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    checkpoint_dir,
                    completed,
                    shared_settings["model_path"],
                    shared_settings,
                )

        latest = checkpoint_dir / "latest"
        if not latest.is_dir():
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                checkpoint_dir,
                total_steps,
                shared_settings["model_path"],
                shared_settings,
            )
        if overwrite and os.path.isdir(final_adapter_dir):
            shutil.rmtree(final_adapter_dir)
        os.makedirs(final_adapter_dir, exist_ok=True)
        for filename in ("lora_weights.safetensors", "lora_config.json"):
            shutil.copy2(latest / filename, os.path.join(final_adapter_dir, filename))

        summary = (
            f"VoxCPM2 LoRA complete: {resolved_name} | "
            f"r={lora_values['r']}, alpha={lora_values['alpha']} | "
            f"steps={total_steps} | cuda:{requested_device}"
        )
        artifacts = {
            "type": "training_artifacts",
            "engine_type": "voxcpm",
            "training_mode": "lora_adapter",
            "model_variant": shared_settings["model_variant"],
            "model_path": final_adapter_dir,
            "job_dir": job_dir,
            "summary": summary,
            "lora_adapter": {
                "type": "voxcpm_lora",
                "adapter_path": final_adapter_dir,
                "base_model_name_or_path": shared_settings["model_path"],
            },
        }
        finalize_training_job(
            node_id,
            status="completed",
            current_step=total_steps,
            total_steps=total_steps,
            artifacts=artifacts,
        )
        _write_progress(
            progress_file,
            status="completed",
            phase="done",
            step=total_steps,
            total_steps=total_steps,
            completed_total_steps=total_steps,
            overall_progress=1.0,
            output_adapter=final_adapter_dir,
            recent_loss_trace=recent_loss,
            summary=summary,
        )
        return artifacts
    except InterruptedError as exc:
        finalize_training_job(node_id, status="cancelled", error=str(exc))
        _write_progress(
            progress_file,
            status="cancelled",
            phase="cancelled",
            error=str(exc),
            summary="VoxCPM training cancelled; latest checkpoint was preserved",
        )
        raise
    except Exception as exc:
        finalize_training_job(node_id, status="error", error=str(exc))
        _write_progress(
            progress_file,
            status="error",
            phase="error",
            error=str(exc),
        )
        raise
