"""
LoRA training runner for MOSS-TTS Delay 8B.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import folder_paths

from engines.training.progress_io import write_json_progress_file
from engines.training.progress_registry import (
    finalize_training_job,
    register_training_job,
    update_training_job,
)
from engines.moss_tts.training.common import (
    dump_jsonl,
    get_moss_training_root,
    load_jsonl,
    next_available_adapter_dir,
    resolve_codec_path,
    resolve_continue_from_adapter_path,
    resolve_delay_training_variant,
    resolve_model_path,
    slugify,
    summarize_lora_mode,
)


def _write_progress(progress_file: str, *, status: str, phase: str, **updates: Any) -> None:
    payload: Dict[str, Any] = {}
    if progress_file and os.path.isfile(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
                if isinstance(existing, dict):
                    payload.update(existing)
        except Exception:
            payload = {}
    payload.update(updates)
    payload["status"] = status
    payload["phase"] = phase
    payload["updated_at"] = datetime.now().isoformat()
    if progress_file:
        write_json_progress_file(progress_file, payload, default=str)


def _build_output_dirs(dataset_info: Dict[str, Any], output_name: str, overwrite: bool) -> Tuple[str, str, str]:
    safe_name = slugify(output_name or dataset_info.get("model_name") or "moss_lora")
    root = os.path.join(get_moss_training_root(), "jobs")
    os.makedirs(root, exist_ok=True)
    dataset_dir = str(dataset_info.get("dataset_dir", "") or "")
    stat = os.stat(dataset_dir) if dataset_dir and os.path.exists(dataset_dir) else None
    fingerprint = f"{safe_name}|{dataset_dir}|{getattr(stat, 'st_mtime_ns', 0)}"
    job_hash = __import__("hashlib").md5(fingerprint.encode("utf-8")).hexdigest()[:10]
    job_dir = os.path.join(root, f"{safe_name}_{job_hash}")
    if os.path.isdir(job_dir) and not overwrite:
        job_dir = os.path.join(root, f"{safe_name}_{job_hash}_{int(time.time())}")
    if overwrite and os.path.isdir(job_dir):
        shutil.rmtree(job_dir)
    os.makedirs(job_dir, exist_ok=True)
    final_adapter_dir = next_available_adapter_dir(safe_name, overwrite=overwrite)
    return job_dir, final_adapter_dir, safe_name


def run_moss_training_job(
    shared_settings: Dict[str, Any],
    dataset_info: Dict[str, Any],
    training_config: Dict[str, Any],
    output_name: str = "",
    resume: bool = False,
    overwrite: bool = False,
    continue_from: Any = None,
    node_id: str = "",
) -> Dict[str, Any]:
    if resume:
        raise RuntimeError("MOSS training does not support exact resume yet. Use continue_from with an existing adapter if you want warm-start LoRA training.")

    resolve_delay_training_variant(shared_settings)
    if str(training_config.get("training_mode", "") or "").strip().lower() != "lora_adapter":
        raise RuntimeError("MOSS training currently supports Delay 8B LoRA mode only.")

    try:
        import torch
        from accelerate import Accelerator
        from peft import LoraConfig, PeftModel, get_peft_model
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise RuntimeError(
            "MOSS LoRA training requires accelerate, peft, and torch runtime dependencies. "
            "Run install.py again after pulling the latest version."
        ) from e

    from engines.moss_tts.impl.delay.modeling_moss_tts import MossTTSDelayModel
    from engines.moss_tts.training.dataset import MossTTSSFTDataset, build_delay_training_processor

    model_path = resolve_model_path("MOSS-TTS")
    codec_path = resolve_codec_path(shared_settings.get("codec_model", "MOSS-Audio-Tokenizer"))
    job_dir, final_adapter_dir, resolved_name = _build_output_dirs(dataset_info, output_name, overwrite)
    progress_file = os.path.join(job_dir, "progress.json")
    continue_from_path = resolve_continue_from_adapter_path(continue_from)

    resolved_training_config_path = os.path.join(job_dir, "resolved_training_config.json")
    with open(resolved_training_config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset_info,
                "training_config": training_config,
                "shared_settings": shared_settings,
                "continue_from": continue_from_path,
                "model_path": model_path,
                "codec_path": codec_path,
                "final_adapter_dir": final_adapter_dir,
            },
            handle,
            indent=2,
            sort_keys=True,
            default=str,
        )

    register_training_job(
        node_id,
        engine_type="moss_tts",
        progress_file=progress_file,
        job_dir=job_dir,
        model_name=resolved_name,
        sample_rate="24k",
        total_epochs=int(training_config.get("epochs", 1)),
    )

    try:
        _write_progress(
            progress_file,
            status="starting",
            phase="setup",
            summary=f"MOSS Delay LoRA training: {summarize_lora_mode(training_config)}",
        )

        train_records = load_jsonl(dataset_info["prepared_train_jsonl"])
        val_records = load_jsonl(dataset_info["prepared_val_jsonl"])
        if not train_records:
            raise RuntimeError("Prepared MOSS training set is empty")
        if not val_records:
            raise RuntimeError("Prepared MOSS validation set is empty")

        processor = build_delay_training_processor(model_path)
        dtype_name = str(training_config.get("mixed_precision", "bf16"))
        if torch.cuda.is_available():
            if dtype_name == "fp16":
                model_dtype = torch.float16
            elif dtype_name == "no":
                model_dtype = torch.float32
            else:
                model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float32
            dtype_name = "no"

        attn_implementation = "sdpa" if torch.cuda.is_available() else "eager"

        train_dataset = MossTTSSFTDataset(train_records, processor)
        val_dataset = MossTTSSFTDataset(val_records, processor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=max(1, int(training_config.get("batch_size", 1))),
            shuffle=True,
            num_workers=max(0, int(training_config.get("num_workers", 0))),
            pin_memory=torch.cuda.is_available(),
            collate_fn=train_dataset.collate_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, int(training_config.get("batch_size", 1))),
            shuffle=False,
            num_workers=max(0, int(training_config.get("num_workers", 0))),
            pin_memory=torch.cuda.is_available(),
            collate_fn=val_dataset.collate_fn,
            drop_last=False,
        )

        model = MossTTSDelayModel.from_pretrained(
            model_path,
            attn_implementation=attn_implementation,
            dtype=model_dtype,
        )

        target_map = {
            "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"],
            "mlp_plus_o": ["gate_proj", "up_proj", "down_proj", "o_proj"],
        }
        module_mode = str(training_config.get("trainable_lora_modules", "mlp"))
        if module_mode not in target_map:
            raise RuntimeError(f"Unsupported MOSS trainable_lora_modules mode: {module_mode}")

        original_get_input_embeddings = type(model).get_input_embeddings
        type(model).get_input_embeddings = lambda self, input_ids=None: (
            original_get_input_embeddings(self, input_ids)
            if input_ids is not None
            else self.language_model.get_input_embeddings()
        )

        lora_config = LoraConfig(
            r=int(training_config.get("lora_r", 16)),
            lora_alpha=int(training_config.get("lora_alpha", 32)),
            target_modules=target_map[module_mode],
            lora_dropout=float(training_config.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
        )

        if continue_from_path:
            model = PeftModel.from_pretrained(model, continue_from_path, is_trainable=True)
        else:
            model = get_peft_model(model, lora_config)

        _original_forward = type(model.get_base_model() if hasattr(model, "get_base_model") else model).forward

        def _patched_forward(self, *args, output_hidden_states=None, return_dict=None, **kwargs):
            return _original_forward(self, *args, **kwargs)

        base_cls = type(model.get_base_model() if hasattr(model, "get_base_model") else model)
        base_cls.forward = _patched_forward

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        if hasattr(base_model, "gradient_checkpointing_enable") and bool(training_config.get("gradient_checkpointing", True)):
            base_model.gradient_checkpointing_enable()
        for cfg_obj in (getattr(model, "config", None), getattr(base_model, "config", None)):
            if cfg_obj is not None and hasattr(cfg_obj, "use_cache"):
                cfg_obj.use_cache = False

        grad_accum = max(1, int(training_config.get("gradient_accumulation_steps", 1)))
        num_epochs = max(1, int(training_config.get("epochs", 3)))
        max_train_steps = int(training_config.get("max_train_steps", 0) or 0)
        num_update_steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
        total_steps = max_train_steps if max_train_steps > 0 else num_epochs * num_update_steps_per_epoch
        warmup_steps = max(0, int(training_config.get("warmup_steps", 100)))
        save_steps = max(0, int(training_config.get("save_steps", 500)))
        eval_steps = max(0, int(training_config.get("eval_steps", 500)))
        log_steps = max(1, int(training_config.get("log_steps", 10)))

        optimizer = AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(training_config.get("learning_rate", 2e-6)),
            weight_decay=float(training_config.get("weight_decay", 0.01)),
        )

        def _lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        accelerator = Accelerator(mixed_precision=dtype_name)
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
        )

        def _evaluate() -> float:
            model.eval()
            total_loss = 0.0
            total_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    gathered = accelerator.gather_for_metrics(outputs.loss.detach().float().unsqueeze(0))
                    total_loss += gathered.mean().item()
                    total_batches += 1
            model.train()
            return total_loss / max(1, total_batches)

        global_step = 0
        best_val_loss = None
        start_time = time.time()
        _write_progress(
            progress_file,
            status="running",
            phase="train",
            step=0,
            total_steps=total_steps,
            train_records=len(train_records),
            val_records=len(val_records),
        )

        for epoch in range(num_epochs):
            for step, batch in enumerate(train_loader, start=1):
                with accelerator.accumulate(model):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            [param for param in model.parameters() if param.requires_grad],
                            float(training_config.get("max_grad_norm", 0.5)),
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    loss_value = accelerator.gather_for_metrics(loss.detach().float().unsqueeze(0)).mean().item()
                    if global_step % log_steps == 0 or global_step == 1:
                        elapsed = max(1e-6, time.time() - start_time)
                        steps_per_sec = global_step / elapsed
                        print(
                            f"MOSS LoRA training | step {global_step}/{total_steps} | "
                            f"loss {loss_value:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                            f"{steps_per_sec:.2f} steps/s"
                        )
                        update_training_job(
                            node_id,
                            current_epoch=epoch + 1,
                            current_step=global_step,
                            total_steps=total_steps,
                            latest_loss=loss_value,
                        )
                        _write_progress(
                            progress_file,
                            status="running",
                            phase="train",
                            step=global_step,
                            total_steps=total_steps,
                            epoch=epoch + 1,
                            latest_loss=loss_value,
                            learning_rate=scheduler.get_last_lr()[0],
                        )

                    if eval_steps > 0 and global_step % eval_steps == 0:
                        val_loss = _evaluate()
                        print(f"MOSS LoRA validation | step {global_step} | val_loss {val_loss:.4f}")
                        if best_val_loss is None or val_loss < best_val_loss:
                            best_val_loss = val_loss
                        _write_progress(
                            progress_file,
                            status="running",
                            phase="validate",
                            step=global_step,
                            total_steps=total_steps,
                            validation_loss=val_loss,
                            best_validation_loss=best_val_loss,
                        )

                    if save_steps > 0 and global_step % save_steps == 0:
                        checkpoint_dir = os.path.join(job_dir, f"checkpoint-step-{global_step}")
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            accelerator.unwrap_model(model).save_pretrained(checkpoint_dir)
                            print(f"MOSS LoRA checkpoint saved: {checkpoint_dir}")

                    if global_step >= total_steps:
                        break

            if global_step >= total_steps:
                break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if overwrite and os.path.isdir(final_adapter_dir):
                shutil.rmtree(final_adapter_dir)
            accelerator.unwrap_model(model).save_pretrained(final_adapter_dir)
        accelerator.wait_for_everyone()

        summary = (
            f"MOSS Delay LoRA training complete: {resolved_name} | "
            f"{summarize_lora_mode(training_config)} | steps={global_step}"
        )
        finalize_training_job(
            node_id,
            status="completed",
            current_step=global_step,
            total_steps=total_steps,
        )
        _write_progress(
            progress_file,
            status="completed",
            phase="done",
            step=global_step,
            total_steps=total_steps,
            output_adapter=final_adapter_dir,
            summary=summary,
        )
        return {
            "type": "training_artifacts",
            "engine_type": "moss_tts",
            "training_mode": "lora_adapter",
            "model_variant": "MOSS-TTS",
            "model_path": final_adapter_dir,
            "job_dir": job_dir,
            "summary": summary,
            "lora_adapter": {
                "type": "moss_lora",
                "adapter_path": final_adapter_dir,
                "base_model_name_or_path": "OpenMOSS-Team/MOSS-TTS",
            },
        }
    except Exception:
        finalize_training_job(node_id, status="error")
        raise

