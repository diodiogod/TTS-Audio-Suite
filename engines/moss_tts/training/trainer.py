"""
LoRA training runner for MOSS-TTS Delay 8B.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from contextlib import contextmanager
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


@contextmanager
def _quiet_transformers_progress(transformers_module):
    logging_module = getattr(getattr(transformers_module, "utils", None), "logging", None)
    if logging_module is None:
        yield
        return
    logging_module.disable_progress_bar()
    try:
        yield
    finally:
        logging_module.enable_progress_bar()


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


def _scalar_metric_value(accelerator, value):
    try:
        import torch
    except ImportError:
        return float(value)

    if isinstance(value, torch.Tensor):
        tensor = value.detach().float()
    else:
        tensor = torch.tensor(float(value), dtype=torch.float32, device=accelerator.device)

    if getattr(accelerator, "num_processes", 1) > 1:
        tensor = accelerator.gather(tensor.reshape(1)).mean()
    return float(tensor.item())


def _interrupt_requested() -> bool:
    try:
        import comfy.model_management as comfy_model_management
    except Exception:
        return False

    try:
        return bool(comfy_model_management.processing_interrupted())
    except Exception:
        return bool(getattr(comfy_model_management, "interrupt_processing", False))


def _raise_if_interrupted() -> None:
    if _interrupt_requested():
        raise InterruptedError("MOSS training interrupted by user")


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
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        import transformers
        from transformers import BitsAndBytesConfig
        from transformers.generation import GenerationMixin
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError as e:
        raise RuntimeError(
            "MOSS LoRA training requires accelerate, peft, transformers, and torch runtime dependencies. "
            "Run install.py again after pulling the latest version."
        ) from e

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
        base_quantization = str(training_config.get("base_quantization", "none") or "none").strip().lower()
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
        quantization_config = None
        if base_quantization == "4bit_nf4":
            if not torch.cuda.is_available():
                raise RuntimeError("MOSS QLoRA 4-bit loading requires CUDA. CPU 4-bit training is not supported here.")
            compute_dtype_mode = str(training_config.get("bnb_4bit_compute_dtype", "auto") or "auto").strip().lower()
            compute_dtype_map = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }
            bnb_compute_dtype = model_dtype if compute_dtype_mode == "auto" else compute_dtype_map.get(compute_dtype_mode)
            if bnb_compute_dtype is None:
                raise RuntimeError(f"Unsupported MOSS bnb_4bit_compute_dtype mode: {compute_dtype_mode}")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=bool(training_config.get("bnb_4bit_use_double_quant", True)),
                bnb_4bit_compute_dtype=bnb_compute_dtype,
            )

        train_dataset = MossTTSSFTDataset(train_records, processor)
        val_dataset = MossTTSSFTDataset(val_records, processor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=max(1, int(training_config.get("batch_size", 1))),
            shuffle=True,
            num_workers=max(0, int(training_config.get("num_workers", 0))),
            pin_memory=torch.cuda.is_available(),
            collate_fn=train_dataset.collate_fn,
            drop_last=False,
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

        if len(train_loader) <= 0:
            raise RuntimeError(
                "MOSS training produced zero train batches. "
                "This usually means the batch size is larger than the prepared train split."
            )

        # Compatibility shim for upstream MOSS remote code on older transformers builds.
        if not hasattr(transformers, "initialization"):
            transformers.initialization = torch.nn.init

        RemoteDelayModel = get_class_from_dynamic_module(
            "modeling_moss_tts.MossTTSDelayModel",
            model_path,
            local_files_only=True,
        )
        if issubclass(RemoteDelayModel, GenerationMixin):
            PatchedRemoteDelayModel = RemoteDelayModel
        else:
            class PatchedRemoteDelayModel(RemoteDelayModel, GenerationMixin):
                pass

            PatchedRemoteDelayModel.__name__ = RemoteDelayModel.__name__
            PatchedRemoteDelayModel.__qualname__ = RemoteDelayModel.__qualname__
            PatchedRemoteDelayModel.__module__ = RemoteDelayModel.__module__

        original_remote_get_input_embeddings = PatchedRemoteDelayModel.get_input_embeddings

        def _remote_get_input_embeddings(self, input_ids=None):
            if input_ids is None:
                return self.language_model.get_input_embeddings()
            return original_remote_get_input_embeddings(self, input_ids)

        PatchedRemoteDelayModel.get_input_embeddings = _remote_get_input_embeddings
        if not hasattr(PatchedRemoteDelayModel, "prepare_inputs_for_generation"):
            def _remote_prepare_inputs_for_generation(
                self,
                input_ids,
                past_key_values=None,
                attention_mask=None,
                inputs_embeds=None,
                cache_position=None,
                position_ids=None,
                use_cache=True,
                **kwargs,
            ):
                if past_key_values is not None:
                    input_ids = input_ids[:, -1:, :]
                    if cache_position is not None:
                        cache_position = cache_position[-1:]
                    if position_ids is not None:
                        position_ids = position_ids[:, -1:]

                return {
                    "input_ids": input_ids,
                    "past_key_values": past_key_values,
                    "attention_mask": attention_mask,
                    "inputs_embeds": inputs_embeds,
                    "cache_position": cache_position,
                    "position_ids": position_ids,
                    "use_cache": use_cache,
                    **kwargs,
                }

            PatchedRemoteDelayModel.prepare_inputs_for_generation = _remote_prepare_inputs_for_generation

        with _quiet_transformers_progress(transformers):
            load_kwargs = {
                "attn_implementation": attn_implementation,
                "dtype": model_dtype,
            }
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = {"": 0}
            model = PatchedRemoteDelayModel.from_pretrained(
                model_path,
                **load_kwargs,
            )

        target_map = {
            "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"],
            "mlp_plus_o": ["gate_proj", "up_proj", "down_proj", "o_proj"],
        }
        module_mode = str(training_config.get("trainable_lora_modules", "mlp"))
        if module_mode not in target_map:
            raise RuntimeError(f"Unsupported MOSS trainable_lora_modules mode: {module_mode}")

        if quantization_config is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool(training_config.get("gradient_checkpointing", True)),
            )
        else:
            for param in model.parameters():
                param.requires_grad = False

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

        allowed_fragments = ("language_model.layers.",)
        allowed_lora_modules = tuple(target_map[module_mode])
        trainable_names = []
        for name, param in model.named_parameters():
            is_lora_param = "lora_" in name
            in_allowed_scope = any(fragment in name for fragment in allowed_fragments)
            in_allowed_lora_module = any(key in name for key in allowed_lora_modules)
            param.requires_grad = is_lora_param and in_allowed_scope and in_allowed_lora_module
            if "emb_ext" in name:
                param.requires_grad = False
            if param.requires_grad:
                trainable_names.append(name)

        if not trainable_names:
            raise RuntimeError(
                "No trainable MOSS LoRA parameters were activated. "
                "Confirm that target modules exist under language_model.layers.* for this model."
            )

        base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        language_model = getattr(base_model, "language_model", None)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        if language_model is not None and hasattr(language_model, "enable_input_require_grads"):
            language_model.enable_input_require_grads()
        if bool(training_config.get("gradient_checkpointing", True)):
            if language_model is not None and hasattr(language_model, "gradient_checkpointing_enable"):
                language_model.gradient_checkpointing_enable()
            elif hasattr(base_model, "gradient_checkpointing_enable"):
                base_model.gradient_checkpointing_enable()
        for cfg_obj in (
            getattr(model, "config", None),
            getattr(base_model, "config", None),
            getattr(language_model, "config", None) if language_model is not None else None,
        ):
            if cfg_obj is not None and hasattr(cfg_obj, "use_cache"):
                cfg_obj.use_cache = False

        grad_accum = max(1, int(training_config.get("gradient_accumulation_steps", 1)))
        num_epochs = max(1, int(training_config.get("epochs", 3)))
        max_train_steps = int(training_config.get("max_train_steps", 0) or 0)
        num_update_steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
        if max_train_steps > 0:
            total_steps = max_train_steps
            num_epochs = max(1, math.ceil(total_steps / num_update_steps_per_epoch))
        else:
            total_steps = num_epochs * num_update_steps_per_epoch
        warmup_steps = max(0, int(training_config.get("warmup_steps", 100)))
        save_steps = max(0, int(training_config.get("save_steps", 500)))
        eval_steps = max(0, int(training_config.get("eval_steps", 500)))
        log_steps = max(1, int(training_config.get("log_steps", 10)))

        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = AdamW(
            trainable_params,
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
                    _raise_if_interrupted()
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
            node_id=str(node_id or ""),
            engine_type="moss_tts",
            model_name=resolved_name,
            sample_rate="24k",
            total_epochs=num_epochs,
            step=0,
            total_steps=total_steps,
            steps_per_epoch=num_update_steps_per_epoch,
            completed_total_steps=0,
            current_epoch=0,
            current_step=0,
            epoch_progress=0.0,
            overall_progress=0.0,
            batch_size=int(training_config.get("batch_size", 1)),
            learning_rate=float(training_config.get("learning_rate", 2e-6)),
            clip_count=len(train_records),
            workers=int(training_config.get("num_workers", 0)),
            started_at=time.time(),
            history=[],
            recent_loss_trace=[],
            train_records=len(train_records),
            val_records=len(val_records),
        )

        recent_loss_trace = []
        history = []
        best_gen_loss = None

        for epoch in range(num_epochs):
            epoch_started_at = time.time()
            epoch_last_loss_value = None
            for step, batch in enumerate(train_loader, start=1):
                _raise_if_interrupted()
                with accelerator.accumulate(model):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                    if loss is None or not loss.requires_grad:
                        active_label_tokens = int((batch["labels"] != -100).sum().item())
                        first_trainable_names = [
                            name for name, param in model.named_parameters() if param.requires_grad
                        ][:8]
                        raise RuntimeError(
                            "MOSS LoRA training produced a detached loss. "
                            f"active_label_tokens={active_label_tokens}, "
                            f"trainable_param_count={len(trainable_params)}, "
                            f"example_trainable_params={first_trainable_names}"
                        )
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
                    loss_value = _scalar_metric_value(accelerator, loss)
                    epoch_last_loss_value = loss_value
                    current_epoch_progress = global_step - ((epoch) * num_update_steps_per_epoch)
                    current_epoch_progress = max(0, min(current_epoch_progress, num_update_steps_per_epoch))
                    completed_total_steps = min(global_step, total_steps)
                    overall_progress = completed_total_steps / max(total_steps, 1)
                    elapsed = max(1e-6, time.time() - start_time)
                    steps_per_sec = global_step / elapsed
                    eta_sec = ((total_steps - completed_total_steps) / steps_per_sec) if steps_per_sec > 0 else 0.0
                    if best_gen_loss is None or loss_value < best_gen_loss:
                        best_gen_loss = loss_value

                    recent_loss_trace.append(
                        {
                            "epoch": epoch + 1,
                            "step": int(current_epoch_progress),
                            "global_step": int(global_step),
                            "total_loss": float(loss_value),
                        }
                    )
                    recent_loss_trace = recent_loss_trace[-120:]

                    metrics_payload = {
                        "loss_gen_all": float(loss_value),
                        "loss_disc_all": 0.0,
                        "loss_mel": 0.0,
                        "loss_kl": 0.0,
                        "loss_fm": 0.0,
                    }

                    progress_payload = {
                        "step": global_step,
                        "total_steps": total_steps,
                        "current_epoch": epoch + 1,
                        "current_step": int(current_epoch_progress),
                        "steps_per_epoch": num_update_steps_per_epoch,
                        "completed_total_steps": int(completed_total_steps),
                        "epoch_progress": float(current_epoch_progress / max(num_update_steps_per_epoch, 1)),
                        "overall_progress": float(overall_progress),
                        "latest_loss": float(loss_value),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "it_per_sec": float(steps_per_sec),
                        "eta_sec": float(eta_sec),
                        "current_metrics": metrics_payload,
                        "recent_loss_trace": recent_loss_trace,
                        "best_gen_loss": float(best_gen_loss),
                        "history": history,
                    }
                    if global_step % log_steps == 0 or global_step == 1:
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
                            completed_total_steps=int(completed_total_steps),
                            steps_per_epoch=num_update_steps_per_epoch,
                            overall_progress=float(overall_progress),
                            epoch_progress=float(current_epoch_progress / max(num_update_steps_per_epoch, 1)),
                            it_per_sec=float(steps_per_sec),
                            eta_sec=float(eta_sec),
                            current_metrics=metrics_payload,
                            recent_loss_trace=recent_loss_trace,
                            best_gen_loss=float(best_gen_loss),
                            history=history,
                        )
                        _write_progress(
                            progress_file,
                            status="running",
                            phase="train",
                            **progress_payload,
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
                            current_epoch=epoch + 1,
                            current_step=int(current_epoch_progress),
                            completed_total_steps=int(completed_total_steps),
                            steps_per_epoch=num_update_steps_per_epoch,
                            overall_progress=float(overall_progress),
                            epoch_progress=float(current_epoch_progress / max(num_update_steps_per_epoch, 1)),
                            current_metrics=metrics_payload,
                            recent_loss_trace=recent_loss_trace,
                            best_gen_loss=float(best_gen_loss),
                            history=history,
                        )

                    if save_steps > 0 and global_step % save_steps == 0:
                        checkpoint_dir = os.path.join(job_dir, f"checkpoint-step-{global_step}")
                        accelerator.wait_for_everyone()
                        _raise_if_interrupted()
                        if accelerator.is_main_process:
                            accelerator.unwrap_model(model).save_pretrained(checkpoint_dir)
                            print(f"MOSS LoRA checkpoint saved: {checkpoint_dir}")

                    if global_step >= total_steps:
                        break

            if global_step >= total_steps:
                break

            if epoch_last_loss_value is not None:
                epoch_time_sec = max(time.time() - epoch_started_at, 0.0)
                history.append(
                    {
                        "epoch": epoch + 1,
                        "epoch_time_sec": float(epoch_time_sec),
                        "total_loss": float(epoch_last_loss_value),
                    }
                )

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
            current_epoch=min(num_epochs, math.ceil(global_step / max(num_update_steps_per_epoch, 1))),
            current_step=min(global_step, num_update_steps_per_epoch),
            completed_total_steps=int(global_step),
            steps_per_epoch=num_update_steps_per_epoch,
            epoch_progress=1.0 if global_step > 0 else 0.0,
            overall_progress=1.0 if total_steps > 0 and global_step >= total_steps else float(global_step / max(total_steps, 1)),
            output_adapter=final_adapter_dir,
            recent_loss_trace=recent_loss_trace,
            best_gen_loss=float(best_gen_loss) if best_gen_loss is not None else None,
            history=history,
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
    except InterruptedError as error:
        finalize_training_job(
            node_id,
            status="cancelled",
            current_step=locals().get("global_step", 0),
            total_steps=locals().get("total_steps", 0),
            error=str(error),
        )
        _write_progress(
            progress_file,
            status="cancelled",
            phase="cancelled",
            step=locals().get("global_step", 0),
            total_steps=locals().get("total_steps", 0),
            current_epoch=locals().get("epoch", -1) + 1 if "epoch" in locals() else 0,
            completed_total_steps=int(locals().get("global_step", 0)),
            recent_loss_trace=locals().get("recent_loss_trace", []),
            best_gen_loss=float(locals().get("best_gen_loss")) if locals().get("best_gen_loss") is not None else None,
            history=locals().get("history", []),
            error=str(error),
            summary="MOSS training cancelled by user",
        )
        raise
    except Exception:
        finalize_training_job(node_id, status="error")
        raise
