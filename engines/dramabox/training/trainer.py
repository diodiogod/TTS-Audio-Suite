"""Process runner for the official DramaBox IC-LoRA trainer."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import folder_paths

from engines.dramabox.dramabox_downloader import DramaBoxDownloader
from engines.training.progress_io import write_json_progress_file
from engines.training.progress_registry import (
    finalize_training_job,
    register_training_job,
    update_training_job,
)

from .dataset import (
    get_dramabox_training_root,
    slugify,
    validate_preprocessed_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = PROJECT_ROOT / "engines" / "dramabox" / "vendor"
PREPROCESS_SCRIPT = VENDOR_ROOT / "src" / "preprocess.py"
TRAIN_SCRIPT = VENDOR_ROOT / "src" / "train.py"


def _write_progress(progress_file: str, *, status: str, phase: str, **updates: Any) -> None:
    payload: Dict[str, Any] = {}
    if progress_file and os.path.isfile(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
                if isinstance(existing, dict):
                    payload.update(existing)
        except Exception:
            pass
    payload.update(updates)
    payload["status"] = status
    payload["phase"] = phase
    payload["updated_at"] = datetime.now().isoformat()
    if progress_file:
        write_json_progress_file(progress_file, payload, default=str)


def _interrupt_requested() -> bool:
    try:
        import comfy.model_management as model_management
    except Exception:
        return False
    try:
        return bool(model_management.processing_interrupted())
    except Exception:
        return bool(getattr(model_management, "interrupt_processing", False))


def _device_environment(shared_settings: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    device = str(shared_settings.get("device", "auto") or "auto").strip().lower()
    if device.startswith("cpu"):
        # CPU mode is explicit. This also prevents a CUDA-enabled torch build
        # from silently taking the user's GPU during preprocessing.
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    return env


def _run_process(
    command: Iterable[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    phase: str,
    progress_file: str = "",
    node_id: str = "",
    total_steps: int = 0,
) -> None:
    command = [str(value) for value in command]
    print(f"🎓 DramaBox {phase} command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    tail: list[str] = []
    recent_loss_trace: list[Dict[str, Any]] = []
    best_loss: Optional[float] = None
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if line:
                telemetry_match = re.fullmatch(
                    r"TTS_SUITE_PROGRESS\s+step=(\d+)\s+total=(\d+)", line
                )
                if telemetry_match is None:
                    print(f"[DramaBox {phase}] {line}")
                    tail.append(line)
                    del tail[:-30]

                if progress_file:
                    match = telemetry_match or re.search(
                        r"(?:Step|step)\s+(\d+)(?:/(\d+))?", line
                    )
                    if match:
                        step = int(match.group(1))
                        parsed_total = int(match.group(2) or total_steps or 0)
                        overall_progress = (step / parsed_total) if parsed_total else 0.0
                        progress_updates: Dict[str, Any] = {
                            "step": step,
                            "total_steps": parsed_total,
                            "overall_progress": overall_progress,
                            "latest_log": line,
                        }
                        loss_match = re.search(
                            r"\bloss=([-+0-9.eE]+)", line, re.IGNORECASE
                        )
                        if loss_match:
                            loss_value = float(loss_match.group(1))
                            lr_match = re.search(
                                r"\blr=([-+0-9.eE]+)", line, re.IGNORECASE
                            )
                            learning_rate = (
                                float(lr_match.group(1)) if lr_match else None
                            )
                            recent_loss_trace.append(
                                {"step": step, "total_loss": loss_value}
                            )
                            recent_loss_trace = recent_loss_trace[-120:]
                            best_loss = (
                                loss_value
                                if best_loss is None
                                else min(best_loss, loss_value)
                            )
                            progress_updates.update(
                                latest_loss=loss_value,
                                best_gen_loss=best_loss,
                                recent_loss_trace=recent_loss_trace,
                                current_metrics={
                                    "loss_gen_all": loss_value,
                                    "loss_disc_all": 0.0,
                                    "loss_mel": 0.0,
                                    "loss_kl": 0.0,
                                    "loss_fm": 0.0,
                                    "learning_rate": learning_rate,
                                },
                            )
                        _write_progress(
                            progress_file,
                            status="running",
                            phase=phase,
                            **progress_updates,
                        )
                        update_training_job(
                            node_id,
                            status="running",
                            phase=phase,
                            **progress_updates,
                        )
                    elif "encoding:" in line.lower():
                        match = re.search(r"(\d+)\s*/\s*(\d+)", line)
                        if match:
                            step = int(match.group(1))
                            parsed_total = int(match.group(2))
                            overall_progress = step / max(parsed_total, 1)
                            _write_progress(
                                progress_file,
                                status="running",
                                phase=phase,
                                step=step,
                                total_steps=parsed_total,
                                overall_progress=overall_progress,
                                latest_log=line,
                            )
                            update_training_job(
                                node_id,
                                status="running",
                                phase=phase,
                                step=step,
                                total_steps=parsed_total,
                                overall_progress=overall_progress,
                                latest_log=line,
                            )

            if _interrupt_requested():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise InterruptedError(f"DramaBox {phase} interrupted by user")

        return_code = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
        raise

    if return_code != 0:
        details = "\n".join(tail[-10:])
        raise RuntimeError(
            f"DramaBox {phase} process failed with exit code {return_code}."
            + (f"\nLast output:\n{details}" if details else "")
        )


def _resolve_model_paths(shared_settings: Dict[str, Any]) -> Dict[str, str]:
    model_name = str(shared_settings.get("model_name", "DramaBox") or "DramaBox")
    return DramaBoxDownloader().resolve_model_path(model_name)


def build_preprocess_command(
    dataset_info: Dict[str, Any],
    shared_settings: Dict[str, Any],
    *,
    batch_size: int = 8,
    skip_existing: bool = True,
) -> list[str]:
    paths = _resolve_model_paths(shared_settings)
    command = [
        sys.executable,
        str(PREPROCESS_SCRIPT),
        "--dataset-type",
        "gemini_synthetic",
        "--index",
        str(dataset_info["index_path"]),
        "--output-dir",
        str(dataset_info["preprocessed_dir"]),
        "--checkpoint",
        paths["audio_components"],
        "--audio-only-ckpt",
        paths["audio_components"],
        "--gemma-root",
        paths["gemma_root"],
        "--max-duration",
        str(float(dataset_info.get("max_duration", 20.0))),
        "--min-duration",
        str(float(dataset_info.get("min_duration", 2.0))),
        "--batch-size",
        str(max(1, int(batch_size))),
    ]
    if skip_existing:
        command.append("--skip-existing")
    return command


def run_dramabox_preprocess(
    dataset_info: Dict[str, Any],
    shared_settings: Dict[str, Any],
    *,
    batch_size: int = 8,
    progress_file: str = "",
    node_id: str = "",
) -> Dict[str, Any]:
    command = build_preprocess_command(
        dataset_info,
        shared_settings,
        batch_size=batch_size,
        skip_existing=True,
    )
    _run_process(
        command,
        cwd=VENDOR_ROOT,
        env=_device_environment(shared_settings),
        phase="preprocess",
        progress_file=progress_file,
        node_id=node_id,
    )
    validate_preprocessed_dataset(
        dataset_info.get("records") or [],
        dataset_info["preprocessed_dir"],
        raise_on_missing=True,
    )
    dataset_info["preprocessed"] = True
    return dataset_info


def _resolve_validation_config(value: str) -> str:
    raw = os.path.expanduser(str(value or "").strip())
    if not raw:
        return ""
    candidates = [Path(raw)]
    if not os.path.isabs(raw):
        candidates.extend(
            (
                Path(folder_paths.get_input_directory()) / raw,
                VENDOR_ROOT / raw,
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"DramaBox validation config not found: {value}")


def _validation_gpu(training_device: str, requested_gpu: Any) -> str:
    value = str(requested_gpu or "").strip()
    if not value:
        raise ValueError(
            "DramaBox validation_config requires validation_gpu because official validation "
            "runs a second full model process. Reserve a GPU different from the training GPU."
        )
    if not value.isdigit():
        raise ValueError("DramaBox validation_gpu must be a non-negative CUDA device index")
    device = str(training_device or "auto").strip().lower()
    training_gpu = device.split(":", 1)[1] if device.startswith("cuda:") else "0"
    if value == training_gpu:
        raise ValueError(
            f"DramaBox validation_gpu ({value}) must differ from the training GPU ({training_gpu})"
        )
    return value


def _resolve_continue_lora(continue_from: Any) -> str:
    if continue_from is None:
        return ""
    if isinstance(continue_from, str):
        value = os.path.abspath(os.path.expanduser(continue_from.strip()))
    elif isinstance(continue_from, dict):
        if str(continue_from.get("engine_type", "") or "").strip().lower() not in {"", "dramabox"}:
            raise ValueError("continue_from TRAINING_ARTIFACTS must come from a DramaBox training run")
        value = str(
            continue_from.get("lora_path")
            or continue_from.get("model_path")
            or (continue_from.get("lora_adapter") or {}).get("adapter_path", "")
        ).strip()
        value = os.path.abspath(os.path.expanduser(value)) if value else ""
    else:
        raise ValueError("Unsupported DramaBox continue_from input")

    if not value:
        return ""
    if os.path.isdir(value):
        candidates = sorted(Path(value).glob("lora_step_*.safetensors"))
        candidates += [Path(value) / "adapter_model.safetensors"]
        for candidate in reversed(candidates):
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(f"No DramaBox LoRA weights found in '{value}'")
    if not os.path.isfile(value):
        raise FileNotFoundError(f"DramaBox LoRA checkpoint not found: {value}")
    return value


def _managed_lora_root() -> Path:
    try:
        from utils.models.extra_paths import get_all_tts_model_paths

        for base_path in get_all_tts_model_paths("TTS"):
            root = Path(base_path) / "dramabox" / "loras"
            root.mkdir(parents=True, exist_ok=True)
            return root
    except Exception:
        pass
    root = Path(folder_paths.models_dir) / "TTS" / "dramabox" / "loras"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _next_managed_lora_dir(name: str, *, overwrite: bool) -> Path:
    target = _managed_lora_root() / slugify(name)
    if overwrite or not target.exists():
        return target
    counter = 2
    while True:
        candidate = target.parent / f"{target.name}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def _latest_lora_file(output_dir: Path) -> Optional[Path]:
    candidates = sorted(
        output_dir.glob("lora_step_*.safetensors"),
        key=lambda path: int(re.search(r"(\d+)", path.stem).group(1))
        if re.search(r"(\d+)", path.stem)
        else -1,
    )
    if candidates:
        return candidates[-1]
    candidate = output_dir / "adapter_model.safetensors"
    return candidate if candidate.is_file() else None


def _build_train_config(
    dataset_info: Dict[str, Any],
    shared_settings: Dict[str, Any],
    training_config: Dict[str, Any],
    *,
    output_dir: Path,
    continue_lora: str,
    resolve_paths: bool = True,
) -> Dict[str, Any]:
    if shared_settings.get("model_paths"):
        paths = dict(shared_settings["model_paths"])
    elif resolve_paths:
        paths = _resolve_model_paths(shared_settings)
    else:
        paths = {
            "transformer": "<dramabox-transformer.safetensors>",
            "audio_components": "<dramabox-audio-components.safetensors>",
        }
    config: Dict[str, Any] = {
        "data_dir": [str(dataset_info["preprocessed_dir"])],
        "speaker_index": [str(dataset_info["index_path"])],
        "output_dir": str(output_dir),
        "checkpoint": paths["transformer"],
        "full_checkpoint": paths["audio_components"],
        "base_model": str(training_config.get("base_model", "dev")),
        "lora_rank": int(training_config.get("lora_rank", 128)),
        "lora_alpha": int(training_config.get("lora_alpha", 128)),
        "lora_dropout": float(training_config.get("lora_dropout", 0.1)),
        "ref_ratio": float(training_config.get("ref_ratio", 0.3)),
        "max_ref_tokens": int(training_config.get("max_ref_tokens", 200)),
        "text_dropout": float(training_config.get("text_dropout", 0.4)),
        "steps": int(training_config.get("steps", 10000)),
        "lr": float(training_config.get("learning_rate", 1e-4)),
        "lr_scheduler": str(training_config.get("lr_scheduler", "cosine")),
        "warmup_steps": int(training_config.get("warmup_steps", 500)),
        "batch_size": int(training_config.get("batch_size", 1)),
        "grad_accum": int(training_config.get("grad_accum", 4)),
        "max_grad_norm": float(training_config.get("max_grad_norm", 1.0)),
        "save_every": max(1, int(training_config.get("save_every", 500))),
        "log_every": int(training_config.get("log_every", 10)),
        "seed": int(training_config.get("seed", 42)),
    }
    if continue_lora:
        config["resume_lora"] = continue_lora
    validation_config = _resolve_validation_config(
        training_config.get("validation_config", "")
    )
    if validation_config:
        config["val_config"] = validation_config
    return config


def _accelerate_command() -> list[str]:
    executable = shutil.which("accelerate")
    if executable:
        return [executable, "launch", "--num_processes", "1"]
    return [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", "1"]


def run_dramabox_training_job(
    shared_settings: Dict[str, Any],
    dataset_info: Dict[str, Any],
    training_config: Dict[str, Any],
    *,
    output_name: str = "",
    resume: bool = False,
    overwrite: bool = False,
    continue_from: Any = None,
    node_id: str = "",
) -> Dict[str, Any]:
    if str(dataset_info.get("engine_type", "") or "").strip().lower() != "dramabox":
        raise ValueError("DramaBox training requires a DramaBox TRAINING_DATASET payload")
    if str(training_config.get("training_mode", "audio_lora") or "").strip().lower() != "audio_lora":
        raise ValueError("DramaBox training currently supports audio_lora mode only")
    if resume:
        raise RuntimeError(
            "DramaBox does not support exact optimizer-state resume. Use continue_from with a saved LoRA checkpoint for a warm start."
        )
    if str(shared_settings.get("device", "auto") or "auto").strip().lower().startswith("cpu") and not bool(
        training_config.get("dry_run", False)
    ):
        raise RuntimeError(
            "DramaBox model training requires CUDA. Use dry_run for CPU-only validation; "
            "no model weights or CUDA process will be started in that mode."
        )
    requested_validation = str(
        training_config.get("validation_config", "") or ""
    ).strip()
    if requested_validation:
        _resolve_validation_config(requested_validation)
        _validation_gpu(
            shared_settings.get("device", "auto"),
            training_config.get("validation_gpu", ""),
        )

    safe_name = slugify(output_name or dataset_info.get("model_name") or "dramabox_lora")
    root = Path(get_dramabox_training_root()) / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = f"{safe_name}|{dataset_info.get('index_path')}|{training_config}"
    job_hash = __import__("hashlib").sha256(fingerprint.encode("utf-8")).hexdigest()[:12]
    job_dir = root / f"{safe_name}_{job_hash}"
    if job_dir.exists() and not overwrite:
        job_dir = root / f"{safe_name}_{job_hash}_{int(time.time())}"
    if overwrite and job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    train_output_dir = job_dir / "lora"
    progress_file = str(job_dir / "progress.json")
    managed_dir = _next_managed_lora_dir(safe_name, overwrite=overwrite)
    continue_lora = _resolve_continue_lora(continue_from)

    register_training_job(
        node_id,
        engine_type="dramabox",
        progress_file=progress_file,
        job_dir=str(job_dir),
        model_name=safe_name,
        sample_rate="48k",
        total_epochs=1,
    )
    try:
        _write_progress(
            progress_file,
            status="starting",
            phase="setup",
            engine_type="dramabox",
            model_name=safe_name,
            dataset_records=int(dataset_info.get("train_records", 0)),
            speakers=dataset_info.get("speakers", []),
            started_at=time.time(),
        )

        if not bool(dataset_info.get("preprocessed")):
            if bool(training_config.get("dry_run", False)):
                print("🧪 DramaBox dry-run: skipping GPU dataset preprocessing")
            else:
                _write_progress(progress_file, status="running", phase="preprocess")
                run_dramabox_preprocess(
                    dataset_info,
                    shared_settings,
                    batch_size=int(training_config.get("preprocess_batch_size", 8)),
                    progress_file=progress_file,
                    node_id=node_id,
                )

        train_config = _build_train_config(
            dataset_info,
            shared_settings,
            training_config,
            output_dir=train_output_dir,
            continue_lora=continue_lora,
            resolve_paths=not bool(training_config.get("dry_run", False)),
        )
        config_path = job_dir / "training_config.yaml"
        import yaml

        config_path.write_text(yaml.safe_dump(train_config, sort_keys=False), encoding="utf-8")
        (job_dir / "resolved_training_config.json").write_text(
            json.dumps(
                {
                    "dataset": dataset_info,
                    "shared_settings": shared_settings,
                    "training_config": training_config,
                    "official_config": train_config,
                    "continue_from": continue_lora,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        command = [*_accelerate_command(), str(TRAIN_SCRIPT), "--config", str(config_path)]
        if bool(training_config.get("dry_run", False)):
            summary = (
                f"DramaBox dry-run ready: {safe_name} | {dataset_info.get('train_records', 0)} rows | "
                f"official command prepared without loading CUDA or model weights"
            )
            _write_progress(
                progress_file,
                status="completed",
                phase="dry_run",
                overall_progress=1.0,
                summary=summary,
                command=command,
            )
            finalize_training_job(node_id, status="completed", summary=summary, dry_run=True)
            return {
                "type": "training_artifacts",
                "engine_type": "dramabox",
                "training_mode": "audio_lora",
                "dry_run": True,
                "job_dir": str(job_dir),
                "training_config": str(config_path),
                "summary": summary,
                "command": command,
            }

        _write_progress(progress_file, status="running", phase="train", total_steps=int(train_config["steps"]))
        train_env = _device_environment(shared_settings)
        if train_config.get("val_config"):
            paths = _resolve_model_paths(shared_settings)
            train_env["LTX_CHECKPOINT"] = paths["transformer"]
            train_env["LTX_FULL_CHECKPOINT"] = paths["audio_components"]
            train_env["GEMMA_ROOT"] = paths["gemma_root"]
            train_env["TRAIN_VAL_GPU"] = _validation_gpu(
                shared_settings.get("device", "auto"),
                training_config.get("validation_gpu", ""),
            )
        _run_process(
            command,
            cwd=VENDOR_ROOT,
            env=train_env,
            phase="train",
            progress_file=progress_file,
            node_id=node_id,
            total_steps=int(train_config["steps"]),
        )

        selected_lora = _latest_lora_file(train_output_dir)
        if selected_lora is None:
            raise RuntimeError(
                f"DramaBox training exited successfully but produced no LoRA file in '{train_output_dir}'."
            )
        if managed_dir.exists():
            shutil.rmtree(managed_dir)
        managed_dir.mkdir(parents=True, exist_ok=True)
        managed_lora = managed_dir / selected_lora.name
        shutil.copy2(selected_lora, managed_lora)
        if selected_lora.name != "adapter_model.safetensors":
            shutil.copy2(selected_lora, managed_dir / "adapter_model.safetensors")
        adapter_config = train_output_dir / "adapter_config.json"
        if adapter_config.is_file():
            shutil.copy2(adapter_config, managed_dir / adapter_config.name)
        shutil.copy2(config_path, managed_dir / "training_config.yaml")

        summary = (
            f"DramaBox audio LoRA training complete: {safe_name} | "
            f"steps={train_config['steps']} | adapter={managed_lora}"
        )
        _write_progress(
            progress_file,
            status="completed",
            phase="done",
            overall_progress=1.0,
            output_adapter=str(managed_lora),
            output_dir=str(managed_dir),
            summary=summary,
        )
        finalize_training_job(
            node_id,
            status="completed",
            output_adapter=str(managed_lora),
            summary=summary,
        )
        return {
            "type": "training_artifacts",
            "engine_type": "dramabox",
            "training_mode": "audio_lora",
            "model_path": str(managed_dir),
            "lora_path": str(managed_lora),
            "job_dir": str(job_dir),
            "summary": summary,
            "lora_adapter": {
                "type": "dramabox_lora",
                "adapter_path": str(managed_lora),
                "adapter_dir": str(managed_dir),
            },
        }
    except InterruptedError as error:
        _write_progress(progress_file, status="cancelled", phase="cancelled", error=str(error))
        finalize_training_job(node_id, status="cancelled", error=str(error))
        raise
    except Exception as error:
        _write_progress(progress_file, status="error", phase="error", error=str(error))
        finalize_training_job(node_id, status="error", error=str(error))
        raise


__all__ = [
    "build_preprocess_command",
    "run_dramabox_preprocess",
    "run_dramabox_training_job",
]
