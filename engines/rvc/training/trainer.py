"""
Training runner for RVC voice models.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from glob import glob
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import folder_paths

from engines.rvc.training.index_builder import build_faiss_index
from engines.rvc.training.reference_bridge import import_reference_module
from engines.training.progress_io import write_json_progress_file
from engines.training.progress_registry import (
    finalize_training_job,
    register_training_job,
    update_training_job,
)
from utils.downloads.unified_downloader import unified_downloader
from utils.models.extra_paths import find_model_in_paths, get_all_tts_model_paths


PROJECT_ROOT = str(Path(__file__).resolve().parents[3])


def _prepare_spawn_import_order() -> None:
    """
    Windows spawn inherits sys.path from the parent process.

    This repo frequently prepends the custom node root to sys.path, which can
    shadow ComfyUI's own ``utils`` package when the child process re-runs
    ``ComfyUI/main.py``. Move this project root to the end so ComfyUI's root
    wins for absolute imports like ``utils.install_util``.
    """
    comfy_root = str(Path(folder_paths.__file__).resolve().parent)

    filtered = [path for path in sys.path if path not in {PROJECT_ROOT, comfy_root}]
    sys.path[:] = [comfy_root, *filtered, PROJECT_ROOT]


def _write_terminal_progress(progress_file: str, *, status: str, phase: str, **updates: Any) -> None:
    if not progress_file:
        return

    payload: Dict[str, Any] = {}
    if os.path.isfile(progress_file):
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
    write_json_progress_file(progress_file, payload, default=str)


def _slugify(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    safe = safe.strip("_")
    return safe or "rvc_model"


def _resolve_default_gpu_ids(device: str) -> str:
    device = str(device or "")
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    if device == "cuda":
        return "0"
    return ""


def _resolve_pretrained_checkpoint(relative_or_local: str) -> str:
    value = (relative_or_local or "").strip()
    if not value:
        return ""

    models_dir = folder_paths.models_dir

    if os.path.isabs(value) and os.path.exists(value):
        return value

    actual_value = value.replace("local:", "", 1) if value.startswith("local:") else value
    actual_basename = os.path.basename(actual_value)

    extra_path_match = None
    if "/" in actual_value or "\\" in actual_value:
        parts = actual_value.replace("\\", "/").split("/")
        if len(parts) >= 2:
            extra_path_match = find_model_in_paths(parts[-1], "TTS", parts[:-1])
    else:
        extra_path_match = find_model_in_paths(actual_basename, "TTS", ["pretrained_v2"])

    if extra_path_match and os.path.exists(extra_path_match):
        return extra_path_match

    candidate_paths = [
        os.path.join(unified_downloader.tts_dir, actual_value),
        os.path.join(models_dir, "TTS", actual_value),
        os.path.join(models_dir, actual_value),
    ]
    if "/" not in actual_value:
        for tts_root in get_all_tts_model_paths("TTS"):
            candidate_paths.append(os.path.join(tts_root, "pretrained_v2", actual_basename))
        candidate_paths.extend(
            [
                os.path.join(models_dir, "TTS", "pretrained_v2", actual_value),
                os.path.join(models_dir, "pretrained_v2", actual_value),
            ]
        )

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate

    if value.startswith("local:"):
        raise FileNotFoundError(f"Local pretrained checkpoint not found: {actual_value}")

    from engines.rvc.impl.rvc_downloader import RVC_DOWNLOAD_LINK

    target_path = os.path.join(unified_downloader.tts_dir, actual_value)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    download_ok = unified_downloader.download_file(
        f"{RVC_DOWNLOAD_LINK}{actual_value}",
        target_path,
        f"RVC pretrained checkpoint {actual_basename}",
    )
    if not download_ok:
        raise FileNotFoundError(f"Failed to download pretrained checkpoint: {actual_value}")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Failed to download pretrained checkpoint: {actual_value}")
    return target_path


def _default_pretrained_name(sample_rate: str, use_f0: bool, kind: str) -> str:
    sample_rate = str(sample_rate or "40k").strip().lower()
    kind = kind.upper()
    if kind not in {"G", "D"}:
        raise ValueError(f"Unsupported pretrained kind: {kind}")
    prefix = "f0" if use_f0 else ""
    return f"pretrained_v2/{prefix}{kind}{sample_rate}.pth"


def _resolve_training_pretrained_paths(
    dataset_info: Dict[str, Any],
    training_config: Dict[str, Any],
) -> Tuple[str, str]:
    sample_rate = str(dataset_info.get("sample_rate", "40k"))
    use_f0 = bool(dataset_info.get("if_f0", True))

    requested_generator = str(training_config.get("pretrained_generator", "") or "").strip()
    requested_discriminator = str(training_config.get("pretrained_discriminator", "") or "").strip()

    if requested_generator.lower() in {"", "auto", "default"}:
        requested_generator = _default_pretrained_name(sample_rate, use_f0, "G")
    if requested_discriminator.lower() in {"", "auto", "default"}:
        requested_discriminator = _default_pretrained_name(sample_rate, use_f0, "D")

    return (
        _resolve_pretrained_checkpoint(requested_generator),
        _resolve_pretrained_checkpoint(requested_discriminator),
    )


def _next_available_model_path(base_path: str) -> str:
    if not os.path.exists(base_path):
        return base_path

    stem, ext = os.path.splitext(base_path)
    counter = 2
    while True:
        candidate = f"{stem}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _prepare_output_paths(
    dataset_info: Dict[str, Any],
    training_config: Dict[str, Any],
    output_name: str,
    resume: bool,
    overwrite: bool,
    continue_from_fingerprint: str = "",
) -> Tuple[str, str, str]:
    safe_name = _slugify(output_name or dataset_info.get("model_name") or "rvc_model")
    sample_rate = dataset_info["sample_rate"]
    models_root = os.path.join(folder_paths.models_dir, "TTS", "RVC")
    index_root = os.path.join(models_root, ".index")
    training_root = os.path.join(folder_paths.get_output_directory(), "tts_audio_suite_training", "rvc", "jobs")
    os.makedirs(models_root, exist_ok=True)
    os.makedirs(index_root, exist_ok=True)
    os.makedirs(training_root, exist_ok=True)

    job_hash = hashlib.md5(
        json.dumps(
            {
                "dataset_dir": dataset_info["dataset_dir"],
                "sample_rate": sample_rate,
                "config": training_config,
                "name": safe_name,
                "continue_from": continue_from_fingerprint,
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:10]
    job_dir = os.path.join(training_root, f"{safe_name}_{job_hash}")
    model_path = os.path.join(models_root, f"{safe_name}_{sample_rate}.pth")

    if resume:
        existing_job_dir = _find_resume_job_dir(
            training_root,
            safe_name,
            dataset_info,
            sample_rate,
        )
        return existing_job_dir or job_dir, model_path, safe_name

    if overwrite:
        if os.path.isdir(job_dir):
            shutil.rmtree(job_dir)
        if os.path.exists(model_path):
            os.remove(model_path)
    else:
        if os.path.isdir(job_dir):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_dir = os.path.join(training_root, f"{safe_name}_{job_hash}_{timestamp}")
        model_path = _next_available_model_path(model_path)

    return job_dir, model_path, safe_name


def _find_resume_checkpoints(job_dir: str) -> Tuple[str, str]:
    def _checkpoint_sort_key(path: str):
        basename = os.path.basename(path)
        stem, _ = os.path.splitext(basename)
        if stem.endswith("_latest"):
            return (2, float("inf"), os.path.getmtime(path))
        digits = "".join(filter(str.isdigit, stem))
        if digits:
            return (1, int(digits), os.path.getmtime(path))
        return (0, 0, os.path.getmtime(path))

    generator_candidates = glob(os.path.join(job_dir, "G_*.pth"))
    discriminator_candidates = glob(os.path.join(job_dir, "D_*.pth"))
    generator_candidates.sort(key=_checkpoint_sort_key)
    discriminator_candidates.sort(key=_checkpoint_sort_key)

    generator_checkpoint = generator_candidates[-1] if generator_candidates else ""
    discriminator_checkpoint = discriminator_candidates[-1] if discriminator_candidates else ""
    return generator_checkpoint, discriminator_checkpoint


def _normalize_resume_identity_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return os.path.normcase(os.path.abspath(raw))
    except Exception:
        return os.path.normcase(raw)


def _read_resolved_training_config(job_dir: str) -> Dict[str, Any]:
    config_path = os.path.join(job_dir, "resolved_training_config.json")
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _find_resume_job_dir(
    training_root: str,
    safe_name: str,
    dataset_info: Dict[str, Any],
    sample_rate: str,
) -> Optional[str]:
    if not os.path.isdir(training_root):
        return None

    prefix = f"{safe_name}_"
    target_dataset_dir = _normalize_resume_identity_path(dataset_info.get("dataset_dir", ""))
    target_if_f0 = bool(dataset_info.get("if_f0", True))
    best_job_dir = None
    best_checkpoint_mtime = -1.0

    for entry in os.listdir(training_root):
        if not entry.startswith(prefix):
            continue

        job_dir = os.path.join(training_root, entry)
        if not os.path.isdir(job_dir):
            continue

        config = _read_resolved_training_config(job_dir)
        dataset_config = config.get("dataset") or {}
        config_dataset_dir = _normalize_resume_identity_path(dataset_config.get("dataset_dir", ""))
        config_sample_rate = str(dataset_config.get("sample_rate", "") or "")
        config_if_f0 = bool(dataset_config.get("if_f0", target_if_f0))

        if config_dataset_dir != target_dataset_dir:
            continue
        if config_sample_rate and config_sample_rate != sample_rate:
            continue
        if config_if_f0 != target_if_f0:
            continue

        generator_checkpoint, discriminator_checkpoint = _find_resume_checkpoints(job_dir)
        if not generator_checkpoint or not discriminator_checkpoint:
            continue

        checkpoint_mtime = max(
            os.path.getmtime(generator_checkpoint),
            os.path.getmtime(discriminator_checkpoint),
        )
        if checkpoint_mtime > best_checkpoint_mtime:
            best_checkpoint_mtime = checkpoint_mtime
            best_job_dir = job_dir

    return best_job_dir


def _resolve_max_checkpoints(training_config: Dict[str, Any]) -> int:
    if "max_checkpoints" in training_config:
        return max(1, int(training_config.get("max_checkpoints", 1)))

    legacy_keep_latest = training_config.get("save_latest_only", True)
    return 1 if bool(legacy_keep_latest) else 0


def _resolve_continue_from_model_path(continue_from: Any) -> str:
    if continue_from is None:
        return ""

    if isinstance(continue_from, str):
        value = continue_from.strip()
        if not value:
            return ""
        if os.path.exists(value):
            return value
        raise FileNotFoundError(f"Continue-from model path not found: {value}")

    if isinstance(continue_from, dict):
        data_type = str(continue_from.get("type", "") or "").strip().lower()

        if data_type == "training_artifacts":
            if str(continue_from.get("engine_type", "") or "").strip().lower() != "rvc":
                raise ValueError("continue_from TRAINING_ARTIFACTS must come from an RVC training run")
            artifact_model = continue_from.get("rvc_model")
            if isinstance(artifact_model, dict):
                model_path = str(artifact_model.get("model_path", "") or "").strip()
                if model_path and os.path.exists(model_path):
                    return model_path
            model_path = str(continue_from.get("model_path", "") or "").strip()
            if model_path and os.path.exists(model_path):
                return model_path
            raise FileNotFoundError("continue_from TRAINING_ARTIFACTS does not contain a valid RVC model_path")

        if data_type == "rvc_model":
            model_path = str(continue_from.get("model_path", "") or "").strip()
            if model_path and os.path.exists(model_path):
                return model_path
            raise FileNotFoundError("continue_from RVC_MODEL does not contain a valid model_path")

    raise ValueError(
        "Unsupported continue_from input. For RVC, use TRAINING_ARTIFACTS, RVC_MODEL, or a direct model path string."
    )


def run_rvc_training_job(
    shared_settings: Dict[str, Any],
    dataset_info: Dict[str, Any],
    training_config: Dict[str, Any],
    output_name: str = "",
    resume: bool = False,
    overwrite: bool = False,
    continue_from: Any = None,
    node_id: str = "",
) -> Dict[str, Any]:
    _prepare_spawn_import_order()
    train_utils = import_reference_module("lib.train.utils")
    training_cli = import_reference_module("training_cli")

    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive for RVC training")

    continue_from_model_path = _resolve_continue_from_model_path(continue_from)
    if resume and continue_from_model_path:
        raise ValueError("resume and continue_from are mutually exclusive for RVC training")

    sample_rate = dataset_info["sample_rate"]
    config_path = (
        Path(__file__).resolve().parents[1] / "impl" / "configs" / f"{sample_rate}{'' if sample_rate == '40k' else '_v2'}.json"
    )
    with open(config_path, "r", encoding="utf-8") as handle:
        config_data = json.load(handle)

    job_dir, model_path, resolved_name = _prepare_output_paths(
        dataset_info,
        training_config,
        output_name,
        resume,
        overwrite,
        continue_from_model_path,
    )
    os.makedirs(job_dir, exist_ok=True)
    progress_file = os.path.join(job_dir, "progress.json")

    epochs = int(training_config.get("epochs", 100))
    gpu_ids = str(training_config.get("gpu_ids", "")).strip() or _resolve_default_gpu_ids(shared_settings.get("device", ""))
    cache_data_in_gpu = bool(training_config.get("cache_data_in_gpu", True)) and gpu_ids != ""
    fp16_run = bool(training_config.get("fp16_run", True)) and gpu_ids != ""
    resolved_pretrain_g, resolved_pretrain_d = _resolve_training_pretrained_paths(dataset_info, training_config)
    initial_generator_path = continue_from_model_path or resolved_pretrain_g
    index_requested = bool(training_config.get("train_index", True))
    index_build_warning = ""

    hparams = train_utils.HParams(**config_data)
    hparams.experiment_dir = dataset_info["dataset_dir"]
    hparams.model_dir = job_dir
    hparams.save_every_epoch = int(training_config.get("save_every_epoch", 0))
    hparams.name = resolved_name
    hparams.total_epoch = epochs
    hparams.pretrainG = initial_generator_path
    hparams.pretrainD = resolved_pretrain_d
    hparams.continue_from_model_path = continue_from_model_path
    hparams.max_checkpoints = _resolve_max_checkpoints(training_config)
    hparams.version = "v2"
    hparams.gpus = gpu_ids
    hparams.sample_rate = sample_rate
    hparams.if_f0 = bool(dataset_info.get("if_f0", True))
    hparams.save_every_weights = bool(training_config.get("save_every_weights", False))
    hparams.if_cache_data_in_gpu = cache_data_in_gpu
    hparams.data.training_files = dataset_info["training_files"]
    hparams.save_best_model = bool(training_config.get("save_best_model", True))
    hparams.best_model_threshold = int(training_config.get("best_model_threshold", 30))
    hparams.log_every_epoch = float(training_config.get("log_every_epoch", 1.0))
    hparams.train.update(
        {
            "epochs": epochs,
            "batch_size": int(training_config.get("batch_size", 4)),
            "learning_rate": float(training_config.get("learning_rate", 1e-4)),
            "fp16_run": fp16_run,
            "c_adv": float(training_config.get("c_adv", 1.0)),
            "c_mel": float(training_config.get("c_mel", 45.0)),
            "c_kl": float(training_config.get("c_kl", 1.0)),
            "c_fm": float(training_config.get("c_fm", 2.0)),
            "c_tefs": float(training_config.get("c_tefs", 0.0)),
            "c_hd": float(training_config.get("c_hd", 0.0)),
            "c_tsi": float(training_config.get("c_tsi", 0.0)),
            "c_gp": float(training_config.get("c_gp", 0.0)),
            "use_multiscale": bool(training_config.get("use_multiscale", False)),
            "use_balancer": bool(training_config.get("use_balancer", False)),
            "use_pareto": bool(training_config.get("use_pareto", False)),
            "fast_mode": bool(training_config.get("fast_mode", False)),
        }
    )
    requested_num_workers = int(training_config.get("num_workers", 1))
    if platform.system() == "Windows" and requested_num_workers > 0:
        print(
            "RVC training: forcing dataloader workers to 0 on Windows "
            "to avoid nested spawn import failures."
        )
        requested_num_workers = 0
    hparams.train.num_workers = requested_num_workers
    hparams.model_path = model_path
    hparams.progress_file = progress_file
    hparams.cancel_flag_path = os.path.join(job_dir, "cancel.flag")
    hparams.node_id = str(node_id or "")

    resolved_config_path = os.path.join(job_dir, "resolved_training_config.json")
    with open(resolved_config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": dataset_info,
                "training_config": training_config,
                "shared_settings": shared_settings,
                "continue_from_model_path": continue_from_model_path,
                "resolved_pretrained_generator": resolved_pretrain_g,
                "resolved_pretrained_discriminator": resolved_pretrain_d,
            },
            handle,
            indent=2,
            sort_keys=True,
            default=str,
        )

    register_training_job(
        node_id,
        engine_type="rvc",
        progress_file=progress_file,
        job_dir=job_dir,
        model_name=resolved_name,
        sample_rate=sample_rate,
        total_epochs=epochs,
    )

    try:
        if resume:
            generator_checkpoint, discriminator_checkpoint = _find_resume_checkpoints(job_dir)
            if not generator_checkpoint or not discriminator_checkpoint:
                raise RuntimeError(
                    "Resume requested, but no saved RVC training checkpoints were found for a compatible "
                    f"RVC job (name='{resolved_name}', sample_rate='{sample_rate}', dataset='{dataset_info['dataset_dir']}'). "
                    "Resume needs saved numbered G_*.pth and D_*.pth checkpoints. "
                    "Set 'save_every_epoch' above 0 for resumable runs, or disable resume."
                )

        update_training_job(
            node_id,
            status="running",
            phase="building_index" if index_requested else "starting",
            model_path=model_path,
            config_path=resolved_config_path,
            pretrained_generator=initial_generator_path,
            pretrained_discriminator=resolved_pretrain_d,
            continue_from_model_path=continue_from_model_path,
        )

        index_path = None
        if index_requested:
            try:
                index_path = build_faiss_index(
                    dataset_dir=dataset_info["dataset_dir"],
                    sample_rate=sample_rate,
                    model_name=resolved_name,
                    index_dir=os.path.join(folder_paths.models_dir, "TTS", "RVC", ".index"),
                    overwrite=overwrite,
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                if "faiss" not in message or "scikit-learn" not in message:
                    raise
                index_build_warning = str(exc)
                print(f"⚠️ RVC training: {exc} Continuing without index build.")

        update_training_job(
            node_id,
            status="running",
            phase="training",
            index_path=index_path,
            index_requested=index_requested,
            index_built=bool(index_path),
            warning=index_build_warning or None,
        )

        if resume or not os.path.isfile(model_path):
            training_cli.train_model(hparams)

        update_training_job(
            node_id,
            status="running",
            phase="finalizing",
        )

        if not os.path.isfile(model_path):
            raise RuntimeError(f"RVC training did not produce a model file: {model_path}")

        rvc_model = {
            "model_path": model_path,
            "index_path": index_path,
            "model_name": os.path.basename(model_path),
            "index_name": os.path.basename(index_path) if index_path else None,
            "type": "rvc_model",
        }

        summary = (
            f"RVC training complete: {resolved_name} | sample rate {sample_rate} | "
            f"model {model_path}"
        )
        if index_requested and not index_path:
            summary += " | index not built"

        artifacts = {
            "type": "training_artifacts",
            "engine_type": "rvc",
            "training_mode": "voice_model",
            "artifact_type": "voice_model",
            "model_path": model_path,
            "index_path": index_path,
            "index_requested": index_requested,
            "index_built": bool(index_path),
            "index_warning": index_build_warning or None,
            "log_dir": job_dir,
            "config_path": resolved_config_path,
            "model_name": resolved_name,
            "rvc_model": rvc_model,
            "summary": summary,
        }
        finalize_training_job(
            node_id,
            status="completed",
            phase="complete",
            artifacts=artifacts,
            model_path=model_path,
            index_path=index_path,
            index_requested=index_requested,
            index_built=bool(index_path),
            warning=index_build_warning or None,
        )
        _write_terminal_progress(
            progress_file,
            status="completed",
            phase="complete",
            artifacts=artifacts,
            model_path=model_path,
            index_path=index_path,
            index_requested=index_requested,
            index_built=bool(index_path),
            warning=index_build_warning or None,
        )
        return artifacts
    except InterruptedError as error:
        finalize_training_job(
            node_id,
            status="cancelled",
            phase="cancelled",
            error=str(error),
            model_path=model_path,
        )
        _write_terminal_progress(
            progress_file,
            status="cancelled",
            phase="cancelled",
            error=str(error),
            model_path=model_path,
        )
        raise
    except Exception as error:
        finalize_training_job(
            node_id,
            status="error",
            phase="error",
            error=str(error),
            model_path=model_path,
        )
        _write_terminal_progress(
            progress_file,
            status="error",
            phase="error",
            error=str(error),
            model_path=model_path,
        )
        raise
