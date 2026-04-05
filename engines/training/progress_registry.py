"""
In-process registry for live training progress.

The training backends can register a progress file for a node, and the web UI
can query this registry through a small API route to render live dashboards.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import threading
import time
from typing import Any, Dict, Optional

import folder_paths


_LOCK = threading.Lock()
_TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}
_ACTIVE_STATUSES = {"starting", "running"}
_TERMINAL_STATUSES = {"completed", "error", "cancelled"}
_STALE_ACTIVE_SECONDS = 180.0


def _normalize_node_id(node_id: Any) -> str:
    return str(node_id or "").strip()


def register_training_job(
    node_id: Any,
    *,
    engine_type: str,
    progress_file: str,
    job_dir: str = "",
    model_name: str = "",
    sample_rate: str = "",
    total_epochs: int = 0,
) -> None:
    normalized = _normalize_node_id(node_id)
    if not normalized:
        return

    with _LOCK:
        _TRAINING_JOBS[normalized] = {
            "node_id": normalized,
            "engine_type": engine_type,
            "progress_file": progress_file,
            "job_dir": job_dir,
            "model_name": model_name,
            "sample_rate": sample_rate,
            "total_epochs": int(total_epochs or 0),
            "status": "starting",
            "phase": "initializing",
            "started_at": time.time(),
            "updated_at": time.time(),
        }


def update_training_job(node_id: Any, **updates: Any) -> None:
    normalized = _normalize_node_id(node_id)
    if not normalized:
        return

    with _LOCK:
        job = _TRAINING_JOBS.setdefault(normalized, {"node_id": normalized})
        job.update(updates)
        job["updated_at"] = time.time()


def finalize_training_job(node_id: Any, *, status: str, **updates: Any) -> None:
    normalized = _normalize_node_id(node_id)
    if not normalized:
        return

    with _LOCK:
        job = _TRAINING_JOBS.setdefault(normalized, {"node_id": normalized})
        job.update(updates)
        job["status"] = status
        job["updated_at"] = time.time()
        job["completed_at"] = time.time()


def _read_progress_file(progress_file: str) -> Dict[str, Any]:
    if not progress_file or not os.path.isfile(progress_file):
        return {}

    try:
        with open(progress_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _normalize_progress_entry(progress_file: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "node_id": _normalize_node_id(data.get("node_id")) or _fallback_key_for_progress(progress_file, data),
        "progress_file": progress_file,
        "job_dir": os.path.dirname(progress_file),
        **data,
    }


def _discover_progress_files() -> list[str]:
    training_root = os.path.join(folder_paths.get_output_directory(), "tts_audio_suite_training")
    pattern = os.path.join(training_root, "*", "jobs", "*", "progress.json")
    return sorted(glob.glob(pattern))


def _fallback_key_for_progress(progress_file: str, data: Dict[str, Any]) -> str:
    node_id = _normalize_node_id(data.get("node_id"))
    if node_id:
        return node_id
    return f"job:{os.path.basename(os.path.dirname(progress_file))}"


def _parse_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value or "").strip()
    if not text:
        return 0.0

    try:
        return float(time.mktime(time.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")))
    except Exception:
        pass

    try:
        return float(time.mktime(time.strptime(text[:19], "%Y-%m-%d %H:%M:%S")))
    except Exception:
        pass

    digits_only = "".join(ch for ch in text if ch.isdigit())
    if digits_only:
        try:
            return float(digits_only[:17])
        except Exception:
            return 0.0
    return 0.0


def _is_stale_active_entry(entry: Dict[str, Any], *, now: Optional[float] = None) -> bool:
    status = str(entry.get("status") or "").strip().lower()
    if status not in _ACTIVE_STATUSES:
        return False

    timestamp = _parse_timestamp(entry.get("updated_at"))
    if timestamp <= 0:
        return False

    reference_time = float(now if now is not None else time.time())
    return (reference_time - timestamp) > _STALE_ACTIVE_SECONDS


def _entry_priority(entry: Dict[str, Any]) -> tuple[float, int]:
    status = str(entry.get("status") or "").strip().lower()
    status_weight = 1 if status in _TERMINAL_STATUSES else 0
    return (_parse_timestamp(entry.get("updated_at")), status_weight)


def _choose_preferred_entry(current: Optional[Dict[str, Any]], candidate: Dict[str, Any]) -> Dict[str, Any]:
    if current is None:
        return candidate
    if _entry_priority(candidate) >= _entry_priority(current):
        return candidate
    return current


def _load_file_backed_jobs(node_id: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    normalized = _normalize_node_id(node_id) if node_id is not None else None
    discovered: Dict[str, Dict[str, Any]] = {}
    now = time.time()

    for progress_file in _discover_progress_files():
        data = _read_progress_file(progress_file)
        if not data:
            continue

        candidate = _normalize_progress_entry(progress_file, data)
        if _is_stale_active_entry(candidate, now=now):
            continue

        current_node_id = _normalize_node_id(data.get("node_id"))
        if normalized is not None and current_node_id and current_node_id != normalized:
            continue

        key = _fallback_key_for_progress(progress_file, data)
        if normalized is not None and key != normalized and current_node_id != normalized:
            continue

        candidate["node_id"] = current_node_id or key
        discovered[key] = _choose_preferred_entry(discovered.get(key), candidate)

    return discovered


def _sort_timestamp(value: Any) -> float:
    return _parse_timestamp(value)


def get_training_progress_snapshot(node_id: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    normalized = _normalize_node_id(node_id) if node_id is not None else None

    with _LOCK:
        if normalized is not None:
            jobs = {}
            if normalized in _TRAINING_JOBS:
                jobs[normalized] = copy.deepcopy(_TRAINING_JOBS[normalized])
        else:
            jobs = {key: copy.deepcopy(value) for key, value in _TRAINING_JOBS.items()}

    snapshot: Dict[str, Dict[str, Any]] = {}
    for current_node_id, entry in jobs.items():
        file_data = _read_progress_file(entry.get("progress_file", ""))
        merged = dict(entry)
        if file_data:
            merged.update(file_data)
        if entry.get("status") in {"completed", "error", "cancelled"}:
            merged["status"] = entry["status"]
            if entry.get("phase"):
                merged["phase"] = entry["phase"]
            if entry.get("error"):
                merged["error"] = entry["error"]
            if entry.get("artifacts"):
                merged["artifacts"] = entry["artifacts"]
        snapshot[current_node_id] = merged

    fallback_jobs = _load_file_backed_jobs(node_id=node_id)
    for current_node_id, entry in fallback_jobs.items():
        existing = snapshot.get(current_node_id)
        if existing is None:
            snapshot[current_node_id] = entry
            continue

        existing_ts = _sort_timestamp(existing.get("updated_at"))
        fallback_ts = _sort_timestamp(entry.get("updated_at"))
        if fallback_ts >= existing_ts:
            merged = dict(existing)
            merged.update(entry)
            snapshot[current_node_id] = merged

    return snapshot


__all__ = [
    "finalize_training_job",
    "get_training_progress_snapshot",
    "register_training_job",
    "update_training_job",
]
