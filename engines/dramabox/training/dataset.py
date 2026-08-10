"""Dataset normalization for the official DramaBox IC-LoRA trainer.

The upstream preprocessor accepts JSONL and TSV, but the upstream training
loop builds its speaker map from ``~``-delimited index rows.  This module keeps
that conversion in the suite so a manifest that is valid for preprocessing is
also valid for training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import folder_paths


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
PREPROCESSED_SAMPLE_PATTERN = re.compile(r"sample_(\d+)\.pt$")


def slugify(value: Any) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in str(value or "").strip()
    )
    safe = safe.strip("_")
    return safe or "dramabox_lora"


def get_dramabox_training_root() -> str:
    root = os.path.join(
        folder_paths.get_output_directory(), "tts_audio_suite_training", "dramabox"
    )
    os.makedirs(root, exist_ok=True)
    return root


def _resolve_source_path(value: str) -> Path:
    raw = os.path.expanduser(str(value or "").strip())
    if not raw:
        raise ValueError("dataset_source is required")

    candidates = [Path(raw)]
    input_root = Path(folder_paths.get_input_directory())
    candidates.extend((input_root / raw, input_root / "datasets" / raw))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"DramaBox dataset source not found: {value}")


def _resolve_audio_path(raw_path: Any, *, source_path: Path, audio_dir: str) -> Path:
    value = os.path.expanduser(str(raw_path or "").strip())
    if not value:
        raise ValueError("Dataset row is missing audio_filepath/audio_path")

    candidates: List[Path] = []
    if os.path.isabs(value):
        candidates.append(Path(value))
    else:
        if audio_dir:
            candidates.append(Path(os.path.expanduser(audio_dir)) / value)
        candidates.append(source_path.parent / value)
        candidates.append(Path(value))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"DramaBox audio file not found: {raw_path}")


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\x00", "")).strip()


def _speaker_value(row: Dict[str, Any], default: str = "speaker_1") -> str:
    value = (
        row.get("speaker")
        or row.get("speaker_id")
        or row.get("voice")
        or row.get("character")
        or default
    )
    return _clean_text(value).replace("~", "_") or default


def _language_value(row: Dict[str, Any]) -> str:
    return _clean_text(row.get("language") or row.get("lang") or "en").replace("~", "_") or "en"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed > 0 else float(default)


def _probe_audio(path: Path) -> Tuple[int, int, float]:
    """Return sample rate, frame count, and duration without loading audio."""
    try:
        import torchaudio

        info = torchaudio.info(str(path))
        sample_rate = int(getattr(info, "sample_rate", 0) or 0)
        frames = int(getattr(info, "num_frames", 0) or 0)
        if sample_rate > 0 and frames > 0:
            return sample_rate, frames, frames / sample_rate
    except Exception:
        pass

    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as handle:
            sample_rate = int(handle.getframerate())
            frames = int(handle.getnframes())
        if sample_rate > 0 and frames > 0:
            return sample_rate, frames, frames / sample_rate

    raise RuntimeError(
        f"Could not inspect audio duration for '{path}'. Add a positive duration "
        "field to the manifest or install a Torchaudio-compatible decoder."
    )


def _parse_manifest(source_path: Path, audio_dir: str) -> Iterable[Dict[str, Any]]:
    text = source_path.read_text(encoding="utf-8-sig")
    stripped = text.lstrip()
    if stripped.startswith("["):
        raw_rows = json.loads(text)
    else:
        raw_rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        yield {
            "audio": _resolve_audio_path(
                row.get("audio_filepath", row.get("audio_path", row.get("audio"))),
                source_path=source_path,
                audio_dir=audio_dir,
            ),
            "text": _clean_text(row.get("text", row.get("transcript", ""))),
            "duration": _coerce_float(row.get("duration")),
            "sample_rate": int(_coerce_float(row.get("sample_rate"))),
            "samples": int(_coerce_float(row.get("samples", row.get("num_frames")))),
            "speaker": _speaker_value(row),
            "language": _language_value(row),
        }


def _parse_tsv(source_path: Path, audio_dir: str) -> Iterable[Dict[str, Any]]:
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if len(row) < 2:
                continue
            yield {
                "audio": _resolve_audio_path(row[0], source_path=source_path, audio_dir=audio_dir),
                "text": _clean_text(row[1]),
                "duration": _coerce_float(row[2]) if len(row) > 2 else 0.0,
                "sample_rate": 0,
                "samples": 0,
                "speaker": _clean_text(row[3]).replace("~", "_") if len(row) > 3 else "speaker_1",
                "language": _clean_text(row[4]).replace("~", "_") if len(row) > 4 else "en",
                "row_number": row_number,
            }


def _parse_gemini(source_path: Path, audio_dir: str) -> Iterable[Dict[str, Any]]:
    for line in source_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split("~")
        if len(parts) < 8:
            continue
        file_id, speaker, language = parts[:3]
        sample_rate = int(_coerce_float(parts[3], 24000))
        samples = int(_coerce_float(parts[4]))
        duration = _coerce_float(parts[5])
        text = _clean_text(parts[-1])
        yield {
            "audio": _resolve_audio_path(file_id, source_path=source_path, audio_dir=audio_dir),
            "text": text,
            "duration": duration,
            "sample_rate": sample_rate,
            "samples": samples,
            "speaker": _clean_text(speaker).replace("~", "_") or "speaker_1",
            "language": _clean_text(language).replace("~", "_") or "en",
        }


def _parse_libriheavy(source_path: Path, audio_dir: str) -> Iterable[Dict[str, Any]]:
    for line in source_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split("~")
        if len(parts) < 7:
            continue
        file_id, speaker, language = parts[:3]
        # Format: id~speaker~lang~samples~duration_ms~phonemes~text.
        sample_rate = 24000
        samples = int(_coerce_float(parts[3]))
        duration = _coerce_float(parts[4]) / 1000.0 if len(parts) >= 5 else 0.0
        yield {
            "audio": _resolve_audio_path(file_id, source_path=source_path, audio_dir=audio_dir),
            "text": _clean_text(parts[-1]),
            "duration": duration,
            "sample_rate": sample_rate,
            "samples": samples,
            "speaker": _clean_text(speaker).replace("~", "_") or "speaker_1",
            "language": _clean_text(language).replace("~", "_") or "en",
        }


def _raw_rows(source_path: Path, dataset_type: str, audio_dir: str) -> Iterable[Dict[str, Any]]:
    parsers = {
        "manifest": _parse_manifest,
        "tsv": _parse_tsv,
        "gemini_synthetic": _parse_gemini,
        "libriheavy": _parse_libriheavy,
    }
    try:
        parser = parsers[str(dataset_type)]
    except KeyError as exc:
        raise ValueError(f"Unsupported DramaBox dataset type: {dataset_type}") from exc
    return parser(source_path, audio_dir)


def _fingerprint(source_path: Path, *, dataset_type: str, audio_dir: str, min_duration: float, max_duration: float) -> str:
    stat = source_path.stat()
    raw = f"{source_path}|{stat.st_size}|{stat.st_mtime_ns}|{dataset_type}|{audio_dir}|{min_duration}|{max_duration}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _normalize_rows(
    source_path: Path,
    *,
    dataset_type: str,
    audio_dir: str,
    min_duration: float,
    max_duration: float,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row_index, row in enumerate(_raw_rows(source_path, dataset_type, audio_dir)):
        text = _clean_text(row.get("text"))
        if not text:
            continue

        audio = Path(row["audio"]).resolve()
        sample_rate = int(row.get("sample_rate") or 0)
        samples = int(row.get("samples") or 0)
        duration = _coerce_float(row.get("duration"))
        if not sample_rate or not samples or not duration:
            try:
                probed_rate, probed_samples, probed_duration = _probe_audio(audio)
                sample_rate = sample_rate or probed_rate
                samples = samples or probed_samples
                duration = duration or probed_duration
            except RuntimeError:
                if duration <= 0:
                    raise
                sample_rate = sample_rate or 24000
                samples = samples or max(1, round(duration * sample_rate))

        if duration < float(min_duration) or duration > float(max_duration):
            continue
        records.append(
            {
                "id": f"sample_{row_index:06d}",
                "audio": str(audio),
                "text": text,
                "duration": float(duration),
                "sample_rate": int(sample_rate),
                "samples": int(samples),
                "speaker": _speaker_value(row),
                "language": _language_value(row),
            }
        )

    if not records:
        raise ValueError(
            "DramaBox dataset preparation produced no usable rows. Check the audio paths, "
            "transcripts, and the min/max duration filters."
        )

    speaker_counts: Dict[str, int] = {}
    for record in records:
        speaker_counts[record["speaker"]] = speaker_counts.get(record["speaker"], 0) + 1
    unusable = sorted(name for name, count in speaker_counts.items() if count < 2)
    if unusable:
        raise ValueError(
            "DramaBox LoRA training needs at least two clips per speaker so the official "
            f"trainer can choose a reference clip. Speakers with fewer than two clips: {', '.join(unusable)}."
        )
    return records


def _write_index(records: List[Dict[str, Any]], index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as handle:
        for record in records:
            text = str(record["text"]).replace("\r", " ").replace("\n", " ")
            handle.write(
                "~".join(
                    (
                        str(Path(record["audio"]).resolve()),
                        str(record["speaker"]),
                        str(record["language"]),
                        str(int(record["sample_rate"])),
                        str(int(record["samples"])),
                        f"{float(record['duration']):.6f}",
                        "_",
                        text,
                    )
                )
                + "\n"
            )


def _preprocessed_indices(directory: Path) -> set[int]:
    indices: set[int] = set()
    if not directory.is_dir():
        return indices
    for path in directory.glob("sample_*.pt"):
        match = PREPROCESSED_SAMPLE_PATTERN.fullmatch(path.name)
        if match:
            indices.add(int(match.group(1)))
    return indices


def validate_preprocessed_dataset(
    records: List[Dict[str, Any]],
    preprocessed_dir: str | Path,
    *,
    raise_on_missing: bool = False,
) -> bool:
    """Require matching text conditions and audio latents for every index row."""
    root = Path(preprocessed_dir)
    expected = set(range(len(records)))
    available = _preprocessed_indices(root / "conditions") & _preprocessed_indices(
        root / "audio_latents"
    )
    missing = sorted(expected - available)
    complete = bool(expected) and not missing
    if raise_on_missing and not complete:
        preview = ", ".join(str(index) for index in missing[:10]) or "all"
        suffix = "..." if len(missing) > 10 else ""
        raise RuntimeError(
            "DramaBox preprocessing did not produce matching condition/audio-latent "
            f"files for {len(missing) or len(expected)} sample(s) (indices: {preview}{suffix}). "
            "Fix the reported source-audio errors and run Dataset Prep again."
        )
    return complete


def prepare_dramabox_dataset(
    shared_settings: Dict[str, Any],
    *,
    dataset_source: str,
    model_name: str,
    dataset_type: str = "manifest",
    audio_dir: str = "",
    min_duration: float = 2.0,
    max_duration: float = 20.0,
    reuse_existing: bool = True,
    preprocess_now: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    source_path = _resolve_source_path(dataset_source)
    fingerprint = _fingerprint(
        source_path,
        dataset_type=dataset_type,
        audio_dir=audio_dir,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    safe_name = slugify(model_name)
    dataset_root = Path(get_dramabox_training_root()) / "datasets" / f"{safe_name}_{fingerprint}"
    index_path = dataset_root / "speaker_index.txt"
    metadata_path = dataset_root / "dataset.json"
    preprocessed_dir = dataset_root / "preprocessed"

    if reuse_existing and metadata_path.is_file() and index_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            records = metadata.get("records") or []
        except Exception:
            records = []
    else:
        records = []

    if not records:
        records = _normalize_rows(
            source_path,
            dataset_type=dataset_type,
            audio_dir=audio_dir,
            min_duration=float(min_duration),
            max_duration=float(max_duration),
        )
        dataset_root.mkdir(parents=True, exist_ok=True)
        _write_index(records, index_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "type": "dramabox_dataset",
                    "source_path": str(source_path),
                    "dataset_type": dataset_type,
                    "audio_dir": audio_dir,
                    "min_duration": float(min_duration),
                    "max_duration": float(max_duration),
                    "records": records,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # Rewrite cached indexes as well so datasets prepared by older suite
    # builds migrate from synthetic sample ids to resolvable audio paths.
    _write_index(records, index_path)

    dataset: Dict[str, Any] = {
        "type": "training_dataset",
        "engine_type": "dramabox",
        "training_mode": "audio_lora",
        "model_name": model_name,
        "dataset_type": dataset_type,
        "source_path": str(source_path),
        "index_path": str(index_path),
        "speaker_index": str(index_path),
        "data_dir": [str(preprocessed_dir)],
        "preprocessed_dir": str(preprocessed_dir),
        "min_duration": float(min_duration),
        "max_duration": float(max_duration),
        "records": records,
        "train_records": len(records),
        "speakers": sorted({str(record["speaker"]) for record in records}),
        "preprocessed": validate_preprocessed_dataset(records, preprocessed_dir),
        "dry_run": bool(dry_run),
        "shared_settings": dict(shared_settings or {}),
    }

    if preprocess_now and not dry_run and not dataset["preprocessed"]:
        from .trainer import run_dramabox_preprocess

        run_dramabox_preprocess(dataset, shared_settings, batch_size=8)
        dataset["preprocessed"] = True

    return dataset


__all__ = [
    "get_dramabox_training_root",
    "prepare_dramabox_dataset",
    "slugify",
    "validate_preprocessed_dataset",
]
