"""Resolve Suite integration capabilities for pinned audio.cpp families."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .catalog import AUDIO_CPP_RELEASE_TAG, PackageRecord, load_catalog


class CapabilityError(ValueError):
    pass


def get_capability_path() -> Path:
    return Path(__file__).resolve().with_name("integration_capabilities.yaml")


@lru_cache(maxsize=1)
def _load_overlay() -> Dict[str, Any]:
    path = get_capability_path()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise CapabilityError(f"Cannot read audio.cpp capability overlay {path}: {exc}") from exc
    if raw.get("schema_version") != 1 or raw.get("release") != AUDIO_CPP_RELEASE_TAG:
        raise CapabilityError("audio.cpp capability overlay release/schema does not match the catalog")
    return dict(raw)


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    value = deepcopy(dict(base))
    for key, item in override.items():
        if isinstance(item, Mapping) and isinstance(value.get(key), Mapping):
            value[key] = _merge(value[key], item)
        else:
            value[key] = deepcopy(item)
    return value


@lru_cache(maxsize=1)
def load_capabilities() -> Dict[str, Dict[str, Any]]:
    raw = _load_overlay()
    catalog = load_catalog()
    families = raw.get("families") or {}
    if set(families) != set(catalog.families):
        missing = sorted(set(catalog.families) - set(families))
        extra = sorted(set(families) - set(catalog.families))
        raise CapabilityError(f"audio.cpp capability families mismatch; missing={missing}, extra={extra}")

    resolved: Dict[str, Dict[str, Any]] = {}
    defaults = raw.get("defaults") or {}
    for family_id, override in families.items():
        family = catalog.family(family_id)
        item = _merge(defaults, override or {})
        item.update(
            id=family.id,
            display_name=family.display_name,
            description=family.description,
            languages=list(family.languages),
            upstream_tasks=list(family.runtime_tasks),
            packages=[package.id for package in family.packages],
            recommended_package_id=family.recommended_package_id,
        )
        asr_features = item.get("asr_features") or {}
        if not isinstance(asr_features, Mapping):
            raise CapabilityError(
                f"audio.cpp {family_id} asr_features must be an object"
            )
        diarization = str(asr_features.get("diarization", "none"))
        timing = str(asr_features.get("timing", "none"))
        if diarization not in {"none", "native"}:
            raise CapabilityError(
                f"audio.cpp {family_id} has invalid ASR diarization capability: {diarization}"
            )
        if timing not in {
            "none",
            "native_word",
            "native_segment",
            "optional_forced_aligner",
        }:
            raise CapabilityError(
                f"audio.cpp {family_id} has invalid ASR timing capability: {timing}"
            )
        resolved[family_id] = item
    return resolved


def get_capability(family: str) -> Dict[str, Any]:
    try:
        return deepcopy(load_capabilities()[str(family)])
    except KeyError as exc:
        raise CapabilityError(f"Unknown audio.cpp capability family: {family!r}") from exc


def public_capabilities() -> Dict[str, Any]:
    catalog = load_catalog()
    raw_sizes = _load_overlay().get("package_sizes") or {}
    if set(raw_sizes) != set(catalog.packages):
        missing = sorted(set(catalog.packages) - set(raw_sizes))
        extra = sorted(set(raw_sizes) - set(catalog.packages))
        raise CapabilityError(f"audio.cpp package sizes mismatch; missing={missing}, extra={extra}")
    packages = {}
    for package_id, package in catalog.packages.items():
        size = raw_sizes[package_id]
        if not isinstance(size, int) or size <= 0:
            raise CapabilityError(f"Invalid estimated size for audio.cpp package {package_id!r}")
        dependencies = get_package_dependencies(package_id)
        dependency_bytes = sum(
            int(item["estimated_download_bytes"]) for item in dependencies
        )
        packages[package_id] = {
            "id": package.id,
            "family": package.family,
            "display_name": package.display_name,
            "format": package.format,
            "precision": package.precision,
            "estimated_download_bytes": size + dependency_bytes,
            "primary_download_bytes": size,
            "dependencies": [item["package"].id for item in dependencies],
        }
    return {
        "schema_version": 1,
        "release": AUDIO_CPP_RELEASE_TAG,
        "sizes_checked_at": "2026-08-13",
        "families": load_capabilities(),
        "packages": packages,
    }


def get_package_dependencies(package_id: str) -> list[Dict[str, Any]]:
    entries = (_load_overlay().get("package_dependencies") or {}).get(package_id, [])
    dependencies = []
    for entry in entries:
        package = PackageRecord(
            family=str(entry["family"]),
            id=str(entry["id"]),
            display_name=str(entry["display_name"]),
            target_directory=str(entry["target_directory"]),
            format=str(entry["format"]),
            precision=str(entry["precision"]),
            files=tuple(str(value) for value in entry["files"]),
            strip_prefix=str(entry.get("strip_prefix", "")),
            download={
                "kind": "huggingface_snapshot",
                "repo": str(entry["repo"]),
                "revision": str(entry.get("revision", "main")),
                "gated": False,
            },
        )
        # Validate local mappings before the downloader touches disk.
        package.local_files
        dependencies.append(
            {
                "package": package,
                "session_option": str(entry["session_option"]),
                "estimated_download_bytes": int(entry["estimated_download_bytes"]),
            }
        )
    return dependencies


def validate_voice_reference(family: str, voice_ref: Any, character: str = "narrator") -> None:
    """Enforce requirements that the frontend panel merely explains."""
    from utils.voice.reference import effective_voice_audio

    capability = get_capability(family)
    audio_requirement = capability["reference_audio"]
    has_audio = isinstance(voice_ref, Mapping) and effective_voice_audio(voice_ref) is not None
    if audio_requirement in {"required", "required_per_speaker"} and not has_audio:
        raise ValueError(f"audio.cpp {family} requires reference audio for '{character}'")
    transcript = ""
    if isinstance(voice_ref, Mapping):
        transcript = str(
            voice_ref.get("reference_text") or voice_ref.get("prompt_text") or voice_ref.get("text") or ""
        ).strip()
    if capability["reference_transcript"] == "required" and not transcript:
        raise ValueError(
            f"audio.cpp {family} requires the transcript matching '{character}' reference audio"
        )
