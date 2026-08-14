"""Pinned audio.cpp release-0.5.1 Suite-compatible model catalog.

The bundled JSON files are exact copies of the selected upstream tag's model
specifications.  The executable's compiled task IDs are kept separately because
several release specs expose broader or differently-spelled task metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple


AUDIO_CPP_RELEASE_VERSION = "0.5.1"
AUDIO_CPP_RELEASE_TAG = "release-0.5.1"
AUDIO_CPP_RELEASE_COMMIT = "238ab6a9e321c17de8e120559f57efeedaeb1345"

MODEL_SPEC_FILENAMES: Tuple[str, ...] = (
    "citrinet_asr.json",
    "chatterbox.json",
    "confucius4_tts.json",
    "dramabox.json",
    "fish_audio.json",
    "fun_asr_nano.json",
    "glm_tts.json",
    "higgs_audio_tts.json",
    "higgs_audio_stt.json",
    "hviske_asr.json",
    "index_tts2.json",
    "inflect_v2.json",
    "irodori_tts.json",
    "kroko_asr.json",
    "miotts.json",
    "moss_tts_local.json",
    "moss_tts_nano.json",
    "nemotron_asr.json",
    "omnivoice.json",
    "outetts.json",
    "parakeet_tdt.json",
    "pocket_tts.json",
    "qwen3_tts.json",
    "qwen3_asr.json",
    "seed_vc.json",
    "supertonic.json",
    "vevo2.json",
    "vibevoice.json",
    "vibevoice_asr.json",
    "vietneu_tts.json",
    "voxcpm2.json",
    "voxtral_realtime.json",
)

# These are the task IDs actually compiled into release-0.5.1.  Do not derive
# them from the broader human-facing ``tasks`` arrays in the JSON specs.
COMPILED_TASKS: Mapping[str, Tuple[str, ...]] = {
    "citrinet_asr": ("asr",),
    "chatterbox": ("clon", "vc"),
    "confucius4_tts": ("clon",),
    "dramabox": ("tts", "clon"),
    "fish_audio": ("tts",),
    "fun_asr_nano": ("asr",),
    "glm_tts": ("tts", "clon"),
    "higgs_audio_tts": ("tts",),
    "higgs_audio_stt": ("asr",),
    "hviske_asr": ("asr",),
    "index_tts2": ("tts", "clon"),
    "inflect_v2": ("tts",),
    "irodori_tts": ("tts", "clon", "vdes"),
    "kroko_asr": ("asr",),
    "miotts": ("tts",),
    "moss_tts_local": ("tts", "clon"),
    "moss_tts_nano": ("tts", "clon"),
    "nemotron_asr": ("asr",),
    "omnivoice": ("tts",),
    "outetts": ("tts", "clon"),
    "parakeet_tdt": ("asr",),
    "pocket_tts": ("tts",),
    "qwen3_tts": ("tts", "vdes"),
    "qwen3_asr": ("asr",),
    "seed_vc": ("vc", "svc"),
    "supertonic": ("tts",),
    "vevo2": ("tts", "vc", "s2s", "svc"),
    "vibevoice": ("tts",),
    "vibevoice_asr": ("asr",),
    "vietneu_tts": ("tts", "vdes"),
    "voxcpm2": ("tts",),
    "voxtral_realtime": ("asr",),
}

_TASK_ALIASES = {
    "clone": "clon",
    "cloning": "clon",
    "voice_clone": "clon",
    "voice_cloning": "clon",
    "voice_design": "vdes",
    "design": "vdes",
    "voice_conversion": "vc",
    "speech_to_speech": "s2s",
    "singing_voice_conversion": "svc",
}


class CatalogError(ValueError):
    """Raised when a pinned model specification is missing or inconsistent."""


def _safe_package_relative_path(value: str, label: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or (path.parts and ":" in path.parts[0])
    ):
        raise CatalogError(f"Unsafe {label}: {value!r}")
    return path


@dataclass(frozen=True)
class PackageRecord:
    family: str
    id: str
    display_name: str
    target_directory: str
    format: str
    precision: str
    files: Tuple[str, ...]
    strip_prefix: str
    download: Mapping[str, Any]
    default: bool = False

    def local_relative_path(self, remote_path: str) -> Path:
        """Map one remote package path to its installed relative path."""

        remote = _safe_package_relative_path(remote_path, "remote file path")
        prefix_text = self.strip_prefix.replace("\\", "/").rstrip("/")
        if prefix_text in ("", "."):
            local = remote
        else:
            prefix = _safe_package_relative_path(prefix_text, "strip_prefix")
            if remote == prefix or remote.parts[: len(prefix.parts)] != prefix.parts:
                raise CatalogError(
                    f"Package {self.id!r} file {remote_path!r} is outside "
                    f"strip_prefix {self.strip_prefix!r}"
                )
            local = PurePosixPath(*remote.parts[len(prefix.parts) :])
        if not local.parts:
            raise CatalogError(f"Package {self.id!r} maps {remote_path!r} to an empty path")
        return Path(*local.parts)

    @property
    def local_files(self) -> Tuple[Path, ...]:
        return tuple(self.local_relative_path(path) for path in self.files)

    @property
    def repo(self) -> str:
        return str(self.download.get("repo", ""))

    @property
    def revision(self) -> str:
        return str(self.download.get("revision", "main"))

    @property
    def gated(self) -> bool:
        return bool(self.download.get("gated", False))


@dataclass(frozen=True)
class FamilyRecord:
    id: str
    display_name: str
    description: str
    category: str
    status: str
    runtime_tasks: Tuple[str, ...]
    languages: Tuple[str, ...]
    options: Mapping[str, Any]
    packages: Tuple[PackageRecord, ...]
    recommended_package_id: str
    spec_filename: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class AudioCppCatalog:
    families: Mapping[str, FamilyRecord]
    packages: Mapping[str, PackageRecord]
    specs_dir: Path
    release_version: str = AUDIO_CPP_RELEASE_VERSION

    def iter_families(self) -> Iterator[FamilyRecord]:
        return iter(self.families.values())

    def iter_packages(self, family: Optional[str] = None) -> Iterator[PackageRecord]:
        if family is None:
            return iter(self.packages.values())
        return iter(self.family(family).packages)

    def family(self, family_id: str) -> FamilyRecord:
        try:
            return self.families[family_id]
        except KeyError as exc:
            raise CatalogError(f"Unknown audio.cpp family: {family_id!r}") from exc

    def package(self, package_id: str) -> PackageRecord:
        try:
            return self.packages[package_id]
        except KeyError as exc:
            raise CatalogError(f"Unknown audio.cpp package: {package_id!r}") from exc


def get_model_specs_dir() -> Path:
    """Return the exact release-0.5.1 spec directory shipped with the node."""

    return Path(__file__).resolve().parent / "model_specs"


def _merged_download(spec: Mapping[str, Any], package: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(spec.get("package_defaults", {}).get("download", {}))
    merged.update(package.get("download", {}))
    return merged


def _load_catalog(specs_dir: Path) -> AudioCppCatalog:
    families: Dict[str, FamilyRecord] = {}
    packages: Dict[str, PackageRecord] = {}

    for filename in MODEL_SPEC_FILENAMES:
        path = specs_dir / filename
        if not path.is_file():
            raise CatalogError(f"Missing pinned audio.cpp model spec: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CatalogError(f"Cannot read audio.cpp model spec {path}: {exc}") from exc

        family_id = str(raw.get("family", ""))
        if family_id not in COMPILED_TASKS:
            raise CatalogError(f"Spec {filename} has unsupported release family {family_id!r}")
        if family_id in families:
            raise CatalogError(f"Duplicate audio.cpp family {family_id!r}")

        family_packages = []
        for package_data in raw.get("packages", []):
            package_id = str(package_data.get("id", ""))
            if not package_id or package_id in packages:
                raise CatalogError(f"Missing or duplicate package ID {package_id!r} in {filename}")
            package = PackageRecord(
                family=family_id,
                id=package_id,
                display_name=str(package_data.get("display_name", package_id)),
                target_directory=str(package_data.get("target_directory", "")),
                format=str(package_data.get("format", "")),
                precision=str(package_data.get("precision", "")),
                files=tuple(str(item) for item in package_data.get("files", [])),
                strip_prefix=str(package_data.get("strip_prefix", "")),
                download=_merged_download(raw, package_data),
                default=bool(package_data.get("default", False)),
            )
            _safe_package_relative_path(package.target_directory, "target_directory")
            if not package.files:
                raise CatalogError(f"Package {package_id!r} has no downloadable files")
            # Validate prefix mappings when loading, before any filesystem mutation.
            package.local_files
            if package.download.get("kind") != "huggingface_snapshot" or not package.repo:
                raise CatalogError(f"Package {package_id!r} has no supported download source")
            packages[package_id] = package
            family_packages.append(package)

        recommendation = str(raw.get("ui", {}).get("recommended_package", ""))
        if not recommendation:
            recommendation = next((p.id for p in family_packages if p.default), "")
        if recommendation not in {package.id for package in family_packages}:
            raise CatalogError(
                f"Family {family_id!r} recommends unknown package {recommendation!r}"
            )

        families[family_id] = FamilyRecord(
            id=family_id,
            display_name=str(raw.get("display_name", family_id)),
            description=str(raw.get("description", "")),
            category=str(raw.get("category", "")),
            status=str(raw.get("status", "")),
            runtime_tasks=COMPILED_TASKS[family_id],
            languages=tuple(str(item) for item in raw.get("languages", [])),
            options=dict(raw.get("options", {})),
            packages=tuple(family_packages),
            recommended_package_id=recommendation,
            spec_filename=filename,
            raw=raw,
        )

    if set(families) != set(COMPILED_TASKS):
        missing = sorted(set(COMPILED_TASKS) - set(families))
        raise CatalogError(f"Pinned audio.cpp catalog is incomplete; missing {missing}")
    if len(packages) != 96:
        raise CatalogError(f"Expected 96 Suite-compatible release-0.5.1 packages, found {len(packages)}")
    return AudioCppCatalog(families=families, packages=packages, specs_dir=specs_dir)


@lru_cache(maxsize=1)
def _load_bundled_catalog() -> AudioCppCatalog:
    return _load_catalog(get_model_specs_dir())


def load_catalog(specs_dir: Optional[Path] = None) -> AudioCppCatalog:
    """Load the bundled catalog, or validate an equivalent override directory."""

    if specs_dir is None:
        return _load_bundled_catalog()
    return _load_catalog(Path(specs_dir).expanduser().resolve())


def family_choices() -> list[str]:
    return list(load_catalog().families)


def package_choices(family: Optional[str] = None) -> list[str]:
    catalog = load_catalog()
    if family is None:
        return list(catalog.packages)
    return [package.id for package in catalog.family(family).packages]


def get_family(family: str) -> FamilyRecord:
    return load_catalog().family(family)


def get_package(package_id: str) -> PackageRecord:
    return load_catalog().package(package_id)


def recommended_package(family: str) -> str:
    return get_family(family).recommended_package_id


def resolve_task(family: str, package_id: Optional[str], requested: str = "auto") -> str:
    """Resolve a UI task name to a release-0.5.1 compiled task ID."""

    catalog = load_catalog()
    family_record = catalog.family(family)
    package = catalog.package(package_id) if package_id is not None else None
    if package is not None and package.family != family:
        raise CatalogError(f"Package {package_id!r} does not belong to family {family!r}")
    normalized = str(requested or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "auto":
        if package is not None and "vdes" in family_record.runtime_tasks:
            package_label = f"{package.id} {package.display_name}".lower().replace("_", "")
            if "voicedesign" in package_label:
                return "vdes"
        return family_record.runtime_tasks[0]
    normalized = _TASK_ALIASES.get(normalized, normalized)
    # Several release families condition cloning through a speaker reference on
    # the compiled ``tts`` task instead of exposing a separate ``clon`` task.
    if normalized == "clon" and "clon" not in family_record.runtime_tasks:
        if "tts" in family_record.runtime_tasks:
            return "tts"
    if normalized not in family_record.runtime_tasks:
        supported = ", ".join(family_record.runtime_tasks)
        raise CatalogError(
            f"Task {requested!r} is unavailable for {family!r} in audio.cpp "
            f"{AUDIO_CPP_RELEASE_VERSION}; supported: {supported}"
        )
    return normalized
