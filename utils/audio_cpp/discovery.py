"""Resolve existing audio.cpp models without copying or cache migration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from .catalog import AudioCppCatalog, PackageRecord, load_catalog
from .settings import AudioCppSettings, get_settings_path, load_settings


def _import_folder_paths():
    try:
        import folder_paths  # type: ignore

        return folder_paths
    except (ImportError, RuntimeError):
        return None


def _registered_paths(folder_paths_module, key: str) -> list[Path]:
    if folder_paths_module is None:
        return []
    registry = getattr(folder_paths_module, "folder_names_and_paths", {})
    if key not in registry:
        return []
    try:
        if hasattr(folder_paths_module, "get_folder_paths"):
            values = folder_paths_module.get_folder_paths(key)
        else:
            values = registry[key][0]
    except (KeyError, TypeError, ValueError):
        return []
    return [Path(value).expanduser() for value in values if value]


def _tts_paths(folder_paths_module) -> list[Path]:
    paths: list[Path] = []
    for key in ("TTS", "tts"):
        paths.extend(_registered_paths(folder_paths_module, key))
    if not paths and folder_paths_module is not None:
        models_dir = getattr(folder_paths_module, "models_dir", None)
        if models_dir:
            paths.append(Path(models_dir) / "TTS")
    return paths


def default_managed_model_root(
    *,
    settings: Optional[AudioCppSettings] = None,
    folder_paths_module=None,
) -> Path:
    current = settings or AudioCppSettings()
    if current.managed_model_root:
        return Path(current.managed_model_root).expanduser()
    module = folder_paths_module if folder_paths_module is not None else _import_folder_paths()
    tts_paths = _tts_paths(module)
    if tts_paths:
        return tts_paths[0] / "audio.cpp" / "models"
    # This is only a non-ComfyUI safety fallback; it remains direct storage, not
    # a Hugging Face cache.
    return get_settings_path(module).parent / "models"


def _canonical(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def resolve_model_roots(
    explicit_roots: Optional[Iterable[Union[str, Path]]] = None,
    *,
    settings: Optional[AudioCppSettings] = None,
    folder_paths_module=None,
) -> list[Path]:
    """Return search roots in priority order, with managed storage last."""

    module = folder_paths_module if folder_paths_module is not None else _import_folder_paths()
    current = settings or load_settings(folder_paths_module=module)
    managed = default_managed_model_root(settings=current, folder_paths_module=module)
    candidates: list[Path] = []
    candidates.extend(Path(path).expanduser() for path in (explicit_roots or ()) if path)
    candidates.extend(Path(path).expanduser() for path in current.model_roots if path)
    candidates.extend(_registered_paths(module, "audio_cpp"))
    candidates.extend(path / "audio.cpp" / "models" for path in _tts_paths(module))

    managed_key = _canonical(managed)
    seen: set[str] = set()
    result: list[Path] = []
    for candidate in candidates:
        key = _canonical(candidate)
        if key == managed_key or key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    result.append(managed)
    return result


def package_install_path(package: PackageRecord, models_root: Union[str, Path]) -> Path:
    return Path(models_root) / Path(*package.target_directory.replace("\\", "/").split("/"))


def package_is_complete(package: PackageRecord, package_directory: Union[str, Path]) -> bool:
    directory = Path(package_directory)
    return directory.is_dir() and all(
        (directory / path).is_file() and (directory / path).stat().st_size > 0
        for path in package.local_files
    )


def find_installed_package(
    package: Union[str, PackageRecord],
    roots: Optional[Sequence[Union[str, Path]]] = None,
    *,
    catalog: Optional[AudioCppCatalog] = None,
    settings: Optional[AudioCppSettings] = None,
    folder_paths_module=None,
) -> Optional[Path]:
    current_catalog = catalog or load_catalog()
    record = current_catalog.package(package) if isinstance(package, str) else package
    search_roots = (
        [Path(root) for root in roots]
        if roots is not None
        else resolve_model_roots(settings=settings, folder_paths_module=folder_paths_module)
    )
    for root in search_roots:
        candidate = package_install_path(record, root)
        if package_is_complete(record, candidate):
            return candidate
    return None


@dataclass(frozen=True)
class ResolvedModel:
    package: PackageRecord
    path: Path
    root: Path


def resolve_model(
    package: Union[str, PackageRecord],
    roots: Optional[Sequence[Union[str, Path]]] = None,
    *,
    catalog: Optional[AudioCppCatalog] = None,
    settings: Optional[AudioCppSettings] = None,
    folder_paths_module=None,
) -> Optional[ResolvedModel]:
    """Resolve one catalog package and retain the root that won precedence."""

    current_catalog = catalog or load_catalog()
    record = current_catalog.package(package) if isinstance(package, str) else package
    search_roots = (
        [Path(root) for root in roots]
        if roots is not None
        else resolve_model_roots(settings=settings, folder_paths_module=folder_paths_module)
    )
    path = find_installed_package(record, search_roots, catalog=current_catalog)
    if path is None:
        return None
    for root in search_roots:
        if _canonical(package_install_path(record, root)) == _canonical(path):
            return ResolvedModel(package=record, path=path, root=root)
    return None


def discover_installed_packages(
    roots: Optional[Sequence[Union[str, Path]]] = None,
    *,
    catalog: Optional[AudioCppCatalog] = None,
    settings: Optional[AudioCppSettings] = None,
    folder_paths_module=None,
) -> dict[str, ResolvedModel]:
    current_catalog = catalog or load_catalog()
    found: dict[str, ResolvedModel] = {}
    for package in current_catalog.packages.values():
        resolved = resolve_model(
            package,
            roots,
            catalog=current_catalog,
            settings=settings,
            folder_paths_module=folder_paths_module,
        )
        if resolved is not None:
            found[package.id] = resolved
    return found
