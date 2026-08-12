"""Verified Windows audio.cpp release-0.5.1 runtime installer."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import sys
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Optional, Sequence, Union

from .catalog import AUDIO_CPP_RELEASE_COMMIT, AUDIO_CPP_RELEASE_TAG, AUDIO_CPP_RELEASE_VERSION
from .downloader import AudioCppDownloadError, ProgressCallback, download_url_to_path


class RuntimeInstallError(RuntimeError):
    """Raised when a managed audio.cpp runtime cannot be verified or installed."""


@dataclass(frozen=True)
class RuntimeAsset:
    filename: str
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class RuntimeManifest:
    backend: str
    assets: tuple[RuntimeAsset, ...]
    required_files: tuple[str, ...]
    release_version: str = AUDIO_CPP_RELEASE_VERSION
    release_tag: str = AUDIO_CPP_RELEASE_TAG
    release_commit: str = AUDIO_CPP_RELEASE_COMMIT
    profile: str = "balance"
    platform: str = "windows"


_RELEASE_URL = "https://github.com/0xShug0/audio.cpp/releases/download/release-0.5.1"

WINDOWS_RUNTIME_MANIFESTS: Mapping[str, RuntimeManifest] = {
    "cpu": RuntimeManifest(
        backend="cpu",
        assets=(
            RuntimeAsset(
                filename="audiocpp-windows-cpu-balance-238ab6a9.zip",
                url=f"{_RELEASE_URL}/audiocpp-windows-cpu-balance-238ab6a9.zip",
                size=11_435_334,
                sha256="c9db54d75becfc9dfa6930d469b6f201f0f3919e36dfa7f488d0ee754ab5fe23",
            ),
        ),
        required_files=("audiocpp_server.exe", "audiocpp_cli.exe"),
    ),
    "cuda": RuntimeManifest(
        backend="cuda",
        assets=(
            RuntimeAsset(
                filename="audiocpp-windows-cuda-balance-238ab6a9.zip",
                url=f"{_RELEASE_URL}/audiocpp-windows-cuda-balance-238ab6a9.zip",
                size=248_519_503,
                sha256="7e20f1fa984960b327700365a9a8434dc8effa0fcaa8d871c6cf5a35bf77b2a7",
            ),
            RuntimeAsset(
                filename="audiocpp-windows-cuda-runtime.zip",
                url=f"{_RELEASE_URL}/audiocpp-windows-cuda-runtime.zip",
                size=575_505_446,
                sha256="46016655aff8f050806d81efd0fe256c15b86527935bfb3896208d4cac6b5ff8",
            ),
        ),
        required_files=(
            "audiocpp_server.exe",
            "audiocpp_cli.exe",
            "cublas64_13.dll",
            "cublasLt64_13.dll",
            "cufft64_12.dll",
        ),
    ),
}


@dataclass(frozen=True)
class RuntimeInstallResult:
    manifest: RuntimeManifest
    path: Path
    executable: Path
    already_present: bool = False


def get_runtime_manifest(backend: str) -> RuntimeManifest:
    normalized = str(backend).strip().lower()
    try:
        return WINDOWS_RUNTIME_MANIFESTS[normalized]
    except KeyError as exc:
        raise RuntimeInstallError("audio.cpp runtime backend must be 'cpu' or 'cuda'") from exc


def runtime_install_path(runtime_root: Union[str, Path], backend: str) -> Path:
    manifest = get_runtime_manifest(backend)
    return (
        Path(runtime_root).expanduser()
        / f"release-{manifest.release_version}"
        / f"windows-{manifest.backend}-{manifest.profile}"
    )


def runtime_is_complete(path: Union[str, Path], manifest: RuntimeManifest) -> bool:
    root = Path(path)
    return root.is_dir() and all(
        (root / relative).is_file() and (root / relative).stat().st_size > 0
        for relative in manifest.required_files
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or (path.parts and ":" in path.parts[0])
    ):
        raise RuntimeInstallError(f"Unsafe path in runtime archive: {member_name!r}")
    return Path(*path.parts)


def _extract_zip(archive: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                relative = _safe_member_path(info.filename)
                unix_mode = info.external_attr >> 16
                if stat.S_ISLNK(unix_mode):
                    raise RuntimeInstallError(f"Runtime archive contains a symlink: {info.filename}")
                target = destination / relative
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info, "r") as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeInstallError(f"Cannot extract runtime archive {archive.name}: {exc}") from exc


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _publish_directory(staging: Path, target: Path, overwrite: bool) -> None:
    if not target.exists() and not target.is_symlink():
        staging.rename(target)
        return
    if not overwrite:
        raise RuntimeInstallError(f"Runtime target exists but is incomplete: {target}")
    backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.backup")
    target.rename(backup)
    try:
        staging.rename(target)
    except BaseException:
        if not target.exists() and backup.exists():
            backup.rename(target)
        raise
    _remove_path(backup)


def install_windows_runtime(
    runtime_root: Union[str, Path],
    backend: str,
    *,
    overwrite: bool = False,
    manifest: Optional[RuntimeManifest] = None,
    platform_name: Optional[str] = None,
    timeout: int = 600,
    opener=None,
    progress: Optional[ProgressCallback] = None,
) -> RuntimeInstallResult:
    """Download, hash, extract, validate, and atomically publish a runtime."""

    platform_value = (platform_name or sys.platform).lower()
    if platform_value not in {"win32", "windows"}:
        raise RuntimeInstallError("Managed audio.cpp release binaries are currently Windows-only")
    selected = manifest or get_runtime_manifest(backend)
    if selected.backend != str(backend).strip().lower():
        raise RuntimeInstallError("Runtime manifest backend does not match the requested backend")
    target = runtime_install_path(runtime_root, selected.backend)
    if runtime_is_complete(target, selected) and not overwrite:
        return RuntimeInstallResult(
            selected, target, target / "audiocpp_server.exe", already_present=True
        )
    if (target.exists() or target.is_symlink()) and not overwrite:
        raise RuntimeInstallError(f"Runtime target exists but is incomplete: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".staging", dir=target.parent))
    archives = work / "archives"
    payload = work / "payload"
    archives.mkdir()
    payload.mkdir()
    try:
        for asset in selected.assets:
            archive = archives / asset.filename
            try:
                downloaded = download_url_to_path(
                    asset.url,
                    archive,
                    timeout=timeout,
                    opener=opener,
                    progress=progress,
                    progress_label=asset.filename,
                )
            except AudioCppDownloadError as exc:
                raise RuntimeInstallError(str(exc)) from exc
            if downloaded != asset.size:
                raise RuntimeInstallError(
                    f"Runtime asset {asset.filename} has size {downloaded}; expected {asset.size}"
                )
            actual_hash = _sha256(archive)
            if actual_hash.lower() != asset.sha256.lower():
                raise RuntimeInstallError(
                    f"SHA256 mismatch for {asset.filename}: expected {asset.sha256}, got {actual_hash}"
                )
            _extract_zip(archive, payload)

        if not runtime_is_complete(payload, selected):
            missing = [
                item
                for item in selected.required_files
                if not (payload / item).is_file() or (payload / item).stat().st_size == 0
            ]
            raise RuntimeInstallError(f"Runtime staging validation failed; missing: {missing}")
        _publish_directory(payload, target, overwrite=overwrite)
        return RuntimeInstallResult(selected, target, target / "audiocpp_server.exe")
    finally:
        _remove_path(work)
