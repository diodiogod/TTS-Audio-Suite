"""Direct, cache-free downloader for pinned audio.cpp model packages."""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Union
from urllib.parse import quote

from .catalog import AudioCppCatalog, PackageRecord, load_catalog
from .discovery import package_install_path, package_is_complete


class AudioCppDownloadError(RuntimeError):
    """Raised when a model package cannot be downloaded safely."""


class IncompleteExistingPackageError(AudioCppDownloadError):
    """Raised when a package target exists but is not complete."""


ProgressCallback = Callable[[str, int, Optional[int]], None]


@dataclass(frozen=True)
class DownloadResult:
    package: PackageRecord
    path: Path
    downloaded_files: tuple[Path, ...]
    bytes_downloaded: int
    already_present: bool = False


def resolve_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    if explicit_token:
        return explicit_token.strip() or None
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        from huggingface_hub import get_token

        value = get_token()
        if value:
            return value.strip()
    except (ImportError, OSError, RuntimeError):
        pass
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    try:
        value = token_path.read_text(encoding="utf-8").strip()
        return value or None
    except OSError:
        return None


def huggingface_resolve_url(repo: str, revision: str, remote_path: str) -> str:
    return (
        "https://huggingface.co/"
        f"{quote(repo, safe='/')}/resolve/{quote(revision, safe='')}/"
        f"{quote(remote_path.replace(chr(92), '/'), safe='/')}"
    )


def download_url_to_path(
    url: str,
    destination: Union[str, Path],
    *,
    headers: Optional[Mapping[str, str]] = None,
    timeout: int = 300,
    opener=None,
    progress: Optional[ProgressCallback] = None,
    progress_label: str = "download",
) -> int:
    """Stream a URL to a new file and validate HTTP Content-Length."""

    destination_path = Path(destination)
    request = urllib.request.Request(url, headers=dict(headers or {}))
    open_request = opener or urllib.request.urlopen
    response = None
    downloaded = 0
    try:
        response = open_request(request, timeout=timeout)
        status = getattr(response, "status", None) or getattr(response, "code", None)
        if status is not None and int(status) >= 400:
            raise AudioCppDownloadError(f"HTTP {status} while downloading {url}")
        content_length_value = response.headers.get("Content-Length") if response.headers else None
        expected = int(content_length_value) if content_length_value else None
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with destination_path.open("xb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if progress is not None:
                    progress(progress_label, downloaded, expected)
        if expected is not None and downloaded != expected:
            raise AudioCppDownloadError(
                f"Incomplete download for {progress_label}: expected {expected} bytes, got {downloaded}"
            )
        return downloaded
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise AudioCppDownloadError(
                f"Hugging Face denied {progress_label} (HTTP {exc.code}). "
                "Accept any model license and configure HF_TOKEN."
            ) from exc
        raise AudioCppDownloadError(f"HTTP {exc.code} while downloading {progress_label}") from exc
    except urllib.error.URLError as exc:
        raise AudioCppDownloadError(f"Network error downloading {progress_label}: {exc.reason}") from exc
    except OSError as exc:
        raise AudioCppDownloadError(f"Cannot write {destination_path}: {exc}") from exc
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _link_or_copy(source: str, destination: str) -> str:
    """Hard-link existing model files into staging, copying only if necessary."""

    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _merge_existing_target(target: Path, staging: Path) -> None:
    """Preserve other precision packages that share this target directory."""

    if target.is_symlink() or not target.is_dir():
        raise IncompleteExistingPackageError(
            f"Model target cannot be safely merged because it is not a regular directory: {target}"
        )
    shutil.copytree(
        target,
        staging,
        dirs_exist_ok=True,
        copy_function=_link_or_copy,
        symlinks=False,
    )


def _publish_directory(staging: Path, target: Path, replace_existing: bool) -> None:
    # PocketTTS targets are nested (for example PocketTTS-GGUF/english).
    # The parent must exist before Windows can rename the staged directory.
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() and not target.is_symlink():
        staging.rename(target)
        return
    if not replace_existing:
        raise IncompleteExistingPackageError(
            f"Model target already exists but is incomplete: {target}. "
            "Choose overwrite explicitly or repair it manually."
        )
    backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.backup")
    target.rename(backup)
    try:
        staging.rename(target)
    except BaseException:
        if not target.exists() and backup.exists():
            backup.rename(target)
        raise
    _remove_path(backup)


def install_package(
    package: Union[str, PackageRecord],
    models_root: Union[str, Path],
    *,
    catalog: Optional[AudioCppCatalog] = None,
    overwrite: bool = False,
    token: Optional[str] = None,
    timeout: int = 300,
    opener=None,
    progress: Optional[ProgressCallback] = None,
) -> DownloadResult:
    """Download every required package file, then atomically publish it."""

    current_catalog = catalog or load_catalog()
    record = current_catalog.package(package) if isinstance(package, str) else package
    if record.download.get("kind") != "huggingface_snapshot":
        raise AudioCppDownloadError(
            f"Unsupported download kind for {record.id}: {record.download.get('kind')!r}"
        )
    if not record.repo:
        raise AudioCppDownloadError(f"Package {record.id!r} has no Hugging Face repository")

    root = Path(models_root).expanduser()
    target = package_install_path(record, root)
    if package_is_complete(record, target) and not overwrite:
        return DownloadResult(record, target, (), 0, already_present=True)
    target_exists = target.exists() or target.is_symlink()
    if target_exists:
        if target.is_symlink() or not target.is_dir():
            raise IncompleteExistingPackageError(
                f"Model target cannot be safely merged because it is not a regular directory: {target}"
            )
        requested_files_exist = any(
            (target / path).exists() or (target / path).is_symlink()
            for path in record.local_files
        )
        if requested_files_exist and not overwrite:
            raise IncompleteExistingPackageError(
                f"Model target contains an incomplete package {record.id!r}: {target}. "
                "Choose overwrite explicitly or repair it manually."
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".staging", dir=target.parent))
    downloaded_files: list[Path] = []
    downloaded_bytes = 0
    hf_token = resolve_hf_token(token)
    headers = {"User-Agent": "TTS-Audio-Suite/audio.cpp-model-installer"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    try:
        if target_exists:
            _merge_existing_target(target, staging)
        for remote_path, local_path in zip(record.files, record.local_files):
            destination = staging / local_path
            # The merge may have hard-linked an older copy of the requested
            # precision.  Remove that link before creating the replacement.
            destination.unlink(missing_ok=True)
            url = huggingface_resolve_url(record.repo, record.revision, remote_path)
            downloaded_bytes += download_url_to_path(
                url,
                destination,
                headers=headers,
                timeout=timeout,
                opener=opener,
                progress=progress,
                progress_label=remote_path,
            )
            downloaded_files.append(local_path)
        if not package_is_complete(record, staging):
            missing = [str(path) for path in record.local_files if not (staging / path).is_file()]
            raise AudioCppDownloadError(
                f"Package {record.id!r} staging validation failed; missing: {missing}"
            )
        _publish_directory(staging, target, replace_existing=target_exists)
        return DownloadResult(
            package=record,
            path=target,
            downloaded_files=tuple(downloaded_files),
            bytes_downloaded=downloaded_bytes,
        )
    except BaseException:
        _remove_path(staging)
        raise
