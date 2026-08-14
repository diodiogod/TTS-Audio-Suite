"""Resolve workflow and machine-local audio.cpp configuration into one session config."""

from __future__ import annotations

import os
import re
import shutil
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .settings import AudioCppSettings, load_settings


class AudioCppResolutionError(RuntimeError):
    """Raised when an audio.cpp engine configuration cannot be made runnable."""


class _SuiteDownloadProgress:
    """Match UnifiedDownloader's single-line console progress convention."""

    def __init__(self) -> None:
        self._completed: set[str] = set()

    def __call__(self, label: str, downloaded: int, total: Optional[int]) -> None:
        if not total or total <= 0:
            return
        filename = Path(label).name
        percent = min(100.0, downloaded * 100.0 / total)
        print(f"\r📥 Downloading {filename}: {percent:.1f}%", end="", flush=True)
        if downloaded >= total and label not in self._completed:
            self._completed.add(label)
            print()


def _print_download_block(
    title: str,
    *,
    model: str,
    description: str,
    repository: str,
    target: Path,
    size_bytes: Optional[int] = None,
) -> None:
    """Use the same boxed pre-download summary as the Suite engine downloaders."""

    print(f"\n{'=' * 60}")
    print(f"📦 {title}")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Description: {description}")
    print(f"Repository: {repository}")
    print(f"Download size: {_format_download_size(size_bytes)}")
    print(f"Target: {target}")
    print(f"{'=' * 60}\n")


def _format_download_size(size_bytes: Optional[int]) -> str:
    if size_bytes is None or size_bytes < 0:
        return "Unknown"
    if size_bytes >= 1024**3:
        return f"{size_bytes / 1024**3:.2f} GB"
    return f"{size_bytes / 1024**2:.1f} MB"


_EXTERNAL_MODES = {
    "external",
    "external_server",
    "existing_server",
    "remote_server",
    "server",
    "connect",
}
_OWNED_MODES = {
    "owned",
    "owned_process",
    "existing_binary",
    "managed_binary",
    "managed",
    "binary",
    "local_binary",
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


def _catalog_module():
    from . import catalog

    return catalog


def _discovery_module():
    from . import discovery

    return discovery


def _downloader_module():
    from . import downloader

    return downloader


def _runtime_installer_module():
    from . import runtime_installer

    return runtime_installer


def _flatten_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("audio.cpp config must be a mapping")
    nested = config.get("config")
    flattened = dict(nested) if isinstance(nested, Mapping) else {}
    flattened.update({key: value for key, value in config.items() if key != "config"})
    return flattened


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _first_value(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and _has_value(config[key]):
            return config[key]
    return default


def _merge_settings(config: Mapping[str, Any], settings: AudioCppSettings) -> dict[str, Any]:
    """Fill blank/absent workflow fields without replacing explicit values."""

    merged = settings.to_mapping()
    for key, value in config.items():
        if _has_value(value) or key not in merged:
            merged[key] = value
    return merged


def _normalize_mode(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "auto").strip().lower()).strip("_")
    if normalized in {"", "auto"}:
        return "auto"
    if normalized in _EXTERNAL_MODES:
        return "external_server"
    if normalized in _OWNED_MODES:
        return "owned_process"
    raise AudioCppResolutionError(f"Unsupported audio.cpp connection mode: {value!r}")


def _canonical_url(value: Any) -> str:
    raw = str(value or "").strip()
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise AudioCppResolutionError(
            "audio.cpp external mode requires a valid HTTP(S) server_url; "
            f"received {value!r}"
        )
    if parsed.query or parsed.fragment:
        raise AudioCppResolutionError(
            "audio.cpp external server_url must not contain a query string or fragment"
        )
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, parsed.path.rstrip("/"), "", "")
    )


def _path_text(value: Any) -> str:
    return os.path.expandvars(os.path.expanduser(os.fspath(value))).strip()


def _existing_path(value: Any, label: str, *, file_only: bool = False) -> Path:
    raw = _path_text(value)
    if not raw:
        raise AudioCppResolutionError(f"Missing audio.cpp {label}")
    candidate = Path(raw)
    if file_only and not candidate.exists():
        located = shutil.which(raw)
        if located:
            candidate = Path(located)
    candidate = candidate.resolve()
    if not candidate.exists():
        raise AudioCppResolutionError(f"audio.cpp {label} does not exist: {candidate}")
    if file_only and not candidate.is_file():
        raise AudioCppResolutionError(f"audio.cpp {label} is not a file: {candidate}")
    return candidate


def _canonical_model_id(value: Any, fallback: str) -> str:
    raw = str(value or fallback or "audio-cpp-model").strip()
    cleaned = "".join(char if char.isalnum() or char in "._-" else "-" for char in raw)
    return cleaned.strip("-.") or "audio-cpp-model"


def _device_index(value: Any) -> int:
    text = str(value if value is not None else 0).strip().lower()
    if text in {"", "auto", "cuda", "cpu", "vulkan", "metal", "hip"}:
        return 0
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    try:
        index = int(text)
    except ValueError as exc:
        raise AudioCppResolutionError(
            f"audio.cpp device must be a non-negative integer index, got {value!r}"
        ) from exc
    if index < 0:
        raise AudioCppResolutionError(
            f"audio.cpp device must be a non-negative integer index, got {value!r}"
        )
    return index


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        raise AudioCppResolutionError(f"Invalid boolean value in audio.cpp config: {value!r}")
    return bool(value)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        return False


def _resolve_backend(config: Mapping[str, Any], settings: AudioCppSettings, *, owned: bool) -> str:
    requested = str(_first_value(config, "backend", default="auto") or "auto").strip().lower()
    if owned and requested == "auto" and settings.runtime_backend in {"cpu", "cuda"}:
        requested = settings.runtime_backend
    if requested == "auto":
        return "cuda" if owned and _cuda_available() else ("cpu" if owned else "auto")
    supported = {"cuda", "cpu", "vulkan", "metal", "hip"} if owned else {
        "cuda",
        "cpu",
        "vulkan",
        "metal",
        "hip",
    }
    if requested not in supported:
        raise AudioCppResolutionError(f"Unsupported audio.cpp backend: {requested!r}")
    return requested


def _explicit_roots(config: Mapping[str, Any]) -> Optional[Sequence[Any]]:
    value = _first_value(config, "model_roots", "model_search_roots")
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    raise AudioCppResolutionError("audio.cpp model_roots must be a path or list of paths")


def _installed_runtime_binary(runtime_root: Any, backend: str, runtime_api) -> Optional[Path]:
    if not _has_value(runtime_root):
        return None
    root = Path(_path_text(runtime_root)).resolve()
    direct = root / "audiocpp_server.exe"
    if direct.is_file() and direct.stat().st_size > 0:
        return direct.resolve()
    if backend not in {"cpu", "cuda"}:
        return None
    candidate = runtime_api.runtime_install_path(root, backend) / "audiocpp_server.exe"
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate.resolve()
    return None


def _external_task(config: Mapping[str, Any]) -> str:
    requested = str(
        _first_value(config, "requested_task", "task", default="auto") or "auto"
    ).strip().lower().replace("-", "_").replace(" ", "_")
    requested = _TASK_ALIASES.get(requested, requested)
    supported = {"auto", "tts", "clon", "vdes", "vc", "s2s", "svc", "asr", "diar"}
    if requested not in supported:
        raise AudioCppResolutionError(f"Unsupported audio.cpp task: {requested!r}")
    return requested


def resolve_audio_cpp_config(
    config: Mapping[str, Any],
    *,
    external: Optional[bool] = None,
) -> dict[str, Any]:
    """Return a complete canonical config without starting an audio.cpp session.

    External mode intentionally returns before importing model discovery or either
    installer.  Owned mode may install only when the corresponding workflow flag
    explicitly permits it, and all installs target suite-managed storage.
    """

    workflow = _flatten_config(config)
    settings = load_settings()
    merged = _merge_settings(workflow, settings)
    workflow_mode = _normalize_mode(_first_value(workflow, "connection_mode", "source", default="auto"))
    workflow_url = _first_value(
        workflow, "server_url", "endpoint", "base_url", "external_server_url"
    )
    merged_url = _first_value(
        merged, "server_url", "endpoint", "base_url", "external_server_url"
    )
    workflow_binary = _first_value(
        workflow, "binary_path", "server_binary", "executable_path", "audio_cpp_binary"
    )

    if external is True:
        mode = "external_server"
    elif external is False:
        mode = "owned_process"
    elif workflow_mode != "auto":
        mode = workflow_mode
    elif _has_value(workflow_url):
        # A URL deliberately supplied by the workflow outranks machine defaults.
        mode = "external_server"
    elif _has_value(workflow_binary):
        mode = "owned_process"
    elif settings.connection_mode == "external" and _has_value(merged_url):
        mode = "external_server"
    elif _has_value(settings.executable_path):
        mode = "owned_process"
    else:
        mode = "owned_process"

    device_index = _device_index(_first_value(merged, "device_index", "device", default=0))
    family = str(_first_value(merged, "family", "model_family", default="") or "").strip()
    package_value = str(_first_value(merged, "package_id", default="auto") or "auto").strip()

    if mode == "external_server":
        endpoint = _canonical_url(merged_url)
        package_id = package_value or "auto"
        result = dict(merged)
        explicit_model_id = _first_value(workflow, "model_id", "server_model_id")
        result.update(
            {
                "connection_mode": "external_server",
                "server_url": endpoint,
                "external_server_url": endpoint,
                "family": family,
                "package_id": package_id,
                "task": _external_task(workflow),
                "backend": _resolve_backend(workflow, settings, owned=False),
                "device": device_index,
                "device_index": device_index,
                "binary_path": "",
                "model_path": "",
            }
        )
        if explicit_model_id:
            result["model_id"] = _canonical_model_id(explicit_model_id, "")
        else:
            # Omission is meaningful: the external session will select the
            # server's sole /v1/models entry.  Its lookup treats blank as an ID.
            result.pop("model_id", None)
            result.pop("server_model_id", None)
        return result

    if not family or family.lower() == "auto":
        raise AudioCppResolutionError(
            "Owned audio.cpp mode requires a model family from the pinned release-0.5.1 catalog"
        )

    catalog_api = _catalog_module()
    try:
        catalog = catalog_api.load_catalog()
        family_record = catalog.family(family)
        package_id = (
            family_record.recommended_package_id
            if package_value.lower() in {"", "auto"}
            else package_value
        )
        package = catalog.package(package_id)
        if package.family != family:
            raise AudioCppResolutionError(
                f"audio.cpp package {package_id!r} belongs to {package.family!r}, not {family!r}"
            )
        requested_task = _first_value(
            workflow, "requested_task", "task", default=_first_value(merged, "task", default="auto")
        )
        task = catalog_api.resolve_task(family, package_id, requested=str(requested_task or "auto"))
    except AudioCppResolutionError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise AudioCppResolutionError(f"Invalid audio.cpp model selection: {exc}") from exc

    backend = _resolve_backend(workflow, settings, owned=True)
    discovery_api = _discovery_module()
    managed_model_root = discovery_api.default_managed_model_root(settings=settings)

    explicit_model = _first_value(workflow, "model_path", "package_path", "gguf_path")
    if _has_value(explicit_model):
        model_path = _existing_path(explicit_model, "model path")
    else:
        resolved_model = discovery_api.resolve_model(
            package,
            _explicit_roots(workflow),
            catalog=catalog,
            settings=settings,
        )
        if resolved_model is not None:
            model_path = Path(resolved_model.path).resolve()
        elif _as_bool(_first_value(workflow, "auto_download_model", default=False)):
            downloader_api = _downloader_module()
            size_resolver = getattr(downloader_api, "package_download_size", None)
            download_size = (
                size_resolver(package, catalog=catalog) if callable(size_resolver) else None
            )
            _print_download_block(
                "audio.cpp Model Download",
                model=package.display_name,
                description=(
                    f"{family_record.display_name} {package.precision.upper()} "
                    f"{package.format.upper()} package"
                ),
                repository=package.repo,
                target=(Path(managed_model_root) / package.target_directory).resolve(),
                size_bytes=download_size,
            )
            print(f"📥 Downloading {package_id} directly (no cache)")
            try:
                download_result = downloader_api.install_package(
                    package,
                    managed_model_root,
                    catalog=catalog,
                    progress=_SuiteDownloadProgress(),
                )
            except Exception as exc:
                raise AudioCppResolutionError(
                    f"Failed to install audio.cpp package {package_id!r} in managed storage "
                    f"{managed_model_root}: {exc}"
                ) from exc
            model_path = Path(download_result.path).resolve()
            print(f"✅ Downloaded: {model_path}")
        else:
            searched = discovery_api.resolve_model_roots(
                _explicit_roots(workflow), settings=settings
            )
            raise AudioCppResolutionError(
                f"audio.cpp package {package_id!r} is not installed. Searched: "
                + ", ".join(str(Path(root)) for root in searched)
                + ". Provide model_path or enable auto_download_model."
            )

    dependency_session_options: Dict[str, Any] = {}
    try:
        from .capabilities import get_package_dependencies

        dependencies = get_package_dependencies(package_id)
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        raise AudioCppResolutionError(
            f"Cannot resolve audio.cpp dependencies for {package_id!r}: {exc}"
        ) from exc
    dependency_roots = discovery_api.resolve_model_roots(
        _explicit_roots(workflow), settings=settings
    )
    for dependency in dependencies:
        dependency_package = dependency["package"]
        dependency_path = discovery_api.find_installed_package(
            dependency_package,
            dependency_roots,
            settings=settings,
        )
        if dependency_path is None:
            if not _as_bool(_first_value(workflow, "auto_download_model", default=False)):
                raise AudioCppResolutionError(
                    f"audio.cpp package {package_id!r} requires {dependency_package.id!r}, "
                    "which is not installed. Enable auto_download_model to install it."
                )
            downloader_api = _downloader_module()
            _print_download_block(
                "audio.cpp Dependency Download",
                model=dependency_package.display_name,
                description=f"Required by {family_record.display_name}",
                repository=dependency_package.repo,
                target=(
                    Path(managed_model_root) / dependency_package.target_directory
                ).resolve(),
                size_bytes=int(dependency["estimated_download_bytes"]),
            )
            print(f"📥 Downloading {dependency_package.id} directly (no cache)")
            try:
                dependency_result = downloader_api.install_package(
                    dependency_package,
                    managed_model_root,
                    progress=_SuiteDownloadProgress(),
                )
            except Exception as exc:
                raise AudioCppResolutionError(
                    f"Failed to install dependency {dependency_package.id!r} for "
                    f"audio.cpp package {package_id!r}: {exc}"
                ) from exc
            dependency_path = Path(dependency_result.path).resolve()
            print(f"✅ Downloaded dependency: {dependency_path}")
        dependency_session_options[str(dependency["session_option"])] = str(
            Path(dependency_path).resolve()
        )

    explicit_binary = _first_value(
        workflow, "binary_path", "server_binary", "executable_path", "audio_cpp_binary"
    )
    managed_runtime_root = Path(managed_model_root).expanduser().resolve().parent / "runtime"
    if _has_value(explicit_binary):
        binary_path = _existing_path(explicit_binary, "server executable", file_only=True)
    elif _has_value(settings.executable_path):
        binary_path = _existing_path(
            settings.executable_path, "configured server executable", file_only=True
        )
    else:
        runtime_api = _runtime_installer_module()
        binary_path = _installed_runtime_binary(settings.runtime_root, backend, runtime_api)
        if binary_path is None:
            binary_path = _installed_runtime_binary(managed_runtime_root, backend, runtime_api)
        if binary_path is None and backend == "cpu":
            # The official CUDA profile also contains the CPU backend. Reuse it
            # before downloading a second executable profile solely for CPU mode.
            binary_path = _installed_runtime_binary(settings.runtime_root, "cuda", runtime_api)
        if binary_path is None and backend == "cpu":
            binary_path = _installed_runtime_binary(managed_runtime_root, "cuda", runtime_api)
        if binary_path is None:
            if backend not in {"cpu", "cuda"}:
                raise AudioCppResolutionError(
                    f"The pinned managed audio.cpp runtime has no {backend!r} Windows artifact. "
                    "Provide binary_path for this backend."
                )
            if not _as_bool(_first_value(workflow, "auto_download_runtime", default=False)):
                expected = runtime_api.runtime_install_path(managed_runtime_root, backend)
                raise AudioCppResolutionError(
                    f"audio.cpp {backend} server runtime is not installed at {expected}. "
                    "Provide binary_path or enable auto_download_runtime."
                )
            runtime_manifest = runtime_api.get_runtime_manifest(backend)
            _print_download_block(
                "audio.cpp Runtime Download",
                model=f"audio.cpp {runtime_manifest.release_version} ({backend})",
                description=(
                    f"Official Windows {runtime_manifest.profile} runtime, "
                    f"pinned to {runtime_manifest.release_tag}"
                ),
                repository="0xShug0/audio.cpp",
                target=runtime_api.runtime_install_path(managed_runtime_root, backend),
                size_bytes=sum(asset.size for asset in runtime_manifest.assets),
            )
            print(f"📥 Downloading audio.cpp {backend} runtime directly (no cache)")
            try:
                runtime_result = runtime_api.install_windows_runtime(
                    managed_runtime_root,
                    backend,
                    progress=_SuiteDownloadProgress(),
                )
            except Exception as exc:
                raise AudioCppResolutionError(
                    f"Failed to install the pinned audio.cpp {backend} balance runtime in "
                    f"{managed_runtime_root}: {exc}"
                ) from exc
            binary_path = Path(runtime_result.executable).resolve()
            print(f"✅ Downloaded: {binary_path}")

    result = dict(merged)
    session_options = dict(result.get("session_options") or {})
    session_options.update(dependency_session_options)
    result.update(
        {
            "connection_mode": "owned_process",
            "server_url": "",
            "external_server_url": "",
            "family": family,
            "package_id": package_id,
            "model_id": _canonical_model_id(
                _first_value(workflow, "model_id", "server_model_id"), package_id
            ),
            "task": task,
            "backend": backend,
            "device": device_index,
            "device_index": device_index,
            "binary_path": str(binary_path),
            "model_path": str(model_path),
            "session_options": session_options,
        }
    )
    return result


__all__ = ["AudioCppResolutionError", "resolve_audio_cpp_config"]
