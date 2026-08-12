"""Machine-local audio.cpp settings stored outside workflow JSON."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


SETTINGS_SCHEMA_VERSION = 1
SETTINGS_FILENAME = "settings.json"


class SettingsError(ValueError):
    """Raised for invalid audio.cpp machine-local settings."""


@dataclass(frozen=True)
class AudioCppSettings:
    schema_version: int = SETTINGS_SCHEMA_VERSION
    connection_mode: str = "managed"
    external_server_url: str = ""
    executable_path: str = ""
    model_roots: Tuple[str, ...] = ()
    managed_model_root: str = ""
    runtime_root: str = ""
    runtime_backend: str = "auto"
    host: str = "127.0.0.1"
    port: int = 0
    extras: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AudioCppSettings":
        if not isinstance(values, Mapping):
            raise SettingsError("audio.cpp settings must be a JSON object")
        known = {
            "schema_version",
            "connection_mode",
            "external_server_url",
            "executable_path",
            "model_roots",
            "managed_model_root",
            "runtime_root",
            "runtime_backend",
            "host",
            "port",
        }
        roots = values.get("model_roots", ())
        if roots is None:
            roots = ()
        if not isinstance(roots, (list, tuple)) or not all(isinstance(item, str) for item in roots):
            raise SettingsError("model_roots must be a list of paths")
        mode = str(values.get("connection_mode", "managed")).strip().lower()
        if mode not in {"managed", "external"}:
            raise SettingsError("connection_mode must be 'managed' or 'external'")
        backend = str(values.get("runtime_backend", "auto")).strip().lower()
        if backend not in {"auto", "cpu", "cuda"}:
            raise SettingsError("runtime_backend must be 'auto', 'cpu', or 'cuda'")
        try:
            port = int(values.get("port", 0))
        except (TypeError, ValueError) as exc:
            raise SettingsError("port must be an integer") from exc
        if not 0 <= port <= 65535:
            raise SettingsError("port must be between 0 and 65535")
        schema_version = int(values.get("schema_version", SETTINGS_SCHEMA_VERSION))
        if schema_version > SETTINGS_SCHEMA_VERSION:
            raise SettingsError(
                f"Unsupported audio.cpp settings schema {schema_version}; "
                f"maximum is {SETTINGS_SCHEMA_VERSION}"
            )
        return cls(
            schema_version=SETTINGS_SCHEMA_VERSION,
            connection_mode=mode,
            external_server_url=str(values.get("external_server_url", "")).strip(),
            executable_path=str(values.get("executable_path", "")).strip(),
            model_roots=tuple(item.strip() for item in roots if item.strip()),
            managed_model_root=str(values.get("managed_model_root", "")).strip(),
            runtime_root=str(values.get("runtime_root", "")).strip(),
            runtime_backend=backend,
            host=str(values.get("host", "127.0.0.1")).strip() or "127.0.0.1",
            port=port,
            extras={key: value for key, value in values.items() if key not in known},
        )

    def to_mapping(self) -> Dict[str, Any]:
        values = dict(self.extras)
        serialized = asdict(self)
        serialized.pop("extras", None)
        serialized["model_roots"] = list(self.model_roots)
        values.update(serialized)
        return values


def _import_folder_paths():
    try:
        import folder_paths  # type: ignore

        return folder_paths
    except (ImportError, RuntimeError):
        return None


def _fallback_settings_directory() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "TTS Audio Suite" / "audio_cpp"
    if os.environ.get("XDG_CONFIG_HOME"):
        return Path(os.environ["XDG_CONFIG_HOME"]) / "tts_audio_suite" / "audio_cpp"
    return Path.home() / ".config" / "tts_audio_suite" / "audio_cpp"


def get_settings_path(folder_paths_module=None) -> Path:
    """Resolve settings below ComfyUI's internal user directory when available."""

    module = folder_paths_module if folder_paths_module is not None else _import_folder_paths()
    if module is not None and hasattr(module, "get_system_user_directory"):
        try:
            base = Path(module.get_system_user_directory("tts_audio_suite"))
            return base / "audio_cpp" / SETTINGS_FILENAME
        except (OSError, TypeError, ValueError):
            pass
    return _fallback_settings_directory() / SETTINGS_FILENAME


def load_settings(
    path: Optional[Path] = None,
    *,
    folder_paths_module=None,
    strict: bool = False,
) -> AudioCppSettings:
    settings_path = Path(path) if path is not None else get_settings_path(folder_paths_module)
    if not settings_path.is_file():
        return AudioCppSettings()
    try:
        values = json.loads(settings_path.read_text(encoding="utf-8"))
        return AudioCppSettings.from_mapping(values)
    except (OSError, json.JSONDecodeError, SettingsError, TypeError, ValueError):
        if strict:
            raise
        # A damaged local preference file must not stop ComfyUI from loading.
        return AudioCppSettings()


def save_settings(
    settings: AudioCppSettings,
    path: Optional[Path] = None,
    *,
    folder_paths_module=None,
) -> Path:
    """Atomically write settings in the same directory as the final file."""

    if not isinstance(settings, AudioCppSettings):
        settings = AudioCppSettings.from_mapping(settings)  # type: ignore[arg-type]
    settings_path = Path(path) if path is not None else get_settings_path(folder_paths_module)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{settings_path.name}.", suffix=".tmp", dir=settings_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(settings.to_mapping(), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, settings_path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return settings_path
