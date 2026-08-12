"""Owned audio.cpp server process launcher."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .client import AudioCppClient, AudioCppClientError


class AudioCppProcessError(RuntimeError):
    """Base error for managed audio.cpp process failures."""


class AudioCppProcessStartupError(AudioCppProcessError):
    """The owned audio.cpp server failed before becoming ready."""


def find_free_loopback_port() -> int:
    """Ask the OS for a currently unused IPv4 loopback port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def normalize_audio_cpp_task(task: Any) -> str:
    normalized = str(task or "tts").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "clone": "clon",
        "cloning": "clon",
        "voice_clone": "clon",
        "voice_cloning": "clon",
        "design": "vdes",
        "voice_design": "vdes",
        "voice_designer": "vdes",
    }
    return aliases.get(normalized, normalized)


def _first(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _resolve_existing_path(value: Any, label: str, *, executable: bool = False) -> Path:
    raw = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    if not raw:
        raise AudioCppProcessStartupError(f"Missing audio.cpp {label}")
    candidate = Path(raw)
    if executable and not candidate.exists():
        located = shutil.which(raw)
        if located:
            candidate = Path(located)
    candidate = candidate.resolve()
    if not candidate.exists():
        raise AudioCppProcessStartupError(f"audio.cpp {label} does not exist: {candidate}")
    if executable and not candidate.is_file():
        raise AudioCppProcessStartupError(f"audio.cpp {label} is not a file: {candidate}")
    return candidate


def _safe_model_id(value: Any, family: str) -> str:
    raw = str(value or family or "audio-cpp-model").strip()
    cleaned = "".join(char if char.isalnum() or char in "._-" else "-" for char in raw)
    cleaned = cleaned.strip("-.")
    return cleaned or "audio-cpp-model"


def _option_dict(config: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AudioCppProcessStartupError(f"audio.cpp {key} must be a JSON object")
    return dict(value)


class AudioCppServerProcess:
    """One suite-owned, single-model ``audiocpp_server`` process."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.binary_path = _resolve_existing_path(
            _first(
                self.config,
                "binary_path",
                "server_binary",
                "executable_path",
                "audio_cpp_binary",
            ),
            "server executable",
            executable=True,
        )
        self.model_path = _resolve_existing_path(
            _first(self.config, "model_path", "package_path", "gguf_path"),
            "model path",
        )
        self.family = str(_first(self.config, "family", "model_family", default="")).strip()
        if not self.family:
            raise AudioCppProcessStartupError("Missing audio.cpp model family")
        self.task = normalize_audio_cpp_task(_first(self.config, "task", default="tts"))
        self.model_id = _safe_model_id(
            _first(self.config, "model_id", "server_model_id", "package_id"),
            self.family,
        )
        self.backend = str(_first(self.config, "backend", default="cuda")).strip().lower()
        if self.backend == "auto":
            self.backend = "cuda"
        if self.backend not in {"cuda", "hip", "cpu", "vulkan", "metal"}:
            raise AudioCppProcessStartupError(f"Unsupported audio.cpp backend: {self.backend}")
        self.device = self._resolve_device_index(
            _first(self.config, "device_index", "device", default=0)
        )
        # Match audio.cpp's CLI default instead of the server's conservative
        # one-thread example configuration.
        self.threads = max(1, int(_first(self.config, "threads", default=4)))
        self.port = int(_first(self.config, "port", "server_port", default=0) or 0)
        self.startup_timeout = max(
            0.1, float(_first(self.config, "startup_timeout", "startup_timeout_seconds", default=30.0))
        )
        self.request_timeout = max(
            0.1, float(_first(self.config, "request_timeout", "request_timeout_seconds", default=600.0))
        )
        self.connect_timeout = max(
            0.05, float(_first(self.config, "connect_timeout", "connect_timeout_seconds", default=2.0))
        )
        self.stop_timeout = max(
            0.1, float(_first(self.config, "stop_timeout", "stop_timeout_seconds", default=5.0))
        )
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._client: Optional[AudioCppClient] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._config_path: Optional[Path] = None
        self._log_path: Optional[Path] = None
        self._log_handle = None
        self._closed = False

    @staticmethod
    def _resolve_device_index(value: Any) -> int:
        text = str(value).strip().lower()
        if text in {"auto", "cuda", "hip", "cpu", "vulkan", "metal", ""}:
            return 0
        if ":" in text:
            text = text.rsplit(":", 1)[-1]
        try:
            return max(0, int(text))
        except ValueError as exc:
            raise AudioCppProcessStartupError(
                f"audio.cpp device must be an integer index, got {value!r}"
            ) from exc

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return self._process

    @property
    def client(self) -> AudioCppClient:
        if self._client is None:
            raise AudioCppProcessError("audio.cpp server process has not started")
        return self._client

    @property
    def config_path(self) -> Optional[Path]:
        return self._config_path

    @property
    def log_path(self) -> Optional[Path]:
        return self._log_path

    @property
    def base_url(self) -> str:
        if self.port <= 0:
            raise AudioCppProcessError("audio.cpp server port is not allocated")
        return f"http://127.0.0.1:{self.port}"

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _model_spec_override(self) -> Optional[Path]:
        if "model_spec_override" in self.config:
            explicit = self.config.get("model_spec_override")
            if explicit in (None, "", False):
                return None
            path = _resolve_existing_path(explicit, "model spec override")
        else:
            path = (Path(__file__).resolve().parent / "model_specs").resolve()
            if not path.is_dir():
                raise AudioCppProcessStartupError(
                    "Bundled audio.cpp release-0.5.1 model specs are missing: " + str(path)
                )
        return path

    def _build_server_config(self) -> Dict[str, Any]:
        model: Dict[str, Any] = {
            "id": self.model_id,
            "family": self.family,
            "path": str(self.model_path),
            "task": self.task,
            "mode": str(_first(self.config, "mode", "run_mode", default="offline")),
            "lazy": bool(_first(self.config, "lazy", "lazy_load", default=True)),
            "load_options": _option_dict(self.config, "load_options"),
            "session_options": _option_dict(self.config, "session_options"),
            "default_request_options": _option_dict(self.config, "default_request_options"),
        }
        for source_key, target_key in (
            ("config_id", "config"),
            ("weight_id", "weight"),
            ("voice_presets", "voice_presets"),
            ("default_voice_preset", "default_voice_preset"),
            ("model_busy_timeout_ms", "busy_timeout_ms"),
        ):
            if source_key in self.config and self.config[source_key] is not None:
                model[target_key] = self.config[source_key]

        server: Dict[str, Any] = {
            "host": "127.0.0.1",
            "port": self.port,
            "cors_origins": "",
            "backend": self.backend,
            "device": self.device,
            "threads": self.threads,
            "lazy_load": bool(_first(self.config, "lazy_load", default=True)),
            "log_request_body": False,
            "max_request_body_bytes": int(
                _first(self.config, "max_request_body_bytes", default=2 * 1024 * 1024 * 1024)
            ),
            "busy_timeout_ms": int(_first(self.config, "busy_timeout_ms", default=300000)),
            "models": [model],
        }
        model_spec_override = self._model_spec_override()
        if model_spec_override is not None:
            server["model_spec_override"] = str(model_spec_override)
        return server

    def _prepare_files(self) -> None:
        temp_root = _first(self.config, "temp_root", "runtime_temp_root")
        if temp_root:
            Path(str(temp_root)).expanduser().resolve().mkdir(parents=True, exist_ok=True)
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix="tts_audio_cpp_",
            dir=str(Path(str(temp_root)).expanduser().resolve()) if temp_root else None,
        )
        temp_path = Path(self._temp_dir.name)
        self._config_path = temp_path / "server.json"
        log_dir = _first(self.config, "log_dir")
        if log_dir:
            resolved_log_dir = Path(str(log_dir)).expanduser().resolve()
            resolved_log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = resolved_log_dir / f"audio_cpp_{self.model_id}_{self.port}.log"
        else:
            self._log_path = temp_path / "server.log"
        with self._config_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self._build_server_config(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        self._log_handle = self._log_path.open("ab", buffering=0)

    def _command(self) -> list[str]:
        binary_args = _first(self.config, "binary_args", "launcher_args", default=[])
        if binary_args is None:
            binary_args = []
        if isinstance(binary_args, (str, bytes)) or not isinstance(binary_args, Sequence):
            raise AudioCppProcessStartupError("audio.cpp binary_args must be a list")
        return [
            str(self.binary_path),
            *[str(item) for item in binary_args],
            "--config",
            str(self._config_path),
        ]

    def start(self) -> "AudioCppServerProcess":
        with self._lock:
            if self._closed:
                raise AudioCppProcessError("audio.cpp server process launcher is closed")
            if self.running:
                return self
            if self.port <= 0:
                self.port = find_free_loopback_port()
            try:
                self._prepare_files()
                env = os.environ.copy()
                extra_env = _first(self.config, "environment", "env", default={})
                if extra_env:
                    if not isinstance(extra_env, Mapping):
                        raise AudioCppProcessStartupError(
                            "audio.cpp environment must be an object"
                        )
                    env.update({str(key): str(value) for key, value in extra_env.items()})
                env.setdefault("PYTHONUTF8", "1")
                creationflags = (
                    getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                )
                self._process = subprocess.Popen(
                    self._command(),
                    cwd=str(self.binary_path.parent),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    creationflags=creationflags,
                )
                self._client = AudioCppClient(
                    self.base_url,
                    connect_timeout=self.connect_timeout,
                    request_timeout=self.request_timeout,
                )
                self._wait_until_ready()
                return self
            except Exception as exc:
                log_tail = self.read_log_tail()
                self._terminate_exact_process()
                self._client = None
                self._cleanup_files()
                if isinstance(exc, AudioCppProcessStartupError):
                    raise
                suffix = f"\nServer log tail:\n{log_tail}" if log_tail else ""
                raise AudioCppProcessStartupError(
                    f"Failed to start audio.cpp server: {exc}{suffix}"
                ) from exc

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_error: Optional[BaseException] = None
        while time.monotonic() < deadline:
            if self._process is None or self._process.poll() is not None:
                code = self._process.poll() if self._process is not None else "unknown"
                log_tail = self.read_log_tail()
                suffix = f"\nServer log tail:\n{log_tail}" if log_tail else ""
                raise AudioCppProcessStartupError(
                    f"audio.cpp server exited during startup with code {code}{suffix}"
                )
            try:
                health = self.client.health(timeout=min(self.connect_timeout, 0.5))
                status = str(health.get("status", "")).lower()
                if status in {"ok", "ready", "healthy"} or health:
                    models = self.client.models(timeout=min(self.connect_timeout, 1.0))
                    if any(str(item.get("id")) == self.model_id for item in models):
                        return
                    last_error = AudioCppProcessStartupError(
                        f"audio.cpp server did not register expected model id '{self.model_id}'"
                    )
            except AudioCppClientError as exc:
                last_error = exc
            time.sleep(0.05)
        log_tail = self.read_log_tail()
        suffix = f"\nServer log tail:\n{log_tail}" if log_tail else ""
        raise AudioCppProcessStartupError(
            f"audio.cpp server was not ready after {self.startup_timeout:.1f}s"
            + (f": {last_error}" if last_error else "")
            + suffix
        )

    def read_log_tail(self, max_bytes: int = 16384) -> str:
        path = self._log_path
        if path is None or not path.exists():
            return ""
        try:
            if self._log_handle is not None:
                self._log_handle.flush()
            with path.open("rb") as handle:
                size = path.stat().st_size
                handle.seek(max(0, size - max(1, int(max_bytes))))
                return handle.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _terminate_exact_process(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=self.stop_timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=self.stop_timeout)
        finally:
            self._process = None

    def _cleanup_files(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except OSError:
                pass
            self._temp_dir = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._terminate_exact_process()
            self._client = None
            self._cleanup_files()

    stop = close

    def __enter__(self) -> "AudioCppServerProcess":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "AudioCppProcessError",
    "AudioCppProcessStartupError",
    "AudioCppServerProcess",
    "find_free_loopback_port",
    "normalize_audio_cpp_task",
]
