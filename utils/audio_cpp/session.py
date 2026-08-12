"""Keyed audio.cpp sessions and ComfyUI lifecycle integration."""

from __future__ import annotations

import atexit
import ipaddress
import json
import os
import re
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .client import (
    AudioCppClient,
    AudioCppConnectionError,
    AudioCppTaskResult,
    AudioCppTimeoutError,
)
from .lifecycle import AudioCppRuntimeProxy
from .process import AudioCppServerProcess, normalize_audio_cpp_task
from .resolver import resolve_audio_cpp_config


def _warn(message: str, exc: Optional[BaseException] = None) -> None:
    """Emit diagnostics without assuming a UTF-8 Windows console."""
    text = f"WARNING: {message}"
    if exc is not None:
        text += f": {exc}"
    encoding = getattr(sys.stderr, "encoding", None) or "ascii"
    try:
        text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except LookupError:
        text = text.encode("ascii", errors="replace").decode("ascii")
    print(text, file=sys.stderr)


def _flatten_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("audio.cpp config must be a mapping")
    nested = config.get("config")
    flattened: Dict[str, Any] = dict(nested) if isinstance(nested, Mapping) else {}
    flattened.update({key: value for key, value in config.items() if key != "config"})
    return flattened


def _first(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _connection_mode(config: Mapping[str, Any]) -> str:
    raw = str(_first(config, "connection_mode", "source", default="auto"))
    normalized = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    external_aliases = {
        "existing_server",
        "external_server",
        "server",
        "remote_server",
        "connect",
    }
    owned_aliases = {
        "owned_process",
        "existing_binary",
        "managed_binary",
        "managed",
        "binary",
        "local_binary",
    }
    if normalized in external_aliases:
        return "external_server"
    if normalized in owned_aliases:
        return "owned_process"
    if normalized not in {"", "auto"}:
        raise ValueError(f"Unsupported audio.cpp connection mode: {raw}")
    endpoint = _first(config, "server_url", "endpoint", "base_url")
    return "external_server" if endpoint else "owned_process"


def _is_loopback_url(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").strip().lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _canonical_url(url: Any) -> str:
    parsed = urllib.parse.urlsplit(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid audio.cpp external server URL: {url!r}")
    if parsed.query or parsed.fragment:
        raise ValueError("audio.cpp external server URL must not contain a query or fragment")
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
    )


def _safe_model_id(config: Mapping[str, Any]) -> str:
    raw = _first(
        config,
        "model_id",
        "server_model_id",
        "package_id",
        default=_first(config, "family", "model_family", default="audio-cpp-model"),
    )
    value = str(raw or "audio-cpp-model").strip()
    cleaned = "".join(char if char.isalnum() or char in "._-" else "-" for char in value)
    return cleaned.strip("-.") or "audio-cpp-model"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _session_key(config: Mapping[str, Any], mode: str, model_id: str, endpoint: str = "") -> str:
    if mode == "external_server":
        identity = {
            "mode": mode,
            "endpoint": endpoint,
            "model_id": model_id,
        }
    else:
        identity_keys = (
            "binary_path",
            "server_binary",
            "executable_path",
            "audio_cpp_binary",
            "binary_args",
            "model_path",
            "package_path",
            "gguf_path",
            "family",
            "model_family",
            "task",
            "backend",
            "device",
            "device_index",
            "threads",
            "load_options",
            "session_options",
            "default_request_options",
            "config_id",
            "weight_id",
            "model_spec_override",
        )
        identity = {"mode": mode, "model_id": model_id}
        for key in identity_keys:
            if key in config:
                identity[key] = config[key]
    return json.dumps(_jsonable(identity), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _normalize_request_paths(request: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(request)
    path_fields = {
        "voice_ref",
        "audio_path",
        "source_audio",
        "target_voice",
        "prosody_ref",
        "style_ref",
    }
    for key in path_fields:
        value = normalized.get(key)
        if isinstance(value, os.PathLike) or (isinstance(value, str) and value.strip()):
            normalized[key] = str(
                Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
            )
    return normalized


class AudioCppSession:
    """Persistent external connection or restartable suite-owned audio.cpp server."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        owned: bool,
        model_id: str,
        client: Optional[AudioCppClient] = None,
        endpoint: str = "",
        model_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = dict(config)
        self.owned = bool(owned)
        self.model_id = str(model_id)
        self.endpoint = endpoint
        self.model_metadata = dict(model_metadata or {})
        self.family = str(
            _first(
                self.model_metadata,
                "family",
                default=_first(self.config, "family", "model_family", default=""),
            )
            or ""
        )
        self.task = normalize_audio_cpp_task(
            _first(
                self.model_metadata,
                "task",
                default=_first(self.config, "task", default="tts"),
            )
        )
        self.unload_models_supported = False
        self._client = client
        self._process: Optional[AudioCppServerProcess] = None
        self._lock = threading.RLock()
        self._closed = False
        self._model_ready_reported = False
        self._proxy = AudioCppRuntimeProxy(self) if self.owned else None

    @property
    def process(self) -> Optional[AudioCppServerProcess]:
        return self._process

    @property
    def running(self) -> bool:
        if not self.owned:
            return not self._closed
        return self._process is not None and self._process.running

    @property
    def proxy(self) -> Optional[AudioCppRuntimeProxy]:
        return self._proxy

    def _probe_opt_in_features(self) -> None:
        if not bool(self.config.get("probe_unload_models", False)) or self._client is None:
            return
        try:
            self.unload_models_supported = self._client.supports_feature("unload_models")
        except Exception as exc:
            _warn("audio.cpp unload_models feature probe failed", exc)
            self.unload_models_supported = False

    def _start_owned_runtime(self) -> None:
        if not self.owned:
            return
        with _RUNTIME_START_LOCK:
            if self._process is not None and self._process.running:
                return
            _stop_conflicting_audio_cpp_sessions(self)
            if str(self.config.get("backend", "cuda")).lower() in {"cuda", "hip", "auto"}:
                _clear_conflicting_suite_tts_models()
            process_config = dict(self.config)
            process_config["model_id"] = self.model_id
            family = self.family or str(self.config.get("family", "unknown"))
            backend = str(self.config.get("backend", "auto"))
            print(
                f"🚀 audio.cpp: Starting {backend} server for {family} "
                f"('{self.model_id}')..."
            )
            started = time.monotonic()
            process = AudioCppServerProcess(process_config)
            process.start()
            self._process = process
            self._client = process.client
            self.endpoint = process.base_url
            self._probe_opt_in_features()
            if self._proxy is not None:
                self._proxy.register()
            print(
                f"✅ audio.cpp: Server ready at {self.endpoint} "
                f"({time.monotonic() - started:.2f}s); model will load on first generation"
            )

    def _ensure_client(self) -> AudioCppClient:
        if self._closed:
            raise RuntimeError("audio.cpp session is closed")
        if self.owned:
            if self._process is None or not self._process.running or self._client is None:
                if self._proxy is not None:
                    self._proxy.unregister()
                if self._process is not None:
                    self._process.close()
                self._process = None
                self._client = None
                self._start_owned_runtime()
        if self._client is None:
            raise RuntimeError("audio.cpp session has no HTTP client")
        return self._client

    def _restart_after_transport_failure(self) -> AudioCppClient:
        if not self.owned:
            raise RuntimeError("Cannot restart an external audio.cpp server")
        self._stop_owned_runtime()
        self._start_owned_runtime()
        if self._client is None:
            raise RuntimeError("audio.cpp owned runtime restart did not create a client")
        return self._client

    def run(self, request: Mapping[str, Any]) -> AudioCppTaskResult:
        if not isinstance(request, Mapping):
            raise TypeError("audio.cpp task request must be a mapping")
        normalized_request = _normalize_request_paths(request)
        timeout = float(_first(self.config, "request_timeout", "request_timeout_seconds", default=600.0))
        with self._lock:
            client = self._ensure_client()
            first_request = not self._model_ready_reported
            started = time.monotonic()
            if first_request:
                action = "Loading model" if self.owned else "Sending first request to model"
                print(f"⏳ audio.cpp: {action} '{self.model_id}'...")
            try:
                result = client.run_task(self.model_id, normalized_request, timeout=timeout)
            except AudioCppConnectionError:
                if not self.owned:
                    raise
                client = self._restart_after_transport_failure()
                result = client.run_task(self.model_id, normalized_request, timeout=timeout)
            except AudioCppTimeoutError:
                # A live server may still be executing after the client times out.
                # Only restart when the exact child has actually exited.
                if not self.owned or (self._process is not None and self._process.running):
                    raise
                client = self._restart_after_transport_failure()
                result = client.run_task(self.model_id, normalized_request, timeout=timeout)
            if first_request:
                self._model_ready_reported = True
                print(
                    f"✅ audio.cpp: Model '{self.model_id}' loaded; first generation completed "
                    f"in {time.monotonic() - started:.2f}s"
                )
            return result

    def voices(self) -> list[str]:
        timeout = float(_first(self.config, "connect_timeout", "connect_timeout_seconds", default=5.0))
        with self._lock:
            client = self._ensure_client()
            try:
                return client.voices(self.model_id, timeout=timeout)
            except AudioCppConnectionError:
                if not self.owned:
                    raise
                return self._restart_after_transport_failure().voices(
                    self.model_id, timeout=timeout
                )

    def _stop_owned_runtime(self, *, unregister: bool = True) -> None:
        if not self.owned:
            return
        with self._lock:
            process = self._process
            self._process = None
            self._client = None
            self._model_ready_reported = False
            if unregister and self._proxy is not None:
                self._proxy.unregister()
            if process is not None:
                process.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.owned:
                self._stop_owned_runtime()
            else:
                # External servers are never unloaded, reconfigured, or terminated.
                self._client = None


_SESSIONS: Dict[str, AudioCppSession] = {}
_SESSIONS_LOCK = threading.RLock()
_RUNTIME_START_LOCK = threading.RLock()


def _stop_conflicting_audio_cpp_sessions(active: AudioCppSession) -> None:
    if str(active.config.get("backend", "cuda")).lower() not in {"cuda", "hip", "auto"}:
        return
    with _SESSIONS_LOCK:
        conflicts = [
            session
            for session in _SESSIONS.values()
            if session is not active
            and session.owned
            and session.running
            and str(session.config.get("backend", "cuda")).lower()
            in {"cuda", "hip", "auto"}
        ]
    for session in conflicts:
        session._stop_owned_runtime()


def _clear_conflicting_suite_tts_models() -> None:
    """Clear known suite-managed TTS resources only when their modules are already live."""
    interface_module = sys.modules.get("utils.models.unified_model_interface")
    interface = getattr(interface_module, "unified_model_interface", None)
    if interface is not None:
        try:
            isolated = getattr(interface, "_isolated_model_cache", None)
            remover = getattr(interface, "_remove_isolated_model", None)
            if isinstance(isolated, dict) and callable(remover):
                for cache_key in list(isolated):
                    if "_tts_" in cache_key:
                        remover(cache_key)
        except Exception as exc:
            _warn("Could not clear a conflicting isolated TTS runtime", exc)

    wrapper_module = sys.modules.get("utils.models.comfyui_model_wrapper")
    manager = getattr(wrapper_module, "tts_model_manager", None)
    cache = getattr(manager, "_model_cache", None)
    remover = getattr(manager, "remove_model", None)
    if isinstance(cache, dict) and callable(remover):
        try:
            for cache_key, wrapper in list(cache.items()):
                model_info = getattr(wrapper, "model_info", None)
                if getattr(model_info, "model_type", None) == "tts":
                    remover(cache_key)
        except Exception as exc:
            _warn("Could not clear a conflicting embedded TTS model", exc)


def _external_client(
    config: Mapping[str, Any],
) -> tuple[AudioCppClient, str, str, Dict[str, Any]]:
    endpoint = _canonical_url(_first(config, "server_url", "endpoint", "base_url"))
    if not _is_loopback_url(endpoint) and not bool(config.get("allow_remote_server", False)):
        raise ValueError(
            "External audio.cpp servers must use loopback by default. "
            "Set allow_remote_server only when transport security and path access are understood."
        )
    client = AudioCppClient(
        endpoint,
        connect_timeout=float(
            _first(config, "connect_timeout", "connect_timeout_seconds", default=5.0)
        ),
        request_timeout=float(
            _first(config, "request_timeout", "request_timeout_seconds", default=600.0)
        ),
    )
    models = client.models()
    requested_id = _first(config, "model_id", "server_model_id")
    available_models = [dict(item) for item in models if item.get("id") is not None]
    available_ids = [str(item["id"]) for item in available_models]
    if requested_id is not None:
        model_id = str(requested_id)
        if model_id not in available_ids:
            raise ValueError(
                f"External audio.cpp server does not expose model '{model_id}'. "
                f"Available: {', '.join(available_ids) or '(none)'}"
            )
    elif len(available_ids) == 1:
        model_id = available_ids[0]
    elif not available_ids:
        raise ValueError("External audio.cpp server does not expose any configured models")
    else:
        raise ValueError(
            "External audio.cpp server exposes multiple models; select model_id explicitly"
        )
    selected_metadata = next(
        (item for item in available_models if str(item.get("id")) == model_id),
        {"id": model_id},
    )
    return client, endpoint, model_id, selected_metadata


def get_audio_cpp_session(config: Mapping[str, Any]) -> AudioCppSession:
    """Return a keyed persistent audio.cpp session for a loose engine config dict."""
    flattened = resolve_audio_cpp_config(_flatten_config(config))
    mode = _connection_mode(flattened)
    if mode == "external_server":
        endpoint_hint = _canonical_url(
            _first(flattened, "server_url", "endpoint", "base_url")
        )
        model_hint = _first(flattened, "model_id", "server_model_id")
        auto_key = None
        if isinstance(model_hint, str) and model_hint.strip():
            hinted_key = _session_key(flattened, mode, model_hint.strip(), endpoint_hint)
            with _SESSIONS_LOCK:
                existing = _SESSIONS.get(hinted_key)
                if existing is not None and not existing._closed:
                    return existing
        else:
            # An omitted model id means "select the server's sole model". Cache
            # that resolution per endpoint so every text chunk does not repeat
            # /v1/models before reaching the already-persistent session.
            auto_key = _session_key(flattened, mode, "", endpoint_hint)
            with _SESSIONS_LOCK:
                existing = _SESSIONS.get(auto_key)
                if existing is not None and not existing._closed:
                    return existing
        client, endpoint, model_id, model_metadata = _external_client(flattened)
        key = _session_key(flattened, mode, model_id, endpoint)
        with _SESSIONS_LOCK:
            existing = _SESSIONS.get(key)
            if existing is not None and not existing._closed:
                if auto_key is not None:
                    _SESSIONS[auto_key] = existing
                return existing
            session = AudioCppSession(
                flattened,
                owned=False,
                model_id=model_id,
                client=client,
                endpoint=endpoint,
                model_metadata=model_metadata,
            )
            session._probe_opt_in_features()
            _SESSIONS[key] = session
            if auto_key is not None:
                _SESSIONS[auto_key] = session
            return session

    model_id = _safe_model_id(flattened)
    key = _session_key(flattened, mode, model_id)
    with _SESSIONS_LOCK:
        existing = _SESSIONS.get(key)
        if existing is not None and not existing._closed:
            return existing
        session = AudioCppSession(flattened, owned=True, model_id=model_id)
        _SESSIONS[key] = session
        return session


def close_all_audio_cpp_sessions() -> None:
    with _SESSIONS_LOCK:
        sessions = list({id(session): session for session in _SESSIONS.values()}.values())
        _SESSIONS.clear()
    for session in sessions:
        try:
            session.close()
        except Exception as exc:
            _warn("Failed to close audio.cpp session during shutdown", exc)


atexit.register(close_all_audio_cpp_sessions)


__all__ = [
    "AudioCppRuntimeProxy",
    "AudioCppSession",
    "close_all_audio_cpp_sessions",
    "get_audio_cpp_session",
]
