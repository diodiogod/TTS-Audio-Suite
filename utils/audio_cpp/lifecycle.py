"""ComfyUI model-management proxy for suite-owned audio.cpp processes."""

from __future__ import annotations

import os
import sys
import threading
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

if TYPE_CHECKING:
    from .session import AudioCppSession


def _first(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _warn(message: str, exc: Optional[BaseException] = None) -> None:
    text = f"WARNING: {message}"
    if exc is not None:
        text += f": {exc}"
    encoding = getattr(sys.stderr, "encoding", None) or "ascii"
    try:
        text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except LookupError:
        text = text.encode("ascii", errors="replace").decode("ascii")
    print(text, file=sys.stderr)


class AudioCppRuntimeProxy:
    """ComfyUI model-like resource representing one owned native server."""

    def __init__(self, session: "AudioCppSession") -> None:
        self._session_ref = weakref.ref(session)
        self.model = self
        self.processor = self
        self.parent = None
        self.currently_used = True
        self.model_options: Dict[str, Any] = {}
        self.model_keys = set()
        self.offload_device = self._torch_device("cpu", 0)
        backend = str(session.config.get("backend", "cuda")).lower()
        device_index = self._device_index(
            session.config.get("device_index", session.config.get("device", 0))
        )
        self.load_device = self._lifecycle_device(backend, device_index)
        self.current_device = self.load_device
        self._estimated_memory_size = self._memory_estimate(session.config)
        self._loaded_model = None
        self._model_management = None
        self._registration_lock = threading.RLock()

    @staticmethod
    def _device_index(value: Any) -> int:
        text = str(value or 0).lower()
        if ":" in text:
            text = text.rsplit(":", 1)[-1]
        try:
            return max(0, int(text))
        except ValueError:
            return 0

    @staticmethod
    def _torch_device(kind: str, index: int):
        try:
            import torch

            return torch.device(kind, index) if kind == "cuda" else torch.device(kind)
        except Exception:
            return f"{kind}:{index}" if kind == "cuda" else kind

    @classmethod
    def _lifecycle_device(cls, backend: str, index: int):
        if backend == "cpu":
            return cls._torch_device("cpu", 0)
        try:
            import comfy.model_management as model_management

            device = model_management.get_torch_device()
            if device is not None:
                return device
        except (ImportError, AttributeError, RuntimeError):
            pass
        if backend == "metal":
            return cls._torch_device("mps", 0)
        return cls._torch_device("cuda", index)

    @staticmethod
    def _memory_estimate(config: Mapping[str, Any]) -> int:
        explicit = _first(config, "estimated_memory_bytes", "model_memory_bytes")
        if explicit is not None:
            return max(1, int(explicit))
        gigabytes = _first(config, "estimated_vram_gb", "model_memory_gb")
        if gigabytes is not None:
            return max(1, int(float(gigabytes) * 1024**3))

        raw_path = _first(config, "model_path", "package_path", "gguf_path")
        if raw_path:
            try:
                model_path = Path(
                    os.path.expandvars(os.path.expanduser(str(raw_path)))
                ).resolve()
                if model_path.is_file():
                    return max(1, model_path.stat().st_size)
                if model_path.is_dir():
                    total = 0
                    for candidate in model_path.rglob("*"):
                        try:
                            if candidate.is_file():
                                total += candidate.stat().st_size
                        except OSError:
                            continue
                    return max(1, total)
            except OSError:
                pass
        # A zero-size entry is ignored by parts of ComfyUI's unload ordering.
        return 1

    def _session(self) -> Optional["AudioCppSession"]:
        return self._session_ref()

    def register(self) -> bool:
        """Register once with ComfyUI after an owned process becomes live."""
        with self._registration_lock:
            if self._loaded_model is not None:
                current = getattr(self._model_management, "current_loaded_models", None)
                if isinstance(current, list) and self._loaded_model in current:
                    return True
                self._loaded_model = None
                self._model_management = None
            try:
                import comfy.model_management as model_management
            except ImportError:
                return False
            try:
                loaded_model_type = getattr(model_management, "LoadedModel", None)
                current_models = getattr(model_management, "current_loaded_models", None)
                if not callable(loaded_model_type) or not isinstance(current_models, list):
                    return False
                loaded_model = loaded_model_type(self)
                loaded_model.real_model = weakref.ref(self)
                cleanup = getattr(model_management, "cleanup_models", None)
                loaded_model.model_finalizer = weakref.finalize(
                    self, cleanup if callable(cleanup) else lambda: None
                )
                loaded_model._tts_wrapper_ref = self
                current_models.insert(0, loaded_model)
                self._loaded_model = loaded_model
                self._model_management = model_management
                return True
            except Exception as exc:
                _warn("Failed to register audio.cpp runtime with ComfyUI", exc)
                return False

    def unregister(self) -> None:
        with self._registration_lock:
            loaded_model = self._loaded_model
            model_management = self._model_management
            self._loaded_model = None
            self._model_management = None
            if loaded_model is None or model_management is None:
                return
            current_models = getattr(model_management, "current_loaded_models", None)
            if not isinstance(current_models, list):
                return
            try:
                for candidate in list(current_models):
                    if candidate is loaded_model or getattr(candidate, "_tts_wrapper_ref", None) is self:
                        current_models.remove(candidate)
            except Exception as exc:
                _warn("Failed to unregister audio.cpp runtime from ComfyUI", exc)

    def to(self, device):
        self.current_device = device
        return self

    def eval(self):
        return self

    def model_size(self) -> int:
        return self._estimated_memory_size

    def loaded_size(self) -> int:
        session = self._session()
        return self._estimated_memory_size if session is not None and session.running else 0

    def model_memory(self) -> int:
        return self.model_size()

    def get_ram_usage(self) -> int:
        return self._estimated_memory_size

    def model_offloaded_memory(self) -> int:
        return max(0, self.model_size() - self.loaded_size())

    def model_mmap_residency(self, free: bool = False) -> tuple[int, int]:
        return 0, self._estimated_memory_size

    def pinned_memory_size(self) -> int:
        return 0

    def lowvram_patch_counter(self) -> int:
        return 0

    def model_dtype(self):
        try:
            import torch

            return torch.float32
        except Exception:
            return None

    def current_loaded_device(self):
        return self.current_device

    def model_patches_models(self):
        return ()

    def model_patches_to(self, target) -> None:
        try:
            import torch

            if isinstance(target, torch.device):
                self.current_device = target
        except Exception:
            pass

    def is_dynamic(self) -> bool:
        return False

    def is_clone(self, other) -> bool:
        return other is self

    def clone_has_same_weights(self, other) -> bool:
        return other is self

    def partially_load(self, device, extra_memory, force_patch_weights=False) -> int:
        self.current_device = device
        return 0

    def partially_unload(self, device, memory_to_free) -> int:
        # audio.cpp 0.5.1 cannot release part of a session. Claiming memory here
        # would make ComfyUI believe VRAM was freed while the server still owns it.
        return 0

    def partially_unload_ram(self, ram_to_unload) -> int:
        return 0

    def patch_model(
        self,
        device_to=None,
        lowvram_model_memory=0,
        load_weights=True,
        force_patch_weights=False,
    ):
        if device_to is not None:
            self.current_device = device_to
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if device_to is not None:
            self.current_device = device_to
        session = self._session()
        if session is not None:
            # LoadedModel removes its list entry after this callback returns.
            session._stop_owned_runtime(unregister=False)
        return self.model

    def model_unload(self, memory_to_free=None, unpatch_weights=True) -> bool:
        self.unpatch_model(self.offload_device, unpatch_weights=unpatch_weights)
        return True

    def detach(self, unpatch_weights=True):
        return self.unpatch_model(self.offload_device, unpatch_weights=unpatch_weights)

    def cleanup(self) -> None:
        session = self._session()
        if session is not None:
            session._stop_owned_runtime(unregister=True)


__all__ = ["AudioCppRuntimeProxy"]
