"""Audio8 TTS proxy for the shared official Transformers 4 runtime."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import uuid
import weakref
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from utils.models.factory_config import ModelLoadConfig
from .bootstrap import PROJECT_ROOT, ensure_runtime
from .launcher import IsolatedRuntimeLauncher
from .profiles import RuntimeProfile, get_runtime_profile
from .protocol import RuntimeJobRequest
from .session import JsonLineWorkerSession


_INTERRUPT_EVENTS: set[threading.Event] = set()
_INTERRUPT_EVENTS_LOCK = threading.Lock()
_INTERRUPT_DISPATCH_LOCK = threading.Lock()


def _install_comfy_interrupt_dispatcher() -> None:
    """Forward Comfy's server interrupt edge to active Audio8 requests."""
    server_module = sys.modules.get("server")
    comfy_nodes = getattr(server_module, "nodes", None)
    if comfy_nodes is None:
        comfy_nodes = sys.modules.get("nodes")
    if comfy_nodes is None:
        return

    with _INTERRUPT_DISPATCH_LOCK:
        current_interrupt = getattr(comfy_nodes, "interrupt_processing", None)
        if not callable(current_interrupt):
            return
        if getattr(current_interrupt, "_tts_audio8_dispatcher", False):
            if (
                getattr(current_interrupt, "_tts_audio8_events", None)
                is _INTERRUPT_EVENTS
            ):
                return
            current_interrupt = getattr(
                current_interrupt,
                "_tts_audio8_original_interrupt",
                current_interrupt,
            )

        def dispatch_interrupt(value=True):
            # TTS Audio Suite patch: preserve Comfy's interrupt edge for
            # blocking Audio8 subprocess requests without replacing the
            # process-wide callback separately for every generation.
            result = current_interrupt(value)
            if value:
                with _INTERRUPT_EVENTS_LOCK:
                    active_events = tuple(_INTERRUPT_EVENTS)
                for active_event in active_events:
                    active_event.set()
            return result

        dispatch_interrupt._tts_audio8_dispatcher = True
        dispatch_interrupt._tts_audio8_original_interrupt = current_interrupt
        dispatch_interrupt._tts_audio8_events = _INTERRUPT_EVENTS
        comfy_nodes.interrupt_processing = dispatch_interrupt


def _register_interrupt_event(interrupt_event: threading.Event) -> None:
    _install_comfy_interrupt_dispatcher()
    with _INTERRUPT_EVENTS_LOCK:
        _INTERRUPT_EVENTS.add(interrupt_event)


def _unregister_interrupt_event(interrupt_event: threading.Event) -> None:
    with _INTERRUPT_EVENTS_LOCK:
        _INTERRUPT_EVENTS.discard(interrupt_event)


class Audio8TTSIsolatedProxy:
    """Expose Audio8's engine API while inference runs under Transformers 4."""

    SAMPLE_RATE = 44100

    def __init__(self, config: ModelLoadConfig, profile: RuntimeProfile):
        self.config = config
        self.profile = profile
        self.model_name = config.model_name or "Audio8-TTS-Preview-0.6b"
        self.model_path = config.model_path
        self.device = str(config.device)
        self.current_device = self.device
        self.dtype_name = str((config.additional_params or {}).get("dtype", "auto"))
        self.dtype = self._resolve_dtype(self.dtype_name)
        self.load_device = self.current_loaded_device()
        self.offload_device = torch.device("cpu")
        self.currently_used = True
        self.model = self
        self.engine = self
        self.parent = None
        self.model_options = {}
        self.model_keys = set()
        estimate_gib = (
            3.5
            if self.device.startswith("cpu") or self.dtype == torch.float32
            else 1.75
        )
        self._estimated_memory_size = int(estimate_gib * 1024**3)
        self._comfy_loaded_model = None
        self._comfy_model_management = None
        self._initialized = False

        launcher = IsolatedRuntimeLauncher(runtime_root=str(PROJECT_ROOT))
        python_path = ensure_runtime(profile)
        worker_script = PROJECT_ROOT / "utils/runtimes/workers/audio8_tts_worker.py"
        self._session = JsonLineWorkerSession(
            python_path=str(python_path),
            worker_script=str(worker_script),
            env=launcher.build_env(profile),
        )
        try:
            self._initialize_remote_engine()
        except Exception:
            self._session.close()
            raise
        self._register_with_comfy_model_management()

    def _resolve_dtype(self, dtype_name: str) -> torch.dtype:
        normalized = str(dtype_name or "auto").lower()
        if not self.device.startswith("cuda"):
            return torch.float32
        if normalized == "float32":
            return torch.float32
        if normalized == "float16":
            return torch.float16
        if normalized == "bfloat16":
            return torch.bfloat16
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability(torch.device(self.device))
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float16

    def _request(self, action: str, payload: Dict[str, Any]):
        response = self._session.request(
            RuntimeJobRequest(
                engine_name="audio8_tts",
                action=action,
                model_name=self.model_name,
                device=self.device,
                runtime_profile=self.profile.name,
                payload=payload,
                request_id=str(uuid.uuid4()),
            )
        )
        if not response.ok:
            details = response.error or (
                f"Audio8 TTS isolated action '{action}' failed"
            )
            if response.logs:
                details = f"{details}\n" + "\n".join(response.logs)
            raise RuntimeError(details)
        return response

    def _request_generation(self, payload: Dict[str, Any]):
        """Run generation while the main process monitors Comfy interruption."""
        try:
            import comfy.model_management as model_management
        except Exception:
            return self._request("generate", payload)

        checker = getattr(
            model_management,
            "throw_exception_if_processing_interrupted",
            None,
        )
        if not callable(checker):
            return self._request("generate", payload)

        interrupt_event = threading.Event()
        _register_interrupt_event(interrupt_event)
        result: Dict[str, Any] = {}
        request_thread = None
        request_started = False

        def run_request() -> None:
            try:
                result["response"] = self._request("generate", payload)
            except BaseException as exc:
                result["error"] = exc

        try:
            checker()
            request_thread = threading.Thread(
                target=run_request,
                name="audio8-tts-worker-request",
                daemon=True,
            )
            request_thread.start()
            request_started = True

            while request_thread.is_alive():
                request_thread.join(timeout=0.05)
                if interrupt_event.is_set():
                    raise model_management.InterruptProcessingException()
                checker()
        except BaseException as exc:
            if request_started:
                self._initialized = False
                try:
                    self._session.terminate_now()
                except Exception:
                    pass
                request_thread.join(timeout=5)
            if isinstance(
                exc,
                model_management.InterruptProcessingException,
            ):
                interrupt_setter = getattr(
                    model_management,
                    "interrupt_current_processing",
                    None,
                )
                if callable(interrupt_setter):
                    interrupt_setter(False)
            raise
        finally:
            _unregister_interrupt_event(interrupt_event)

        if "error" in result:
            raise result["error"]
        return result["response"]

    def _initialize_remote_engine(self) -> None:
        self._request(
            "initialize",
            {
                "model_path": self.model_path,
                "dtype": self.dtype_name,
            },
        )
        self._initialized = True
        print(f"✅ Audio8 TTS isolated runtime ready ({self.profile.name})")

    def _ensure_remote_engine(self) -> None:
        process = getattr(self._session, "_process", None)
        if not self._initialized or process is None or process.poll() is not None:
            self._initialize_remote_engine()

    @staticmethod
    def _serialize_reference_audio(
        reference_audio: Any,
        bundle_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        if reference_audio is None:
            return None
        if isinstance(reference_audio, (str, os.PathLike)):
            return {
                "kind": "audio_path",
                "audio_path": os.fspath(reference_audio),
            }

        waveform = None
        sample_rate = None
        if isinstance(reference_audio, dict):
            waveform = reference_audio.get(
                "waveform",
                reference_audio.get("array"),
            )
            sample_rate = reference_audio.get(
                "sample_rate",
                reference_audio.get("sampling_rate"),
            )
        elif isinstance(reference_audio, (tuple, list)) and len(reference_audio) == 2:
            waveform, sample_rate = reference_audio

        if waveform is None or sample_rate is None:
            raise TypeError(
                "Audio8 TTS isolated reference audio must be a path, "
                "audio dict, or (waveform, sample_rate) pair"
            )

        tensor_path = bundle_dir / "reference_audio.pt"
        waveform_tensor = torch.as_tensor(waveform).detach().float().cpu()
        torch.save(
            {
                "waveform": waveform_tensor,
                "sample_rate": int(sample_rate),
            },
            tensor_path,
        )
        return {
            "kind": "tensor_path",
            "tensor_path": str(tensor_path),
        }

    def generate(
        self,
        text: str,
        reference_audio: Any = None,
        reference_text: Optional[str] = None,
        *,
        max_new_tokens: int = 1024,
        retry_max_new_tokens: int = 2000,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        seed: int = 42,
    ) -> torch.Tensor:
        self._ensure_remote_engine()
        with tempfile.TemporaryDirectory(prefix="tts_audio8_iso_") as temp_dir:
            bundle_dir = Path(temp_dir)
            output_path = bundle_dir / "result.pt"
            serialized_reference = self._serialize_reference_audio(
                reference_audio,
                bundle_dir,
            )
            self._request_generation(
                {
                    "text": text,
                    "reference_audio": serialized_reference,
                    "reference_text": reference_text,
                    "max_new_tokens": int(max_new_tokens),
                    "retry_max_new_tokens": int(retry_max_new_tokens),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "do_sample": bool(do_sample),
                    "seed": int(seed),
                    "output_path": str(output_path),
                }
            )
            if not output_path.exists():
                raise RuntimeError(
                    "Audio8 TTS isolated worker returned no output payload"
                )
            result = torch.load(output_path, map_location="cpu")
            audio = torch.as_tensor(result["audio"]).detach().float().cpu()
            sample_rate = int(result["sample_rate"])
            if sample_rate != self.SAMPLE_RATE:
                raise RuntimeError(
                    f"Audio8 TTS worker returned {sample_rate} Hz; "
                    f"expected {self.SAMPLE_RATE} Hz"
                )
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            if audio.ndim != 2 or audio.shape[0] != 1:
                raise RuntimeError(
                    "Audio8 TTS worker returned invalid audio shape "
                    f"{tuple(audio.shape)}"
                )
            return audio.contiguous()

    def _register_with_comfy_model_management(self) -> None:
        self._ensure_comfy_model_registration()

    def _ensure_comfy_model_registration(self) -> None:
        try:
            import comfy.model_management as model_management
        except Exception:
            return
        if not hasattr(model_management, "LoadedModel") or not hasattr(
            model_management,
            "current_loaded_models",
        ):
            return
        for loaded_model in model_management.current_loaded_models:
            wrapper = getattr(loaded_model, "_tts_wrapper_ref", None)
            real_model = getattr(loaded_model, "real_model", None)
            resolved_real_model = (
                real_model()
                if isinstance(real_model, weakref.ReferenceType)
                else real_model
            )
            if wrapper is self or resolved_real_model is self:
                self._comfy_loaded_model = loaded_model
                self._comfy_model_management = model_management
                return
        try:
            loaded_model = model_management.LoadedModel(self)
            loaded_model.real_model = weakref.ref(self)
            finalizer = getattr(
                model_management,
                "cleanup_models",
                lambda: None,
            )
            loaded_model.model_finalizer = weakref.finalize(self, finalizer)
            loaded_model._tts_wrapper_ref = self
            model_management.current_loaded_models.insert(0, loaded_model)
            self._comfy_loaded_model = loaded_model
            self._comfy_model_management = model_management
        except Exception as exc:
            print(
                "⚠️ Failed to register isolated Audio8 TTS runtime with "
                f"ComfyUI model management: {exc}"
            )

    def _unregister_from_comfy_model_management(self) -> None:
        model_management = self._comfy_model_management
        loaded_model = self._comfy_loaded_model
        if model_management is None or loaded_model is None:
            return
        try:
            if loaded_model in model_management.current_loaded_models:
                model_management.current_loaded_models.remove(loaded_model)
        except Exception as exc:
            print(
                "⚠️ Failed to remove isolated Audio8 TTS runtime from "
                f"ComfyUI tracking: {exc}"
            )
        finally:
            self._comfy_loaded_model = None
            self._comfy_model_management = None

    def to(self, device):
        self.device = str(device)
        self.current_device = self.device
        self.load_device = self.current_loaded_device()
        return self

    def eval(self):
        return self

    def loaded_size(self) -> int:
        if self._initialized and self.device.startswith("cuda"):
            return self._estimated_memory_size
        return 0

    def model_size(self) -> int:
        return self._estimated_memory_size

    def model_memory(self) -> int:
        return self._estimated_memory_size

    def get_ram_usage(self) -> int:
        return self._estimated_memory_size

    def model_offloaded_memory(self) -> int:
        return (
            0
            if self._initialized and self.device.startswith("cuda")
            else self._estimated_memory_size
        )

    def model_mmap_residency(self, free: bool = False) -> tuple[int, int]:
        del free
        return 0, self._estimated_memory_size

    def pinned_memory_size(self) -> int:
        return 0

    def lowvram_patch_counter(self) -> int:
        return 0

    def model_dtype(self):
        return self.dtype

    def model_patches_models(self):
        return ()

    def model_patches_to(self, target) -> None:
        if isinstance(target, torch.dtype):
            self.dtype = target
        elif isinstance(target, torch.device):
            self.device = str(target)

    def is_dynamic(self) -> bool:
        return False

    def current_loaded_device(self) -> torch.device:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(self.device)
        return torch.device("cpu")

    def partially_load(
        self,
        device,
        extra_memory,
        force_patch_weights=False,
    ) -> int:
        del extra_memory, force_patch_weights
        self.to(device)
        return 0

    def partially_unload_ram(self, ram_to_unload: int) -> int:
        return self.partially_unload("cpu", ram_to_unload)

    def partially_unload(self, device, memory_to_free) -> int:
        del device, memory_to_free
        freed = self.loaded_size()
        self.cleanup(unregister=False)
        return freed

    def model_unload(
        self,
        memory_to_free=None,
        unpatch_weights=True,
    ) -> bool:
        del memory_to_free, unpatch_weights
        self.cleanup(unregister=False)
        return True

    def detach(self, unpatch_weights=True) -> None:
        del unpatch_weights
        self.cleanup(unregister=False)

    def is_clone(self, other: Any) -> bool:
        return other is self

    def cleanup(self, unregister: bool = True) -> None:
        if unregister:
            self._unregister_from_comfy_model_management()
        else:
            self._comfy_loaded_model = None
            self._comfy_model_management = None
        self._initialized = False
        if getattr(self, "_session", None) is not None:
            self._session.close()

    close = cleanup

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


def build_audio8_tts_isolated_proxy(
    config: ModelLoadConfig,
) -> Audio8TTSIsolatedProxy:
    profile_name = config.runtime_profile or "vibevoice_transformers4_shared"
    profile = get_runtime_profile(profile_name)
    if profile is None:
        raise RuntimeError(
            f"Unknown isolated runtime profile '{profile_name}' for Audio8 TTS"
        )
    if profile.name != "vibevoice_transformers4_shared":
        raise RuntimeError(
            "Audio8 TTS requires the shared Transformers 4 runtime profile "
            "'vibevoice_transformers4_shared'"
        )
    config.runtime_profile = profile.name
    return Audio8TTSIsolatedProxy(config=config, profile=profile)
