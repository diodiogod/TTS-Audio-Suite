from __future__ import annotations

"""
Qwen3-ASR isolated runtime proxy.
"""

import tempfile
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from utils.models.factory_config import ModelLoadConfig
from .bootstrap import PROJECT_ROOT, ensure_runtime
from .launcher import IsolatedRuntimeLauncher
from .profiles import RuntimeProfile, get_runtime_profile
from .protocol import RuntimeJobRequest
from .session import JsonLineWorkerSession


@dataclass(frozen=True)
class ForcedAlignItem:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class ForcedAlignResult:
    items: List[ForcedAlignItem]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> ForcedAlignItem:
        return self.items[idx]


@dataclass
class ASRTranscriptionResult:
    language: str
    text: str
    time_stamps: Optional[ForcedAlignResult] = None


class Qwen3ASRIsolatedProxy:
    def __init__(self, config: ModelLoadConfig, profile: RuntimeProfile):
        self.config = config
        self.profile = profile
        self.current_model_name = config.model_name
        self.current_model_path = config.model_path
        self.device = config.device
        self.current_device = config.device
        self.load_device = self.current_loaded_device()
        self.offload_device = "cpu"
        self.dtype = self._resolve_dtype(config.additional_params.get("precision", "auto"))
        self.model = self
        self.processor = self
        self.parent = None
        self.currently_used = True
        self.model_options = {}
        self.model_keys = set()
        self._estimated_memory_size = 8 * 1024 * 1024 * 1024
        self._comfy_loaded_model = None
        self._comfy_model_management = None
        self._initialized = False

        launcher = IsolatedRuntimeLauncher(runtime_root=str(PROJECT_ROOT))
        python_path = ensure_runtime(profile)
        worker_script = str(PROJECT_ROOT / "utils" / "runtimes" / "workers" / "qwen3_asr_worker.py")

        self._session = JsonLineWorkerSession(
            python_path=str(python_path),
            worker_script=worker_script,
            env=launcher.build_env(profile),
        )
        self._register_with_comfy_model_management()
        self._initialize_remote_engine()

    def _resolve_dtype(self, dtype_name: str):
        dtype_name = (dtype_name or "auto").lower()
        if dtype_name in ("bf16", "bfloat16"):
            return torch.bfloat16
        if dtype_name in ("fp32", "float32"):
            return torch.float32
        return torch.float16

    def _register_with_comfy_model_management(self) -> None:
        try:
            import comfy.model_management as model_management
        except Exception:
            return

        if not hasattr(model_management, "LoadedModel") or not hasattr(model_management, "current_loaded_models"):
            return

        try:
            loaded_model = model_management.LoadedModel(self)
            loaded_model.real_model = weakref.ref(self)
            if hasattr(model_management, "cleanup_models"):
                loaded_model.model_finalizer = weakref.finalize(self, model_management.cleanup_models)
            else:
                loaded_model.model_finalizer = weakref.finalize(self, lambda: None)
            loaded_model._tts_wrapper_ref = self
            model_management.current_loaded_models.insert(0, loaded_model)
            self._comfy_loaded_model = loaded_model
            self._comfy_model_management = model_management
        except Exception as e:
            print(f"⚠️ Failed to register isolated Qwen3-ASR runtime with ComfyUI model management: {e}")

    def _unregister_from_comfy_model_management(self) -> None:
        model_management = self._comfy_model_management
        loaded_model = self._comfy_loaded_model
        if model_management is None or loaded_model is None:
            return

        try:
            if hasattr(model_management, "current_loaded_models") and loaded_model in model_management.current_loaded_models:
                model_management.current_loaded_models.remove(loaded_model)
        except Exception as e:
            print(f"⚠️ Failed to remove isolated Qwen3-ASR runtime from ComfyUI tracking: {e}")
        finally:
            self._comfy_loaded_model = None
            self._comfy_model_management = None

    def _initialize_remote_engine(self) -> None:
        response = self._session.request(
            RuntimeJobRequest(
                engine_name="qwen3_asr",
                action="initialize",
                model_name=self.current_model_name,
                device=str(self.device),
                runtime_profile=self.profile.name,
                payload={
                    "precision": self.config.additional_params.get("precision", "auto"),
                    "attn_implementation": self.config.additional_params.get("attn_implementation", "sdpa"),
                    "max_new_tokens": self.config.additional_params.get("max_new_tokens", 256),
                    "forced_aligner": self.config.additional_params.get("forced_aligner"),
                    "forced_aligner_kwargs": self.config.additional_params.get("forced_aligner_kwargs"),
                    "model_path": self.current_model_path,
                },
                request_id=str(uuid.uuid4()),
            )
        )
        if not response.ok:
            details = response.error or "Failed to initialize isolated Qwen3-ASR runtime"
            if response.logs:
                details = f"{details}\n" + "\n".join(response.logs)
            raise RuntimeError(details)

        self._initialized = True
        print(f"✅ Qwen3-ASR isolated runtime ready ({self.profile.name})")

    def _ensure_remote_engine(self) -> None:
        process = getattr(self._session, "_process", None)
        if not self._initialized or process is None or process.poll() is not None:
            self._initialize_remote_engine()

    def _serialize_audio_payload(self, audio: Any, bundle_dir: Path) -> Dict[str, Any]:
        items = audio if isinstance(audio, list) else [audio]
        serialized_items = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                serialized_items.append({"kind": "audio_path", "audio_path": item})
                continue

            if isinstance(item, tuple) and len(item) == 2:
                waveform, sample_rate = item
                tensor_path = bundle_dir / f"audio_{index}.pt"
                if isinstance(waveform, np.ndarray):
                    waveform = torch.from_numpy(waveform)
                elif not isinstance(waveform, torch.Tensor):
                    waveform = torch.tensor(waveform)
                torch.save(
                    {
                        "waveform": waveform.detach().cpu().float(),
                        "sample_rate": int(sample_rate),
                    },
                    tensor_path,
                )
                serialized_items.append({"kind": "tensor_path", "tensor_path": str(tensor_path)})
                continue

            raise TypeError(f"Unsupported Qwen3-ASR isolated audio item type: {type(item)}")

        return {"items": serialized_items}

    def _deserialize_results(self, payload: List[Dict[str, Any]]) -> List[ASRTranscriptionResult]:
        results = []
        for item in payload:
            time_stamps = None
            raw_time_stamps = item.get("time_stamps")
            if raw_time_stamps:
                time_stamps = ForcedAlignResult(
                    items=[
                        ForcedAlignItem(
                            text=str(ts.get("text", "")),
                            start_time=float(ts.get("start_time", 0.0)),
                            end_time=float(ts.get("end_time", 0.0)),
                        )
                        for ts in raw_time_stamps
                    ]
                )
            results.append(
                ASRTranscriptionResult(
                    language=str(item.get("language", "")),
                    text=str(item.get("text", "")),
                    time_stamps=time_stamps,
                )
            )
        return results

    def transcribe(
        self,
        audio,
        context="",
        language=None,
        return_time_stamps: bool = False,
    ) -> List[ASRTranscriptionResult]:
        self._ensure_remote_engine()
        with tempfile.TemporaryDirectory(prefix="tts_qwen3_asr_iso_") as temp_dir:
            bundle_dir = Path(temp_dir)
            response = self._session.request(
                RuntimeJobRequest(
                    engine_name="qwen3_asr",
                    action="transcribe",
                    model_name=self.current_model_name,
                    device=str(self.device),
                    runtime_profile=self.profile.name,
                    request_id=str(uuid.uuid4()),
                    payload={
                        "audio": self._serialize_audio_payload(audio, bundle_dir),
                        "context": context,
                        "language": language,
                        "return_time_stamps": return_time_stamps,
                    },
                )
            )

            if not response.ok:
                details = response.error or "Isolated Qwen3-ASR transcribe failed"
                if response.logs:
                    details = f"{details}\n" + "\n".join(response.logs)
                raise RuntimeError(details)

            return self._deserialize_results(response.result.get("results", []))

    def to(self, device):
        self.device = str(device)
        self.current_device = self.device
        self.load_device = self.current_loaded_device()
        return self

    def eval(self):
        return self

    def loaded_size(self) -> int:
        return self._estimated_memory_size if str(self.device).startswith("cuda") else 0

    def model_size(self) -> int:
        return self._estimated_memory_size

    def model_memory(self) -> int:
        return self._estimated_memory_size

    def get_ram_usage(self) -> int:
        return self._estimated_memory_size

    def model_offloaded_memory(self) -> int:
        return 0 if str(self.device).startswith("cuda") else self._estimated_memory_size

    def model_mmap_residency(self, free: bool = False) -> tuple[int, int]:
        return 0, self._estimated_memory_size

    def pinned_memory_size(self) -> int:
        return 0

    def lowvram_patch_counter(self) -> int:
        return 0

    def model_dtype(self):
        return self.dtype

    def model_patches_models(self):
        return ()

    def is_dynamic(self) -> bool:
        return False

    def model_patches_to(self, target) -> None:
        if isinstance(target, torch.dtype):
            self.dtype = target
        elif isinstance(target, torch.device):
            self.device = str(target)

    def current_loaded_device(self) -> torch.device:
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def partially_load(self, device: str, extra_memory: int, force_patch_weights: bool = False) -> int:
        self.to(device)
        return 0

    def partially_unload_ram(self, ram_to_unload: int) -> int:
        return self.partially_unload("cpu", ram_to_unload)

    def partially_unload(self, device: str, memory_to_free: int) -> int:
        freed = self._estimated_memory_size if str(self.device).startswith("cuda") else 0
        self.cleanup(unregister=False)
        return freed

    def model_unload(self, memory_to_free: Optional[int] = None, unpatch_weights: bool = True) -> bool:
        self.cleanup(unregister=False)
        return True

    def detach(self, unpatch_weights: bool = True) -> None:
        self.cleanup(unregister=False)

    def is_clone(self, other: Any) -> bool:
        return other is self

    def cleanup(self, unregister: bool = True):
        if unregister:
            self._unregister_from_comfy_model_management()
        self._initialized = False
        self._session.close()

    def close(self):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


def build_qwen3_asr_isolated_proxy(config: ModelLoadConfig) -> Qwen3ASRIsolatedProxy:
    profile_name = config.runtime_profile or "vibevoice_transformers4_shared"
    profile = get_runtime_profile(profile_name)
    if profile is None:
        raise RuntimeError(f"Unknown isolated runtime profile '{profile_name}' for Qwen3-ASR")
    return Qwen3ASRIsolatedProxy(config=config, profile=profile)
