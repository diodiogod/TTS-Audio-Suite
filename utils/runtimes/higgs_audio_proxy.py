from __future__ import annotations

"""
Higgs Audio isolated runtime proxy.
"""

import tempfile
import uuid
import weakref
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from utils.models.factory_config import ModelLoadConfig
from .bootstrap import PROJECT_ROOT, ensure_runtime
from .launcher import IsolatedRuntimeLauncher
from .profiles import RuntimeProfile, get_runtime_profile
from .protocol import RuntimeJobRequest
from .session import JsonLineWorkerSession


class HiggsAudioIsolatedProxy:
    def __init__(self, config: ModelLoadConfig, profile: RuntimeProfile):
        self.config = config
        self.profile = profile
        self.current_model_name = config.model_name
        self.current_model_path = config.model_path
        self.device = config.device
        self.current_device = config.device
        self.load_device = self.current_loaded_device()
        self.offload_device = "cpu"
        self.enable_cuda_graphs = bool(config.additional_params.get("enable_cuda_graphs", True))
        self.currently_used = True
        self.model = self
        self.processor = self
        self.parent = None
        self.model_options = {}
        self.model_keys = set()
        self._estimated_memory_size = 10 * 1024 * 1024 * 1024
        self._comfy_loaded_model = None
        self._comfy_model_management = None
        self._initialized = False
        self.engine = self

        launcher = IsolatedRuntimeLauncher(runtime_root=str(PROJECT_ROOT))
        python_path = ensure_runtime(profile)
        worker_script = str(PROJECT_ROOT / "utils" / "runtimes" / "workers" / "higgs_audio_worker.py")

        self._session = JsonLineWorkerSession(
            python_path=str(python_path),
            worker_script=worker_script,
            env=launcher.build_env(profile),
        )
        self._register_with_comfy_model_management()
        self._initialize_remote_engine()

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
            print(f"⚠️ Failed to register isolated Higgs Audio runtime with ComfyUI model management: {e}")

    def _unregister_from_comfy_model_management(self) -> None:
        model_management = self._comfy_model_management
        loaded_model = self._comfy_loaded_model
        if model_management is None or loaded_model is None:
            return

        try:
            if hasattr(model_management, "current_loaded_models") and loaded_model in model_management.current_loaded_models:
                model_management.current_loaded_models.remove(loaded_model)
        except Exception as e:
            print(f"⚠️ Failed to remove isolated Higgs Audio runtime from ComfyUI tracking: {e}")
        finally:
            self._comfy_loaded_model = None
            self._comfy_model_management = None

    def _initialize_remote_engine(self) -> None:
        response = self._session.request(
            RuntimeJobRequest(
                engine_name="higgs_audio",
                action="initialize",
                model_name=self.current_model_name,
                device=str(self.device),
                runtime_profile=self.profile.name,
                payload={
                    "model_path": self.current_model_path,
                    "tokenizer_path": self.config.additional_params.get("tokenizer_path"),
                    "enable_cuda_graphs": self.enable_cuda_graphs,
                },
                request_id=str(uuid.uuid4()),
            )
        )
        if not response.ok:
            details = response.error or "Failed to initialize isolated Higgs Audio runtime"
            if response.logs:
                details = f"{details}\n" + "\n".join(response.logs)
            raise RuntimeError(details)

        self._initialized = True
        print(f"✅ Higgs Audio isolated runtime ready ({self.profile.name})")

    def _ensure_remote_engine(self) -> None:
        process = getattr(self._session, "_process", None)
        if not self._initialized or process is None or process.poll() is not None:
            self._initialize_remote_engine()

    def _serialize_audio_ref(self, ref_audio: Optional[Dict[str, Any]], bundle_dir: Path, name: str) -> Optional[Dict[str, Any]]:
        if ref_audio is None:
            return None

        if isinstance(ref_audio, str):
            return {"kind": "audio_path", "audio_path": ref_audio}

        if not isinstance(ref_audio, dict):
            raise TypeError(f"Unsupported Higgs Audio isolated ref_audio type: {type(ref_audio)}")

        if "audio_path" in ref_audio and ref_audio["audio_path"]:
            return {"kind": "audio_path", "audio_path": ref_audio["audio_path"]}

        nested_audio = ref_audio.get("audio")
        if isinstance(nested_audio, dict) and "audio_path" in nested_audio and nested_audio["audio_path"]:
            return {"kind": "audio_path", "audio_path": nested_audio["audio_path"]}
        if isinstance(nested_audio, dict) and "waveform" in nested_audio:
            ref_audio = nested_audio

        if "waveform" not in ref_audio:
            raise ValueError("Unsupported Higgs Audio isolated reference audio format")

        tensor_path = bundle_dir / f"{name}.pt"
        waveform = ref_audio["waveform"]
        torch.save(
            {
                "waveform": waveform.detach().cpu() if isinstance(waveform, torch.Tensor) else waveform,
                "sample_rate": int(ref_audio.get("sample_rate", 24000)),
            },
            tensor_path,
        )
        return {"kind": "tensor_path", "tensor_path": str(tensor_path)}

    def _run_generation(self, action: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        self._ensure_remote_engine()
        with tempfile.TemporaryDirectory(prefix="tts_higgs_iso_") as temp_dir:
            bundle_dir = Path(temp_dir)
            output_path = bundle_dir / "result.pt"

            for key, stem in (
                ("reference_audio", "reference_audio"),
                ("primary_reference_audio", "primary_reference_audio"),
                ("secondary_reference_audio", "secondary_reference_audio"),
            ):
                ref_audio = payload.get(key)
                if ref_audio is not None:
                    payload[key] = self._serialize_audio_ref(ref_audio, bundle_dir, stem)

            payload["output_path"] = str(output_path)

            response = self._session.request(
                RuntimeJobRequest(
                    engine_name="higgs_audio",
                    action=action,
                    model_name=self.current_model_name,
                    device=str(self.device),
                    runtime_profile=self.profile.name,
                    request_id=str(uuid.uuid4()),
                    payload=payload,
                )
            )

            if not response.ok:
                details = response.error or f"Isolated Higgs Audio action '{action}' failed"
                if response.logs:
                    details = f"{details}\n" + "\n".join(response.logs)
                raise RuntimeError(details)
            if not output_path.exists():
                raise RuntimeError("Isolated Higgs Audio worker returned no output payload")

            result = torch.load(output_path, map_location="cpu")
            return result["audio_result"], result["generation_info"]

    def generate_stateless(self, **kwargs) -> Tuple[Dict[str, Any], str]:
        return self._run_generation("generate_stateless", kwargs)

    def generate_native_multispeaker_stateless(self, **kwargs) -> Tuple[Dict[str, Any], str]:
        return self._run_generation("generate_native_multispeaker_stateless", kwargs)

    def get_available_models(self):
        return [self.current_model_name]

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
        return torch.float16

    def model_patches_models(self):
        return ()

    def is_dynamic(self) -> bool:
        return False

    def model_patches_to(self, target) -> None:
        if isinstance(target, torch.device):
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


def build_higgs_audio_isolated_proxy(config: ModelLoadConfig) -> HiggsAudioIsolatedProxy:
    profile_name = config.runtime_profile or "vibevoice_transformers4_shared"
    profile = get_runtime_profile(profile_name)
    if profile is None:
        raise RuntimeError(f"Unknown isolated runtime profile '{profile_name}' for Higgs Audio")
    return HiggsAudioIsolatedProxy(config=config, profile=profile)
