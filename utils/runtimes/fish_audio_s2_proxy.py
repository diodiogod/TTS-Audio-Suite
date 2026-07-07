from __future__ import annotations

import tempfile
import uuid
import os
import sys
import weakref
from pathlib import Path

import torch

from .bootstrap import PROJECT_ROOT
from .protocol import RuntimeJobRequest
from .session import JsonLineWorkerSession


class FishAudioS2Proxy:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if str(config.device) == "auto" and torch.cuda.is_available() else str(config.device)
        if self.device == "auto":
            self.device = "cpu"
        self.model = self
        self.parent = None
        self.currently_used = True
        self.model_options = {}
        self.model_keys = set()
        self.load_device = self.current_loaded_device()
        self.offload_device = torch.device("cpu")
        self.model_variant = self.config.additional_params.get(
            "model_variant", self.config.model_name or "s2-pro"
        )
        self.quantization = self.config.additional_params.get("quantization", "none")
        memory_estimates_gb = {
            "s2-pro": 16,
            "s2-pro-fp8": 12,
        }
        if self.model_variant == "s2-pro" and self.quantization == "bnb_int8":
            estimate_gb = 13
        elif self.model_variant == "s2-pro" and self.quantization == "bnb_nf4":
            estimate_gb = 10
        else:
            estimate_gb = memory_estimates_gb.get(self.model_variant, 16)
        self._estimated_memory_size = estimate_gb * 1024**3
        self._comfy_loaded_model = None
        self._comfy_model_management = None
        self._initialized = False
        worker_env = os.environ.copy()
        worker_env.setdefault("PYTHONUTF8", "1")
        self._session = JsonLineWorkerSession(
            python_path=sys.executable,
            worker_script=str(PROJECT_ROOT / "utils/runtimes/workers/fish_audio_s2_worker.py"),
            env=worker_env,
        )
        self._initialize_remote_engine()

    def _initialize_remote_engine(self):
        response = self._request("initialize", {
            "model_path": self.config.model_path, "device": self.device,
            "model_variant": self.model_variant,
            "quantization": self.quantization,
            "precision": self.config.additional_params.get("precision", "bfloat16"),
            "compile": bool(self.config.additional_params.get("compile", False)),
            "context_length": int(self.config.additional_params.get("context_length", 8192)),
        })
        self.sample_rate = int((response.result or {}).get("sample_rate", 44100))
        self._initialized = True
        self._register_with_comfy_model_management()

    def _register_with_comfy_model_management(self):
        try:
            import comfy.model_management as model_management
        except Exception:
            return
        if not hasattr(model_management, "LoadedModel") or not hasattr(model_management, "current_loaded_models"):
            return
        try:
            loaded_model = model_management.LoadedModel(self)
            loaded_model.real_model = weakref.ref(self)
            finalizer = model_management.cleanup_models if hasattr(model_management, "cleanup_models") else lambda: None
            loaded_model.model_finalizer = weakref.finalize(self, finalizer)
            loaded_model._tts_wrapper_ref = self
            model_management.current_loaded_models.insert(0, loaded_model)
            self._comfy_loaded_model = loaded_model
            self._comfy_model_management = model_management
        except Exception as exc:
            print(f"⚠️ Failed to register Fish Audio S2 runtime with ComfyUI model management: {exc}")

    def _unregister_from_comfy_model_management(self):
        model_management = self._comfy_model_management
        loaded_model = self._comfy_loaded_model
        if model_management is None or loaded_model is None:
            return
        try:
            if loaded_model in model_management.current_loaded_models:
                model_management.current_loaded_models.remove(loaded_model)
        except Exception as exc:
            print(f"⚠️ Failed to remove Fish Audio S2 runtime from ComfyUI tracking: {exc}")
        finally:
            self._comfy_loaded_model = None
            self._comfy_model_management = None

    def _request(self, action, payload):
        response = self._session.request(RuntimeJobRequest(
            engine_name="fish_audio_s2", action=action,
            model_name=f"{self.model_variant}:{self.quantization}",
            device=self.device, runtime_profile=None,
            payload=payload, request_id=str(uuid.uuid4()),
        ))
        if not response.ok:
            raise RuntimeError(response.error or f"Fish S2 {action} failed")
        return response

    def generate(self, **payload):
        if not self._initialized:
            self._initialize_remote_engine()
        with tempfile.TemporaryDirectory(prefix="tts_fish_s2_") as temp_dir:
            output = Path(temp_dir) / "result.pt"
            payload["output_path"] = str(output)
            self._request("generate", payload)
            result = torch.load(output, map_location="cpu")
            if result.get("peak_memory_gb"):
                label = (
                    self.model_variant
                    if self.quantization == "none"
                    else f"{self.model_variant}+{self.quantization}"
                )
                print(
                    f"Fish Audio S2 {label}: "
                    f"peak GPU memory {result['peak_memory_gb']:.2f} GB"
                )
            return result["audio"].float(), int(result["sample_rate"])

    def to(self, device):
        self.device = str(device)
        return self

    def model_size(self): return self._estimated_memory_size
    def loaded_size(self): return self._estimated_memory_size
    def model_memory(self): return self._estimated_memory_size
    def model_dtype(self): return torch.bfloat16
    def current_loaded_device(self):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    def model_patches_models(self): return ()
    def model_patches_to(self, target): return None
    def is_dynamic(self): return False
    def partially_load(self, device, extra_memory, force_patch_weights=False): return 0
    def partially_unload_ram(self, ram_to_unload): return self.partially_unload("cpu", ram_to_unload)
    def partially_unload(self, device, memory_to_free):
        self.cleanup(unregister=False)
        return self._estimated_memory_size
    def model_unload(self, *args, **kwargs): self.cleanup(unregister=False); return True
    def detach(self, unpatch_weights=True): self.cleanup(unregister=False)
    def is_clone(self, other): return other is self

    def cleanup(self, unregister=True):
        if unregister:
            self._unregister_from_comfy_model_management()
        else:
            self._comfy_loaded_model = None
            self._comfy_model_management = None
        self._initialized = False
        if getattr(self, "_session", None):
            self._session.close()

    def close(self):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


def build_fish_audio_s2_proxy(config):
    return FishAudioS2Proxy(config)
