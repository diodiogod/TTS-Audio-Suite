from __future__ import annotations

"""ComfyUI-side proxy for TADA in the shared Transformers 4 runtime."""

import os
import tempfile
import uuid
import weakref
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from engines.tada.languages import validate_tada_model_language
from engines.tada.tada_downloader import TadaDownloader
from utils.device import resolve_torch_device
from .bootstrap import PROJECT_ROOT, ensure_runtime
from .launcher import IsolatedRuntimeLauncher
from .profiles import RuntimeProfile, get_runtime_profile
from .protocol import RuntimeJobRequest
from .session import JsonLineWorkerSession


TADA_RUNTIME_PROFILE = "vibevoice_transformers4_shared"


class TadaIsolatedProxy:
    """Persistent TADA worker with explicit cleanup for ComfyUI's Clear VRAM action."""

    def __init__(self, config, profile: RuntimeProfile):
        if profile.name != TADA_RUNTIME_PROFILE:
            raise RuntimeError(
                f"TADA only supports the '{TADA_RUNTIME_PROFILE}' shared runtime; got '{profile.name}'."
            )

        self.config = config
        self.profile = profile
        self.current_model_name = str(config.model_name or "TADA-1B")
        self.device = resolve_torch_device(str(config.device or "auto"))
        self.current_device = self.device
        self.load_device = self.current_loaded_device()
        self.offload_device = torch.device("cpu")
        self.dtype = torch.bfloat16
        self.model = self
        self.parent = None
        self.currently_used = True
        self.model_options = {}
        self.model_keys = set()
        self.sample_rate = 24000

        params = dict(config.additional_params or {})
        self.model_path = str(params.get("model_path") or config.model_path or "")
        self.codec_path = str(params.get("codec_path") or "")
        self.tokenizer_path = str(params.get("tokenizer_path") or "")
        self.load_dtype = str(params.get("dtype", "auto"))
        self.attn_implementation = str(params.get("attn_implementation", "sdpa"))
        self.prompt_cache_size = int(params.get("prompt_cache_size", 8))

        if not self.model_path or not self.codec_path or not self.tokenizer_path:
            assets = TadaDownloader().resolve_model_assets(self.current_model_name, language=None)
            self.current_model_name = assets["model_name"]
            self.model_path = assets["model_path"]
            self.codec_path = assets["codec_path"]
            self.tokenizer_path = assets["tokenizer_path"]

        estimate_gb = 10 if "3b" in self.current_model_name.lower() else 6
        self._estimated_memory_size = estimate_gb * 1024**3
        self._initialized = False
        self._comfy_loaded_model = None
        self._comfy_model_management = None

        launcher = IsolatedRuntimeLauncher(runtime_root=str(PROJECT_ROOT))
        worker_env = launcher.build_env(profile)
        worker_env["HF_HUB_OFFLINE"] = "1"
        worker_env["TRANSFORMERS_OFFLINE"] = "1"
        worker_env.setdefault("PYTHONUTF8", "1")
        python_path = ensure_runtime(profile)
        self._session = JsonLineWorkerSession(
            python_path=str(python_path),
            worker_script=str(PROJECT_ROOT / "utils" / "runtimes" / "workers" / "tada_worker.py"),
            env=worker_env,
        )
        self._initialize_remote_engine()
        self._register_with_comfy_model_management()

    def _request(self, action: str, payload: Optional[Dict[str, Any]] = None):
        response = self._session.request(
            RuntimeJobRequest(
                engine_name="tada",
                action=action,
                model_name=self.current_model_name,
                device=self.device,
                runtime_profile=self.profile.name,
                payload=payload or {},
                request_id=str(uuid.uuid4()),
            )
        )
        if not response.ok:
            details = response.error or f"TADA {action} failed"
            if response.logs:
                details = f"{details}\n" + "\n".join(response.logs)
            raise RuntimeError(details)
        return response

    def _initialize_remote_engine(self) -> None:
        self._request(
            "initialize",
            {
                "model_path": self.model_path,
                "codec_path": self.codec_path,
                "tokenizer_path": self.tokenizer_path,
                "dtype": self.load_dtype,
                "attn_implementation": self.attn_implementation,
                "prompt_cache_size": self.prompt_cache_size,
            },
        )
        self._initialized = True

    def _ensure_remote_engine(self) -> None:
        process = getattr(self._session, "_process", None)
        if not self._initialized or process is None or process.poll() is not None:
            self._initialize_remote_engine()
            if self._comfy_loaded_model is None:
                self._register_with_comfy_model_management()

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
            finalizer = model_management.cleanup_models if hasattr(model_management, "cleanup_models") else lambda: None
            loaded_model.model_finalizer = weakref.finalize(self, finalizer)
            loaded_model._tts_wrapper_ref = self
            model_management.current_loaded_models.insert(0, loaded_model)
            self._comfy_loaded_model = loaded_model
            self._comfy_model_management = model_management
        except Exception as exc:
            print(f"⚠️ Failed to register TADA runtime with ComfyUI model management: {exc}")

    def _unregister_from_comfy_model_management(self) -> None:
        model_management = self._comfy_model_management
        loaded_model = self._comfy_loaded_model
        if model_management is None or loaded_model is None:
            return
        try:
            if loaded_model in model_management.current_loaded_models:
                model_management.current_loaded_models.remove(loaded_model)
        except Exception as exc:
            print(f"⚠️ Failed to remove TADA runtime from ComfyUI tracking: {exc}")
        finally:
            self._comfy_loaded_model = None
            self._comfy_model_management = None

    @staticmethod
    def _serialize_reference(reference_audio: object, bundle_dir: Path) -> Dict[str, Any]:
        if isinstance(reference_audio, (str, os.PathLike)):
            path = os.path.abspath(str(reference_audio))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"TADA reference audio file not found: {path}")
            return {"kind": "audio_path", "audio_path": path}

        audio = reference_audio
        sample_rate = 24000
        if isinstance(audio, (tuple, list)) and len(audio) == 2:
            audio, sample_rate = audio
        elif isinstance(audio, dict):
            if "audio" in audio and isinstance(audio["audio"], dict):
                audio = audio["audio"]
            if isinstance(audio, dict):
                path = audio.get("audio_path") or audio.get("path")
                if path:
                    path = os.path.abspath(str(path))
                    if not os.path.isfile(path):
                        raise FileNotFoundError(f"TADA reference audio file not found: {path}")
                    return {"kind": "audio_path", "audio_path": path}
                sample_rate = audio.get("sample_rate", audio.get("sampling_rate", sample_rate))
                audio = audio.get("waveform", audio.get("samples"))

        if audio is None:
            raise ValueError("TADA requires reference audio.")
        waveform = audio.detach().cpu() if isinstance(audio, torch.Tensor) else torch.as_tensor(audio)
        tensor_path = bundle_dir / "reference.pt"
        torch.save({"waveform": waveform, "sample_rate": int(sample_rate)}, tensor_path)
        return {"kind": "tensor_path", "tensor_path": str(tensor_path)}

    def generate_speech(
        self,
        text: str,
        reference_audio: object,
        reference_text: str,
        seed: int = 42,
        language: object = "English",
        acoustic_cfg_scale: float = 1.6,
        duration_cfg_scale: float = 1.0,
        cfg_schedule: str = "cosine",
        time_schedule: str = "logsnr",
        num_flow_matching_steps: int = 10,
        noise_temperature: float = 0.9,
        speed_up_factor: Optional[float] = None,
        num_transition_steps: int = 5,
        negative_condition_source: str = "negative_step_output",
    ) -> Tuple[torch.Tensor, int]:
        if not str(reference_text or "").strip():
            raise ValueError(
                "TADA requires the exact transcript of the reference audio; ASR fallback is disabled."
            )
        validate_tada_model_language(self.current_model_name, language)

        # A segment may switch language without reloading the base model. Ensure
        # just its aligner exists before the worker enters enforced offline mode.
        codec_parent = str(Path(self.codec_path).resolve().parent)
        TadaDownloader(base_path=codec_parent).ensure_codec_assets(
            language=language,
            codec_path=self.codec_path,
        )
        self._ensure_remote_engine()

        with tempfile.TemporaryDirectory(prefix="tts_tada_") as temp_dir:
            bundle_dir = Path(temp_dir)
            output_path = bundle_dir / "result.pt"
            serialized_reference = self._serialize_reference(reference_audio, bundle_dir)
            self._request(
                "generate",
                {
                    "text": str(text),
                    "reference_audio": serialized_reference,
                    "reference_text": str(reference_text),
                    "seed": int(seed),
                    "language": language,
                    "acoustic_cfg_scale": float(acoustic_cfg_scale),
                    "duration_cfg_scale": float(duration_cfg_scale),
                    "cfg_schedule": str(cfg_schedule),
                    "time_schedule": str(time_schedule),
                    "num_flow_matching_steps": int(num_flow_matching_steps),
                    "noise_temperature": float(noise_temperature),
                    "speed_up_factor": speed_up_factor,
                    "num_transition_steps": int(num_transition_steps),
                    "negative_condition_source": str(negative_condition_source),
                    "output_path": str(output_path),
                },
            )
            if not output_path.is_file():
                raise RuntimeError("TADA worker returned without writing its audio result.")
            result = torch.load(output_path, map_location="cpu")
            audio = result.get("audio")
            if not isinstance(audio, torch.Tensor):
                audio = torch.as_tensor(audio)
            return audio.detach().cpu().float(), int(result.get("sample_rate", 24000))

    def to(self, device):
        self.device = resolve_torch_device(str(device))
        self.current_device = self.device
        self.load_device = self.current_loaded_device()
        return self

    def eval(self):
        return self

    def model_size(self):
        return self._estimated_memory_size

    def loaded_size(self):
        return self._estimated_memory_size if self._initialized else 0

    def model_memory(self):
        return self._estimated_memory_size

    def get_ram_usage(self):
        return self._estimated_memory_size

    def model_offloaded_memory(self):
        return 0 if self._initialized else self._estimated_memory_size

    def model_mmap_residency(self, free: bool = False):
        return 0, self._estimated_memory_size

    def pinned_memory_size(self):
        return 0

    def lowvram_patch_counter(self):
        return 0

    def model_dtype(self):
        return self.dtype

    def current_loaded_device(self):
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def model_patches_models(self):
        return ()

    def model_patches_to(self, target):
        return None

    def is_dynamic(self):
        return False

    def partially_load(self, device, extra_memory, force_patch_weights=False):
        return 0

    def partially_unload_ram(self, ram_to_unload):
        return self.partially_unload("cpu", ram_to_unload)

    def partially_unload(self, device, memory_to_free):
        self.cleanup(unregister=False)
        return self._estimated_memory_size

    def model_unload(self, *args, **kwargs):
        self.cleanup(unregister=False)
        return True

    def detach(self, unpatch_weights=True):
        self.cleanup(unregister=False)

    def is_clone(self, other):
        return other is self

    def cleanup(self, unregister: bool = True) -> None:
        if unregister:
            self._unregister_from_comfy_model_management()
        else:
            self._comfy_loaded_model = None
            self._comfy_model_management = None
        process = getattr(getattr(self, "_session", None), "_process", None)
        if self._initialized and process is not None and process.poll() is None:
            try:
                self._request("cleanup")
            except Exception:
                pass
        self._initialized = False
        session = getattr(self, "_session", None)
        if session is not None:
            session.close()

    def close(self) -> None:
        self.cleanup()


def build_tada_isolated_proxy(config):
    requested_profile = getattr(config, "runtime_profile", None) or TADA_RUNTIME_PROFILE
    if requested_profile != TADA_RUNTIME_PROFILE:
        raise RuntimeError(
            f"TADA only supports runtime profile '{TADA_RUNTIME_PROFILE}', not '{requested_profile}'."
        )
    profile = get_runtime_profile(TADA_RUNTIME_PROFILE)
    if profile is None:
        raise RuntimeError(f"TADA runtime profile '{TADA_RUNTIME_PROFILE}' is not configured.")
    return TadaIsolatedProxy(config, profile)


__all__ = ["TADA_RUNTIME_PROFILE", "TadaIsolatedProxy", "build_tada_isolated_proxy"]
