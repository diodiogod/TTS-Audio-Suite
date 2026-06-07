from __future__ import annotations

"""
Qwen3-TTS isolated runtime proxy.
"""

import tempfile
import uuid
import weakref
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from utils.models.factory_config import ModelLoadConfig
from .bootstrap import PROJECT_ROOT, ensure_runtime
from .launcher import IsolatedRuntimeLauncher
from .profiles import RuntimeProfile, get_runtime_profile
from .protocol import RuntimeJobRequest
from .session import JsonLineWorkerSession


class Qwen3TTSIsolatedProxy:
    def __init__(self, config: ModelLoadConfig, profile: RuntimeProfile):
        self.config = config
        self.profile = profile
        self.current_model_name = config.model_name
        self.current_model_path = config.model_path
        self.device = config.device
        self.current_device = config.device
        self.load_device = self.current_loaded_device()
        self.offload_device = "cpu"
        self.dtype = self._resolve_dtype(config.additional_params.get("dtype", "auto"))
        self.attn_implementation = config.additional_params.get("attn_implementation", "auto")
        self.model = self
        self.processor = self
        self.parent = None
        self.currently_used = True
        self.model_options = {}
        self.model_keys = set()
        self._estimated_memory_size = 10 * 1024 * 1024 * 1024
        self._comfy_loaded_model = None
        self._comfy_model_management = None

        launcher = IsolatedRuntimeLauncher(runtime_root=str(PROJECT_ROOT))
        python_path = ensure_runtime(profile)
        worker_script = str(PROJECT_ROOT / "utils" / "runtimes" / "workers" / "qwen3_tts_worker.py")

        self._session = JsonLineWorkerSession(
            python_path=str(python_path),
            worker_script=worker_script,
            env=launcher.build_env(profile),
        )
        self._register_with_comfy_model_management()
        self._initialize_remote_engine()

    def _resolve_dtype(self, dtype_name: str):
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float32":
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
            print(f"⚠️ Failed to register isolated Qwen3-TTS runtime with ComfyUI model management: {e}")

    def _unregister_from_comfy_model_management(self) -> None:
        model_management = self._comfy_model_management
        loaded_model = self._comfy_loaded_model
        if model_management is None or loaded_model is None:
            return

        try:
            if hasattr(model_management, "current_loaded_models") and loaded_model in model_management.current_loaded_models:
                model_management.current_loaded_models.remove(loaded_model)
        except Exception as e:
            print(f"⚠️ Failed to remove isolated Qwen3-TTS runtime from ComfyUI tracking: {e}")
        finally:
            self._comfy_loaded_model = None
            self._comfy_model_management = None

    def _initialize_remote_engine(self) -> None:
        response = self._session.request(
            RuntimeJobRequest(
                engine_name="qwen3_tts",
                action="initialize",
                model_name=self.current_model_name,
                device=str(self.device),
                runtime_profile=self.profile.name,
                payload={
                    "dtype": self.config.additional_params.get("dtype", "auto"),
                    "attn_implementation": self.attn_implementation,
                    "model_path": self.current_model_path,
                },
                request_id=str(uuid.uuid4()),
            )
        )
        if not response.ok:
            details = response.error or "Failed to initialize isolated Qwen3-TTS runtime"
            if response.logs:
                details = f"{details}\n" + "\n".join(response.logs)
            raise RuntimeError(details)

        print(f"✅ Qwen3-TTS isolated runtime ready ({self.profile.name})")

    def _serialize_ref_audio(self, ref_audio: Any, bundle_dir: Path) -> Optional[Dict[str, Any]]:
        if ref_audio is None:
            return None

        if isinstance(ref_audio, str):
            return {"kind": "audio_path", "audio_path": ref_audio}

        if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
            waveform, sample_rate = ref_audio
            tensor_path = bundle_dir / "ref_audio.pt"
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
            return {"kind": "tensor_path", "tensor_path": str(tensor_path)}

        if isinstance(ref_audio, dict) and "waveform" in ref_audio:
            tensor_path = bundle_dir / "ref_audio.pt"
            waveform = ref_audio["waveform"]
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform)
            torch.save(
                {
                    "waveform": waveform.detach().cpu().float() if isinstance(waveform, torch.Tensor) else waveform,
                    "sample_rate": int(ref_audio.get("sample_rate", 24000)),
                },
                tensor_path,
            )
            return {"kind": "tensor_path", "tensor_path": str(tensor_path)}

        raise TypeError(f"Unsupported Qwen3-TTS isolated ref_audio type: {type(ref_audio)}")

    def _run_generation(self, action: str, payload: Dict[str, Any]) -> tuple[list[np.ndarray], int]:
        with tempfile.TemporaryDirectory(prefix="tts_qwen3_iso_") as temp_dir:
            bundle_dir = Path(temp_dir)
            output_path = bundle_dir / "result.pt"

            ref_audio = payload.pop("ref_audio", None)
            serialized_ref_audio = self._serialize_ref_audio(ref_audio, bundle_dir)
            if serialized_ref_audio is not None:
                payload["ref_audio"] = serialized_ref_audio

            payload["output_path"] = str(output_path)

            response = self._session.request(
                RuntimeJobRequest(
                    engine_name="qwen3_tts",
                    action=action,
                    model_name=self.current_model_name,
                    device=str(self.device),
                    runtime_profile=self.profile.name,
                    request_id=str(uuid.uuid4()),
                    payload=payload,
                )
            )

            if not response.ok:
                details = response.error or f"Isolated Qwen3-TTS action '{action}' failed"
                if response.logs:
                    details = f"{details}\n" + "\n".join(response.logs)
                raise RuntimeError(details)
            if not output_path.exists():
                raise RuntimeError("Isolated Qwen3-TTS worker returned no output payload")

            result = torch.load(output_path, map_location="cpu")
            wavs = result.get("wavs", [])
            sr = int(result.get("sample_rate", 24000))
            wavs_np = []
            for wav in wavs:
                if isinstance(wav, torch.Tensor):
                    wavs_np.append(wav.cpu().numpy())
                else:
                    wavs_np.append(np.asarray(wav, dtype=np.float32))
            return wavs_np, sr

    def generate_custom_voice(self, text, language, speaker, instruct=None, progress_bar=None, **kwargs):
        kwargs.pop("streamer", None)
        return self._run_generation(
            "generate_custom_voice",
            {
                "text": text,
                "language": language,
                "speaker": speaker,
                "instruct": instruct,
                **kwargs,
            },
        )

    def generate_voice_design(self, text, language, instruct, progress_bar=None, **kwargs):
        kwargs.pop("streamer", None)
        return self._run_generation(
            "generate_voice_design",
            {
                "text": text,
                "language": language,
                "instruct": instruct,
                **kwargs,
            },
        )

    def generate_voice_clone(
        self,
        text,
        language,
        ref_audio,
        ref_text=None,
        x_vector_only_mode=False,
        voice_clone_prompt=None,
        progress_bar=None,
        **kwargs,
    ):
        kwargs.pop("streamer", None)
        if voice_clone_prompt is not None:
            raise RuntimeError("Qwen3-TTS isolated runtime does not support precomputed voice_clone_prompt yet")
        return self._run_generation(
            "generate_voice_clone",
            {
                "text": text,
                "language": language,
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "x_vector_only_mode": x_vector_only_mode,
                **kwargs,
            },
        )

    def enable_streaming_optimizations(self, use_compile=True, use_cuda_graphs=True, compile_mode="reduce-overhead", decode_window_frames=80):
        response = self._session.request(
            RuntimeJobRequest(
                engine_name="qwen3_tts",
                action="enable_streaming_optimizations",
                model_name=self.current_model_name,
                device=str(self.device),
                runtime_profile=self.profile.name,
                request_id=str(uuid.uuid4()),
                payload={
                    "use_compile": use_compile,
                    "use_cuda_graphs": use_cuda_graphs,
                    "compile_mode": compile_mode,
                    "decode_window_frames": decode_window_frames,
                },
            )
        )
        if not response.ok:
            details = response.error or "Failed to enable isolated Qwen3-TTS optimizations"
            if response.logs:
                details = f"{details}\n" + "\n".join(response.logs)
            raise RuntimeError(details)
        return self

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
        self._session.close()

    def close(self):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


def build_qwen3_tts_isolated_proxy(config: ModelLoadConfig) -> Qwen3TTSIsolatedProxy:
    profile_name = config.runtime_profile or "vibevoice_transformers4_shared"
    profile = get_runtime_profile(profile_name)
    if profile is None:
        raise RuntimeError(f"Unknown isolated runtime profile '{profile_name}' for Qwen3-TTS")
    return Qwen3TTSIsolatedProxy(config=config, profile=profile)
