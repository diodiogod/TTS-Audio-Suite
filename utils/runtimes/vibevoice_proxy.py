from __future__ import annotations

"""
VibeVoice isolated runtime proxy.
"""

import tempfile
import uuid
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from utils.models.factory_config import ModelLoadConfig
from utils.voice.reference import effective_voice_audio
from .bootstrap import PROJECT_ROOT, ensure_runtime
from .launcher import IsolatedRuntimeLauncher
from .profiles import RuntimeProfile, get_runtime_profile
from .protocol import RuntimeJobRequest
from .session import JsonLineWorkerSession


class VibeVoiceIsolatedProxy:
    def __init__(self, config: ModelLoadConfig, profile: RuntimeProfile):
        self.config = config
        self.profile = profile
        self.current_model_name = config.model_name
        self.device = config.device
        self.current_device = config.device
        self.load_device = self.current_loaded_device()
        self.offload_device = "cpu"
        self.attention_mode = config.additional_params.get("attention_mode", "auto")
        self.quantize_llm_4bit = config.additional_params.get("quantize_llm_4bit", False)
        self.model = self
        self.processor = self
        self.parent = None
        self.is_kugelaudio = False
        self.dtype = torch.float16
        self.currently_used = True
        self.model_options = {}
        self.model_keys = set()
        self._estimated_memory_size = 6 * 1024 * 1024 * 1024
        self._comfy_loaded_model = None
        self._comfy_model_management = None
        self._initialized = False

        launcher = IsolatedRuntimeLauncher(runtime_root=str(PROJECT_ROOT))
        python_path = ensure_runtime(profile)
        worker_script = str(PROJECT_ROOT / "utils" / "runtimes" / "workers" / "vibevoice_worker.py")

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
            print(f"⚠️ Failed to register isolated VibeVoice runtime with ComfyUI model management: {e}")

    def _unregister_from_comfy_model_management(self) -> None:
        model_management = self._comfy_model_management
        loaded_model = self._comfy_loaded_model
        if model_management is None or loaded_model is None:
            return

        try:
            if hasattr(model_management, "current_loaded_models") and loaded_model in model_management.current_loaded_models:
                model_management.current_loaded_models.remove(loaded_model)
        except Exception as e:
            print(f"⚠️ Failed to remove isolated VibeVoice runtime from ComfyUI tracking: {e}")
        finally:
            self._comfy_loaded_model = None
            self._comfy_model_management = None

    def _initialize_remote_engine(self) -> None:
        response = self._session.request(
            RuntimeJobRequest(
                engine_name="vibevoice",
                action="initialize",
                model_name=self.current_model_name,
                device=str(self.device),
                runtime_profile=self.profile.name,
                payload={
                    "attention_mode": self.attention_mode,
                    "quantize_llm_4bit": self.quantize_llm_4bit,
                },
                request_id=str(uuid.uuid4()),
            )
        )
        if not response.ok:
            details = response.error or "Failed to initialize isolated VibeVoice runtime"
            if response.logs:
                details = f"{details}\n" + "\n".join(response.logs)
            raise RuntimeError(details)

        self._initialized = True
        self.is_kugelaudio = bool(response.result.get("is_kugelaudio", False))
        print(
            f"✅ VibeVoice isolated runtime ready "
            f"({self.profile.name}, kugel={self.is_kugelaudio})"
        )

    def _ensure_remote_engine(self) -> None:
        process = getattr(self._session, "_process", None)
        if not self._initialized or process is None or process.poll() is not None:
            self._initialize_remote_engine()

    def _prepare_voice_samples(self, voice_refs: List[Optional[Dict]]) -> List[Optional[Dict]]:
        # Raw refs cross the process boundary and are prepared in the worker.
        return voice_refs

    def _serialize_voice_ref(self, voice_ref: Optional[Dict[str, Any]], bundle_dir: Path, index: int) -> Optional[Dict[str, Any]]:
        if voice_ref is None:
            return None

        if not isinstance(voice_ref, dict):
            raise TypeError(f"Unsupported VibeVoice isolated voice reference type: {type(voice_ref)}")

        reference_text = voice_ref.get("reference_text") or voice_ref.get("text") or ""
        character_name = voice_ref.get("character_name") or "narrator"

        effective_audio = effective_voice_audio(voice_ref)
        if isinstance(effective_audio, str):
            return {
                "kind": "audio_path",
                "audio_path": effective_audio,
                "reference_text": reference_text,
                "character_name": character_name,
            }

        nested_audio = effective_audio
        if isinstance(nested_audio, dict):
            nested_audio_path = nested_audio.get("audio_path")
            if nested_audio_path:
                return {
                    "kind": "audio_path",
                    "audio_path": nested_audio_path,
                    "reference_text": reference_text,
                    "character_name": character_name,
                }
            if "waveform" in nested_audio:
                tensor_path = bundle_dir / f"voice_{index}.pt"
                torch.save(
                    {
                        "waveform": nested_audio["waveform"].detach().cpu()
                        if isinstance(nested_audio["waveform"], torch.Tensor)
                        else nested_audio["waveform"],
                        "sample_rate": nested_audio.get("sample_rate", 24000),
                    },
                    tensor_path,
                )
                return {
                    "kind": "tensor_path",
                    "tensor_path": str(tensor_path),
                    "reference_text": reference_text,
                    "character_name": character_name,
                }
        elif nested_audio is not None:
            tensor_path = bundle_dir / f"voice_{index}.pt"
            torch.save(
                {
                    "waveform": nested_audio.detach().cpu()
                    if isinstance(nested_audio, torch.Tensor)
                    else nested_audio,
                    "sample_rate": voice_ref.get("sample_rate", 24000),
                },
                tensor_path,
            )
            return {
                "kind": "tensor_path",
                "tensor_path": str(tensor_path),
                "reference_text": reference_text,
                "character_name": character_name,
            }

        if "waveform" in voice_ref:
            tensor_path = bundle_dir / f"voice_{index}.pt"
            waveform = voice_ref["waveform"]
            torch.save(
                {
                    "waveform": waveform.detach().cpu() if isinstance(waveform, torch.Tensor) else waveform,
                    "sample_rate": voice_ref.get("sample_rate", 24000),
                },
                tensor_path,
            )
            return {
                "kind": "tensor_path",
                "tensor_path": str(tensor_path),
                "reference_text": reference_text,
                "character_name": character_name,
            }

        raise ValueError("Unsupported VibeVoice isolated voice reference format")

    def generate_speech(
        self,
        text: str,
        voice_samples: List[Optional[Dict]],
        cfg_scale: float = 1.3,
        seed: int = 42,
        use_sampling: bool = False,
        temperature: float = 0.95,
        top_p: float = 0.95,
        inference_steps: int = 20,
        max_new_tokens: Optional[int] = None,
        enable_cache: bool = True,
        character: str = "narrator",
        stable_audio_component: str = "",
        multi_speaker_mode: str = "Custom Character Switching",
    ) -> Dict[str, Any]:
        self._ensure_remote_engine()
        with tempfile.TemporaryDirectory(prefix="tts_vibevoice_iso_") as temp_dir:
            bundle_dir = Path(temp_dir)
            output_path = bundle_dir / "result.pt"
            serialized_refs = [
                self._serialize_voice_ref(ref, bundle_dir, index)
                for index, ref in enumerate(voice_samples)
            ]

            response = self._session.request(
                RuntimeJobRequest(
                    engine_name="vibevoice",
                    action="generate_from_refs",
                    model_name=self.current_model_name,
                    device=str(self.device),
                    runtime_profile=self.profile.name,
                    request_id=str(uuid.uuid4()),
                    payload={
                        "text": text,
                        "voice_refs": serialized_refs,
                        "cfg_scale": cfg_scale,
                        "seed": seed,
                        "use_sampling": use_sampling,
                        "temperature": temperature,
                        "top_p": top_p,
                        "inference_steps": inference_steps,
                        "max_new_tokens": max_new_tokens,
                        "enable_cache": enable_cache,
                        "character": character,
                        "stable_audio_component": stable_audio_component,
                        "multi_speaker_mode": multi_speaker_mode,
                        "output_path": str(output_path),
                    },
                )
            )

            if not response.ok:
                details = response.error or "Isolated VibeVoice generation failed"
                if response.logs:
                    details = f"{details}\n" + "\n".join(response.logs)
                raise RuntimeError(details)
            if not output_path.exists():
                raise RuntimeError("Isolated VibeVoice worker returned no output payload")

            result = torch.load(output_path, map_location="cpu")
            waveform = result.get("waveform")
            if isinstance(waveform, torch.Tensor):
                result["waveform"] = waveform.cpu()
            return result

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


def build_vibevoice_isolated_proxy(config: ModelLoadConfig) -> VibeVoiceIsolatedProxy:
    profile_name = config.runtime_profile or "vibevoice_transformers4_shared"
    profile = get_runtime_profile(profile_name)
    if profile is None:
        raise RuntimeError(f"Unknown isolated runtime profile '{profile_name}' for VibeVoice")
    return VibeVoiceIsolatedProxy(config=config, profile=profile)
