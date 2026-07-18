from __future__ import annotations

"""Step Audio EditX proxy for its official Transformers 4 runtime."""

import tempfile
import uuid
import weakref
from pathlib import Path

import torch

from utils.models.factory_config import ModelLoadConfig
from .bootstrap import PROJECT_ROOT, ensure_runtime
from .launcher import IsolatedRuntimeLauncher
from .profiles import RuntimeProfile, get_runtime_profile
from .protocol import RuntimeJobRequest
from .session import JsonLineWorkerSession


class StepAudioEditXIsolatedProxy:
    def __init__(self, config: ModelLoadConfig, profile: RuntimeProfile):
        from engines.step_audio_editx.step_audio_editx_downloader import StepAudioEditXDownloader

        self.config = config
        self.profile = profile
        self.device = config.device
        self.current_device = config.device
        self.load_device = self.current_loaded_device()
        self.offload_device = "cpu"
        self.currently_used = True
        self.model = self
        self.engine = self
        self.parent = None
        self.model_options = {}
        self.model_keys = set()
        self._estimated_memory_size = 12 * 1024 * 1024 * 1024
        self._comfy_loaded_model = None
        self._comfy_model_management = None
        self._initialized = False

        self.model_path = StepAudioEditXDownloader().resolve_model_path(config.model_path)
        launcher = IsolatedRuntimeLauncher(runtime_root=str(PROJECT_ROOT))
        python_path = ensure_runtime(profile)
        worker_script = PROJECT_ROOT / "utils/runtimes/workers/step_audio_editx_worker.py"
        self._session = JsonLineWorkerSession(
            python_path=str(python_path),
            worker_script=str(worker_script),
            env=launcher.build_env(profile),
        )
        self._register_with_comfy_model_management()
        self._initialize_remote_engine()

    def _request(self, action, payload):
        response = self._session.request(RuntimeJobRequest(
            engine_name="step_audio_editx",
            action=action,
            model_name="Step-Audio-EditX",
            device=str(self.device),
            runtime_profile=self.profile.name,
            request_id=str(uuid.uuid4()),
            payload=payload,
        ))
        if not response.ok:
            raise RuntimeError(response.error or f"Step Audio EditX isolated action '{action}' failed")
        return response

    def _initialize_remote_engine(self):
        self._request("initialize", {
            "model_path": self.model_path,
            "torch_dtype": self.config.additional_params.get("torch_dtype", "bfloat16"),
            "quantization": self.config.additional_params.get("quantization"),
        })
        self._initialized = True
        print(f"✅ Step Audio EditX isolated runtime ready ({self.profile.name})")

    def _run(self, action, payload):
        with tempfile.TemporaryDirectory(prefix="tts_step_editx_") as temp_dir:
            output_path = Path(temp_dir) / "result.pt"
            payload["output_path"] = str(output_path)
            self._request(action, payload)
            if not output_path.exists():
                raise RuntimeError("Step Audio EditX isolated worker returned no output")
            result = torch.load(output_path, map_location="cpu")
            return result["audio"], int(result["sample_rate"])

    def clone(self, prompt_wav_path, prompt_text, target_text, temperature=0.7,
              do_sample=True, max_new_tokens=1024, progress_bar=None):
        return self._run("clone", {
            "prompt_wav_path": prompt_wav_path,
            "prompt_text": prompt_text,
            "target_text": target_text,
            "temperature": temperature,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
        })

    def edit_single(self, input_audio_path, audio_text, edit_type, edit_info=None,
                    text=None, progress_bar=None, max_new_tokens=1024,
                    temperature=0.7, do_sample=True):
        audio, _ = self._run("edit", {
            "input_audio_path": input_audio_path,
            "audio_text": audio_text,
            "edit_type": edit_type,
            "edit_info": edit_info,
            "text": text,
            "temperature": temperature,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
        })
        return audio

    def edit(self, input_audio_path, audio_text, edit_type, edit_info=None,
             text=None, progress_bar=None, max_new_tokens=1024,
             temperature=0.7, do_sample=True):
        return self.edit_single(
            input_audio_path,
            audio_text,
            edit_type,
            edit_info,
            text,
            progress_bar,
            max_new_tokens,
            temperature,
            do_sample,
        )

    def get_sample_rate(self):
        """Match the in-process Step engine interface."""
        return 24000

    def _register_with_comfy_model_management(self):
        try:
            import comfy.model_management as mm
            loaded = mm.LoadedModel(self)
            loaded.real_model = weakref.ref(self)
            loaded._tts_wrapper_ref = self
            mm.current_loaded_models.insert(0, loaded)
            self._comfy_loaded_model = loaded
            self._comfy_model_management = mm
        except Exception as exc:
            print(f"⚠️ Failed to register isolated Step runtime: {exc}")

    def cleanup(self):
        if self._comfy_model_management and self._comfy_loaded_model:
            try:
                self._comfy_model_management.current_loaded_models.remove(self._comfy_loaded_model)
            except (ValueError, AttributeError):
                pass
        self._initialized = False
        self._session.close()

    close = cleanup

    def to(self, device):
        self.device = str(device)
        self.current_device = self.device
        self.load_device = self.current_loaded_device()
        return self

    def eval(self): return self
    def loaded_size(self): return self._estimated_memory_size
    def model_size(self): return self._estimated_memory_size
    def model_memory(self): return self._estimated_memory_size
    def get_ram_usage(self): return self._estimated_memory_size
    def model_offloaded_memory(self): return 0
    def model_mmap_residency(self, free=False): return 0, self._estimated_memory_size
    def pinned_memory_size(self): return 0
    def model_dtype(self): return torch.bfloat16
    def model_patches_models(self): return ()
    def is_dynamic(self): return False
    def model_patches_to(self, target): return None
    def current_loaded_device(self):
        return torch.device("cuda" if str(self.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    def partially_load(self, device, extra_memory, force_patch_weights=False): return 0
    def partially_unload(self, device, memory_to_free):
        self.cleanup()
        return self._estimated_memory_size
    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        self.cleanup()
        return True
    def detach(self, unpatch_weights=True): self.cleanup()
    def is_clone(self, other): return other is self


def build_step_audio_editx_isolated_proxy(config: ModelLoadConfig):
    profile_name = config.runtime_profile or "vibevoice_transformers4_shared"
    profile = get_runtime_profile(profile_name)
    if profile is None:
        raise RuntimeError(f"Unknown isolated runtime profile '{profile_name}' for Step Audio EditX")
    return StepAudioEditXIsolatedProxy(config, profile)
