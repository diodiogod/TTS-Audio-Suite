"""
OmniVoice engine wrapper.

Wraps the official `omnivoice` package with ComfyUI-friendly lifecycle hooks.
"""

import importlib
import logging
import os
import sys
import types
from contextlib import contextmanager
from typing import Any, Optional

import torch

current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.omnivoice.omnivoice_downloader import OmniVoiceDownloader
from engines.omnivoice.progress_callback import OmniVoiceGenerationProgress
from utils.device import resolve_torch_device


@contextmanager
def _suppress_transformers_logs():
    logger_names = ("transformers", "transformers.modeling_utils", "transformers.tokenization_utils_base")
    original_levels = {}
    try:
        for name in logger_names:
            logger = logging.getLogger(name)
            original_levels[name] = logger.level
            logger.setLevel(logging.WARNING)
        yield
    finally:
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)


class OmniVoiceEngine:
    """Thin wrapper around the official OmniVoice model."""

    SAMPLE_RATE = 24000

    def __init__(
        self,
        model_name: str = "OmniVoice",
        device: str = "auto",
        dtype: str = "auto",
        model_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = resolve_torch_device(device)
        self.dtype = self._resolve_dtype(dtype, self.device)
        self.downloader = OmniVoiceDownloader()
        self.model_dir = model_dir or self.downloader.resolve_model_path(model_name)
        self._model = None

    def _install_progress_hook(self):
        if self._model is None or getattr(self._model, "_tts_suite_progress_patched", False):
            return

        original_forward = self._model.forward
        original_generate_iterative = self._model._generate_iterative

        def wrapped_forward(model_self, *args, **kwargs):
            result = original_forward(*args, **kwargs)
            tracker = getattr(model_self, "_tts_suite_progress_tracker", None)
            if tracker is not None:
                current_step = getattr(model_self, "_tts_suite_progress_step", 0) + 1
                model_self._tts_suite_progress_step = current_step
                tracker.update(current_step, force=(current_step == 1 or current_step >= tracker.total_steps))
            return result

        def wrapped_generate_iterative(model_self, task, gen_config):
            total_steps = int(getattr(gen_config, "num_step", 0) or 0)
            tracker = OmniVoiceGenerationProgress(total_steps)
            model_self._tts_suite_progress_tracker = tracker
            model_self._tts_suite_progress_step = 0
            try:
                result = original_generate_iterative(task, gen_config)
                tracker.end(getattr(model_self, "_tts_suite_progress_step", 0))
                return result
            except Exception:
                tracker.abort()
                raise
            finally:
                model_self._tts_suite_progress_tracker = None
                model_self._tts_suite_progress_step = 0

        self._model.forward = types.MethodType(wrapped_forward, self._model)
        self._model._generate_iterative = types.MethodType(wrapped_generate_iterative, self._model)
        self._model._tts_suite_progress_tracker = None
        self._model._tts_suite_progress_step = 0
        self._model._tts_suite_progress_patched = True

    @staticmethod
    def _resolve_dtype(dtype: str, device: str):
        normalized = str(dtype or "auto").lower()
        explicit = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized in explicit:
            if device == "cpu" and explicit[normalized] != torch.float32:
                return torch.float32
            return explicit[normalized]

        if device == "cpu":
            return torch.float32
        if device == "cuda" and torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
            if major >= 8:
                return torch.bfloat16
            return torch.float16
        if device == "xpu":
            return torch.bfloat16
        return torch.float16

    def _import_official_package(self):
        importlib.invalidate_caches()

        nodes_dir = os.path.join(project_root, "nodes")
        blocked_paths = {
            os.path.abspath(engines_dir),
            os.path.abspath(nodes_dir),
        }
        original_sys_path = list(sys.path)
        sys.path[:] = [
            path for path in sys.path
            if os.path.abspath(path or os.getcwd()) not in blocked_paths
        ]

        stale_modules = []
        for module_name, module in list(sys.modules.items()):
            if not (module_name == "omnivoice" or module_name.startswith("omnivoice.")):
                continue
            module_file = getattr(module, "__file__", "") or ""
            if module_file and os.path.abspath(module_file).startswith(os.path.abspath(project_root)):
                stale_modules.append((module_name, module))
                del sys.modules[module_name]

        try:
            from omnivoice import OmniVoice  # type: ignore
        except Exception as e:
            for module_name, module in stale_modules:
                sys.modules.setdefault(module_name, module)
            raise ImportError(
                "Failed to import the official `omnivoice` package in the active ComfyUI Python "
                "environment. This is usually a missing install or import-path collision. "
                f"Active Python: {sys.executable}"
            ) from e
        finally:
            sys.path[:] = original_sys_path

        return OmniVoice

    def _loaded_model_device(self) -> Optional[str]:
        if self._model is None:
            return None
        try:
            first_param = next(self._model.parameters())
            return str(first_param.device)
        except Exception:
            return None

    def _ensure_model_device(self):
        if self._model is None:
            return
        current_device = self._loaded_model_device()
        if current_device == self.device:
            return
        self.to(self.device)

    def _ensure_model_loaded(self):
        if self._model is not None:
            self._ensure_model_device()
            return

        OmniVoice = self._import_official_package()
        print(f"🔄 Loading OmniVoice model via official package")
        print(f"   Path: {self.model_dir}")
        print(f"   Device: {self.device} | Dtype: {self.dtype}")

        with _suppress_transformers_logs():
            self._model = OmniVoice.from_pretrained(
                self.model_dir,
                device_map=self.device,
                dtype=self.dtype,
                load_asr=False,
            )
        self._install_progress_hook()
        print("✅ OmniVoice model loaded successfully")

    @property
    def sampling_rate(self) -> int:
        if self._model is not None and getattr(self._model, "sampling_rate", None):
            return int(self._model.sampling_rate)
        return self.SAMPLE_RATE

    def create_voice_clone_prompt(
        self,
        ref_audio: Any,
        ref_text: Optional[str] = None,
        preprocess_prompt: bool = True,
    ):
        self._ensure_model_loaded()
        if ref_text is None or not str(ref_text).strip():
            raise ValueError(
                "OmniVoice voice cloning in this suite requires reference text. "
                "Use Character Voices or a narrator voice with a matching .reference.txt file."
            )

        return self._model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            preprocess_prompt=preprocess_prompt,
        )

    def generate(self, **kwargs):
        self._ensure_model_loaded()
        return self._model.generate(**kwargs)

    def to(self, device: str):
        self.device = resolve_torch_device(device)
        if self._model is None:
            return self

        try:
            if hasattr(self._model, "to"):
                self._model = self._model.to(self.device)
        except Exception as e:
            print(f"⚠️ OmniVoice base model move to {self.device} failed: {e}")

        for attr_name in ("audio_tokenizer",):
            attr_value = getattr(self._model, attr_name, None)
            if hasattr(attr_value, "to"):
                try:
                    setattr(self._model, attr_name, attr_value.to(self.device))
                except Exception as e:
                    print(f"⚠️ OmniVoice {attr_name} move to {self.device} failed: {e}")

        return self

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._model is not None and hasattr(self._model, name):
            return getattr(self._model, name)
        raise AttributeError(name)
