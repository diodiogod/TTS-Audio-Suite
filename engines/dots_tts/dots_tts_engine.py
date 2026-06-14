"""
Dots TTS engine wrapper.

Wraps the official dots.tts runtime with ComfyUI-friendly model lifecycle hooks.
"""

import os
import sys
import types
import importlib
import logging
import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch

current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.dots_tts.dots_tts_downloader import DotsTTSDownloader
from engines.dots_tts.progress_callback import DotsTTSProgressTracker
from utils.device import resolve_torch_device


class _SuppressDotsModelTypeWarning(logging.Filter):
    """Drop the upstream bogus model-type warning while keeping real ones."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        return "model of type `dots_tts` to instantiate a model of type" not in message


class DotsTTSEngine:
    """Thin wrapper around the official DotsTtsRuntime."""

    SAMPLE_RATE = 48000
    _normalizer_warning_shown = False

    def __init__(
        self,
        model_name: str = "dots.tts-soar",
        device: str = "auto",
        precision: str = "auto",
        optimize: bool = False,
        max_generate_length: int = 500,
        model_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = resolve_torch_device(device)
        self.precision = self._resolve_precision(precision, self.device)
        self.optimize = bool(optimize)
        self.max_generate_length = int(max_generate_length)
        self.model_dir = model_dir or DotsTTSDownloader().resolve_model_path(model_name)
        self._runtime = None

    @staticmethod
    def _resolve_precision(precision: str, device: str) -> str:
        normalized = str(precision or "auto").lower()
        if normalized in {"bfloat16", "float16", "float32"}:
            if device == "cpu" and normalized != "float32":
                return "float32"
            return normalized

        if device == "cpu":
            return "float32"

        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
            if major >= 8:
                return "bfloat16"
        return "float16"

    def _import_runtime(self):
        self._ensure_text_normalizer_fallback()
        importlib.invalidate_caches()

        nodes_dir = os.path.join(project_root, "nodes")
        blocked_paths = {
            os.path.abspath(engines_dir),
            os.path.abspath(nodes_dir),
        }
        original_sys_path = list(sys.path)

        # ComfyUI can place the suite's `engines/` directory directly on sys.path.
        # That makes our local `engines/dots_tts` package shadow the pip-installed
        # official `dots_tts` package we actually need here.
        sys.path[:] = [
            path for path in sys.path
            if os.path.abspath(path or os.getcwd()) not in blocked_paths
        ]

        stale_modules = []
        for module_name, module in list(sys.modules.items()):
            if not (module_name == "dots_tts" or module_name.startswith("dots_tts.")):
                continue
            module_file = getattr(module, "__file__", "") or ""
            if module_file and os.path.abspath(module_file).startswith(os.path.abspath(project_root)):
                stale_modules.append((module_name, module))
                del sys.modules[module_name]
        try:
            from dots_tts.runtime import DotsTtsRuntime
        except Exception as e:
            for module_name, module in stale_modules:
                sys.modules.setdefault(module_name, module)
            raise ImportError(
                "Failed to import the official `dots_tts` package in the active ComfyUI Python "
                "environment. This is usually either a missing install or an import-path "
                f"collision. Active Python: {sys.executable}"
            ) from e
        finally:
            sys.path[:] = original_sys_path
        return DotsTtsRuntime

    @staticmethod
    @contextmanager
    def _suppress_noisy_runtime_logs():
        """Mute verbose Dots chatter while keeping actionable Transformers warnings."""
        dots_logger = None
        dots_logger_disabled = False
        transformers_filter = _SuppressDotsModelTypeWarning()
        transformer_loggers = []

        try:
            from loguru import logger as dots_logger

            dots_logger.disable("dots_tts")
            dots_logger_disabled = True

            transformer_loggers = [
                logging.getLogger("transformers"),
                logging.getLogger("transformers.configuration_utils"),
                logging.getLogger("transformers.modeling_utils"),
                logging.getLogger("transformers.tokenization_utils_base"),
            ]
            for logger in transformer_loggers:
                logger.addFilter(transformers_filter)
        except Exception:
            pass

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*model of type `dots_tts` to instantiate a model of type.*",
                )
                yield
        finally:
            if dots_logger is not None and dots_logger_disabled:
                try:
                    dots_logger.enable("dots_tts")
                except Exception:
                    pass

            for logger in transformer_loggers:
                try:
                    logger.removeFilter(transformers_filter)
                except Exception:
                    pass

    @staticmethod
    @contextmanager
    def _patch_tokenizer_regex_fix():
        """Force the upstream Dots tokenizer load to opt into the Mistral regex fix."""
        try:
            from transformers import AutoTokenizer
        except Exception:
            yield
            return

        original_from_pretrained = AutoTokenizer.from_pretrained

        def patched_from_pretrained(*args, **kwargs):
            kwargs.setdefault("fix_mistral_regex", True)
            return original_from_pretrained(*args, **kwargs)

        AutoTokenizer.from_pretrained = patched_from_pretrained
        try:
            yield
        finally:
            AutoTokenizer.from_pretrained = original_from_pretrained

    @staticmethod
    def _ensure_text_normalizer_fallback():
        """Provide a no-op WeTextProcessing fallback when tn is unavailable."""
        try:
            import tn  # noqa: F401
            return
        except Exception:
            pass

        class _NoOpNormalizer:
            def normalize(self, text: str) -> str:
                return text

        tn_module = types.ModuleType("tn")
        chinese_module = types.ModuleType("tn.chinese")
        chinese_normalizer_module = types.ModuleType("tn.chinese.normalizer")
        english_module = types.ModuleType("tn.english")
        english_normalizer_module = types.ModuleType("tn.english.normalizer")

        chinese_normalizer_module.Normalizer = _NoOpNormalizer
        english_normalizer_module.Normalizer = _NoOpNormalizer

        chinese_module.normalizer = chinese_normalizer_module
        english_module.normalizer = english_normalizer_module
        tn_module.chinese = chinese_module
        tn_module.english = english_module

        sys.modules.setdefault("tn", tn_module)
        sys.modules.setdefault("tn.chinese", chinese_module)
        sys.modules.setdefault("tn.chinese.normalizer", chinese_normalizer_module)
        sys.modules.setdefault("tn.english", english_module)
        sys.modules.setdefault("tn.english.normalizer", english_normalizer_module)
        if not DotsTTSEngine._normalizer_warning_shown:
            print("[Dots TTS] WeTextProcessing not available; normalize_text will use a no-op fallback")
            DotsTTSEngine._normalizer_warning_shown = True

    def _ensure_runtime_loaded(self):
        if self._runtime is not None:
            return

        DotsTtsRuntime = self._import_runtime()
        with self._patch_tokenizer_regex_fix(), self._suppress_noisy_runtime_logs():
            self._runtime = DotsTtsRuntime.from_pretrained(
                self.model_dir,
                precision=self.precision,
                optimize=self.optimize,
                max_generate_length=self.max_generate_length,
            )
        self._ensure_runtime_device()

    def _current_runtime_device(self) -> Optional[str]:
        if self._runtime is None:
            return None
        try:
            first_param = next(self._runtime.model.parameters())
            return str(first_param.device)
        except Exception:
            return str(getattr(self._runtime, "device", self.device))

    def _move_runtime_to(self, target_device: str):
        if self._runtime is None:
            return

        device_obj = torch.device(target_device)
        runtime = self._runtime
        runtime.model = runtime.model.to(device_obj).eval()
        runtime.device = device_obj

    def _ensure_runtime_device(self):
        if self._runtime is None:
            return
        target_device = resolve_torch_device(self.device)
        current_device = self._current_runtime_device()
        if self._devices_equivalent(current_device, target_device):
            return
        if current_device != target_device:
            print(f"🔄 Dots TTS: moving runtime from {current_device} to {target_device}")
            self._move_runtime_to(target_device)

    @staticmethod
    def _devices_equivalent(current_device: Optional[str], target_device: Optional[str]) -> bool:
        if current_device == target_device:
            return True
        if not current_device or not target_device:
            return False
        current = str(current_device)
        target = str(target_device)
        if current.startswith("cuda") and target.startswith("cuda"):
            return current in {"cuda", "cuda:0"} and target in {"cuda", "cuda:0"}
        return False

    def to(self, device):
        """ComfyUI unload/reload hook."""
        self.device = str(device) if isinstance(device, str) else str(torch.device(device))
        self._ensure_runtime_device()
        return self

    def generate(
        self,
        text: str,
        prompt_audio_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        language: Optional[str] = None,
        num_steps: int = 10,
        guidance_scale: float = 1.2,
        speaker_scale: float = 1.5,
        normalize_text: bool = False,
        template_name: Optional[str] = "tts",
        seed: int = 42,
    ) -> Dict[str, Any]:
        self._ensure_runtime_loaded()
        self._ensure_runtime_device()

        seed_value = int(seed or 0)
        if seed_value > 0:
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)

        chunks = []
        tracker = DotsTTSProgressTracker(
            max_generate_length=self.max_generate_length,
            sample_rate=self.SAMPLE_RATE,
        )
        start_time = time.time()

        with self._patch_tokenizer_regex_fix(), self._suppress_noisy_runtime_logs():
            stream = self._runtime.generate_stream(
                text=text,
                prompt_audio_path=prompt_audio_path,
                prompt_text=prompt_text,
                language=language,
                template_name=template_name,
                num_steps=int(num_steps),
                guidance_scale=float(guidance_scale),
                speaker_scale=float(speaker_scale),
                normalize_text=bool(normalize_text),
            )
            for chunk in stream:
                if not isinstance(chunk, torch.Tensor):
                    chunk = torch.tensor(chunk, dtype=torch.float32)
                chunk = chunk.detach().float().cpu()
                if chunk.dim() > 1:
                    chunk = chunk.squeeze()
                chunks.append(chunk)
                tracker.update(chunk.shape[-1] if chunk.numel() > 0 else 0)

        tracker.end()

        if chunks:
            audio = torch.cat(chunks, dim=-1)
        else:
            audio = torch.zeros(0, dtype=torch.float32)

        time_used = time.time() - start_time
        duration_seconds = audio.shape[-1] / self.SAMPLE_RATE if audio.numel() > 0 else 0.0
        rtf = time_used / duration_seconds if duration_seconds > 0 else float("inf")

        return {
            "fid": None,
            "audio": audio,
            "sample_rate": self.SAMPLE_RATE,
            "time_used": time_used,
            "rtf": rtf,
            "profiling": None,
        }
