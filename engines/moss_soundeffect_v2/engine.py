"""Embedded MOSS-SoundEffect v2 engine for the configured ComfyUI environment."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

import torch

from utils.device import resolve_torch_device


class MossSoundEffectV2Engine:
    """Small lifecycle wrapper around the official v2 diffusion pipeline."""

    def __init__(self, model_path: str, device: str = "auto", dtype: str = "auto"):
        self.model_path = str(model_path)
        self.device_name = resolve_torch_device(device)
        self.dtype_name = dtype
        self.pipeline = None

    @property
    def dtype(self) -> torch.dtype:
        return self._resolve_dtype()

    def _resolve_dtype(self) -> torch.dtype:
        if not str(self.device_name).startswith("cuda"):
            return torch.float32
        explicit = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if self.dtype_name in explicit:
            return explicit[self.dtype_name]
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    @staticmethod
    def _load_pipeline_class():
        impl_dir = Path(__file__).resolve().parent / "impl"
        impl_path = str(impl_dir)
        if impl_path not in sys.path:
            sys.path.insert(0, impl_path)
        from moss_soundeffect_v2 import MossSoundEffectPipeline

        return MossSoundEffectPipeline

    def _ensure_model_loaded(self):
        if self.pipeline is not None:
            return

        pipeline_class = self._load_pipeline_class()
        torch_dtype = self._resolve_dtype()
        print("🔄 Loading MOSS-SoundEffect v2 via unified interface")
        print(f"   Path: {self.model_path}")
        print(f"   Device: {self.device_name} | Dtype: {torch_dtype}")
        self.pipeline = pipeline_class.from_pretrained(
            self.model_path,
            device=self.device_name,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        print("✅ MOSS-SoundEffect v2 model loaded")

    @staticmethod
    def _interruptible_progress(iterable: Iterable):
        """Keep official tqdm progress while honoring ComfyUI cancellation."""
        from tqdm.auto import tqdm

        for item in tqdm(iterable, desc="Generating MOSS-SoundEffect v2"):
            try:
                import comfy.model_management as model_management

                checker = getattr(model_management, "throw_exception_if_processing_interrupted", None)
                if callable(checker):
                    checker()
                elif getattr(model_management, "interrupt_processing", False):
                    raise InterruptedError("MOSS-SoundEffect v2 generation interrupted by user")
            except ImportError:
                pass
            yield item

    def generate_sound_effect(
        self,
        description: str,
        duration_seconds: float,
        inference_steps: int = 100,
        cfg_scale: float = 4.0,
        sigma_shift: float = 5.0,
        negative_prompt: str = "",
        seed: int = 0,
        **_unused,
    ) -> tuple[torch.Tensor, int]:
        self._ensure_model_loaded()
        waveform = self.pipeline(
            prompt=str(description),
            seconds=float(duration_seconds),
            num_inference_steps=int(inference_steps),
            cfg_scale=float(cfg_scale),
            sigma_shift=float(sigma_shift),
            negative_prompt=str(negative_prompt or ""),
            seed=int(seed),
            progress_bar_cmd=self._interruptible_progress,
        )
        return waveform.detach().cpu().float(), int(self.pipeline.sample_rate)

    def to(self, device):
        self.device_name = str(device)
        if self.pipeline is not None:
            self.pipeline.to(device)
        return self

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        if self.pipeline is None:
            return
        engine = getattr(self.pipeline, "engine", None)
        if engine is not None and hasattr(engine, "parameters"):
            yield from engine.parameters()

    def cleanup(self):
        if self.pipeline is None:
            return
        try:
            self.pipeline.to("cpu")
        except Exception:
            pass
        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release(self):
        """Drop model references without copying a large pipeline into system RAM."""
        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["MossSoundEffectV2Engine"]
