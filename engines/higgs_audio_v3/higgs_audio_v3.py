"""Higgs Audio v3 engine wrapper."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from .higgs_audio_v3_downloader import HiggsAudioV3Downloader

logger = logging.getLogger(__name__)


@dataclass
class HiggsAudioV3Bundle:
    model: torch.nn.Module
    codec: Any
    tokenizer: Any
    model_dir: Path
    device: torch.device
    torch_dtype: torch.dtype
    dtype_name: str
    attention: str


class HiggsAudioV3Engine:
    """Native Higgs Audio v3 TTS engine."""

    SAMPLE_RATE = 24000

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        attention: str = "auto",
    ):
        self.model_path = model_path
        self.device_name = device
        self.dtype_name = dtype
        self.attention_name = attention
        self.downloader = HiggsAudioV3Downloader()
        self.bundle: Optional[HiggsAudioV3Bundle] = None
        self._load()

    def _resolve_device(self) -> torch.device:
        if self.device_name == "auto":
            try:
                import comfy.model_management as model_management

                return torch.device(model_management.get_torch_device())
            except Exception:
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(self.device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected for Higgs Audio v3, but torch.cuda is not available.")
        return device

    def _resolve_dtype(self, device: torch.device) -> torch.dtype:
        if self.dtype_name == "auto":
            if device.type == "cuda":
                try:
                    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
                except Exception:
                    return torch.float32
            return torch.float32
        if self.dtype_name == "bf16":
            if device.type == "cuda":
                try:
                    if not torch.cuda.is_bf16_supported():
                        raise RuntimeError("bf16 was selected but this CUDA device does not report bf16 support.")
                except RuntimeError:
                    raise
                except Exception:
                    pass
            return torch.bfloat16
        if self.dtype_name == "fp32":
            return torch.float32
        raise ValueError(f"Unsupported Higgs Audio v3 dtype: {self.dtype_name}")

    def _resolve_attention(self) -> tuple[str, str | None]:
        attention = self.attention_name or "auto"
        if attention in {"auto", "sdpa"}:
            return "sdpa", "sdpa"
        if attention == "eager":
            return "eager", "eager"
        if attention == "flash_attention":
            if importlib.util.find_spec("flash_attn") is None:
                raise ImportError("flash_attention was selected for Higgs Audio v3, but flash_attn is not installed.")
            return "flash_attention", "flash_attention_2"
        if attention == "sageattention":
            if importlib.util.find_spec("sageattention") is None:
                raise ImportError("sageattention was selected for Higgs Audio v3, but sageattention is not installed.")
            return "sageattention", "sdpa"
        raise ValueError(f"Unsupported Higgs Audio v3 attention mode: {attention}")

    def _load(self) -> None:
        from .native import (
            HiggsAudioCodec,
            build_native_model,
            load_native_weights,
            load_tokenizer,
            read_config,
        )

        model_dir = Path(self.downloader.resolve_model_path(self.model_path))
        device = self._resolve_device()
        torch_dtype = self._resolve_dtype(device)
        runtime_attention, hf_attention = self._resolve_attention()

        config = read_config(model_dir)
        print(f"🔄 Loading Higgs Audio v3: {model_dir}")
        print(f"   Device: {device} | Dtype: {torch_dtype} | Attention: {runtime_attention}")

        model = build_native_model(config, torch_dtype, hf_attention)
        load_native_weights(model, model_dir, device, torch_dtype)
        codec = HiggsAudioCodec.from_pretrained(model_dir, device=device, dtype=torch_dtype)
        tokenizer = load_tokenizer(model_dir)

        self.bundle = HiggsAudioV3Bundle(
            model=model,
            codec=codec,
            tokenizer=tokenizer,
            model_dir=model_dir,
            device=device,
            torch_dtype=torch_dtype,
            dtype_name=self.dtype_name,
            attention=runtime_attention,
        )
        print("✅ Higgs Audio v3 model loaded")

    def to(self, device):
        """Move model components for ComfyUI model management."""
        if self.bundle is None:
            return self
        device = torch.device(device)
        self.bundle.model.to(device)
        self.bundle.codec.model.to(device)
        self.bundle.device = device
        self.bundle.codec.device = device
        return self

    def parameters(self):
        if self.bundle is None:
            return
        if hasattr(self.bundle.model, "parameters"):
            yield from self.bundle.model.parameters()
        codec_model = getattr(self.bundle.codec, "model", None)
        if codec_model is not None and hasattr(codec_model, "parameters"):
            yield from codec_model.parameters()

    def generate(
        self,
        text: str,
        reference_audio: Optional[dict] = None,
        reference_text: str = "",
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        seed: int = 0,
        progress_callback=None,
    ) -> tuple[torch.Tensor, int]:
        if self.bundle is None:
            self._load()

        # If ComfyUI offloaded to CPU, keep all internal references coherent.
        try:
            first_param = next(self.bundle.model.parameters())
            if first_param.device != self.bundle.device:
                self.to(self.bundle.device)
        except Exception:
            pass

        from .native import generate_higgs_audio

        audio = generate_higgs_audio(
            self.bundle,
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            progress_callback=progress_callback,
        )
        waveform = audio["waveform"]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            return waveform.contiguous(), int(audio["sample_rate"])
        return waveform.view(1, -1).contiguous(), int(audio["sample_rate"])

    def cleanup(self):
        if self.bundle is None:
            return
        try:
            self.bundle.model.to("cpu")
            self.bundle.codec.model.to("cpu")
        except Exception:
            pass
        self.bundle = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
