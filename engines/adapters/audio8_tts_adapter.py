"""Adapter between suite processors and the official Audio8 TTS engine."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch

from engines.audio8_tts.downloader import Audio8TTSDownloader
from utils.audio.audio_hash import generate_stable_audio_component
from utils.audio.cache import get_audio_cache
from utils.device import resolve_torch_device
from utils.models.factory_config import ModelLoadConfig, RUNTIME_MODE_SHARED
from utils.voice.character_logging import resolved_character_label
from utils.voice.reference import effective_voice_audio


class Audio8TTSEngineAdapter:
    """Translate unified TTS calls into Audio8's official inference API."""

    SAMPLE_RATE = 44100

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self.audio_cache = get_audio_cache()
        self.downloader = Audio8TTSDownloader()
        self._last_config: Optional[ModelLoadConfig] = None
        self._load_signature: Optional[Tuple[Any, ...]] = None

    def update_config(self, new_config: Dict[str, Any]):
        self.config = dict(new_config or {})

    def _model_selection(self) -> str:
        return str(
            self.config.get(
                "model_variant",
                self.config.get(
                    "model_name",
                    Audio8TTSDownloader.MODEL_NAME,
                ),
            )
            or Audio8TTSDownloader.MODEL_NAME
        )

    def _build_load_signature(self) -> Tuple[Any, ...]:
        return (
            self._model_selection(),
            resolve_torch_device(self.config.get("device", "auto")),
            self.config.get("dtype", "auto"),
        )

    def load_model(
        self,
        model_variant: str,
        device: str = "auto",
        dtype: str = "auto",
    ):
        """Resolve the organized model path and load through the unified factory."""
        from utils.models.unified_model_interface import unified_model_interface

        model_path = self.downloader.resolve_model_path(model_variant)
        config = ModelLoadConfig(
            engine_name="audio8_tts",
            model_type="tts",
            model_name=model_variant,
            model_path=model_path,
            device=device,
            additional_params={"dtype": dtype},
            runtime_mode=RUNTIME_MODE_SHARED,
            runtime_profile="vibevoice_transformers4_shared",
        )
        self._last_config = config
        return unified_model_interface.load_model(config)

    def _ensure_model_loaded(self):
        signature = self._build_load_signature()
        if signature != self._load_signature or self._last_config is None:
            self.load_model(
                model_variant=self._model_selection(),
                device=self.config.get("device", "auto"),
                dtype=self.config.get("dtype", "auto"),
            )
            self._load_signature = signature

    def _get_engine(self):
        if self._last_config is None:
            self._ensure_model_loaded()
        from utils.models.unified_model_interface import unified_model_interface

        return unified_model_interface.load_model(self._last_config)

    @staticmethod
    def _mono_waveform(waveform: Any) -> torch.Tensor:
        audio = torch.as_tensor(waveform, dtype=torch.float32).detach().cpu()
        if audio.ndim == 3:
            if audio.shape[0] != 1:
                raise ValueError(
                    "Audio8 TTS reference audio must contain one batch item"
                )
            audio = audio[0]
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        if audio.ndim != 1 or audio.numel() == 0:
            raise ValueError(
                "Audio8 TTS reference audio must be non-empty mono or "
                "channels-first audio"
            )
        return audio.contiguous()

    def _in_memory_reference(
        self,
        waveform: Any,
        sample_rate: Any,
    ) -> Tuple[Dict[str, Any], str]:
        mono = self._mono_waveform(waveform)
        sample_rate = int(sample_rate)
        comfy_audio = {
            "waveform": mono.unsqueeze(0),
            "sample_rate": sample_rate,
        }
        processor_audio = {
            "array": mono,
            "sampling_rate": sample_rate,
        }
        return (
            processor_audio,
            generate_stable_audio_component(reference_audio=comfy_audio),
        )

    def _extract_voice_reference(
        self,
        voice_ref: Optional[Dict[str, Any]],
    ) -> Tuple[Any, str, str]:
        if not isinstance(voice_ref, dict):
            return None, "", "default_voice"

        reference_text = str(
            voice_ref.get("reference_text")
            or voice_ref.get("prompt_text")
            or voice_ref.get("text")
            or ""
        ).strip()
        reference_audio = effective_voice_audio(voice_ref)
        if reference_audio is None:
            return None, reference_text, "default_voice"

        if isinstance(reference_audio, (str, os.PathLike)):
            path = os.fspath(reference_audio)
            component = generate_stable_audio_component(audio_file_path=path)
            return path, reference_text, component

        if isinstance(reference_audio, dict):
            waveform = reference_audio.get(
                "waveform",
                reference_audio.get("array"),
            )
            sample_rate = reference_audio.get(
                "sample_rate",
                reference_audio.get("sampling_rate"),
            )
            if waveform is None or sample_rate is None:
                raise ValueError(
                    "Audio8 TTS in-memory reference requires waveform and sample_rate"
                )
            normalized, component = self._in_memory_reference(
                waveform,
                sample_rate,
            )
            return normalized, reference_text, component

        if isinstance(reference_audio, (tuple, list)) and len(reference_audio) == 2:
            normalized, component = self._in_memory_reference(
                reference_audio[0],
                reference_audio[1],
            )
            return normalized, reference_text, component

        if torch.is_tensor(reference_audio):
            normalized, component = self._in_memory_reference(
                reference_audio,
                voice_ref.get("sample_rate", self.SAMPLE_RATE),
            )
            return normalized, reference_text, component

        raise TypeError(
            f"Unsupported Audio8 TTS reference type: {type(reference_audio)}"
        )

    def generate_single(
        self,
        text: str,
        voice_ref: Optional[Dict[str, Any]],
        seed: int = 0,
        enable_audio_cache: bool = True,
        character_name: Optional[str] = None,
    ) -> torch.Tensor:
        """Generate one raw utterance; chunking remains processor-owned."""
        text = str(text or "").strip()
        if not text:
            return torch.zeros(1, 0, dtype=torch.float32)

        reference_audio, reference_text, audio_component = (
            self._extract_voice_reference(voice_ref)
        )
        if reference_audio is not None and not reference_text:
            raise ValueError(
                "Audio8 TTS voice cloning requires the exact transcript of "
                "the reference audio. Add reference text to Character Voices "
                "or the narrator voice."
            )

        params = {
            "model_variant": self._model_selection(),
            "max_new_tokens": int(self.config.get("max_new_tokens", 1024)),
            "retry_max_new_tokens": int(self.config.get("retry_max_new_tokens", 2000)),
            "temperature": float(self.config.get("temperature", 0.8)),
            "top_p": float(self.config.get("top_p", 0.95)),
            "top_k": int(self.config.get("top_k", 50)),
            "do_sample": bool(self.config.get("do_sample", True)),
            "dtype": self.config.get("dtype", "auto"),
            "device": resolve_torch_device(self.config.get("device", "auto")),
            "seed": int(seed),
        }

        cache_key = None
        if enable_audio_cache:
            cache_key = self.audio_cache.generate_cache_key(
                "audio8_tts",
                text=text,
                audio_component=audio_component,
                reference_text=reference_text,
                character=character_name or "narrator",
                **params,
            )
            cached = self.audio_cache.get_cached_audio(cache_key)
            if cached:
                display_name = resolved_character_label(
                    character_name or "narrator",
                    voice_ref,
                )
                print(
                    "💾 Using cached Audio8 TTS audio for "
                    f"'{display_name}': '{text[:30]}...'"
                )
                return cached[0]

        self._ensure_model_loaded()
        audio = self._get_engine().generate(
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text or None,
            max_new_tokens=params["max_new_tokens"],
            retry_max_new_tokens=params["retry_max_new_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            top_k=params["top_k"],
            do_sample=params["do_sample"],
            seed=params["seed"],
        )
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio, dtype=torch.float32)
        audio = audio.detach().float().cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim != 2:
            raise RuntimeError(
                f"Audio8 TTS returned invalid audio shape: {tuple(audio.shape)}"
            )

        if cache_key:
            self.audio_cache.cache_audio(
                cache_key,
                audio,
                audio.shape[-1] / self.SAMPLE_RATE,
            )
        return audio
