"""
Dots TTS engine adapter.

Translates TTS Audio Suite processor calls into the official dots.tts runtime.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio.audio_hash import generate_stable_audio_component
from utils.audio.cache import get_audio_cache
from utils.audio.processing import AudioProcessingUtils
from utils.models.factory_config import ModelLoadConfig
from utils.models.language_mapper import resolve_language_alias
from engines.dots_tts.languages import normalize_dots_language


class DotsTTSEngineAdapter:
    """Adapter for official Dots TTS inference."""

    SAMPLE_RATE = 48000

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config.copy() if config else {}
        self.audio_cache = get_audio_cache()
        self._last_config: Optional[ModelLoadConfig] = None
        self._load_signature = None

    def update_config(self, new_config: Dict[str, Any]):
        self.config = new_config.copy() if new_config else {}

    def _build_load_signature(self) -> Tuple[Any, ...]:
        return (
            self.config.get("model_variant", "dots.tts-soar"),
            self.config.get("device", "auto"),
            self.config.get("precision", "auto"),
            bool(self.config.get("optimize", False)),
            int(self.config.get("max_generate_length", 500)),
        )

    def load_model(
        self,
        model_variant: str,
        device: str = "auto",
        precision: str = "auto",
        optimize: bool = False,
        max_generate_length: int = 500,
    ):
        from utils.models.unified_model_interface import unified_model_interface

        config = ModelLoadConfig(
            engine_name="dots_tts",
            model_type="tts",
            model_name=model_variant,
            model_path=model_variant,
            device=device,
            additional_params={
                "precision": precision,
                "optimize": bool(optimize),
                "max_generate_length": int(max_generate_length),
            },
        )
        self._last_config = config
        unified_model_interface.load_model(config)

    def _ensure_model_loaded(self):
        signature = self._build_load_signature()
        if signature == self._load_signature and self._last_config is not None:
            return

        self.load_model(
            model_variant=self.config.get("model_variant", "dots.tts-soar"),
            device=self.config.get("device", "auto"),
            precision=self.config.get("precision", "auto"),
            optimize=self.config.get("optimize", False),
            max_generate_length=self.config.get("max_generate_length", 500),
        )
        self._load_signature = signature

    def _get_engine(self):
        if self._last_config is None:
            self._ensure_model_loaded()
        from utils.models.unified_model_interface import unified_model_interface

        return unified_model_interface.load_model(self._last_config)

    def _extract_voice_reference(self, voice_ref: Optional[Dict[str, Any]]) -> Tuple[Optional[str], str, str]:
        if not voice_ref or not isinstance(voice_ref, dict):
            return None, "", "default_voice"

        prompt_text = (
            voice_ref.get("reference_text")
            or voice_ref.get("prompt_text")
            or voice_ref.get("text")
            or ""
        ).strip()

        ref_audio = (
            voice_ref.get("prompt_audio_path")
            or voice_ref.get("audio_path")
            or voice_ref.get("audio")
            or voice_ref.get("waveform")
        )

        if ref_audio is None:
            return None, prompt_text, "default_voice"

        if isinstance(ref_audio, str):
            return ref_audio, prompt_text, generate_stable_audio_component(audio_file_path=ref_audio)

        if isinstance(ref_audio, dict) and "waveform" in ref_audio:
            temp_path = AudioProcessingUtils.save_audio_to_temp_file(
                ref_audio["waveform"],
                ref_audio.get("sample_rate", self.SAMPLE_RATE),
            )
            return temp_path, prompt_text, generate_stable_audio_component(reference_audio=ref_audio)

        if torch.is_tensor(ref_audio):
            sample_rate = int(voice_ref.get("sample_rate", self.SAMPLE_RATE))
            audio_dict = {"waveform": ref_audio, "sample_rate": sample_rate}
            temp_path = AudioProcessingUtils.save_audio_to_temp_file(ref_audio, sample_rate)
            return temp_path, prompt_text, generate_stable_audio_component(reference_audio=audio_dict)

        raise TypeError(f"Unsupported Dots TTS voice reference type: {type(ref_audio)}")

    @staticmethod
    def _normalize_language(language: Optional[str]) -> Optional[str]:
        if language is None:
            return None
        normalized = str(language).strip()
        if not normalized:
            return None
        normalized_language = normalize_dots_language(normalized)
        if normalized_language is not None:
            return normalized_language
        resolved = resolve_language_alias(normalized)
        return normalize_dots_language(resolved)

    def generate_single(
        self,
        text: str,
        voice_ref: Optional[Dict[str, Any]],
        seed: int = 0,
        enable_audio_cache: bool = True,
        character_name: Optional[str] = None,
    ) -> torch.Tensor:
        stripped = (text or "").strip()
        if not stripped:
            return torch.zeros(1, 0, dtype=torch.float32)

        self._ensure_model_loaded()
        engine = self._get_engine()

        prompt_audio_path, prompt_text, audio_component = self._extract_voice_reference(voice_ref)
        language = self._normalize_language(self.config.get("language", "auto"))
        model_variant = self.config.get("model_variant", "dots.tts-soar")
        template_name = self.config.get("template_name", "tts")
        num_steps = int(self.config.get("num_steps", 10))
        guidance_scale = float(self.config.get("guidance_scale", 1.2))
        speaker_scale = float(self.config.get("speaker_scale", 1.5))
        normalize_text = bool(self.config.get("normalize_text", False))
        precision = self.config.get("precision", "auto")
        optimize = bool(self.config.get("optimize", False))
        max_generate_length = int(self.config.get("max_generate_length", 500))

        cache_key = None
        if enable_audio_cache:
            cache_key = self.audio_cache.generate_cache_key(
                "dots_tts",
                text=stripped,
                audio_component=audio_component,
                prompt_text=prompt_text,
                model_variant=model_variant,
                language=language or "none",
                template_name=template_name,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
                normalize_text=normalize_text,
                max_generate_length=max_generate_length,
                precision=precision,
                optimize=optimize,
                seed=int(seed or 0),
                device=self.config.get("device", "auto"),
                character=character_name or "narrator",
            )
            cached_audio = self.audio_cache.get_cached_audio(cache_key)
            if cached_audio:
                print(f"💾 Using cached Dots TTS audio for '{character_name or 'narrator'}': '{stripped[:30]}...'")
                return cached_audio[0]

        result = engine.generate(
            text=stripped,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text or None,
            language=language,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            speaker_scale=speaker_scale,
            normalize_text=normalize_text,
            template_name=template_name,
            seed=int(seed or 0),
        )

        audio_tensor = result["audio"]
        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)
        audio_tensor = audio_tensor.detach().float().cpu()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        if enable_audio_cache and cache_key:
            duration = self.audio_cache._calculate_duration(audio_tensor, "dots_tts")
            self.audio_cache.cache_audio(cache_key, audio_tensor, duration)

        return audio_tensor
