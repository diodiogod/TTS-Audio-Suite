"""
OmniVoice engine adapter.

Translates TTS Audio Suite processor calls into the official OmniVoice runtime.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import torch

current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio.audio_hash import generate_stable_audio_component
from utils.audio.cache import get_audio_cache
from utils.models.factory_config import ModelLoadConfig
from utils.models.language_mapper import resolve_language_alias
from utils.voice.reference import effective_voice_audio


class OmniVoiceEngineAdapter:
    """Adapter for official OmniVoice inference."""

    SAMPLE_RATE = 24000

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config.copy() if config else {}
        self.audio_cache = get_audio_cache()
        self._last_config: Optional[ModelLoadConfig] = None
        self._load_signature = None

    def update_config(self, new_config: Dict[str, Any]):
        self.config = new_config.copy() if new_config else {}

    def _build_load_signature(self) -> Tuple[Any, ...]:
        return (
            self.config.get("model_variant", "OmniVoice"),
            self.config.get("device", "auto"),
            self.config.get("dtype", "auto"),
        )

    def load_model(
        self,
        model_variant: str,
        device: str = "auto",
        dtype: str = "auto",
    ):
        from utils.models.unified_model_interface import unified_model_interface

        config = ModelLoadConfig(
            engine_name="omnivoice",
            model_type="tts",
            model_name=model_variant,
            model_path=model_variant,
            device=device,
            additional_params={
                "dtype": dtype,
            },
        )
        self._last_config = config
        unified_model_interface.load_model(config)

    def _ensure_model_loaded(self):
        signature = self._build_load_signature()
        if signature == self._load_signature and self._last_config is not None:
            return

        self.load_model(
            model_variant=self.config.get("model_variant", "OmniVoice"),
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
    def _seed_everything(seed: int):
        if int(seed or 0) <= 0:
            return
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(int(seed))

    @staticmethod
    def _first_non_none(*values):
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return None

    @staticmethod
    def _normalize_waveform_tensor(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform.detach().float().cpu()

    def _extract_voice_reference(
        self,
        voice_ref: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Union[str, Tuple[torch.Tensor, int]]], str, str]:
        if not voice_ref or not isinstance(voice_ref, dict):
            return None, "", "default_voice"

        prompt_text = (
            voice_ref.get("reference_text")
            or voice_ref.get("prompt_text")
            or voice_ref.get("text")
            or ""
        ).strip()

        ref_audio = effective_voice_audio(voice_ref)

        if ref_audio is None:
            return None, prompt_text, "default_voice"

        if isinstance(ref_audio, str):
            return ref_audio, prompt_text, generate_stable_audio_component(audio_file_path=ref_audio)

        if isinstance(ref_audio, dict) and "waveform" in ref_audio:
            waveform = self._normalize_waveform_tensor(ref_audio["waveform"])
            sample_rate = int(ref_audio.get("sample_rate", self.SAMPLE_RATE))
            audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
            return (
                (waveform, sample_rate),
                prompt_text,
                generate_stable_audio_component(reference_audio=audio_dict),
            )

        if torch.is_tensor(ref_audio):
            waveform = self._normalize_waveform_tensor(ref_audio)
            sample_rate = int(voice_ref.get("sample_rate", self.SAMPLE_RATE))
            audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
            return (
                (waveform, sample_rate),
                prompt_text,
                generate_stable_audio_component(reference_audio=audio_dict),
            )

        raise TypeError(f"Unsupported OmniVoice reference type: {type(ref_audio)}")

    @staticmethod
    def _normalize_language(language: Optional[str]) -> Optional[str]:
        if language is None:
            return None

        normalized = str(language).strip()
        if not normalized:
            return None

        lowered = normalized.lower()
        if lowered in {"auto", "none"}:
            return None

        resolved = resolve_language_alias(normalized)

        # OmniVoice uses the base ISO code for Portuguese rather than regional
        # variants accepted by other suite engines.
        if resolved in {"pt", "pt-br", "pt-pt"}:
            return "pt"

        if resolved and resolved.lower() != lowered:
            return resolved
        return normalized

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

        ref_audio, prompt_text, audio_component = self._extract_voice_reference(voice_ref)
        language = self._normalize_language(self.config.get("language", "Auto"))
        instruct = str(self.config.get("instruct", "") or "").strip() or None
        model_variant = self.config.get("model_variant", "OmniVoice")
        device = self.config.get("device", "auto")
        dtype = self.config.get("dtype", "auto")
        num_step = int(self.config.get("num_steps", 32))
        guidance_scale = float(self.config.get("guidance_scale", 2.0))
        t_shift = float(self.config.get("t_shift", 0.1))
        speed = float(self.config.get("speed", 1.0))
        duration_value = float(self.config.get("duration", 0.0))
        duration = duration_value if duration_value > 0 else None
        layer_penalty_factor = float(self.config.get("layer_penalty_factor", 5.0))
        position_temperature = float(self.config.get("position_temperature", 5.0))
        class_temperature = float(self.config.get("class_temperature", 0.0))
        denoise = bool(self.config.get("denoise", True))
        preprocess_prompt = bool(self.config.get("preprocess_prompt", True))
        postprocess_output = bool(self.config.get("postprocess_output", True))
        audio_chunk_duration = float(self.config.get("audio_chunk_duration", 15.0))
        audio_chunk_threshold = float(self.config.get("audio_chunk_threshold", 30.0))

        cache_key = None
        if enable_audio_cache:
            cache_key = self.audio_cache.generate_cache_key(
                "omnivoice",
                text=stripped,
                audio_component=audio_component,
                prompt_text=prompt_text,
                model_variant=model_variant,
                language=language or "none",
                instruct=instruct or "",
                num_step=num_step,
                guidance_scale=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                duration=duration if duration is not None else "auto",
                layer_penalty_factor=layer_penalty_factor,
                position_temperature=position_temperature,
                class_temperature=class_temperature,
                denoise=denoise,
                preprocess_prompt=preprocess_prompt,
                postprocess_output=postprocess_output,
                audio_chunk_duration=audio_chunk_duration,
                audio_chunk_threshold=audio_chunk_threshold,
                seed=int(seed or 0),
                device=device,
                dtype=dtype,
                character=character_name or "narrator",
            )
            cached_audio = self.audio_cache.get_cached_audio(cache_key)
            if cached_audio:
                print(f"💾 Using cached OmniVoice audio for '{character_name or 'narrator'}': '{stripped[:30]}...'")
                return cached_audio[0]

        voice_clone_prompt = None
        if ref_audio is not None:
            if not prompt_text:
                raise ValueError(
                    "OmniVoice voice cloning requires reference text. "
                    "Direct audio-only narrator input is unsupported. "
                    "Use Character Voices or a narrator voice file with matching reference text."
                )
            voice_clone_prompt = engine.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=prompt_text,
                preprocess_prompt=preprocess_prompt,
            )

        self._seed_everything(seed)

        result = engine.generate(
            text=stripped,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
            instruct=instruct,
            duration=duration,
            speed=speed,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
            denoise=denoise,
            preprocess_prompt=preprocess_prompt,
            postprocess_output=postprocess_output,
            layer_penalty_factor=layer_penalty_factor,
            position_temperature=position_temperature,
            class_temperature=class_temperature,
            audio_chunk_duration=audio_chunk_duration,
            audio_chunk_threshold=audio_chunk_threshold,
        )

        audio_tensor = result[0]
        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)
        audio_tensor = audio_tensor.detach().float().cpu()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        if enable_audio_cache and cache_key:
            duration_seconds = self.audio_cache._calculate_duration(audio_tensor, "omnivoice")
            self.audio_cache.cache_audio(cache_key, audio_tensor, duration_seconds)

        return audio_tensor
