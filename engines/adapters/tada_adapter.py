"""Adapter between TTS Audio Suite processors and the isolated TADA runtime."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.tada.languages import format_tada_language_display, validate_tada_model_language
from engines.tada.tada_downloader import TadaDownloader
from utils.audio.audio_hash import generate_stable_audio_component
from utils.audio.cache import get_audio_cache
from utils.device import resolve_torch_device
from utils.models.factory_config import ModelLoadConfig, RUNTIME_MODE_SHARED
from utils.voice.reference import effective_voice_audio


class TadaEngineAdapter:
    """Resolve assets, load the shared runtime proxy, and cache TADA waveforms."""

    SAMPLE_RATE = 24000

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config.copy() if config else {}
        self.audio_cache = get_audio_cache()
        self.downloader = TadaDownloader()
        self._last_config: Optional[ModelLoadConfig] = None
        self._load_signature: Optional[Tuple[Any, ...]] = None
        self._ensured_assets: Dict[Tuple[str, str], Dict[str, str]] = {}

    def update_config(self, new_config: Dict[str, Any]) -> None:
        self.config = new_config.copy() if new_config else {}

    def _model_identifier(self) -> str:
        return str(
            self.config.get("model_variant")
            or self.config.get("model_path")
            or self.config.get("model_name")
            or "TADA-1B"
        )

    def _ensure_assets(self, language: str) -> Dict[str, str]:
        identifier = self._model_identifier()
        display_language = format_tada_language_display(language)
        key = (identifier, display_language)
        assets = self._ensured_assets.get(key)
        if assets is None:
            assets = self.downloader.resolve_model_assets(identifier, display_language)
            self._ensured_assets[key] = assets
        return assets

    def _build_load_signature(self, assets: Dict[str, str]) -> Tuple[Any, ...]:
        resolved_device = resolve_torch_device(self.config.get("device", "auto"))
        return (
            assets["model_name"],
            os.path.abspath(assets["model_path"]),
            os.path.abspath(assets["codec_path"]),
            os.path.abspath(assets["tokenizer_path"]),
            resolved_device,
            self.config.get("dtype", "auto"),
            self.config.get("attn_implementation", "sdpa"),
            bool(self.config.get("use_torch_compile", False)),
            int(self.config.get("prompt_cache_size", 8)),
            self.config.get("runtime_mode", RUNTIME_MODE_SHARED),
            self.config.get("runtime_profile", "vibevoice_transformers4_shared"),
        )

    def _ensure_model_loaded(self, language: str) -> None:
        from utils.models.unified_model_interface import unified_model_interface

        assets = self._ensure_assets(language)
        signature = self._build_load_signature(assets)
        if signature == self._load_signature and self._last_config is not None:
            return

        resolved_device = resolve_torch_device(self.config.get("device", "auto"))
        load_config = ModelLoadConfig(
            engine_name="tada",
            model_type="tts",
            model_name=assets["model_name"],
            model_path=assets["model_path"],
            device=resolved_device,
            additional_params={
                "codec_path": assets["codec_path"],
                "tokenizer_path": assets["tokenizer_path"],
                "dtype": self.config.get("dtype", "auto"),
                "attn_implementation": self.config.get("attn_implementation", "sdpa"),
                "use_torch_compile": bool(self.config.get("use_torch_compile", False)),
                "prompt_cache_size": int(self.config.get("prompt_cache_size", 8)),
            },
            runtime_mode=self.config.get("runtime_mode", RUNTIME_MODE_SHARED),
            runtime_profile=self.config.get(
                "runtime_profile", "vibevoice_transformers4_shared"
            ),
        )
        self._last_config = load_config
        unified_model_interface.load_model(load_config)
        self._load_signature = signature

    def _get_engine(self):
        if self._last_config is None:
            raise RuntimeError("TADA model was requested before its assets were resolved")
        from utils.models.unified_model_interface import unified_model_interface

        return unified_model_interface.load_model(self._last_config)

    @staticmethod
    def _reference_text(voice_ref: Optional[Dict[str, Any]]) -> str:
        if not isinstance(voice_ref, dict):
            return ""
        return str(
            voice_ref.get("reference_text")
            or voice_ref.get("prompt_text")
            or voice_ref.get("text")
            or ""
        ).strip()

    @staticmethod
    def _audio_component(reference_audio: Any, voice_ref: Dict[str, Any]) -> str:
        if isinstance(reference_audio, str):
            return generate_stable_audio_component(audio_file_path=reference_audio)
        if isinstance(reference_audio, dict) and "waveform" in reference_audio:
            return generate_stable_audio_component(reference_audio=reference_audio)
        if torch.is_tensor(reference_audio):
            return generate_stable_audio_component(
                reference_audio={
                    "waveform": reference_audio,
                    "sample_rate": int(voice_ref.get("sample_rate", TadaEngineAdapter.SAMPLE_RATE)),
                }
            )
        if isinstance(reference_audio, (tuple, list)) and len(reference_audio) == 2:
            waveform, sample_rate = reference_audio
            if torch.is_tensor(waveform):
                return generate_stable_audio_component(
                    reference_audio={"waveform": waveform, "sample_rate": int(sample_rate)}
                )
        raise TypeError(f"Unsupported TADA reference audio type: {type(reference_audio)}")

    def generate_single(
        self,
        text: str,
        voice_ref: Optional[Dict[str, Any]],
        seed: int = 0,
        enable_audio_cache: bool = True,
        character_name: Optional[str] = None,
    ) -> torch.Tensor:
        stripped = str(text or "").strip()
        if not stripped:
            return torch.zeros(1, 0, dtype=torch.float32)
        if not isinstance(voice_ref, dict):
            raise ValueError("TADA requires reference audio and its exact transcript")

        reference_audio = effective_voice_audio(voice_ref)
        reference_text = self._reference_text(voice_ref)
        if reference_audio is None:
            raise ValueError("TADA requires reference audio; random-voice mode is not exposed")
        if not reference_text:
            raise ValueError(
                "TADA requires the exact reference transcript. Automatic Parakeet ASR is "
                "intentionally disabled to prevent a hidden 4.25 GB download."
            )

        model_variant = self._model_identifier()
        language = format_tada_language_display(self.config.get("language", "English"))
        validate_tada_model_language(model_variant, language)
        speed_up_factor = float(self.config.get("speed_up_factor", 0.0) or 0.0)
        native_speed = speed_up_factor if speed_up_factor > 0 else None
        audio_component = self._audio_component(reference_audio, voice_ref)

        generation_params = {
            "model_variant": model_variant,
            "decoder_source": "official_tada_codec_v1",
            "language": language,
            "reference_text": reference_text,
            "audio_component": audio_component,
            "seed": int(seed or 0),
            "acoustic_cfg_scale": float(self.config.get("acoustic_cfg_scale", 1.6)),
            "duration_cfg_scale": float(self.config.get("duration_cfg_scale", 1.0)),
            "noise_temperature": float(self.config.get("noise_temperature", 0.9)),
            "num_flow_matching_steps": int(self.config.get("num_flow_matching_steps", 10)),
            "cfg_schedule": self.config.get("cfg_schedule", "cosine"),
            "time_schedule": self.config.get("time_schedule", "logsnr"),
            "negative_condition_source": self.config.get(
                "negative_condition_source", "negative_step_output"
            ),
            "speed_up_factor": native_speed,
            "num_transition_steps": int(self.config.get("num_transition_steps", 5)),
            "device": self.config.get("device", "auto"),
            "dtype": self.config.get("dtype", "auto"),
            "runtime_mode": self.config.get("runtime_mode", RUNTIME_MODE_SHARED),
            "runtime_profile": self.config.get(
                "runtime_profile", "vibevoice_transformers4_shared"
            ),
            "character": character_name or "narrator",
        }

        cache_key = None
        if enable_audio_cache:
            cache_key = self.audio_cache.generate_cache_key(
                "tada", text=stripped, **generation_params
            )
            cached = self.audio_cache.get_cached_audio(cache_key)
            if cached:
                print(
                    f"💾 Using cached TADA audio for '{character_name or 'narrator'}': "
                    f"'{stripped[:30]}...'"
                )
                return cached[0]

        self._ensure_model_loaded(language)
        engine = self._get_engine()
        audio, sample_rate = engine.generate_speech(
            text=stripped,
            reference_audio=reference_audio,
            reference_text=reference_text,
            seed=generation_params["seed"],
            language=language,
            acoustic_cfg_scale=generation_params["acoustic_cfg_scale"],
            duration_cfg_scale=generation_params["duration_cfg_scale"],
            cfg_schedule=generation_params["cfg_schedule"],
            time_schedule=generation_params["time_schedule"],
            num_flow_matching_steps=generation_params["num_flow_matching_steps"],
            noise_temperature=generation_params["noise_temperature"],
            speed_up_factor=native_speed,
            num_transition_steps=generation_params["num_transition_steps"],
            negative_condition_source=generation_params["negative_condition_source"],
        )
        if int(sample_rate) != self.SAMPLE_RATE:
            raise RuntimeError(
                f"TADA returned {sample_rate} Hz audio; expected {self.SAMPLE_RATE} Hz"
            )
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio, dtype=torch.float32)
        audio = audio.detach().cpu().float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() != 2:
            raise RuntimeError(f"TADA returned an invalid audio shape: {tuple(audio.shape)}")
        if audio.shape[-1] == 0:
            raise RuntimeError("TADA returned empty audio")

        if enable_audio_cache and cache_key:
            duration = self.audio_cache._calculate_duration(audio, "tada")
            self.audio_cache.cache_audio(cache_key, audio, duration)
        return audio


__all__ = ["TadaEngineAdapter"]
