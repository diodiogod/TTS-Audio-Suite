"""Higgs Audio v3 adapter for the unified TTS pipeline."""

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

from utils.audio.audio_hash import generate_stable_audio_component
from utils.audio.cache import get_audio_cache
from utils.models.factory_config import ModelLoadConfig
from utils.text.pause_processor import PauseTagProcessor


class HiggsAudioV3EngineAdapter:
    """Adapter for native Higgs Audio v3 generation."""

    SAMPLE_RATE = 24000

    def __init__(self, node_instance=None):
        self.node = node_instance
        self.audio_cache = get_audio_cache()
        self._last_config: Optional[ModelLoadConfig] = None

    def load_model(
        self,
        model: str,
        device: str = "auto",
        dtype: str = "auto",
        attention: str = "auto",
    ):
        from utils.models.unified_model_interface import unified_model_interface

        config = ModelLoadConfig(
            engine_name="higgs_audio_v3",
            model_type="tts",
            model_name=model,
            model_path=model,
            device=device,
            additional_params={
                "dtype": dtype,
                "attention": attention,
            },
        )
        self._last_config = config
        unified_model_interface.load_model(config)

    def update_model_config(
        self,
        model: str,
        device: str = "auto",
        dtype: str = "auto",
        attention: str = "auto",
    ):
        self.load_model(model, device, dtype, attention)

    def _get_engine(self):
        if self._last_config is None:
            raise RuntimeError("Higgs Audio v3 model has not been loaded")
        from utils.models.unified_model_interface import unified_model_interface

        return unified_model_interface.load_model(self._last_config)

    def generate_with_pause_tags(
        self,
        text: str,
        voice_ref: Optional[Dict[str, Any]],
        params: Dict[str, Any],
        process_pauses: bool = True,
        character_name: Optional[str] = None,
    ) -> torch.Tensor:
        engine = self._get_engine()
        if process_pauses and PauseTagProcessor.has_pause_tags(text):
            return self._generate_with_pauses(text, voice_ref, params, character_name, engine)
        return self._generate_direct(text, voice_ref, params, character_name, engine)

    def _generate_direct(
        self,
        text: str,
        voice_ref: Optional[Dict[str, Any]],
        params: Dict[str, Any],
        character_name: Optional[str] = None,
        engine=None,
    ) -> torch.Tensor:
        if engine is None:
            engine = self._get_engine()

        reference_audio, reference_text, audio_component = self._extract_voice_reference(voice_ref)
        model_name = params.get("model", "higgs-audio-v3-tts-4b")
        seed = int(params.get("seed", 0) or 0)
        max_new_tokens = int(params.get("max_new_tokens", 2048))
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 0.95))
        top_k = int(params.get("top_k", 50))

        cache_key = self.audio_cache.generate_cache_key(
            "higgs_audio_v3",
            text=text,
            model_variant=model_name,
            audio_component=audio_component,
            reference_text=reference_text,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            seed=seed,
            device=params.get("device", "auto"),
            dtype=params.get("dtype", "auto"),
            attention=params.get("attention", "auto"),
            character=character_name or "narrator",
        )

        if params.get("enable_audio_cache", True):
            cached_audio = self.audio_cache.get_cached_audio(cache_key)
            if cached_audio:
                print(f"💾 Using cached Higgs Audio v3 audio for '{character_name or 'narrator'}': '{text[:30]}...'")
                return cached_audio[0]

        audio_tensor, sample_rate = engine.generate(
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )

        if sample_rate != self.SAMPLE_RATE:
            raise RuntimeError(f"Higgs Audio v3 returned {sample_rate} Hz; expected {self.SAMPLE_RATE} Hz")

        if params.get("enable_audio_cache", True):
            duration = self.audio_cache._calculate_duration(audio_tensor, "higgs_audio_v3")
            self.audio_cache.cache_audio(cache_key, audio_tensor, duration)
        return audio_tensor

    def _generate_with_pauses(
        self,
        text: str,
        voice_ref: Optional[Dict[str, Any]],
        params: Dict[str, Any],
        character_name: Optional[str] = None,
        engine=None,
    ) -> torch.Tensor:
        segments, _ = PauseTagProcessor.parse_pause_tags(text)
        audio_parts = []
        for segment_type, content in segments:
            if segment_type == "text":
                audio_tensor = self._generate_direct(content, voice_ref, params, character_name, engine)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                elif audio_tensor.dim() == 3:
                    audio_tensor = audio_tensor.squeeze(0)
                audio_parts.append(audio_tensor)
            elif segment_type == "pause":
                audio_parts.append(PauseTagProcessor.create_silence_segment(content, self.SAMPLE_RATE))

        if not audio_parts:
            return torch.zeros(1, 0)
        return torch.cat(audio_parts, dim=-1)

    def _extract_voice_reference(self, voice_ref: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str, str]:
        if not voice_ref or not isinstance(voice_ref, dict):
            return None, "", "zero_shot"

        ref_audio = None
        for key in ("audio", "audio_dict", "reference_audio", "waveform"):
            if key in voice_ref and voice_ref.get(key) is not None:
                ref_audio = voice_ref.get(key)
                break
        reference_text = (
            voice_ref.get("reference_text")
            or voice_ref.get("text")
            or voice_ref.get("prompt_text")
            or ""
        )

        if ref_audio is None and "audio_path" in voice_ref:
            audio_path = voice_ref.get("audio_path")
            if audio_path and os.path.exists(audio_path):
                try:
                    from utils.audio.processing import AudioProcessingUtils

                    waveform, sample_rate = AudioProcessingUtils.safe_load_audio(audio_path)
                    ref_audio = {"waveform": waveform, "sample_rate": sample_rate}
                except Exception as e:
                    print(f"⚠️ Failed to load Higgs Audio v3 reference audio {audio_path}: {e}")

        if isinstance(ref_audio, dict) and "waveform" in ref_audio:
            return ref_audio, str(reference_text), generate_stable_audio_component(reference_audio=ref_audio)

        if torch.is_tensor(ref_audio):
            sample_rate = int(voice_ref.get("sample_rate", self.SAMPLE_RATE))
            audio_dict = {"waveform": ref_audio, "sample_rate": sample_rate}
            return audio_dict, str(reference_text), generate_stable_audio_component(reference_audio=audio_dict)

        return None, str(reference_text), "zero_shot"
