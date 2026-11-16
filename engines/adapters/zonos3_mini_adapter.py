"""
ZONOS3-Mini engine adapter.

Translates TTS Audio Suite processor calls into native ZONOS3 GenerateRequests.
"""

import base64
import io
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.zonos3_mini.languages import DEFAULT_LANGUAGE, to_zonos_language
from engines.zonos3_mini.voices import DEFAULT_PRESET
from utils.audio.audio_hash import generate_stable_audio_component
from utils.audio.cache import get_audio_cache
from utils.audio.processing import AudioProcessingUtils
from utils.models.factory_config import ModelLoadConfig
from utils.voice.reference import effective_voice_audio


class Zonos3MiniEngineAdapter:
    """Adapter for the official ZONOS3-Mini reference inference."""

    SAMPLE_RATE = 44100
    MAX_VOICE_ENTRIES = 32

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config.copy() if config else {}
        self.audio_cache = get_audio_cache()
        self._last_config: Optional[ModelLoadConfig] = None
        self._load_signature = None
        # (voice_mode, audio component) -> speaker embedding / reference codes of an already encoded clip
        self._voice_cache: Dict[Tuple[str, str], Any] = {}

    def update_config(self, new_config: Dict[str, Any]):
        self.config = new_config.copy() if new_config else {}

    def _build_load_signature(self) -> Tuple[Any, ...]:
        return (
            self.config.get("model_variant", "ZONOS3-Mini"),
            self.config.get("device", "auto"),
            self.config.get("dtype", "auto"),
        )

    def load_model(self, model_variant: str, device: str = "auto", dtype: str = "auto"):
        from utils.models.unified_model_interface import unified_model_interface

        config = ModelLoadConfig(
            engine_name="zonos3_mini",
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

        self._voice_cache.clear()
        self.load_model(
            model_variant=self.config.get("model_variant", "ZONOS3-Mini"),
            device=self.config.get("device", "auto"),
            dtype=self.config.get("dtype", "auto"),
        )
        self._load_signature = signature

    def _get_engine(self):
        if self._last_config is None:
            self._ensure_model_loaded()
        from utils.models.unified_model_interface import unified_model_interface

        return unified_model_interface.load_model(self._last_config)

    # ------------------------------------------------------------------ voice

    @staticmethod
    def _normalize_waveform_tensor(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform.detach().float().cpu()

    def _extract_voice_reference(self, voice_ref: Optional[Dict[str, Any]]):
        """``(loader, audio component)``: ``loader()`` gives ``(waveform [channels, samples], sample rate)``;
        ``(None, None)`` without reference audio."""
        if not voice_ref or not isinstance(voice_ref, dict):
            return None, None
        ref_audio = effective_voice_audio(voice_ref)
        if ref_audio is None:
            return None, None

        if isinstance(ref_audio, str):
            return (
                lambda: AudioProcessingUtils.safe_load_audio(ref_audio),
                generate_stable_audio_component(audio_file_path=ref_audio),
            )

        if isinstance(ref_audio, dict) and "waveform" in ref_audio:
            waveform = self._normalize_waveform_tensor(ref_audio["waveform"])
            sample_rate = int(ref_audio.get("sample_rate", self.SAMPLE_RATE))
        elif torch.is_tensor(ref_audio):
            waveform = self._normalize_waveform_tensor(ref_audio)
            sample_rate = int(voice_ref.get("sample_rate", self.SAMPLE_RATE))
        else:
            raise TypeError(f"Unsupported ZONOS3-Mini reference type: {type(ref_audio)}")
        component = generate_stable_audio_component(
            reference_audio={"waveform": waveform, "sample_rate": sample_rate}
        )
        return (lambda: (waveform, sample_rate)), component

    def _remember_voice(self, key: Tuple[str, str], value):
        if len(self._voice_cache) >= self.MAX_VOICE_ENTRIES:
            self._voice_cache.pop(next(iter(self._voice_cache)))
        self._voice_cache[key] = value
        return value

    def _speaker_fields(self, engine, voice_mode: str, loader, audio_component: str, preset: str) -> Dict[str, Any]:
        key = (voice_mode, audio_component)
        value = self._voice_cache.get(key)
        if value is None:
            if loader is None:
                value = engine.preset(preset)["codes" if voice_mode == "reference" else "embedding"]
            else:
                waveform, sample_rate = loader()
                waveform = self._normalize_waveform_tensor(waveform)
                if voice_mode == "reference":
                    value = engine.encode_reference(waveform, int(sample_rate))
                else:
                    value = engine.embed_speaker(waveform, int(sample_rate))
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            self._remember_voice(key, value)

        if voice_mode == "reference":
            return {"speaker_mode": "reference", "speaker_reference_codes": np.asarray(value).tolist()}
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(value, dtype=np.float32), allow_pickle=False)
        return {
            "speaker_mode": "embedding",
            "speaker_embedding_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }

    # ------------------------------------------------------------------ request

    def _build_request(self, text: str, seed: int, voice_mode: str) -> Dict[str, Any]:
        cfg = self.config
        request: Dict[str, Any] = {
            "text": text,
            "language": to_zonos_language(cfg.get("language", DEFAULT_LANGUAGE)),
            "text_normalization": bool(cfg.get("text_normalization", True)),
            "temperature": float(cfg.get("temperature", 0.75)),
            "topk": int(cfg.get("top_k", 80)),
            "top_p": float(cfg.get("top_p", 0.69)),
            "min_p": float(cfg.get("min_p", 0.0)),
            "repetition_penalty": float(cfg.get("repetition_penalty", 2.4)),
            "repetition_window": int(cfg.get("repetition_window", 96)),
            "repetition_codebooks": int(cfg.get("repetition_codebooks", 6)),
            "seed": int(seed or 0),
            "speaker_cfg_scale": float(cfg.get("speaker_cfg_scale", 1.0)),
            "controls_cfg_scale": float(cfg.get("controls_cfg_scale", 1.0)),
            "condition_buckets": {
                "loudness": int(cfg.get("loudness_bucket", 10)),
                "fmax": int(cfg.get("fmax_bucket", 10)),
                "speaking_rate": int(cfg.get("speaking_rate_bucket", 30)),
            },
            "decoder": str(cfg.get("decoder", "keep2_plain") or "keep2_plain"),
            "laci": bool(cfg.get("laci", False)),
            "short_text": str(cfg.get("short_text", "auto") or "auto"),
            "silence_prefill_frames": int(cfg.get("silence_prefill_frames", 3)),
            "headroom_db": float(cfg.get("headroom_db", 3.0)),
            "fade_out_ms": float(cfg.get("fade_out_ms", 0.0)),
        }
        max_new_frames = int(cfg.get("max_new_frames", 0) or 0)
        if max_new_frames > 0:
            request["max_tokens"] = max_new_frames
        if voice_mode == "reference":
            request["decode_context"] = bool(cfg.get("decode_context", False))
        controls = {str(k): float(v) for k, v in (cfg.get("controls") or {}).items() if float(v) != 0.0}
        if controls:
            request["controls"] = controls
        for key in ("steer_hold", "steer_onset_seconds", "steer_ramp_seconds"):
            value = cfg.get(key)
            if value is not None and float(value) >= 0:
                request[key] = float(value)
        return request

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

        voice_mode = str(self.config.get("voice_mode", "embedding") or "embedding")
        if voice_mode not in ("embedding", "reference"):
            raise ValueError(f"ZONOS3-Mini voice_mode must be 'embedding' or 'reference', got '{voice_mode}'")
        preset = str(self.config.get("voice_preset", DEFAULT_PRESET) or DEFAULT_PRESET)
        request = self._build_request(stripped, seed, voice_mode)

        loader, audio_component = self._extract_voice_reference(voice_ref)
        if loader is None:
            audio_component = f"preset:{preset}"

        cache_key = None
        if enable_audio_cache:
            cache_key = self.audio_cache.generate_cache_key(
                "zonos3_mini",
                **request,
                voice_mode=voice_mode,
                audio_component=audio_component,
                model_variant=self.config.get("model_variant", "ZONOS3-Mini"),
                dtype=self.config.get("dtype", "auto"),
                character=character_name or "narrator",
            )
            cached_audio = self.audio_cache.get_cached_audio(cache_key)
            if cached_audio:
                print(f"💾 Using cached ZONOS3-Mini audio for '{character_name or 'narrator'}': '{stripped[:30]}...'")
                return cached_audio[0]

        self._ensure_model_loaded()
        engine = self._get_engine()
        engine._ensure_model_loaded(request["decoder"])
        request.update(self._speaker_fields(engine, voice_mode, loader, audio_component, preset))

        audio_tensor, laci = engine.generate(request)
        if laci:
            print(
                f"   LACI: {laci.get('attempts', 0)} rollback retry(ies), "
                f"{len(laci.get('failures') or [])} detected failure(s), ended on {laci.get('end')}"
            )

        if enable_audio_cache and cache_key:
            duration_seconds = self.audio_cache._calculate_duration(audio_tensor, "zonos3_mini")
            self.audio_cache.cache_audio(cache_key, audio_tensor, duration_seconds)

        return audio_tensor
