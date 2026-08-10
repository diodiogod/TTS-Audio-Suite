"""Adapter between unified TTS processing and official DramaBox inference."""

import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torchaudio

from utils.audio.audio_hash import generate_stable_audio_component
from utils.audio.cache import get_audio_cache
from utils.audio.processing import AudioProcessingUtils
from utils.models.factory_config import ModelLoadConfig
from utils.voice.reference import effective_voice_audio


class DramaBoxEngineAdapter:
    """Translate suite voice/config/cache data into DramaBox calls."""

    SAMPLE_RATE = 48000
    SILENCE_RMS_THRESHOLD = 1e-3
    SILENCE_PEAK_THRESHOLD = 2e-2

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config.copy() if config else {}
        self.audio_cache = get_audio_cache()
        self._last_config: Optional[ModelLoadConfig] = None
        self._load_signature = None
        self._lora_signature = None
        self.last_generation_status: Dict[str, Any] = {"near_silent": False}

    def update_config(self, new_config: Dict[str, Any]):
        self.config = new_config.copy() if new_config else {}

    @staticmethod
    def _lora_revision(path: Any) -> str:
        """Return a cheap cache token that changes when a managed adapter is replaced."""
        value = str(path or "").strip()
        if not value:
            return ""
        try:
            candidate = os.path.abspath(os.path.expanduser(value))
            if os.path.isfile(candidate):
                stat = os.stat(candidate)
                return f"{candidate}:{stat.st_size}:{stat.st_mtime_ns}"
            if os.path.isdir(candidate):
                entries = []
                for item in os.listdir(candidate):
                    if not item.endswith(".safetensors"):
                        continue
                    item_path = os.path.join(candidate, item)
                    stat = os.stat(item_path)
                    entries.append(f"{item}:{stat.st_size}:{stat.st_mtime_ns}")
                return f"{candidate}|{'|'.join(sorted(entries))}"
        except OSError:
            pass
        return value

    @classmethod
    def _warn_if_near_silent(
        cls,
        audio: torch.Tensor,
        *,
        character_name: Optional[str],
        seed: int,
        cached: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Warn about clearly near-silent model output without altering it."""
        if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
            return None

        samples = torch.nan_to_num(audio.detach().float().cpu())
        rms = float(samples.square().mean().sqrt())
        peak = float(samples.abs().max())
        if rms >= cls.SILENCE_RMS_THRESHOLD or peak >= cls.SILENCE_PEAK_THRESHOLD:
            return None

        rms_db = 20.0 * math.log10(max(rms, 1e-12))
        peak_db = 20.0 * math.log10(max(peak, 1e-12))
        source = "cached " if cached else ""
        print(
            f"\n⚠️ DramaBox generated a near-silent {source}segment for "
            f"'{character_name or 'narrator'}' "
            f"(RMS {rms_db:.1f} dBFS, peak {peak_db:.1f} dBFS)."
        )
        print(
            "⚠️ This can depend on generation duration, reference duration, "
            "reference audio, guidance settings, and seed."
        )
        print(
            "⚠️ Try changing those parameters for this segment; another seed "
            "may help, but is not guaranteed to fix it.\n"
        )
        return {
            "near_silent": True,
            "character": character_name or "narrator",
            "seed": int(seed),
            "rms_dbfs": rms_db,
            "peak_dbfs": peak_db,
            "cached": bool(cached),
        }

    def _build_load_signature(self) -> Tuple[Any, ...]:
        """Identity of the expensive base runtime, excluding live LoRA state."""
        return (
            self.config.get("model_name", "DramaBox"),
            self.config.get("device", "auto"),
            self.config.get("precision", "auto"),
            self.config.get("memory_mode", "fast"),
            self.config.get("transformer_quantization", "none"),
            bool(self.config.get("compile_model", False)),
        )

    def _build_lora_signature(self) -> Tuple[Any, ...]:
        path = self.config.get("lora_path", "")
        return (
            str(path or "").strip(),
            self._lora_revision(path),
            float(self.config.get("lora_strength", 1.0)),
        )

    def _ensure_model_loaded(self):
        signature = self._build_load_signature()
        from utils.models.unified_model_interface import unified_model_interface

        if signature != self._load_signature or self._last_config is None:
            self._last_config = ModelLoadConfig(
                engine_name="dramabox",
                model_type="tts",
                model_name=self.config.get("model_name", "DramaBox"),
                device=self.config.get("device", "auto"),
                additional_params={
                    "precision": self.config.get("precision", "auto"),
                    "memory_mode": self.config.get("memory_mode", "fast"),
                    "transformer_quantization": self.config.get(
                        "transformer_quantization", "none"
                    ),
                    "compile_model": bool(self.config.get("compile_model", False)),
                },
            )
            self._load_signature = signature
            self._lora_signature = None

        engine = unified_model_interface.load_model(self._last_config)
        lora_signature = self._build_lora_signature()
        if lora_signature != self._lora_signature:
            lora_path, lora_revision, lora_strength = lora_signature
            engine.set_lora(
                lora_path=lora_path,
                strength=lora_strength,
                revision=lora_revision,
            )
            self._lora_signature = lora_signature
        return engine

    def _get_engine(self):
        return self._ensure_model_loaded()

    def _extract_voice_reference(
        self, voice_ref: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[str], str, bool]:
        if not isinstance(voice_ref, dict):
            return None, "default_voice", False

        audio = effective_voice_audio(voice_ref)
        if audio is None:
            return None, "default_voice", False
        if isinstance(audio, str):
            return (
                audio,
                generate_stable_audio_component(audio_file_path=audio),
                False,
            )
        if isinstance(audio, dict) and "waveform" in audio:
            path = AudioProcessingUtils.save_audio_to_temp_file(
                audio["waveform"], audio.get("sample_rate", self.SAMPLE_RATE)
            )
            return path, generate_stable_audio_component(reference_audio=audio), True
        if torch.is_tensor(audio):
            sample_rate = int(voice_ref.get("sample_rate", self.SAMPLE_RATE))
            audio_dict = {"waveform": audio, "sample_rate": sample_rate}
            path = AudioProcessingUtils.save_audio_to_temp_file(audio, sample_rate)
            return (
                path,
                generate_stable_audio_component(reference_audio=audio_dict),
                True,
            )
        raise TypeError(f"Unsupported DramaBox voice reference: {type(audio)}")

    def generate_single(
        self,
        text: str,
        voice_ref: Optional[Dict[str, Any]],
        seed: int = 42,
        enable_audio_cache: bool = True,
        character_name: Optional[str] = None,
    ) -> torch.Tensor:
        prompt = (text or "").strip()
        if not prompt:
            return torch.zeros(1, 0, dtype=torch.float32)

        voice_path, audio_component, remove_voice_path = self._extract_voice_reference(
            voice_ref
        )
        cfg_scale = float(self.config.get("cfg_scale", 2.5))
        stg_scale = float(self.config.get("stg_scale", 1.5))
        duration_multiplier = float(self.config.get("duration_multiplier", 1.1))
        gen_duration = float(self.config.get("gen_duration", 0.0))
        ref_duration = float(self.config.get("ref_duration", 10.0))
        rescale_scale = self.config.get("rescale_scale", "auto")
        watermark = bool(self.config.get("watermark", False))
        negative_prompt = str(self.config.get("negative_prompt", ""))
        model_name = self.config.get("model_name", "DramaBox")

        cache_key = None
        if enable_audio_cache:
            cache_key = self.audio_cache.generate_cache_key(
                "dramabox",
                text=prompt,
                audio_component=audio_component,
                model_name=model_name,
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                duration_multiplier=duration_multiplier,
                gen_duration=gen_duration,
                ref_duration=ref_duration,
                rescale_scale=rescale_scale,
                watermark=watermark,
                prompt_template=str(
                    self.config.get("prompt_template", '"{seg}"')
                ),
                negative_prompt=negative_prompt,
                precision=self.config.get("precision", "auto"),
                transformer_quantization=self.config.get(
                    "transformer_quantization", "none"
                ),
                memory_mode=self.config.get("memory_mode", "fast"),
                compile_model=bool(self.config.get("compile_model", False)),
                lora_path=self.config.get("lora_path", ""),
                lora_strength=float(self.config.get("lora_strength", 1.0)),
                lora_revision=self._lora_revision(self.config.get("lora_path", "")),
                seed=int(seed),
                character=character_name or "narrator",
            )
            cached = self.audio_cache.get_cached_audio(cache_key)
            if cached:
                print(
                    f"💾 Using cached DramaBox audio for "
                    f"'{character_name or 'narrator'}': '{prompt[:30]}...'"
                )
                self.last_generation_status = self._warn_if_near_silent(
                    cached[0],
                    character_name=character_name,
                    seed=int(seed),
                    cached=True,
                ) or {"near_silent": False}
                return cached[0]

        try:
            result = self._get_engine().generate(
                prompt=prompt,
                voice_ref_path=voice_path,
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                duration_multiplier=duration_multiplier,
                gen_duration=gen_duration,
                ref_duration=ref_duration,
                rescale_scale=rescale_scale,
                watermark=watermark,
                negative_prompt=negative_prompt,
                seed=int(seed),
            )
        finally:
            if remove_voice_path and voice_path:
                try:
                    os.unlink(voice_path)
                except OSError:
                    pass
        audio = result["audio"]
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
        audio = audio.detach().float().cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        if int(result.get("sample_rate", self.SAMPLE_RATE)) != self.SAMPLE_RATE:
            audio = torchaudio.functional.resample(
                audio, int(result["sample_rate"]), self.SAMPLE_RATE
            )

        self.last_generation_status = self._warn_if_near_silent(
            audio,
            character_name=character_name,
            seed=int(seed),
        ) or {"near_silent": False}

        if enable_audio_cache and cache_key:
            duration = audio.shape[-1] / self.SAMPLE_RATE
            self.audio_cache.cache_audio(cache_key, audio, duration)
        return audio
