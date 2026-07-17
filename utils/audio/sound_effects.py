"""Provider dispatch for unified sound-effect generation."""

from __future__ import annotations

import time
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Protocol

import torch


@dataclass
class SoundEffectResult:
    audio: Dict[str, Any]
    generation_info: str


class SoundEffectProvider(Protocol):
    def generate(
        self,
        engine_data: Dict[str, Any],
        description: str,
        duration_seconds: float,
        seed: int,
        enable_audio_cache: bool,
    ) -> SoundEffectResult: ...


def _engine_config(engine_data: Dict[str, Any]) -> Dict[str, Any]:
    config = engine_data.get("config", engine_data)
    if not isinstance(config, dict):
        raise TypeError("TTS_ENGINE config must be a dictionary")
    return dict(config)


def _audio_output(waveform: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().float().cpu()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() != 3:
        raise ValueError(f"Sound-effect engine returned unsupported waveform shape: {tuple(waveform.shape)}")
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _check_interrupted() -> None:
    try:
        import comfy.model_management as model_management

        checker = getattr(model_management, "throw_exception_if_processing_interrupted", None)
        if callable(checker):
            checker()
        elif getattr(model_management, "interrupt_processing", False):
            raise InterruptedError("Sound-effect generation interrupted by user")
    except ImportError:
        pass


class MossSoundEffectV1Provider:
    MODEL_NAME = "MOSS-SoundEffect"

    def generate(self, engine_data, description, duration_seconds, seed, enable_audio_cache):
        from engines.adapters.moss_tts_adapter import MossTTSEngineAdapter
        from engines.moss_tts.model_specs import MOSS_MODEL_SPECS

        config = _engine_config(engine_data)
        selected_model = str(config.get("model_variant") or "")
        if config.get("model_role") != "sound_effects" or "MOSS-SoundEffect" not in selected_model:
            raise ValueError(
                f"MOSS model '{selected_model or 'unknown'}' cannot generate sound effects. "
                "In the MOSS-TTS Engine, select 'Sound Effects 8B v1 (MOSS-SoundEffect)'."
            )

        spec = MOSS_MODEL_SPECS[self.MODEL_NAME]
        duration_tokens = max(1, int(round(float(duration_seconds) * 12.5)))
        adapter = MossTTSEngineAdapter()
        adapter.load_model(
            model_variant=selected_model,
            device=config.get("device", "auto"),
            dtype=config.get("dtype", "auto"),
            attn_implementation=config.get("attn_implementation", "auto"),
            codec_model=spec["codec_model"],
            lora_adapter=config.get("lora_adapter"),
            defer_load=True,
        )
        params = {
            "model_variant": selected_model,
            "ambient_sound": description,
            "duration_tokens": duration_tokens,
            "seed": int(seed or 0),
            "device": config.get("device", "auto"),
            "dtype": config.get("dtype", "auto"),
            "attn_implementation": config.get("attn_implementation", "auto"),
            "audio_temperature": config.get("temperature", spec["audio_temperature"]),
            "audio_top_p": config.get("top_p", spec["audio_top_p"]),
            "audio_top_k": config.get("top_k", spec["audio_top_k"]),
            "audio_repetition_penalty": config.get("repetition_penalty", spec["audio_repetition_penalty"]),
            "max_new_tokens": config.get("max_new_tokens", spec["max_new_tokens"]),
            "lora_adapter": config.get("lora_adapter"),
            "enable_audio_cache": bool(enable_audio_cache),
        }
        started = time.perf_counter()
        _check_interrupted()
        waveform = adapter.generate_sound_effect(description, params)
        _check_interrupted()
        elapsed = time.perf_counter() - started
        audio = _audio_output(waveform, int(spec["sample_rate"]))
        actual_seconds = audio["waveform"].shape[-1] / float(audio["sample_rate"])
        info = (
            "Sound effect generated with MOSS-SoundEffect v1\n"
            f"Requested duration: {float(duration_seconds):.1f}s ({duration_tokens} audio tokens)\n"
            f"Actual duration: {actual_seconds:.2f}s\n"
            f"Seed: {int(seed or 0)}\n"
            f"LoRA: {config.get('lora_adapter') or 'None'}\n"
            f"Elapsed: {elapsed:.2f}s"
        )
        return SoundEffectResult(audio=audio, generation_info=info)


class MossSoundEffectV2Provider:
    def generate(self, engine_data, description, duration_seconds, seed, enable_audio_cache):
        from engines.adapters.moss_soundeffect_v2_adapter import MossSoundEffectV2Adapter

        config = _engine_config(engine_data)
        adapter = MossSoundEffectV2Adapter(config)
        started = time.perf_counter()
        _check_interrupted()
        waveform, sample_rate, cache_hit = adapter.generate(
            description=description,
            duration_seconds=duration_seconds,
            seed=seed,
            enable_audio_cache=enable_audio_cache,
        )
        _check_interrupted()
        elapsed = time.perf_counter() - started
        audio = _audio_output(waveform, sample_rate)
        actual_seconds = audio["waveform"].shape[-1] / float(sample_rate)
        info = (
            "Sound effect generated with MOSS-SoundEffect v2\n"
            f"Requested duration: {float(duration_seconds):.1f}s\n"
            f"Actual duration: {actual_seconds:.2f}s\n"
            f"Seed: {int(seed or 0)}\n"
            f"Cache: {'hit' if cache_hit else 'generated'}\nElapsed: {elapsed:.2f}s"
        )
        return SoundEffectResult(audio=audio, generation_info=info)


_PROVIDERS: Dict[str, SoundEffectProvider] = {
    "moss_tts": MossSoundEffectV1Provider(),
    "moss_soundeffect_v2": MossSoundEffectV2Provider(),
}


def generate_sound_effect(
    engine_data: Dict[str, Any],
    description: str,
    duration_seconds: float,
    seed: int = 0,
    enable_audio_cache: bool = True,
) -> SoundEffectResult:
    if not isinstance(engine_data, dict):
        raise TypeError("🌩️ Sound Effects requires a TTS_ENGINE connection")
    capabilities = engine_data.get("capabilities") or []
    if "sound_effects" not in capabilities:
        raise ValueError(
            "The connected engine is not configured for sound effects. Select a SoundEffect model "
            "in a compatible engine node, then run the workflow again."
        )
    clean_description = str(description or "").strip()
    if not clean_description:
        raise ValueError("Sound Effects description cannot be empty")
    requested_seconds = float(duration_seconds)
    if requested_seconds <= 0:
        raise ValueError("Sound Effects duration must be greater than zero")

    engine_type = str(engine_data.get("engine_type") or engine_data.get("config", {}).get("engine_type") or "")
    provider = _PROVIDERS.get(engine_type)
    if provider is None:
        supported = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Engine '{engine_type or 'unknown'}' is not wired for Sound Effects. Supported: {supported}")

    requested_seed = int(seed or 0)
    resolved_seed = requested_seed if requested_seed > 0 else secrets.randbelow(2**63 - 1) + 1

    print("🌩️ Sound Effects")
    print("=" * 60)
    print(clean_description)
    print("=" * 60)
    return provider.generate(
        engine_data=engine_data,
        description=clean_description,
        duration_seconds=requested_seconds,
        seed=resolved_seed,
        enable_audio_cache=bool(enable_audio_cache),
    )
