"""Provider dispatch for unified sound-effect generation."""

from __future__ import annotations

import time
import secrets
import math
from dataclasses import dataclass
from typing import Any, Dict, Protocol

import torch

from utils.text.segment_parameters import apply_segment_parameters, parse_parameter_segments
from utils.audio.processing import AudioProcessingUtils
from utils.voice.character_logging import format_resolved_character_block
from utils.text.pause_processor import PauseTagProcessor


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


def _parse_sound_effect_timeline(description: str):
    """Combine standalone pause tags with generic parameter-only text tags."""
    timeline = []
    pause_segments, _ = PauseTagProcessor.parse_pause_tags(description)
    for segment_type, content in pause_segments:
        if segment_type == "pause":
            timeline.append(("pause", float(content), {}))
        else:
            for segment_text, parameters in parse_parameter_segments(str(content)):
                timeline.append(("audio", segment_text, parameters))
    return timeline


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
        duration_tokens = int(config.get("duration_tokens") or 0)
        if duration_tokens <= 0:
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
    crossfade_seconds: float = 1.0,
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
    crossfade_seconds = max(0.0, float(crossfade_seconds))

    engine_type = str(engine_data.get("engine_type") or engine_data.get("config", {}).get("engine_type") or "")
    provider = _PROVIDERS.get(engine_type)
    if provider is None:
        supported = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Engine '{engine_type or 'unknown'}' is not wired for Sound Effects. Supported: {supported}")

    requested_seed = int(seed or 0)
    resolved_seed = requested_seed if requested_seed > 0 else secrets.randbelow(2**63 - 1) + 1

    print("🌩️ Sound Effects: resolving timeline and inline parameters")

    base_config = _engine_config(engine_data)
    base_config.update({"duration_seconds": requested_seconds, "seed": resolved_seed})
    timeline = _parse_sound_effect_timeline(clean_description)

    if not timeline:
        raise ValueError("Sound Effects description contains no text to generate")

    generated_infos = []
    sample_rate = None
    assembled = None
    pending_leading_pause = 0.0
    previous_was_audio = False
    audio_index = 0
    audio_count = sum(item_type == "audio" for item_type, _, _ in timeline)
    if audio_count == 0:
        raise ValueError("Sound Effects description contains pauses but no sound description to generate")
    print(
        f"🔄 Sound Effects: Processing {len(timeline)} timeline segment(s) "
        f"({audio_count} generation{'s' if audio_count != 1 else ''})"
    )

    def append_audio(current, next_audio, allow_crossfade):
        if current is None:
            return next_audio
        if allow_crossfade and crossfade_seconds > 0:
            return AudioProcessingUtils.crossfade_audio(current, next_audio, crossfade_seconds, sample_rate)
        return torch.cat([current, next_audio], dim=-1)

    def generate_one(segment_text, segment_config, segment_seconds, segment_seed):
        nonlocal sample_rate
        segment_engine_data = dict(engine_data)
        segment_engine_data["config"] = segment_config

        # v2 has a hard 30-second generation window. Longer requests are split
        # into equal pieces whose overlaps still assemble to the exact target time.
        chunk_count = 1
        if engine_type == "moss_soundeffect_v2" and segment_seconds > 30.0:
            overlap = min(crossfade_seconds, 29.0)
            chunk_count = max(2, math.ceil((segment_seconds - overlap) / (30.0 - overlap)))
        generated_total = segment_seconds + min(crossfade_seconds, 29.0) * (chunk_count - 1)
        chunk_seconds = generated_total / chunk_count
        chunk_waveform = None
        chunk_infos = []
        if chunk_count > 1:
            print(
                f"  📝 Splitting {segment_seconds:.2f}s into {chunk_count} overlapping chunk(s) "
                f"(crossfade={crossfade_seconds:.2f}s)"
            )
        for chunk_index in range(chunk_count):
            chunk_seed = (segment_seed + chunk_index) % (2**63 - 1) or 1
            if chunk_count > 1:
                print(
                    f"🧩 Sound Effects chunk {chunk_index + 1}/{chunk_count}: "
                    f"{chunk_seconds:.2f}s (seed {chunk_seed})"
                )
            result = provider.generate(
                engine_data=segment_engine_data,
                description=segment_text,
                duration_seconds=chunk_seconds,
                seed=chunk_seed,
                enable_audio_cache=bool(enable_audio_cache),
            )
            result_rate = int(result.audio["sample_rate"])
            if sample_rate is None:
                sample_rate = result_rate
            elif sample_rate != result_rate:
                raise ValueError("Sound Effects segments returned inconsistent sample rates")
            chunk_waveform = append_audio(chunk_waveform, result.audio["waveform"], chunk_waveform is not None)
            chunk_infos.append(result.generation_info)

        if chunk_count > 1:
            target_samples = round(segment_seconds * sample_rate)
            current_samples = chunk_waveform.shape[-1]
            if current_samples > target_samples:
                chunk_waveform = chunk_waveform[..., :target_samples]
            elif current_samples < target_samples:
                chunk_waveform = torch.nn.functional.pad(chunk_waveform, (0, target_samples - current_samples))
        return chunk_waveform, chunk_infos, chunk_count

    for timeline_index, (item_type, content, parameters) in enumerate(timeline, start=1):
        if item_type == "pause":
            if sample_rate is None:
                # A leading pause is materialized after the first generated segment reveals the rate.
                pending_leading_pause += float(content)
            else:
                silence = torch.zeros(
                    (*assembled.shape[:-1], round(float(content) * sample_rate)),
                    dtype=assembled.dtype,
                    device=assembled.device,
                )
                assembled = torch.cat([assembled, silence], dim=-1)
            print(
                f"   ⏸️ Segment {timeline_index}/{len(timeline)}: "
                f"inserting {float(content):.2f}s silence"
            )
            previous_was_audio = False
            continue

        audio_index += 1
        segment_config = apply_segment_parameters(base_config, parameters, engine_type)
        segment_seconds = float(segment_config.get("duration_seconds", requested_seconds))
        segment_seed = int(segment_config.get("seed", resolved_seed))
        print(f"\n🎤 Segment {timeline_index}/{len(timeline)}: Sound effect ({segment_seconds:.2f}s)")
        if parameters:
            print(f"  📊 Applying parameters: {parameters}")
        print(format_resolved_character_block("Sound Effect", content, None))
        waveform, infos, chunk_count = generate_one(content, segment_config, segment_seconds, segment_seed)
        if pending_leading_pause > 0:
            silence = torch.zeros(
                (*waveform.shape[:-1], round(pending_leading_pause * sample_rate)),
                dtype=waveform.dtype,
                device=waveform.device,
            )
            assembled = silence
            pending_leading_pause = 0.0
        assembled = append_audio(assembled, waveform, previous_was_audio)
        previous_was_audio = True
        generated_infos.append(
            f"Segment {audio_index}" + (f" ({chunk_count} chunks)" if chunk_count > 1 else "") + ":\n" + "\n".join(infos)
        )

    info = (
        f"Sound Effects timeline: {audio_count} generated segment(s), "
        f"crossfade {crossfade_seconds:.2f}s\n\n" + "\n\n".join(generated_infos)
    )
    actual_seconds = assembled.shape[-1] / float(sample_rate)
    print(
        f"✅ Sound Effects generation complete: {audio_count} generated segment(s), "
        f"{actual_seconds:.2f}s total"
    )
    return SoundEffectResult(audio={"waveform": assembled, "sample_rate": sample_rate}, generation_info=info)
