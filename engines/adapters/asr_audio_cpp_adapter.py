"""audio.cpp adapter for the Suite's unified ASR pipeline."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Iterable, Mapping, Optional

import torch

from utils.asr.types import ASRRequest, ASRResult, ASRSegment, ASRWord
from utils.audio.processing import AudioProcessingUtils


_NATIVE_CHUNK_FAMILIES = {
    "fun_asr_nano",
    "higgs_audio_stt",
    "hviske_asr",
    "qwen3_asr",
    "vibevoice_asr",
    "voxtral_realtime",
}

# audio.cpp release-0.5.1 keeps stale offline decoder state for these loaders:
# the first request transcribes normally and later requests return empty text.
# A fresh owned process is currently the only reliable reset contract.
_RESTART_BETWEEN_CHUNKS_FAMILIES = {"nemotron_asr", "voxtral_realtime"}


def _session(config: Mapping[str, Any]):
    from utils.audio_cpp.session import get_audio_cpp_session

    return get_audio_cpp_session(dict(config))


def _advanced_options(config: Mapping[str, Any]) -> Dict[str, Any]:
    value = config.get("advanced_options", config.get("request_options", {}))
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid audio.cpp advanced JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("audio.cpp advanced options must be a JSON object")
    return dict(value)


def _audio_path(audio: Mapping[str, Any]) -> str:
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not torch.is_tensor(waveform):
        raise TypeError("audio.cpp ASR input must contain a waveform tensor")
    if sample_rate is None or int(sample_rate) <= 0:
        raise ValueError("audio.cpp ASR input must contain a positive sample_rate")
    return os.path.abspath(
        AudioProcessingUtils.save_audio_to_temp_file(waveform, int(sample_rate))
    )


def _waveform_3d(audio: Mapping[str, Any]) -> tuple[torch.Tensor, int]:
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate") or 0)
    if not torch.is_tensor(waveform):
        raise TypeError("audio.cpp ASR input must contain a waveform tensor")
    if sample_rate <= 0:
        raise ValueError("audio.cpp ASR input must contain a positive sample_rate")
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 3:
        raise ValueError(
            "audio.cpp ASR waveform must have [samples], [channels, samples], or "
            "[batch, channels, samples] shape"
        )
    if waveform.shape[0] != 1:
        raise ValueError("audio.cpp ASR accepts one audio item at a time")
    if waveform.shape[-1] <= 0:
        raise ValueError("audio.cpp ASR input audio is empty")
    return waveform.detach().cpu(), sample_rate


def _chunk_ranges(
    total_samples: int,
    sample_rate: int,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        return [(0, total_samples)]
    if overlap < 0:
        raise ValueError("ASR overlap must be zero or greater")
    if overlap >= chunk_size:
        raise ValueError("ASR overlap must be smaller than chunk_size")

    chunk_samples = chunk_size * sample_rate
    if total_samples <= chunk_samples:
        return [(0, total_samples)]
    step_samples = (chunk_size - overlap) * sample_rate
    ranges = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        ranges.append((start, end))
        if end >= total_samples:
            break
        start += step_samples
    return ranges


def _normalized_token(value: str) -> str:
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE).casefold()


def _merge_transcript(parts: Iterable[str]) -> str:
    merged: list[str] = []
    for part in parts:
        incoming = str(part or "").strip().split()
        if not incoming:
            continue
        if not merged:
            merged.extend(incoming)
            continue
        limit = min(len(merged), len(incoming), 80)
        duplicate_count = 0
        for size in range(limit, 0, -1):
            left = [_normalized_token(token) for token in merged[-size:]]
            right = [_normalized_token(token) for token in incoming[:size]]
            if all(left) and left == right:
                duplicate_count = size
                break
        merged.extend(incoming[duplicate_count:])
    return " ".join(merged).strip()


def _offset_words(
    words: Iterable[ASRWord], offset: float, unique_after: Optional[float]
) -> list[ASRWord]:
    shifted = []
    for word in words:
        item = ASRWord(start=word.start + offset, end=word.end + offset, text=word.text)
        if unique_after is not None and (item.start + item.end) / 2.0 < unique_after:
            continue
        shifted.append(item)
    return shifted


def _offset_segments(
    segments: Iterable[ASRSegment], offset: float, unique_after: Optional[float]
) -> list[ASRSegment]:
    shifted = []
    for segment in segments:
        item = ASRSegment(
            start=segment.start + offset,
            end=segment.end + offset,
            text=segment.text,
            speaker=segment.speaker,
        )
        if unique_after is not None and (item.start + item.end) / 2.0 < unique_after:
            continue
        shifted.append(item)
    return shifted


def _seconds(value: Any, sample_rate: int) -> float:
    try:
        return max(0.0, float(value) / float(sample_rate))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _words(payload: Mapping[str, Any], sample_rate: int) -> list[ASRWord]:
    words = []
    for item in payload.get("words") or []:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("word", item.get("text", ""))).strip()
        if not text:
            continue
        words.append(
            ASRWord(
                start=_seconds(item.get("start_sample"), sample_rate),
                end=_seconds(item.get("end_sample"), sample_rate),
                text=text,
            )
        )
    return words


def _plain_segments(payload: Mapping[str, Any], sample_rate: int) -> list[ASRSegment]:
    segments = []
    for item in payload.get("segments") or []:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text", "")).strip()
        segments.append(
            ASRSegment(
                start=_seconds(item.get("start_sample"), sample_rate),
                end=_seconds(item.get("end_sample"), sample_rate),
                text=text,
            )
        )
    return segments


def _speaker_segments(payload: Mapping[str, Any], sample_rate: int) -> list[ASRSegment]:
    segments = []
    for item in payload.get("speaker_turns") or []:
        if not isinstance(item, Mapping):
            continue
        speaker = str(item.get("speaker_id", "")).strip()
        if speaker and not speaker.lower().startswith("speaker"):
            speaker = f"Speaker {speaker}"
        segments.append(
            ASRSegment(
                start=_seconds(item.get("start_sample"), sample_rate),
                end=_seconds(item.get("end_sample"), sample_rate),
                text=str(item.get("text", "")).strip(),
                speaker=speaker or None,
            )
        )
    return segments


def _attach_words(segments: Iterable[ASRSegment], words: Iterable[ASRWord]) -> None:
    segment_list = list(segments)
    for word in words:
        midpoint = (word.start + word.end) / 2.0
        target = next(
            (segment for segment in segment_list if segment.start <= midpoint <= segment.end),
            None,
        )
        if target is not None:
            target.words.append(word)


class AudioCppASREngineAdapter:
    """Normalize audio.cpp transcript/timing output into ``ASRResult``."""

    def __init__(self, engine_data: Dict[str, Any]):
        self.engine_data = dict(engine_data)
        self.config = dict(engine_data.get("config", engine_data))

    def _session_config(self) -> Dict[str, Any]:
        config = dict(self.config)
        if str(config.get("connection_mode", "auto")).lower() != "external_server":
            config["requested_task"] = "asr"
            config["task"] = "asr"
        return config

    def transcribe(self, req: ASRRequest) -> ASRResult:
        if req.task != "transcribe":
            raise ValueError(
                "audio.cpp release-0.5.1 ASR loaders support transcription, not the "
                "Unified ASR translate mode"
            )

        config = self._session_config()
        family = str(config.get("family", "")).strip()
        warnings: list[str] = []
        notes: list[str] = []
        options = _advanced_options(config)

        # VibeVoice-ASR owns diarization across its full recording. Independent
        # Suite requests can restart speaker numbering, so preserve its native
        # chunking only for this mode. All other ASR uses Suite-side windows.
        native_diarization = (
            family == "vibevoice_asr" and req.diarization and req.chunk_size > 0
        )
        if native_diarization:
            options.setdefault("audio_chunk_mode", "fixed")
            options.setdefault("audio_chunk_seconds", int(req.chunk_size))
            if req.overlap > 0:
                notes.append(
                    "VibeVoice-ASR diarization uses native chunking to preserve speaker "
                    "identity; the Suite overlap setting is not applied."
                )
        elif family in _NATIVE_CHUNK_FAMILIES:
            options.setdefault("audio_chunk_mode", "none")

        if req.timestamps == "word" and family == "qwen3_asr":
            session_options = config.get("session_options") or {}
            aligner = session_options.get("qwen3_asr.forced_aligner_model_path")
            if aligner:
                options["return_timestamps"] = True
            else:
                warnings.append(
                    "Qwen3-ASR word timestamps require the optional Qwen3 Forced Aligner; "
                    "transcription continued without downloading that auxiliary model."
                )

        waveform, source_rate = _waveform_3d(req.audio)
        ranges = (
            [(0, waveform.shape[-1])]
            if native_diarization
            else _chunk_ranges(
                waveform.shape[-1], source_rate, int(req.chunk_size), int(req.overlap)
            )
        )
        session = _session(config)
        if str(getattr(session, "task", "asr")) != "asr":
            raise ValueError(
                f"audio.cpp model '{session.model_id}' is configured for task "
                f"'{session.task}', not ASR"
            )
        restart_between_chunks = (
            len(ranges) > 1 and family in _RESTART_BETWEEN_CHUNKS_FAMILIES
        )
        if restart_between_chunks and not bool(getattr(session, "owned", False)):
            raise RuntimeError(
                f"audio.cpp release-0.5.1 {family} returns empty text after its first "
                "offline request. Suite-side chunking therefore requires a managed "
                "audio.cpp server so the Suite can reset it between chunks. Set "
                "connection_mode to managed, or set ASR chunk_size to 0 when using "
                "an external server."
            )
        if restart_between_chunks:
            notes.append(
                f"audio.cpp release-0.5.1 {family} requires a managed server reset "
                "between Suite chunks to avoid empty repeated-request results."
            )

        display_family = family or "external model"
        print(f"🎧 audio.cpp ASR: Transcribing with {display_family}...")
        if len(ranges) > 1:
            notes.append(
                f"Suite-side ASR chunking used {len(ranges)} windows of "
                f"{int(req.chunk_size)}s with {int(req.overlap)}s overlap."
            )
            print(
                f"🧩 audio.cpp ASR: {len(ranges)} chunks "
                f"({int(req.chunk_size)}s, {int(req.overlap)}s overlap)"
            )

        payloads: list[Mapping[str, Any]] = []
        chunk_timings: list[Mapping[str, Any]] = []
        chunk_diagnostics: list[Dict[str, Any]] = []
        started_at = time.time()
        for index, (start, end) in enumerate(ranges, start=1):
            if index > 1 and restart_between_chunks:
                print(
                    f"🔄 audio.cpp ASR: Resetting {family} session for chunk "
                    f"{index}/{len(ranges)}"
                )
                session.restart_owned_runtime()
            chunk_waveform = waveform[..., start:end]
            chunk_rms = float(torch.sqrt(torch.mean(chunk_waveform.float().square())).item())
            chunk_peak = float(chunk_waveform.float().abs().max().item())
            temp_path = _audio_path({
                "waveform": chunk_waveform,
                "sample_rate": source_rate,
            })
            try:
                request: Dict[str, Any] = {"audio": temp_path, "options": dict(options)}
                if req.language:
                    request["language"] = req.language
                result = session.run(request)
                payload = result.raw if isinstance(result.raw, Mapping) else {}
                payloads.append(payload)
                if isinstance(payload.get("timing"), Mapping):
                    chunk_timings.append(payload["timing"])
                chunk_diagnostics.append({
                    "index": index,
                    "start": round(start / source_rate, 3),
                    "end": round(end / source_rate, 3),
                    "rms": round(chunk_rms, 6),
                    "peak": round(chunk_peak, 6),
                    "text": str(payload.get("text", "")).strip(),
                    "characters": len(str(payload.get("text", "")).strip()),
                    "upstream_timing": (
                        dict(payload["timing"])
                        if isinstance(payload.get("timing"), Mapping)
                        else None
                    ),
                })
            finally:
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass
            if len(ranges) > 1:
                chunk_chars = len(str(payload.get("text", "")).strip())
                print(
                    f"   ASR chunk {index}/{len(ranges)} complete "
                    f"({chunk_chars} chars, RMS {chunk_rms:.4f}, peak {chunk_peak:.4f})"
                )

        words: list[ASRWord] = []
        speaker_segments: list[ASRSegment] = []
        plain_segments: list[ASRSegment] = []
        overlap_seconds = float(req.overlap) if len(ranges) > 1 else 0.0
        for index, ((start, _end), payload) in enumerate(zip(ranges, payloads)):
            offset = start / source_rate
            unique_after = offset + overlap_seconds if index > 0 else None
            words.extend(_offset_words(_words(payload, source_rate), offset, unique_after))
            speaker_segments.extend(
                _offset_segments(
                    _speaker_segments(payload, source_rate), offset, unique_after
                )
            )
            plain_segments.extend(
                _offset_segments(_plain_segments(payload, source_rate), offset, unique_after)
            )

        if req.diarization:
            segments = speaker_segments
            if segments:
                _attach_words(segments, words)
            else:
                warnings.append(
                    f"audio.cpp {family or 'ASR model'} returned no speaker-attributed turns."
                )
                segments = plain_segments
        elif req.timestamps == "word" and words:
            segments = [
                ASRSegment(start=word.start, end=word.end, text=word.text, words=[word])
                for word in words
            ]
        elif req.timestamps == "word":
            segments = plain_segments
        else:
            segments = []

        text = _merge_transcript(payload.get("text", "") for payload in payloads)
        if req.diarization and speaker_segments:
            text = " ".join(
                f"[{segment.speaker}] {segment.text}" if segment.speaker else segment.text
                for segment in speaker_segments
                if segment.text
            ).strip()
        if not text and speaker_segments:
            text = " ".join(segment.text for segment in speaker_segments if segment.text).strip()
        if req.timestamps == "word" and not words:
            warnings.append(f"audio.cpp {family or 'ASR model'} returned no word timestamps.")
        empty_chunks = sum(
            1 for payload in payloads if not str(payload.get("text", "")).strip()
        )
        if len(payloads) > 1 and empty_chunks:
            warnings.append(
                f"audio.cpp {family or 'ASR model'} returned no text for "
                f"{empty_chunks} of {len(payloads)} Suite chunks."
            )

        raw: Dict[str, Any] = {}
        if warnings:
            raw["warnings"] = warnings
        if notes:
            raw["notes"] = notes
        if len(payloads) == 1 and chunk_timings:
            raw["timing"] = dict(chunk_timings[0])
        elif len(payloads) > 1:
            raw["timing"] = {
                "wall_ms": round((time.time() - started_at) * 1000.0, 3),
                "suite_chunks": len(payloads),
                "suite_chunk_size_seconds": int(req.chunk_size),
                "suite_overlap_seconds": int(req.overlap),
                "upstream_wall_ms": round(
                    sum(float(item.get("wall_ms", 0.0)) for item in chunk_timings), 3
                ),
            }
            raw["chunks"] = chunk_diagnostics
        output_language = next(
            (
                str(payload.get("language", "")).strip()
                for payload in payloads
                if str(payload.get("language", "")).strip()
            ),
            str(req.language or "").strip(),
        ) or None
        print(
            f"✅ audio.cpp ASR: Complete ({len(text)} chars, "
            f"{len(segments)} timed/speaker segments)"
        )
        return ASRResult(
            text=text,
            language=output_language,
            segments=segments,
            raw=raw or None,
        )


__all__ = ["AudioCppASREngineAdapter"]
