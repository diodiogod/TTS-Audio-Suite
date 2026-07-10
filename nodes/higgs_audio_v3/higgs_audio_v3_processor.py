"""Higgs Audio v3 TTS processor."""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional

import torch
import comfy.model_management as model_management

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.adapters.higgs_audio_v3_adapter import HiggsAudioV3EngineAdapter
from utils.audio.chunk_combiner import ChunkCombiner
from utils.audio.processing import AudioProcessingUtils
from utils.text.character_parser import character_parser
from utils.text.segment_parameters import apply_segment_parameters
from utils.voice.discovery import get_available_characters, voice_discovery
from utils.voice.character_logging import (
    format_resolved_character_block,
    resolved_character_label,
)


DELIVERY_PROSODY_VALUES = {
    "speed_very_slow",
    "speed_slow",
    "speed_fast",
    "speed_very_fast",
    "pitch_low",
    "pitch_high",
    "expressive_high",
    "expressive_low",
}


def _normalize_higgs_native_tag_aliases(text: str) -> str:
    """Accept `<emotion:...>` style aliases and rewrite them to native Higgs `<|...|>` tags."""

    def _replace(match: re.Match[str]) -> str:
        category = match.group(1)
        value = match.group(2).strip()
        return f"<|{category}:{value}|>"

    return re.sub(r"<(emotion|style|prosody|sfx):([^<>]+)>", _replace, text)


def _delivery_state_from_prefix(prefix: str) -> dict[str, str]:
    state: dict[str, str] = {}
    for kind, value in re.findall(r"<\|(emotion|style|prosody):([^|]+)\|>", prefix):
        if kind == "prosody" and value not in DELIVERY_PROSODY_VALUES:
            continue
        state[kind] = value
    return state


def _delivery_state_prefix(state: dict[str, str], skip_categories: set[str] | None = None) -> str:
    skip_categories = skip_categories or set()
    parts: list[str] = []
    if "emotion" not in skip_categories and state.get("emotion"):
        parts.append(f"<|emotion:{state['emotion']}|>")
    if "style" not in skip_categories and state.get("style"):
        parts.append(f"<|style:{state['style']}|>")
    if "prosody" not in skip_categories and state.get("prosody"):
        parts.append(f"<|prosody:{state['prosody']}|>")
    return "".join(parts)


def _update_delivery_state_from_text(state: dict[str, str], text: str) -> None:
    for kind, value in re.findall(r"<\|(emotion|style|prosody):([^|]+)\|>", text):
        if kind == "prosody" and value not in DELIVERY_PROSODY_VALUES:
            continue
        state[kind] = value


def _initial_delivery_categories(text: str) -> set[str]:
    categories: set[str] = set()
    pos = 0
    pattern = re.compile(r"\s*<\|(emotion|style|prosody):([^|]+)\|>")
    for match in pattern.finditer(text):
        if match.start() != pos:
            break
        kind, value = match.group(1), match.group(2)
        if kind != "prosody" or value in DELIVERY_PROSODY_VALUES:
            categories.add(kind)
        pos = match.end()
    return categories


def _bare_tag_start(text: str) -> int | None:
    match = re.search(r"<\|(?:emotion|style|prosody|sfx):[^|]+\|>\s*$", text)
    return None if match is None else match.start()


def _is_cjk(char: str) -> bool:
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x3040 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
        or 0x0E00 <= cp <= 0x0E7F
        or 0x1000 <= cp <= 0x109F
        or 0x1780 <= cp <= 0x17FF
    )


def _tag_safe_boundary(segment: str) -> int | None:
    boundary = re.compile(
        r"(<\|prosody:(?:pause|long_pause)\|>|[.!?]+(?:\s|$)|[。？！\u0964\u0965\u061F\u104B\u0F0D]+)"
    )
    tag_ranges = [(m.start(), m.end()) for m in re.finditer(r"<\|[^|]*\|>", segment)]
    last_end = None
    for match in boundary.finditer(segment):
        end = match.end()
        if any(start < end < stop for start, stop in tag_ranges):
            continue
        last_end = end
    return last_end


def _chunk_by_characters(text: str, chars_per_chunk: int) -> list[str]:
    if len(text) <= chars_per_chunk:
        return [text]
    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        target = min(pos + chars_per_chunk, len(text))
        if target >= len(text):
            tail = text[pos:].strip()
            if tail:
                chunks.append(tail)
            break
        segment = text[pos:target]
        split = _tag_safe_boundary(segment)
        if split is None or split < max(20, chars_per_chunk // 3):
            split = target - pos
            next_tag = segment.rfind("<|", 0, split)
            next_close = segment.rfind("|>", 0, split)
            if next_tag > next_close:
                split = next_tag
            else:
                bare_tag = _bare_tag_start(segment[:split])
                if bare_tag is not None and bare_tag > 0:
                    split = bare_tag
        chunk = text[pos : pos + split].strip()
        if chunk:
            chunks.append(chunk)
        pos += max(split, 1)
    return chunks or [text]


def _smart_chunk_text(text: str, max_chars_per_chunk: int, enabled: bool) -> list[str]:
    if not enabled or max_chars_per_chunk <= 0:
        return [text.strip()]
    text = text.strip()
    if not text:
        return []

    cjk_count = sum(1 for ch in text if _is_cjk(ch))
    alpha_count = sum(1 for ch in text if ch.isalpha() or _is_cjk(ch))
    if alpha_count > 0 and cjk_count / alpha_count > 0.3:
        return _chunk_by_characters(text, max_chars_per_chunk)

    if len(text) <= max_chars_per_chunk:
        return [text]

    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        candidate = " ".join(current)
        if len(candidate) >= max_chars_per_chunk:
            split = _tag_safe_boundary(candidate)
            if split is not None and split >= max(20, len(candidate) // 3):
                final = candidate[:split].strip()
                rest = candidate[split:].strip()
                if final:
                    chunks.append(final)
                current = rest.split() if rest else []
            else:
                bare_tag = _bare_tag_start(candidate)
                if bare_tag is not None and bare_tag > 0:
                    final = candidate[:bare_tag].strip()
                    rest = candidate[bare_tag:].strip()
                    if final:
                        chunks.append(final)
                    current = rest.split() if rest else []
                else:
                    chunks.append(candidate.strip())
                    current = []
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


class HiggsAudioV3Processor:
    """Internal processor for Higgs Audio v3 generation."""

    SAMPLE_RATE = 24000

    def __init__(self, node_instance, engine_config: Dict[str, Any]):
        self.node = node_instance
        self.config = engine_config.copy()
        self.adapter = HiggsAudioV3EngineAdapter(node_instance)
        self.adapter.load_model(
            model=self.config.get("model", "higgs-audio-v3-tts-4b"),
            device=self.config.get("device", "auto"),
            dtype=self.config.get("dtype", "auto"),
            attention=self.config.get("attention", "auto"),
        )

    def update_config(self, new_config: Dict[str, Any]):
        old_load_key = (
            self.config.get("model", "higgs-audio-v3-tts-4b"),
            self.config.get("device", "auto"),
            self.config.get("dtype", "auto"),
            self.config.get("attention", "auto"),
        )
        self.config.update(new_config)
        new_load_key = (
            self.config.get("model", "higgs-audio-v3-tts-4b"),
            self.config.get("device", "auto"),
            self.config.get("dtype", "auto"),
            self.config.get("attention", "auto"),
        )
        if old_load_key != new_load_key:
            self.adapter.update_model_config(*new_load_key)

    def process_text(
        self,
        text: str,
        voice_mapping: Dict[str, Any],
        seed: int,
        enable_chunking: bool = True,
        max_chars_per_chunk: int = 400,
    ) -> List[Dict]:
        text = _normalize_higgs_native_tag_aliases(text)
        params = self.config.copy()
        params["seed"] = seed

        available_chars = get_available_characters()
        character_aliases = voice_discovery.get_character_aliases()
        all_available = set(["narrator"])
        if available_chars:
            all_available.update(available_chars)
        for alias, target in character_aliases.items():
            all_available.add(alias.lower())
            all_available.add(target.lower())
        for tag in re.findall(r"\[([^\]]+)\]", text):
            if not tag.startswith("pause:"):
                all_available.add(tag.split("|")[0].strip().lower())

        character_parser.set_available_characters(list(all_available))
        character_parser.reset_session_cache()
        segment_objects = character_parser.parse_text_segments(text, engine_type="higgs_audio_v3")
        return self._process_character_switching(
            segment_objects,
            voice_mapping,
            params,
            enable_chunking,
            max_chars_per_chunk,
        )

    def _process_character_switching(
        self,
        segment_objects,
        voice_mapping: Dict[str, Any],
        params: Dict[str, Any],
        enable_chunking: bool,
        max_chars: int,
    ) -> List[Dict]:
        audio_segments = []
        print(f"🔄 Higgs Audio v3: Processing {len(segment_objects)} segment(s)")

        for seg_idx, segment in enumerate(segment_objects):
            if model_management.interrupt_processing:
                raise InterruptedError(f"Higgs Audio v3 segment {seg_idx + 1}/{len(segment_objects)} interrupted by user")

            segment_params = params.copy()
            if segment.parameters:
                updates = apply_segment_parameters(segment_params, segment.parameters, "higgs_audio_v3")
                segment_params.update(updates)
                print(f"📊 Higgs Audio v3 segment parameters: {segment.parameters}")

            self._process_character_block(
                segment.character,
                segment.text.strip(),
                voice_mapping,
                segment_params,
                enable_chunking,
                max_chars,
                audio_segments,
            )

        return audio_segments

    def _process_character_block(
        self,
        character: str,
        text: str,
        voice_mapping: Dict[str, Any],
        params: Dict[str, Any],
        enable_chunking: bool,
        max_chars: int,
        audio_segments: List[Dict],
    ) -> None:
        if not text:
            return
        voice_ref = voice_mapping.get(character) or voice_mapping.get("narrator")
        chunks = _smart_chunk_text(text, int(max_chars), enable_chunking and len(text) > max_chars)
        delivery_state: dict[str, str] = {}

        for chunk_idx, chunk in enumerate(chunks):
            if model_management.interrupt_processing:
                raise InterruptedError(f"Higgs Audio v3 chunk {chunk_idx + 1}/{len(chunks)} interrupted by user")

            local_seed = int(params.get("seed", 0) or 0)
            if local_seed:
                local_seed += chunk_idx
            chunk_params = params.copy()
            chunk_params["seed"] = local_seed
            skip_categories = _initial_delivery_categories(chunk)
            prefix = _delivery_state_prefix(delivery_state, skip_categories)
            prompt_text = prefix + chunk
            chunk_note = f" chunk {chunk_idx + 1}/{len(chunks)}" if len(chunks) > 1 else ""
            display_name = resolved_character_label(character, voice_ref)
            print(f"🎭 Higgs Audio v3 - Generating for '{display_name}'{chunk_note}:")
            print(format_resolved_character_block(character, prompt_text, voice_ref))

            audio_tensor = self.adapter.generate_with_pause_tags(
                prompt_text,
                voice_ref,
                chunk_params,
                process_pauses=True,
                character_name=character,
            )
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(0)

            audio_segments.append(
                {
                    "waveform": audio_tensor,
                    "sample_rate": self.SAMPLE_RATE,
                    "character": character,
                    "text": chunk,
                }
            )
            _update_delivery_state_from_text(delivery_state, chunk)

    def combine_audio_segments(
        self,
        segments: List[Dict],
        method: str = "auto",
        silence_ms: int = 100,
        text_length: Optional[int] = None,
        return_info: bool = False,
    ):
        if not segments:
            empty = torch.zeros(1, 0)
            return (empty, {}) if return_info else empty

        audio_tensors = []
        text_chunks = []
        for segment in segments:
            tensor = segment["waveform"]
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 3:
                tensor = tensor.squeeze(0)
            audio_tensors.append(tensor)
            text_chunks.append(segment.get("text", ""))

        result = ChunkCombiner.combine_chunks(
            audio_tensors,
            method=method,
            silence_ms=silence_ms,
            sample_rate=self.SAMPLE_RATE,
            text_length=text_length or 0,
            original_text=" ".join(chunk for chunk in text_chunks if chunk).strip(),
            text_chunks=text_chunks,
            return_info=return_info,
        )

        if return_info:
            combined, chunk_info = result
            chunk_info["sample_rate"] = self.SAMPLE_RATE
            return combined, chunk_info
        return result

    def format_for_comfyui(self, audio_tensor: torch.Tensor) -> Dict[str, Any]:
        return AudioProcessingUtils.format_for_comfyui(audio_tensor, self.SAMPLE_RATE)
