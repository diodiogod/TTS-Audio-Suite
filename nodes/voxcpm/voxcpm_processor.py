"""VoxCPM text orchestration for TTS Audio Suite."""

import os
import re
import sys
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple, Union

import comfy.model_management as model_management
import torch

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio.chunk_timing import ChunkTimingHelper
from utils.audio.edit_post_processor import process_segments as apply_edit_post_processing
from utils.models.language_mapper import resolve_language_alias
from utils.text.character_parser import character_parser
from utils.text.pause_processor import PauseTagProcessor
from utils.text.segment_parameters import ParameterValidator, apply_segment_parameters
from utils.text.step_audio_editx_special_tags import get_edit_tags_for_segment
from utils.voice.character_logging import (
    format_resolved_character_block,
    resolved_character_label,
)
from utils.voice.discovery import (
    get_available_characters,
    get_character_mapping,
    voice_discovery,
)
from utils.voice.reference import effective_voice_audio


class VoxCPMProcessor:
    """Handle suite-level orchestration around one-call VoxCPM inference."""

    ENGINE_TYPE = "voxcpm"

    def __init__(self, adapter, engine_config: Dict[str, Any]):
        self.adapter = adapter
        self.config = dict(engine_config or {})

    def update_config(self, new_config: Dict[str, Any]):
        self.config = dict(new_config or {})
        self.adapter.update_config(self.config)

    @property
    def sample_rate(self) -> int:
        """Return the selected model's native sample rate without hardcoding it."""
        for attribute_name in ("get_sample_rate", "sample_rate", "SAMPLE_RATE"):
            value = getattr(self.adapter, attribute_name, None)
            if callable(value):
                value = value()
            if value is not None:
                try:
                    sample_rate = int(value)
                except (TypeError, ValueError):
                    continue
                if sample_rate > 0:
                    return sample_rate

        configured_rate = self.config.get("sample_rate")
        if configured_rate is not None:
            try:
                sample_rate = int(configured_rate)
            except (TypeError, ValueError):
                sample_rate = 0
            if sample_rate > 0:
                return sample_rate

        raise RuntimeError(
            "VoxCPM adapter did not expose a valid model sample rate. "
            "Expected get_sample_rate(), sample_rate, or SAMPLE_RATE."
        )

    @property
    def SAMPLE_RATE(self) -> int:
        """Compatibility property for callers that use the suite's legacy name."""
        return self.sample_rate

    def get_sample_rate(self) -> int:
        return self.sample_rate

    @staticmethod
    def _check_interrupt(context: str):
        if model_management.interrupt_processing:
            raise InterruptedError(f"VoxCPM generation interrupted {context}")

    def _setup_character_parser(self, text: str, voice_mapping: Dict[str, Any]):
        configured_language = str(self.config.get("language", "auto") or "auto").strip()
        parser_language = resolve_language_alias(configured_language)
        if parser_language in ("", "auto", "automatic", "none"):
            parser_language = "en"

        character_parser.language_resolver.default_language = parser_language
        character_parser.default_language = parser_language

        all_available = set(get_available_characters() or ())
        for alias, target in voice_discovery.get_character_aliases().items():
            all_available.add(alias.lower())
            all_available.add(target.lower())

        for character_name in (voice_mapping or {}):
            if character_name:
                all_available.add(str(character_name).lower())

        for tag in re.findall(r"\[([^\]]+)\]", text or ""):
            if tag.lower().startswith(("pause:", "wait:", "stop:")):
                continue
            all_available.add(tag.split("|", 1)[0].split(":")[-1].strip().lower())

        all_available.add("narrator")
        character_parser.set_available_characters(list(all_available))

        for character_name, language in voice_discovery.get_character_language_defaults().items():
            character_parser.set_character_language_default(character_name, language)

        character_parser.reset_session_cache()

    @staticmethod
    def _coerce_voice_reference(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            copied = dict(value)
            if "waveform" in copied and "sample_rate" in copied:
                return {"audio": copied}
            return copied
        if isinstance(value, (str, os.PathLike)):
            return {"audio_path": os.fspath(value)}
        return {"audio": value}

    def _resolve_voice_reference(
        self,
        character_name: str,
        narrator_voice: Dict[str, Any],
        voice_mapping: Dict[str, Any],
        character_mapping: Dict[str, Tuple[Any, Any]],
    ) -> Dict[str, Any]:
        if character_name != "narrator" and character_name in voice_mapping:
            return self._coerce_voice_reference(voice_mapping[character_name])

        if character_name != "narrator":
            audio_path, reference_text = character_mapping.get(character_name, (None, None))
            if audio_path:
                return {
                    "audio_path": audio_path,
                    "reference_text": reference_text or "",
                }

        return dict(narrator_voice)

    @staticmethod
    def _voice_log_note(voice_ref: Dict[str, Any]) -> str:
        if not voice_ref or effective_voice_audio(voice_ref) is None:
            return " [no reference audio]"
        reference_text = (
            voice_ref.get("reference_text")
            or voice_ref.get("prompt_text")
            or voice_ref.get("text")
            or ""
        )
        if reference_text:
            return f" [reference transcript: {len(reference_text)} chars]"
        return ""

    def _log_generation(
        self,
        character_name: str,
        text: str,
        voice_ref: Dict[str, Any],
        chunk_count: int,
        filtered_params: Dict[str, Any],
        show_text_logging: bool,
    ):
        display_name = resolved_character_label(character_name, voice_ref)
        print(
            f"🎭 VoxCPM - Generating for '{display_name}'"
            f"{self._voice_log_note(voice_ref)}"
        )
        if filtered_params:
            parameter_text = ", ".join(
                f"{name}={value}" for name, value in sorted(filtered_params.items())
            )
            print(f"🎛️ VoxCPM segment parameters: {parameter_text}")
        if show_text_logging:
            print(format_resolved_character_block(character_name, text, voice_ref))
        if chunk_count > 1:
            print(f"📝 VoxCPM: Split '{display_name}' into {chunk_count} chunks")

    @staticmethod
    def _normalize_generated_audio(audio: Any) -> torch.Tensor:
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio, dtype=torch.float32)
        audio = audio.detach().float().cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        if audio.dim() != 2:
            raise ValueError(
                f"VoxCPM adapter returned invalid audio shape {tuple(audio.shape)}; "
                "expected [samples] or [channels, samples]"
            )
        return audio

    def process_text(
        self,
        text: str,
        voice_mapping: Dict[str, Any],
        seed: int,
        enable_chunking: bool = True,
        max_chars_per_chunk: int = 400,
        chunk_combination_method: str = "auto",
        silence_between_chunks_ms: int = 100,
        enable_audio_cache: bool = True,
        apply_edit_postprocessing: bool = True,
        show_text_logging: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate record-form audio while preserving suite tag behavior."""
        del chunk_combination_method, silence_between_chunks_ms

        voice_mapping = dict(voice_mapping or {})
        self._setup_character_parser(text, voice_mapping)

        narrator_voice = self._coerce_voice_reference(voice_mapping.get("narrator"))
        base_config = dict(self.config)
        segment_objects = character_parser.parse_text_segments(
            text or "", engine_type=self.ENGINE_TYPE
        )

        characters = list(
            {segment.character for segment in segment_objects if segment.character}
        )
        character_mapping = get_character_mapping(characters, engine_type="audio_only")
        segment_records: List[Dict[str, Any]] = []

        try:
            for segment_index, segment in enumerate(segment_objects):
                character_name = segment.character or "narrator"
                self._check_interrupt(
                    f"before segment {segment_index + 1}/{len(segment_objects)} "
                    f"('{character_name}')"
                )

                segment_text = (segment.text or "").strip()
                if not segment_text:
                    continue

                raw_segment_params = dict(segment.parameters or {})
                filtered_params = ParameterValidator.filter_parameters_for_engine(
                    raw_segment_params, self.ENGINE_TYPE
                )
                current_config = apply_segment_parameters(
                    base_config, filtered_params, self.ENGINE_TYPE
                )
                current_seed = int(current_config.get("seed", seed))

                if (
                    getattr(segment, "explicit_language", False)
                    and "language" in base_config
                    and getattr(segment, "language", None)
                ):
                    current_config["language"] = segment.language

                self.adapter.update_config(current_config)
                voice_ref = self._resolve_voice_reference(
                    character_name,
                    narrator_voice,
                    voice_mapping,
                    character_mapping,
                )

                def generate_chunks(text_content: str, edit_tags: list):
                    cleaned = (text_content or "").strip()
                    if not cleaned:
                        return

                    if enable_chunking:
                        from utils.text.chunking import ImprovedChatterBoxChunker

                        maximum = ImprovedChatterBoxChunker.validate_chunking_params(
                            max_chars_per_chunk
                        )
                        chunks = ImprovedChatterBoxChunker.split_into_chunks(
                            cleaned, max_chars=maximum
                        )
                    else:
                        chunks = [cleaned]

                    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
                    self._log_generation(
                        character_name,
                        cleaned,
                        voice_ref,
                        len(chunks),
                        filtered_params,
                        show_text_logging,
                    )

                    for chunk_index, chunk in enumerate(chunks):
                        self._check_interrupt(
                            f"before chunk {chunk_index + 1}/{len(chunks)} "
                            f"for '{character_name}'"
                        )
                        audio = self.adapter.generate_single(
                            text=chunk,
                            voice_ref=voice_ref,
                            seed=(
                                0
                                if current_seed == 0
                                else current_seed + chunk_index
                            ),
                            enable_audio_cache=enable_audio_cache,
                            character_name=character_name,
                        )
                        audio = self._normalize_generated_audio(audio)
                        segment_records.append(
                            {
                                "waveform": audio,
                                "sample_rate": self.sample_rate,
                                "text": chunk,
                                "edit_tags": edit_tags if chunk_index == 0 else [],
                            }
                        )

                if PauseTagProcessor.has_pause_tags(segment_text):
                    pause_parts, _ = PauseTagProcessor.parse_pause_tags(segment_text)
                    for part_index, (part_type, content) in enumerate(pause_parts):
                        self._check_interrupt(
                            f"before pause fragment {part_index + 1}/{len(pause_parts)} "
                            f"for '{character_name}'"
                        )
                        if part_type == "text":
                            clean_text, edit_tags = get_edit_tags_for_segment(content)
                            generate_chunks(clean_text, edit_tags)
                        elif part_type == "pause":
                            sample_rate = self.sample_rate
                            silence = PauseTagProcessor.create_silence_segment(
                                content,
                                sample_rate,
                                torch.device("cpu"),
                                torch.float32,
                            )
                            silence = self._normalize_generated_audio(silence)
                            segment_records.append(
                                {
                                    "waveform": silence,
                                    "sample_rate": sample_rate,
                                    "text": f"[pause:{content}s]",
                                    "edit_tags": [],
                                }
                            )
                else:
                    clean_text, edit_tags = get_edit_tags_for_segment(segment_text)
                    generate_chunks(clean_text, edit_tags)
        finally:
            # Segment-local overrides must never leak into the next invocation.
            self.adapter.update_config(base_config)

        if (
            apply_edit_postprocessing
            and segment_records
            and any(record.get("edit_tags") for record in segment_records)
        ):
            self._check_interrupt("before inline edit post-processing")
            segment_records = apply_edit_post_processing(
                segment_records, engine_config=base_config
            )
            self._check_interrupt("after inline edit post-processing")

        return segment_records

    def combine_audio_segments(
        self,
        segments: List[Dict[str, Any]],
        method: str = "auto",
        silence_ms: int = 100,
        original_text: str = "",
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        if not segments:
            empty = torch.zeros(1, 0, dtype=torch.float32)
            return (empty, {}) if return_info else empty

        sample_rates = {
            int(segment.get("sample_rate", 0))
            for segment in segments
            if segment.get("sample_rate") is not None
        }
        if len(sample_rates) != 1 or next(iter(sample_rates), 0) <= 0:
            raise ValueError(
                f"VoxCPM segments have inconsistent sample rates: {sorted(sample_rates)}"
            )
        sample_rate = next(iter(sample_rates))

        audio_segments = [
            self._normalize_generated_audio(segment["waveform"]) for segment in segments
        ]
        text_chunks = [segment.get("text", "") for segment in segments]
        combined_audio, chunk_info = ChunkTimingHelper.combine_audio_with_timing(
            audio_segments=audio_segments,
            combination_method=method,
            silence_ms=silence_ms,
            crossfade_duration=0.1,
            sample_rate=sample_rate,
            text_length=len(" ".join(text_chunks)),
            original_text=original_text,
            text_chunks=text_chunks,
        )

        if return_info:
            return combined_audio, chunk_info or {}
        return combined_audio
