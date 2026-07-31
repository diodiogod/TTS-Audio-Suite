"""Suite orchestration for Audio8 TTS text generation."""

import os
import re
import sys
from typing import Any, Dict, List, Tuple, Union

import comfy.model_management as model_management
import torch

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio.chunk_timing import ChunkTimingHelper
from utils.audio.edit_post_processor import (
    process_segments as apply_edit_post_processing,
)
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


class Audio8TTSProcessor:
    """Handle suite-level tags, voices, chunking, and Audio8 generation."""

    SAMPLE_RATE = 44100

    def __init__(self, adapter, engine_config: Dict[str, Any]):
        self.adapter = adapter
        self.config = engine_config.copy() if engine_config else {}

    def update_config(self, new_config: Dict[str, Any]):
        self.config = new_config.copy() if new_config else {}
        self.adapter.update_config(self.config)

    @staticmethod
    def _check_interrupt(context: str = "") -> None:
        if not model_management.interrupt_processing:
            return
        suffix = f" {context}" if context else ""
        raise InterruptedError(f"Audio8 TTS generation interrupted{suffix}")

    def _setup_character_parser(self, text: str) -> None:
        character_tags = re.findall(r"\[([^\]]+)\]", text or "")
        characters_from_tags = [
            tag.split("|")[0].strip()
            for tag in character_tags
            if not tag.lower().startswith("pause:")
        ]

        all_available = set(get_available_characters() or [])
        for alias, target in voice_discovery.get_character_aliases().items():
            all_available.add(alias.lower())
            all_available.add(target.lower())
        all_available.update(
            character.lower() for character in characters_from_tags if character
        )
        all_available.add("narrator")

        character_parser.set_available_characters(list(all_available))
        character_parser.reset_session_cache()

    @staticmethod
    def _as_voice_reference(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value.copy()
        if value is None:
            return {}
        return {"audio": value}

    def _resolve_voice(
        self,
        character: str,
        voice_mapping: Dict[str, Any],
        discovered_mapping: Dict[str, Tuple[Any, Any]],
    ) -> Dict[str, Any]:
        if character in voice_mapping:
            mapped_voice = self._as_voice_reference(voice_mapping[character])
            if effective_voice_audio(mapped_voice) is not None or mapped_voice.get(
                "reference_text"
            ):
                return mapped_voice

        if character != "narrator":
            audio_path, reference_text = discovered_mapping.get(character, (None, None))
            if audio_path:
                return {
                    "audio_path": audio_path,
                    "reference_text": reference_text or "",
                }
            print(
                f"⚠️ Audio8 TTS: No voice found for '{character}'; "
                "using narrator/no-reference fallback"
            )

        return self._as_voice_reference(voice_mapping.get("narrator"))

    @staticmethod
    def _voice_log_note(voice_ref: Dict[str, Any]) -> str:
        reference_text = ""
        if isinstance(voice_ref, dict):
            reference_text = str(voice_ref.get("reference_text") or "").strip()
        has_audio = (
            isinstance(voice_ref, dict) and effective_voice_audio(voice_ref) is not None
        )

        if not has_audio and not reference_text:
            return " [no reference voice]"
        if has_audio and reference_text:
            return f" [reference transcript: {len(reference_text)} chars]"
        if has_audio:
            return " [reference audio has no transcript; adapter will validate]"
        return " [reference transcript has no audio; adapter will validate]"

    @staticmethod
    def _format_parameter_log(
        filtered_params: Dict[str, Any],
        current_config: Dict[str, Any],
        current_seed: int,
    ) -> str:
        if not filtered_params:
            return ""

        values = []
        for key in ("seed", "temperature", "top_p", "top_k", "max_new_tokens"):
            if key not in filtered_params:
                continue
            value = current_seed if key == "seed" else current_config.get(key)
            values.append(f"{key}={value}")
        return ", ".join(values)

    @staticmethod
    def _validate_segment_config(
        config: Dict[str, Any],
        filtered_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply Audio8-specific bounds and keep retry budgets consistent."""
        if "top_p" in filtered_params and float(config["top_p"]) <= 0:
            raise ValueError("Audio8 segment top_p must be greater than 0")

        if "max_new_tokens" in filtered_params:
            max_new_tokens = int(config["max_new_tokens"])
            if max_new_tokens > 2048:
                raise ValueError("Audio8 segment max_new_tokens must be at most 2048")
            config["retry_max_new_tokens"] = max(
                max_new_tokens,
                int(config.get("retry_max_new_tokens", 2000)),
            )
        return config

    def _log_generation(
        self,
        character: str,
        text: str,
        voice_ref: Dict[str, Any],
        config: Dict[str, Any],
        chunk_count: int,
        parameter_log: str,
        show_text_logging: bool,
    ) -> None:
        display_name = resolved_character_label(character, voice_ref)
        print(
            f"🎭 Audio8 TTS - Generating for '{display_name}'"
            f"{self._voice_log_note(voice_ref)}"
        )
        print(
            "   Settings: "
            f"mode={'Sampling' if config.get('do_sample', True) else 'Greedy'}, "
            f"temperature={config.get('temperature', 0.8)}, "
            f"top_p={config.get('top_p', 0.95)}, "
            f"top_k={config.get('top_k', 50)}, "
            f"max_new_tokens={config.get('max_new_tokens', 1024)}"
        )
        if parameter_log:
            print(f"🎛️ Audio8 TTS segment params: {parameter_log}")
        if show_text_logging:
            print(format_resolved_character_block(character, text, voice_ref))
        if chunk_count > 1:
            print(
                f"📝 Audio8 TTS: Chunking '{display_name}' into "
                f"{chunk_count} suite chunks"
            )

    @staticmethod
    def _normalize_audio(audio: Any) -> torch.Tensor:
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio, dtype=torch.float32)
        audio = audio.detach().to(device="cpu", dtype=torch.float32)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        if audio.dim() != 2:
            raise ValueError(
                "Audio8 adapter returned an invalid waveform shape; "
                "expected [channels, samples]"
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
        **_unused,
    ) -> List[Dict[str, Any]]:
        del chunk_combination_method, silence_between_chunks_ms

        self._check_interrupt()
        self._setup_character_parser(text)
        base_config = self.config.copy()
        segment_objects = character_parser.parse_text_segments(text)
        if not segment_objects:
            segment_objects = character_parser.parse_text_segments(
                "narrator " + (text or "")
            )

        characters = list(
            {segment.character for segment in segment_objects if segment.character}
        )
        discovered_mapping = get_character_mapping(characters, engine_type="audio_only")
        segment_records: List[Dict[str, Any]] = []

        try:
            for segment_index, segment in enumerate(segment_objects):
                character = segment.character or "narrator"
                self._check_interrupt(
                    f"before segment {segment_index + 1}/{len(segment_objects)} "
                    f"for '{character}'"
                )
                segment_text = (segment.text or "").strip()
                if not segment_text:
                    continue

                segment_params = segment.parameters or {}
                filtered_params = ParameterValidator.filter_parameters_for_engine(
                    segment_params, "audio8_tts"
                )
                current_config = (
                    apply_segment_parameters(base_config, filtered_params, "audio8_tts")
                    if filtered_params
                    else base_config.copy()
                )
                current_config = self._validate_segment_config(
                    current_config, filtered_params
                )
                current_seed = int(current_config.get("seed", seed))
                parameter_log = self._format_parameter_log(
                    filtered_params, current_config, current_seed
                )
                self.adapter.update_config(current_config)

                voice_ref = self._resolve_voice(
                    character, voice_mapping, discovered_mapping
                )
                seed_offset = 0

                def generate_chunks(text_content: str, edit_tags: list) -> None:
                    nonlocal seed_offset
                    clean_content = (text_content or "").strip()
                    if not clean_content:
                        return

                    if enable_chunking:
                        from utils.text.chunking import ImprovedChatterBoxChunker

                        max_chars = ImprovedChatterBoxChunker.validate_chunking_params(
                            max_chars_per_chunk
                        )
                        chunks = ImprovedChatterBoxChunker.split_into_chunks(
                            clean_content, max_chars=max_chars
                        )
                    else:
                        chunks = [clean_content]
                    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

                    self._log_generation(
                        character=character,
                        text=clean_content,
                        voice_ref=voice_ref,
                        config=current_config,
                        chunk_count=len(chunks),
                        parameter_log=parameter_log,
                        show_text_logging=show_text_logging,
                    )

                    for chunk_index, chunk in enumerate(chunks):
                        self._check_interrupt(
                            f"before chunk {chunk_index + 1}/{len(chunks)} "
                            f"for '{character}'"
                        )
                        chunk_seed = current_seed + seed_offset
                        seed_offset += 1
                        audio = self.adapter.generate_single(
                            text=chunk,
                            voice_ref=voice_ref,
                            seed=chunk_seed,
                            enable_audio_cache=enable_audio_cache,
                            character_name=character,
                        )
                        segment_records.append(
                            {
                                "waveform": self._normalize_audio(audio),
                                "sample_rate": self.SAMPLE_RATE,
                                "text": chunk,
                                "edit_tags": (edit_tags if chunk_index == 0 else []),
                            }
                        )

                if PauseTagProcessor.has_pause_tags(segment_text):
                    pause_segments, _ = PauseTagProcessor.parse_pause_tags(segment_text)
                    for fragment_type, fragment_content in pause_segments:
                        self._check_interrupt(
                            f"while processing segment {segment_index + 1}"
                        )
                        if fragment_type == "text":
                            clean_text, edit_tags = get_edit_tags_for_segment(
                                fragment_content
                            )
                            generate_chunks(clean_text, edit_tags)
                        elif fragment_type == "pause":
                            silence = PauseTagProcessor.create_silence_segment(
                                fragment_content,
                                self.SAMPLE_RATE,
                                torch.device("cpu"),
                                torch.float32,
                            )
                            if silence.dim() == 1:
                                silence = silence.unsqueeze(0)
                            segment_records.append(
                                {
                                    "waveform": silence.cpu(),
                                    "sample_rate": self.SAMPLE_RATE,
                                    "text": f"[pause:{fragment_content}s]",
                                    "edit_tags": [],
                                }
                            )
                else:
                    clean_text, edit_tags = get_edit_tags_for_segment(segment_text)
                    generate_chunks(clean_text, edit_tags)
        finally:
            self.adapter.update_config(base_config)

        if (
            apply_edit_postprocessing
            and segment_records
            and any(record.get("edit_tags") for record in segment_records)
        ):
            self._check_interrupt("before edit post-processing")
            segment_records = apply_edit_post_processing(
                segment_records, engine_config=base_config
            )
            for record in segment_records:
                record["waveform"] = self._normalize_audio(record["waveform"])

        return segment_records

    def combine_audio_segments(
        self,
        segments: List[Dict[str, Any]],
        method: str = "auto",
        silence_ms: int = 100,
        original_text: str = "",
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        self._check_interrupt("before audio assembly")
        if not segments:
            empty = torch.zeros(0, dtype=torch.float32)
            return (empty, {}) if return_info else empty

        text_chunks = [segment.get("text", "") for segment in segments]
        combined_audio, chunk_info = ChunkTimingHelper.combine_audio_with_timing(
            audio_segments=[segment["waveform"] for segment in segments],
            combination_method=method,
            silence_ms=silence_ms,
            crossfade_duration=0.1,
            sample_rate=self.SAMPLE_RATE,
            text_length=len(" ".join(text_chunks)),
            original_text=original_text,
            text_chunks=text_chunks,
        )
        return (combined_audio, chunk_info) if return_info else combined_audio
