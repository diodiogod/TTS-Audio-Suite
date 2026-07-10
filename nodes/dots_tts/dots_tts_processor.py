"""
Dots TTS processor.

Handles chunking, character switching, pause tags, and orchestration for Dots TTS.
"""

import os
import re
import sys
from typing import Any, Dict, List, Tuple, Union

import torch

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio.chunk_timing import ChunkTimingHelper
from utils.audio.edit_post_processor import process_segments as apply_edit_post_processing
from engines.dots_tts.languages import format_dots_language_display, normalize_dots_language
from utils.text.character_parser import character_parser
from utils.text.pause_processor import PauseTagProcessor
from utils.voice.character_logging import (
    format_resolved_character_block,
    resolved_character_label,
)
from utils.text.segment_parameters import ParameterValidator, apply_segment_parameters
from utils.text.step_audio_editx_special_tags import get_edit_tags_for_segment
from utils.voice.discovery import get_available_characters, get_character_mapping, voice_discovery


class DotsTTSProcessor:
    """Internal processor for Dots TTS generation."""

    SAMPLE_RATE = 48000

    def __init__(self, adapter, engine_config: Dict[str, Any]):
        self.adapter = adapter
        self.config = engine_config.copy() if engine_config else {}

    def update_config(self, new_config: Dict[str, Any]):
        self.config = new_config.copy() if new_config else {}
        self.adapter.update_config(self.config)

    @staticmethod
    def _voice_log_note(voice_ref: Dict[str, Any]) -> str:
        if not isinstance(voice_ref, dict):
            return " [⚠️ No voice reference - will use default]"
        audio_path = voice_ref.get("audio_path") or voice_ref.get("audio")
        if not audio_path:
            return " [⚠️ No voice reference - will use default]"
        reference_text = voice_ref.get("reference_text")
        if reference_text:
            return f" [ref text: {len(reference_text)} chars]"
        return ""

    def _log_generation_text(
        self,
        character_name: str,
        text_content: str,
        language: str,
        voice_ref: Dict[str, Any],
        chunk_count: int,
        parameter_log: str = "",
        show_text_content: bool = True,
    ) -> None:
        voice_note = self._voice_log_note(voice_ref)
        display_name = resolved_character_label(character_name, voice_ref)
        print(f"🎭 Dots TTS - Generating for '{display_name}' (Language: {language}){voice_note}:")
        if parameter_log:
            print(f"🎛️ Dots TTS params: {parameter_log}")
        if show_text_content:
            print(format_resolved_character_block(character_name, text_content, voice_ref))
        if chunk_count > 1:
            print(
                f"📝 Chunking {display_name}'s text into {chunk_count} chunks "
                f"(Language: {language}){voice_note}"
            )

    @staticmethod
    def _format_parameter_log(filtered_params: Dict[str, Any], current_config: Dict[str, Any], current_seed: int) -> str:
        if not filtered_params:
            return ""

        ordered_parts = []
        if "seed" in filtered_params:
            ordered_parts.append(f"seed={current_seed}")

        for key in ("num_steps", "guidance_scale", "speaker_scale", "max_generate_length"):
            if key in filtered_params:
                ordered_parts.append(f"{key}={current_config.get(key)}")

        for key in ("normalize_text", "language"):
            if key in filtered_params:
                value = current_config.get(key)
                if key == "language":
                    value = format_dots_language_display(value)
                ordered_parts.append(f"{key}={value}")

        remaining_keys = [
            key for key in filtered_params.keys()
            if key not in {"seed", "num_steps", "guidance_scale", "speaker_scale", "max_generate_length", "normalize_text", "language"}
        ]
        for key in remaining_keys:
            ordered_parts.append(f"{key}={current_config.get(key, filtered_params[key])}")

        return ", ".join(ordered_parts)

    def _setup_character_parser(self, text: str):
        configured_language = str(self.config.get("language", "auto") or "auto").strip()
        normalized_language = normalize_dots_language(configured_language)
        if normalized_language is None:
            resolved_language = "en"
        elif normalized_language == "auto_detect":
            resolved_language = "en"
        else:
            resolved_language = normalized_language.lower()

        character_parser.language_resolver.default_language = resolved_language
        character_parser.default_language = resolved_language

        character_tags = re.findall(r"\[([^\]]+)\]", text or "")
        characters_from_tags = []
        for tag in character_tags:
            if not tag.startswith("pause:"):
                characters_from_tags.append(tag.split("|")[0].strip())

        available_chars = get_available_characters()
        character_aliases = voice_discovery.get_character_aliases()

        all_available = set()
        if available_chars:
            all_available.update(available_chars)
        for alias, target in character_aliases.items():
            all_available.add(alias.lower())
            all_available.add(target.lower())
        for char in characters_from_tags:
            all_available.add(char.lower())
        all_available.add("narrator")

        character_parser.set_available_characters(list(all_available))

        char_lang_defaults = voice_discovery.get_character_language_defaults()
        for char, lang in char_lang_defaults.items():
            character_parser.set_character_language_default(char, lang)

        character_parser.reset_session_cache()

    @staticmethod
    def _should_apply_segment_language(seg: Any, base_config: Dict[str, Any]) -> bool:
        segment_language = getattr(seg, "language", None)
        if not segment_language:
            return False

        if getattr(seg, "explicit_language", False):
            return True

        global_language = normalize_dots_language(base_config.get("language", "auto"))
        parser_default = normalize_dots_language(character_parser.default_language)
        segment_language_code = normalize_dots_language(segment_language)

        # Character parser always resolves untagged text to its fallback default
        # language. For Dots we must not feed that fallback back into generation,
        # or Auto/None collapse into English cache/runtime behavior.
        if segment_language_code == parser_default:
            return False

        return segment_language_code != global_language

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
        self._setup_character_parser(text)

        narrator_info = voice_mapping.get("narrator", {})
        narrator_voice = narrator_info if isinstance(narrator_info, dict) else {"audio": narrator_info}

        base_config = self.config.copy()
        segment_objects = character_parser.parse_text_segments(text)
        if not segment_objects:
            segment_objects = character_parser.parse_text_segments("narrator " + text)

        characters = list({seg.character for seg in segment_objects if seg.character})
        character_mapping = get_character_mapping(characters, engine_type="audio_only")

        segment_records: List[Dict[str, Any]] = []
        for seg in segment_objects:
            segment_text = (seg.text or "").strip()
            if not segment_text:
                continue

            segment_params = seg.parameters if seg.parameters else {}
            current_config = base_config
            current_seed = seed
            filtered_params: Dict[str, Any] = {}
            if segment_params:
                filtered_params = ParameterValidator.filter_parameters_for_engine(segment_params, "dots_tts")
                current_config = apply_segment_parameters(base_config, segment_params, "dots_tts")
                if "seed" in current_config:
                    current_seed = int(current_config.get("seed", seed))

            if self._should_apply_segment_language(seg, base_config):
                current_config = current_config.copy()
                current_config["language"] = seg.language
                global_language = normalize_dots_language(base_config.get("language", "auto"))
                segment_language_code = normalize_dots_language(seg.language)
                if segment_language_code != global_language:
                    print(
                        "  🌍 Dots TTS language switched to: "
                        f"{format_dots_language_display(seg.language)}"
                    )

            self.adapter.update_config(current_config)
            segment_language = format_dots_language_display(current_config.get("language", "Auto"))
            parameter_log = self._format_parameter_log(filtered_params, current_config, current_seed)

            voice_ref = narrator_voice.copy()
            char_name = seg.character or "narrator"
            if char_name != "narrator" and char_name in voice_mapping:
                char_info = voice_mapping[char_name]
                if isinstance(char_info, dict):
                    voice_ref = char_info.copy()
            elif char_name != "narrator":
                audio_path, ref_text = character_mapping.get(char_name, (None, None))
                if audio_path:
                    voice_ref = {
                        "audio_path": audio_path,
                        "reference_text": ref_text or "",
                    }

            def _generate_chunks(text_content: str, edit_tags: list):
                if enable_chunking:
                    from utils.text.chunking import ImprovedChatterBoxChunker

                    max_chars = ImprovedChatterBoxChunker.validate_chunking_params(max_chars_per_chunk)
                    chunks = ImprovedChatterBoxChunker.split_into_chunks(text_content, max_chars=max_chars)
                else:
                    chunks = [text_content]

                self._log_generation_text(
                    character_name=char_name,
                    text_content=text_content,
                    language=segment_language,
                    voice_ref=voice_ref,
                    chunk_count=len(chunks),
                    parameter_log=parameter_log,
                    show_text_content=show_text_logging,
                )

                for chunk_idx, chunk in enumerate(chunks):
                    audio = self.adapter.generate_single(
                        text=chunk,
                        voice_ref=voice_ref,
                        seed=current_seed + chunk_idx,
                        enable_audio_cache=enable_audio_cache,
                        character_name=char_name,
                    )
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                    segment_records.append(
                        {
                            "waveform": audio.cpu(),
                            "sample_rate": self.SAMPLE_RATE,
                            "text": chunk,
                            "edit_tags": edit_tags if chunk_idx == 0 else [],
                        }
                    )

            if PauseTagProcessor.has_pause_tags(segment_text):
                raw_pause_segments, _ = PauseTagProcessor.parse_pause_tags(segment_text)
                for frag_type, frag_content in raw_pause_segments:
                    if frag_type == "text":
                        frag_clean, frag_edit_tags = get_edit_tags_for_segment(frag_content)
                        frag_clean = frag_clean.strip()
                        if frag_clean:
                            _generate_chunks(frag_clean, frag_edit_tags)
                    elif frag_type == "pause":
                        silence = PauseTagProcessor.create_silence_segment(
                            frag_content,
                            self.SAMPLE_RATE,
                            torch.device("cpu"),
                            torch.float32,
                        )
                        if silence.dim() == 1:
                            silence = silence.unsqueeze(0)
                        segment_records.append(
                            {
                                "waveform": silence,
                                "sample_rate": self.SAMPLE_RATE,
                                "text": f"[pause:{frag_content}s]",
                                "edit_tags": [],
                            }
                        )
            else:
                clean_text, edit_tags = get_edit_tags_for_segment(segment_text)
                _generate_chunks(clean_text, edit_tags)

        if apply_edit_postprocessing and segment_records and any(seg["edit_tags"] for seg in segment_records):
            segment_records = apply_edit_post_processing(segment_records, engine_config=base_config)

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
            empty = torch.zeros(0, dtype=torch.float32)
            if return_info:
                return empty, {}
            return empty

        audio_segments = [seg["waveform"] for seg in segments]
        text_chunks = [seg.get("text", "") for seg in segments]
        text_length = len(" ".join(text_chunks))

        combined_audio, chunk_info = ChunkTimingHelper.combine_audio_with_timing(
            audio_segments=audio_segments,
            combination_method=method,
            silence_ms=silence_ms,
            crossfade_duration=0.1,
            sample_rate=self.SAMPLE_RATE,
            text_length=text_length,
            original_text=original_text,
            text_chunks=text_chunks,
        )

        if return_info:
            return combined_audio, chunk_info
        return combined_audio
