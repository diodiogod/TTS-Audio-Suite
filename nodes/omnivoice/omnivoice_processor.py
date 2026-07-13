"""
OmniVoice processor.

Handles character switching, pause tags, parameter switching, and native
OmniVoice long-form generation orchestration.
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import comfy.model_management as model_management
import torch

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio.chunk_timing import ChunkTimingHelper
from utils.text.character_parser import character_parser
from utils.text.omnivoice_special_tags import convert_omnivoice_special_tags
from utils.text.pause_processor import PauseTagProcessor
from utils.voice.character_logging import (
    format_resolved_character_block,
    resolved_character_label,
)
from utils.voice.reference import effective_voice_audio
from utils.text.segment_parameters import ParameterValidator, apply_segment_parameters
from utils.voice.discovery import get_available_characters, get_character_mapping, voice_discovery


class OmniVoiceProcessor:
    """Internal processor for OmniVoice generation."""

    SAMPLE_RATE = 24000
    MIN_AUTO_SPEECH_FRAGMENT_DURATION = 0.12

    def __init__(self, adapter, engine_config: Dict[str, Any]):
        self.adapter = adapter
        self.config = engine_config.copy() if engine_config else {}

    def update_config(self, new_config: Dict[str, Any]):
        self.config = new_config.copy() if new_config else {}
        self.adapter.update_config(self.config)

    @staticmethod
    def _check_interrupt(
        item_index: Optional[int] = None,
        total_items: Optional[int] = None,
        character_name: Optional[str] = None,
    ) -> None:
        if not model_management.interrupt_processing:
            return

        if item_index is not None and total_items is not None:
            if character_name:
                raise InterruptedError(
                    f"OmniVoice text generation interrupted at item "
                    f"{item_index + 1}/{total_items}, character '{character_name}'"
                )
            raise InterruptedError(
                f"OmniVoice text generation interrupted at item {item_index + 1}/{total_items}"
            )

        raise InterruptedError("OmniVoice text generation interrupted by user")

    @staticmethod
    def _first_non_none(*values):
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return None

    @staticmethod
    def _voice_log_note(voice_ref: Dict[str, Any]) -> str:
        if not isinstance(voice_ref, dict):
            return " [auto/default voice]"
        audio_ref = effective_voice_audio(voice_ref)
        if audio_ref is None:
            return " [auto/default voice]"
        reference_text = voice_ref.get("reference_text")
        if reference_text:
            return f" [ref text: {len(reference_text)} chars]"
        return " [auto-transcribe reference]"

    def _log_generation_text(
        self,
        character_name: str,
        text_content: str,
        language: str,
        voice_ref: Dict[str, Any],
        parameter_log: str = "",
        show_text_content: bool = True,
    ) -> None:
        voice_note = self._voice_log_note(voice_ref)
        display_name = resolved_character_label(character_name, voice_ref)
        print(f"🎭 OmniVoice - Generating for '{display_name}' (Language: {language}){voice_note}:")
        if parameter_log:
            print(f"🎛️ OmniVoice params: {parameter_log}")
        if show_text_content:
            print(format_resolved_character_block(character_name, text_content, voice_ref))

    @staticmethod
    def _format_parameter_log(
        filtered_params: Dict[str, Any],
        current_config: Dict[str, Any],
        current_seed: int,
    ) -> str:
        if not filtered_params:
            return ""

        ordered_parts = []
        if "seed" in filtered_params:
            ordered_parts.append(f"seed={current_seed}")

        for key in (
            "num_steps",
            "guidance_scale",
            "t_shift",
            "speed",
            "duration",
            "layer_penalty_factor",
            "position_temperature",
            "class_temperature",
            "audio_chunk_duration",
            "audio_chunk_threshold",
        ):
            if key in filtered_params:
                ordered_parts.append(f"{key}={current_config.get(key)}")

        if "language" in filtered_params:
            ordered_parts.append(f"language={current_config.get('language')}")

        remaining_keys = [
            key for key in filtered_params.keys()
            if key not in {
                "seed",
                "num_steps",
                "guidance_scale",
                "t_shift",
                "speed",
                "duration",
                "layer_penalty_factor",
                "position_temperature",
                "class_temperature",
                "audio_chunk_duration",
                "audio_chunk_threshold",
                "language",
            }
        ]
        for key in remaining_keys:
            ordered_parts.append(f"{key}={current_config.get(key, filtered_params[key])}")

        return ", ".join(ordered_parts)

    def _setup_character_parser(self, text: str):
        configured_language = str(self.config.get("language", "Auto") or "Auto").strip()
        if configured_language.lower() in {"auto", "none"}:
            resolved_language = "en"
        else:
            resolved_language = configured_language.lower()

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

        global_language = str(base_config.get("language", "Auto") or "Auto").strip().lower()
        parser_default = str(character_parser.default_language or "en").strip().lower()
        segment_language_code = str(segment_language).strip().lower()

        if segment_language_code == parser_default:
            return False
        return segment_language_code != global_language

    @staticmethod
    def _text_weight(text: str) -> int:
        compact = re.sub(r"\s+", "", text or "")
        return max(len(compact), 1)

    def _build_generation_items(
        self,
        text: str,
        voice_mapping: Dict[str, Any],
        seed: int,
        show_text_logging: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self._setup_character_parser(text)

        narrator_info = voice_mapping.get("narrator", {})
        narrator_voice = narrator_info if isinstance(narrator_info, dict) else {"audio": narrator_info}

        base_config = self.config.copy()
        segment_objects = character_parser.parse_text_segments(text, engine_type="omnivoice")
        if not segment_objects:
            segment_objects = character_parser.parse_text_segments("narrator " + text, engine_type="omnivoice")

        characters = list({seg.character for seg in segment_objects if seg.character})
        character_mapping = get_character_mapping(characters, engine_type="audio_only")

        generation_items: List[Dict[str, Any]] = []
        native_chunking_notice_shown = False

        for seg in segment_objects:
            self._check_interrupt()
            segment_text = (seg.text or "").strip()
            if not segment_text:
                continue
            segment_text = convert_omnivoice_special_tags(segment_text)

            segment_params = seg.parameters if seg.parameters else {}
            current_config = base_config.copy()
            current_seed = seed
            filtered_params: Dict[str, Any] = {}
            if segment_params:
                filtered_params = ParameterValidator.filter_parameters_for_engine(segment_params, "omnivoice")
                current_config = apply_segment_parameters(base_config, segment_params, "omnivoice")
                if "seed" in current_config:
                    current_seed = int(current_config.get("seed", seed))

            if self._should_apply_segment_language(seg, base_config):
                current_config = current_config.copy()
                current_config["language"] = seg.language
                print(f"  🌍 OmniVoice language switched to: {seg.language}")

            segment_language = str(current_config.get("language", "Auto") or "Auto")
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

            if not native_chunking_notice_shown and len(segment_text) > 400:
                print(
                    "ℹ️ OmniVoice uses its official native long-form chunking "
                    "(audio_chunk_duration / audio_chunk_threshold) instead of suite char-based chunk splitting."
                )
                native_chunking_notice_shown = True

            explicit_duration = None
            if "duration" in filtered_params:
                duration_value = float(current_config.get("duration", 0.0))
                if duration_value > 0:
                    explicit_duration = duration_value

            def _append_text_item(text_content: str):
                text_content = (text_content or "").strip()
                if not text_content:
                    return
                generation_items.append(
                    {
                        "kind": "text",
                        "text": text_content,
                        "config": current_config.copy(),
                        "seed": current_seed,
                        "voice_ref": voice_ref.copy() if isinstance(voice_ref, dict) else voice_ref,
                        "character_name": char_name,
                        "language": segment_language,
                        "parameter_log": parameter_log,
                        "show_text_logging": show_text_logging,
                        "explicit_target_duration": explicit_duration,
                    }
                )

            if PauseTagProcessor.has_pause_tags(segment_text):
                raw_pause_segments, _ = PauseTagProcessor.parse_pause_tags(segment_text)
                for frag_type, frag_content in raw_pause_segments:
                    if frag_type == "text":
                        _append_text_item(frag_content)
                    elif frag_type == "pause":
                        generation_items.append(
                            {
                                "kind": "pause",
                                "duration": float(frag_content),
                                "effective_duration": float(frag_content),
                                "text": f"[pause:{float(frag_content):.3f}s]",
                            }
                        )
            else:
                _append_text_item(segment_text)

        return generation_items, base_config

    def _apply_duration_budget(
        self,
        generation_items: List[Dict[str, Any]],
        total_duration: float,
    ) -> None:
        if total_duration <= 0:
            return

        pause_items = [item for item in generation_items if item["kind"] == "pause"]
        spoken_items = [item for item in generation_items if item["kind"] == "text"]
        variable_spoken_items = [item for item in spoken_items if not item.get("explicit_target_duration")]
        explicit_spoken_items = [item for item in spoken_items if item.get("explicit_target_duration")]

        total_pause_duration = sum(float(item.get("duration", 0.0)) for item in pause_items)
        total_explicit_speech = sum(float(item.get("explicit_target_duration", 0.0)) for item in explicit_spoken_items)

        desired_variable_speech = min(
            max(total_duration - total_explicit_speech, 0.0),
            self.MIN_AUTO_SPEECH_FRAGMENT_DURATION * len(variable_spoken_items),
        )
        max_pause_budget = max(0.0, total_duration - total_explicit_speech - desired_variable_speech)

        pause_scale = 1.0
        if total_pause_duration > 0 and total_pause_duration > max_pause_budget:
            pause_scale = max_pause_budget / total_pause_duration if max_pause_budget > 0 else 0.0
            print(
                "⚠️ OmniVoice SRT: Pause tags exceed available subtitle time. "
                f"Compressing pauses from {total_pause_duration:.3f}s to {total_pause_duration * pause_scale:.3f}s."
            )

        adjusted_pause_total = total_pause_duration * pause_scale
        for pause_item in pause_items:
            pause_item["effective_duration"] = float(pause_item.get("duration", 0.0)) * pause_scale

        remaining_variable_speech = total_duration - total_explicit_speech - adjusted_pause_total
        if remaining_variable_speech <= 0 and variable_spoken_items:
            fallback_duration = max(0.05, min(total_duration, self.MIN_AUTO_SPEECH_FRAGMENT_DURATION))
            remaining_variable_speech = fallback_duration * len(variable_spoken_items)
            print(
                "⚠️ OmniVoice SRT: Subtitle timing leaves no room for auto-allocated speech after pauses/fixed durations. "
                f"Using minimal speech targets of {fallback_duration:.3f}s per spoken fragment."
            )

        total_weight = sum(self._text_weight(item["text"]) for item in variable_spoken_items)
        allocated_total = 0.0
        for index, item in enumerate(variable_spoken_items):
            if total_weight <= 0:
                target_duration = remaining_variable_speech / len(variable_spoken_items)
            elif index == len(variable_spoken_items) - 1:
                target_duration = max(0.0, remaining_variable_speech - allocated_total)
            else:
                target_duration = remaining_variable_speech * (self._text_weight(item["text"]) / total_weight)
                allocated_total += target_duration
            item["allocated_target_duration"] = max(0.0, float(target_duration))

        for item in explicit_spoken_items:
            item["allocated_target_duration"] = float(item["explicit_target_duration"])

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
        duration_budget: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        del enable_chunking, max_chars_per_chunk  # OmniVoice uses native duration-based chunking
        del apply_edit_postprocessing  # OmniVoice uses native inline tags instead of Step Audio EditX inline post-processing
        generation_items, _base_config = self._build_generation_items(
            text=text,
            voice_mapping=voice_mapping,
            seed=seed,
            show_text_logging=show_text_logging,
        )

        if duration_budget is not None and duration_budget > 0:
            self._apply_duration_budget(generation_items, float(duration_budget))

        segment_records: List[Dict[str, Any]] = []
        for item_index, item in enumerate(generation_items):
            self._check_interrupt(
                item_index=item_index,
                total_items=len(generation_items),
                character_name=item.get("character_name"),
            )
            if item["kind"] == "pause":
                silence_duration = float(item.get("effective_duration", item.get("duration", 0.0)))
                silence = PauseTagProcessor.create_silence_segment(
                    silence_duration,
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
                        "text": f"[pause:{silence_duration:.3f}s]",
                    }
                )
                continue

            current_config = item["config"].copy()
            allocated_target_duration = item.get("allocated_target_duration")
            if allocated_target_duration is not None and allocated_target_duration > 0:
                current_config["duration"] = float(allocated_target_duration)

            self.adapter.update_config(current_config)
            self._log_generation_text(
                character_name=item["character_name"],
                text_content=item["text"],
                language=item["language"],
                voice_ref=item["voice_ref"],
                parameter_log=item["parameter_log"],
                show_text_content=item["show_text_logging"],
            )
            self._check_interrupt(
                item_index=item_index,
                total_items=len(generation_items),
                character_name=item.get("character_name"),
            )
            audio = self.adapter.generate_single(
                text=item["text"],
                voice_ref=item["voice_ref"],
                seed=item["seed"],
                enable_audio_cache=enable_audio_cache,
                character_name=item["character_name"],
            )
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            segment_records.append(
                {
                    "waveform": audio.cpu(),
                    "sample_rate": self.SAMPLE_RATE,
                    "text": item["text"],
                }
            )

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
