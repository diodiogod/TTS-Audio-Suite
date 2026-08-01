"""TADA text processor for chunking, character voices, pauses, and tags."""

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

from engines.tada.languages import format_tada_language_display, normalize_tada_language
from utils.audio.chunk_timing import ChunkTimingHelper
from utils.audio.edit_post_processor import process_segments as apply_edit_post_processing
from utils.text.character_parser import character_parser
from utils.text.pause_processor import PauseTagProcessor
from utils.text.segment_parameters import ParameterValidator, apply_segment_parameters
from utils.text.step_audio_editx_special_tags import get_edit_tags_for_segment
from utils.voice.character_logging import format_resolved_character_block, resolved_character_label
from utils.voice.discovery import get_available_characters, get_character_mapping, voice_discovery
from utils.voice.reference import effective_voice_audio


class TadaProcessor:
    """Orchestrate standard TTS Audio Suite text features around raw TADA calls."""

    SAMPLE_RATE = 24000

    def __init__(self, adapter, engine_config: Dict[str, Any]):
        self.adapter = adapter
        self.config = engine_config.copy() if engine_config else {}

    def update_config(self, new_config: Dict[str, Any]):
        self.config = new_config.copy() if new_config else {}
        self.adapter.update_config(self.config)

    @staticmethod
    def _check_interrupt(context: str = "generation") -> None:
        if model_management.interrupt_processing:
            raise InterruptedError(f"TADA {context} interrupted by user")

    @staticmethod
    def _reference_text(voice_ref: Dict[str, Any]) -> str:
        if not isinstance(voice_ref, dict):
            return ""
        return str(
            voice_ref.get("reference_text")
            or voice_ref.get("prompt_text")
            or voice_ref.get("text")
            or ""
        ).strip()

    def _validate_voice(self, character_name: str, voice_ref: Dict[str, Any]) -> None:
        if not isinstance(voice_ref, dict) or effective_voice_audio(voice_ref) is None:
            raise ValueError(
                f"TADA requires reference audio for '{character_name}'. Connect a narrator voice "
                "or save a character voice with audio and its exact transcript."
            )
        if not self._reference_text(voice_ref):
            raise ValueError(
                f"TADA requires the exact reference transcript for '{character_name}'. "
                "Use Character Voices or add the matching .txt/.reference.txt file. "
                "Automatic Parakeet transcription is intentionally disabled to prevent a hidden large-model download."
            )

    def _setup_character_parser(self, text: str) -> None:
        language_code = normalize_tada_language(self.config.get("language", "English")) or "en"
        character_parser.language_resolver.default_language = language_code
        character_parser.default_language = language_code

        text_characters = []
        for tag in re.findall(r"\[([^\]]+)\]", text or ""):
            if not tag.lower().startswith(("pause:", "wait:", "stop:")):
                text_characters.append(tag.split("|")[0].split(":")[-1].strip())

        available = set(get_available_characters() or [])
        for alias, target in voice_discovery.get_character_aliases().items():
            available.add(alias.lower())
            available.add(target.lower())
        available.update(name.lower() for name in text_characters if name)
        available.add("narrator")
        character_parser.set_available_characters(list(available))

        for character, language in voice_discovery.get_character_language_defaults().items():
            character_parser.set_character_language_default(character, language)
        character_parser.reset_session_cache()

    def _segment_config(self, segment: Any, base_config: Dict[str, Any], seed: int):
        params = segment.parameters or {}
        filtered = ParameterValidator.filter_parameters_for_engine(params, "tada") if params else {}
        current = apply_segment_parameters(base_config, params, "tada") if params else base_config.copy()
        current_seed = int(current.get("seed", seed))

        segment_language = getattr(segment, "language", None)
        base_language = normalize_tada_language(base_config.get("language", "English"))
        segment_language_code = (
            normalize_tada_language(segment_language) if segment_language else base_language
        )
        should_switch_language = bool(
            segment_language
            and (
                getattr(segment, "explicit_language", False)
                or segment_language_code != base_language
            )
        )
        if should_switch_language:
            current["language"] = format_tada_language_display(segment_language)
            print(f"  🌍 TADA language switched to: {current['language']}")
        return current, current_seed, filtered

    @staticmethod
    def _parameter_log(filtered: Dict[str, Any], config: Dict[str, Any], seed: int) -> str:
        if not filtered:
            return ""
        values = []
        if "seed" in filtered:
            values.append(f"seed={seed}")
        keys = (
            "acoustic_cfg_scale",
            "duration_cfg_scale",
            "num_flow_matching_steps",
            "noise_temperature",
            "speed_up_factor",
        )
        for key in keys:
            if key in config and (
                key in filtered
                or (key == "acoustic_cfg_scale" and "cfg" in filtered)
                or (key == "num_flow_matching_steps" and "num_steps" in filtered)
                or (key == "speed_up_factor" and "speed" in filtered)
            ):
                values.append(f"{key}={config[key]}")
        return ", ".join(values)

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
        del chunk_combination_method, silence_between_chunks_ms
        self._check_interrupt("text setup")
        self._setup_character_parser(text)

        narrator = voice_mapping.get("narrator", {})
        narrator_voice = narrator.copy() if isinstance(narrator, dict) else {"audio": narrator}
        base_config = self.config.copy()
        segments = character_parser.parse_text_segments(text)
        if not segments:
            segments = character_parser.parse_text_segments("narrator " + text)

        execution_config = base_config.copy()
        selected_model = str(
            execution_config.get("model_variant")
            or execution_config.get("model_name")
            or "TADA-1B"
        )
        segment_languages = {
            format_tada_language_display(segment.language)
            for segment in segments
            if getattr(segment, "language", None)
            and normalize_tada_language(segment.language) is not None
        }
        if "tada-1b" in selected_model.lower().replace("_", "-") and segment_languages:
            execution_config.update(
                {
                    "model_variant": "TADA-3B-ML",
                    "model_name": "TADA-3B-ML",
                    "model_path": "TADA-3B-ML",
                }
            )
            languages = ", ".join(sorted(segment_languages))
            print(f"🌍 Non-English character language detected ({languages})")
            print("🔄 Switching TADA-1B → TADA-3B-ML for this generation")

        characters = list({segment.character for segment in segments if segment.character})
        discovered_voices = get_character_mapping(characters, engine_type="tada")
        records: List[Dict[str, Any]] = []

        try:
            for segment_index, segment in enumerate(segments):
                self._check_interrupt(f"text segment {segment_index + 1}/{len(segments)}")
                segment_text = (segment.text or "").strip()
                if not segment_text:
                    continue

                current_config, current_seed, filtered = self._segment_config(
                    segment, execution_config, seed
                )
                self.adapter.update_config(current_config)
                character = segment.character or "narrator"
                voice_ref = narrator_voice.copy()

                mapped_voice = voice_mapping.get(character)
                if character != "narrator" and isinstance(mapped_voice, dict):
                    voice_ref = mapped_voice.copy()
                elif character != "narrator":
                    audio_path, reference_text = discovered_voices.get(character, (None, None))
                    if audio_path:
                        voice_ref = {"audio_path": audio_path, "reference_text": reference_text or ""}

                self._validate_voice(character, voice_ref)

                def generate_text(text_content: str, edit_tags: list) -> None:
                    if enable_chunking:
                        from utils.text.chunking import ImprovedChatterBoxChunker

                        max_chars = ImprovedChatterBoxChunker.validate_chunking_params(max_chars_per_chunk)
                        chunks = ImprovedChatterBoxChunker.split_into_chunks(text_content, max_chars=max_chars)
                    else:
                        chunks = [text_content]

                    display_name = resolved_character_label(character, voice_ref)
                    language = format_tada_language_display(current_config.get("language", "English"))
                    print(f"🎭 TADA - Generating for '{display_name}' (Language: {language})")
                    parameter_log = self._parameter_log(filtered, current_config, current_seed)
                    if parameter_log:
                        print(f"🎛️ TADA params: {parameter_log}")
                    if show_text_logging:
                        print(format_resolved_character_block(character, text_content, voice_ref))
                    if len(chunks) > 1:
                        print(f"📝 TADA chunking: {len(chunks)} chunks")

                    for chunk_index, chunk in enumerate(chunks):
                        self._check_interrupt(
                            f"chunk {chunk_index + 1}/{len(chunks)} for '{display_name}'"
                        )
                        audio = self.adapter.generate_single(
                            text=chunk,
                            voice_ref=voice_ref,
                            seed=current_seed + chunk_index,
                            enable_audio_cache=enable_audio_cache,
                            character_name=character,
                        )
                        if audio.dim() == 1:
                            audio = audio.unsqueeze(0)
                        records.append(
                            {
                                "waveform": audio.detach().cpu().float(),
                                "sample_rate": self.SAMPLE_RATE,
                                "text": chunk,
                                "edit_tags": edit_tags if chunk_index == 0 else [],
                            }
                        )

                if PauseTagProcessor.has_pause_tags(segment_text):
                    pause_segments, _ = PauseTagProcessor.parse_pause_tags(segment_text)
                    for fragment_type, fragment in pause_segments:
                        self._check_interrupt("pause-tag processing")
                        if fragment_type == "text":
                            clean_text, edit_tags = get_edit_tags_for_segment(fragment)
                            if clean_text.strip():
                                generate_text(clean_text.strip(), edit_tags)
                        elif fragment_type == "pause":
                            silence = PauseTagProcessor.create_silence_segment(
                                fragment, self.SAMPLE_RATE, torch.device("cpu"), torch.float32
                            )
                            if silence.dim() == 1:
                                silence = silence.unsqueeze(0)
                            records.append(
                                {
                                    "waveform": silence,
                                    "sample_rate": self.SAMPLE_RATE,
                                    "text": f"[pause:{fragment}s]",
                                    "edit_tags": [],
                                }
                            )
                else:
                    clean_text, edit_tags = get_edit_tags_for_segment(segment_text)
                    if clean_text.strip():
                        generate_text(clean_text.strip(), edit_tags)
        finally:
            self.adapter.update_config(base_config)

        if apply_edit_postprocessing and records and any(record["edit_tags"] for record in records):
            self._check_interrupt("edit post-processing")
            records = apply_edit_post_processing(records, engine_config=base_config)
        return records

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
            return (empty, {}) if return_info else empty

        self._check_interrupt("audio assembly")
        waveforms = [segment["waveform"] for segment in segments]
        text_chunks = [segment.get("text", "") for segment in segments]
        combined, chunk_info = ChunkTimingHelper.combine_audio_with_timing(
            audio_segments=waveforms,
            combination_method=method,
            silence_ms=silence_ms,
            crossfade_duration=0.1,
            sample_rate=self.SAMPLE_RATE,
            text_length=len(" ".join(text_chunks)),
            original_text=original_text,
            text_chunks=text_chunks,
        )
        return (combined, chunk_info) if return_info else combined
