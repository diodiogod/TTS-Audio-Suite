"""Fish Audio S2 Pro text orchestration."""

import re
from typing import Any, Dict, List

import torch
import comfy.model_management as model_management

from utils.audio.chunk_timing import ChunkTimingHelper
from utils.text.character_parser import character_parser
from utils.text.fish_audio_s2_tags import get_fish_language_instruction
from utils.text.pause_processor import PauseTagProcessor
from utils.text.segment_parameters import ParameterValidator, apply_segment_parameters
from utils.voice.discovery import (
    get_available_characters,
    get_character_default_language,
    get_character_mapping,
    voice_discovery,
)


class FishAudioS2Processor:
    SAMPLE_RATE = 44100

    def __init__(self, adapter, engine_config: Dict[str, Any]):
        self.adapter = adapter
        self.config = dict(engine_config or {})

    def update_config(self, config):
        self.config = dict(config or {})
        self.adapter.update_config(self.config)

    def _setup_character_parser(self, text):
        tags = [tag.split("|")[0].strip() for tag in re.findall(r"\[([^\]]+)\]", text or "") if not tag.lower().startswith(("pause:", "wait:", "stop:"))]
        available = set(get_available_characters() or [])
        aliases = voice_discovery.get_character_aliases()
        available.update(aliases.keys())
        available.update(aliases.values())
        available.update(tag.lower() for tag in tags)
        available.add("narrator")
        character_parser.set_available_characters(list(available))
        for character, language in voice_discovery.get_character_language_defaults().items():
            character_parser.set_character_language_default(character, language)
        character_parser.reset_session_cache()

    def get_character_order(self, text):
        self._setup_character_parser(text)
        order = []
        for segment in character_parser.parse_text_segments(text, engine_type="fish_audio_s2"):
            character = segment.character or "narrator"
            if character not in order:
                order.append(character)
        if "narrator" in order:
            order = ["narrator"] + [character for character in order if character != "narrator"]
        return order

    def _apply_language_instruction(self, text: str, segment) -> str:
        if self.config.get("language_prompting", "Auto Inline Tag") != "Auto Inline Tag":
            return text
        instruction = get_fish_language_instruction(
            getattr(segment, "language", None),
            explicit=bool(getattr(segment, "explicit_language", False)),
        )
        if not instruction:
            return text
        stripped = (text or "").lstrip()
        if stripped.startswith("<"):
            return text
        return f"<{instruction}> {text}"

    def _get_effective_language_instruction(self, segment, override_voice_ref: Dict[str, Any] | None) -> str | None:
        explicit_language = bool(getattr(segment, "explicit_language", False))
        if explicit_language:
            return get_fish_language_instruction(
                getattr(segment, "language", None),
                explicit=True,
            )
        if isinstance(override_voice_ref, dict):
            override_character = override_voice_ref.get("character_name")
            if override_character:
                override_language = get_character_default_language(str(override_character))
                return get_fish_language_instruction(override_language, explicit=False)
            return None
        return get_fish_language_instruction(
            getattr(segment, "language", None),
            explicit=False,
        )

    def _apply_effective_language_instruction(
        self, text: str, segment, override_voice_ref: Dict[str, Any] | None
    ) -> tuple[str, str | None]:
        if self.config.get("language_prompting", "Auto Inline Tag") != "Auto Inline Tag":
            return text, None
        instruction = self._get_effective_language_instruction(segment, override_voice_ref)
        if not instruction:
            return text, None
        stripped = (text or "").lstrip()
        if stripped.startswith("<"):
            return text, instruction
        return f"<{instruction}> {text}", instruction

    @staticmethod
    def _voice_for(character, voice_mapping, discovered, narrator):
        if character in voice_mapping and isinstance(voice_mapping[character], dict):
            return dict(voice_mapping[character])
        audio_path, ref_text = discovered.get(character, (None, None))
        if audio_path:
            return {"audio_path": audio_path, "reference_text": ref_text or ""}
        return dict(narrator)

    @staticmethod
    def _speaker_display_name(speaker_number: int, voice_ref: Dict[str, Any] | None) -> str:
        if isinstance(voice_ref, dict):
            if voice_ref.get("character_name"):
                return str(voice_ref["character_name"])
            if voice_ref.get("audio_path"):
                import os
                return os.path.splitext(os.path.basename(str(voice_ref["audio_path"])))[0]
        if speaker_number == 1:
            return "Speaker 1 (Narrator)"
        return f"Speaker {speaker_number}"

    def process_text(self, text: str, voice_mapping: Dict[str, Any], seed: int,
                     enable_chunking: bool = True, max_chars_per_chunk: int = 400,
                     chunk_combination_method: str = "auto", silence_between_chunks_ms: int = 100,
                     enable_audio_cache: bool = True, apply_edit_postprocessing: bool = True,
                     show_text_logging: bool = True, reference_order=None) -> List[Dict[str, Any]]:
        del enable_chunking, max_chars_per_chunk, chunk_combination_method
        del silence_between_chunks_ms, apply_edit_postprocessing
        self._setup_character_parser(text)
        segments = character_parser.parse_text_segments(text, engine_type="fish_audio_s2")
        characters = []
        for segment in segments:
            character = segment.character or "narrator"
            if character not in characters:
                characters.append(character)
        reference_characters = list(reference_order or characters)
        narrator_exists_globally = "narrator" in reference_characters
        if narrator_exists_globally:
            reference_characters = ["narrator"] + [character for character in reference_characters if character != "narrator"]
        global_speaker_number = {
            character: index + 1
            for index, character in enumerate(reference_characters)
        }
        discovered = get_character_mapping(characters, engine_type="audio_only")
        narrator_value = voice_mapping.get("narrator", {})
        narrator = dict(narrator_value) if isinstance(narrator_value, dict) else {"audio": narrator_value}
        configured_refs = list(self.config.get("speaker_references") or [])
        speaker_map = {character: index for index, character in enumerate(characters)}
        voice_refs = []
        narrator_has_reference = bool(narrator)
        for character in characters:
            effective_speaker_number = global_speaker_number.get(character, speaker_map[character] + 1)
            if narrator_has_reference and narrator_exists_globally and character == "narrator":
                voice_refs.append(dict(narrator))
                continue
            if narrator_has_reference and not narrator_exists_globally and effective_speaker_number == 1:
                voice_refs.append(dict(narrator))
                continue
            configured_index = effective_speaker_number - 2
            if 0 <= configured_index < len(configured_refs):
                voice_refs.append(configured_refs[configured_index])
                continue
            voice_refs.append(self._voice_for(character, voice_mapping, discovered, narrator))
        voice_ref_by_character = dict(zip(characters, voice_refs))
        custom_switching = self.config.get(
            "multi_speaker_mode", "Native Multi-Speaker"
        ) == "Custom Character Switching"
        records = []
        pending_turns = []
        pending_logs = []
        pending_config = None
        group_index = 0
        speaker_labels = {index: character for character, index in speaker_map.items()}
        connected_speakers = []
        if narrator_has_reference:
            connected_speakers.append("Speaker 1 (Narrator)")
        connected_speakers.extend(
            f"Speaker {idx + 2}"
            for idx, ref in enumerate(configured_refs)
            if ref is not None
        )
        logged_priority_overrides = set()

        def generation_key(config):
            return tuple(config.get(name) for name in (
                "temperature", "top_p", "repetition_penalty", "native_chunk_length",
                "max_new_tokens", "normalize", "cache_reference",
            ))

        def flush_turns():
            nonlocal pending_turns, pending_logs, group_index
            if not pending_turns:
                return
            self.adapter.update_config(pending_config)
            if show_text_logging:
                print(f"🎭 Fish Audio S2 native dialogue: Processing {len(pending_turns)} segment(s)")
                if connected_speakers:
                    print(f"🎤 Fish speaker inputs connected: {connected_speakers}")
                mapping_display = ", ".join(
                    f"{character}->Speaker {global_speaker_number.get(character, speaker_idx + 1)}"
                    for character, speaker_idx in sorted(speaker_map.items(), key=lambda item: item[1])
                )
                if mapping_display:
                    print(f"🎭 Fish character mapping: {mapping_display}")
                print("============================================================")
                for log_label, turn_text in pending_logs:
                    print(f"[{log_label}] {turn_text}")
                print("============================================================")
            audio = self.adapter.generate_dialogue(
                turns=pending_turns, voice_refs=voice_refs,
                seed=int(pending_config.get("seed", seed)) + group_index,
                enable_audio_cache=enable_audio_cache,
            )
            combined_text = "\n".join(turn_text for _, turn_text in pending_turns)
            records.append({
                "waveform": audio.cpu(), "sample_rate": self.SAMPLE_RATE,
                "text": combined_text, "edit_tags": [],
            })
            pending_turns = []
            pending_logs = []
            group_index += 1

        for segment in segments:
            if model_management.interrupt_processing:
                raise InterruptedError("Fish Audio S2 text generation interrupted")
            segment_text = (segment.text or "").strip()
            if not segment_text:
                continue
            params = segment.parameters or {}
            filtered = ParameterValidator.filter_parameters_for_engine(params, "fish_audio_s2")
            config = apply_segment_parameters(self.config, params, "fish_audio_s2") if filtered else dict(self.config)
            character = segment.character or "narrator"
            original_character = getattr(segment, "original_character", None) or character
            speaker_idx = speaker_map[character]
            effective_speaker_number = global_speaker_number.get(character, speaker_idx + 1)
            configured_index = effective_speaker_number - 2
            override_voice_ref = None
            if (
                not custom_switching
                and narrator_has_reference
                and (
                    (narrator_exists_globally and character == "narrator")
                    or (not narrator_exists_globally and effective_speaker_number == 1)
                )
            ):
                discovered_ref = discovered.get(character, (None, None))[0]
                override_voice_ref = narrator
                if discovered_ref and show_text_logging:
                    override_key = (speaker_idx, character)
                    if override_key not in logged_priority_overrides:
                        print(f"⚠️ Fish priority: Speaker 1 (Narrator) input overrides ['{original_character}'] alias")
                        logged_priority_overrides.add(override_key)
            elif not custom_switching and 0 <= configured_index < len(configured_refs):
                configured_ref = configured_refs[configured_index]
                override_voice_ref = configured_ref
                discovered_ref = discovered.get(character, (None, None))[0]
                if configured_ref is not None and discovered_ref:
                    override_key = (speaker_idx, character)
                    if override_key not in logged_priority_overrides and show_text_logging:
                        print(f"⚠️ Fish priority: Speaker {effective_speaker_number} input overrides ['{original_character}'] alias")
                        logged_priority_overrides.add(override_key)
            if pending_config is not None and generation_key(config) != generation_key(pending_config):
                flush_turns()
            pending_config = config

            pause_parts, _ = PauseTagProcessor.parse_pause_tags(segment_text)
            if not pause_parts:
                pause_parts = [("text", segment_text)]
            for part_type, content in pause_parts:
                if model_management.interrupt_processing:
                    raise InterruptedError("Fish Audio S2 text generation interrupted")
                if part_type == "pause":
                    flush_turns()
                    records.append({
                        "waveform": PauseTagProcessor.create_silence_segment(content, self.SAMPLE_RATE),
                        "sample_rate": self.SAMPLE_RATE, "text": f"[pause:{content}s]", "edit_tags": [],
                    })
                    continue
                turn_text = str(content).strip()
                if not turn_text:
                    continue
                turn_text, language_instruction = self._apply_effective_language_instruction(
                    turn_text, segment, override_voice_ref if not custom_switching else None
                )
                if show_text_logging and self.config.get("language_prompting", "Auto Inline Tag") == "Auto Inline Tag":
                    if language_instruction:
                        print(f"  🌍 Fish Audio S2 prompting language via <{language_instruction}>")
                if show_text_logging and custom_switching:
                    print(f"🎭 Fish Audio S2 Pro - {character}: {turn_text}")
                if custom_switching:
                    flush_turns()
                    self.adapter.update_config(config)
                    audio = self.adapter.generate_dialogue(
                        turns=[(0, turn_text)],
                        voice_refs=[voice_ref_by_character[character]],
                        seed=int(config.get("seed", seed)) + group_index,
                        enable_audio_cache=enable_audio_cache,
                    )
                    records.append({
                        "waveform": audio.cpu(), "sample_rate": self.SAMPLE_RATE,
                        "text": turn_text, "edit_tags": [],
                    })
                    group_index += 1
                else:
                    pending_turns.append((speaker_map[character], turn_text))
                    if override_voice_ref is not None:
                        log_label = self._speaker_display_name(effective_speaker_number, override_voice_ref)
                    else:
                        log_label = character
                    pending_logs.append((log_label, turn_text))
        flush_turns()
        return records

    def combine_audio_segments(self, segments, method="auto", silence_ms=100,
                               original_text="", return_info=False):
        if not segments:
            empty = torch.zeros(0)
            return (empty, {}) if return_info else empty
        return ChunkTimingHelper.combine_audio_with_timing(
            audio_segments=[item["waveform"] for item in segments],
            combination_method=method, silence_ms=silence_ms, crossfade_duration=0.1,
            sample_rate=self.SAMPLE_RATE, text_length=len(original_text), original_text=original_text,
            text_chunks=[item.get("text", "") for item in segments],
        ) if return_info else ChunkTimingHelper.combine_audio_with_timing(
            audio_segments=[item["waveform"] for item in segments],
            combination_method=method, silence_ms=silence_ms, crossfade_duration=0.1,
            sample_rate=self.SAMPLE_RATE, text_length=len(original_text), original_text=original_text,
            text_chunks=[item.get("text", "") for item in segments],
        )[0]
