"""
MOSS-TTS internal processor.

Handles unified text generation orchestration for the official MOSS-TTS models.
"""

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

from engines.adapters.moss_tts_adapter import MossTTSEngineAdapter
from utils.audio.processing import AudioProcessingUtils
from utils.audio.edit_post_processor import process_segments as apply_edit_post_processing
from utils.models.language_mapper import resolve_language_alias
from utils.text.character_parser import character_parser
from utils.text.chunking import ImprovedChatterBoxChunker
from utils.text.pause_processor import PauseTagProcessor
from utils.text.segment_parameters import apply_segment_parameters
from utils.text.step_audio_editx_special_tags import get_edit_tags_for_segment
from utils.voice.discovery import (
    get_available_characters,
    get_character_mapping,
    voice_discovery,
)


class MossTTSProcessor:
    """Processor for MOSS-TTS text generation through unified nodes."""

    SAMPLE_RATE = 24000
    OFFICIAL_INLINE_FIELDS = ("instruction", "quality", "sound_event", "ambient_sound")
    EXPERIMENTAL_TTS_FIELDS = {"quality", "sound_event", "ambient_sound"}
    NATIVE_MAPPING_META_KEY = "__moss_native_character_map__"

    LANGUAGE_NAME_TO_CODE = {
        "auto": "auto",
        "chinese": "zh",
        "mandarin": "zh",
        "zh": "zh",
        "english": "en",
        "en": "en",
        "german": "de",
        "de": "de",
        "spanish": "es",
        "es": "es",
        "french": "fr",
        "fr": "fr",
        "japanese": "ja",
        "ja": "ja",
        "italian": "it",
        "it": "it",
        "hungarian": "hu",
        "hu": "hu",
        "korean": "ko",
        "ko": "ko",
        "russian": "ru",
        "ru": "ru",
        "persian": "fa",
        "farsi": "fa",
        "fa": "fa",
        "arabic": "ar",
        "ar": "ar",
        "polish": "pl",
        "pl": "pl",
        "portuguese": "pt",
        "pt": "pt",
        "czech": "cs",
        "cs": "cs",
        "danish": "da",
        "da": "da",
        "swedish": "sv",
        "sv": "sv",
        "greek": "el",
        "el": "el",
        "turkish": "tr",
        "tr": "tr",
    }

    def __init__(self, node_instance, engine_config: Dict[str, Any]):
        self.node = node_instance
        self.config = engine_config.copy()
        self.adapter = MossTTSEngineAdapter(node_instance)
        self.chunker = ImprovedChatterBoxChunker()
        self._load_model_from_config(self.config)

    def _load_model_from_config(self, config: Dict[str, Any]) -> None:
        self.adapter.load_model(
            model_variant=config.get("model_variant", "MOSS-TTS-Local-Transformer"),
            device=config.get("device", "auto"),
            dtype=config.get("dtype", "auto"),
            attn_implementation=config.get("attn_implementation", "auto"),
            codec_model=config.get("codec_model", "MOSS-Audio-Tokenizer"),
        )

    def update_config(self, new_config: Dict[str, Any]) -> None:
        old_identity = self._model_identity(self.config)
        self.config.update(new_config)
        new_identity = self._model_identity(self.config)
        if new_identity != old_identity:
            self._load_model_from_config(self.config)

    @staticmethod
    def _model_identity(config: Dict[str, Any]):
        return (
            config.get("model_variant", "MOSS-TTS-Local-Transformer"),
            config.get("device", "auto"),
            config.get("dtype", "auto"),
            config.get("attn_implementation", "auto"),
            config.get("codec_model", "MOSS-Audio-Tokenizer"),
        )

    def _language_name_to_code(self, language_input: Optional[str]) -> str:
        if language_input is None:
            return "auto"
        value = str(language_input).strip()
        if not value:
            return "auto"

        lowered = value.lower()
        if lowered in self.LANGUAGE_NAME_TO_CODE:
            return self.LANGUAGE_NAME_TO_CODE[lowered]

        resolved = resolve_language_alias(value).lower()
        # MOSS currently expects generic Portuguese code `pt`.
        if resolved in {"pt-br", "pt-pt"}:
            resolved = "pt"
        return self.LANGUAGE_NAME_TO_CODE.get(resolved, resolved)

    def _prepare_character_parser(self, text: str, language_code: str) -> None:
        available_chars = get_available_characters()
        character_aliases = voice_discovery.get_character_aliases()

        all_available = set()
        if available_chars:
            all_available.update(available_chars)
        for alias, target in character_aliases.items():
            all_available.add(alias.lower())
            all_available.add(target.lower())

        import re

        for tag in re.findall(r"\[([^\]]+)\]", text):
            if not tag.startswith("pause:"):
                all_available.add(tag.split("|")[0].strip().lower())
        all_available.add("narrator")

        character_parser.set_available_characters(list(all_available))
        character_parser.language_resolver.default_language = language_code
        character_parser.default_language = language_code

        for char, lang in voice_discovery.get_character_language_defaults().items():
            character_parser.set_character_language_default(char, lang)

        character_parser.reset_session_cache()

    def build_voice_mapping(
        self,
        text: str,
        narrator_audio: Any = None,
        reference_text: str = "",
        narrator_audio_path: Optional[str] = None,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Build character-to-reference mapping from text tags and voice folders."""
        if self.config.get("multi_speaker_mode") == "Native Multi-Speaker Dialogue":
            text = self._normalize_native_dialogue_input(text)
        language_code = self._language_name_to_code(self.config.get("language", "Auto"))
        self._prepare_character_parser(text, language_code)
        segments = character_parser.parse_text_segments(text)
        characters = sorted({seg.character for seg in segments} or {"narrator"})

        character_mapping = get_character_mapping(characters, engine_type="moss_tts")
        narrator_ref = self._make_voice_ref(narrator_audio, reference_text, narrator_audio_path)

        voice_mapping: Dict[str, Optional[Dict[str, Any]]] = {}
        for character in characters:
            if character == "narrator" and narrator_ref:
                voice_mapping[character] = narrator_ref
                continue

            audio_path, char_text = character_mapping.get(character, (None, None))
            if audio_path:
                voice_mapping[character] = {
                    "audio_path": audio_path,
                    "reference_text": char_text or "",
                }
                if self.config.get("multi_speaker_mode") != "Native Multi-Speaker Dialogue" or self._manual_speaker_number(character) is None:
                    print(f"🎭 MOSS-TTS: Using character voice for '{character}'")
            elif narrator_ref:
                voice_mapping[character] = narrator_ref
                if character != "narrator" and not (
                    self.config.get("multi_speaker_mode") == "Native Multi-Speaker Dialogue"
                    and self._manual_speaker_number(character) is not None
                ):
                    print(f"🔄 MOSS-TTS: Using narrator voice fallback for '{character}'")
            else:
                voice_mapping[character] = None
                if character != "narrator" and not (
                    self.config.get("multi_speaker_mode") == "Native Multi-Speaker Dialogue"
                    and self._manual_speaker_number(character) is not None
                ):
                    print(f"ℹ️ MOSS-TTS: No reference for '{character}', using direct TTS")

        return voice_mapping

    @staticmethod
    def _make_voice_ref(audio: Any, reference_text: str = "", audio_path: Optional[str] = None):
        if audio_path:
            return {"audio_path": audio_path, "reference_text": reference_text or ""}
        if audio is None:
            return None
        if isinstance(audio, dict) and "waveform" in audio:
            return {"audio": audio, "reference_text": reference_text or ""}
        if torch.is_tensor(audio):
            return {
                "waveform": audio,
                "sample_rate": MossTTSProcessor.SAMPLE_RATE,
                "reference_text": reference_text or "",
            }
        if isinstance(audio, str):
            return {"audio_path": audio, "reference_text": reference_text or ""}
        return {"audio": audio, "reference_text": reference_text or ""}

    def process_text(
        self,
        text: str,
        voice_mapping: Dict[str, Any],
        seed: int,
        enable_chunking: bool = True,
        max_chars_per_chunk: int = 400,
        apply_edit_postprocessing: bool = True,
    ) -> List[Dict[str, Any]]:
        params = self.config.copy()
        params["seed"] = seed
        params["language"] = self._language_name_to_code(params.get("language", "Auto"))
        native_character_map = (voice_mapping or {}).get(self.NATIVE_MAPPING_META_KEY)

        chunk_minutes = int(self.config.get("chunk_minutes", 0) or 0)
        chunk_chars = int(self.config.get("chunk_chars", 0) or 0)
        if chunk_minutes > 0 and chunk_chars > 0:
            enable_chunking = True
            max_chars_per_chunk = chunk_chars
        elif chunk_minutes == 0:
            # VibeVoice-style override behavior: explicit 0 disables chunking here.
            enable_chunking = False
            max_chars_per_chunk = 999999

        if params.get("multi_speaker_mode") == "Custom Character Switching":
            if self._contains_manual_speaker_format(text):
                print("🔄 MOSS-TTS: Auto-switching to Native Multi-Speaker Dialogue (detected manual 'Speaker N:' format)")
                params["multi_speaker_mode"] = "Native Multi-Speaker Dialogue"
            elif re.search(r"\[(?:s\s*\d+|speaker\s*\d+|\d+)\]", str(text or ""), re.IGNORECASE):
                print("⚠️ MOSS-TTS: Numeric/native speaker tags detected in Custom Character Switching mode")
                print("⚠️ These tags are intended for native dialogue mapping. Switch to Native Multi-Speaker Dialogue for [1]/[S1]/Speaker 1 usage.")

        if params.get("multi_speaker_mode") == "Native Multi-Speaker Dialogue":
            normalized_text = self._normalize_native_dialogue_input(text)
            if normalized_text != text:
                print("🔄 MOSS-TTSD: Normalized native dialogue input to canonical [Sx]/[Character] form")
            text = normalized_text

        self._prepare_character_parser(text, params["language"])
        segment_objects = character_parser.parse_text_segments(text)
        if params.get("multi_speaker_mode") == "Native Multi-Speaker Dialogue":
            return self._process_native_multispeaker_or_fallback(
                segment_objects,
                voice_mapping,
                params,
                enable_chunking,
                max_chars_per_chunk,
                native_character_map=native_character_map,
            )
        audio_segments = self._process_character_switching(
            segment_objects,
            voice_mapping,
            params,
            enable_chunking,
            max_chars_per_chunk,
        )
        if apply_edit_postprocessing and any(seg.get("edit_tags") for seg in audio_segments):
            print("🎨 Applying Step Audio EditX inline edit tags post-processing...")
            audio_segments = apply_edit_post_processing(audio_segments, self.config)
        return audio_segments

    def _get_native_fallback_reasons(self, segment_objects) -> List[str]:
        full_text = " ".join(seg.text for seg in segment_objects)
        unique_characters = []
        for segment in segment_objects:
            character_key = self._native_character_key(segment.character)
            if character_key not in unique_characters:
                unique_characters.append(character_key)

        fallback_reasons = []
        if len(unique_characters) > 5:
            fallback_reasons.append("more than 5 speakers")
        if PauseTagProcessor.has_pause_tags(full_text):
            fallback_reasons.append("pause tags")
        _, inline_edit_tags = get_edit_tags_for_segment(full_text)
        if inline_edit_tags:
            fallback_reasons.append("inline edit tags")
        if any(getattr(segment, "parameters", None) for segment in segment_objects):
            fallback_reasons.append("per-segment parameter changes")
        return fallback_reasons

    def get_native_srt_fallback_reasons(self, texts: List[str]) -> List[str]:
        fallback_reasons = []
        nonempty_texts = [str(text or "").strip() for text in texts if str(text or "").strip()]
        if not nonempty_texts:
            return fallback_reasons

        native_character_map = self.build_native_character_map_from_texts(nonempty_texts)
        if len(native_character_map) > 5:
            fallback_reasons.append("more than 5 speakers")

        language = self._language_name_to_code(self.config.get("language", "Auto"))
        for text in nonempty_texts:
            normalized = self._normalize_native_dialogue_input(text)
            self._prepare_character_parser(normalized, language)
            segment_objects = character_parser.parse_text_segments(normalized)
            for reason in self._get_native_fallback_reasons(segment_objects):
                if reason not in fallback_reasons:
                    fallback_reasons.append(reason)
        return fallback_reasons

    def _process_native_multispeaker_or_fallback(
        self,
        segment_objects,
        voice_mapping: Dict[str, Any],
        params: Dict[str, Any],
        enable_chunking: bool,
        max_chars: int,
        native_character_map: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        fallback_reasons = self._get_native_fallback_reasons(segment_objects)

        if fallback_reasons:
            reason_text = ", ".join(fallback_reasons)
            raise RuntimeError(
                "MOSS-TTSD Native Multi-Speaker Dialogue does not support this input "
                f"({reason_text}). Switch to 'Custom Character Switching' and choose a standard MOSS model."
            )

        target_variant = str(params.get("model_variant") or "MOSS-TTSD-v1.0")
        current_variant = str(self.config.get("model_variant") or "")
        if current_variant != target_variant:
            print(f"🔄 MOSS-TTS: Loading {target_variant} for Native Multi-Speaker Dialogue")
            self.config["model_variant"] = target_variant
            params["model_variant"] = target_variant
            self._load_model_from_config(self.config)

        return self._process_native_multispeaker(
            segment_objects,
            voice_mapping,
            params,
            enable_chunking=enable_chunking,
            max_chars=max_chars,
            native_character_map=native_character_map,
        )

    @staticmethod
    def _manual_speaker_number(character: str) -> Optional[int]:
        value = str(character or "").strip().lower()
        if value.isdigit():
            number = int(value)
            return number if 1 <= number <= 5 else None
        match = re.fullmatch(r"s(?:peaker)?\s*(\d+)", value)
        if match:
            number = int(match.group(1))
            return number if 1 <= number <= 5 else None
        return None

    @classmethod
    def _contains_manual_speaker_format(cls, text: str) -> bool:
        return bool(re.search(r"(?im)^\s*speaker\s*(\d+)\s*:\s*", str(text or "")))

    @classmethod
    def _normalize_native_dialogue_input(cls, text: str) -> str:
        lines = []
        for raw_line in str(text or "").splitlines():
            match = re.match(r"^\s*speaker\s*(\d+)\s*:\s*(.*)$", raw_line, re.IGNORECASE)
            if match:
                number = int(match.group(1))
                if 1 <= number <= 5:
                    lines.append(f"[S{number}] {match.group(2).strip()}")
                    continue
            lines.append(raw_line)
        return "\n".join(lines)

    @classmethod
    def _native_character_key(cls, character: str) -> str:
        speaker_num = cls._manual_speaker_number(character)
        if speaker_num is not None:
            return f"s{speaker_num}"
        return str(character or "").strip()

    @staticmethod
    def _resolve_native_map_key_with_aliases(
        character_key: str,
        native_character_map: Optional[Dict[str, int]],
    ) -> str:
        """
        When character parser resolves [Alice] -> alias target (e.g. female_01),
        map it back to the canonical key used in native_character_map if possible.
        """
        if not native_character_map:
            return character_key
        if character_key in native_character_map:
            return character_key

        # Case-insensitive direct map key match first.
        lower_to_canonical = {str(k).strip().lower(): k for k in native_character_map.keys()}
        lowered_key = str(character_key).strip().lower()
        if lowered_key in lower_to_canonical:
            return lower_to_canonical[lowered_key]

        try:
            alias_map = voice_discovery.get_character_aliases() or {}
        except Exception:
            alias_map = {}

        candidates = []
        for source, target in alias_map.items():
            if str(target).strip().lower() == lowered_key:
                candidates.append(str(source).strip())

        for source_key in sorted(candidates, key=lambda s: (s.lower() != "narrator", s.lower())):
            source_lower = source_key.lower()
            if source_lower in lower_to_canonical:
                return lower_to_canonical[source_lower]

        return character_key

    @staticmethod
    def _is_voice_reference(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (str, torch.Tensor)):
            return True
        if isinstance(value, dict):
            return any(key in value for key in ("audio", "audio_path", "waveform"))
        return False

    @classmethod
    def build_native_character_map_from_texts(cls, texts: List[str]) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        used_slots = set()
        has_narrator = False
        discovered_keys: List[str] = []

        # Build discovery order from line-based parsing so implicit narrator lines are included.
        for text in texts:
            normalized = cls._normalize_native_dialogue_input(text)
            for raw_line in normalized.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                tag_match = re.match(r"^\[([^\]]+)\]\s*(.*)$", line)
                if tag_match:
                    character = tag_match.group(1).split("|")[0].strip()
                    if not character or character.lower().startswith("pause:"):
                        continue
                    key = cls._native_character_key(character)
                    if key not in discovered_keys:
                        discovered_keys.append(key)
                else:
                    has_narrator = True

        # Prefer narrator on S1 for native dialogue consistency when implicit narrator exists.
        if has_narrator:
            mapping["narrator"] = 1
            used_slots.add(1)

        for key in discovered_keys:
            if key in mapping:
                continue
            manual_speaker = cls._manual_speaker_number(key)
            if manual_speaker is not None:
                mapping[key] = manual_speaker
                used_slots.add(manual_speaker)
                continue
            candidates = range(2, 6) if has_narrator else range(1, 6)
            for candidate in candidates:
                if candidate not in used_slots:
                    mapping[key] = candidate
                    used_slots.add(candidate)
                    break
            if key not in mapping:
                mapping[key] = 5
                used_slots.add(5)

        return mapping

    def _process_native_multispeaker(
        self,
        segment_objects,
        voice_mapping: Dict[str, Any],
        params: Dict[str, Any],
        enable_chunking: bool = True,
        max_chars: int = 400,
        native_character_map: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        main_narrator_voice = voice_mapping.get("narrator") if voice_mapping else None
        if main_narrator_voice is None:
            available_voices = [voice for voice in (voice_mapping or {}).values() if self._is_voice_reference(voice)]
            main_narrator_voice = available_voices[0] if available_voices else None

        speaker_inputs = {
            1: main_narrator_voice,
            2: params.get("speaker2_voice"),
            3: params.get("speaker3_voice"),
            4: params.get("speaker4_voice"),
            5: params.get("speaker5_voice"),
        }
        speaker_voices = [speaker_inputs.get(idx) for idx in range(1, 6)]
        character_map: Dict[str, int] = {}
        used_slots = set()
        logged_priority_overrides = set()
        formatted_lines = []

        print(f"🎭 MOSS-TTSD native dialogue: Processing {len(segment_objects)} segment(s)")
        connected = [f"S{idx}" for idx, voice in speaker_inputs.items() if voice is not None]
        print(f"🎤 MOSS-TTSD speaker inputs connected: {connected or ['none']}")

        if native_character_map:
            for key, speaker_num in native_character_map.items():
                if 1 <= int(speaker_num) <= 5:
                    character_map[key] = int(speaker_num) - 1
                    used_slots.add(int(speaker_num))

        normalized_segments = []
        for segment in segment_objects:
            character = segment.character
            text = segment.text.strip()
            if not text:
                continue

            character_key = self._native_character_key(character)
            character_key = self._resolve_native_map_key_with_aliases(character_key, native_character_map)
            manual_speaker = self._manual_speaker_number(character)
            if manual_speaker is not None:
                speaker_idx = manual_speaker - 1
                character_map.setdefault(character_key, speaker_idx)
                used_slots.add(manual_speaker)
            elif character_key not in character_map:
                next_free = next((candidate for candidate in range(1, 6) if candidate not in used_slots), None)
                if next_free is None:
                    print(f"⚠️ MOSS-TTSD: More than 5 speakers found; '{character}' will use S5")
                    speaker_idx = 4
                else:
                    speaker_idx = next_free - 1
                    used_slots.add(next_free)
                character_map[character_key] = speaker_idx
            else:
                speaker_idx = character_map[character_key]

            speaker_num = speaker_idx + 1
            connected_voice = speaker_inputs.get(speaker_num)
            character_voice = (voice_mapping or {}).get(character)
            if not self._is_voice_reference(character_voice):
                character_voice = None

            if connected_voice is not None and character_voice is not None:
                override_key = (speaker_num, self._native_character_key(character))
                if override_key not in logged_priority_overrides:
                    print(f"⚠️ MOSS-TTSD priority: S{speaker_num} input overrides ['{character}'] alias")
                    logged_priority_overrides.add(override_key)
                speaker_voices[speaker_idx] = connected_voice
            elif connected_voice is not None:
                speaker_voices[speaker_idx] = connected_voice
            elif character_voice is not None:
                speaker_voices[speaker_idx] = character_voice

            normalized_segments.append((speaker_num, text))
            formatted_lines.append(f"[S{speaker_num}] {text}")

        if not formatted_lines:
            return []

        max_speaker_idx = max(character_map.values()) if character_map else 0
        speaker_count = min(max_speaker_idx + 1, 5)
        speaker_voices = speaker_voices[:speaker_count]

        mapping_display = ", ".join(
            f"{character}->S{speaker_idx + 1}"
            for character, speaker_idx in sorted(character_map.items(), key=lambda item: item[1])
        )
        print(f"🎭 MOSS-TTSD character mapping: {mapping_display}")
        for field_name in self.OFFICIAL_INLINE_FIELDS:
            field_value = params.get(field_name)
            if field_value:
                print(f"  🔹 {field_name}: {self._format_prompt_field_log(field_name, field_value, params)}")
        if enable_chunking and len("\n".join(formatted_lines)) > max_chars:
            segment_groups = self._split_native_segments_by_chars(normalized_segments, max_chars)
            print(f"✂️ MOSS-TTSD: Split native dialogue into {len(segment_groups)} chunk(s) (max {max_chars} chars)")
        else:
            segment_groups = [normalized_segments]

        outputs = []
        for chunk_idx, chunk_group in enumerate(segment_groups, start=1):
            dialogue_text = "\n".join(f"[S{speaker_num}] {text}" for speaker_num, text in chunk_group)
            if len(segment_groups) > 1:
                print(f"🎭 MOSS-TTSD formatted dialogue chunk {chunk_idx}/{len(segment_groups)}:")
            else:
                print("🎭 MOSS-TTSD formatted dialogue:")
            print("=" * 60)
            print(dialogue_text)
            print("=" * 60)

            audio = self.adapter.generate_native_dialogue(
                dialogue_text=dialogue_text,
                speaker_voices=speaker_voices,
                params=params,
            )
            outputs.append(audio)
        return outputs

    @staticmethod
    def _split_native_segments_by_chars(
        segments: List[tuple[int, str]],
        max_chars: int,
    ) -> List[List[tuple[int, str]]]:
        if not segments:
            return []
        if max_chars <= 0:
            return [segments]

        groups: List[List[tuple[int, str]]] = []
        current_group: List[tuple[int, str]] = []
        current_chars = 0

        for speaker_num, text in segments:
            line_len = len(text) + 8  # account for [Sx] + separators
            if current_group and current_chars + line_len > max_chars:
                groups.append(current_group)
                current_group = []
                current_chars = 0
            current_group.append((speaker_num, text))
            current_chars += line_len

        if current_group:
            groups.append(current_group)
        return groups

    def _process_character_switching(
        self,
        segment_objects,
        voice_mapping: Dict[str, Any],
        params: Dict[str, Any],
        enable_chunking: bool,
        max_chars: int,
    ) -> List[Dict[str, Any]]:
        audio_segments: List[Dict[str, Any]] = []
        grouped_segments = self._group_consecutive_character_segments(segment_objects)
        print(
            f"🔄 MOSS-TTS: Processing {len(grouped_segments)} character segment(s)"
            + (f" (grouped from {len(segment_objects)})" if len(grouped_segments) != len(segment_objects) else "")
        )

        for seg_idx, segment in enumerate(grouped_segments):
            if model_management.interrupt_processing:
                raise InterruptedError(f"MOSS-TTS segment {seg_idx + 1}/{len(grouped_segments)} interrupted by user")

            print(f"\n🎤 Segment {seg_idx + 1}/{len(grouped_segments)}: Character '{segment.character}'")

            segment_params = params.copy()
            if getattr(segment, "language", None):
                segment_params["language"] = self._language_name_to_code(segment.language)
                if segment.language != params.get("language", "auto"):
                    print(f"  🌍 Language switched to: {segment_params['language']}")

            if segment.parameters:
                updates = apply_segment_parameters(segment_params, segment.parameters, "moss_tts")
                segment_params.update(updates)
                print(f"  📊 MOSS-TTS segment parameters: {segment.parameters}")

            self._process_character_block(
                character=segment.character,
                combined_text=segment.text.strip(),
                voice_mapping=voice_mapping,
                params=segment_params,
                enable_chunking=enable_chunking,
                max_chars=max_chars,
                audio_segments=audio_segments,
            )

        return audio_segments

    @staticmethod
    def _group_consecutive_character_segments(segment_objects):
        if not segment_objects:
            return []

        grouped = []
        current = None

        def can_merge(prev, nxt) -> bool:
            if prev.character != nxt.character:
                return False
            if getattr(prev, "language", None) != getattr(nxt, "language", None):
                return False
            if getattr(prev, "parameters", None) != getattr(nxt, "parameters", None):
                return False
            # Keep pause/edit-tag text boundaries as hard split points.
            if PauseTagProcessor.has_pause_tags(prev.text) or PauseTagProcessor.has_pause_tags(nxt.text):
                return False
            if get_edit_tags_for_segment(prev.text)[1] or get_edit_tags_for_segment(nxt.text)[1]:
                return False
            return True

        for segment in segment_objects:
            if current is None:
                current = segment
                continue
            if can_merge(current, segment):
                current.text = f"{current.text}\n{segment.text}"
                continue
            grouped.append(current)
            current = segment

        if current is not None:
            grouped.append(current)
        return grouped

    @staticmethod
    def _is_soundeffect_model(params: Dict[str, Any]) -> bool:
        model_variant = str(params.get("model_variant", "") or "")
        return "soundeffect" in model_variant.lower()

    @classmethod
    def _format_prompt_field_log(cls, field_name: str, field_value: Any, params: Dict[str, Any]) -> str:
        value = str(field_value)
        if field_name in cls.EXPERIMENTAL_TTS_FIELDS and not cls._is_soundeffect_model(params):
            return f"{value}  ⚠️ On base MOSS-TTS this may have little or no audible effect"
        return value

    def _process_character_block(
        self,
        character: str,
        combined_text: str,
        voice_mapping: Dict[str, Any],
        params: Dict[str, Any],
        enable_chunking: bool,
        max_chars: int,
        audio_segments: List[Dict[str, Any]],
    ) -> None:
        if not combined_text:
            return

        if PauseTagProcessor.has_pause_tags(combined_text):
            pause_segments, _ = PauseTagProcessor.parse_pause_tags(combined_text)
            for segment_type, content in pause_segments:
                if segment_type == "text":
                    self._process_character_block(
                        character,
                        content,
                        voice_mapping,
                        params,
                        enable_chunking,
                        max_chars,
                        audio_segments,
                    )
                elif segment_type == "pause":
                    audio_segments.append({
                        "waveform": PauseTagProcessor.create_silence_segment(content, self.SAMPLE_RATE),
                        "sample_rate": self.SAMPLE_RATE,
                        "character": character,
                        "text": f"[pause:{content}s]",
                    })
            return

        voice_ref = voice_mapping.get(character)
        language = params.get("language", "auto")
        generation_params = params.copy()
        ambient_sound = generation_params.get("ambient_sound")
        duration_tokens = generation_params.get("duration_tokens")
        max_new_tokens = generation_params.get("max_new_tokens", 4096)

        if ambient_sound and not duration_tokens and int(max_new_tokens) >= 1024:
            print(
                "⚠️ MOSS-TTS: ambient_sound is set without duration_tokens, and max_new_tokens is high. "
                "MOSS may keep generating ambience until it reaches the token cap. "
                "Set duration_tokens or lower max_new_tokens for short clips."
            )

        clean_text, edit_tags = get_edit_tags_for_segment(combined_text)
        if not clean_text:
            return

        if enable_chunking and len(clean_text) > max_chars:
            chunks = self.chunker.split_into_chunks(clean_text, max_chars)
            print(f"📝 MOSS-TTS: Chunking '{character}' into {len(chunks)} chunk(s) (language={language})")
        else:
            chunks = [clean_text]

        for chunk_idx, chunk in enumerate(chunks, start=1):
            if model_management.interrupt_processing:
                raise InterruptedError("MOSS-TTS generation interrupted by user")

            chunk_note = f" chunk {chunk_idx}/{len(chunks)}" if len(chunks) > 1 else ""
            print(f"🎭 MOSS-TTS - Generating for '{character}' (language={language}){chunk_note}:")
            for field_name in self.OFFICIAL_INLINE_FIELDS:
                field_value = generation_params.get(field_name)
                if field_value:
                    print(
                        f"  🔹 {field_name}: "
                        f"{self._format_prompt_field_log(field_name, field_value, generation_params)}"
                    )
            print("=" * 60)
            print(chunk)
            print("=" * 60)

            audio_tensor = self.adapter.generate_with_pause_tags(chunk, voice_ref, generation_params, True, character)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(0)

            audio_segments.append({
                "waveform": audio_tensor,
                "sample_rate": self.SAMPLE_RATE,
                "character": character,
                "text": chunk,
                "original_text": combined_text,
                "edit_tags": edit_tags,
            })

    def combine_audio_segments(
        self,
        segments: List[Dict[str, Any]],
        method: str = "auto",
        silence_ms: int = 100,
        text_length: int = 0,
        return_info: bool = False,
    ):
        if not segments:
            empty = torch.zeros(1, 0)
            if return_info:
                return empty, {"method_used": "none", "total_chunks": 0, "chunk_timings": []}
            return empty

        waveforms = []
        texts = []
        for seg in segments:
            wave = seg["waveform"]
            if wave.dim() == 3:
                wave = wave.squeeze(0)
            if wave.dim() == 1:
                wave = wave.unsqueeze(0)
            waveforms.append(wave)
            texts.append(seg.get("text", ""))

        if len(waveforms) == 1:
            combined = waveforms[0]
            if return_info:
                return combined, {
                    "method_used": "none",
                    "total_chunks": 1,
                    "chunk_timings": [{
                        "start": 0.0,
                        "end": combined.shape[-1] / self.SAMPLE_RATE,
                        "text": texts[0],
                    }],
                }
            return combined

        from utils.audio.chunk_combiner import ChunkCombiner

        return ChunkCombiner.combine_chunks(
            audio_segments=waveforms,
            method=method,
            silence_ms=silence_ms,
            crossfade_duration=0.1,
            sample_rate=self.SAMPLE_RATE,
            text_length=text_length,
            original_text=" ".join(texts),
            text_chunks=texts,
            return_info=return_info,
        )

    def format_for_comfyui(self, audio_tensor: torch.Tensor):
        return AudioProcessingUtils.format_for_comfyui(audio_tensor, self.SAMPLE_RATE)

    def cleanup(self):
        if self.adapter:
            self.adapter.cleanup()
