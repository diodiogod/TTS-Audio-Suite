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
from utils.models.language_mapper import resolve_language_alias
from utils.text.character_parser import character_parser
from utils.text.chunking import ImprovedChatterBoxChunker
from utils.text.pause_processor import PauseTagProcessor
from utils.text.segment_parameters import apply_segment_parameters
from utils.voice.discovery import (
    get_available_characters,
    get_character_mapping,
    voice_discovery,
)


class MossTTSProcessor:
    """Processor for MOSS-TTS text generation through unified nodes."""

    SAMPLE_RATE = 24000
    OFFICIAL_INLINE_FIELDS = ("instruction", "quality", "sound_event", "ambient_sound")

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
                print(f"🎭 MOSS-TTS: Using character voice for '{character}'")
            elif narrator_ref:
                voice_mapping[character] = narrator_ref
                if character != "narrator":
                    print(f"🔄 MOSS-TTS: Using narrator voice fallback for '{character}'")
            else:
                voice_mapping[character] = None
                if character != "narrator":
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
    ) -> List[Dict[str, Any]]:
        params = self.config.copy()
        params["seed"] = seed
        params["language"] = self._language_name_to_code(params.get("language", "Auto"))

        self._prepare_character_parser(text, params["language"])
        segment_objects = character_parser.parse_text_segments(text)
        if params.get("multi_speaker_mode") == "Native Multi-Speaker Dialogue":
            return self._process_native_multispeaker_or_fallback(
                segment_objects,
                voice_mapping,
                params,
                enable_chunking,
                max_chars_per_chunk,
            )
        return self._process_character_switching(
            segment_objects,
            voice_mapping,
            params,
            enable_chunking,
            max_chars_per_chunk,
        )

    def _process_native_multispeaker_or_fallback(
        self,
        segment_objects,
        voice_mapping: Dict[str, Any],
        params: Dict[str, Any],
        enable_chunking: bool,
        max_chars: int,
    ) -> List[Dict[str, Any]]:
        full_text = " ".join(seg.text for seg in segment_objects)
        unique_characters = []
        for segment in segment_objects:
            if segment.character not in unique_characters:
                unique_characters.append(segment.character)

        fallback_reasons = []
        if len(unique_characters) > 5:
            fallback_reasons.append("more than 5 speakers")
        if PauseTagProcessor.has_pause_tags(full_text):
            fallback_reasons.append("pause tags")
        if any(getattr(segment, "parameters", None) for segment in segment_objects):
            fallback_reasons.append("per-segment parameter changes")

        if fallback_reasons:
            print(f"⚡ MOSS-TTS fallback: Native Multi-Speaker Dialogue → Custom Character Switching ({', '.join(fallback_reasons)})")
            return self._process_character_switching(
                segment_objects,
                voice_mapping,
                params,
                enable_chunking,
                max_chars,
            )

        if self.config.get("model_variant") != "MOSS-TTSD-v1.0":
            print("🔄 MOSS-TTS: Loading MOSS-TTSD-v1.0 for Native Multi-Speaker Dialogue")
            self.config["model_variant"] = "MOSS-TTSD-v1.0"
            params["model_variant"] = "MOSS-TTSD-v1.0"
            self._load_model_from_config(self.config)

        return self._process_native_multispeaker(segment_objects, voice_mapping, params)

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

    def _process_native_multispeaker(
        self,
        segment_objects,
        voice_mapping: Dict[str, Any],
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        main_narrator_voice = voice_mapping.get("narrator") if voice_mapping else None
        if main_narrator_voice is None:
            available_voices = [voice for voice in (voice_mapping or {}).values() if voice is not None]
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
        formatted_lines = []

        print(f"🎭 MOSS-TTSD native dialogue: Processing {len(segment_objects)} segment(s)")
        connected = [f"S{idx}" for idx, voice in speaker_inputs.items() if voice is not None]
        print(f"🎤 MOSS-TTSD speaker inputs connected: {connected or ['none']}")

        for segment in segment_objects:
            character = segment.character
            text = segment.text.strip()
            if not text:
                continue

            manual_speaker = self._manual_speaker_number(character)
            if manual_speaker is not None:
                speaker_idx = manual_speaker - 1
                character_map.setdefault(character, speaker_idx)
            elif character not in character_map:
                speaker_idx = len(character_map)
                if speaker_idx >= 5:
                    print(f"⚠️ MOSS-TTSD: More than 5 speakers found; '{character}' will use S5")
                    speaker_idx = 4
                character_map[character] = speaker_idx
            else:
                speaker_idx = character_map[character]

            speaker_num = speaker_idx + 1
            connected_voice = speaker_inputs.get(speaker_num)
            character_voice = (voice_mapping or {}).get(character)

            if connected_voice is not None and character_voice is not None:
                print(f"⚠️ MOSS-TTSD priority: S{speaker_num} input overrides ['{character}'] alias")
                speaker_voices[speaker_idx] = connected_voice
            elif connected_voice is not None:
                speaker_voices[speaker_idx] = connected_voice
            elif character_voice is not None:
                speaker_voices[speaker_idx] = character_voice

            formatted_lines.append(f"[S{speaker_num}] {text}")

        if not formatted_lines:
            return []

        max_speaker_idx = max(character_map.values()) if character_map else 0
        speaker_count = min(max_speaker_idx + 1, 5)
        speaker_voices = speaker_voices[:speaker_count]
        dialogue_text = "\n".join(formatted_lines)

        mapping_display = ", ".join(
            f"{character}->S{speaker_idx + 1}"
            for character, speaker_idx in sorted(character_map.items(), key=lambda item: item[1])
        )
        print(f"🎭 MOSS-TTSD character mapping: {mapping_display}")
        for field_name in self.OFFICIAL_INLINE_FIELDS:
            field_value = params.get(field_name)
            if field_value:
                print(f"  🔹 Official {field_name}: {field_value}")
        print("🎭 MOSS-TTSD formatted dialogue:")
        print("=" * 60)
        print(dialogue_text)
        print("=" * 60)

        audio = self.adapter.generate_native_dialogue(
            dialogue_text=dialogue_text,
            speaker_voices=speaker_voices,
            params=params,
        )
        return [audio]

    def _process_character_switching(
        self,
        segment_objects,
        voice_mapping: Dict[str, Any],
        params: Dict[str, Any],
        enable_chunking: bool,
        max_chars: int,
    ) -> List[Dict[str, Any]]:
        audio_segments: List[Dict[str, Any]] = []
        print(f"🔄 MOSS-TTS: Processing {len(segment_objects)} character segment(s)")

        for seg_idx, segment in enumerate(segment_objects):
            if model_management.interrupt_processing:
                raise InterruptedError(f"MOSS-TTS segment {seg_idx + 1}/{len(segment_objects)} interrupted by user")

            print(f"\n🎤 Segment {seg_idx + 1}/{len(segment_objects)}: Character '{segment.character}'")

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

        if enable_chunking and len(combined_text) > max_chars:
            chunks = self.chunker.split_into_chunks(combined_text, max_chars)
            print(f"📝 MOSS-TTS: Chunking '{character}' into {len(chunks)} chunk(s) (language={language})")
        else:
            chunks = [combined_text]

        for chunk_idx, chunk in enumerate(chunks, start=1):
            if model_management.interrupt_processing:
                raise InterruptedError("MOSS-TTS generation interrupted by user")

            chunk_note = f" chunk {chunk_idx}/{len(chunks)}" if len(chunks) > 1 else ""
            print(f"🎭 MOSS-TTS - Generating for '{character}' (language={language}){chunk_note}:")
            for field_name in self.OFFICIAL_INLINE_FIELDS:
                field_value = generation_params.get(field_name)
                if field_value:
                    print(f"  🔹 Official {field_name}: {field_value}")
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
