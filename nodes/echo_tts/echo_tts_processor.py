"""
Echo-TTS Internal Processor - Handles TTS generation orchestration
Called by unified TTS nodes when using Echo-TTS engine
"""

import os
import re
import sys
from typing import Dict, Any, List, Tuple, Union

import torch

# Add project root to path
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.text.pause_processor import PauseTagProcessor
from utils.text.segment_parameters import apply_segment_parameters
from utils.text.step_audio_editx_special_tags import get_edit_tags_for_segment
from utils.audio.edit_post_processor import process_segments as apply_edit_post_processing
from utils.voice.discovery import get_available_characters, voice_discovery, get_character_mapping
from utils.text.character_parser import character_parser
from utils.audio.chunk_timing import ChunkTimingHelper


class EchoTTSProcessor:
    """
    Internal processor for Echo-TTS generation.
    Handles character switching, pause tags, segment parameters, and orchestration.
    """

    SAMPLE_RATE = 44100

    def __init__(self, adapter, engine_config: Dict[str, Any]):
        """
        Initialize Echo-TTS processor.

        Args:
            adapter: EchoTTSEngineAdapter instance used for segment generation
            engine_config: Engine configuration from Echo-TTS Engine node
        """
        self.adapter = adapter
        self.config = engine_config.copy() if engine_config else {}

    def update_config(self, new_config: Dict[str, Any]):
        """Update processor configuration with new parameters."""
        self.config = new_config.copy() if new_config else {}

    @staticmethod
    def _strip_s1_tag(text_value: str) -> str:
        return re.sub(r'\[s1\]\s*', '', text_value or "", flags=re.IGNORECASE)

    def _setup_character_parser(self, text: str):
        """Configure character parser for this text run (preserve unknown tags)."""
        character_tags = re.findall(r'\[([^\]]+)\]', text or "")
        characters_from_tags = []
        for tag in character_tags:
            if not tag.startswith('pause:'):
                character_name = tag.split('|')[0].strip()
                characters_from_tags.append(character_name)

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

    def process_text(
        self,
        text: str,
        narrator_audio: Any,
        narrator_reference_text: str,
        seed: int,
        enable_chunking: bool = True,
        max_chars_per_chunk: int = 400,
        chunk_combination_method: str = "auto",
        silence_between_chunks_ms: int = 100,
        enable_audio_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process text and generate Echo-TTS segment records.

        Returns:
            List of segment dicts:
            - waveform: torch.Tensor
            - sample_rate: int
            - text: str
            - edit_tags: list
        """
        self._setup_character_parser(text)

        base_config = self.config.copy()
        parsed_text = self._strip_s1_tag(text)
        segment_objects = character_parser.parse_text_segments(parsed_text)
        if not segment_objects:
            segment_objects = character_parser.parse_text_segments("narrator " + parsed_text)

        characters = list(set([seg.character for seg in segment_objects]))
        character_mapping = get_character_mapping(characters, engine_type="audio_only")

        segment_records: List[Dict[str, Any]] = []
        active_segments = []
        block_texts = []
        for seg in segment_objects:
            segment_text = (seg.text or "").strip()
            if not segment_text:
                continue
            clean_segment_text, _ = get_edit_tags_for_segment(segment_text)
            block_texts.append(max(1, len((clean_segment_text or "").strip())))
            active_segments.append(seg)

        self.adapter.start_job(total_blocks=len(active_segments), block_texts=block_texts)
        try:
            for block_idx, seg in enumerate(active_segments):
                self.adapter.set_current_block(block_idx)

                segment_text = (seg.text or "").strip()
                segment_params = seg.parameters if seg.parameters else {}
                current_config = base_config
                current_seed = seed
                if segment_params:
                    current_config = apply_segment_parameters(base_config, segment_params, "echo_tts")
                    if 'seed' in current_config:
                        current_seed = int(current_config.get('seed', seed))
                    print(f"📊 Echo-TTS segment: Character '{seg.character}' with parameters {segment_params}")

                self.adapter.update_config(current_config)

                speaker_audio = narrator_audio
                current_ref_text = narrator_reference_text or ""
                if seg.character == "narrator":
                    if narrator_audio is not None:
                        speaker_audio = narrator_audio
                        current_ref_text = narrator_reference_text or current_ref_text
                elif seg.character and seg.character in character_mapping:
                    char_audio, char_text = character_mapping[seg.character]
                    if char_audio:
                        speaker_audio = char_audio
                        current_ref_text = char_text or current_ref_text
                    else:
                        print(f"⚠️ Echo-TTS: No voice file found for '{seg.character}', using narrator voice")

                clean_text, edit_tags = get_edit_tags_for_segment(segment_text)
                clean_text, pause_segments = PauseTagProcessor.preprocess_text_with_pause_tags(
                    clean_text,
                    enable_pause_tags=True
                )
                if pause_segments and any(
                    current_config.get(k) is not None for k in (
                        "speaker_kv_scale", "speaker_kv_max_layers", "speaker_kv_min_t"
                    )
                ):
                    raise ValueError(
                        "Echo-TTS: Pause tags are not compatible with force_speaker_kv settings. "
                        "Disable force_speaker_kv (speaker_kv_*) or remove pause tags."
                    )

                pause_mode = pause_segments is not None
                seed_offset = 0

                def _tts_generate_func(text_content: str) -> torch.Tensor:
                    nonlocal seed_offset
                    segment_seed = current_seed + seed_offset
                    seed_offset += 1

                    # Ensure full config (including speaker_kv_*) is applied for each text segment.
                    self.adapter.update_config(current_config)
                    audio = self.adapter.process_text(
                        text=text_content,
                        speaker_audio=speaker_audio,
                        reference_text=current_ref_text,
                        seed=segment_seed,
                        enable_chunking=False if pause_mode else enable_chunking,
                        max_chars_per_chunk=max_chars_per_chunk,
                        chunk_combination_method=chunk_combination_method,
                        silence_between_chunks_ms=0 if pause_mode else silence_between_chunks_ms,
                        enable_audio_cache=enable_audio_cache,
                        return_info=False
                    )
                    if isinstance(audio, tuple):
                        audio = audio[0]
                    if not isinstance(audio, torch.Tensor):
                        audio = torch.tensor(audio, dtype=torch.float32)
                    if audio.dim() > 1:
                        audio = audio.squeeze()
                    return audio

                if pause_segments:
                    audio_segment = PauseTagProcessor.generate_audio_with_pauses(
                        pause_segments,
                        _tts_generate_func,
                        sample_rate=self.SAMPLE_RATE
                    )
                    if audio_segment.dim() > 1:
                        audio_segment = audio_segment.squeeze()
                else:
                    audio_segment = _tts_generate_func(clean_text)

                segment_records.append({
                    "waveform": audio_segment,
                    "sample_rate": self.SAMPLE_RATE,
                    "text": clean_text,
                    "edit_tags": edit_tags
                })
                self.adapter.complete_block()

            if segment_records and any(seg["edit_tags"] for seg in segment_records):
                segment_records = apply_edit_post_processing(segment_records, engine_config=base_config)
        finally:
            self.adapter.end_job()

        return segment_records

    def combine_audio_segments(
        self,
        segments: List[Dict[str, Any]],
        method: str = "auto",
        silence_ms: int = 100,
        original_text: str = "",
        return_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Combine Echo-TTS segment records into final audio.

        Returns:
            (combined_audio, chunk_info) when return_info=True
            (combined_audio, {}) otherwise
        """
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
            text_chunks=text_chunks
        )

        if return_info:
            return combined_audio, chunk_info
        return combined_audio
