"""
Qwen3-TTS Internal Processor - Handles TTS generation orchestration
Called by unified TTS nodes when using Qwen3-TTS engine
"""

import torch
from typing import Dict, Any, Optional, List, Tuple
import os
import sys
import comfy.model_management as model_management

# Add project root to path
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.text.chunking import ImprovedChatterBoxChunker
from utils.audio.processing import AudioProcessingUtils
from utils.text.character_parser import character_parser
from utils.text.segment_parameters import apply_segment_parameters
from utils.text.pause_processor import PauseTagProcessor
from engines.adapters.qwen3_tts_adapter import Qwen3TTSEngineAdapter
from utils.models.language_mapper import resolve_language_alias
from utils.text.step_audio_editx_special_tags import get_edit_tags_for_segment
from utils.audio.edit_post_processor import process_segments as apply_edit_post_processing


class Qwen3TTSProcessor:
    """
    Internal processor for Qwen3-TTS generation.
    Handles chunking, character processing, intelligent model selection, and generation orchestration.

    Supports all 3 model types:
    - CustomVoice: 9 preset speakers with optional instruction
    - VoiceDesign: Text-to-voice design (handled by Voice Designer node)
    - Base: Zero-shot voice cloning from reference audio
    """

    # Supported languages for Qwen3-TTS (official API)
    SUPPORTED_LANGUAGES = {
        "auto", "chinese", "english", "japanese", "korean",
        "german", "french", "russian", "portuguese", "spanish", "italian"
    }

    # CustomVoice preset speakers (for character mapping)
    PRESET_SPEAKERS = [
        "Vivian", "Serena", "Uncle_Fu", "Dylan",
        "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"
    ]

    def __init__(self, node_instance, engine_config: Dict[str, Any]):
        """
        Initialize Qwen3-TTS processor.

        Args:
            node_instance: Parent node instance
            engine_config: Engine configuration from Qwen3-TTS Engine node
        """
        self.node = node_instance
        self.config = engine_config
        self.adapter = Qwen3TTSEngineAdapter(node_instance)
        self.chunker = ImprovedChatterBoxChunker()

        # Session-based character-to-speaker mapping for CustomVoice
        self._character_speaker_mapping = {}

        # Extract config parameters
        model_size = engine_config.get('model_size', '1.7B')
        device = engine_config.get('device', 'auto')
        dtype = engine_config.get('dtype', 'auto')
        attn_implementation = engine_config.get('attn_implementation', 'auto')
        voice_preset = engine_config.get('voice_preset', 'None (Zero-shot / Custom)')

        # Create context for model type determination
        context = {
            "voice_preset": voice_preset,
            "model_size": model_size
        }

        # Determine model type based on voice_preset
        if voice_preset == "None (Zero-shot / Custom)":
            model_type = "Base"
        else:
            model_type = "CustomVoice"

        # Build model path based on determined type
        model_path = f'Qwen3-TTS-12Hz-{model_size}-{model_type}'

        # Load model via adapter (with intelligent model selection)
        self.adapter.load_base_model(
            model_path=model_path,
            device=device,
            dtype=dtype,
            model_size=model_size,
            attn_implementation=attn_implementation,
            context=context
        )

    def update_config(self, new_config: Dict[str, Any]):
        """Update processor configuration with new parameters."""
        self.config.update(new_config)

    def _language_name_to_code(self, language_input: str) -> str:
        """Convert language name or code to Qwen3-TTS language parameter."""
        # First resolve using the centralized language mapper
        resolved_code = resolve_language_alias(language_input)

        # Map to Qwen3-TTS supported language names (exact API strings)
        code_mapping = {
            # Chinese variations
            "zh": "Chinese",
            "zh-cn": "Chinese",
            "zh-tw": "Chinese",
            # English variations
            "en": "English",
            # Japanese variations
            "ja": "Japanese",
            "jp": "Japanese",
            # Korean variations
            "ko": "Korean",
            "kr": "Korean",
            # German variations
            "de": "German",
            # French variations
            "fr": "French",
            # Russian variations
            "ru": "Russian",
            # Portuguese variations
            "pt": "Portuguese",
            "pt-br": "Portuguese",
            "pt-pt": "Portuguese",
            "po": "Portuguese",  # Common abbreviation
            # Spanish variations
            "es": "Spanish",
            # Italian variations
            "it": "Italian",
            # Auto
            "auto": "Auto",
        }

        mapped_language = code_mapping.get(resolved_code.lower())

        if mapped_language:
            return mapped_language

        # If not in mapping, check if it's already a valid Qwen3-TTS language name
        if resolved_code.lower() in self.SUPPORTED_LANGUAGES:
            return resolved_code.capitalize()

        # Fallback to Auto if unsupported
        print(f"⚠️ Language '{language_input}' not recognized by Qwen3-TTS. Falling back to Auto.")
        return "Auto"

    def _map_character_to_speaker(self, character_name: str, narrator_preset: str) -> str:
        """
        Map character name to CustomVoice preset speaker for character switching.

        Priority:
        1. If character name matches preset speaker name → use it directly
        2. If already mapped in session → use cached mapping
        3. Otherwise → deterministically assign unused speaker based on character name hash

        Args:
            character_name: Character name from text tags
            narrator_preset: Default narrator preset speaker

        Returns:
            Preset speaker name to use for this character
        """
        import hashlib

        # Normalize for matching
        char_lower = character_name.lower()

        # Priority 1: Check if character name IS a preset speaker
        for speaker in self.PRESET_SPEAKERS:
            if speaker.lower() == char_lower:
                print(f"🚨🎭🚨 Character '{character_name}' matches preset speaker - using CustomVoice {speaker}")
                return speaker

        # Priority 2: Check session cache
        if character_name in self._character_speaker_mapping:
            cached_speaker = self._character_speaker_mapping[character_name]
            print(f"🚨🎭🚨 Character '{character_name}' using cached preset '{cached_speaker}'")
            return cached_speaker

        # Priority 3: Deterministically assign unused speaker
        # Use MD5 hash for consistent assignment across sessions (cache-friendly)
        char_hash = int(hashlib.md5(character_name.encode()).hexdigest(), 16)

        # Get available speakers (exclude narrator and already-used)
        used_speakers = set(self._character_speaker_mapping.values())
        used_speakers.add(narrator_preset)
        available = [s for s in self.PRESET_SPEAKERS if s not in used_speakers]

        if not available:
            # All speakers used, wrap around (exclude narrator if possible)
            available = [s for s in self.PRESET_SPEAKERS if s != narrator_preset]
            if not available:
                available = self.PRESET_SPEAKERS

        assigned_speaker = available[char_hash % len(available)]
        self._character_speaker_mapping[character_name] = assigned_speaker

        print(f"🚨🎭🚨 Character '{character_name}' auto-mapped to preset '{assigned_speaker}' (CustomVoice doesn't support voice cloning)")
        return assigned_speaker

    def process_text(self,
                    text: str,
                    voice_mapping: Dict[str, Any],
                    seed: int,
                    enable_chunking: bool = True,
                    max_chars_per_chunk: int = 400) -> List[Dict]:
        """
        Process text and generate audio.

        Args:
            text: Input text with potential character tags
            voice_mapping: Mapping of character names to voice references
            seed: Random seed for generation
            enable_chunking: Whether to chunk long text
            max_chars_per_chunk: Maximum characters per chunk

        Returns:
            List of audio segments
        """

        # Add seed to params
        params = self.config.copy()
        params['seed'] = seed

        # Map language to Qwen3-TTS format
        if 'language' in params:
            original_lang = params['language']
            params['language'] = self._language_name_to_code(params['language'])
            if original_lang != params['language']:
                print(f"🌐 Language mapped: '{original_lang}' → '{params['language']}'")

        # Parse character segments with parameter support
        from utils.voice.discovery import get_available_characters, voice_discovery

        # Get available characters and aliases
        available_chars = get_available_characters()
        character_aliases = voice_discovery.get_character_aliases()

        # Build complete available set
        all_available = set()
        if available_chars:
            all_available.update(available_chars)
        for alias, target in character_aliases.items():
            all_available.add(alias.lower())
            all_available.add(target.lower())

        # Also add characters from text (extract from tags)
        import re
        character_tags = re.findall(r'\[([^\]]+)\]', text)
        for tag in character_tags:
            if not tag.startswith('pause:'):
                character_name = tag.split('|')[0].strip().lower()
                all_available.add(character_name)

        # Add "narrator"
        all_available.add("narrator")

        character_parser.set_available_characters(list(all_available))

        # Set global default language for character parser BEFORE parsing segments
        # This ensures segments without explicit language tags use the engine's configured language
        # Convert from Qwen3-TTS format (Spanish, English) to standard codes (es, en)
        global_language = params.get('language', 'Auto')

        # Map Qwen3 language names back to standard codes for character parser
        qwen3_to_code = {
            'Chinese': 'zh', 'English': 'en', 'Japanese': 'ja', 'Korean': 'ko',
            'German': 'de', 'French': 'fr', 'Russian': 'ru', 'Portuguese': 'pt',
            'Spanish': 'es', 'Italian': 'it', 'Auto': 'auto'
        }
        standard_lang_code = qwen3_to_code.get(global_language, global_language.lower())

        # Directly set the default language on the language resolver
        character_parser.language_resolver.default_language = standard_lang_code
        character_parser.default_language = standard_lang_code

        # Set character-specific language defaults from alias system
        char_lang_defaults = voice_discovery.get_character_language_defaults()
        for char, lang in char_lang_defaults.items():
            character_parser.set_character_language_default(char, lang)

        character_parser.reset_session_cache()
        segment_objects = character_parser.parse_text_segments(text)

        # Info if using CustomVoice preset with character switching
        voice_preset = params.get('voice_preset', 'None (Zero-shot / Custom)')
        if voice_preset != 'None (Zero-shot / Custom)' and len(segment_objects) > 1:
            # Check if there are different characters (not just parameter changes)
            unique_chars = set(seg.character for seg in segment_objects)
            if len(unique_chars) > 1:
                print(f"ℹ️ Qwen3-TTS: Character switching with CustomVoice preset '{voice_preset}'")
                print(f"   CustomVoice model doesn't support voice cloning - characters will be auto-mapped to preset speakers")

        # Process using character switching mode
        return self._process_character_switching(
            segment_objects, voice_mapping, params,
            enable_chunking, max_chars_per_chunk
        )

    def _process_character_switching(self,
                                    segment_objects,  # List of CharacterSegment objects
                                    voice_mapping: Dict[str, Any],
                                    params: Dict,
                                    enable_chunking: bool,
                                    max_chars: int) -> List[Dict]:
        """
        Process character segments individually with Qwen3-TTS generation.
        Each segment is processed separately to respect character switching and parameter changes.

        Args:
            segment_objects: List of CharacterSegment objects (with parameters)
            voice_mapping: Voice mapping
            params: Base generation parameters
            enable_chunking: Whether to chunk
            max_chars: Max chars per chunk

        Returns:
            List of audio segments
        """
        audio_segments = []

        print(f"🔄 Qwen3-TTS: Processing {len(segment_objects)} character segments")

        # Calculate total chunks across all segments for time estimation
        chunk_texts = []
        for seg in segment_objects:
            seg_text = seg.text.strip()
            if enable_chunking and len(seg_text) > max_chars:
                # Estimate chunk count and sizes
                chunks = self.chunker.split_into_chunks(seg_text, max_chars)
                for chunk in chunks:
                    chunk_texts.append(len(chunk))
            else:
                chunk_texts.append(len(seg_text))

        # Only start job if not already tracking (SRT processor manages job at higher level)
        self._srt_mode = self.adapter.job_tracker is not None
        if not self._srt_mode:
            self.adapter.start_job(total_blocks=len(chunk_texts), block_texts=chunk_texts)
        self._chunk_idx = 0  # Track chunk index across all segments

        for seg_idx, segment in enumerate(segment_objects):
            # Check for interruption before processing each segment
            if model_management.interrupt_processing:
                raise InterruptedError(f"Qwen3-TTS segment {seg_idx + 1}/{len(segment_objects)} ({segment.character}) interrupted by user")

            print(f"\n🎤 Segment {seg_idx + 1}/{len(segment_objects)}: Character '{segment.character}'")

            # Apply per-segment parameters
            segment_params = params.copy()

            # Apply segment language if character has language default
            if hasattr(segment, 'language') and segment.language:
                segment_params['language'] = self._language_name_to_code(segment.language)
                if segment.language != params.get('language', 'Auto'):
                    print(f"  🌍 Language switched to: {segment_params['language']}")

            if segment.parameters:
                seg_param_updates = apply_segment_parameters(segment_params, segment.parameters, 'qwen3_tts')
                segment_params.update(seg_param_updates)
                print(f"  📊 Applying parameters: {segment.parameters}")

            # Process the segment text
            self._process_character_block(segment.character, segment.text.strip(), voice_mapping,
                                        segment_params, enable_chunking, max_chars, audio_segments)

            print()  # New line after progress bar

        # End job tracking (only if we started it)
        if not self._srt_mode:
            self.adapter.end_job()

        # Apply inline edit tags post-processing if any segments have edit_tags
        if any(seg.get('edit_tags') for seg in audio_segments):
            print(f"🎨 Applying Step Audio EditX inline edit tags post-processing...")
            audio_segments = apply_edit_post_processing(
                audio_segments,
                self.config,
                pre_loaded_engine=None  # Will load Step EditX engine if needed
            )

        return audio_segments

    def _process_character_block(self, character: str, combined_text: str,
                               voice_mapping: Dict[str, Any], params: Dict,
                               enable_chunking: bool, max_chars: int,
                               audio_segments: List[Dict]) -> None:
        """
        Process a combined character block (potentially with chunking).

        Handles pause tags correctly:
        - Split by pause tags FIRST (each pause-segment gets separate generation)

        Args:
            character: Character name
            combined_text: Combined text for this character
            voice_mapping: Voice mapping
            params: Generation parameters
            enable_chunking: Whether chunking is enabled
            max_chars: Max characters per chunk
            audio_segments: List to append results to
        """
        # If using CustomVoice preset, map character to preset speaker
        voice_preset = params.get('voice_preset', 'None (Zero-shot / Custom)')
        if voice_preset != 'None (Zero-shot / Custom)':
            # Special case: narrator uses the selected preset directly
            if character.lower() == 'narrator':
                # Narrator uses the user's chosen preset - no mapping needed
                pass
            else:
                # Map non-narrator characters to preset speakers
                mapped_speaker = self._map_character_to_speaker(character, voice_preset)
                # Override params with mapped speaker for this character
                params = params.copy()
                params['voice_preset'] = mapped_speaker

        # Check if text has pause tags - if so, split by pause FIRST
        from utils.text.pause_processor import PauseTagProcessor

        if PauseTagProcessor.has_pause_tags(combined_text):
            # Split by pause tags to get individual pause-delimited segments
            pause_segments, _ = PauseTagProcessor.parse_pause_tags(combined_text)

            for seg_type, content in pause_segments:
                if seg_type == 'text':
                    # Process this pause-delimited text segment
                    self._process_single_text_segment(
                        character, content, voice_mapping, params, audio_segments
                    )
                elif seg_type == 'pause':
                    # Add silence segment
                    silence = PauseTagProcessor.create_silence_segment(content, 24000)
                    audio_dict = {
                        'waveform': silence,
                        'sample_rate': 24000,
                        'character': character,
                        'text': f'[pause:{content}s]'
                    }
                    audio_segments.append(audio_dict)
            return

        # No pause tags - process normally
        # Apply chunking if enabled and text is long
        if enable_chunking and len(combined_text) > max_chars:
            voice_ref = voice_mapping.get(character)

            # Show voice info
            voice_note = ""
            if not isinstance(voice_ref, dict):
                voice_note = " [⚠️ No voice reference - will use default]"

            # Get language for logging
            segment_lang = params.get('language', 'Auto')

            # Extract inline edit tags BEFORE chunking
            combined_text_clean, combined_text_edit_tags = get_edit_tags_for_segment(combined_text)

            chunks = self.chunker.split_into_chunks(combined_text_clean, max_chars)
            print(f"📝 Chunking {character}'s combined text into {len(chunks)} chunks (Language: {segment_lang}){voice_note}")

            for chunk_idx, chunk in enumerate(chunks):
                # Check for interruption during chunk processing
                if model_management.interrupt_processing:
                    raise InterruptedError(f"Qwen3-TTS chunk {chunk_idx + 1}/{len(chunks)} interrupted by user")

                # Set current chunk for time tracking (skip in SRT mode - managed at subtitle level)
                if not self._srt_mode:
                    self.adapter.set_current_block(self._chunk_idx)

                # Process pause tags
                audio_tensor = self.adapter.generate_with_pause_tags(
                    chunk, voice_ref, params, True, character
                )

                # Mark chunk as completed for time tracking (skip in SRT mode)
                if not self._srt_mode:
                    self.adapter.complete_block()
                self._chunk_idx += 1

                # Ensure 2D shape [channels, samples]
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                elif audio_tensor.dim() == 3:
                    audio_tensor = audio_tensor.squeeze(0)

                audio_dict = {
                    'waveform': audio_tensor,
                    'sample_rate': 24000,
                    'character': character,
                    'text': chunk,
                    'edit_tags': combined_text_edit_tags  # Pass edit tags for post-processing
                }
                audio_segments.append(audio_dict)
        else:
            # Generate without chunking - the entire combined block at once
            voice_ref = voice_mapping.get(character)

            # Show voice info
            voice_note = ""
            if not isinstance(voice_ref, dict):
                voice_note = " [⚠️ No voice reference - will use default]"

            # Get language for logging
            segment_lang = params.get('language', 'Auto')

            # Extract inline edit tags BEFORE generation
            combined_text_clean, combined_text_edit_tags = get_edit_tags_for_segment(combined_text)

            print(f"🎭 Qwen3-TTS - Generating for '{character}' (Language: {segment_lang}){voice_note}:")
            print("="*60)
            print(combined_text_clean)
            print("="*60)

            # Set current segment for time tracking (skip in SRT mode - managed at subtitle level)
            if not self._srt_mode:
                self.adapter.set_current_block(self._chunk_idx)

            # Process pause tags with clean text
            audio_tensor = self.adapter.generate_with_pause_tags(
                combined_text_clean, voice_ref, params, True, character
            )

            # Mark segment as completed for time tracking (skip in SRT mode)
            if not self._srt_mode:
                self.adapter.complete_block()
            self._chunk_idx += 1

            # Ensure 2D shape [channels, samples]
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(0)

            audio_dict = {
                'waveform': audio_tensor,
                'sample_rate': 24000,
                'character': character,
                'text': combined_text_clean,
                'edit_tags': combined_text_edit_tags  # Pass edit tags for post-processing
            }
            audio_segments.append(audio_dict)

    def _process_single_text_segment(self,
                                    character: str,
                                    text: str,
                                    voice_mapping: Dict[str, Any],
                                    params: Dict,
                                    audio_segments: List[Dict]) -> None:
        """
        Process a single text segment (no pause tags, no chunking).

        NOTE: This is called from pause-split path, so job tracking is NOT updated here.
        The parent already set up job tracking at the block level.

        Args:
            character: Character name
            text: Text segment
            voice_mapping: Voice mapping
            params: Generation parameters
            audio_segments: List to append result to
        """
        voice_ref = voice_mapping.get(character)

        # Generate audio
        # NOTE: Don't update job tracker - parent handles it
        audio_tensor = self.adapter._generate_direct(text, voice_ref, params, character)

        # Ensure correct shape [channels, samples]
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)

        audio_dict = {
            'waveform': audio_tensor,
            'sample_rate': 24000,
            'character': character,
            'text': text
        }
        audio_segments.append(audio_dict)

    def combine_audio_segments(self,
                              segments: List[Dict],
                              method: str = "auto",
                              silence_ms: int = 100,
                              text_length: int = 0,
                              return_info: bool = False):
        """
        Combine multiple audio segments with optional timing info.

        Args:
            segments: List of audio dicts with 'waveform' and 'text' keys
            method: Combination method
            silence_ms: Silence between segments
            text_length: Total text length
            return_info: If True, return (audio, chunk_info) tuple

        Returns:
            Combined audio tensor, or (tensor, chunk_info) if return_info=True
        """
        if not segments:
            empty = torch.zeros(1, 1, 0)
            if return_info:
                return empty, {"method_used": "none", "total_chunks": 0, "chunk_timings": []}
            return empty

        sample_rate = 24000

        # Extract waveforms and text
        waveforms = []
        texts = []
        for seg in segments:
            wave = seg['waveform']
            if wave.dim() == 3:
                wave = wave.squeeze(0)  # Remove batch dim
            if wave.dim() == 1:
                wave = wave.unsqueeze(0)  # Add channel dim
            waveforms.append(wave)
            texts.append(seg.get('text', ''))

        # Single segment
        if len(waveforms) == 1:
            combined = waveforms[0]
            # Ensure 2D shape [channels, samples]
            if combined.dim() == 1:
                combined = combined.unsqueeze(0)
            elif combined.dim() == 3:
                combined = combined.squeeze(0)

            if return_info:
                chunk_info = {
                    "method_used": "none",
                    "total_chunks": 1,
                    "chunk_timings": [{"start": 0.0, "end": combined.shape[-1] / sample_rate, "text": texts[0]}]
                }
                return combined, chunk_info
            return combined

        # Use unified ChunkCombiner for consistency
        from utils.audio.chunk_combiner import ChunkCombiner
        result = ChunkCombiner.combine_chunks(
            audio_segments=waveforms,
            method=method,
            silence_ms=silence_ms,
            crossfade_duration=0.1,
            sample_rate=sample_rate,
            text_length=text_length,
            original_text=" ".join(texts),
            text_chunks=texts,
            return_info=return_info
        )

        return result

    def cleanup(self):
        """Clean up resources"""
        if self.adapter:
            self.adapter.cleanup()
