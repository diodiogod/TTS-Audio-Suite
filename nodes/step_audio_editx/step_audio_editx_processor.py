"""
Step Audio EditX Internal Processor - Handles TTS generation orchestration
Called by unified TTS nodes when using Step Audio EditX engine
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
from utils.text.step_audio_editx_special_tags import get_edit_tags_for_segment
from utils.audio.edit_post_processor import process_segments as apply_edit_post_processing
from engines.adapters.step_audio_editx_adapter import StepAudioEditXEngineAdapter


class StepAudioEditXProcessor:
    """
    Internal processor for Step Audio EditX TTS generation.
    Handles chunking, character processing, and generation orchestration.

    Note: This processor handles CLONE mode only. For audio editing (emotion/style/speed),
    use the dedicated Audio Editor node which works with any TTS engine output.
    """

    def __init__(self, node_instance, engine_config: Dict[str, Any]):
        """
        Initialize Step Audio EditX processor.

        Args:
            node_instance: Parent node instance
            engine_config: Engine configuration from Step Audio EditX Engine node
        """
        self.node = node_instance
        self.config = engine_config
        self.adapter = StepAudioEditXEngineAdapter(node_instance)
        self.chunker = ImprovedChatterBoxChunker()

        # Load model with configuration
        model_path = engine_config.get('model_path', 'Step-Audio-EditX')
        device = engine_config.get('device', 'auto')
        torch_dtype = engine_config.get('torch_dtype', 'auto')
        quantization = engine_config.get('quantization', None)

        # Load model via adapter
        self.adapter.load_base_model(model_path, device, torch_dtype, quantization)

    def update_config(self, new_config: Dict[str, Any]):
        """Update processor configuration with new parameters."""
        self.config.update(new_config)

    def process_text(self,
                    text: str,
                    voice_mapping: Dict[str, Any],
                    seed: int,
                    enable_chunking: bool = True,
                    max_chars_per_chunk: int = 400) -> List[Dict]:
        """
        Process text and generate audio using clone mode.

        Args:
            text: Input text with potential character tags
            voice_mapping: Mapping of character names to voice references
            seed: Random seed for generation (currently unused by Step Audio EditX)
            enable_chunking: Whether to chunk long text
            max_chars_per_chunk: Maximum characters per chunk

        Returns:
            List of audio segments
        """

        # Add seed to params - Step Audio EditX uses global torch seed (not passed to generate())
        params = self.config.copy()
        params['seed'] = seed

        # Parse character segments with parameter support
        # Configure character parser like IndexTTS does
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

        # Set language defaults
        char_lang_defaults = voice_discovery.get_character_language_defaults()
        for char, lang in char_lang_defaults.items():
            character_parser.set_character_language_default(char, lang)

        character_parser.reset_session_cache()
        segment_objects = character_parser.parse_text_segments(text)

        # Process using character switching mode (clone for each character)
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
        Process character segments individually with Step Audio EditX clone generation.
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

        print(f"🔄 Step Audio EditX: Processing {len(segment_objects)} character segments")

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
                raise InterruptedError(f"Step Audio EditX segment {seg_idx + 1}/{len(segment_objects)} ({segment.character}) interrupted by user")

            print(f"\n🎤 Segment {seg_idx + 1}/{len(segment_objects)}: Character '{segment.character}'")

            # Apply per-segment parameters
            segment_params = params.copy()
            if segment.parameters:
                seg_param_updates = apply_segment_parameters(segment_params, segment.parameters, 'step_audio_editx')
                segment_params.update(seg_param_updates)
                print(f"  📊 Applying parameters: {segment.parameters}")

            # Process the segment text
            self._process_character_block(segment.character, segment.text.strip(), voice_mapping,
                                        segment_params, enable_chunking, max_chars, audio_segments)

            print()  # New line after progress bar

        # End job tracking (only if we started it)
        if not self._srt_mode:
            self.adapter.end_job()

        # Apply edit post-processing to segments with edit tags
        # ONLY when NOT in SRT mode - SRT processor will batch process all segments at the end
        if not self._srt_mode:
            # This applies Step Audio EditX edits (emotion, style, speed, paralinguistic)
            # to segments that had inline edit tags like <Laughter:2>, <style:whisper>
            # Pass the already-loaded engine from the adapter to avoid reloading
            audio_segments = apply_edit_post_processing(
                audio_segments,
                self.config,
                pre_loaded_engine=self.adapter.engine
            )

        return audio_segments


    def _process_character_block(self, character: str, combined_text: str,
                               voice_mapping: Dict[str, Any], params: Dict,
                               enable_chunking: bool, max_chars: int,
                               audio_segments: List[Dict]) -> None:
        """
        Process a combined character block (potentially with chunking).

        IMPORTANT: Handles pause tags + edit tags correctly:
        - Split by pause tags FIRST (each pause-segment gets separate generation)
        - Extract edit tags from EACH pause segment
        - This allows: "Hello <Laughter> [pause:1] World <Sigh>"
          to apply Laughter to "Hello" and Sigh to "World"

        Args:
            character: Character name
            combined_text: Combined text for this character
            voice_mapping: Voice mapping
            params: Generation parameters
            enable_chunking: Whether chunking is enabled
            max_chars: Max characters per chunk
            audio_segments: List to append results to
        """
        # Check if text has pause tags - if so, split by pause FIRST
        from utils.text.pause_processor import PauseTagProcessor

        if PauseTagProcessor.has_pause_tags(combined_text):
            # Split by pause tags to get individual pause-delimited segments
            pause_segments, _ = PauseTagProcessor.parse_pause_tags(combined_text)

            for seg_type, content in pause_segments:
                if seg_type == 'text':
                    # Process this pause-delimited text segment
                    # (extract edit tags, generate, store with tags)
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
                        'text': f'[pause:{content}s]',
                        'edit_tags': None
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
            elif not voice_ref.get('prompt_audio_path') or not voice_ref.get('prompt_text'):
                voice_note = " [⚠️ Missing prompt audio/text - will use default]"

            chunks = self.chunker.split_into_chunks(combined_text, max_chars)
            print(f"📝 Chunking {character}'s combined text into {len(chunks)} chunks{voice_note}")

            for chunk_idx, chunk in enumerate(chunks):
                # Check for interruption during chunk processing
                if model_management.interrupt_processing:
                    raise InterruptedError(f"Step Audio EditX chunk {chunk_idx + 1}/{len(chunks)} interrupted by user")

                # Extract edit tags BEFORE TTS generation
                clean_chunk, edit_tags = get_edit_tags_for_segment(chunk)
                if edit_tags:
                    print(f"  🎨 Found {len(edit_tags)} edit tag(s) in chunk {chunk_idx + 1}")

                # Set current chunk for time tracking (skip in SRT mode - managed at subtitle level)
                if not self._srt_mode:
                    self.adapter.set_current_block(self._chunk_idx)

                # Process pause tags with CLEAN text (edit tags removed)
                audio_tensor = self.adapter.generate_with_pause_tags(
                    clean_chunk, voice_ref, params, True, character
                )

                # Mark chunk as completed for time tracking (skip in SRT mode)
                if not self._srt_mode:
                    self.adapter.complete_block()
                self._chunk_idx += 1

                # Convert tensor back to dict format
                # Ensure 2D shape [channels, samples]
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                elif audio_tensor.dim() == 3:
                    audio_tensor = audio_tensor.squeeze(0)

                audio_dict = {
                    'waveform': audio_tensor,
                    'sample_rate': 24000,  # Step Audio EditX native sample rate
                    'character': character,
                    'text': clean_chunk,  # Clean text for transcript
                    'original_text': chunk,  # Keep original for fallback parsing
                    'edit_tags': edit_tags  # Keep as-is (list or empty list, not None)
                }
                audio_segments.append(audio_dict)
        else:
            # Generate without chunking - the entire combined block at once
            voice_ref = voice_mapping.get(character)

            # Show voice info
            voice_note = ""
            if not isinstance(voice_ref, dict):
                voice_note = " [⚠️ No voice reference - will use default]"
            elif not voice_ref.get('prompt_audio_path') or not voice_ref.get('prompt_text'):
                voice_note = " [⚠️ Missing prompt audio/text - will use default]"

            # Extract edit tags BEFORE TTS generation
            clean_text, edit_tags = get_edit_tags_for_segment(combined_text)
            print(f"🔍 DEBUG: Extracted {len(edit_tags)} edit tags from combined_text (len={len(combined_text)})")
            if edit_tags:
                print(f"🎨 Found {len(edit_tags)} edit tag(s) for post-processing")

            print(f"🎭 Step Audio EditX - Generating for '{character}'{voice_note}:")
            print("="*60)
            print(clean_text)
            print("="*60)

            # Set current segment for time tracking (skip in SRT mode - managed at subtitle level)
            if not self._srt_mode:
                self.adapter.set_current_block(self._chunk_idx)

            # Process pause tags with CLEAN text (edit tags removed)
            audio_tensor = self.adapter.generate_with_pause_tags(
                clean_text, voice_ref, params, True, character
            )

            # Mark segment as completed for time tracking (skip in SRT mode)
            if not self._srt_mode:
                self.adapter.complete_block()
            self._chunk_idx += 1

            # Convert tensor back to dict format
            # Ensure 2D shape [channels, samples]
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(0)

            audio_dict = {
                'waveform': audio_tensor,
                'sample_rate': 24000,  # Step Audio EditX native sample rate
                'character': character,
                'text': clean_text,  # Clean text for transcript
                'original_text': combined_text,  # Keep original for fallback parsing
                'edit_tags': edit_tags  # Keep as-is (list or empty list, not None)
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
        Extracts edit tags and generates audio.

        NOTE: This is called from pause-split path, so job tracking is NOT updated here.
        The parent already set up job tracking at the block level.

        Args:
            character: Character name
            text: Text segment (may contain edit tags)
            voice_mapping: Voice mapping
            params: Generation parameters
            audio_segments: List to append result to
        """
        voice_ref = voice_mapping.get(character)

        # Extract edit tags BEFORE generation
        clean_text, edit_tags = get_edit_tags_for_segment(text)
        if edit_tags:
            print(f"  🎨 Found {len(edit_tags)} edit tag(s) in pause segment")

        # Generate audio with CLEAN text (no edit tags, no pause tags)
        # NOTE: Don't update job tracker - parent handles it
        audio_tensor = self.adapter._generate_direct(clean_text, voice_ref, params, character)

        # Ensure correct shape [1, samples] or [channels, samples]
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)

        audio_dict = {
            'waveform': audio_tensor,
            'sample_rate': 24000,
            'character': character,
            'text': clean_text,
            'original_text': text,  # Keep original for fallback parsing
            'edit_tags': edit_tags  # Keep as-is (list or empty list, not None)
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
