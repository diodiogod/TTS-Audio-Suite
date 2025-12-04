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
        torch_dtype = engine_config.get('torch_dtype', 'bfloat16')
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
        Process using character switching mode with Step Audio EditX clone generation.
        Groups consecutive same-character segments for better long-form generation.
        Supports per-segment parameters.

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

        # Group consecutive same-character segments for optimization
        grouped_segments = self._group_consecutive_character_objects(segment_objects)
        print(f"🔄 Step Audio EditX: Grouped {len(segment_objects)} segments into {len(grouped_segments)} character blocks")

        # Calculate total chunks across all blocks for time estimation
        # We need to estimate chunks based on text length and max_chars
        chunk_texts = []
        for _, segment_list in grouped_segments:
            block_text = ' '.join(seg.text.strip() for seg in segment_list)
            if enable_chunking and len(block_text) > max_chars:
                # Estimate chunk count and sizes
                chunks = self.chunker.split_into_chunks(block_text, max_chars)
                for chunk in chunks:
                    chunk_texts.append(len(chunk))
            else:
                chunk_texts.append(len(block_text))
        # Only start job if not already tracking (SRT processor manages job at higher level)
        self._srt_mode = self.adapter.job_tracker is not None
        if not self._srt_mode:
            self.adapter.start_job(total_blocks=len(chunk_texts), block_texts=chunk_texts)
        self._chunk_idx = 0  # Track chunk index across all blocks

        for group_idx, (character, segment_list) in enumerate(grouped_segments):
            # Check for interruption before processing each character block
            if model_management.interrupt_processing:
                raise InterruptedError(f"Step Audio EditX character block {group_idx + 1}/{len(grouped_segments)} ({character}) interrupted by user")
            print(f"\n🎤 Block {group_idx + 1}: Character '{character}' with {len(segment_list)} segments")

            # Combine text blocks for this character
            combined_text = ' '.join(seg.text.strip() for seg in segment_list)

            # All segments in a group have the same parameters (grouping broke on parameter changes)
            # So just take parameters from first segment
            group_params = params.copy()
            if segment_list and segment_list[0].parameters:
                segment_params = apply_segment_parameters(group_params, segment_list[0].parameters, 'step_audio_editx')
                group_params.update(segment_params)
                if segment_list[0].parameters:
                    print(f"  📊 Applying parameters: {segment_list[0].parameters}")

            # Process the combined character block with group parameters
            self._process_character_block(character, combined_text, voice_mapping, group_params,
                                        enable_chunking, max_chars, audio_segments)

            print()  # New line after progress bar

        # End job tracking (only if we started it)
        if not self._srt_mode:
            self.adapter.end_job()

        return audio_segments

    def _group_consecutive_character_objects(self, segment_objects) -> List[Tuple[str, list]]:
        """
        Group consecutive same-character segment objects for optimization.
        IMPORTANT: Groups are broken when parameters change (each parameter set gets own generation).

        Args:
            segment_objects: List of CharacterSegment objects

        Returns:
            List of (character, segment_object_list) tuples with grouped segments
        """
        if not segment_objects:
            return []

        grouped = []
        current_character = None
        current_parameters = None
        current_segments = []

        for segment in segment_objects:
            # Check if character OR parameters changed
            character_changed = segment.character != current_character
            parameters_changed = segment.parameters != current_parameters

            if character_changed or parameters_changed:
                # Character or parameters changed - finalize previous group
                if current_character is not None:
                    grouped.append((current_character, current_segments))

                # Start new group
                current_character = segment.character
                current_parameters = segment.parameters.copy() if segment.parameters else {}
                current_segments = [segment]
            else:
                # Same character AND same parameters, add to current group
                current_segments.append(segment)

        # Don't forget the last group
        if current_character is not None:
            grouped.append((current_character, current_segments))

        return grouped

    def _process_character_block(self, character: str, combined_text: str,
                               voice_mapping: Dict[str, Any], params: Dict,
                               enable_chunking: bool, max_chars: int,
                               audio_segments: List[Dict]) -> None:
        """
        Process a combined character block (potentially with chunking).

        Args:
            character: Character name
            combined_text: Combined text for this character
            voice_mapping: Voice mapping
            params: Generation parameters
            enable_chunking: Whether chunking is enabled
            max_chars: Max characters per chunk
            audio_segments: List to append results to
        """
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
                    'text': chunk
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

            print(f"🎭 Step Audio EditX - Generating for '{character}'{voice_note}:")
            print("="*60)
            print(combined_text)
            print("="*60)

            # Set current segment for time tracking (skip in SRT mode - managed at subtitle level)
            if not self._srt_mode:
                self.adapter.set_current_block(self._chunk_idx)

            # Process pause tags
            audio_tensor = self.adapter.generate_with_pause_tags(
                combined_text, voice_ref, params, True, character
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
                'text': combined_text
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
