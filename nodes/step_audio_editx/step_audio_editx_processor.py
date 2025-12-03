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

        for group_idx, (character, segment_list) in enumerate(grouped_segments):
            # Check for interruption before processing each character block
            if model_management.interrupt_processing:
                raise InterruptedError(f"Step Audio EditX character block {group_idx + 1}/{len(grouped_segments)} ({character}) interrupted by user")
            print(f"🎤 Block {group_idx + 1}: Character '{character}' with {len(segment_list)} segments")

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

                # Process pause tags
                audio_tensor = self.adapter.generate_with_pause_tags(
                    chunk, voice_ref, params, True, character
                )

                # Convert tensor back to dict format
                audio_dict = {
                    'waveform': audio_tensor.unsqueeze(0) if audio_tensor.dim() == 2 else audio_tensor,
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

            # Process pause tags
            audio_tensor = self.adapter.generate_with_pause_tags(
                combined_text, voice_ref, params, True, character
            )

            # Convert tensor back to dict format
            audio_dict = {
                'waveform': audio_tensor.unsqueeze(0) if audio_tensor.dim() == 2 else audio_tensor,
                'sample_rate': 24000,  # Step Audio EditX native sample rate
                'character': character,
                'text': combined_text
            }
            audio_segments.append(audio_dict)

    def combine_audio_segments(self,
                              segments: List[Dict],
                              method: str = "auto",
                              silence_ms: int = 100) -> torch.Tensor:
        """
        Combine multiple audio segments.

        Args:
            segments: List of audio dicts
            method: Combination method
            silence_ms: Silence between segments

        Returns:
            Combined audio tensor
        """
        if not segments:
            return torch.zeros(1, 1, 0)

        # Extract waveforms
        waveforms = []
        for seg in segments:
            wave = seg['waveform']
            if wave.dim() == 3:
                wave = wave.squeeze(0)  # Remove batch dim
            if wave.dim() == 1:
                wave = wave.unsqueeze(0)  # Add channel dim
            waveforms.append(wave)

        # Determine combination method
        if method == "auto":
            # Auto-select based on content
            total_samples = sum(w.shape[-1] for w in waveforms)
            if total_samples > 24000 * 10:  # > 10 seconds
                method = "silence_padding"
            else:
                method = "concatenate"

        # Combine based on method
        if method == "silence_padding" and silence_ms > 0:
            sample_rate = 24000
            silence_samples = int(silence_ms * sample_rate / 1000)
            silence = torch.zeros(1, silence_samples)

            combined_parts = []
            for i, wave in enumerate(waveforms):
                combined_parts.append(wave)
                if i < len(waveforms) - 1:
                    combined_parts.append(silence)

            combined = torch.cat(combined_parts, dim=-1)
        else:
            # Simple concatenation
            combined = torch.cat(waveforms, dim=-1)

        # Ensure proper shape
        if combined.dim() == 2:
            combined = combined.unsqueeze(0)  # Add batch dim

        return combined

    def cleanup(self):
        """Clean up resources"""
        if self.adapter:
            self.adapter.cleanup()
