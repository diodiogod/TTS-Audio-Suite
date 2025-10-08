"""
VibeVoice Internal Processor - Handles TTS generation orchestration
Called by unified TTS nodes when using VibeVoice engine
"""

import torch
from typing import Dict, Any, Optional, List, Tuple
import os
import sys

# Add project root to path
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.text.chunking import ImprovedChatterBoxChunker
from utils.audio.processing import AudioProcessingUtils
from utils.text.character_parser import parse_character_text
from engines.adapters.vibevoice_adapter import VibeVoiceEngineAdapter


class VibeVoiceProcessor:
    """
    Internal processor for VibeVoice TTS generation.
    Handles chunking, character processing, and generation orchestration.
    """
    
    def __init__(self, node_instance, engine_config: Dict[str, Any]):
        """
        Initialize VibeVoice processor.
        
        Args:
            node_instance: Parent node instance
            engine_config: Engine configuration from VibeVoice Engine node
        """
        self.node = node_instance
        self.config = engine_config
        self.adapter = VibeVoiceEngineAdapter(node_instance)
        self.chunker = ImprovedChatterBoxChunker()
        
        # Load model with enhanced parameters
        model_name = engine_config.get('model', 'vibevoice-1.5B')
        device = engine_config.get('device', 'auto')
        attention_mode = engine_config.get('attention_mode', 'auto')
        quantize_llm_4bit = engine_config.get('quantize_llm_4bit', False)
        
        # Load model with new parameters (Credits: based on wildminder/ComfyUI-VibeVoice enhancements)
        self.adapter.load_base_model(model_name, device, attention_mode, quantize_llm_4bit)
    
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
        
        # Check for time-based chunking from config
        chunk_chars = self.config.get('chunk_chars', 0)
        chunk_minutes = self.config.get('chunk_minutes', 0)
        
        # IMPORTANT: chunk_minutes from VibeVoice Engine overrides TTS Text chunking
        if chunk_minutes > 0:
            # Use time-based chunking
            enable_chunking = True
            max_chars_per_chunk = chunk_chars
        elif chunk_minutes == 0:
            # chunk_minutes=0 means NO CHUNKING AT ALL (override TTS Text settings)
            enable_chunking = False
            max_chars_per_chunk = 999999  # Effectively disable chunking
        # Note: If chunk_minutes is not set (None), fall back to TTS Text settings
        
        # Parse character segments (allow auto-discovery like ChatterBox)
        character_segments = parse_character_text(text, None)  # Auto-discover all characters from text
        
        # Auto-detect manual "Speaker N:" format and suggest Native mode
        multi_speaker_mode = self.config.get('multi_speaker_mode', 'Custom Character Switching')
        if multi_speaker_mode == "Custom Character Switching":
            if self._contains_manual_speaker_format(text):
                print("🔄 Auto-switching to Native Multi-Speaker mode (detected manual 'Speaker N:' format)")
                print("💡 TIP: Use 'Native Multi-Speaker' mode for better performance with manual format")
                multi_speaker_mode = "Native Multi-Speaker"
        
        if multi_speaker_mode == "Native Multi-Speaker":
            # Check if we can use native mode (max 4 characters)
            unique_chars = list(set([char for char, _ in character_segments]))
            if len(unique_chars) <= 4:
                print(f"🎙️ Using VibeVoice native multi-speaker mode for {len(unique_chars)} speakers")
                return self._process_native_multispeaker(character_segments, voice_mapping, params)
        
        # Use Custom Character Switching mode
        return self._process_character_switching(
            character_segments, voice_mapping, params,
            enable_chunking, max_chars_per_chunk
        )
    
    def _process_character_switching(self,
                                    segments: List[Tuple[str, str]],
                                    voice_mapping: Dict[str, Any],
                                    params: Dict,
                                    enable_chunking: bool,
                                    max_chars: int) -> List[Dict]:
        """
        Process using character switching mode with VibeVoice-optimized grouping.
        Groups consecutive same-character segments for better long-form generation.
        
        Args:
            segments: Character segments
            voice_mapping: Voice mapping
            params: Generation parameters
            enable_chunking: Whether to chunk
            max_chars: Max chars per chunk
            
        Returns:
            List of audio segments
        """
        audio_segments = []
        
        # Group consecutive same-character segments for VibeVoice optimization
        grouped_segments = self._group_consecutive_characters(segments)
        print(f"🔄 VibeVoice Custom: Grouped {len(segments)} segments into {len(grouped_segments)} character blocks")
        
        for group_idx, (character, text_list) in enumerate(grouped_segments):
            print(f"🎤 Block {group_idx + 1}: Character '{character}' with {len(text_list)} segments")
            
            # Combine text blocks for this character (VibeVoice style)
            combined_text = '\n'.join(f"Speaker 1: {text.strip()}" for text in text_list)
            
            # Process the combined character block
            self._process_character_block(character, combined_text, voice_mapping, params, 
                                        enable_chunking, max_chars, audio_segments)
        
        return audio_segments
    
    def _group_consecutive_characters(self, segments: List[Tuple[str, str]]) -> List[Tuple[str, List[str]]]:
        """
        Group consecutive same-character segments for VibeVoice optimization.
        
        Args:
            segments: Original character segments
            
        Returns:
            List of (character, text_list) tuples with grouped segments
        """
        if not segments:
            return []
            
        grouped = []
        current_character = None
        current_texts = []
        
        for character, text in segments:
            if character == current_character:
                # Same character, add to current group
                current_texts.append(text)
            else:
                # Different character, finalize previous group
                if current_character is not None:
                    grouped.append((current_character, current_texts))
                
                # Start new group
                current_character = character
                current_texts = [text]
        
        # Don't forget the last group
        if current_character is not None:
            grouped.append((current_character, current_texts))
        
        return grouped
    
    def _process_character_block(self, character: str, combined_text: str, 
                               voice_mapping: Dict[str, Any], params: Dict,
                               enable_chunking: bool, max_chars: int, 
                               audio_segments: List[Dict]) -> None:
        """
        Process a combined character block (potentially with chunking).
        
        Args:
            character: Character name
            combined_text: Combined text for this character (VibeVoice format)
            voice_mapping: Voice mapping
            params: Generation parameters  
            enable_chunking: Whether chunking is enabled
            max_chars: Max characters per chunk
            audio_segments: List to append results to
        """
        # Apply chunking if enabled and text is long
        if enable_chunking and len(combined_text) > max_chars:
            chunks = self.chunker.split_into_chunks(combined_text, max_chars)
            print(f"📝 Chunking {character}'s combined text into {len(chunks)} chunks")
            
            for chunk in chunks:
                voice_ref = voice_mapping.get(character)
                audio_tensor = self.adapter.generate_vibevoice_with_pause_tags(
                    chunk, voice_ref, params, True, character
                )
                # Convert tensor back to dict format
                audio_dict = {
                    'waveform': audio_tensor.unsqueeze(0) if audio_tensor.dim() == 2 else audio_tensor,
                    'sample_rate': 24000,
                    'character': character,
                    'text': chunk
                }
                audio_segments.append(audio_dict)
        else:
            # Generate without chunking - the entire combined block at once
            print(f"🎭 CUSTOM CHARACTER BLOCK - Generating combined text for '{character}':")
            print("="*60)
            print(combined_text)
            print("="*60)
            
            voice_ref = voice_mapping.get(character)
            audio_tensor = self.adapter.generate_vibevoice_with_pause_tags(
                combined_text, voice_ref, params, True, character
            )
            # Convert tensor back to dict format
            audio_dict = {
                'waveform': audio_tensor.unsqueeze(0) if audio_tensor.dim() == 2 else audio_tensor,
                'sample_rate': 24000,
                'character': character,
                'text': combined_text
            }
            audio_segments.append(audio_dict)
    
    def _process_native_multispeaker(self,
                                    segments: List[Tuple[str, str]],
                                    voice_mapping: Dict[str, Any],
                                    params: Dict) -> List[Dict]:
        """
        Process using native multi-speaker mode with chunking support.

        Args:
            segments: Character segments
            voice_mapping: Voice mapping
            params: Generation parameters

        Returns:
            List of audio segments (single if no chunking, multiple if chunked)
        """
        # Check if chunking is needed based on config
        chunk_chars = self.config.get('chunk_chars', 0)
        chunk_minutes = self.config.get('chunk_minutes', 0)

        # Calculate total text length
        total_text = ' '.join([text for _, text in segments])
        total_length = len(total_text)

        # Determine if chunking should be applied
        should_chunk = chunk_minutes > 0 and total_length > chunk_chars

        if not should_chunk:
            # No chunking - generate everything at once
            audio = self.adapter._generate_native_multispeaker(
                segments, voice_mapping, params, None
            )
            return [audio]

        # Chunking enabled - rebuild text and split using existing chunker
        print(f"📝 Native Multi-Speaker: Chunking {len(segments)} segments (max {chunk_chars} chars per chunk)")

        # Reconstruct the full formatted text with Speaker tags
        formatted_lines = []
        character_map = {}
        for character, text in segments:
            if character not in character_map:
                speaker_idx = len(character_map) + 1  # 1-based indexing
                if speaker_idx > 4:
                    speaker_idx = 4  # Max 4 speakers
                character_map[character] = speaker_idx
            speaker_idx = character_map[character]
            formatted_lines.append(f"Speaker {speaker_idx}: {text.strip()}")

        full_text = "\n".join(formatted_lines)

        # Use existing chunker to intelligently split the text
        text_chunks = self.chunker.split_into_chunks(full_text, chunk_chars)
        print(f"✂️ Split {len(full_text)} chars into {len(text_chunks)} chunks")

        # Parse each chunk back into (character, text) segments
        chunk_groups = []
        for chunk_text in text_chunks:
            chunk_segments = []
            # Parse Speaker N: format back to segments
            for line in chunk_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Match "Speaker N: text"
                import re
                match = re.match(r'^Speaker\s+(\d+):\s*(.+)$', line)
                if match:
                    speaker_num = int(match.group(1))
                    text_content = match.group(2)
                    # Find character name for this speaker number
                    char_name = next((char for char, idx in character_map.items() if idx == speaker_num), f"Speaker {speaker_num}")
                    chunk_segments.append((char_name, text_content))
                else:
                    # Fallback: treat as narrator text
                    chunk_segments.append(("narrator", line))

            if chunk_segments:
                chunk_groups.append(chunk_segments)

        print(f"🔄 Split into {len(chunk_groups)} chunks for native multi-speaker generation")

        # Generate each chunk using native multi-speaker
        audio_results = []
        for i, chunk_segments in enumerate(chunk_groups):
            chunk_chars_total = sum(len(text) for _, text in chunk_segments)
            print(f"📦 Chunk {i+1}/{len(chunk_groups)}: {len(chunk_segments)} segments, {chunk_chars_total} chars")

            audio = self.adapter._generate_native_multispeaker(
                chunk_segments, voice_mapping, params, None
            )
            audio_results.append(audio)

        return audio_results
    
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
    
    def _contains_manual_speaker_format(self, text: str) -> bool:
        """
        Check if text contains manual 'Speaker N:' format.
        
        Args:
            text: Input text to check
            
        Returns:
            True if manual Speaker format is detected
        """
        import re
        return bool(re.search(r'speaker\s*\d+\s*:', text, re.IGNORECASE))
    
    def cleanup(self):
        """Clean up resources"""
        if self.adapter:
            self.adapter.cleanup()