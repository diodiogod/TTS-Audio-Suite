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
        
        # Process based on mode
        multi_speaker_mode = self.config.get('multi_speaker_mode', 'Custom Character Switching')
        
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
        Process using character switching mode (generate per character).
        
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
        
        for character, text in segments:
            # Apply chunking if enabled and text is long
            if enable_chunking and len(text) > max_chars:
                chunks = self.chunker.split_into_chunks(text, max_chars)
                print(f"📝 Chunking {character}'s text into {len(chunks)} chunks")
                
                for chunk in chunks:
                    # Fix: Call generate_segment directly with proper Dict parameter, not generate_segment_audio
                    # print(f"🐛 VIBEVOICE_PROCESSOR: Calling generate_segment for character '{character}' (chunked)")
                    voice_ref = voice_mapping.get(character)
                    # print(f"🐛 VIBEVOICE_PROCESSOR: voice_ref type: {type(voice_ref)}")
                    # Use modular pause tag processing like F5 does
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
                    # generate_segment already returns dict format
                    # (audio_dict already added above)
            else:
                # Generate without chunking
                # print(f"🐛 VIBEVOICE_PROCESSOR: Calling generate_segment for character '{character}' (no chunking)")
                voice_ref = voice_mapping.get(character)
                # print(f"🐛 VIBEVOICE_PROCESSOR: voice_ref type: {type(voice_ref)}")
                # Use modular pause tag processing like F5 does
                audio_tensor = self.adapter.generate_vibevoice_with_pause_tags(
                    text, voice_ref, params, True, character
                )
                # Convert tensor back to dict format
                audio_dict = {
                    'waveform': audio_tensor.unsqueeze(0) if audio_tensor.dim() == 2 else audio_tensor,
                    'sample_rate': 24000,
                    'character': character,
                    'text': text
                }
                # generate_segment already returns dict format
                audio_segments.append(audio_dict)
        
        return audio_segments
    
    def _process_native_multispeaker(self,
                                    segments: List[Tuple[str, str]],
                                    voice_mapping: Dict[str, Any],
                                    params: Dict) -> List[Dict]:
        """
        Process using native multi-speaker mode.
        
        Args:
            segments: Character segments
            voice_mapping: Voice mapping
            params: Generation parameters
            
        Returns:
            List with single combined audio segment
        """
        # Use adapter's native multi-speaker generation
        audio = self.adapter._generate_native_multispeaker(
            segments, voice_mapping, params
        )
        return [audio]
    
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