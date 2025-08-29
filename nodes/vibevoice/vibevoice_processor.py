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
        
        # Load model
        model_name = engine_config.get('model', 'vibevoice-1.5B')
        device = engine_config.get('device', 'auto')
        self.adapter.load_base_model(model_name, device)
    
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
        if chunk_chars > 0:
            # Override with time-based chunking
            enable_chunking = True
            max_chars_per_chunk = chunk_chars
        
        # Parse character segments
        character_segments = parse_character_text(text, list(voice_mapping.keys()))
        
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
                chunks = self.chunker.chunk_text(text, max_chars)
                print(f"📝 Chunking {character}'s text into {len(chunks)} chunks")
                
                for chunk in chunks:
                    audio = self.adapter.generate_segment_audio(
                        chunk, 
                        voice_mapping.get(character),
                        "",  # char_text (VibeVoice doesn't use reference text)
                        character,
                        **params
                    )
                    # Convert tensor back to dict format for consistency
                    audio_dict = {
                        "waveform": audio,
                        "sample_rate": 24000
                    }
                    audio_segments.append(audio_dict)
            else:
                # Generate without chunking
                audio = self.adapter.generate_segment_audio(
                    text,
                    voice_mapping.get(character),
                    "",  # char_text (VibeVoice doesn't use reference text)
                    character,
                    **params
                )
                # Convert tensor back to dict format for consistency
                audio_dict = {
                    "waveform": audio,
                    "sample_rate": 24000
                }
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