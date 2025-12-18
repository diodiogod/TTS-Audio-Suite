"""
CosyVoice3 Processor - Handles TTS generation orchestration
Called by unified TTS nodes when using CosyVoice3 engine

Provides:
- Character switching with [CharacterName] syntax
- Pause tag support ([pause:2s], [wait:500ms])
- Chunking for long text
- Caching integration
"""

import torch
from typing import Dict, Any, Optional, List, Tuple
import os
import sys

# Add project root to path
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.text.chunking import ImprovedChatterBoxChunker
from utils.audio.processing import AudioProcessingUtils
from utils.text.character_parser import CharacterParser
from utils.text.pause_processor import PauseTagProcessor
from utils.text.segment_parameters import apply_segment_parameters
from utils.voice.discovery import get_character_mapping, get_available_characters, voice_discovery
from engines.adapters.cosyvoice_adapter import CosyVoiceAdapter


class CosyVoiceProcessor:
    """
    Internal processor for CosyVoice3 TTS generation.
    Handles character processing, pause tags, and generation orchestration.
    """

    SAMPLE_RATE = 22050  # CosyVoice3 native sample rate

    def __init__(self, engine_config: Dict[str, Any]):
        """
        Initialize CosyVoice3 processor.
        
        Args:
            engine_config: Engine configuration from CosyVoice3 Engine node
        """
        self.engine_config = engine_config
        
        # Extract configuration
        self.model_path = engine_config.get('model_path', 'Fun-CosyVoice3-0.5B')
        self.device = engine_config.get('device', 'auto')
        self.mode = engine_config.get('mode', 'zero_shot')
        self.speed = engine_config.get('speed', 1.0)
        self.use_fp16 = engine_config.get('use_fp16', True)
        self.load_trt = engine_config.get('load_trt', False)
        self.load_vllm = engine_config.get('load_vllm', False)
        self.instruct_text = engine_config.get('instruct_text', '')
        self.reference_text = engine_config.get('reference_text', '')
        
        # Initialize adapter
        self.adapter = CosyVoiceAdapter()
        self.adapter.initialize_engine(
            model_path=self.model_path,
            device=self.device,
            use_fp16=self.use_fp16,
            load_trt=self.load_trt,
            load_vllm=self.load_vllm
        )
        
        # Setup character parser
        self._setup_character_parser()

    def _setup_character_parser(self):
        """Set up character parser with available characters and aliases."""
        self.character_parser = CharacterParser()
        
        # Get available characters
        available_chars = get_available_characters()
        if available_chars:
            self.character_parser.set_available_characters(list(available_chars))
        
        # Get character aliases
        character_aliases = voice_discovery.get_character_aliases()
        for alias, target in character_aliases.items():
            self.character_parser.set_character_fallback(alias, target)
        
        # Get character language defaults
        char_lang_defaults = voice_discovery.get_character_language_defaults()
        for char, lang in char_lang_defaults.items():
            self.character_parser.set_character_language_default(char, lang)

    def process_text(
        self,
        text: str,
        speaker_audio: Optional[Dict[str, Any]] = None,
        seed: int = 1,
        enable_chunking: bool = True,
        max_chars_per_chunk: int = 400,
        silence_between_chunks_ms: int = 100,
        enable_pause_tags: bool = True,
        return_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Process text and generate audio with CosyVoice3.
        
        Args:
            text: Input text with potential character tags and pause tags
            speaker_audio: Speaker reference audio tensor dict
            seed: Random seed for reproducibility
            enable_chunking: Enable text chunking for long text
            max_chars_per_chunk: Maximum characters per chunk
            silence_between_chunks_ms: Silence between chunks in ms
            enable_pause_tags: Enable pause tag processing
            return_info: Whether to return generation info
            
        Returns:
            Tuple of (audio_tensor, generation_info or None)
        """
        generation_info = {
            'engine': 'cosyvoice3',
            'mode': self.mode,
            'speed': self.speed,
            'seed': seed,
            'character_segments': [],
            'pause_tags_processed': False,
            'chunks_processed': 0
        }

        # Get speaker audio path
        speaker_audio_path = None
        if speaker_audio:
            if isinstance(speaker_audio, dict):
                speaker_audio_path = speaker_audio.get('audio_path') or speaker_audio.get('waveform_path')
            elif isinstance(speaker_audio, str):
                speaker_audio_path = speaker_audio

        # Parse character segments
        segments = self.character_parser.split_by_character(text, include_language=True)
        
        if len(segments) > 1:
            print(f"🎭 CosyVoice3: Processing {len(segments)} character segment(s)")
            generation_info['character_segments'] = [
                {'character': seg[0], 'text_preview': seg[1][:50]} for seg in segments
            ]

        # Get character mapping for voice switching
        unique_characters = set(seg[0] for seg in segments if seg[0])
        character_mapping = {}
        if unique_characters:
            character_mapping = get_character_mapping(list(unique_characters), engine_type="cosyvoice")

        audio_segments = []
        
        for character, segment_text, language in segments:
            segment_text = segment_text.strip()
            if not segment_text:
                continue

            # Determine speaker audio for this segment
            current_speaker_audio = speaker_audio_path
            current_reference_text = self.reference_text

            if character and character in character_mapping:
                char_audio, char_text = character_mapping[character]
                if char_audio:
                    current_speaker_audio = char_audio
                    if char_text:
                        current_reference_text = char_text
                    print(f"📖 Using character voice '{character}'")

            # Process pause tags if enabled
            if enable_pause_tags and PauseTagProcessor.has_pause_tags(segment_text):
                generation_info['pause_tags_processed'] = True
                
                # Define TTS generation function for pause processor
                def tts_generate_func(text_content: str, segment_params: Optional[Dict] = None):
                    return self._generate_audio_segment(
                        text=text_content,
                        speaker_audio=current_speaker_audio,
                        reference_text=current_reference_text,
                        seed=seed
                    )
                
                pause_segments, clean_text = PauseTagProcessor.parse_pause_tags(segment_text)
                segment_audio = PauseTagProcessor.generate_audio_with_pauses(
                    segments=pause_segments,
                    tts_generate_func=tts_generate_func,
                    sample_rate=self.SAMPLE_RATE
                )
            else:
                # Chunking for long text
                if enable_chunking and len(segment_text) > max_chars_per_chunk:
                    chunks = ImprovedChatterBoxChunker.chunk_text(
                        segment_text, 
                        max_chars=max_chars_per_chunk
                    )
                    generation_info['chunks_processed'] += len(chunks)
                    
                    chunk_audios = []
                    for chunk in chunks:
                        chunk_audio = self._generate_audio_segment(
                            text=chunk,
                            speaker_audio=current_speaker_audio,
                            reference_text=current_reference_text,
                            seed=seed
                        )
                        chunk_audios.append(chunk_audio)
                    
                    # Combine chunks with silence
                    segment_audio = self._combine_with_silence(
                        chunk_audios, 
                        silence_ms=silence_between_chunks_ms
                    )
                else:
                    segment_audio = self._generate_audio_segment(
                        text=segment_text,
                        speaker_audio=current_speaker_audio,
                        reference_text=current_reference_text,
                        seed=seed
                    )

            audio_segments.append(segment_audio)

        # Combine all character segments
        if audio_segments:
            final_audio = self._combine_with_silence(
                audio_segments,
                silence_ms=silence_between_chunks_ms
            )
        else:
            final_audio = torch.zeros(1, self.SAMPLE_RATE, dtype=torch.float32)

        # Normalize dimensions
        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0)
        elif final_audio.dim() > 2:
            final_audio = final_audio.squeeze()
            if final_audio.dim() == 1:
                final_audio = final_audio.unsqueeze(0)

        if return_info:
            return final_audio, generation_info
        return final_audio, None

    def _generate_audio_segment(
        self,
        text: str,
        speaker_audio: Optional[str],
        reference_text: Optional[str],
        seed: int
    ) -> torch.Tensor:
        """Generate audio for a single text segment."""
        return self.adapter.generate(
            text=text,
            speaker_audio=speaker_audio,
            reference_text=reference_text,
            mode=self.mode,
            instruct_text=self.instruct_text,
            speed=self.speed,
            seed=seed
        )

    def _combine_with_silence(
        self, 
        audio_segments: List[torch.Tensor],
        silence_ms: int = 100
    ) -> torch.Tensor:
        """Combine audio segments with silence between them."""
        if not audio_segments:
            return torch.zeros(1, 0, dtype=torch.float32)
        
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        # Create silence
        silence_samples = int(silence_ms * self.SAMPLE_RATE / 1000)
        
        combined_parts = []
        for i, segment in enumerate(audio_segments):
            # Ensure 2D shape
            if segment.dim() == 1:
                segment = segment.unsqueeze(0)
            
            combined_parts.append(segment)
            
            # Add silence between segments (not after last)
            if i < len(audio_segments) - 1:
                silence = torch.zeros(
                    segment.shape[0], 
                    silence_samples, 
                    device=segment.device, 
                    dtype=segment.dtype
                )
                combined_parts.append(silence)
        
        return torch.cat(combined_parts, dim=-1)

    def combine_audio_segments(
        self,
        segments: List[torch.Tensor],
        method: str = "auto",
        silence_ms: int = 100,
        text_chunks: Optional[List[str]] = None,
        text_length: int = 0,
        return_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Combine audio segments using unified ChunkCombiner.
        
        Args:
            segments: List of audio tensors
            method: Combination method
            silence_ms: Silence between segments
            text_chunks: Text for each segment
            text_length: Total text length
            return_info: Whether to return combination info
            
        Returns:
            Tuple of (combined_audio, combination_info or None)
        """
        from utils.audio.chunk_combiner import ChunkCombiner
        
        # Prepare segment info for combiner
        segment_info = []
        for i, seg in enumerate(segments):
            if seg.dim() == 1:
                seg = seg.unsqueeze(0)
            text = text_chunks[i] if text_chunks and i < len(text_chunks) else ""
            segment_info.append({
                'audio': seg,
                'text': text,
                'index': i
            })
        
        combined, info = ChunkCombiner.combine_chunks(
            segment_info,
            sample_rate=self.SAMPLE_RATE,
            method=method,
            silence_ms=silence_ms,
            return_info=True
        )
        
        if return_info:
            return combined, info
        return combined, None

    def cleanup(self):
        """Clean up resources"""
        if self.adapter:
            self.adapter.unload()
            self.adapter = None
