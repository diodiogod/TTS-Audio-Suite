"""
Pause Tag Processing Module
Handles parsing and processing of pause tags in text for TTS generation.

Supports formats:
- [pause:2] - 2 seconds
- [pause:1.5] - 1.5 seconds  
- [pause:500ms] - 500 milliseconds
- [pause:2s] - 2 seconds (explicit)

Aliases supported:
- [wait:2] - same as [pause:2]
- [stop:1.5] - same as [pause:1.5]
"""

import re
import torch
from typing import List, Tuple, Union, Optional
from utils.audio.processing import AudioProcessingUtils


class PauseTagProcessor:
    """Handles pause tag parsing and audio generation with pauses"""
    
    # Regex pattern for flexible pause tag matching (supports pause, wait, stop aliases)
    PAUSE_PATTERN = r'\[(pause|wait|stop):(\d+(?:\.\d+)?)(s|ms)?\]'
    
    @staticmethod
    def parse_pause_tags(text: str) -> Tuple[List[Tuple[str, Union[str, float]]], str]:
        """
        Parse pause tags from text and return segments with pause information.
        
        Args:
            text: Input text with pause tags
            
        Returns:
            Tuple of (segments, clean_text) where segments contain 
            ('text', content) or ('pause', duration_seconds)
        """
        def normalize_duration(duration_str: str, unit: Optional[str] = None) -> float:
            """Convert pause duration to seconds"""
            duration = float(duration_str)
            if unit == 'ms':
                return duration / 1000.0
            return duration
        
        segments = []
        last_end = 0
        
        for match in re.finditer(PauseTagProcessor.PAUSE_PATTERN, text):
            # Add text before pause tag
            if match.start() > last_end:
                text_content = text[last_end:match.start()].strip()
                if text_content:
                    segments.append(('text', text_content))
            
            # Add pause segment (group 1 is pause/wait/stop, group 2 is duration, group 3 is unit)
            duration = normalize_duration(match.group(2), match.group(3))
            segments.append(('pause', duration))
            last_end = match.end()
        
        # Add remaining text
        if last_end < len(text):
            remaining_text = text[last_end:].strip()
            if remaining_text:
                segments.append(('text', remaining_text))
        
        # Create clean text without pause tags
        clean_text = re.sub(PauseTagProcessor.PAUSE_PATTERN, ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return segments, clean_text
    
    @staticmethod
    def has_pause_tags(text: str) -> bool:
        """Check if text contains pause tags"""
        return bool(re.search(PauseTagProcessor.PAUSE_PATTERN, text))
    
    @staticmethod
    def create_silence_segment(duration_seconds: float, sample_rate: int, 
                             device: Optional[torch.device] = None, 
                             dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Create silence tensor for pause segments.
        
        Args:
            duration_seconds: Duration of silence
            sample_rate: Audio sample rate
            device: Target device for tensor
            dtype: Target dtype for tensor
            
        Returns:
            Silence tensor with shape (1, num_samples)
        """
        # Clamp duration to reasonable limits
        duration_seconds = max(0.0, min(duration_seconds, 30.0))  # 0 to 30 seconds max
        
        num_samples = int(duration_seconds * sample_rate)
        silence = torch.zeros(1, num_samples, device=device, dtype=dtype)
        return silence
    
    @staticmethod
    def generate_audio_with_pauses(segments: List[Tuple[str, Union[str, float]]], 
                                 tts_generate_func, sample_rate: int,
                                 **generation_kwargs) -> torch.Tensor:
        """
        Generate audio from parsed segments with pauses.
        
        Args:
            segments: List of ('text', content) or ('pause', duration) tuples
            tts_generate_func: Function to call for text generation
            sample_rate: Audio sample rate
            **generation_kwargs: Additional arguments for TTS generation
            
        Returns:
            Combined audio tensor with pauses
        """
        if not segments:
            return torch.zeros(1, 0)
        
        audio_segments = []
        
        for segment_type, content in segments:
            if segment_type == 'text':
                # Generate audio for text segment
                audio = tts_generate_func(content, **generation_kwargs)
                
                # Ensure correct shape (1, num_samples)
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                elif audio.dim() > 2:
                    audio = audio.squeeze()
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                
                audio_segments.append(audio)
                
            elif segment_type == 'pause':
                # Create silence segment
                device = audio_segments[-1].device if audio_segments else None
                dtype = audio_segments[-1].dtype if audio_segments else None
                
                silence = PauseTagProcessor.create_silence_segment(
                    content, sample_rate, device, dtype
                )
                audio_segments.append(silence)
        
        # Concatenate all segments
        if audio_segments:
            return torch.cat(audio_segments, dim=-1)
        else:
            return torch.zeros(1, 0)
    
    @staticmethod
    def preprocess_text_with_pause_tags(text: str, enable_pause_tags: bool = True) -> Tuple[str, Optional[List]]:
        """
        Preprocess text for TTS generation, handling pause tags if enabled.
        
        Args:
            text: Input text
            enable_pause_tags: Whether to process pause tags
            
        Returns:
            Tuple of (processed_text, segments_or_none)
            - If pause tags disabled or no tags found: (original_text, None)
            - If pause tags enabled and found: (clean_text, segments)
        """
        if not enable_pause_tags or not PauseTagProcessor.has_pause_tags(text):
            return text, None
        
        segments, clean_text = PauseTagProcessor.parse_pause_tags(text)
        return clean_text, segments