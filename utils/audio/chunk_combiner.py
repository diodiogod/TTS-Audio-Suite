"""
Chunk Combination Utility - Modular audio chunk combination for all TTS engines
Provides standardized chunk combination methods to avoid code duplication
"""

import torch
from typing import List, Optional
from .processing import AudioProcessingUtils


class ChunkCombiner:
    """
    Unified chunk combination utility for all TTS engines.
    Provides consistent chunk combination methods with auto-selection logic.
    """
    
    @staticmethod
    def combine_chunks(audio_segments: List[torch.Tensor], 
                      method: str = "auto",
                      silence_ms: int = 100,
                      crossfade_duration: float = 0.1,
                      sample_rate: int = 24000,
                      text_length: int = 0,
                      original_text: str = "",
                      text_chunks: List[str] = None) -> torch.Tensor:
        """
        Combine audio chunks using specified method.
        
        Args:
            audio_segments: List of audio tensors to combine
            method: Combination method ("auto", "concatenate", "silence", "crossfade")
            silence_ms: Silence duration in milliseconds (for "silence" method)
            crossfade_duration: Crossfade duration in seconds (for "crossfade" method)
            sample_rate: Audio sample rate
            text_length: Original text length for auto-selection (legacy fallback)
            original_text: Original text before chunking (for smart auto-selection)
            text_chunks: List of text chunks after splitting (for smart auto-selection)
            
        Returns:
            Combined audio tensor
        """
        if not audio_segments:
            return torch.empty(0)
            
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        # Auto-select best method based on chunking analysis
        if method == "auto":
            method = ChunkCombiner._smart_auto_select_method(
                original_text, text_chunks, text_length, len(audio_segments)
            )
        
        # Convert method names to AudioProcessingUtils format
        if method == "concatenate":
            return AudioProcessingUtils.concatenate_audio_segments(audio_segments, "simple")
        
        elif method == "silence":
            silence_duration = silence_ms / 1000.0  # Convert to seconds
            return AudioProcessingUtils.concatenate_audio_segments(
                audio_segments, "silence", 
                silence_duration=silence_duration, 
                sample_rate=sample_rate
            )
        
        elif method == "crossfade":
            return AudioProcessingUtils.concatenate_audio_segments(
                audio_segments, "crossfade", 
                crossfade_duration=crossfade_duration, 
                sample_rate=sample_rate
            )
        
        else:
            raise ValueError(f"Unknown combination method: {method}")
    
    @staticmethod
    def _smart_auto_select_method(original_text: str, text_chunks: List[str] = None, 
                                 text_length: int = 0, chunk_count: int = 1) -> str:
        """
        Smart auto-select optimal combination method based on chunking analysis.
        
        Args:
            original_text: Original text before chunking
            text_chunks: List of text chunks after splitting
            text_length: Text length for fallback (legacy)
            chunk_count: Number of chunks for fallback
            
        Returns:
            Selected method name
        """
        # If we have chunk analysis data, use smart selection
        if text_chunks and len(text_chunks) > 1:
            return ChunkCombiner._analyze_chunk_split_patterns(original_text, text_chunks)
        
        # Fallback to legacy character-count method
        return ChunkCombiner._legacy_auto_select_method(text_length, chunk_count)
    
    @staticmethod
    def _analyze_chunk_split_patterns(original_text: str, text_chunks: List[str]) -> str:
        """
        Analyze how the chunker split the text to determine optimal combination method.
        
        Args:
            original_text: Original text before chunking
            text_chunks: List of text chunks after splitting
            
        Returns:
            Selected method name based on split analysis
        """
        import re
        
        sentence_boundary_splits = 0
        comma_splits = 0
        forced_character_splits = 0
        short_responses = 0
        
        # Analyze each chunk boundary
        for i in range(len(text_chunks) - 1):
            current_chunk = text_chunks[i].strip()
            next_chunk = text_chunks[i + 1].strip()
            
            # Check if current chunk ends with sentence punctuation
            if re.search(r'[.!?]\s*$', current_chunk):
                sentence_boundary_splits += 1
            # Check if current chunk ends with comma
            elif re.search(r',\s*$', current_chunk):
                comma_splits += 1
            # Check if split appears to be mid-word or forced
            elif not re.search(r'[.!?,;:]\s*$', current_chunk):
                forced_character_splits += 1
            
            # Check for short conversational responses
            if len(current_chunk) < 50 and re.search(r'^(yes|no|okay|ok|sure|hello|hi|thanks|thank you)\b', current_chunk.lower()):
                short_responses += 1
        
        total_splits = len(text_chunks) - 1
        
        # Decision logic based on split analysis
        
        # If mostly sentence boundary splits, use silence for natural pauses
        if sentence_boundary_splits >= total_splits * 0.7:
            return "silence"
        
        # If we have forced character splits, definitely use crossfade to smooth over artificial breaks
        if forced_character_splits > 0:
            return "crossfade"
        
        # If mostly comma splits, use crossfade for smooth flow
        if comma_splits >= total_splits * 0.5:
            return "crossfade"
        
        # If we have short conversational responses, use silence for clarity
        if short_responses > 0:
            return "silence"
        
        # If chunks are very short (likely natural speech units), concatenate
        avg_chunk_length = sum(len(chunk) for chunk in text_chunks) / len(text_chunks)
        if avg_chunk_length < 100:
            return "concatenate"
        
        # Default to crossfade for smooth transitions
        return "crossfade"
    
    @staticmethod
    def _legacy_auto_select_method(text_length: int, chunk_count: int) -> str:
        """
        Legacy auto-select method based on character count (fallback only).
        
        Args:
            text_length: Length of original text in characters
            chunk_count: Number of chunks being combined
            
        Returns:
            Selected method name
        """
        # For very short text or single chunks, use simple concatenation
        if text_length < 200 or chunk_count <= 2:
            return "concatenate"
        
        # For medium text, use crossfade for smooth transitions
        elif text_length < 800:
            return "crossfade" 
            
        # For long text with many chunks, use silence padding for clarity
        else:
            return "silence"
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available combination methods."""
        return ["auto", "concatenate", "silence", "crossfade"]
    
    @staticmethod
    def get_method_description(method: str) -> str:
        """Get description of combination method."""
        descriptions = {
            "auto": "Smart analysis of chunk split patterns (sentence boundaries, commas, etc.)",
            "concatenate": "Direct joining with no gap or transition", 
            "silence": "Add silence padding between chunks for clarity",
            "crossfade": "Smooth crossfade transitions between chunks"
        }
        return descriptions.get(method, "Unknown method")