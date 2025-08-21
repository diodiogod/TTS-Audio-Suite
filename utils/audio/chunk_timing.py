"""
Chunk Timing Utility - Modular chunk combination timing information for TTS engines
Provides standardized chunk timing info integration for audio generation results
"""

from typing import Dict, Any, List, Tuple, Optional
from .chunk_combiner import ChunkCombiner


class ChunkTimingHelper:
    """
    Helper class to integrate chunk combination timing information into TTS engine generation results.
    Provides consistent chunk timing info formatting across all engines.
    """
    
    @staticmethod
    def combine_audio_with_timing(audio_segments: List[Any], 
                                 combination_method: str = "auto",
                                 silence_ms: int = 100,
                                 crossfade_duration: float = 0.1,
                                 sample_rate: int = 24000,
                                 text_length: int = 0,
                                 original_text: str = "",
                                 text_chunks: List[str] = None,
                                 combiner_func=None) -> Tuple[Any, Optional[Dict]]:
        """
        Combine audio segments and return timing information.
        
        Args:
            audio_segments: List of audio segments to combine
            combination_method: Method to use for combination
            silence_ms: Silence duration in milliseconds
            crossfade_duration: Crossfade duration in seconds
            sample_rate: Audio sample rate
            text_length: Original text length
            original_text: Original text before chunking
            text_chunks: List of text chunks
            combiner_func: Optional custom combiner function (for engine-specific wrappers)
            
        Returns:
            Tuple of (combined_audio, timing_info_dict)
        """
        if len(audio_segments) <= 1:
            return audio_segments[0] if audio_segments else None, None
        
        # Use ChunkCombiner directly or custom combiner function
        if combiner_func:
            # For engines that need custom audio format handling
            combined_audio, chunk_info = combiner_func(
                audio_segments=audio_segments,
                combination_method=combination_method,
                silence_ms=silence_ms,
                crossfade_duration=crossfade_duration,
                sample_rate=sample_rate,
                text_length=text_length,
                original_text=original_text,
                text_chunks=text_chunks,
                return_info=True
            )
        else:
            # Direct ChunkCombiner usage for simple tensor lists
            combined_audio, chunk_info = ChunkCombiner.combine_chunks(
                audio_segments=audio_segments,
                method=combination_method,
                silence_ms=silence_ms,
                crossfade_duration=crossfade_duration,
                sample_rate=sample_rate,
                text_length=text_length,
                original_text=original_text,
                text_chunks=text_chunks,
                return_info=True
            )
        
        return combined_audio, chunk_info
    
    @staticmethod
    def enhance_generation_info(base_info: str, chunk_info: Optional[Dict]) -> str:
        """
        Enhance generation info string with timing information.
        
        Args:
            base_info: Base generation info string
            chunk_info: Chunk timing information dictionary
            
        Returns:
            Enhanced info string with timing details
        """
        if not chunk_info:
            return base_info
        
        timing_info = ChunkCombiner.format_timing_info(chunk_info)
        if timing_info:
            return f"{base_info}\n\n{timing_info}"
        
        return base_info
    
    @staticmethod
    def create_engine_combiner_wrapper(engine_combine_func):
        """
        Create a wrapper for engine-specific combine functions that returns timing info.
        
        Args:
            engine_combine_func: Engine's existing combine function
            
        Returns:
            Wrapped function that can return timing info
        """
        def wrapped_combiner(audio_segments, combination_method="auto", silence_ms=100,
                           crossfade_duration=0.1, sample_rate=24000, text_length=0,
                           original_text="", text_chunks=None, return_info=False, **kwargs):
            
            if not return_info:
                # Use original function for backward compatibility
                return engine_combine_func(
                    audio_segments, combination_method, silence_ms, 
                    text_length=text_length, **kwargs
                )
            
            # For timing info, we need to extract tensors and use ChunkCombiner
            if hasattr(audio_segments[0], 'get'):
                # Handle dict format (like Higgs Audio)
                tensors = [seg["waveform"] for seg in audio_segments]
            else:
                # Handle direct tensor format
                tensors = audio_segments
            
            combined_tensor, chunk_info = ChunkCombiner.combine_chunks(
                audio_segments=tensors,
                method=combination_method,
                silence_ms=silence_ms,
                crossfade_duration=crossfade_duration,
                sample_rate=sample_rate,
                text_length=text_length,
                original_text=original_text,
                text_chunks=text_chunks,
                return_info=True
            )
            
            # Convert back to engine format if needed
            if hasattr(audio_segments[0], 'get'):
                combined_audio = {"waveform": combined_tensor, "sample_rate": sample_rate}
            else:
                combined_audio = combined_tensor
            
            return combined_audio, chunk_info
        
        return wrapped_combiner