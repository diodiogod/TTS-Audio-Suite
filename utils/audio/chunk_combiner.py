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
                      text_chunks: List[str] = None,
                      return_info: bool = False) -> torch.Tensor:
        """
        Combine audio chunks using specified method with per-junction analysis for auto mode.
        
        Args:
            audio_segments: List of audio tensors to combine
            method: Combination method ("auto", "concatenate", "silence", "crossfade")
            silence_ms: Silence duration in milliseconds (for "silence" method)
            crossfade_duration: Crossfade duration in seconds (for "crossfade" method)
            sample_rate: Audio sample rate
            text_length: Original text length for auto-selection (legacy fallback)
            original_text: Original text before chunking (for smart auto-selection)
            text_chunks: List of text chunks after splitting (for smart auto-selection)
            return_info: If True, return tuple of (combined_audio, timing_info)
            
        Returns:
            Combined audio tensor, or tuple of (audio, info) if return_info=True
        """
        if not audio_segments:
            return torch.empty(0)
            
        if len(audio_segments) == 1:
            if return_info:
                chunk_info = {
                    "method_used": "none",
                    "total_chunks": 1,
                    "chunk_timings": [{"start": 0.0, "end": audio_segments[0].size(-1) / sample_rate, "text": text_chunks[0] if text_chunks else ""}],
                    "auto_selected": False,
                    "junction_methods": []
                }
                return audio_segments[0], chunk_info
            return audio_segments[0]
        
        # Handle method name aliases for backward compatibility
        if method == "silence_padding":
            method = "silence"
        
        # Auto-select method(s) - either per-junction or global
        auto_selected = method == "auto"
        junction_methods = []
        
        if method == "auto":
            # Smart per-junction analysis
            junction_methods = ChunkCombiner._analyze_per_junction_methods(
                original_text, text_chunks, text_length, len(audio_segments)
            )
            # Use most common method for display purposes
            method = ChunkCombiner._get_dominant_method(junction_methods)
        else:
            # Single method for all junctions
            junction_methods = [method] * (len(audio_segments) - 1)
        
        # Combine chunks junction by junction
        combined = ChunkCombiner._combine_with_junction_methods(
            audio_segments, junction_methods, silence_ms, crossfade_duration, sample_rate
        )
        
        # Calculate timing information if requested
        chunk_timings = []
        junction_details = []
        if return_info:
            current_time = 0.0
            for i, segment in enumerate(audio_segments):
                duration = segment.size(-1) / sample_rate
                chunk_timings.append({
                    "start": current_time,
                    "end": current_time + duration,
                    "text": text_chunks[i] if text_chunks and i < len(text_chunks) else f"Chunk {i+1}"
                })
                current_time += duration
                
                # Add junction info and transition time
                if i < len(audio_segments) - 1:
                    junction_method = junction_methods[i]
                    junction_details.append({
                        "junction": i + 1,
                        "method": junction_method,
                        "after_chunk": i + 1,
                        "before_chunk": i + 2
                    })
                    
                    # Add transition time based on method
                    if junction_method in ["silence", "silence_padding"]:
                        current_time += silence_ms / 1000.0
        
        # Return with timing info if requested
        if return_info:
            chunk_info = {
                "method_used": method,
                "total_chunks": len(audio_segments),
                "chunk_timings": chunk_timings,
                "auto_selected": auto_selected,
                "junction_methods": junction_methods,
                "junction_details": junction_details,
                "silence_ms": silence_ms,
                "crossfade_duration": crossfade_duration
            }
            return combined, chunk_info
        
        return combined
    
    @staticmethod
    def _analyze_per_junction_methods(original_text: str, text_chunks: List[str] = None, 
                                     text_length: int = 0, chunk_count: int = 1) -> List[str]:
        """
        Analyze each individual junction to determine optimal combination method.
        
        Args:
            original_text: Original text before chunking
            text_chunks: List of text chunks after splitting
            text_length: Text length for fallback (legacy)
            chunk_count: Number of chunks for fallback
            
        Returns:
            List of method names for each junction
        """
        if not text_chunks or len(text_chunks) < 2:
            # Fallback to legacy method
            fallback_method = ChunkCombiner._legacy_auto_select_method(text_length, chunk_count)
            return [fallback_method] * max(1, chunk_count - 1)
        
        import re
        junction_methods = []
        
        # Analyze each junction individually
        for i in range(len(text_chunks) - 1):
            current_chunk = text_chunks[i].strip()
            next_chunk = text_chunks[i + 1].strip()
            
            # Analyze this specific junction
            method = ChunkCombiner._analyze_single_junction(current_chunk, next_chunk)
            junction_methods.append(method)
        
        return junction_methods
    
    @staticmethod
    def _analyze_single_junction(current_chunk: str, next_chunk: str) -> str:
        """
        Analyze a single junction between two chunks to determine optimal method.
        
        Args:
            current_chunk: Text of the chunk before the junction
            next_chunk: Text of the chunk after the junction
            
        Returns:
            Optimal method for this specific junction
        """
        import re
        
        # Clean up chunks
        current = current_chunk.strip()
        next_chunk = next_chunk.strip()
        
        # Priority 1: Sentence boundary - natural pause
        if re.search(r'[.!?]\s*$', current):
            return "silence"
        
        # Priority 2: Paragraph break or dialogue change - longer pause
        if '\n' in current_chunk or current.endswith(':'):
            return "silence"
        
        # Priority 3: Short conversational responses - clarity
        if len(current) < 50 and re.search(r'^(yes|no|okay|ok|sure|hello|hi|thanks|thank you|hmm|ah|oh)\b', current.lower()):
            return "silence"
        
        # Priority 4: Current chunk ends with comma - smooth continuation
        if re.search(r',\s*$', current):
            return "crossfade"
        
        # Priority 5: Next chunk starts with conjunction - smooth flow
        if re.search(r'^(and|but|or|so|then|however|therefore|because|since|while|although)\b', next_chunk.lower()):
            return "crossfade"
        
        # Priority 6: Forced character split (no natural boundary) - smooth over artificial break
        if not re.search(r'[.!?,;:]\s*$', current):
            return "crossfade"
        
        # Priority 7: Both chunks are very short - likely natural speech units
        if len(current) < 100 and len(next_chunk) < 100:
            return "concatenate"
        
        # Default: Use crossfade for smooth transitions
        return "crossfade"
    
    @staticmethod
    def _get_dominant_method(junction_methods: List[str]) -> str:
        """
        Get the most commonly used method from junction analysis for display purposes.
        
        Args:
            junction_methods: List of methods for each junction
            
        Returns:
            Most common method name
        """
        if not junction_methods:
            return "concatenate"
        
        # Count method usage
        method_counts = {}
        for method in junction_methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Return most common method
        return max(method_counts.items(), key=lambda x: x[1])[0]
    
    @staticmethod
    def _combine_with_junction_methods(audio_segments: List[torch.Tensor], 
                                      junction_methods: List[str],
                                      silence_ms: int,
                                      crossfade_duration: float,
                                      sample_rate: int) -> torch.Tensor:
        """
        Combine audio segments using different methods for each junction.
        
        Args:
            audio_segments: List of audio tensors to combine
            junction_methods: List of methods for each junction
            silence_ms: Silence duration in milliseconds
            crossfade_duration: Crossfade duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            Combined audio tensor
        """
        if not audio_segments:
            return torch.empty(0)
        
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        # Normalize all segments to consistent format
        normalized_segments = []
        for segment in audio_segments:
            if segment.dim() == 1:
                segment = segment.unsqueeze(0)  # [samples] -> [1, samples]
            elif segment.dim() == 2 and segment.shape[0] > segment.shape[1]:
                segment = segment.transpose(0, 1)  # [samples, channels] -> [channels, samples]
            normalized_segments.append(segment)
        
        # Start with first segment
        result = normalized_segments[0]
        
        # Add each subsequent segment with its specific junction method
        for i in range(1, len(normalized_segments)):
            junction_method = junction_methods[i - 1] if i - 1 < len(junction_methods) else "concatenate"
            next_segment = normalized_segments[i]
            
            result = ChunkCombiner._apply_junction_method(
                result, next_segment, junction_method, silence_ms, crossfade_duration, sample_rate
            )
        
        return result
    
    @staticmethod
    def _apply_junction_method(left_audio: torch.Tensor, 
                              right_audio: torch.Tensor,
                              method: str,
                              silence_ms: int,
                              crossfade_duration: float,
                              sample_rate: int) -> torch.Tensor:
        """
        Apply a specific combination method to join two audio segments.
        
        Args:
            left_audio: Left audio segment
            right_audio: Right audio segment  
            method: Junction method ("concatenate", "silence", "crossfade")
            silence_ms: Silence duration in milliseconds
            crossfade_duration: Crossfade duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            Combined audio tensor
        """
        # Handle aliases
        if method == "silence_padding":
            method = "silence"
        
        if method == "concatenate":
            return torch.cat([left_audio, right_audio], dim=-1)
        
        elif method == "silence":
            if silence_ms > 0:
                silence_duration = silence_ms / 1000.0
                silence = AudioProcessingUtils.create_silence(
                    silence_duration, sample_rate,
                    channels=left_audio.shape[0],
                    device=left_audio.device,
                    dtype=left_audio.dtype
                )
                
                # Ensure silence has same dimensions as audio segments
                if silence.dim() != left_audio.dim():
                    if silence.dim() == 1 and left_audio.dim() == 2:
                        silence = silence.unsqueeze(0)  # [samples] -> [1, samples]
                    elif silence.dim() == 2 and left_audio.dim() == 1:
                        silence = silence.squeeze(0)   # [1, samples] -> [samples]
                
                return torch.cat([left_audio, silence, right_audio], dim=-1)
            else:
                return torch.cat([left_audio, right_audio], dim=-1)
        
        elif method == "crossfade":
            # Apply crossfade between segments
            return ChunkCombiner._apply_crossfade(left_audio, right_audio, crossfade_duration, sample_rate)
        
        else:
            # Unknown method, fallback to concatenate
            return torch.cat([left_audio, right_audio], dim=-1)
    
    @staticmethod  
    def _apply_crossfade(left_audio: torch.Tensor, 
                        right_audio: torch.Tensor,
                        crossfade_duration: float,
                        sample_rate: int) -> torch.Tensor:
        """
        Apply crossfade transition between two audio segments.
        
        Args:
            left_audio: Left audio segment [channels, samples]
            right_audio: Right audio segment [channels, samples]
            crossfade_duration: Crossfade duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            Crossfaded audio tensor
        """
        crossfade_samples = int(crossfade_duration * sample_rate)
        
        # Ensure we don't crossfade more than available audio
        left_samples = left_audio.shape[-1]
        right_samples = right_audio.shape[-1]
        crossfade_samples = min(crossfade_samples, left_samples, right_samples)
        
        if crossfade_samples <= 0:
            # No crossfade possible, just concatenate
            return torch.cat([left_audio, right_audio], dim=-1)
        
        # Extract crossfade regions
        left_fade_out = left_audio[..., -crossfade_samples:]  # End of left
        right_fade_in = right_audio[..., :crossfade_samples]  # Start of right
        
        # Create fade curves
        fade_out_curve = torch.linspace(1.0, 0.0, crossfade_samples, device=left_audio.device, dtype=left_audio.dtype)
        fade_in_curve = torch.linspace(0.0, 1.0, crossfade_samples, device=right_audio.device, dtype=right_audio.dtype)
        
        # Apply fades and mix
        faded_left = left_fade_out * fade_out_curve
        faded_right = right_fade_in * fade_in_curve
        crossfaded_region = faded_left + faded_right
        
        # Combine: left_prefix + crossfaded_region + right_suffix
        left_prefix = left_audio[..., :-crossfade_samples] if crossfade_samples < left_samples else torch.empty_like(left_audio)[..., :0]
        right_suffix = right_audio[..., crossfade_samples:] if crossfade_samples < right_samples else torch.empty_like(right_audio)[..., :0]
        
        return torch.cat([left_prefix, crossfaded_region, right_suffix], dim=-1)
    
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
        return ["auto", "concatenate", "silence", "silence_padding", "crossfade"]
    
    @staticmethod
    def get_method_description(method: str) -> str:
        """Get description of combination method."""
        descriptions = {
            "auto": "Smart per-junction analysis (sentence boundaries, commas, conjunctions, forced splits, etc.)",
            "concatenate": "Direct joining with no gap or transition", 
            "silence": "Add silence padding between chunks for clarity",
            "silence_padding": "Add silence padding between chunks for clarity",  # Alias for silence
            "crossfade": "Smooth crossfade transitions between chunks"
        }
        return descriptions.get(method, "Unknown method")
    
    @staticmethod
    def format_timing_info(chunk_info: dict) -> str:
        """Format chunk timing information for generation info display with per-junction details."""
        if not chunk_info:
            return ""
        
        method = chunk_info.get("method_used", "unknown")
        total_chunks = chunk_info.get("total_chunks", 0)
        auto_selected = chunk_info.get("auto_selected", False)
        junction_methods = chunk_info.get("junction_methods", [])
        junction_details = chunk_info.get("junction_details", [])
        
        if total_chunks <= 1:
            return f"Single chunk, no combination needed"
        
        # Method info - show if per-junction analysis was used
        if auto_selected and junction_methods:
            # Count different methods used
            method_counts = {}
            for jm in junction_methods:
                method_counts[jm] = method_counts.get(jm, 0) + 1
            
            if len(method_counts) == 1:
                # All junctions use same method
                method_str = f"'{method}' (auto-analyzed, consistent across all junctions)"
            else:
                # Mixed methods per junction
                method_breakdown = ", ".join([f"{count}x {method}" for method, count in method_counts.items()])
                method_str = f"per-junction analysis ({method_breakdown})"
        else:
            method_str = f"'{method}'"
            if auto_selected:
                method_str += " (auto-selected)"
        
        # Timing breakdown with junction details
        timings = chunk_info.get("chunk_timings", [])
        timing_lines = []
        
        for i, timing in enumerate(timings):
            start = timing.get("start", 0)
            end = timing.get("end", 0)
            text = timing.get("text", f"Chunk {i+1}")
            # Truncate long text for display
            display_text = text[:40] + "..." if len(text) > 40 else text
            timing_line = f"  {start:.1f}s-{end:.1f}s: {display_text}"
            
            # Add junction method info if available
            if i < len(junction_details):
                junction = junction_details[i]
                junction_method = junction.get("method", "unknown")
                timing_line += f" → [{junction_method}]"
            
            timing_lines.append(timing_line)
        
        result = f"Combined {total_chunks} chunks using {method_str}\n"
        result += "\n".join(timing_lines)
        
        # Add global method-specific details
        if method in ["silence", "silence_padding"]:
            silence_ms = chunk_info.get("silence_ms", 0)
            result += f"\n  Default silence padding: {silence_ms}ms"
        elif method == "crossfade":
            crossfade_duration = chunk_info.get("crossfade_duration", 0)
            result += f"\n  Default crossfade duration: {crossfade_duration:.1f}s"
        
        # Add per-junction summary if mixed methods
        if auto_selected and junction_methods and len(set(junction_methods)) > 1:
            result += f"\n  Junction analysis: Smart per-junction method selection based on text patterns"
        
        return result