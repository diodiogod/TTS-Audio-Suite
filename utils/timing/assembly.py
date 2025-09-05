"""
SRT Audio Assembly - Handles audio segment assembly and time-stretching
Extracted from the massive SRT TTS node for better maintainability
"""

import torch
import numpy as np
import tempfile
import os
from typing import Dict, Any, Optional, List, Tuple
from utils.audio.processing import AudioProcessingUtils


class AudioAssemblyEngine:
    """
    Handles assembly of audio segments with different timing modes
    Integrates with existing ChatterBox audio timing utilities
    """
    
    def __init__(self, sample_rate: int):
        """
        Initialize audio assembly engine
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.stretcher_cache = {}  # Cache for time stretching instances
        
    def assemble_concatenation(self, audio_segments: List[torch.Tensor],
                             fade_duration: float = 0.0) -> torch.Tensor:
        """
        Concatenate audio segments naturally with optional crossfading
        
        Args:
            audio_segments: List of audio tensors to concatenate
            fade_duration: Duration for crossfading between segments (seconds)
            
        Returns:
            Concatenated audio tensor
        """
        if not audio_segments:
            return torch.empty((1, 0))
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        if fade_duration > 0.0:
            return self._concatenate_with_crossfade(audio_segments, fade_duration)
        else:
            # Simple concatenation - handle tensor dimensions properly
            if not audio_segments:
                return torch.empty(0)
            
            # Filter out empty segments first
            non_empty_segments = [seg for seg in audio_segments if seg.numel() > 0]
            if not non_empty_segments:
                return torch.empty(0)
            
            # Determine target dimension from first non-empty segment
            first_segment = non_empty_segments[0]
            target_dim = first_segment.dim()
            
            # Normalize all segments to same dimension
            normalized_segments = []
            for segment in non_empty_segments:
                if segment.dim() == target_dim:
                    normalized_segments.append(segment)
                elif target_dim == 1 and segment.dim() == 2:
                    # Convert 2D to 1D (average channels if multi-channel)
                    if segment.shape[0] > 1:
                        normalized_segments.append(torch.mean(segment, dim=0))
                    else:
                        normalized_segments.append(segment.squeeze(0))
                elif target_dim == 2 and segment.dim() == 1:
                    # Convert 1D to 2D
                    normalized_segment = segment.unsqueeze(0)
                    if first_segment.shape[0] > 1:
                        normalized_segment = normalized_segment.repeat(first_segment.shape[0], 1)
                    normalized_segments.append(normalized_segment)
            
            # Concatenate on appropriate dimension
            if target_dim == 1:
                return torch.cat(normalized_segments, dim=0)
            else:
                return torch.cat(normalized_segments, dim=1)
    
    def _concatenate_with_crossfade(self, audio_segments: List[torch.Tensor], 
                                  fade_duration: float) -> torch.Tensor:
        """Concatenate audio segments with crossfading between them"""
        # Filter out empty segments first
        non_empty_segments = [seg for seg in audio_segments if seg.numel() > 0]
        if not non_empty_segments:
            return torch.empty(0)
        if len(non_empty_segments) == 1:
            return non_empty_segments[0]
        
        fade_samples = int(fade_duration * self.sample_rate)
        result = non_empty_segments[0]
        
        for i in range(1, len(non_empty_segments)):
            current_segment = non_empty_segments[i]
            
            # Handle tensor dimensions properly - check last dimension regardless of tensor shape
            result_samples = result.size(-1)
            current_samples = current_segment.size(-1)
            
            if fade_samples > 0 and result_samples >= fade_samples and current_samples >= fade_samples:
                # Create fade out for end of previous segment
                fade_out = torch.linspace(1.0, 0.0, fade_samples, device=result.device)
                fade_in = torch.linspace(0.0, 1.0, fade_samples, device=current_segment.device)
                
                # Normalize both tensors to same dimension before applying fades
                if result.dim() != current_segment.dim():
                    if result.dim() == 1 and current_segment.dim() == 2:
                        # Convert current_segment to 1D (average channels if multi-channel)
                        if current_segment.shape[0] > 1:
                            current_segment = torch.mean(current_segment, dim=0)
                        else:
                            current_segment = current_segment.squeeze(0)
                    elif result.dim() == 2 and current_segment.dim() == 1:
                        # Convert current_segment to 2D to match result
                        current_segment = current_segment.unsqueeze(0)
                        if result.shape[0] > 1:
                            current_segment = current_segment.repeat(result.shape[0], 1)
                
                # Apply fades based on result tensor dimensions (both are now same dimension)
                if result.dim() == 1:
                    # 1D tensors (mono)
                    result_end = result[-fade_samples:] * fade_out
                    current_start = current_segment[:fade_samples] * fade_in
                    
                    # Crossfade by overlapping
                    crossfaded = result_end + current_start
                    
                    # Combine: previous audio (minus fade region) + crossfaded region + rest of current segment
                    result = torch.cat([
                        result[:-fade_samples],
                        crossfaded,
                        current_segment[fade_samples:]
                    ], dim=0)
                else:
                    # 2D tensors (multi-channel)
                    result_end = result[:, -fade_samples:] * fade_out
                    current_start = current_segment[:, :fade_samples] * fade_in
                    
                    # Crossfade by overlapping
                    crossfaded = result_end + current_start
                    
                    # Combine: previous audio (minus fade region) + crossfaded region + rest of current segment
                    result = torch.cat([
                        result[:, :-fade_samples],
                        crossfaded,
                        current_segment[:, fade_samples:]
                    ], dim=1)
            else:
                # No crossfading, just concatenate on the appropriate dimension
                if result.dim() == 1 and current_segment.dim() == 1:
                    result = torch.cat([result, current_segment], dim=0)
                elif result.dim() == 2 and current_segment.dim() == 2:
                    result = torch.cat([result, current_segment], dim=1)
                else:
                    # Mixed dimensions - normalize to match result's dimensions
                    if result.dim() == 1 and current_segment.dim() == 2:
                        # Convert current_segment to 1D (average channels if multi-channel)
                        if current_segment.shape[0] > 1:
                            current_segment = torch.mean(current_segment, dim=0)
                        else:
                            current_segment = current_segment.squeeze(0)
                        result = torch.cat([result, current_segment], dim=0)
                    elif result.dim() == 2 and current_segment.dim() == 1:
                        # Convert current_segment to 2D to match result
                        current_segment = current_segment.unsqueeze(0)
                        if result.shape[0] > 1:
                            current_segment = current_segment.repeat(result.shape[0], 1)
                        result = torch.cat([result, current_segment], dim=1)
        
        return result
        
    def assemble_stretch_to_fit(self, audio_segments: List[torch.Tensor],
                               target_timings: List[Tuple[float, float]],
                               fade_duration: float = 0.01) -> torch.Tensor:
        """
        Assemble audio segments using stretch-to-fit mode - ORIGINAL BEHAVIOR
        Uses the original TimedAudioAssembler from chatterbox
        """
        if len(audio_segments) != len(target_timings):
            raise ValueError("Number of segments must match number of timings")
        
        if not audio_segments:
            return torch.empty(0)
        
        # Use the original TimedAudioAssembler from chatterbox - EXACT ORIGINAL BEHAVIOR
        try:
            from engines.chatterbox.audio_timing import TimedAudioAssembler
            
            assembler = TimedAudioAssembler(self.sample_rate)
            final_audio = assembler.assemble_timed_audio(
                audio_segments, target_timings, fade_duration=fade_duration
            )
            return final_audio
            
        except ImportError as e:
            # Fallback to basic assembly if chatterbox modules not available
            return self._basic_stretch_assembly(audio_segments, target_timings, fade_duration)
    
    def assemble_with_overlaps(self, audio_segments: List[torch.Tensor],
                              subtitles: List, device) -> torch.Tensor:
        """
        ORIGINAL: Assemble audio by placing segments at their SRT start times, allowing audible overlaps.
        Silence is implicitly added in gaps. EXACT COPY FROM ORIGINAL LINES 1776-1875
        """
        if not audio_segments:
            return torch.empty(0) # Return empty tensor if no segments

        # Use device directly for all operations within this function.
        # device is the consistent target device (e.g., 'cuda' or 'cpu')
        # established during model loading.
        target_device = device

        # Determine output buffer properties (num_channels, dtype) from the first segment,
        # ensuring it's on the target_device first.
        # audio_segments (which is normalized_segments) should already be on device.
        _first_segment_for_props = audio_segments[0].to(target_device)
        num_channels = _first_segment_for_props.shape[0] if _first_segment_for_props.dim() == 2 else 1
        dtype = _first_segment_for_props.dtype

        # Calculate total duration needed for the output buffer
        max_end_time = 0.0
        for i, (segment_from_list, subtitle) in enumerate(zip(audio_segments, subtitles)):
            # Ensure segment is on target_device for size calculation
            segment_on_target = segment_from_list.to(target_device)
            segment_end_time = subtitle.start_time + (segment_on_target.size(-1) / self.sample_rate)
            max_end_time = max(max_end_time, segment_end_time)
        
        # Ensure the buffer is at least as long as the last subtitle's end time
        if subtitles:
            max_end_time = max(max_end_time, subtitles[-1].end_time)

        total_samples = int(max_end_time * self.sample_rate)

        # Initialize output buffer with zeros
        if num_channels == 1:
            output_audio = torch.zeros(total_samples, device=target_device, dtype=dtype)
        else:
            output_audio = torch.zeros(num_channels, total_samples, device=target_device, dtype=dtype)

        for i, (original_segment_from_list, subtitle) in enumerate(zip(audio_segments, subtitles)):
            # Process segment

            # Explicitly move the current segment to target_device before any processing
            current_processing_segment = original_segment_from_list.to(target_device)

            # Ensure normalized_audio matches the channel dimension of output_audio
            # current_processing_segment is now guaranteed to be on target_device.
            normalized_audio_for_add = current_processing_segment

            if num_channels == 1: # Output is mono (1D)
                if normalized_audio_for_add.dim() == 2:
                    # If segment is 2D (e.g., [1, samples] or [channels, samples]), squeeze to 1D
                    if normalized_audio_for_add.shape[0] == 1: # If it's [1, samples], just squeeze
                        normalized_audio_for_add = normalized_audio_for_add.squeeze(0)
                    else: # If it's multi-channel, sum to mono
                        normalized_audio_for_add = torch.sum(normalized_audio_for_add, dim=0)
                # If it's already 1D, no change needed
            else: # Output is stereo/multi-channel (2D)
                if normalized_audio_for_add.dim() == 1:
                    # If segment is mono (1D), expand to 2D and repeat channels
                    normalized_audio_for_add = normalized_audio_for_add.unsqueeze(0).repeat(num_channels, 1)
                elif normalized_audio_for_add.dim() == 2 and normalized_audio_for_add.shape[0] != num_channels:
                    # If segment is 2D but has wrong channel count, raise error
                    raise RuntimeError(f"Channel mismatch: output buffer has {num_channels} channels, but segment {i} has {normalized_audio_for_add.shape[0]} channels.")
            
            # Segment normalized
            # At this point, normalized_audio_for_add is on target_device and has the correct dimensions for addition.

            start_sample = int(subtitle.start_time * self.sample_rate)
            end_sample_segment = start_sample + normalized_audio_for_add.size(-1)

            # Resize output_audio if current segment extends beyond current buffer size
            if end_sample_segment > output_audio.size(-1):
                new_total_samples = end_sample_segment
                if num_channels == 1:
                    new_output_audio = torch.zeros(new_total_samples, device=target_device, dtype=dtype)
                else:
                    new_output_audio = torch.zeros(num_channels, new_total_samples, device=target_device, dtype=dtype)
                
                # Copy existing audio to the new larger buffer
                current_len = output_audio.size(-1)
                if num_channels == 1:
                    new_output_audio[:current_len] = output_audio
                else:
                    new_output_audio[:, :current_len] = output_audio
                output_audio = new_output_audio
                # Buffer resized

            # Both output_audio and normalized_audio_for_add are on target_device.
            if output_audio.dim() == 1:
                output_audio[start_sample:end_sample_segment] += normalized_audio_for_add
            else:
                output_audio[:, start_sample:end_sample_segment] += normalized_audio_for_add
            
            # Segment placed

        # Assembly complete
        return output_audio
    
    def assemble_smart_natural(self, audio_segments: List[torch.Tensor],
                              processed_segments: List[torch.Tensor],
                              adjustments: List[Dict], subtitles: List, device: str) -> torch.Tensor:
        """
        Assemble audio using smart natural timing with processed segments
        ORIGINAL FINAL ASSEMBLY LOGIC FROM LINES 1703-1774
        """
        if not processed_segments:
            return torch.empty(0)
        
        # Import required for assembly using import manager
        try:
            from utils.system.import_manager import import_manager
            success, modules, source = import_manager.import_srt_modules()
            if not success:
                raise ImportError("SRT modules not available")
            
            AudioTimingUtils = modules.get("AudioTimingUtils")
            SRTSubtitle = modules.get("SRTSubtitle")
            
            if not all([AudioTimingUtils, SRTSubtitle]):
                raise ImportError("Required SRT modules not found")
        except ImportError:
            # Fallback to simple concatenation
            return AudioProcessingUtils.concatenate_audio_segments(processed_segments, "simple")
        
        # ORIGINAL FINAL ASSEMBLY LOGIC - Extract mutable_subtitles from adjustments
        # Since we don't have access to the mutable_subtitles from timing engine,
        # we'll reconstruct the final timing from the adjustments
        
        final_audio_parts = []
        current_output_time = 0.0
        
        for i, segment_audio in enumerate(processed_segments):
            # Get timing info from adjustments
            adj = adjustments[i] if i < len(adjustments) else {}
            
            # Determine start time for this segment
            segment_start_time = adj.get('final_srt_start', subtitles[i].start_time if i < len(subtitles) else current_output_time)
            
            # Add silence if there's a gap between current_output_time and segment's start_time
            if current_output_time < segment_start_time:
                gap_duration = segment_start_time - current_output_time
                # Add silence gap
                silence = AudioTimingUtils.create_silence(gap_duration, self.sample_rate,
                                                         channels=segment_audio.shape[0] if segment_audio.dim() == 2 else 1,
                                                         device=segment_audio.device)
                final_audio_parts.append(silence)
                current_output_time += gap_duration
            
            final_audio_parts.append(segment_audio)
            current_output_time += AudioTimingUtils.get_audio_duration(segment_audio, self.sample_rate)
        
        if not final_audio_parts:
            # Handle case where no segments were processed (e.g., immediate interruption)
            if processed_segments:
                return torch.empty(0, device=processed_segments[0].device, dtype=processed_segments[0].dtype)
            else:
                return torch.empty(0)
        
        # ORIGINAL DIMENSION NORMALIZATION LOGIC
        # Ensure all parts have the same number of dimensions before concatenating
        target_dim = processed_segments[0].dim() if processed_segments else 1
        target_channels = processed_segments[0].shape[0] if target_dim == 2 else 1
        
        normalized_final_audio_parts = []
        target_device_for_concat = device  # Use passed device parameter
        
        for part in final_audio_parts:
            # Ensure each part is on the target_device before normalization
            part_on_device = part.to(target_device_for_concat)
            
            processed_part_for_append = None
            current_part_dim = part_on_device.dim()
            
            if current_part_dim == target_dim:
                if target_dim == 2 and part_on_device.shape[0] != target_channels:
                    if part_on_device.shape[0] == 1:
                        processed_part_for_append = part_on_device.repeat(target_channels, 1)
                    else:
                        # This branch raises an error, so no tensor is assigned here
                        raise RuntimeError(f"Channel mismatch in final assembly: Expected {target_channels} channels, got {part_on_device.shape[0]}")
                else:
                    # Part is already on target_device_for_concat and correctly shaped
                    processed_part_for_append = part_on_device
            elif current_part_dim == 1 and target_dim == 2: # Mono part, target is multi-channel
                processed_part_for_append = part_on_device.unsqueeze(0).repeat(target_channels, 1)
            elif current_part_dim == 2 and target_dim == 1: # Multi-channel part, target is mono
                processed_part_for_append = torch.sum(part_on_device, dim=0)
            else:
                # This branch raises an error
                raise RuntimeError(f"Dimension mismatch in final assembly: Expected {target_dim}D, got {current_part_dim}D for part with shape {part_on_device.shape}")
            
            # Ensure the final tensor to be appended is on the target_device_for_concat.
            if processed_part_for_append is not None:
                 normalized_final_audio_parts.append(processed_part_for_append.to(target_device_for_concat))
        
        # All parts in normalized_final_audio_parts are on the target device
        return torch.cat(normalized_final_audio_parts, dim=-1)
    
    def apply_time_stretching(self, audio: torch.Tensor, 
                            stretch_factor: float,
                            method: str = "auto") -> torch.Tensor:
        """
        Apply time stretching to audio segment
        
        Args:
            audio: Input audio tensor
            stretch_factor: Stretching factor (>1 = slower, <1 = faster)
            method: Stretching method ("auto", "ffmpeg", "phase_vocoder")
            
        Returns:
            Time-stretched audio tensor
        """
        if abs(stretch_factor - 1.0) < 0.01:  # 1% tolerance
            return audio
        
        try:
            # Try to use chatterbox audio timing utilities
            if method == "auto" or method == "ffmpeg":
                from engines.chatterbox.audio_timing import FFmpegTimeStretcher
                
                cache_key = f"ffmpeg_{self.sample_rate}"
                if cache_key not in self.stretcher_cache:
                    try:
                        self.stretcher_cache[cache_key] = FFmpegTimeStretcher()
                    except Exception:
                        # Fall back to phase vocoder if FFmpeg not available
                        cache_key = f"phase_vocoder_{self.sample_rate}"
                        if cache_key not in self.stretcher_cache:
                            from engines.chatterbox.audio_timing import PhaseVocoderTimeStretcher
                            self.stretcher_cache[cache_key] = PhaseVocoderTimeStretcher()
                
                stretcher = self.stretcher_cache[cache_key]
                return stretcher.time_stretch(audio, stretch_factor, self.sample_rate)
            
            elif method == "phase_vocoder":
                from engines.chatterbox.audio_timing import PhaseVocoderTimeStretcher
                
                cache_key = f"phase_vocoder_{self.sample_rate}"
                if cache_key not in self.stretcher_cache:
                    self.stretcher_cache[cache_key] = PhaseVocoderTimeStretcher()
                
                stretcher = self.stretcher_cache[cache_key]
                return stretcher.time_stretch(audio, stretch_factor, self.sample_rate)
            
            else:
                raise ValueError(f"Unknown time stretching method: {method}")
                
        except ImportError:
            # Fallback to simple interpolation if chatterbox modules not available
            return self._simple_time_stretch(audio, stretch_factor)
    
    def create_silence(self, duration_seconds: float, 
                      channels: int = 1, 
                      device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Create silence tensor of specified duration
        
        Args:
            duration_seconds: Duration in seconds
            channels: Number of audio channels
            device: Target device for tensor
            
        Returns:
            Silence tensor
        """
        try:
            # Use chatterbox utilities if available
            from engines.chatterbox.audio_timing import AudioTimingUtils
            return AudioTimingUtils.create_silence(
                duration_seconds, self.sample_rate, channels, device
            )
        except ImportError:
            # Fallback implementation
            num_samples = int(duration_seconds * self.sample_rate)
            if channels == 1:
                return torch.zeros(num_samples, device=device)
            else:
                return torch.zeros(channels, num_samples, device=device)
    
    def _get_audio_duration(self, audio: torch.Tensor) -> float:
        """Get duration of audio tensor in seconds"""
        try:
            from engines.chatterbox.audio_timing import AudioTimingUtils
            return AudioTimingUtils.get_audio_duration(audio, self.sample_rate)
        except ImportError:
            # Fallback implementation
            if audio.dim() == 1:
                return audio.size(0) / self.sample_rate
            elif audio.dim() == 2:
                return audio.size(-1) / self.sample_rate
            else:
                raise ValueError(f"Unsupported audio tensor dimensions: {audio.dim()}")
    
    def _basic_stretch_assembly(self, audio_segments: List[torch.Tensor], 
                               target_timings: List[Tuple[float, float]],
                               fade_duration: float) -> torch.Tensor:
        """
        Basic stretch-to-fit assembly fallback implementation
        """
        if not audio_segments:
            return torch.empty(0)
        
        total_duration = max(end_time for _, end_time in target_timings)
        total_samples = int(total_duration * self.sample_rate)
        
        # Create output buffer
        first_segment = audio_segments[0]
        if first_segment.dim() == 1:
            output = torch.zeros(total_samples, device=first_segment.device, dtype=first_segment.dtype)
        else:
            output = torch.zeros(first_segment.size(0), total_samples, 
                               device=first_segment.device, dtype=first_segment.dtype)
        
        # Process each segment
        for segment, (start_time, end_time) in zip(audio_segments, target_timings):
            target_duration = end_time - start_time
            current_duration = self._get_audio_duration(segment)
            
            # Apply time stretching if needed
            if abs(current_duration - target_duration) > 0.01:
                stretch_factor = target_duration / current_duration if current_duration > 0 else 1.0
                segment = self._simple_time_stretch(segment, stretch_factor)
            
            # Place segment in output
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + segment.size(-1)
            
            if output.dim() == 1:
                output[start_sample:min(end_sample, output.size(0))] = segment[:min(segment.size(0), output.size(0) - start_sample)]
            else:
                output[:, start_sample:min(end_sample, output.size(-1))] = segment[:, :min(segment.size(-1), output.size(-1) - start_sample)]
        
        return output
    
    def _simple_time_stretch(self, audio: torch.Tensor, stretch_factor: float) -> torch.Tensor:
        """
        Simple time stretching using interpolation (fallback method)
        """
        if abs(stretch_factor - 1.0) < 0.01:
            return audio
        
        original_shape = audio.shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        stretched_channels = []
        for channel_idx in range(audio.size(0)):
            channel_audio = audio[channel_idx].cpu().numpy()
            
            original_length = len(channel_audio)
            new_length = int(original_length * stretch_factor)
            
            # Create new time indices and interpolate
            old_indices = np.linspace(0, original_length - 1, original_length)
            new_indices = np.linspace(0, original_length - 1, new_length)
            stretched = np.interp(new_indices, old_indices, channel_audio)
            
            stretched_channels.append(torch.from_numpy(stretched))
        
        # Combine channels and restore original shape
        result = torch.stack(stretched_channels, dim=0).to(audio.device)
        if len(original_shape) == 1:
            result = result.squeeze(0)
        
        return result
    
    def get_assembly_info(self) -> Dict[str, Any]:
        """
        Get information about the assembly engine configuration
        
        Returns:
            Dictionary with assembly engine info
        """
        return {
            'sample_rate': self.sample_rate,
            'cached_stretchers': list(self.stretcher_cache.keys()),
            'available_methods': self._get_available_methods()
        }
    
    def _get_available_methods(self) -> List[str]:
        """Get list of available time stretching methods"""
        methods = ['simple_interpolation']  # Always available fallback
        
        try:
            from engines.chatterbox.audio_timing import FFmpegTimeStretcher
            FFmpegTimeStretcher()  # Test if FFmpeg is available
            methods.append('ffmpeg')
        except Exception:
            pass
        
        try:
            from engines.chatterbox.audio_timing import PhaseVocoderTimeStretcher
            methods.append('phase_vocoder')
        except Exception:
            pass
        
        return methods
    
    def assemble_by_timing_mode(self, audio_segments: List[torch.Tensor], 
                               subtitles: List, timing_mode: str, device,
                               adjustments: Optional[List[Dict]] = None,
                               processed_segments: Optional[List[torch.Tensor]] = None,
                               fade_duration: float = 0.01) -> torch.Tensor:
        """
        Route to the correct assembly method based on timing mode
        
        Args:
            audio_segments: Raw audio segments
            subtitles: SRT subtitle objects
            timing_mode: The timing mode to use
            device: Target device
            adjustments: Timing adjustments (for smart_natural mode)
            processed_segments: Processed segments (for smart_natural mode)  
            fade_duration: Fade duration for stretch modes
            
        Returns:
            Assembled audio tensor
        """
        if timing_mode == "pad_with_silence":
            # Use overlap assembly - places audio at SRT start times with overlaps allowed
            print(f"🔧 Assembly: Using overlap assembly for {timing_mode} mode")
            return self.assemble_with_overlaps(audio_segments, subtitles, device)
            
        elif timing_mode == "smart_natural":
            # Use smart natural assembly with processed segments and adjustments
            if processed_segments is None or adjustments is None:
                raise ValueError("smart_natural mode requires processed_segments and adjustments")
            print(f"🔧 Assembly: Using smart natural assembly for {timing_mode} mode")
            return self.assemble_smart_natural(audio_segments, processed_segments, adjustments, subtitles, device)
            
        elif timing_mode == "stretch_to_fit":
            # Use stretch-to-fit assembly
            target_timings = [(sub.start_time, sub.end_time) for sub in subtitles]
            print(f"🔧 Assembly: Using stretch-to-fit assembly for {timing_mode} mode")
            return self.assemble_stretch_to_fit(audio_segments, target_timings, fade_duration)
            
        elif timing_mode == "concatenate":
            # Use simple concatenation
            print(f"🔧 Assembly: Using concatenation assembly for {timing_mode} mode")
            return self.assemble_concatenation(audio_segments, fade_duration)
            
        else:
            raise ValueError(f"Unknown timing mode: {timing_mode}")