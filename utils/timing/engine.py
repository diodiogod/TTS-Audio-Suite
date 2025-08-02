"""
SRT Timing Engine - Handles complex timing calculations and adjustments
Extracted from the massive SRT TTS node for better maintainability
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from utils.audio.processing import AudioProcessingUtils


class TimingEngine:
    """
    Advanced timing engine for SRT audio synchronization
    Handles smart timing calculations, adjustments, and optimizations
    """
    
    def __init__(self, sample_rate: int):
        """
        Initialize timing engine
        
        Args:
            sample_rate: Audio sample rate for calculations
        """
        self.sample_rate = sample_rate
        
    def calculate_concatenation_adjustments(self, audio_segments: List[torch.Tensor],
                                          subtitles: List) -> List[Dict]:
        """
        Calculate timing adjustments for concatenation mode
        Ignores original SRT timings and creates new sequential timings
        """
        adjustments = []
        current_time = 0.0
        
        for i, (audio_segment, subtitle) in enumerate(zip(audio_segments, subtitles)):
            # Get natural duration of this segment
            natural_duration = self._get_audio_duration(audio_segment)
            
            # Calculate new timing for this segment
            start_time = current_time
            end_time = current_time + natural_duration
            
            # Create adjustment record
            adjustment = {
                'sequence': subtitle.sequence,
                'start_time': start_time,
                'end_time': end_time,
                'natural_duration': natural_duration,
                'original_text': subtitle.text,
                'original_srt_start': subtitle.start_time,
                'original_srt_end': subtitle.end_time,
                'original_srt_duration': subtitle.duration,
                'timing_change': end_time - subtitle.end_time,  # How much timing changed
                'needs_stretching': False,  # No stretching in concatenate mode
                'mode': 'concatenate'
            }
            adjustments.append(adjustment)
            
            # Update current time for next segment
            current_time = end_time
        
        return adjustments
        
    def calculate_smart_timing_adjustments(self, audio_segments: List[torch.Tensor],
                                         subtitles: List, tolerance: float,
                                         max_stretch_ratio: float,
                                         min_stretch_ratio: float, device: str) -> Tuple[List[Dict], List[torch.Tensor]]:
        """
        ORIGINAL Smart balanced timing: Adjusts SRT segment timings based on actual spoken duration
        and a user-defined timing_tolerance - EXACT ORIGINAL IMPLEMENTATION FROM LINES 1497-1774
        """
        # Import required modules using the same pattern as SRT node
        try:
            # Use the import manager to get the modules
            from utils.system.import_manager import import_manager
            success, modules, source = import_manager.import_srt_modules()
            if not success:
                raise ImportError("SRT modules not available")
            
            AudioTimingUtils = modules.get("AudioTimingUtils")
            FFmpegTimeStretcher = modules.get("FFmpegTimeStretcher")
            PhaseVocoderTimeStretcher = modules.get("PhaseVocoderTimeStretcher")
            AudioTimingError = modules.get("AudioTimingError")
            SRTSubtitle = modules.get("SRTSubtitle")
            
            if not all([AudioTimingUtils, FFmpegTimeStretcher, PhaseVocoderTimeStretcher, AudioTimingError, SRTSubtitle]):
                raise ImportError("Required SRT modules not found")
                
            import comfy.model_management as model_management
        except ImportError:
            # Fallback if modules not available
            return self._fallback_smart_timing(audio_segments, subtitles)
        
        processed_segments = []
        smart_adjustments_report = []
        
        # Create a mutable copy of subtitles to adjust start/end times
        mutable_subtitles = [SRTSubtitle(s.sequence, s.start_time, s.end_time, s.text) for s in subtitles]
        
        # Initialize stretcher for smart_natural mode
        try:
            # Try FFmpeg first
            print("Smart natural mode: Trying FFmpeg stretcher...")
            time_stretcher = FFmpegTimeStretcher()
            self._smart_natural_stretcher = "ffmpeg"
            print("Smart natural mode: Using FFmpeg stretcher")
        except AudioTimingError as e:
            # Fall back to Phase Vocoder
            print(f"Smart natural mode: FFmpeg initialization failed ({str(e)}), falling back to Phase Vocoder")
            time_stretcher = PhaseVocoderTimeStretcher()
            self._smart_natural_stretcher = "phase_vocoder"
            print("Smart natural mode: Using Phase Vocoder stretcher")
        
        # Process audio with smart natural timing
        for i, audio in enumerate(audio_segments):
            # Check for interruption during smart natural processing
            if model_management.interrupt_processing:
                print(f"⚠️ Smart natural processing interrupted at segment {i+1}/{len(audio_segments)}")
                raise InterruptedError(f"Smart natural processing interrupted at segment {i+1}/{len(audio_segments)}")
                
            current_subtitle = mutable_subtitles[i]
            natural_duration = AudioTimingUtils.get_audio_duration(audio, self.sample_rate)
            
            original_srt_start = subtitles[i].start_time # Use original for reference
            original_srt_end = subtitles[i].end_time
            
            # Step 1: Check if natural duration fits within current SRT slot
            # This is the duration the SRT *originally* allocated for this segment
            initial_target_duration = original_srt_end - original_srt_start
            
            segment_report = {
                'segment_index': i,
                'sequence': current_subtitle.sequence,
                'original_srt_start': original_srt_start,
                'original_srt_end': original_srt_end,
                'original_srt_duration': initial_target_duration,
                'natural_audio_duration': natural_duration,
                'next_segment_shifted_by': 0.0,
                'stretch_factor_applied': 1.0,
                'padding_added': 0.0,
                'truncated_by': 0.0,
                'final_segment_duration': natural_duration, # Will be updated
                'final_srt_start': original_srt_start, # Will be updated
                'final_srt_end': original_srt_end, # Will be updated
                'original_text': subtitles[i].text, # Add original subtitle text
                'actions': []
            }
            
            # Process segment timing
            
            # Calculate how much extra time is needed for the natural audio
            time_needed_beyond_srt = natural_duration - initial_target_duration
            
            adjusted_current_segment_end = original_srt_end # This will be updated
            
            # Step 2 & 3: Adjust Next Segment Start (if needed)
            if time_needed_beyond_srt > 0: # Natural audio is longer than original SRT slot
                # Audio is longer than slot
                segment_report['actions'].append(f"Natural audio ({natural_duration:.3f}s) is longer than original SRT slot ({initial_target_duration:.3f}s) by {time_needed_beyond_srt:.3f}s.")
                
                if i + 1 < len(mutable_subtitles):
                    next_subtitle = mutable_subtitles[i+1]
                    original_next_srt_start = subtitles[i+1].start_time # Use original for reference
                    
                    # First, try to consume any existing gap to the next subtitle
                    existing_gap = original_next_srt_start - original_srt_end
                    if existing_gap > 0:
                        time_to_consume_from_gap = min(time_needed_beyond_srt, existing_gap)
                        time_needed_beyond_srt -= time_to_consume_from_gap
                        adjusted_current_segment_end += time_to_consume_from_gap
                        segment_report['actions'].append(f"Consumed {time_to_consume_from_gap:.3f}s from existing gap. Remaining excess: {time_needed_beyond_srt:.3f}s.")
                        # Gap consumed
                    
                    if time_needed_beyond_srt > 0: # Still need more time after consuming gap
                        next_natural_audio_duration = AudioTimingUtils.get_audio_duration(audio_segments[i+1], self.sample_rate)
                        
                        # Calculate "room" in the next segment: how much shorter its natural audio is than its SRT slot
                        next_segment_room = max(0.0, next_subtitle.duration - next_natural_audio_duration)
                        # Calculate available room
                        segment_report['actions'].append(f"Next segment (Seq {next_subtitle.sequence}) has {next_segment_room:.3f}s room.")

                        # How much can we shift the next subtitle without exceeding tolerance?
                        # This is the amount of time we can "borrow" from the next segment's start.
                        max_shift_allowed = min(tolerance, next_segment_room) # Only shift into its room, within tolerance
                        
                        # How much do we *want* to shift the next subtitle?
                        desired_shift = time_needed_beyond_srt
                        
                        actual_shift = min(desired_shift, max_shift_allowed)
                        
                        if actual_shift > 0:
                            # Shift the next subtitle's start and end times
                            next_subtitle.start_time += actual_shift
                            next_subtitle.end_time += actual_shift
                            adjusted_current_segment_end += actual_shift # Add to the already adjusted end
                            segment_report['next_segment_shifted_by'] = actual_shift
                            segment_report['actions'].append(f"Shifted next subtitle (Seq {next_subtitle.sequence}) by {actual_shift:.3f}s. New next SRT start: {next_subtitle.start_time:.3f}s.")
                            # Subtitle shifted
                        else:
                            segment_report['actions'].append("Cannot shift next subtitle within tolerance/available room.")
                            # Cannot shift subtitle
                    else: # No next subtitle or no excess after consuming gap
                        segment_report['actions'].append("No next subtitle to shift or excess consumed by gap.")
                        print("   No next subtitle to shift or excess consumed by gap.")
                else:
                    segment_report['actions'].append("No next subtitle to shift.")
                    print("   No next subtitle to shift.")
            
            # Step 4: Stretch/Shrink (if still needed)
            # The new target duration for the current segment is from its original start to its (potentially) adjusted end
            new_target_duration = adjusted_current_segment_end - original_srt_start
            
            # Calculate stretch factor needed to fit natural audio into the new target duration
            stretch_factor = new_target_duration / natural_duration if natural_duration > 0 else 1.0
            
            # Apply stretch factor limits based on max_stretch_ratio and min_stretch_ratio
            clamped_stretch_factor = max(min_stretch_ratio, min(max_stretch_ratio, stretch_factor))
            
            # Check if stretching is actually needed and if it's within acceptable limits
            if abs(clamped_stretch_factor - 1.0) > 0.01: # Apply stretch if deviation is more than 1%
                # Apply audio stretching
                segment_report['actions'].append(f"⏱️ Applying stretch/shrink: natural {natural_duration:.3f}s -> target {new_target_duration:.3f}s (factor: {clamped_stretch_factor:.3f}x).")
                segment_report['stretch_factor_applied'] = clamped_stretch_factor
                try:
                    stretched_audio = time_stretcher.time_stretch(audio, clamped_stretch_factor, self.sample_rate)
                    processed_audio = stretched_audio
                except Exception as e:
                    segment_report['actions'].append("Time stretching failed, using padding/truncation")
                    # Time stretching failed, use fallback
                    processed_audio = audio # Use original audio if stretching fails
            else:
                processed_audio = audio
                segment_report['actions'].append("No significant stretch/shrink needed.")
                # No stretching needed
            
            # Step 5: Pad with Silence (last resort) or Truncate
            final_processed_duration = AudioTimingUtils.get_audio_duration(processed_audio, self.sample_rate)
            
            if final_processed_duration < new_target_duration:
                padding_needed = new_target_duration - final_processed_duration
                if padding_needed > 0:
                    segment_report['padding_added'] = padding_needed
                    segment_report['actions'].append(f"Padding with {padding_needed:.3f}s silence to reach target duration.")
                    # Add silence padding
                    processed_audio = AudioTimingUtils.pad_audio_to_duration(processed_audio, new_target_duration, self.sample_rate, "end")
            elif final_processed_duration > new_target_duration:
                # Truncate if still too long
                truncated_by = final_processed_duration - new_target_duration
                segment_report['truncated_by'] = truncated_by
                
                # Define threshold for insignificant truncations (50ms)
                INSIGNIFICANT_TRUNCATION_THRESHOLD = 0.05
                
                if truncated_by > INSIGNIFICANT_TRUNCATION_THRESHOLD:
                    # Significant truncation - show warning emoji
                    segment_report['actions'].append(f"🚧 Truncating audio by {truncated_by:.3f}s.")
                else:
                    # Insignificant truncation - show without alarming emoji
                    segment_report['actions'].append(f"Fine-tuning audio duration (-{truncated_by:.3f}s for precision).")
                
                # Truncate audio
                target_samples = AudioTimingUtils.seconds_to_samples(new_target_duration, self.sample_rate)
                processed_audio = processed_audio[..., :target_samples]
            
            processed_segments.append(processed_audio)
            
            # Update the current subtitle's end time to reflect the final processed duration
            # This is important for calculating the gap to the *next* segment in the final assembly
            current_subtitle.end_time = original_srt_start + AudioTimingUtils.get_audio_duration(processed_audio, self.sample_rate)
            
            segment_report['final_segment_duration'] = AudioTimingUtils.get_audio_duration(processed_audio, self.sample_rate)
            segment_report['final_srt_start'] = current_subtitle.start_time
            segment_report['final_srt_end'] = current_subtitle.end_time
            
            smart_adjustments_report.append(segment_report)
        
        return smart_adjustments_report, processed_segments
    
    def _fallback_smart_timing(self, audio_segments: List[torch.Tensor], subtitles: List) -> Tuple[List[Dict], List[torch.Tensor]]:
        """Fallback when chatterbox modules not available - SHOULD NOT HAPPEN IN NORMAL OPERATION"""
        # This should not show in reports as it means the import failed completely
        raise ImportError("Smart timing not available - missing chatterbox modules")
    
    def calculate_overlap_timing(self, audio_segments: List[torch.Tensor], 
                               subtitles: List) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Calculate timing for pad_with_silence mode allowing overlaps
        
        Args:
            audio_segments: List of audio tensors
            subtitles: List of SRTSubtitle objects
            
        Returns:
            Tuple of (final assembled audio, timing adjustments)
        """
        if not audio_segments:
            return torch.empty(0), []
        
        # Calculate total duration needed
        max_end_time = 0.0
        adjustments = []
        
        for i, (segment, subtitle) in enumerate(zip(audio_segments, subtitles)):
            natural_duration = self._get_audio_duration(segment)
            segment_end_time = subtitle.start_time + natural_duration
            max_end_time = max(max_end_time, segment_end_time)
            
            # Create timing adjustment record
            adjustment = {
                'segment_index': i,
                'sequence': subtitle.sequence,
                'start_time': subtitle.start_time,
                'end_time': segment_end_time,
                'natural_duration': natural_duration,
                'target_duration': subtitle.duration,
                'stretch_factor': 1.0,  # No stretching in overlap mode
                'needs_stretching': False,
                'stretch_type': 'none',
                'placement': 'natural_at_start_time'
            }
            adjustments.append(adjustment)
        
        if subtitles:
            max_end_time = max(max_end_time, subtitles[-1].end_time)
        
        # Assemble audio with overlaps
        final_audio = self._assemble_overlapping_audio(audio_segments, subtitles, max_end_time)
        
        return final_audio, adjustments
    
    def _get_audio_duration(self, audio: torch.Tensor) -> float:
        """Get duration of audio tensor in seconds"""
        if audio.dim() == 1:
            return audio.size(0) / self.sample_rate
        elif audio.dim() == 2:
            return audio.size(-1) / self.sample_rate
        else:
            raise ValueError(f"Unsupported audio tensor dimensions: {audio.dim()}")
    
    def _apply_time_stretch(self, audio: torch.Tensor, stretch_factor: float) -> torch.Tensor:
        """Apply time stretching to audio (placeholder - would use actual stretching)"""
        # This would integrate with the actual time stretching utilities
        # For now, return original audio (the actual implementation would be in audio_assembly.py)
        return audio
    
    def _add_silence_padding(self, audio: torch.Tensor, padding_duration: float) -> torch.Tensor:
        """Add silence padding to audio"""
        padding_samples = int(padding_duration * self.sample_rate)
        
        if audio.dim() == 1:
            padding = torch.zeros(padding_samples, device=audio.device, dtype=audio.dtype)
            return torch.cat([audio, padding], dim=0)
        else:
            padding = torch.zeros(audio.size(0), padding_samples, device=audio.device, dtype=audio.dtype)
            return torch.cat([audio, padding], dim=-1)
    
    def _assemble_overlapping_audio(self, audio_segments: List[torch.Tensor], 
                                  subtitles: List, max_end_time: float) -> torch.Tensor:
        """Assemble audio segments allowing overlaps at their SRT start times"""
        total_samples = int(max_end_time * self.sample_rate)
        
        # Initialize output buffer
        device = audio_segments[0].device
        dtype = audio_segments[0].dtype
        
        if audio_segments[0].dim() == 1:
            output_audio = torch.zeros(total_samples, device=device, dtype=dtype)
        else:
            channels = audio_segments[0].shape[0]
            output_audio = torch.zeros(channels, total_samples, device=device, dtype=dtype)
        
        # Place each segment at its start time
        for segment, subtitle in zip(audio_segments, subtitles):
            start_sample = int(subtitle.start_time * self.sample_rate)
            end_sample = start_sample + segment.size(-1)
            
            # Resize output if needed
            if end_sample > output_audio.size(-1):
                new_size = end_sample
                if output_audio.dim() == 1:
                    new_output = torch.zeros(new_size, device=device, dtype=dtype)
                    new_output[:output_audio.size(-1)] = output_audio
                else:
                    new_output = torch.zeros(output_audio.shape[0], new_size, device=device, dtype=dtype)
                    new_output[:, :output_audio.size(-1)] = output_audio
                output_audio = new_output
            
            # Add segment to output (allowing overlaps)
            if output_audio.dim() == 1:
                output_audio[start_sample:end_sample] += segment
            else:
                output_audio[:, start_sample:end_sample] += segment
        
        return output_audio
    
    def validate_timing_feasibility(self, subtitles: List, 
                                  max_stretch_ratio: float, 
                                  min_stretch_ratio: float) -> List[str]:
        """
        Validate if SRT timing is feasible for TTS generation
        
        Args:
            subtitles: List of SRTSubtitle objects
            max_stretch_ratio: Maximum allowed stretch ratio
            min_stretch_ratio: Minimum allowed stretch ratio
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        for subtitle in subtitles:
            # Estimate natural speech duration (rough: 150 words per minute)
            word_count = len(subtitle.text.split())
            estimated_duration = word_count / 2.5  # 150 words/min = 2.5 words/sec
            
            if estimated_duration > 0:
                required_stretch = subtitle.duration / estimated_duration
                
                if required_stretch > max_stretch_ratio:
                    warnings.append(
                        f"Subtitle {subtitle.sequence}: May need extreme compression "
                        f"({required_stretch:.2f}x, text: '{subtitle.text[:50]}...')"
                    )
                elif required_stretch < min_stretch_ratio:
                    warnings.append(
                        f"Subtitle {subtitle.sequence}: May need extreme stretching "
                        f"({required_stretch:.2f}x, text: '{subtitle.text[:50]}...')"
                    )
        
        return warnings