"""
Audio Timing Utilities for ChatterBox TTS SRT Support
Provides time-stretching, silence padding, and sample-accurate timing conversion
"""

# Smart numba compatibility for audio timing
from utils.compatibility import setup_numba_compatibility
import sys
import os
setup_numba_compatibility(quick_startup=True, verbose=False)

import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional, List, Union
import librosa
from scipy.signal import stft, istft
import warnings
import subprocess
import tempfile
import os
import shutil
import soundfile as sf


class AudioTimingError(Exception):
    """Exception raised when audio timing operations fail"""
    pass


class AudioTimingUtils:
    """
    Utilities for audio timing manipulation and synchronization
    """
    
    @staticmethod
    def seconds_to_samples(seconds: float, sample_rate: int) -> int:
        """
        Convert time in seconds to sample count
        
        Args:
            seconds: Time in seconds
            sample_rate: Audio sample rate
            
        Returns:
            Number of samples (integer)
        """
        return int(seconds * sample_rate)
    
    @staticmethod
    def samples_to_seconds(samples: int, sample_rate: int) -> float:
        """
        Convert sample count to time in seconds
        
        Args:
            samples: Number of samples
            sample_rate: Audio sample rate
            
        Returns:
            Time in seconds
        """
        return samples / sample_rate
    
    @staticmethod
    def get_audio_duration(audio: torch.Tensor, sample_rate: int) -> float:
        """
        Get duration of audio tensor in seconds
        
        Args:
            audio: Audio tensor (1D or 2D)
            sample_rate: Audio sample rate
            
        Returns:
            Duration in seconds
        """
        if audio.dim() == 1:
            return audio.size(0) / sample_rate
        elif audio.dim() == 2:
            return audio.size(-1) / sample_rate
        else:
            raise AudioTimingError(f"Unsupported audio tensor dimensions: {audio.dim()}")
    
    @staticmethod
    def create_silence(duration_seconds: float, sample_rate: int, 
                      channels: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Create silence tensor of specified duration
        
        Args:
            duration_seconds: Duration of silence in seconds
            sample_rate: Audio sample rate
            channels: Number of audio channels
            device: Target device for tensor
            
        Returns:
            Silence tensor of shape [channels, samples] or [samples] if channels=1
        """
        if duration_seconds < 0:
            raise AudioTimingError(f"Duration cannot be negative: {duration_seconds}")
        
        num_samples = AudioTimingUtils.seconds_to_samples(duration_seconds, sample_rate)
        
        if channels == 1:
            silence = torch.zeros(num_samples, device=device)
        else:
            silence = torch.zeros(channels, num_samples, device=device)
        
        return silence
    
    @staticmethod
    def pad_audio_to_duration(audio: torch.Tensor, target_duration: float, 
                            sample_rate: int, pad_mode: str = "end") -> torch.Tensor:
        """
        Pad audio to reach target duration
        
        Args:
            audio: Input audio tensor
            target_duration: Target duration in seconds
            sample_rate: Audio sample rate
            pad_mode: Where to add padding ("start", "end", "both")
            
        Returns:
            Padded audio tensor
        """
        current_duration = AudioTimingUtils.get_audio_duration(audio, sample_rate)
        
        if current_duration >= target_duration:
            return audio  # No padding needed
        
        padding_duration = target_duration - current_duration
        padding_samples = AudioTimingUtils.seconds_to_samples(padding_duration, sample_rate)
        
        if audio.dim() == 1:
            padding = torch.zeros(padding_samples, device=audio.device, dtype=audio.dtype)
        else:
            padding = torch.zeros(audio.size(0), padding_samples, device=audio.device, dtype=audio.dtype)
        
        if pad_mode == "start":
            return torch.cat([padding, audio], dim=-1)
        elif pad_mode == "end":
            return torch.cat([audio, padding], dim=-1)
        elif pad_mode == "both":
            half_padding = padding_samples // 2
            if audio.dim() == 1:
                start_pad = torch.zeros(half_padding, device=audio.device, dtype=audio.dtype)
                end_pad = torch.zeros(padding_samples - half_padding, device=audio.device, dtype=audio.dtype)
            else:
                start_pad = torch.zeros(audio.size(0), half_padding, device=audio.device, dtype=audio.dtype)
                end_pad = torch.zeros(audio.size(0), padding_samples - half_padding, device=audio.device, dtype=audio.dtype)
            return torch.cat([start_pad, audio, end_pad], dim=-1)
        else:
            raise AudioTimingError(f"Invalid pad_mode: {pad_mode}. Use 'start', 'end', or 'both'")


class PhaseVocoderTimeStretcher:
    """
    Phase vocoder-based time stretching implementation
    """
    
    def __init__(self, hop_length: int = 512, win_length: int = 2048):
        """
        Initialize phase vocoder
        
        Args:
            hop_length: STFT hop length
            win_length: STFT window length
        """
        self.hop_length = hop_length
        self.win_length = win_length
    
    def time_stretch(self, audio: torch.Tensor, stretch_factor: float, 
                    sample_rate: int) -> torch.Tensor:
        """
        Time-stretch audio using phase vocoder
        
        Args:
            audio: Input audio tensor (1D or 2D)
            stretch_factor: Time stretching factor (>1 = slower, <1 = faster)
            sample_rate: Audio sample rate
            
        Returns:
            Time-stretched audio tensor
        """
        if stretch_factor <= 0:
            raise AudioTimingError(f"Stretch factor must be positive: {stretch_factor}")
        
        if abs(stretch_factor - 1.0) < 1e-6:
            return audio  # No stretching needed
        
        # Handle different tensor dimensions
        original_shape = audio.shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add channel dimension
        elif audio.dim() > 2:
            raise AudioTimingError(f"Unsupported audio tensor dimensions: {audio.dim()}")
        
        stretched_channels = []
        
        for channel in range(audio.size(0)):
            channel_audio = audio[channel].cpu().numpy()
            
            try:
                # Use librosa for phase vocoder time stretching
                stretched = librosa.effects.time_stretch(
                    channel_audio, 
                    rate=1.0/stretch_factor,  # librosa uses rate = 1/stretch_factor
                    hop_length=self.hop_length
                )
                stretched_channels.append(torch.from_numpy(stretched))
                
            except Exception as e:
                # Fallback to simple resampling if phase vocoder fails
                warnings.warn(f"Phase vocoder failed, using simple resampling: {e}")
                stretched = self._simple_time_stretch(channel_audio, stretch_factor)
                stretched_channels.append(torch.from_numpy(stretched))
        
        # Combine channels
        result = torch.stack(stretched_channels, dim=0).to(audio.device)
        
        # Restore original shape if input was 1D
        if len(original_shape) == 1:
            result = result.squeeze(0)
        
        return result
    
    def _simple_time_stretch(self, audio: np.ndarray, stretch_factor: float) -> np.ndarray:
        """
        Simple time stretching using interpolation (fallback method)
        
        Args:
            audio: Input audio array
            stretch_factor: Time stretching factor
            
        Returns:
            Time-stretched audio array
        """
        original_length = len(audio)
        new_length = int(original_length * stretch_factor)
        
        # Create new time indices
        old_indices = np.linspace(0, original_length - 1, original_length)
        new_indices = np.linspace(0, original_length - 1, new_length)
        
        # Interpolate
        stretched = np.interp(new_indices, old_indices, audio)
        
        return stretched


class FFmpegTimeStretcher:
    """FFmpeg time stretching using atempo filter"""
    
    def __init__(self):
        """Verify FFmpeg is available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
            if result.returncode != 0:
                raise AudioTimingError("FFmpeg check failed")
        except Exception as e:
            raise AudioTimingError(f"FFmpeg not found: {str(e)}")
    def _build_filter_chain(self, stretch_factor: float) -> str:
        """Build safe FFmpeg filter chain for any stretch factor"""
        if stretch_factor <= 0:
            raise AudioTimingError("Invalid stretch factor")
            
        speed = 1/stretch_factor
        
        # Direct speed if within safe range
        if 0.5 <= speed <= 2.0:
            return f"aresample=async=0,atempo={speed:0.6f}"
            
        # Chain filters for extreme speeds
        filters = []
        remaining = speed
        
        while remaining < 0.5:
            filters.append('atempo=0.5')  # Max slowdown
            remaining *= 2
            if len(filters) > 4:  # Safety limit
                raise AudioTimingError("Too extreme slowdown")
                
        while remaining > 2.0:
            filters.append('atempo=2.0')  # Max speedup
            remaining /= 2
            if len(filters) > 4:
                raise AudioTimingError("Too extreme speedup")
                
        # Final adjustment
        if abs(remaining - 1.0) > 1e-6:
            filters.append(f'atempo={remaining:0.6f}')
            
        filter_str = f"aresample=async=0,{','.join(filters)}"
        return filter_str
        
    def time_stretch(self, audio: torch.Tensor, stretch_factor: float, sample_rate: int) -> torch.Tensor:
        """Time stretch audio using FFmpeg with safe filter chaining"""
        # Full validation
        if not isinstance(audio, torch.Tensor):
            raise AudioTimingError("Input must be a tensor")
            
        if not isinstance(stretch_factor, (int, float)) or stretch_factor <= 0:
            raise AudioTimingError(f"Invalid stretch factor: {stretch_factor}")
            
        if abs(stretch_factor - 1.0) < 1e-6:
            return audio
            
        try:
            # Get filter chain
            filter_str = self._build_filter_chain(stretch_factor)
            
            # Shape handling
            original_shape = audio.shape
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            elif audio.dim() > 2:
                raise AudioTimingError("Only mono/stereo supported")
                
            # Process channels
            stretched = []
            with tempfile.TemporaryDirectory() as temp_dir:
                
                for i in range(audio.shape[0]):
                    # Setup paths
                    in_path = os.path.join(temp_dir, f'in_{i}.wav')
                    out_path = os.path.join(temp_dir, f'out_{i}.wav')
                    
                    try:
                        # Save input
                        data = audio[i].cpu().numpy()
                        sf.write(in_path, data, sample_rate,
                                format='WAV', subtype='FLOAT')
                        
                        # Process with FFmpeg
                        cmd = [
                            'ffmpeg',
                            '-nostdin',      # No interaction
                            '-y',            # Overwrite output
                            '-hide_banner',  # Less noise
                            '-i', in_path,   # Input
                            '-filter:a', filter_str,  # Audio filter
                            '-c:a', 'pcm_f32le',  # Output format
                            '-ar', str(sample_rate),
                            '-ac', '1',      # Force mono
                            '-v', 'error',   # Only errors (less verbose)
                            out_path
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            raise AudioTimingError(f"FFmpeg processing failed: {result.stderr}")
                            
                        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                            raise AudioTimingError("No output produced")
                            
                        # Load and verify
                        audio_data, sr = sf.read(out_path)
                        if sr != sample_rate:
                            raise AudioTimingError(f"Sample rate mismatch: {sr}")
                            
                        if len(audio_data) == 0:
                            raise AudioTimingError("Empty output")
                            
                        stretched.append(torch.from_numpy(audio_data))
                        
                    except Exception as e:
                        raise AudioTimingError(f"Channel {i} failed: {str(e)}")
            
            # Combine results
            if not stretched:
                raise AudioTimingError("No audio was processed successfully")
                
            result = torch.stack(stretched, dim=0).to(audio.device)
            return result.squeeze(0) if len(original_shape) == 1 else result
            
        except Exception as e:
            raise AudioTimingError(f"Time stretching failed: {str(e)}")
                    
                    # Run FFmpeg with checks
    def _process_audio_file(self, in_path: str, out_path: str, filter_str: str,
                          sample_rate: int) -> np.ndarray:
        """Process a single audio file through FFmpeg"""
        cmd = [
            'ffmpeg',
            '-hide_banner',   # Reduce noise
            '-nostdin',       # No interactive mode
            '-y',            # Overwrite output
            '-i', in_path,
            '-filter:a', filter_str,
            '-acodec', 'pcm_f32le',
            '-ar', str(sample_rate),
            '-ac', '1',
            '-v', 'info',     # Informative output level
            out_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise AudioTimingError(f"FFmpeg failed: {result.stderr}")
                
            # Verify output
            if not os.path.exists(out_path):
                raise AudioTimingError("No output file produced")
                
            out_size = os.path.getsize(out_path)
            if out_size == 0:
                raise AudioTimingError("Empty output file")
                
            # Load and validate
            audio_data, sr = sf.read(out_path)
            
            if sr != sample_rate:
                raise AudioTimingError(f"Sample rate mismatch: got {sr}, expected {sample_rate}")
                
            if len(audio_data) == 0:
                raise AudioTimingError("No audio data in output")
                
            return audio_data
            
        except subprocess.TimeoutExpired:
            raise AudioTimingError("FFmpeg process timed out")
        except Exception as e:
            raise AudioTimingError(f"FFmpeg processing failed: {str(e)}")

        if not stretched:
            raise AudioTimingError("No channels were processed successfully")
            
        try:
            # Stack channels and restore shape
            result = torch.stack(stretched, dim=0).to(audio.device)
            print(f"Successfully processed all channels")
            return result.squeeze(0) if len(original_shape) == 1 else result
            
        except Exception as e:
            raise AudioTimingError(f"Failed to combine channels: {str(e)}")
    
    def _build_atempo_chain(self, stretch_factor: float) -> str:
        """Simple atempo command that stays within safe ranges"""
        if stretch_factor <= 0:
            raise AudioTimingError("Invalid stretch factor")
            
        speed = 1/stretch_factor
        if not (0.5 <= speed <= 2.0):
            raise AudioTimingError("Speed factor out of supported range")
            
        return f'atempo={speed:0.6f}'


class TimedAudioAssembler:
    """
    Assembles audio segments with precise timing control
    """
    
    def __init__(self, sample_rate: int, stretcher_type: str = "ffmpeg",
                 time_stretcher: Optional[Union[PhaseVocoderTimeStretcher, FFmpegTimeStretcher]] = None):
        """
        Initialize audio assembler
        
        Args:
            sample_rate: Target sample rate for output
            stretcher_type: Type of time stretcher to use ("ffmpeg" or "phase_vocoder")
            time_stretcher: Custom time stretching utility (creates default if None)
        """
        self.sample_rate = sample_rate
        self.stretch_method_used = None  # Track which stretching method was used
        
        if time_stretcher is not None:
            if not isinstance(time_stretcher, (PhaseVocoderTimeStretcher, FFmpegTimeStretcher)):
                raise AudioTimingError("time_stretcher must be PhaseVocoderTimeStretcher or FFmpegTimeStretcher")
            self.time_stretcher = time_stretcher
        else:
            if stretcher_type == "ffmpeg":
                try:
                    self.time_stretcher = FFmpegTimeStretcher()
                except AudioTimingError as e:
                    # Fall back to phase vocoder silently
                    self.time_stretcher = PhaseVocoderTimeStretcher()
            elif stretcher_type == "phase_vocoder":
                self.time_stretcher = PhaseVocoderTimeStretcher()
            else:
                raise AudioTimingError(f"Invalid stretcher_type: {stretcher_type}. Use 'ffmpeg' or 'phase_vocoder'")
    
    def assemble_timed_audio(self, audio_segments: List[torch.Tensor], 
                           target_timings: List[Tuple[float, float]],
                           total_duration: Optional[float] = None,
                           fade_duration: float = 0.01) -> torch.Tensor:
        """
        Assemble audio segments with precise timing
        
        Args:
            audio_segments: List of audio tensors to assemble
            target_timings: List of (start_time, end_time) tuples in seconds
            total_duration: Total duration of output (auto-calculated if None)
            fade_duration: Crossfade duration in seconds for overlaps
            
        Returns:
            Assembled audio tensor
        """
        if len(audio_segments) != len(target_timings):
            raise AudioTimingError(
                f"Number of audio segments ({len(audio_segments)}) must match "
                f"number of timings ({len(target_timings)})"
            )
        
        if not audio_segments:
            raise AudioTimingError("No audio segments provided")
        
        # Calculate total duration if not provided
        if total_duration is None:
            total_duration = max(end_time for _, end_time in target_timings)
        
        # Create output buffer
        total_samples = AudioTimingUtils.seconds_to_samples(total_duration, self.sample_rate)
        
        # Determine output shape based on first segment
        first_segment = audio_segments[0]
        if first_segment.dim() == 1:
            output = torch.zeros(total_samples, device=first_segment.device, dtype=first_segment.dtype)
        else:
            output = torch.zeros(first_segment.size(0), total_samples, 
                               device=first_segment.device, dtype=first_segment.dtype)
        
        # Process each segment
        for i, (audio_segment, (start_time, end_time)) in enumerate(zip(audio_segments, target_timings)):
            if start_time < 0 or end_time <= start_time:
                raise AudioTimingError(f"Invalid timing for segment {i}: {start_time} -> {end_time}")
            
            target_duration = end_time - start_time
            current_duration = AudioTimingUtils.get_audio_duration(audio_segment, self.sample_rate)
            
            # If input audio is empty but target duration is positive, create silence.
            if current_duration == 0.0 and target_duration > 0.0:
                num_output_channels = first_segment.size(0) if first_segment.dim() > 1 else 1
                audio_segment = AudioTimingUtils.create_silence(
                    target_duration,
                    self.sample_rate,
                    channels=num_output_channels,
                    device=first_segment.device, # Use device/dtype consistent with output buffer
                    dtype=first_segment.dtype
                )
                current_duration = target_duration # Update duration, no stretching needed for this segment
                self.stretch_method_used = "silence_created"
            
            # Time-stretch if needed
            if abs(current_duration - target_duration) > 0.01:  # 10ms tolerance
                if current_duration == 0.0: # Should only be true if target_duration is also ~0.0
                    stretch_factor = 1.0 # Avoid division by zero
                else:
                    stretch_factor = target_duration / current_duration
                
                # Determine and track stretching method
                if isinstance(self.time_stretcher, FFmpegTimeStretcher):
                    self.stretch_method_used = "ffmpeg"
                elif isinstance(self.time_stretcher, PhaseVocoderTimeStretcher):
                    self.stretch_method_used = "phase_vocoder"
                
                audio_segment = self.time_stretcher.time_stretch(
                    audio_segment, stretch_factor, self.sample_rate
                )
            else:
                # No stretching needed
                self.stretch_method_used = "none"
            
            # Calculate sample positions
            start_sample = AudioTimingUtils.seconds_to_samples(start_time, self.sample_rate)
            end_sample = AudioTimingUtils.seconds_to_samples(end_time, self.sample_rate)
            
            # Ensure segment fits exactly
            target_samples = end_sample - start_sample
            current_samples = audio_segment.size(-1)
            
            if current_samples != target_samples:
                # Fine-tune length to match exactly
                if current_samples > target_samples:
                    audio_segment = audio_segment[..., :target_samples]
                else:
                    padding_needed = target_samples - current_samples
                    if audio_segment.dim() == 1:
                        padding = torch.zeros(padding_needed, device=audio_segment.device, dtype=audio_segment.dtype)
                    else:
                        padding = torch.zeros(audio_segment.size(0), padding_needed, 
                                            device=audio_segment.device, dtype=audio_segment.dtype)
                    audio_segment = torch.cat([audio_segment, padding], dim=-1)
            
            # Place segment in output buffer with crossfading for overlaps
            self._place_segment_with_fade(output, audio_segment, start_sample, end_sample, fade_duration)
        
        return output
    
    def _place_segment_with_fade(self, output: torch.Tensor, segment: torch.Tensor,
                               start_sample: int, end_sample: int, fade_duration: float):
        """
        Place audio segment in output buffer with crossfading for overlaps
        """
        segment_length = segment.size(-1)
        fade_samples = min(
            AudioTimingUtils.seconds_to_samples(fade_duration, self.sample_rate),
            segment_length // 4  # Don't fade more than 25% of segment
        )
        
        # Check for overlap with existing content
        if output.dim() == 1:
            existing_content = output[start_sample:end_sample]
            has_overlap = torch.any(existing_content != 0)
        else:
            existing_content = output[:, start_sample:end_sample]
            has_overlap = torch.any(existing_content != 0)
        
        if has_overlap and fade_samples > 0:
            # Apply crossfade
            fade_in = torch.linspace(0, 1, fade_samples, device=segment.device)
            fade_out = torch.linspace(1, 0, fade_samples, device=segment.device)
            
            # Fade in the beginning of new segment
            if segment.dim() == 1:
                segment[:fade_samples] *= fade_in
                # Fade out existing content at the beginning
                output[start_sample:start_sample + fade_samples] *= fade_out
            else:
                segment[:, :fade_samples] *= fade_in.unsqueeze(0)
                output[:, start_sample:start_sample + fade_samples] *= fade_out.unsqueeze(0)
            
            # Fade out the end of new segment if there's content after
            end_check_start = min(end_sample, output.size(-1) - fade_samples)
            end_check_end = min(end_sample + fade_samples, output.size(-1))
            
            if output.dim() == 1:
                if end_check_start < end_check_end and torch.any(output[end_check_start:end_check_end] != 0):
                    segment[-fade_samples:] *= fade_out
            else:
                if end_check_start < end_check_end and torch.any(output[:, end_check_start:end_check_end] != 0):
                    segment[:, -fade_samples:] *= fade_out.unsqueeze(0)
        
        # Add segment to output
        if output.dim() == 1:
            output[start_sample:end_sample] += segment
        else:
            output[:, start_sample:end_sample] += segment


def calculate_timing_adjustments(natural_durations: List[float], 
                               target_timings: List[Tuple[float, float]]) -> List[dict]:
    """
    Calculate timing adjustments needed for each audio segment
    
    Args:
        natural_durations: Natural durations of TTS-generated segments
        target_timings: Target (start_time, end_time) tuples from SRT
        
    Returns:
        List of adjustment dictionaries with timing information
    """
    adjustments = []
    
    for i, (natural_duration, (start_time, end_time)) in enumerate(zip(natural_durations, target_timings)):
        target_duration = end_time - start_time
        stretch_factor = target_duration / natural_duration if natural_duration > 0 else 1.0
        
        adjustment = {
            'segment_index': i,
            'natural_duration': natural_duration,
            'target_duration': target_duration,
            'start_time': start_time,
            'end_time': end_time,
            'stretch_factor': stretch_factor,
            'needs_stretching': abs(stretch_factor - 1.0) > 0.05,  # 5% tolerance
            'stretch_type': 'compress' if stretch_factor < 1.0 else 'expand' if stretch_factor > 1.0 else 'none'
        }
        
        adjustments.append(adjustment)
    
    return adjustments