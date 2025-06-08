"""
Audio Timing Utilities for ChatterBox TTS SRT Support
Provides time-stretching, silence padding, and sample-accurate timing conversion
"""

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
    """
    FFmpeg-based time stretching implementation using the atempo filter
    """
    
    def __init__(self):
        """
        Initialize FFmpeg time stretcher
        Verifies FFmpeg is available on the system
        """
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise AudioTimingError("FFmpeg is required but not found. Please install FFmpeg and ensure it's in your PATH.")
    
    def time_stretch(self, audio: torch.Tensor, stretch_factor: float,
                    sample_rate: int) -> torch.Tensor:
        """
        Time-stretch audio using FFmpeg's atempo filter
        
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
        
        # Create temporary directory for audio files
        with tempfile.TemporaryDirectory() as temp_dir:
            stretched_channels = []
            
            for channel in range(audio.size(0)):
                # Save channel audio to WAV file
                channel_audio = audio[channel].cpu().numpy()
                input_path = os.path.join(temp_dir, f'input_{channel}.wav')
                output_path = os.path.join(temp_dir, f'output_{channel}.wav')
                
                # Save as 32-bit float WAV
                import soundfile as sf
                sf.write(input_path, channel_audio, sample_rate, subtype='FLOAT')
                
                try:
                    # Construct FFmpeg command
                    # Use multiple atempo filters for large stretch factors
                    atempo_chain = self._build_atempo_chain(stretch_factor)
                    
                    cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output
                        '-i', input_path,
                        '-filter:a', atempo_chain,
                        '-acodec', 'pcm_f32le',  # 32-bit float output
                        output_path
                    ]
                    
                    # Run FFmpeg
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise AudioTimingError(f"FFmpeg error: {result.stderr}")
                    
                    # Load processed audio
                    stretched_audio, _ = sf.read(output_path)
                    stretched_channels.append(torch.from_numpy(stretched_audio))
                    
                except Exception as e:
                    raise AudioTimingError(f"FFmpeg processing failed: {str(e)}")
            
            # Combine channels
            result = torch.stack(stretched_channels, dim=0).to(audio.device)
            
            # Restore original shape if input was 1D
            if len(original_shape) == 1:
                result = result.squeeze(0)
            
            return result
    
    def _build_atempo_chain(self, stretch_factor: float) -> str:
        """
        Build FFmpeg filter chain for time stretching
        Handles stretch factors outside the normal atempo range (0.5 to 2.0)
        by chaining multiple atempo filters
        """
        if 0.5 <= stretch_factor <= 2.0:
            return f'atempo={1/stretch_factor}'  # Inverse because FFmpeg uses speed factor
        
        # For factors outside 0.5-2.0 range, chain multiple atempo filters
        factors = []
        remaining = stretch_factor
        
        while remaining < 0.5:
            factors.append(2.0)  # Speed up by 2x
            remaining *= 2
        while remaining > 2.0:
            factors.append(0.5)  # Slow down by 2x
            remaining *= 0.5
        
        # Add remaining factor
        if abs(remaining - 1.0) > 1e-6:
            factors.append(1/remaining)
        
        # Build filter chain
        return ','.join(f'atempo={f}' for f in factors)


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
        
        if time_stretcher is not None:
            if not isinstance(time_stretcher, (PhaseVocoderTimeStretcher, FFmpegTimeStretcher)):
                raise AudioTimingError("time_stretcher must be PhaseVocoderTimeStretcher or FFmpegTimeStretcher")
            self.time_stretcher = time_stretcher
        else:
            if stretcher_type == "ffmpeg":
                try:
                    self.time_stretcher = FFmpegTimeStretcher()
                except AudioTimingError as e:
                    print(f"Warning: FFmpeg stretcher initialization failed ({str(e)}). Falling back to phase vocoder.")
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
            
            # Time-stretch if needed
            if abs(current_duration - target_duration) > 0.01:  # 10ms tolerance
                stretch_factor = target_duration / current_duration
                print(f"Time-stretching segment {i}: {stretch_factor:.3f}x "
                      f"({current_duration:.3f}s -> {target_duration:.3f}s)")
                
                audio_segment = self.time_stretcher.time_stretch(
                    audio_segment, stretch_factor, self.sample_rate
                )
            
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