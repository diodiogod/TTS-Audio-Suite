"""
Audio Processing - Utility functions for audio manipulation in ChatterBox Voice
Common audio operations and processing functions
"""

import torch
import torchaudio
import tempfile
import os
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any


class AudioProcessingUtils:
    """
    Utility class for common audio processing operations.
    """
    
    @staticmethod
    def get_audio_duration(audio: torch.Tensor, sample_rate: int) -> float:
        """
        Get duration of audio tensor in seconds.
        
        Args:
            audio: Audio tensor
            sample_rate: Sample rate of the audio
            
        Returns:
            Duration in seconds
        """
        if audio.dim() == 1:
            return audio.size(0) / sample_rate
        elif audio.dim() == 2:
            return audio.size(-1) / sample_rate
        elif audio.dim() == 3:
            return audio.size(-1) / sample_rate
        else:
            raise ValueError(f"Unsupported audio tensor dimensions: {audio.dim()}")
    
    @staticmethod
    def normalize_audio_tensor(audio: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio tensor to standard format.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Normalized audio tensor
        """
        # Remove batch dimension if present
        if audio.dim() == 3 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        
        # Ensure we have proper channel dimension
        if audio.dim() == 1:
            # Mono audio - keep as 1D
            return audio
        elif audio.dim() == 2:
            # Multi-channel audio - keep as 2D [channels, samples]
            return audio
        else:
            raise ValueError(f"Unsupported audio tensor shape: {audio.shape}")
    
    @staticmethod
    def create_silence(duration_seconds: float, sample_rate: int, 
                      channels: int = 1, device: Optional[torch.device] = None,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Create silence tensor.
        
        Args:
            duration_seconds: Duration of silence in seconds
            sample_rate: Sample rate
            channels: Number of audio channels
            device: Target device for the tensor
            dtype: Data type for the tensor
            
        Returns:
            Silence tensor
        """
        num_samples = int(duration_seconds * sample_rate)
        
        if channels == 1:
            return torch.zeros(num_samples, device=device, dtype=dtype)
        else:
            return torch.zeros(channels, num_samples, device=device, dtype=dtype)
    
    @staticmethod
    def pad_audio_to_duration(audio: torch.Tensor, target_duration: float, 
                             sample_rate: int, pad_type: str = "end") -> torch.Tensor:
        """
        Pad audio to a specific duration.
        
        Args:
            audio: Input audio tensor
            target_duration: Target duration in seconds
            sample_rate: Sample rate
            pad_type: Where to add padding ("start", "end", "both")
            
        Returns:
            Padded audio tensor
        """
        current_duration = AudioProcessingUtils.get_audio_duration(audio, sample_rate)
        
        if current_duration >= target_duration:
            return audio
        
        pad_duration = target_duration - current_duration
        silence = AudioProcessingUtils.create_silence(
            pad_duration, sample_rate,
            channels=audio.shape[0] if audio.dim() == 2 else 1,
            device=audio.device,
            dtype=audio.dtype
        )
        
        if pad_type == "start":
            return torch.cat([silence, audio], dim=-1)
        elif pad_type == "end":
            return torch.cat([audio, silence], dim=-1)
        elif pad_type == "both":
            half_silence = AudioProcessingUtils.create_silence(
                pad_duration / 2, sample_rate,
                channels=audio.shape[0] if audio.dim() == 2 else 1,
                device=audio.device,
                dtype=audio.dtype
            )
            return torch.cat([half_silence, audio, half_silence], dim=-1)
        else:
            raise ValueError(f"Unknown pad_type: {pad_type}")
    
    @staticmethod
    def crossfade_audio(audio1: torch.Tensor, audio2: torch.Tensor, 
                       fade_duration: float, sample_rate: int) -> torch.Tensor:
        """
        Crossfade between two audio segments.
        
        Args:
            audio1: First audio segment
            audio2: Second audio segment
            fade_duration: Crossfade duration in seconds
            sample_rate: Sample rate
            
        Returns:
            Crossfaded audio
        """
        fade_samples = int(fade_duration * sample_rate)
        
        if audio1.size(-1) < fade_samples or audio2.size(-1) < fade_samples:
            # Not enough samples for crossfade, just concatenate
            return torch.cat([audio1, audio2], dim=-1)
        
        # Create fade curves
        fade_out = torch.linspace(1.0, 0.0, fade_samples, device=audio1.device)
        fade_in = torch.linspace(0.0, 1.0, fade_samples, device=audio2.device)
        
        # Apply fades to the overlapping regions
        if audio1.dim() == 1:
            audio1_end = audio1[-fade_samples:] * fade_out
            audio2_start = audio2[:fade_samples] * fade_in
        else:
            audio1_end = audio1[..., -fade_samples:] * fade_out
            audio2_start = audio2[..., :fade_samples] * fade_in
        
        # Create crossfaded region
        crossfaded = audio1_end + audio2_start
        
        # Combine all parts
        if audio1.dim() == 1:
            return torch.cat([
                audio1[:-fade_samples],
                crossfaded,
                audio2[fade_samples:]
            ], dim=-1)
        else:
            return torch.cat([
                audio1[..., :-fade_samples],
                crossfaded,
                audio2[..., fade_samples:]
            ], dim=-1)
    
    @staticmethod
    def concatenate_audio_segments(segments: List[torch.Tensor], 
                                 method: str = "simple",
                                 silence_duration: float = 0.0,
                                 crossfade_duration: float = 0.01,
                                 sample_rate: int = 22050) -> torch.Tensor:
        """
        Concatenate multiple audio segments with various methods.
        
        Args:
            segments: List of audio tensors to concatenate
            method: Concatenation method ("simple", "silence", "crossfade")
            silence_duration: Duration of silence between segments (for "silence" method)
            crossfade_duration: Duration of crossfade (for "crossfade" method)
            sample_rate: Sample rate
            
        Returns:
            Concatenated audio tensor
        """
        if not segments:
            return torch.empty(0)
        
        if len(segments) == 1:
            return segments[0]
        
        # Normalize all segments to have consistent dimensions
        normalized_segments = []
        for segment in segments:
            if segment.dim() == 1:
                # Convert 1D to 2D: [samples] -> [1, samples]
                segment = segment.unsqueeze(0)
            elif segment.dim() == 2 and segment.shape[0] > segment.shape[1]:
                # If shape is [samples, channels], transpose to [channels, samples]
                segment = segment.transpose(0, 1)
            normalized_segments.append(segment)
        
        segments = normalized_segments
        
        if method == "simple":
            return torch.cat(segments, dim=-1)
        
        elif method == "silence":
            result = segments[0]
            for segment in segments[1:]:
                if silence_duration > 0:
                    silence = AudioProcessingUtils.create_silence(
                        silence_duration, sample_rate,
                        channels=result.shape[0],
                        device=result.device,
                        dtype=result.dtype
                    )
                    # Ensure all tensors have same dimensionality before concatenation
                    tensors_to_cat = [result, silence, segment]
                    normalized_tensors = []
                    
                    for tensor in tensors_to_cat:
                        if tensor.dim() == 1:
                            tensor = tensor.unsqueeze(0)
                        elif tensor.dim() == 2 and tensor.shape[0] > tensor.shape[1]:
                            tensor = tensor.transpose(0, 1)
                        normalized_tensors.append(tensor)
                    
                    result = torch.cat(normalized_tensors, dim=-1)
                else:
                    # Ensure tensors have same dimensionality 
                    if result.dim() != segment.dim():
                        if result.dim() == 1:
                            result = result.unsqueeze(0)
                        if segment.dim() == 1:
                            segment = segment.unsqueeze(0)
                    result = torch.cat([result, segment], dim=-1)
            return result
        
        elif method == "crossfade":
            result = segments[0]
            for segment in segments[1:]:
                result = AudioProcessingUtils.crossfade_audio(
                    result, segment, crossfade_duration, sample_rate
                )
            return result
        
        else:
            raise ValueError(f"Unknown concatenation method: {method}")
    
    @staticmethod
    def seconds_to_samples(seconds: float, sample_rate: int) -> int:
        """
        Convert seconds to sample count.
        
        Args:
            seconds: Time in seconds
            sample_rate: Sample rate
            
        Returns:
            Number of samples
        """
        return int(seconds * sample_rate)
    
    @staticmethod
    def samples_to_seconds(samples: int, sample_rate: int) -> float:
        """
        Convert sample count to seconds.
        
        Args:
            samples: Number of samples
            sample_rate: Sample rate
            
        Returns:
            Time in seconds
        """
        return samples / sample_rate
    
    @staticmethod
    def save_audio_to_temp_file(audio: torch.Tensor, sample_rate: int, 
                               suffix: str = ".wav") -> str:
        """
        Save audio tensor to a temporary file.
        
        Args:
            audio: Audio tensor to save
            sample_rate: Sample rate
            suffix: File suffix/extension
            
        Returns:
            Path to the temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_file.close()
        
        # Normalize audio for saving
        if audio.dim() == 3:
            audio = audio.squeeze(0)  # Remove batch dimension
        
        torchaudio.save(temp_file.name, audio, sample_rate)
        return temp_file.name
    
    @staticmethod
    def load_audio_from_file(file_path: str, target_sample_rate: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """
        Load audio from file with optional resampling.
        
        Args:
            file_path: Path to audio file
            target_sample_rate: Optional target sample rate for resampling
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        audio, sample_rate = torchaudio.load(file_path)
        
        if target_sample_rate is not None and sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            audio = resampler(audio)
            sample_rate = target_sample_rate
        
        return audio, sample_rate
    
    @staticmethod
    def format_for_comfyui(audio: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
        """
        Format audio tensor for ComfyUI output.
        
        Args:
            audio: Audio tensor
            sample_rate: Sample rate
            
        Returns:
            ComfyUI audio format dictionary
        """
        # Ensure proper dimensions for ComfyUI
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add channel dimension [samples] -> [1, samples]
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # Add batch dimension [channels, samples] -> [1, channels, samples]
        
        # Ensure audio is on CPU and proper dtype for ComfyUI compatibility
        if hasattr(audio, 'cpu'):
            audio = audio.cpu()
        
        # Ensure float32 dtype for ComfyUI video nodes
        if hasattr(audio, 'float'):
            audio = audio.float()  # Converts to float32
        
        return {
            "waveform": audio,
            "sample_rate": sample_rate
        }
    
    @staticmethod
    def normalize_audio_input(audio_input, input_name: str = "audio") -> Dict[str, Any]:
        """
        Universal audio input normalizer - handles all ComfyUI audio formats.
        Supports: AUDIO dict, Character Voices output, VideoHelper LazyAudioMap, etc.
        
        Args:
            audio_input: Audio input in any supported format
            input_name: Name for error messages
            
        Returns:
            Standard AUDIO dict with 'waveform' and 'sample_rate' keys
            
        Raises:
            ValueError: If audio format is not supported
        """
        if audio_input is None:
            raise ValueError(f"{input_name} input is required")
        
        try:
            # Character Voices node output (NARRATOR_VOICE)
            if isinstance(audio_input, dict) and "audio" in audio_input:
                # Extract the nested audio component
                return AudioProcessingUtils.normalize_audio_input(audio_input["audio"], input_name)
            
            # Standard AUDIO format or VideoHelper LazyAudioMap
            elif hasattr(audio_input, "__getitem__"):
                # Check for required keys
                if "waveform" in audio_input and "sample_rate" in audio_input:
                    # Already in correct format - just ensure it's a dict
                    return {
                        "waveform": audio_input["waveform"], 
                        "sample_rate": audio_input["sample_rate"]
                    }
                else:
                    # Try to find missing keys in mapping-like objects
                    available_keys = []
                    if hasattr(audio_input, "keys"):
                        try:
                            available_keys = list(audio_input.keys())
                        except:
                            pass
                    
                    raise ValueError(f"Audio input missing required keys. Expected 'waveform' and 'sample_rate', found: {available_keys}")
            
            else:
                # Unknown format
                audio_type = type(audio_input).__name__
                raise ValueError(f"Unsupported audio format: {audio_type}")
                
        except Exception as e:
            if "Audio input missing required keys" in str(e) or "Unsupported audio format" in str(e):
                raise  # Re-raise our specific errors
            else:
                raise ValueError(f"Failed to normalize {input_name}: {e}")
    
    @staticmethod
    def apply_volume(audio: torch.Tensor, volume: float) -> torch.Tensor:
        """
        Apply volume scaling to audio.
        
        Args:
            audio: Input audio tensor
            volume: Volume multiplier (1.0 = no change)
            
        Returns:
            Volume-adjusted audio tensor
        """
        return audio * volume
    
    @staticmethod
    def normalize_loudness(audio: torch.Tensor, target_lufs: float = -23.0) -> torch.Tensor:
        """
        Normalize audio loudness (basic implementation).
        
        Args:
            audio: Input audio tensor
            target_lufs: Target LUFS level
            
        Returns:
            Normalized audio tensor
        """
        # Simple RMS-based normalization (not true LUFS)
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 0:
            # Convert target LUFS to linear scale (approximation)
            target_rms = 10 ** (target_lufs / 20)
            scale_factor = target_rms / rms
            return audio * scale_factor
        return audio
    
    @staticmethod
    def detect_silence(audio: torch.Tensor, threshold: float = 0.01,
                      min_duration: float = 0.1, sample_rate: int = 22050) -> List[Tuple[float, float]]:
        """
        Detect silent regions in audio. Handles 1D to 4D tensors without quality loss.

        Args:
            audio: Input audio tensor (supports 1D to 4D shapes)
            threshold: Amplitude threshold for silence detection
            min_duration: Minimum duration for a silence region
            sample_rate: Sample rate

        Returns:
            List of (start_time, end_time) tuples for silent regions
        """
        # Flatten to 1D while preserving sample values
        # Remove batch and channel dimensions, keep only sample data
        while audio.dim() > 1:
            if audio.shape[0] == 1:
                # Remove batch dimension if it's 1
                audio = audio.squeeze(0)
            else:
                # Multiple channels/batches: average along first dimension to convert to mono
                audio = torch.mean(audio, dim=0)
        
        # Find samples below threshold
        silent_mask = torch.abs(audio) < threshold
        
        # Find transitions
        transitions = torch.diff(silent_mask.float())
        silence_starts = torch.where(transitions == 1)[0] + 1
        silence_ends = torch.where(transitions == -1)[0] + 1
        
        # Handle edge cases
        if silent_mask[0]:
            silence_starts = torch.cat([torch.tensor([0], device=silence_starts.device), silence_starts])
        if silent_mask[-1]:
            silence_ends = torch.cat([silence_ends, torch.tensor([len(audio)], device=silence_ends.device)])
        
        # Filter by minimum duration
        silent_regions = []
        min_samples = int(min_duration * sample_rate)
        
        for start, end in zip(silence_starts, silence_ends):
            if end - start >= min_samples:
                start_time = start.item() / sample_rate
                end_time = end.item() / sample_rate
                silent_regions.append((start_time, end_time))
        
        return silent_regions


class AudioCache:
    """
    Simple audio caching system for performance optimization.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize audio cache.
        
        Args:
            max_size: Maximum number of cached items
        """
        self.cache: Dict[str, Tuple[torch.Tensor, int]] = {}
        self.max_size = max_size
        self.access_order: List[str] = []
    
    def get(self, key: str) -> Optional[Tuple[torch.Tensor, int]]:
        """
        Get cached audio.
        
        Args:
            key: Cache key
            
        Returns:
            Cached audio tuple or None if not found
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            audio, sample_rate = self.cache[key]
            return audio.clone(), sample_rate  # Return a copy
        return None
    
    def put(self, key: str, audio: torch.Tensor, sample_rate: int):
        """
        Cache audio.
        
        Args:
            key: Cache key
            audio: Audio tensor to cache
            sample_rate: Sample rate
        """
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        # Update cache
        self.cache[key] = (audio.clone(), sample_rate)  # Store a copy
        
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


# Global audio cache instance
audio_cache = AudioCache()