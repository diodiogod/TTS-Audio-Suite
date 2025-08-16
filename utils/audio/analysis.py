"""
Audio Analysis - Core functionality for audio waveform analysis and timing extraction
Provides precise timing extraction for F5-TTS speech editing through waveform analysis
"""

import torch
import torchaudio
import numpy as np
import json
import os
import tempfile
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

# Add support for more audio formats
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from utils.audio.processing import AudioProcessingUtils


@dataclass
class WaveformData:
    """Data structure for waveform visualization"""
    samples: np.ndarray
    sample_rate: int
    duration: float
    peaks: List[float]
    rms_values: List[float]
    time_axis: np.ndarray


@dataclass
class TimingRegion:
    """Data structure for timing regions"""
    start_time: float
    end_time: float
    label: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = None


class AudioAnalyzer:
    """
    Core audio analysis functionality for precise timing extraction.
    Provides waveform analysis, silence detection, and timing region extraction.
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize audio analyzer.
        
        Args:
            sample_rate: Target sample rate for analysis
        """
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.win_length = 1024
        self.n_fft = 2048
        
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio file for analysis with support for multiple formats.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Get file extension
        ext = os.path.splitext(audio_path)[1].lower()
        supported_exts = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
        
        if ext not in supported_exts:
            raise ValueError(f"Unsupported audio format: {ext}. Supported formats: {supported_exts}")
        
        try:
            # Try torchaudio first
            audio, sr = torchaudio.load(audio_path)
            # print(f"✅ Loaded audio using torchaudio: {audio_path}")  # Debug: load method
            
        except Exception as e:
            print(f"⚠️ Torchaudio failed: {e}")  # Keep: important fallback info
            if LIBROSA_AVAILABLE:
                try:
                    # Fallback to librosa for better format support
                    audio_np, sr = librosa.load(audio_path, sr=None, mono=False)
                    
                    # Convert to torch tensor
                    if audio_np.ndim == 1:
                        audio = torch.from_numpy(audio_np).unsqueeze(0)
                    else:
                        audio = torch.from_numpy(audio_np)
                    
                    # print(f"✅ Loaded audio using librosa: {audio_path}")  # Debug: fallback success
                    
                except Exception as e2:
                    print(f"❌ Librosa also failed: {e2}")  # Keep: critical error
                    raise RuntimeError(f"Failed to load audio file with both torchaudio and librosa: {e}, {e2}")
            else:
                raise RuntimeError(f"Failed to load audio file: {e}. Consider installing librosa for better format support.")
        
        # Convert to mono if stereo (ensure 1D output for consistency)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0)  # Remove keepdim=True to get 1D tensor
        elif audio.dim() == 2:
            audio = audio.squeeze(0)  # Remove channel dimension if already mono
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
            sr = self.sample_rate
        
        # print(f"✅ Audio loaded successfully: {audio_path} - Duration: {audio.shape[-1] / sr:.2f}s")  # Debug: final load
        
        return audio, sr
    
    def analyze_audio(self, audio: torch.Tensor, sample_rate: int = None) -> WaveformData:
        """
        Analyze audio for waveform visualization.
        
        Args:
            audio: Audio tensor
            sample_rate: Sample rate (uses self.sample_rate if None)
            
        Returns:
            WaveformData object with analysis results
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Ensure audio is 1D
        if audio.dim() > 1:
            audio = audio.squeeze()
        
        # Convert to numpy for analysis
        audio_np = audio.detach().cpu().numpy() if audio.requires_grad else audio.cpu().numpy()
        duration = len(audio_np) / sample_rate
        
        # Create time axis
        time_axis = np.linspace(0, duration, len(audio_np))
        
        # Calculate RMS values for visualization (downsample for efficiency)
        rms_window = self.hop_length
        rms_values = []
        
        for i in range(0, len(audio_np), rms_window):
            window = audio_np[i:i + rms_window]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
        
        # Find peaks for timing markers
        peaks = self._find_peaks(audio_np, sample_rate)
        
        return WaveformData(
            samples=audio_np,
            sample_rate=sample_rate,
            duration=duration,
            peaks=peaks,
            rms_values=rms_values,
            time_axis=time_axis
        )
    
    def _find_peaks(self, audio: np.ndarray, sample_rate: int, 
                   min_distance: float = 0.1, threshold: float = 0.1, region_size: float = 0.1) -> List[float]:
        """
        Find peaks in audio signal for timing markers.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
            min_distance: Minimum distance between peaks in seconds
            threshold: Minimum amplitude threshold for peaks
            region_size: Size of region around peaks (not used in this method, passed to caller)
            
        Returns:
            List of peak times in seconds
        """
        # Calculate RMS envelope - use smaller window for better peak detection
        window_size = int(0.005 * sample_rate)  # 5ms window for better resolution
        window_size = max(1, window_size)  # Ensure at least 1 sample
        
        # Calculate RMS values (downsampled, not repeated)
        rms_values = []
        rms_times = []
        
        for i in range(0, len(audio), window_size):
            window = audio[i:i + window_size]
            if len(window) > 0:
                rms = np.sqrt(np.mean(window ** 2))
                rms_values.append(rms)
                rms_times.append(i / sample_rate)  # Time of this RMS window
        
        rms_values = np.array(rms_values)
        rms_times = np.array(rms_times)
        
        # Find peaks in RMS envelope
        peaks = []
        min_distance_windows = max(1, int(min_distance / (window_size / sample_rate)))
        
        for i in range(1, len(rms_values) - 1):
            if (rms_values[i] > rms_values[i-1] and 
                rms_values[i] > rms_values[i+1] and 
                rms_values[i] > threshold):
                
                # Check minimum distance from previous peaks
                if not peaks or i - peaks[-1] >= min_distance_windows:
                    peaks.append(i)
        
        # Convert peak indices to time using RMS time array
        peak_times = [rms_times[peak_idx] for peak_idx in peaks]
        
        return peak_times
    
    def detect_silence_regions(self, audio: torch.Tensor, 
                             threshold: float = 0.01, 
                             min_duration: float = 0.1,
                             invert: bool = False) -> List[TimingRegion]:
        """
        Detect silence regions in audio, optionally inverted to get speech regions.
        
        Args:
            audio: Audio tensor
            threshold: Amplitude threshold for silence
            min_duration: Minimum duration for silence region
            invert: If True, return speech regions instead of silence regions
            
        Returns:
            List of TimingRegion objects for silence or speech regions
        """
        silent_regions = AudioProcessingUtils.detect_silence(
            audio, threshold, min_duration, self.sample_rate
        )
        
        if invert:
            # Convert silence regions to speech regions by inverting
            speech_regions = self._invert_silence_to_speech(silent_regions, audio)
            return [
                TimingRegion(
                    start_time=start,
                    end_time=end,
                    label="speech",
                    confidence=1.0,
                    metadata={"type": "speech", "threshold": threshold, "inverted_from": "silence"}
                )
                for start, end in speech_regions
            ]
        else:
            # Return original silence regions
            return [
                TimingRegion(
                    start_time=start,
                    end_time=end,
                    label="silence",
                    confidence=1.0,
                    metadata={"type": "silence", "threshold": threshold}
                )
                for start, end in silent_regions
            ]
    
    def _invert_silence_to_speech(self, silence_regions: List[Tuple[float, float]], audio: torch.Tensor) -> List[Tuple[float, float]]:
        """
        Convert silence regions to speech regions by inverting the detection.
        
        Args:
            silence_regions: List of (start, end) tuples for silence
            audio: Audio tensor to get total duration
            
        Returns:
            List of (start, end) tuples for speech regions
        """
        if not silence_regions:
            # No silence detected, entire audio is speech
            duration = len(audio) / self.sample_rate
            return [(0.0, duration)]
        
        speech_regions = []
        total_duration = len(audio) / self.sample_rate
        
        # Sort silence regions by start time
        sorted_silence = sorted(silence_regions, key=lambda x: x[0])
        
        # Check for speech before first silence
        if sorted_silence[0][0] > 0:
            speech_regions.append((0.0, sorted_silence[0][0]))
        
        # Check for speech between silence regions
        for i in range(len(sorted_silence) - 1):
            current_silence_end = sorted_silence[i][1]
            next_silence_start = sorted_silence[i + 1][0]
            
            if current_silence_end < next_silence_start:
                speech_regions.append((current_silence_end, next_silence_start))
        
        # Check for speech after last silence
        if sorted_silence[-1][1] < total_duration:
            speech_regions.append((sorted_silence[-1][1], total_duration))
        
        return speech_regions
    
    def detect_word_boundaries(self, audio: torch.Tensor, 
                             sensitivity: float = 0.5) -> List[TimingRegion]:
        """
        Detect potential word boundaries using energy analysis.
        
        Args:
            audio: Audio tensor
            sensitivity: Detection sensitivity (0.0 to 1.0)
            
        Returns:
            List of TimingRegion objects for word boundaries
        """
        # Convert to numpy
        if audio.dim() > 1:
            audio = audio.squeeze()
        audio_np = audio.detach().numpy() if audio.requires_grad else audio.numpy()
        
        # Calculate energy envelope
        window_size = int(0.01 * self.sample_rate)  # 10ms window
        energy = []
        
        for i in range(0, len(audio_np), window_size):
            window = audio_np[i:i + window_size]
            energy.append(np.sum(window ** 2))
        
        energy = np.array(energy)
        
        # Find energy drops (potential word boundaries)
        # Use derivative to find rapid changes
        energy_diff = np.diff(energy)
        
        # Threshold based on sensitivity
        threshold = np.std(energy_diff) * (1.0 - sensitivity)
        
        # Find significant drops
        boundaries = []
        for i in range(1, len(energy_diff) - 1):
            if (energy_diff[i] < -threshold and 
                energy_diff[i-1] > -threshold/2 and 
                energy_diff[i+1] > -threshold/2):
                
                time_pos = i * window_size / self.sample_rate
                boundaries.append(TimingRegion(
                    start_time=max(0, time_pos - 0.05),
                    end_time=min(len(audio_np) / self.sample_rate, time_pos + 0.05),
                    label="word_boundary",
                    confidence=min(1.0, abs(energy_diff[i]) / threshold),
                    metadata={"type": "word_boundary", "energy_drop": float(energy_diff[i])}
                ))
        
        return boundaries
    
    def generate_visualization_data(self, audio: torch.Tensor, 
                                  target_points: int = 2000) -> Dict[str, Any]:
        """
        Generate data for waveform visualization.
        
        Args:
            audio: Audio tensor
            target_points: Target number of points for visualization
            
        Returns:
            Dictionary with visualization data
        """
        waveform_data = self.analyze_audio(audio)
        
        # Downsample for visualization efficiency
        downsample_factor = max(1, len(waveform_data.samples) // target_points)
        
        viz_samples = waveform_data.samples[::downsample_factor]
        viz_time = waveform_data.time_axis[::downsample_factor]
        
        # Calculate RMS for visualization
        rms_window = max(1, len(viz_samples) // 200)  # 200 RMS points
        rms_values = []
        rms_times = []
        
        for i in range(0, len(viz_samples), rms_window):
            window = viz_samples[i:i + rms_window]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(float(rms))  # Convert to Python float
            rms_times.append(float(viz_time[i] if i < len(viz_time) else viz_time[-1]))  # Convert to Python float
        
        return {
            "waveform": {
                "samples": [float(x) for x in viz_samples.tolist()],  # Ensure Python floats
                "time": [float(x) for x in viz_time.tolist()]  # Ensure Python floats
            },
            "rms": {
                "values": rms_values,
                "time": rms_times
            },
            "peaks": [float(x) for x in waveform_data.peaks],  # Convert to Python floats
            "duration": float(waveform_data.duration),  # Convert to Python float
            "sample_rate": int(waveform_data.sample_rate)  # Convert to Python int
        }
    
    def extract_timing_regions(self, audio: torch.Tensor, 
                             method: str = "silence", **kwargs) -> List[TimingRegion]:
        """
        Extract timing regions using specified method.
        
        Args:
            audio: Audio tensor
            method: Extraction method ("silence", "energy", "peaks")
            
        Returns:
            List of TimingRegion objects
        """
        if method == "silence":
            return self.detect_silence_regions(audio)
        elif method == "energy":
            return self.detect_word_boundaries(audio)
        elif method == "peaks":
            # Extract peak detection parameters from kwargs
            peak_threshold = kwargs.get("peak_threshold", 0.02)
            peak_min_distance = kwargs.get("peak_min_distance", 0.05)
            peak_region_size = kwargs.get("peak_region_size", 0.1)
            
            # Convert to numpy for peak detection
            if audio.dim() > 1:
                audio = audio.squeeze()
            audio_np = audio.detach().numpy() if audio.requires_grad else audio.numpy()
            
            # Find peaks with custom parameters
            peak_times = self._find_peaks(
                audio_np, self.sample_rate, 
                min_distance=peak_min_distance, 
                threshold=peak_threshold,
                region_size=peak_region_size
            )
            
            regions = []
            duration = len(audio_np) / self.sample_rate
            
            for i, peak_time in enumerate(peak_times):
                # Create regions around peaks with adjustable size
                half_region = peak_region_size / 2
                start = max(0, peak_time - half_region)
                end = min(duration, peak_time + half_region)
                
                regions.append(TimingRegion(
                    start_time=start,
                    end_time=end,
                    label=f"peak_{i+1}",
                    confidence=1.0,
                    metadata={
                        "type": "peak", 
                        "peak_time": peak_time,
                        "threshold": peak_threshold,
                        "min_distance": peak_min_distance,
                        "region_size": peak_region_size
                    }
                ))
            
            return regions
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
    def group_regions(self, regions: List[TimingRegion], threshold: float = 0.0) -> List[TimingRegion]:
        """
        Group nearby regions together based on time threshold.
        
        Args:
            regions: List of TimingRegion objects to group
            threshold: Maximum time gap between regions to group them (0.0 = no grouping)
            
        Returns:
            List of grouped TimingRegion objects
        """
        if threshold <= 0.000 or len(regions) <= 1:
            return regions
        
        # Sort regions by start time
        sorted_regions = sorted(regions, key=lambda r: r.start_time)
        grouped_regions = []
        
        current_group = [sorted_regions[0]]
        
        for i in range(1, len(sorted_regions)):
            current_region = sorted_regions[i]
            last_in_group = current_group[-1]
            
            # Check if current region should be grouped with the current group
            # Either overlapping or within threshold distance
            gap = current_region.start_time - last_in_group.end_time
            
            if gap <= threshold or current_region.start_time <= last_in_group.end_time:
                # Add to current group
                current_group.append(current_region)
            else:
                # Finalize current group and start new one
                grouped_regions.append(self._merge_region_group(current_group))
                current_group = [current_region]
        
        # Don't forget the last group
        if current_group:
            grouped_regions.append(self._merge_region_group(current_group))
        
        return grouped_regions
    
    def _merge_region_group(self, region_group: List[TimingRegion]) -> TimingRegion:
        """
        Merge a group of regions into a single region.
        
        Args:
            region_group: List of regions to merge
            
        Returns:
            Single merged TimingRegion
        """
        if len(region_group) == 1:
            return region_group[0]
        
        # Find the overall start and end times
        start_time = min(r.start_time for r in region_group)
        end_time = max(r.end_time for r in region_group)
        
        # Create simple label for grouped regions
        if len(region_group) == 2:
            label = f"group_{region_group[0].label.split('_')[-1]}+{region_group[1].label.split('_')[-1]}"
        else:
            # Use first and last numbers for multi-region groups
            first_num = region_group[0].label.split('_')[-1]
            last_num = region_group[-1].label.split('_')[-1]
            label = f"group_{first_num}-{last_num}"
        
        # Average confidence
        avg_confidence = sum(r.confidence for r in region_group) / len(region_group)
        
        # Merge metadata
        merged_metadata = {
            "type": "grouped",
            "source_regions": len(region_group),
            "original_labels": [r.label for r in region_group],
            "group_span": end_time - start_time
        }
        
        # Include original metadata if all regions have the same type
        original_types = [r.metadata.get("type") if r.metadata else None for r in region_group]
        if len(set(original_types)) == 1 and original_types[0] is not None:
            merged_metadata["original_type"] = original_types[0]
        
        return TimingRegion(
            start_time=start_time,
            end_time=end_time,
            label=label,
            confidence=avg_confidence,
            metadata=merged_metadata
        )
    
    def format_timing_for_f5tts(self, regions: List[TimingRegion]) -> str:
        """
        Format timing regions for F5-TTS edit node.
        
        Args:
            regions: List of TimingRegion objects
            
        Returns:
            Formatted string for F5-TTS edit node (start,end format)
        """
        lines = []
        for region in regions:
            lines.append(f"{region.start_time:.3f},{region.end_time:.3f}")
        return "\n".join(lines)
    
    def export_timing_data(self, regions: List[TimingRegion], 
                          format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export timing data in various formats.
        
        Args:
            regions: List of TimingRegion objects
            format: Export format ("json", "csv", "f5tts")
            
        Returns:
            Formatted timing data
        """
        if format == "json":
            return {
                "regions": [
                    {
                        "start_time": float(region.start_time),
                        "end_time": float(region.end_time),
                        "duration": float(region.end_time - region.start_time),
                        "label": str(region.label),
                        "confidence": float(region.confidence),
                        "metadata": region.metadata or {}
                    }
                    for region in regions
                ],
                "total_regions": len(regions),
                "export_format": "json"
            }
        elif format == "csv":
            lines = ["start_time,end_time,duration,label,confidence"]
            for region in regions:
                duration = region.end_time - region.start_time
                lines.append(f"{region.start_time:.3f},{region.end_time:.3f},{duration:.3f},{region.label},{region.confidence:.3f}")
            return "\n".join(lines)
        elif format == "f5tts":
            return self.format_timing_for_f5tts(regions)
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    def save_audio_segment(self, audio: torch.Tensor, start_time: float, 
                          end_time: float, output_path: str):
        """
        Save audio segment to file.
        
        Args:
            audio: Audio tensor
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output file path
        """
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        # Extract segment
        segment = audio[start_sample:end_sample]
        
        # Ensure proper dimensions for saving
        if segment.dim() == 1:
            segment = segment.unsqueeze(0)
        
        torchaudio.save(output_path, segment, self.sample_rate)


class AudioAnalysisCache:
    """
    Cache system for audio analysis results to improve performance.
    """
    
    def __init__(self, max_size: int = 50):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached items
        """
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_order: List[str] = []
    
    def _generate_key(self, audio_path: str, **kwargs) -> str:
        """Generate cache key from audio path and parameters."""
        import hashlib
        key_data = f"{audio_path}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, audio_path: str, **kwargs) -> Optional[Any]:
        """Get cached analysis result."""
        key = self._generate_key(audio_path, **kwargs)
        
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        return None
    
    def put(self, audio_path: str, result: Any, **kwargs):
        """Cache analysis result."""
        key = self._generate_key(audio_path, **kwargs)
        
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        # Update cache
        self.cache[key] = result
        
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()


# Global cache instance
analysis_cache = AudioAnalysisCache()