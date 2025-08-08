"""
Merge Audio Node - Advanced audio mixing and merging for TTS Audio Suite  
Combines multiple audio sources with various mixing algorithms
Adapted from reference RVC implementation for TTS Suite integration
"""

import os
import sys
import importlib.util
from typing import Dict, Any, Tuple, Optional, List
import hashlib

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load base_node module directly
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseTTSNode = base_module.BaseTTSNode

from utils.audio.processing import AudioProcessingUtils

# AnyType for flexible input types
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


class MergeAudioNode(BaseTTSNode):
    """
    Merge Audio Node - Advanced audio mixing and merging.
    Combines multiple audio sources using various mathematical algorithms.
    Supports multiple mixing modes and quality controls.
    """
    
    @classmethod
    def NAME(cls):
        return "🎵 Merge Audio"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Mixing algorithms
        merge_options = [
            "mean",      # Average of all inputs
            "median",    # Median value mixing (reduces outliers)  
            "max",       # Maximum amplitude mixing
            "min",       # Minimum amplitude mixing
            "sum",       # Simple addition (may need normalization)
            "overlay",   # Layer audio sources
            "crossfade", # Smooth crossfade between sources
            "weighted"   # Weighted average mixing
        ]
        
        # Sample rate options
        sample_rates = ["auto", 16000, 22050, 44100, 48000]
        
        # Output formats
        output_formats = ["wav", "flac", "mp3"]
        
        return {
            "required": {
                "audio1": (any_typ, {
                    "tooltip": "Primary audio input. Accepts AUDIO format."
                }),
                "audio2": (any_typ, {
                    "tooltip": "Secondary audio input. Accepts AUDIO format."
                }),
                "merge_algorithm": (merge_options, {
                    "default": "mean",
                    "tooltip": "Mixing algorithm. Mean=balanced mix, Median=outlier reduction, Max=prominence mixing"
                }),
            },
            "optional": {
                "audio3": (any_typ, {
                    "tooltip": "Optional third audio input"
                }),
                "audio4": (any_typ, {
                    "tooltip": "Optional fourth audio input"
                }),
                "sample_rate": (sample_rates, {
                    "default": "auto",
                    "tooltip": "Output sample rate. Auto=use highest input rate"
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize output to prevent clipping"
                }),
                "crossfade_duration": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Crossfade duration in seconds (for crossfade algorithm)"
                }),
                "volume_balance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Volume balance between audio1 (0.0) and audio2 (1.0)"
                }),
                "output_format": (output_formats, {
                    "default": "wav",
                    "tooltip": "Output audio format"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("merged_audio", "merge_info")
    
    CATEGORY = "🎵 TTS Audio Suite/Audio"
    
    FUNCTION = "merge_audio"
    
    DESCRIPTION = """
    Merge Audio - Advanced audio mixing and combining
    
    Combines multiple audio sources using sophisticated mixing algorithms.
    Perfect for layering TTS voices, adding background music, or creating complex soundscapes.
    
    Key Features:
    • Multiple mixing algorithms (mean, median, max, overlay, crossfade)
    • Up to 4 audio input support
    • Automatic sample rate handling
    • Volume balance controls
    • Crossfade transitions
    • Normalization to prevent clipping
    
    Algorithm Guide:
    • Mean: Balanced average of all inputs
    • Median: Reduces outliers and noise
    • Max: Emphasizes loudest elements
    • Overlay: Natural audio layering
    • Crossfade: Smooth transitions between sources
    • Weighted: Custom balance between sources
    """
    
    def merge_audio(
        self,
        audio1,
        audio2,
        merge_algorithm="mean",
        audio3=None,
        audio4=None,
        sample_rate="auto",
        normalize=True,
        crossfade_duration=0.1,
        volume_balance=0.5,
        output_format="wav"
    ):
        """
        Merge multiple audio sources using specified algorithm.
        
        Args:
            audio1: Primary audio input
            audio2: Secondary audio input  
            merge_algorithm: Mixing algorithm to use
            audio3: Optional third audio input
            audio4: Optional fourth audio input
            sample_rate: Output sample rate ("auto" or specific rate)
            normalize: Whether to normalize output
            crossfade_duration: Crossfade duration for crossfade algorithm
            volume_balance: Balance between audio1 and audio2
            output_format: Output audio format
            
        Returns:
            Tuple of (merged_audio, merge_info)
        """
        try:
            print(f"🎵 Merge Audio: Starting {merge_algorithm} merge")
            
            # Collect non-None audio inputs
            audio_inputs = [audio for audio in [audio1, audio2, audio3, audio4] if audio is not None]
            
            if len(audio_inputs) < 2:
                raise ValueError("At least 2 audio inputs are required for merging")
            
            print(f"Merging {len(audio_inputs)} audio sources")
            
            # Convert all inputs to processing format
            processed_audios = []
            sample_rates = []
            
            for i, audio in enumerate(audio_inputs):
                if not self._validate_audio_input(audio):
                    raise ValueError(f"Invalid audio input format for input {i+1}")
                
                audio_data, sr = self._convert_input_audio(audio)
                processed_audios.append(audio_data)
                sample_rates.append(sr)
            
            # Determine output sample rate
            if sample_rate == "auto":
                target_sr = max(sample_rates)  # Use highest input sample rate
            else:
                target_sr = int(sample_rate)
            
            # Resample all audio to target sample rate and align lengths
            aligned_audios = self._align_and_resample_audio(
                processed_audios, sample_rates, target_sr
            )
            
            # Apply mixing algorithm
            merged_audio = self._apply_merge_algorithm(
                aligned_audios, merge_algorithm, volume_balance, crossfade_duration
            )
            
            # Apply normalization if requested
            if normalize:
                merged_audio = self._normalize_audio(merged_audio)
            
            # Convert back to ComfyUI format
            merged_output = self._convert_output_audio(merged_audio, target_sr)
            
            # Create merge info
            merge_info = (
                f"Audio Merge: {merge_algorithm} algorithm | "
                f"Sources: {len(audio_inputs)} | "
                f"Sample Rate: {target_sr}Hz | "
                f"Normalized: {normalize} | "
                f"Format: {output_format}"
            )
            
            print(f"✅ Audio merge completed successfully")
            return merged_output, merge_info
            
        except Exception as e:
            print(f"❌ Audio merge failed: {e}")
            # Return primary audio on error
            error_info = f"Audio Merge Error: {str(e)} - Returning primary audio"
            return audio1, error_info
    
    def _validate_audio_input(self, audio) -> bool:
        """Validate audio input format."""
        if isinstance(audio, dict) and "waveform" in audio:
            return True
        return False
    
    def _convert_input_audio(self, audio) -> Tuple:
        """Convert ComfyUI audio to processing format."""
        try:
            import torch
            import numpy as np
            
            waveform = audio["waveform"]
            sample_rate = audio.get("sample_rate", 44100)
            
            # Convert to numpy
            if isinstance(waveform, torch.Tensor):
                audio_np = waveform.detach().cpu().numpy()
            else:
                audio_np = np.array(waveform)
            
            # Handle different input shapes
            if audio_np.ndim == 3:  # (batch, channels, samples)
                audio_np = audio_np[0]  # Take first batch
            
            if audio_np.ndim == 2:
                if audio_np.shape[0] == 1:  # (1, samples) - mono
                    audio_np = audio_np[0]
                elif audio_np.shape[0] == 2:  # (2, samples) - stereo
                    audio_np = audio_np.mean(axis=0)  # Convert to mono for mixing
                else:  # (samples, channels)
                    audio_np = audio_np.mean(axis=1)
            
            return audio_np, sample_rate
            
        except Exception as e:
            raise ValueError(f"Failed to convert input audio: {e}")
    
    def _align_and_resample_audio(self, audio_list, sample_rates, target_sr):
        """Align audio lengths and resample to target sample rate."""
        import numpy as np
        import scipy.signal
        
        resampled_audios = []
        
        # Resample all audio to target sample rate
        for audio, sr in zip(audio_list, sample_rates):
            if sr != target_sr:
                # Simple resampling (in production, would use librosa or similar)
                resample_ratio = target_sr / sr
                new_length = int(len(audio) * resample_ratio)
                resampled = scipy.signal.resample(audio, new_length)
            else:
                resampled = audio
            
            resampled_audios.append(resampled)
        
        # Align lengths (pad shorter audio with zeros)
        max_length = max(len(audio) for audio in resampled_audios)
        
        aligned_audios = []
        for audio in resampled_audios:
            if len(audio) < max_length:
                padded = np.pad(audio, (0, max_length - len(audio)), mode='constant')
                aligned_audios.append(padded)
            else:
                aligned_audios.append(audio[:max_length])  # Truncate if longer
        
        return aligned_audios
    
    def _apply_merge_algorithm(self, aligned_audios, algorithm, volume_balance, crossfade_duration):
        """Apply the specified merging algorithm."""
        import numpy as np
        
        audio_array = np.array(aligned_audios)
        
        if algorithm == "mean":
            return np.mean(audio_array, axis=0)
        
        elif algorithm == "median":
            return np.median(audio_array, axis=0)
        
        elif algorithm == "max":
            return np.max(audio_array, axis=0)
        
        elif algorithm == "min":
            return np.min(audio_array, axis=0)
        
        elif algorithm == "sum":
            return np.sum(audio_array, axis=0)
        
        elif algorithm == "overlay":
            # Simple overlay mixing
            result = aligned_audios[0].copy()
            for audio in aligned_audios[1:]:
                result = result + audio * 0.5  # Reduce volume to prevent clipping
            return result
        
        elif algorithm == "weighted":
            # Weighted mixing based on volume_balance
            if len(aligned_audios) == 2:
                return (aligned_audios[0] * (1 - volume_balance) + 
                       aligned_audios[1] * volume_balance)
            else:
                # For more than 2 sources, use mean with slight weighting
                weights = [1.0] + [volume_balance] * (len(aligned_audios) - 1)
                weights = np.array(weights) / np.sum(weights)  # Normalize weights
                return np.average(audio_array, axis=0, weights=weights)
        
        elif algorithm == "crossfade":
            # Simple crossfade between sources
            if len(aligned_audios) >= 2:
                result = aligned_audios[0].copy()
                fade_samples = int(crossfade_duration * 44100)  # Assume 44.1kHz for fade calculation
                
                for audio in aligned_audios[1:]:
                    # Apply crossfade in the middle section
                    if len(result) > fade_samples * 2:
                        mid_point = len(result) // 2
                        start_fade = mid_point - fade_samples // 2
                        end_fade = mid_point + fade_samples // 2
                        
                        fade_in = np.linspace(0, 1, fade_samples)
                        fade_out = np.linspace(1, 0, fade_samples)
                        
                        result[start_fade:end_fade] *= fade_out
                        result[start_fade:end_fade] += audio[start_fade:end_fade] * fade_in
                    else:
                        # If too short for crossfade, use simple overlay
                        result = (result + audio) * 0.5
                
                return result
            else:
                return aligned_audios[0]
        
        else:
            # Default to mean if algorithm not recognized
            print(f"⚠️ Unknown algorithm '{algorithm}', using mean")
            return np.mean(audio_array, axis=0)
    
    def _normalize_audio(self, audio):
        """Normalize audio to prevent clipping."""
        import numpy as np
        
        max_amp = np.max(np.abs(audio))
        if max_amp > 1.0:
            return audio / max_amp
        return audio
    
    def _convert_output_audio(self, audio_np, sample_rate):
        """Convert processed audio back to ComfyUI format."""
        import torch
        import numpy as np
        
        # Ensure proper data type and range
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        
        # Convert to tensor in ComfyUI format (batch, channels, samples)
        if audio_np.ndim == 1:
            waveform = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
        
        return {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs for audio merging."""
        return True