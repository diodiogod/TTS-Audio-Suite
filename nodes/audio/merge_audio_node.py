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
import numpy as np
import torch
import librosa
import scipy.signal

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
        return "🥪 Merge Audio"
    
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
        sample_rates = ["auto", 16000, 22050, 24000, 44100, 48000]
        
        
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
                    "tooltip": """🎚️ AUDIO MERGING ALGORITHMS

Choose how to combine multiple audio sources:

📊 MATHEMATICAL ALGORITHMS:
• MEAN: ⭐ Average all inputs - smooth, balanced mix (recommended)
• MEDIAN: Reduces outliers and noise - cleaner output, less distortion
• MAX: Takes loudest signal at each point - preserves peaks, can be harsh
• MIN: Takes quietest signal at each point - gentle, subdued mix
• SUM: Simple addition - louder but may clip without normalization

🎵 CREATIVE ALGORITHMS:
• OVERLAY: Professional layering with gain compensation - natural mixing
• WEIGHTED: Custom balance using volume_balance slider - precise control
• CROSSFADE: Smooth transitions between sources - cinematic blending

💡 USE CASES:
🎤 Voice + Music: MEAN or OVERLAY
🎼 Multiple instruments: MEDIAN (reduces conflicts)
🔊 Emphasize loudest: MAX
🎛️ Custom balance: WEIGHTED
🎬 Smooth transitions: CROSSFADE
🔧 Precise control: SUM + normalization"""
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
                    "tooltip": "Output sample rate - Auto=use highest input rate, Higher=better quality but larger files"
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize output volume - ON=prevents distortion/clipping, OFF=preserves original levels"
                }),
                "crossfade_duration": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Crossfade blend duration in seconds (for crossfade algorithm) - Low=quick transitions, High=smooth gradual blending"
                }),
                "volume_balance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Volume balance between audio1 (0.0) and audio2 (1.0) - ONLY used by 'weighted' algorithm"
                }),
                # Instrumental Pitch Control (Replay parity feature)
                "vocal_pitch_shift": ("FLOAT", {
                    "default": 0.0,
                    "min": -24.0,
                    "max": 24.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Change vocal pitch - Negative=deeper voice (like man), Positive=higher voice (like child), 12=twice as high"
                }),
                "instrumental_pitch_shift": ("FLOAT", {
                    "default": 0.0,
                    "min": -24.0,
                    "max": 24.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Change music pitch - Negative=deeper/slower sound, Positive=higher/faster sound, 12=twice as high"
                }),
                "enable_pitch_control": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable pitch shifting - ON=allows separate pitch control for vocal/instrumental, OFF=no pitch changes"
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("merged_audio", "merge_info")
    
    CATEGORY = "TTS Audio Suite/🎵 Audio Processing"
    
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
        vocal_pitch_shift=0.0,
        instrumental_pitch_shift=0.0,
        enable_pitch_control=False
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
            
        Returns:
            Tuple of (merged_audio, merge_info)
        """
        try:
            print(f"🥪 Merge Audio: Starting {merge_algorithm} merge")
            
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
            
            # Apply instrumental pitch control if enabled (Replay parity feature)
            if enable_pitch_control and (vocal_pitch_shift != 0.0 or instrumental_pitch_shift != 0.0):
                print(f"🎵 Applying pitch control: vocal={vocal_pitch_shift:.1f}, instrumental={instrumental_pitch_shift:.1f} semitones")
                processed_audios = self._apply_pitch_shifting(
                    processed_audios, 
                    sample_rates, 
                    vocal_pitch_shift, 
                    instrumental_pitch_shift
                )
            
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
                f"Normalized: {normalize}"
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
            # Use _get_audio for VideoHelper compatibility
            normalized_audio = self._get_audio(audio, "input_audio")
            
            waveform = normalized_audio["waveform"]
            sample_rate = normalized_audio.get("sample_rate", 44100)
            
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
                    print("⚠️ Converting stereo to mono for mixing - stereo information will be lost")
                    audio_np = audio_np.mean(axis=0)  # Convert to mono for mixing
                else:  # (samples, channels)
                    print(f"⚠️ Converting {audio_np.shape[0]}-channel audio to mono - spatial information will be lost")
                    audio_np = audio_np.mean(axis=1)
            
            return audio_np, sample_rate
            
        except Exception as e:
            raise ValueError(f"Failed to convert input audio: {e}")
    
    def _align_and_resample_audio(self, audio_list, sample_rates, target_sr):
        """Align audio lengths and resample to target sample rate."""
        
        resampled_audios = []
        
        # Resample all audio to target sample rate
        for i, (audio, sr) in enumerate(zip(audio_list, sample_rates)):
            if sr != target_sr:
                print(f"⚠️ Resampling audio {i+1}: {sr}Hz → {target_sr}Hz - may introduce artifacts")
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
        for i, audio in enumerate(resampled_audios):
            if len(audio) < max_length:
                pad_seconds = (max_length - len(audio)) / target_sr
                if pad_seconds > 0.1:  # Only warn for significant padding
                    print(f"⚠️ Padding audio {i+1} with {pad_seconds:.2f}s of silence")
                padded = np.pad(audio, (0, max_length - len(audio)), mode='constant')
                aligned_audios.append(padded)
            elif len(audio) > max_length:
                truncate_seconds = (len(audio) - max_length) / target_sr
                if truncate_seconds > 0.1:  # Only warn for significant truncation
                    print(f"⚠️ Truncating audio {i+1} by {truncate_seconds:.2f}s")
                aligned_audios.append(audio[:max_length])  # Truncate if longer
            else:
                aligned_audios.append(audio)
        
        return aligned_audios
    
    def _apply_merge_algorithm(self, aligned_audios, algorithm, volume_balance, crossfade_duration):
        """Apply the specified merging algorithm using real implementations."""
        
        try:
            # Use the same merge functions as the reference implementation
            audio_array = np.array(aligned_audios)
            
            if algorithm == "mean":
                return self._get_merge_func("mean")(audio_array, axis=0)
            
            elif algorithm == "median":
                return self._get_merge_func("median")(audio_array, axis=0)
            
            elif algorithm == "max":
                return self._get_merge_func("max")(audio_array, axis=0)
            
            elif algorithm == "min":
                return self._get_merge_func("min")(audio_array, axis=0)
            
            elif algorithm == "sum":
                return np.sum(audio_array, axis=0)
            
            elif algorithm == "overlay":
                # Advanced overlay mixing with proper gain staging
                return self._overlay_mix(aligned_audios)
            
            elif algorithm == "weighted":
                # Weighted mixing based on volume_balance
                return self._weighted_mix(aligned_audios, volume_balance)
            
            elif algorithm == "crossfade":
                # Professional crossfade implementation
                return self._crossfade_mix(aligned_audios, crossfade_duration)
            
            else:
                # Default to mean if algorithm not recognized
                print(f"⚠️ Unknown algorithm '{algorithm}', using mean")
                return self._get_merge_func("mean")(audio_array, axis=0)
                
        except Exception as e:
            print(f"Merge algorithm error: {e}, using fallback")
            # Fallback to simple mean
            return np.mean(audio_array, axis=0)

    def _get_merge_func(self, merge_type: str):
        """Get merge function following reference implementation"""
        if merge_type == "min":
            return np.nanmin
        elif merge_type == "max": 
            return np.nanmax
        elif merge_type == "median":
            return np.nanmedian
        else:
            return np.nanmean

    def _pad_audio(self, *audios, axis=0):
        """Pad audio arrays to same length (from reference implementation)"""
        try:
            maxlen = max(len(a) if a is not None else 0 for a in audios)
            if maxlen > 0:
                stack = librosa.util.stack([
                    librosa.util.fix_length(a, size=maxlen) 
                    for a in audios if a is not None
                ], axis=axis)
                return stack
            else:
                return np.stack(audios, axis=axis)
        except ImportError:
            # Fallback without librosa
            maxlen = max(len(a) for a in audios if a is not None)
            padded_audios = []
            for audio in audios:
                if audio is not None:
                    if len(audio) < maxlen:
                        padded = np.pad(audio, (0, maxlen - len(audio)), mode='constant')
                    else:
                        padded = audio[:maxlen]
                    padded_audios.append(padded)
            return np.stack(padded_audios, axis=axis)

    def _overlay_mix(self, aligned_audios):
        """Professional overlay mixing with gain compensation"""
        result = aligned_audios[0].copy()
        
        # Calculate gain compensation based on number of sources
        gain_compensation = 1.0 / np.sqrt(len(aligned_audios))
        
        for i, audio in enumerate(aligned_audios[1:], 1):
            # Progressive volume reduction to prevent clipping
            mix_ratio = 1.0 / (i + 1)
            result = result * (1 - mix_ratio) + audio * mix_ratio
        
        # Apply gain compensation
        result *= gain_compensation
        
        return result

    def _weighted_mix(self, aligned_audios, volume_balance):
        """Weighted mixing with proper balance control"""
        if len(aligned_audios) == 2:
            # Simple two-source weighting
            return (aligned_audios[0] * (1 - volume_balance) + 
                   aligned_audios[1] * volume_balance)
        else:
            # Multi-source weighting
            weights = [1.0] + [volume_balance] * (len(aligned_audios) - 1)
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            
            result = np.zeros_like(aligned_audios[0])
            for i, audio in enumerate(aligned_audios):
                result += audio * weights[i]
            
            return result

    def _crossfade_mix(self, aligned_audios, crossfade_duration):
        """Professional crossfade implementation"""
        if len(aligned_audios) < 2:
            return aligned_audios[0]
            
        result = aligned_audios[0].copy()
        
        for audio in aligned_audios[1:]:
            result = self._crossfade_two_sources(result, audio, crossfade_duration)
        
        return result

    def _crossfade_two_sources(self, audio1, audio2, crossfade_duration):
        """Crossfade between two audio sources"""
        # Calculate crossfade samples (assume 44.1kHz sample rate)
        fade_samples = int(crossfade_duration * 44100)
        audio_length = min(len(audio1), len(audio2))
        
        if fade_samples >= audio_length // 2:
            # If crossfade too long, use simple overlay
            return (audio1 + audio2) * 0.5
        
        # Create crossfade in the middle of the audio
        mid_point = audio_length // 2
        start_fade = max(0, mid_point - fade_samples // 2)
        end_fade = min(audio_length, mid_point + fade_samples // 2)
        fade_length = end_fade - start_fade
        
        # Create fade curves
        fade_out = np.linspace(1, 0, fade_length)
        fade_in = np.linspace(0, 1, fade_length)
        
        # Apply crossfade
        result = audio1.copy()
        result[start_fade:end_fade] = (
            audio1[start_fade:end_fade] * fade_out + 
            audio2[start_fade:end_fade] * fade_in
        )
        
        return result
    
    def _normalize_audio(self, audio):
        """Normalize audio to prevent clipping."""
        
        max_amp = np.max(np.abs(audio))
        if max_amp > 1.0:
            return audio / max_amp
        return audio
    
    def _convert_output_audio(self, audio_np, sample_rate):
        """Convert processed audio back to ComfyUI format."""
        
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
    
    def _apply_pitch_shifting(self, processed_audios, sample_rates, vocal_pitch_shift, instrumental_pitch_shift):
        """
        Apply separate pitch shifting to vocal (audio1) and instrumental (audio2) tracks.
        Implements Replay's Instrumental Pitch Control feature.
        """
        try:
            # Apply pitch shifts to the first two audio tracks
            # audio1 = vocal (gets vocal_pitch_shift)
            # audio2 = instrumental (gets instrumental_pitch_shift)
            pitch_shifted_audios = []
            
            for i, (audio_data, sr) in enumerate(zip(processed_audios, sample_rates)):
                if i == 0 and vocal_pitch_shift != 0.0:
                    # Apply vocal pitch shift to first audio (typically the vocal track)
                    print(f"🎤 Pitch shifting vocal track: {vocal_pitch_shift:.1f} semitones")
                    shifted_audio = librosa.effects.pitch_shift(
                        y=audio_data.astype(np.float32),
                        sr=sr,
                        n_steps=vocal_pitch_shift
                    )
                    pitch_shifted_audios.append(shifted_audio)
                    
                elif i == 1 and instrumental_pitch_shift != 0.0:
                    # Apply instrumental pitch shift to second audio (typically the instrumental track)
                    print(f"🎼 Pitch shifting instrumental track: {instrumental_pitch_shift:.1f} semitones")
                    shifted_audio = librosa.effects.pitch_shift(
                        y=audio_data.astype(np.float32),
                        sr=sr,
                        n_steps=instrumental_pitch_shift
                    )
                    pitch_shifted_audios.append(shifted_audio)
                    
                else:
                    # No pitch shift for this track or additional tracks beyond audio1/audio2
                    pitch_shifted_audios.append(audio_data)
            
            return pitch_shifted_audios
            
        except ImportError:
            print("⚠️ Pitch shifting requires librosa - skipping pitch control")
            return processed_audios
        except Exception as e:
            print(f"⚠️ Pitch shifting failed: {e} - using original audio")
            return processed_audios
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs for audio merging."""
        return True