"""
LibROSA Fallback Functions for Python 3.13 Compatibility

This module provides fallback implementations for librosa functions that fail on Python 3.13
due to numba compatibility issues. It tries librosa first, then falls back to torchaudio/numpy
implementations if librosa fails.
"""

import sys
import numpy as np
import torch
import torchaudio
import warnings


def safe_load(file_path, sr=None, mono=True):
    """
    Safe audio loading with librosa fallback to torchaudio
    
    Args:
        file_path: Path to audio file or BytesIO object
        sr: Target sample rate (None for original)
        mono: Convert to mono if True
    
    Returns:
        (audio_array, sample_rate): NumPy array and sample rate
    """
    try:
        # Try librosa first (best quality)
        import librosa
        return librosa.load(file_path, sr=sr, mono=mono)
    except Exception as e:
        # Fallback to torchaudio for Python 3.13 compatibility
        audio_tensor, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if requested
        if mono and audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
        
        # Resample if needed
        if sr is not None and sample_rate != sr:
            resampler = torchaudio.transforms.Resample(sample_rate, sr)
            audio_tensor = resampler(audio_tensor)
            sample_rate = sr
        
        # Convert to numpy and squeeze
        audio_array = audio_tensor.squeeze().numpy()
        
        return audio_array, sample_rate


def safe_resample(audio, orig_sr, target_sr, res_type='kaiser_fast'):
    """
    Safe audio resampling with librosa fallback to torchaudio
    
    Args:
        audio: Audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate
        res_type: Resampling type (ignored in fallback)
    
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
        
    try:
        # Try librosa first (best quality)
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr, res_type=res_type)
    except Exception as e:
        # Fallback to torchaudio
        audio_tensor = torch.from_numpy(audio).unsqueeze(0) if audio.ndim == 1 else torch.from_numpy(audio)
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        resampled_tensor = resampler(audio_tensor)
        return resampled_tensor.squeeze().numpy()


def safe_trim(audio, top_db=30, frame_length=2048, hop_length=512):
    """
    Safe audio trimming with librosa fallback to simple energy-based trimming
    
    Args:
        audio: Audio array
        top_db: Threshold in dB below peak
        frame_length: Frame length (ignored in fallback)
        hop_length: Hop length (ignored in fallback)
    
    Returns:
        Trimmed audio array
    """
    try:
        # Try librosa first (best quality)
        import librosa
        return librosa.effects.trim(audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length)[0]
    except Exception as e:
        # Fallback to simple energy-based trimming
        return _simple_energy_trim(audio, top_db)


def _simple_energy_trim(audio, top_db=30):
    """Simple energy-based audio trimming fallback"""
    if len(audio) < 1024:
        return audio
        
    # Convert dB to linear scale
    threshold = 10 ** (-top_db / 20)
    
    # Calculate RMS energy in sliding windows
    window_size = 1024  # Fixed window size for simplicity
    hop_size = 512
    
    # Calculate RMS for each window
    rms_values = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i + window_size]
        rms_values.append(np.sqrt(np.mean(window**2)))
    
    if not rms_values:
        return audio
    
    max_rms = max(rms_values)
    threshold_rms = max_rms * threshold
    
    # Find start and end indices where audio is above threshold
    start_idx = 0
    end_idx = len(audio)
    
    for i, rms in enumerate(rms_values):
        if rms > threshold_rms:
            start_idx = i * hop_size
            break
    
    for i, rms in enumerate(reversed(rms_values)):
        if rms > threshold_rms:
            end_idx = len(audio) - i * hop_size
            break
    
    # Ensure we don't trim too aggressively
    start_idx = max(0, min(start_idx, len(audio) // 4))
    end_idx = min(len(audio), max(end_idx, 3 * len(audio) // 4))
    
    return audio[start_idx:end_idx]


def safe_mel_filters(sr, n_fft, n_mels=80, fmin=0.0, fmax=None):
    """
    Safe mel filter bank creation with librosa fallback to torchaudio
    
    Args:
        sr: Sample rate
        n_fft: FFT size
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
    
    Returns:
        Mel filter bank matrix
    """
    try:
        # Try librosa first (best quality)
        import librosa
        return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    except Exception as e:
        # Fallback to torchaudio
        if fmax is None:
            fmax = sr // 2
        mel_transform = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=sr, f_min=fmin, f_max=fmax, n_stft=n_fft // 2 + 1)
        return mel_transform.fb.numpy()


def safe_stft(audio, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect'):
    """
    Safe STFT computation with librosa fallback to torch
    
    Args:
        audio: Audio array
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        window: Window type
        center: Center the audio
        pad_mode: Padding mode (only used with librosa)
    
    Returns:
        STFT matrix
    """
    try:
        # Try librosa first (best quality)
        import librosa
        return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode)
    except Exception as e:
        # Fallback to torch STFT (pad_mode not supported)
        if win_length is None:
            win_length = n_fft
        
        audio_tensor = torch.from_numpy(audio).float()
        stft_result = torch.stft(
            audio_tensor, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length,
            window=torch.hann_window(win_length),
            center=center,
            return_complex=True
        )
        return stft_result.numpy()


def get_python313_status():
    """Check if running on Python 3.13+"""
    return sys.version_info >= (3, 13)


def log_fallback_usage(function_name, reason="librosa compatibility"):
    """Log when fallback functions are used"""
    if get_python313_status():
        print(f"🔄 Using torchaudio fallback for {function_name} ({reason})")