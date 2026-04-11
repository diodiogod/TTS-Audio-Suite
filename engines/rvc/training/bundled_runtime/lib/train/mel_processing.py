import functools
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn


MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


# Reusable banks
@functools.lru_cache(None)
def get_mel_filters(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float):
    return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

@functools.lru_cache(None)
def get_hann_window(win_size: int):
    torch.windows.hann
    return torch.hann_window(win_size)


def spectrogram_torch(y: torch.Tensor, n_fft: int, hop_size: int, win_size: int, center=False):
    """Convert waveform into Linear-frequency Linear-amplitude spectrogram.

    Returns:
        :: (B, Freq, Frame) - Linear-frequency Linear-amplitude spectrogram
    """
    # Validation
    if torch.min(y) < -1.05:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.05:
        print("max value is ", torch.max(y))
    y = y.clamp(min=-1.05, max=1.05)

    # Window - Cache if needed
    hann_window = get_hann_window(win_size).to(device=y.device)

    # Padding
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # Complex Spectrogram :: (B, T) -> (B, Freq, Frame, RealComplex=2)
    spec = torch.view_as_real(torch.stft(
        y.float(),
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    ))

    # Linear-frequency Linear-amplitude spectrogram :: (B, Freq, Frame, RealComplex=2) -> (B, Freq, Frame)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-8)
    return spec.to(device=y.device,dtype=y.dtype)

def spec_to_mel_torch(spec: torch.Tensor, n_fft: int, num_mels: int, sampling_rate: int, fmin: float, fmax: float):
    mel = get_mel_filters(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)

    # Mel-frequency Log-amplitude spectrogram :: (B, Freq=num_mels, Frame)
    mel_basis = torch.from_numpy(mel).to(device=spec.device, dtype=spec.dtype)
    melspec = torch.matmul(mel_basis, spec)
    melspec = spectral_normalize_torch(melspec)
    return melspec


# def mel_spectrogram_torch(
#     y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
# ):
#     """Convert waveform into Mel-frequency Log-amplitude spectrogram.

#     Args:
#         y       :: (B, T)           - Waveforms
#     Returns:
#         melspec :: (B, Freq, Frame) - Mel-frequency Log-amplitude spectrogram
#     """
#     # Linear-frequency Linear-amplitude spectrogram :: (B, T) -> (B, Freq, Frame)
#     spec = spectrogram_torch(y, n_fft, hop_size, win_size, center)

#     # Mel-frequency Log-amplitude spectrogram :: (B, Freq, Frame) -> (B, Freq=num_mels, Frame)
#     melspec = spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)

#     return melspec

def mel_spectrogram_torch(
    wav: torch.Tensor, n_fft: int, n_mels: int, sampling_rate: int, hop_length: int, window_length: int, fmin: float, fmax: float, center=False
):
        """
        Mirrors AudioSignal.mel_spectrogram used by BigVGAN-v2 training from: 
        https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        """
        B, C, T = wav.shape

        # Padding
        wav = torch.nn.functional.pad(wav.reshape(-1, T),
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        )

        window = get_hann_window(window_length).to(device=wav.device)

        stft = torch.stft(
            wav.float(),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_length,
            window=window.float(),
            return_complex=True,
            center=center,
        )
        magnitude = torch.abs(stft)

        mel_basis = get_mel_filters(sampling_rate, n_fft, n_mels, fmin, fmax)
        mel_basis = torch.from_numpy(mel_basis).to(wav.device)
        mel_spectrogram = magnitude.transpose(-2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, -2)

        return spectral_normalize_torch(mel_spectrogram).to(dtype=wav.dtype)