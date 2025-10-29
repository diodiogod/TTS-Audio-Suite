import torch
import torch.nn as nn
import numpy as np
import librosa
from voicefixer_bundled.tools.modules.pqmf import PQMF


class LibrosaSTFT:
    """Librosa-based STFT wrapper for compatibility with torchlibrosa API"""
    def __init__(self, n_fft, hop_length, win_length, window, center, pad_mode, freeze_parameters=True):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def __call__(self, input_tensor):
        """Convert tensor to STFT (real, imag parts)"""
        # Convert torch tensor to numpy
        if isinstance(input_tensor, torch.Tensor):
            input_np = input_tensor.detach().cpu().numpy()
            device = input_tensor.device
            dtype = input_tensor.dtype
        else:
            input_np = input_tensor
            device = torch.device('cpu')
            dtype = torch.float32

        # Handle batch dimension
        if input_np.ndim == 2:  # [batch, samples]
            batch_size = input_np.shape[0]
            results_real = []
            results_imag = []

            for b in range(batch_size):
                D = librosa.stft(input_np[b], n_fft=self.n_fft, hop_length=self.hop_length,
                                win_length=self.win_length, window=self.window, center=self.center,
                                pad_mode=self.pad_mode)
                results_real.append(np.real(D))
                results_imag.append(np.imag(D))

            real = np.stack(results_real, axis=0)  # [batch, freq, time]
            imag = np.stack(results_imag, axis=0)
        else:  # [samples]
            D = librosa.stft(input_np, n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.win_length, window=self.window, center=self.center,
                            pad_mode=self.pad_mode)
            real = np.real(D)[np.newaxis, ...]  # Add batch dimension [1, freq, time]
            imag = np.imag(D)[np.newaxis, ...]

        # Transpose to match torchlibrosa output format: [batch, time, freq]
        # librosa gives us [batch, freq, time], so transpose last two dims
        real = np.transpose(real, (0, 2, 1))  # [batch, time, freq]
        imag = np.transpose(imag, (0, 2, 1))  # [batch, time, freq]

        # Convert back to torch tensors
        real_tensor = torch.from_numpy(real).to(device=device, dtype=dtype)
        imag_tensor = torch.from_numpy(imag).to(device=device, dtype=dtype)

        return real_tensor, imag_tensor


class LibrosaISTFT:
    """Librosa-based ISTFT wrapper for compatibility with torchlibrosa API"""
    def __init__(self, n_fft, hop_length, win_length, window, center, pad_mode, freeze_parameters=True):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def __call__(self, real_tensor, imag_tensor, length=None):
        """Convert STFT (real, imag parts) back to waveform"""
        # Convert torch tensors to numpy
        if isinstance(real_tensor, torch.Tensor):
            real_np = real_tensor.detach().cpu().numpy()
            device = real_tensor.device
            dtype = real_tensor.dtype
        else:
            real_np = real_tensor
            device = torch.device('cpu')
            dtype = torch.float32

        imag_np = imag_tensor.detach().cpu().numpy() if isinstance(imag_tensor, torch.Tensor) else imag_tensor

        # Input is [batch, time, freq] or [time, freq] from torchlibrosa format
        # Transpose back to [batch, freq, time] for librosa
        if real_np.ndim == 3:  # [batch, time, freq]
            real_np = np.transpose(real_np, (0, 2, 1))  # [batch, freq, time]
            imag_np = np.transpose(imag_np, (0, 2, 1))  # [batch, freq, time]
        else:  # [time, freq]
            real_np = real_np.T  # [freq, time]
            imag_np = imag_np.T  # [freq, time]

        # Reconstruct complex spectrogram
        D = real_np + 1j * imag_np  # [batch, freq, time] or [freq, time]

        # Handle batch dimension
        if D.ndim == 3:  # [batch, freq, time]
            batch_size = D.shape[0]
            results = []

            for b in range(batch_size):
                wav = librosa.istft(D[b], hop_length=self.hop_length, win_length=self.win_length,
                                   window=self.window, center=self.center, length=length)
                results.append(wav)

            output_np = np.stack(results, axis=0)  # [batch, samples]
        else:  # [freq, time]
            output_np = librosa.istft(D, hop_length=self.hop_length, win_length=self.win_length,
                                     window=self.window, center=self.center, length=length)
            output_np = output_np[np.newaxis, ...]  # Add batch dimension [1, samples]

        # Convert back to torch tensor
        output_tensor = torch.from_numpy(output_np).to(device=device, dtype=dtype)

        return output_tensor


class FDomainHelper(nn.Module):
    def __init__(
        self,
        window_size=2048,
        hop_size=441,
        center=True,
        pad_mode="reflect",
        window="hann",
        freeze_parameters=True,
        subband=None,
        root="/Users/admin/Documents/projects/",
    ):
        super(FDomainHelper, self).__init__()
        self.subband = subband

        if self.subband is None:
            self.stft = LibrosaSTFT(
                n_fft=window_size,
                hop_length=hop_size,
                win_length=window_size,
                window=window,
                center=center,
                pad_mode=pad_mode,
                freeze_parameters=freeze_parameters,
            )

            self.istft = LibrosaISTFT(
                n_fft=window_size,
                hop_length=hop_size,
                win_length=window_size,
                window=window,
                center=center,
                pad_mode=pad_mode,
                freeze_parameters=freeze_parameters,
            )
        else:
            self.stft = LibrosaSTFT(
                n_fft=window_size // self.subband,
                hop_length=hop_size // self.subband,
                win_length=window_size // self.subband,
                window=window,
                center=center,
                pad_mode=pad_mode,
                freeze_parameters=freeze_parameters,
            )

            self.istft = LibrosaISTFT(
                n_fft=window_size // self.subband,
                hop_length=hop_size // self.subband,
                win_length=window_size // self.subband,
                window=window,
                center=center,
                pad_mode=pad_mode,
                freeze_parameters=freeze_parameters,
            )

        if subband is not None and root is not None:
            self.qmf = PQMF(subband, 64, root)

    def complex_spectrogram(self, input, eps=0.0):
        # [batchsize, samples]
        # return [batchsize, 2, t-steps, f-bins]
        real, imag = self.stft(input)
        return torch.cat([real, imag], dim=1)

    def reverse_complex_spectrogram(self, input, eps=0.0, length=None):
        # [batchsize, 2[real,imag], t-steps, f-bins]
        wav = self.istft(input[:, 0:1, ...], input[:, 1:2, ...], length=length)
        return wav

    def spectrogram(self, input, eps=0.0):
        (real, imag) = self.stft(input.float())
        return torch.clamp(real**2 + imag**2, eps, np.inf) ** 0.5

    def spectrogram_phase(self, input, eps=0.0):
        (real, imag) = self.stft(input.float())
        mag = torch.clamp(real**2 + imag**2, eps, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def wav_to_spectrogram_phase(self, input, eps=1e-8):
        """Waveform to spectrogram.

        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        cos_list = []
        sin_list = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            mag, cos, sin = self.spectrogram_phase(input[:, channel, :], eps=eps)
            # mag/cos/sin shape: [batch, freq, time] - add channel dimension
            sp_list.append(mag.unsqueeze(1))  # [batch, 1, freq, time]
            cos_list.append(cos.unsqueeze(1))
            sin_list.append(sin.unsqueeze(1))

        # Stack to [batch, channels, freq, time]
        sps = torch.cat(sp_list, dim=1)
        coss = torch.cat(cos_list, dim=1)
        sins = torch.cat(sin_list, dim=1)
        return sps, coss, sins

    def spectrogram_phase_to_wav(self, sps, coss, sins, length):
        channels_num = sps.size()[1]
        res = []
        for i in range(channels_num):
            res.append(
                self.istft(
                    sps[:, i : i + 1, ...] * coss[:, i : i + 1, ...],
                    sps[:, i : i + 1, ...] * sins[:, i : i + 1, ...],
                    length,
                )
            )
            res[-1] = res[-1].unsqueeze(1)
        return torch.cat(res, dim=1)

    def wav_to_spectrogram(self, input, eps=1e-8):
        """Waveform to spectrogram.

        Args:
          input: (batch_size,channels_num, segment_samples)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            sp_list.append(self.spectrogram(input[:, channel, :], eps=eps))
        output = torch.cat(sp_list, dim=1)
        return output

    def spectrogram_to_wav(self, input, spectrogram, length=None):
        """Spectrogram to waveform.
        Args:
          input: (batch_size, segment_samples, channels_num)
          spectrogram: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, segment_samples, channels_num)
        """
        channels_num = input.shape[1]
        wav_list = []
        for channel in range(channels_num):
            (real, imag) = self.stft(input[:, channel, :])
            (_, cos, sin) = magphase(real, imag)
            wav_list.append(
                self.istft(
                    spectrogram[:, channel : channel + 1, :, :] * cos,
                    spectrogram[:, channel : channel + 1, :, :] * sin,
                    length,
                )
            )

        output = torch.stack(wav_list, dim=1)
        return output

    # todo the following code is not bug free!
    def wav_to_complex_spectrogram(self, input, eps=0.0):
        # [batchsize , channels, samples]
        # [batchsize, 2[real,imag]*channels, t-steps, f-bins]
        res = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            res.append(self.complex_spectrogram(input[:, channel, :], eps=eps))
        return torch.cat(res, dim=1)

    def complex_spectrogram_to_wav(self, input, eps=0.0, length=None):
        # [batchsize, 2[real,imag]*channels, t-steps, f-bins]
        # return  [batchsize, channels, samples]
        channels = input.size()[1] // 2
        wavs = []
        for i in range(channels):
            wavs.append(
                self.reverse_complex_spectrogram(
                    input[:, 2 * i : 2 * i + 2, ...], eps=eps, length=length
                )
            )
            wavs[-1] = wavs[-1].unsqueeze(1)
        return torch.cat(wavs, dim=1)

    def wav_to_complex_subband_spectrogram(self, input, eps=0.0):
        # [batchsize, channels, samples]
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        subwav = self.qmf.analysis(input)  # [batchsize, subband*channels, samples]
        subspec = self.wav_to_complex_spectrogram(subwav)
        return subspec

    def complex_subband_spectrogram_to_wav(self, input, eps=0.0):
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        # [batchsize, channels, samples]
        subwav = self.complex_spectrogram_to_wav(input)
        data = self.qmf.synthesis(subwav)
        return data

    def wav_to_mag_phase_subband_spectrogram(self, input, eps=1e-8):
        """
        :param input:
        :param eps:
        :return:
            loss = torch.nn.L1Loss()
            models = FDomainHelper(subband=4)
            data = torch.randn((3,1, 44100*3))

            sps, coss, sins = models.wav_to_mag_phase_subband_spectrogram(data)
            wav = models.mag_phase_subband_spectrogram_to_wav(sps,coss,sins,44100*3//4)

            print(loss(data,wav))
            print(torch.max(torch.abs(data-wav)))

        """
        # [batchsize, channels, samples]
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        subwav = self.qmf.analysis(input)  # [batchsize, subband*channels, samples]
        sps, coss, sins = self.wav_to_spectrogram_phase(subwav, eps=eps)
        return sps, coss, sins

    def mag_phase_subband_spectrogram_to_wav(self, sps, coss, sins, length, eps=0.0):
        # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
        # [batchsize, channels, samples]
        subwav = self.spectrogram_phase_to_wav(
            sps, coss, sins, length + self.qmf.pad_samples // self.qmf.N
        )
        data = self.qmf.synthesis(subwav)
        return data
