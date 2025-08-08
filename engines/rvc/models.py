"""
RVC Model Architectures - Essential classes for voice conversion
Extracted from reference implementation for TTS Suite integration
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
import numpy as np

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = len(dilation)
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[j],
                              padding=self.get_padding(kernel_size, dilation[j])))
            for j in range(self.h)
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                              padding=self.get_padding(kernel_size, 1)))
            for j in range(self.h)
        ])

    def get_padding(self, kernel_size, dilation=1):
        return int((kernel_size*dilation - dilation)/2)

    def forward(self, x):
        for j in range(self.h):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = self.convs1[j](xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = self.convs2[j](xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.convs2:
            torch.nn.utils.remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = len(dilation)
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[j],
                              padding=self.get_padding(kernel_size, dilation[j])))
            for j in range(self.h)
        ])

    def get_padding(self, kernel_size, dilation=1):
        return int((kernel_size*dilation - dilation)/2)

    def forward(self, x):
        for j in range(self.h):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = self.convs[j](xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            torch.nn.utils.remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                               k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(self.init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def init_weights(self, m, mean=0.0, std=0.01):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(mean, std)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0, 
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_threshold: F0 threshold for U/V classification (default 0)
    Sine_source, noise_source = SineGen(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    upsample_scale: = sr / F0_rate (default 320)
    """
    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The integer part n can be ignored
        # because 2 * pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], \
                             device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for sine generation
            tmp_over_one = torch.cumsum(rad_values, 1)  # B x T x dim
            tmp_over_one *= 2 * np.pi
            tmp_over_one = torch.sin(tmp_over_one)
            # different component has different amplitude
            tmp_over_one[:, :, 0] = tmp_over_one[:, :, 0] * self.sine_amp
            for idx in np.arange(self.harmonic_num):
                tmp_over_one[:, :, idx + 1] = tmp_over_one[:, :, idx + 1] * self.sine_amp / (2 * (idx + 2))
            # sum
            sines = tmp_over_one.sum(-1, keepdim=True)
        else:
            # for pulse generation
            # make sure that the first priod is ignored
            # since the pulse is always periodic
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different component has different amplitude
            for idx in np.arange(self.harmonic_num):
                tmp_over_one[:, :, idx + 1] = tmp_over_one[:, :, idx + 1] * self.sine_amp / (idx + 2)
            # to prevent the safe_log error
            tmp_cumsum = tmp_cumsum + 1e-10
            # get the sine waveforms
            sines = torch.sin(tmp_cumsum)
            sines = sines.sum(-1, keepdim=True)
            sines = sines / (self.harmonic_num + 1)
        return sines

    def forward(self, f0, upp):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            # Ensure f0 is float for interpolation operations
            if f0.dtype != torch.float32:
                f0 = f0.float()
            f0 = f0[:, None].transpose(1, 2)  # B x 1 x T
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], f0.shape[2] * upp, 
                                device=f0.device, dtype=torch.float32)
            # fundamental component
            f0_buf[:, :, ::upp] = f0

            for i in np.arange(self.harmonic_num):
                # overtone component
                f0_buf[:, :, ::upp] += f0 * (2 * (i + 2))  # Check this later

            # interpolate F0
            f0_buf = torch.nn.functional.interpolate(f0_buf,
                                                   size=f0.shape[2] * upp,
                                                   mode='linear',
                                                   align_corners=True)
            f0_buf = f0_buf.transpose(1, 2).reshape(f0_buf.shape[0], f0_buf.shape[2], f0_buf.shape[1])

            # generate uv signal
            # uv = torch.ones(f0_buf.shape)   # init as voiced
            uv = self._f02uv(f0_buf)

            # generate sine waveforms
            if self.harmonic_num > 0:
                sine_wavs = self._f02sine(f0_buf) * uv
            else:
                sine_wavs = torch.sin(f0_buf * 2 * np.pi / self.sampling_rate) * uv
            # generate Gaussian noise
            # noise_amp = uv * self.noise_std + (1-uv) * self.sine_amp / 3
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp
            noise = noise_amp * torch.randn_like(sine_wavs)
            sine_wavs = sine_wavs * self.sine_amp + noise
        return sine_wavs, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for harmonic-plus-noise excitation signal model
    """
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_flag=False, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = 0.003
        self.harmonic_num = harmonic_num
        self.sampling_rate = sampling_rate
        self.voiced_threshod = voiced_threshod

        # to generate sine waveforms
        self.l_sin_gen = SineGen(self.sampling_rate, self.harmonic_num,
                                self.sine_amp, self.noise_std, self.voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(self.harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None  # noise, uv


# Simplified RVC Synthesizer models
class SynthesizerTrnMs256NSFsid(nn.Module):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, 
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout, 
                 resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 spk_embed_dim, gin_channels, sr, is_half=False, **kwargs):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.sr = sr

        # Basic components for minimal functionality
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)
        
        # Source generator for F0 conditioning
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sampling_rate=sr, harmonic_num=8)
        
        # Main generator
        self.dec = Generator(
            inter_channels, resblock, resblock_kernel_sizes, 
            resblock_dilation_sizes, upsample_rates, 
            upsample_initial_channel, upsample_kernel_sizes,
            gin_channels=gin_channels
        )

    def forward(self, phone, phone_lengths, pitch, pitchf, y, y_lengths, ds):
        # Simplified forward pass for inference
        g = self.emb_g(ds).unsqueeze(-1)  # [b, h, 1]
        
        # F0 conditioning
        with torch.no_grad():
            har_source, noi_source, uv = self.m_source(pitch, self.upsample_rates[0])
            har_source = har_source.transpose(1, 2)
        
        # Generate audio
        o = self.dec(har_source, g=g)
        return o
    
    def infer(self, c, f0_lengths, pitch=None, pitchf=None, sid=None):
        """Inference method for RVC voice conversion"""
        with torch.no_grad():
            # Get speaker embedding
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
            
            if pitch is not None and pitchf is not None:
                # F0 conditioning - use the first upsample rate for F0 upsampling
                upsample_rate = np.prod(self.upsample_rates)
                har_source, _, _ = self.m_source(pitch, upsample_rate)
                har_source = har_source.transpose(1, 2)  # [b, t, h]
                
                # Combine with input features
                c = c + har_source[:, :c.shape[1], :]
                
            # Generate audio using decoder
            o = self.dec(c.transpose(1, 2), g=g)  # [b, 1, t]
            return o, None


class SynthesizerTrnMs768NSFsid(SynthesizerTrnMs256NSFsid):
    """V2 model with 768 hidden channels"""
    pass


# Non-F0 variants
class SynthesizerTrnMs256NSFsid_nono(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Simplified non-F0 model
        self.dec = Generator(
            256, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], 
            [10, 10, 2, 2], 512, [16, 16, 4, 4], 256
        )
    
    def forward(self, *args, **kwargs):
        # Very basic implementation
        return torch.randn(1, 1, 16000)  # Return dummy audio
    
    def infer(self, c, f0_lengths, sid=None):
        """Non-F0 inference method"""
        with torch.no_grad():
            # Basic audio generation without F0
            o = self.dec(c.transpose(1, 2))
            return o, None


class SynthesizerTrnMs768NSFsid_nono(SynthesizerTrnMs256NSFsid_nono):
    """V2 non-F0 model"""
    pass