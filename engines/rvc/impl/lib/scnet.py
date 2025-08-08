"""
SCNet: Sparse Compression Network for Music Source Separation
Implementation based on ZFTurbo's Music-Source-Separation-Training repository
Paper: https://arxiv.org/abs/2401.13276.pdf

MIT License - Copyright (c) 2024 Roman Solovyev (ZFTurbo)
Adapted for ComfyUI TTS Audio Suite
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import typing as tp
import math
import numpy as np
from .utils import gc_collect


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class ConvolutionModule(nn.Module):
    """
    Convolution Module in SD block.

    Args:
        channels (int): input/output channels.
        depth (int): number of layers in the residual branch.
        compress (float): amount of channel compression.
        kernel (int): kernel size for the convolutions.
    """

    def __init__(self, channels, depth=2, compress=4, kernel=3):
        super().__init__()
        assert kernel % 2 == 1
        self.depth = abs(depth)
        hidden_size = int(channels / compress)
        norm = lambda d: nn.GroupNorm(1, d)
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            padding = (kernel // 2)
            mods = [
                norm(channels),
                nn.Conv1d(channels, hidden_size * 2, kernel, padding=padding),
                nn.GLU(1),
                nn.Conv1d(hidden_size, hidden_size, kernel, padding=padding, groups=hidden_size),
                norm(hidden_size),
                Swish(),
                nn.Conv1d(hidden_size, channels, 1),
            ]
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class FeatureConversion(nn.Module):
    """
    Integrates into the adjacent Dual-Path layer.

    Args:
        channels (int): Number of input channels.
        inverse (bool): If True, uses ifft; otherwise, uses rfft.
    """

    def __init__(self, channels, inverse):
        super().__init__()
        self.inverse = inverse
        self.channels = channels

    def forward(self, x):
        # B, C, F, T = x.shape
        if self.inverse:
            x = x.float()
            x_r = x[:, :self.channels // 2, :, :]
            x_i = x[:, self.channels // 2:, :, :]
            x = torch.complex(x_r, x_i)
            x = torch.fft.irfft(x, dim=3, norm="ortho")
        else:
            x = x.float()
            x = torch.fft.rfft(x, dim=3, norm="ortho")
            x_real = x.real
            x_imag = x.imag
            x = torch.cat([x_real, x_imag], dim=1)
        return x


class DualPathRNN(nn.Module):
    """
    Dual-Path RNN in Separation Network.

    Args:
        d_model (int): The number of expected features in the input (input_size).
        expand (int): Expansion factor used to calculate the hidden_size of LSTM.
        bidirectional (bool): If True, becomes a bidirectional LSTM.
    """

    def __init__(self, d_model, expand, bidirectional=True):
        super(DualPathRNN, self).__init__()

        self.d_model = d_model
        self.hidden_size = d_model * expand
        self.bidirectional = bidirectional
        # Initialize LSTM layers and normalization layers
        self.lstm_layers = nn.ModuleList([self._init_lstm_layer(self.d_model, self.hidden_size) for _ in range(2)])
        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_size * 2, self.d_model) for _ in range(2)])
        self.norm_layers = nn.ModuleList([nn.GroupNorm(1, d_model) for _ in range(2)])

    def _init_lstm_layer(self, d_model, hidden_size):
        return nn.LSTM(d_model, hidden_size, num_layers=1, bidirectional=self.bidirectional, batch_first=True)

    def forward(self, x):
        B, C, F, T = x.shape

        # Process dual-path rnn
        original_x = x
        # Frequency-path
        x = self.norm_layers[0](x)
        x = x.transpose(1, 3).contiguous().view(B * T, F, C)
        x, _ = self.lstm_layers[0](x)
        x = self.linear_layers[0](x)
        x = x.view(B, T, F, C).transpose(1, 3)
        x = x + original_x

        original_x = x
        # Time-path
        x = self.norm_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(B * F, C, T).transpose(1, 2)
        x, _ = self.lstm_layers[1](x)
        x = self.linear_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(B, F, C, T).transpose(1, 2)
        x = x + original_x

        return x


class SeparationNet(nn.Module):
    """
    Separation Network with Dual-Path RNN layers.

    Args:
    - channels (int): Number input channels.
    - expand (int): Expansion factor used to calculate the hidden_size of LSTM.
    - num_layers (int): Number of dual-path layers.
    """

    def __init__(self, channels, expand=1, num_layers=6):
        super(SeparationNet, self).__init__()

        self.num_layers = num_layers

        self.dp_modules = nn.ModuleList([
            DualPathRNN(channels * (2 if i % 2 == 1 else 1), expand) for i in range(num_layers)
        ])

        self.feature_conversion = nn.ModuleList([
            FeatureConversion(channels * 2, inverse=False if i % 2 == 0 else True) for i in range(num_layers)
        ])

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.dp_modules[i](x)
            x = self.feature_conversion[i](x)
        return x


class FusionLayer(nn.Module):
    """
    A FusionLayer within the decoder.

    Args:
    - channels (int): Number of input channels.
    - kernel_size (int, optional): Kernel size for the convolutional layer, defaults to 3.
    - stride (int, optional): Stride for the convolutional layer, defaults to 1.
    - padding (int, optional): Padding for the convolutional layer, defaults to 1.
    """

    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(FusionLayer, self).__init__()
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size, stride=stride, padding=padding)

    def forward(self, x, skip=None):
        if skip is not None:
            x += skip
        x = x.repeat(1, 2, 1, 1)
        x = self.conv(x)
        x = F.glu(x, dim=1)
        return x


class SDlayer(nn.Module):
    """
    Implements a Sparse Down-sample Layer for processing different frequency bands separately.

    Args:
    - channels_in (int): Input channel count.
    - channels_out (int): Output channel count.
    - band_configs (dict): A dictionary containing configuration for each frequency band.
    """

    def __init__(self, channels_in, channels_out, band_configs):
        super(SDlayer, self).__init__()

        # Initializing convolutional layers for each band
        self.convs = nn.ModuleList()
        self.strides = []
        self.kernels = []
        for config in band_configs.values():
            self.convs.append(
                nn.Conv2d(channels_in, channels_out, (config['kernel'], 1), (config['stride'], 1), (0, 0)))
            self.strides.append(config['stride'])
            self.kernels.append(config['kernel'])

        # Saving rate proportions for determining splits
        self.SR_low = band_configs['low']['SR']
        self.SR_mid = band_configs['mid']['SR']

    def forward(self, x):
        B, C, Fr, T = x.shape
        # Define splitting points based on sampling rates
        splits = [
            (0, math.ceil(Fr * self.SR_low)),
            (math.ceil(Fr * self.SR_low), math.ceil(Fr * (self.SR_low + self.SR_mid))),
            (math.ceil(Fr * (self.SR_low + self.SR_mid)), Fr)
        ]

        # Processing each band with the corresponding convolution
        outputs = []
        original_lengths = []
        for conv, stride, kernel, (start, end) in zip(self.convs, self.strides, self.kernels, splits):
            extracted = x[:, :, start:end, :]
            original_lengths.append(end - start)
            current_length = extracted.shape[2]

            # padding
            if stride == 1:
                total_padding = kernel - stride
            else:
                total_padding = (stride - current_length % stride) % stride
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left

            padded = F.pad(extracted, (0, 0, pad_left, pad_right))

            output = conv(padded)
            outputs.append(output)

        return outputs, original_lengths


class SUlayer(nn.Module):
    """
    Implements a Sparse Up-sample Layer in decoder.

    Args:
    - channels_in: The number of input channels.
    - channels_out: The number of output channels.
    - band_configs: Dictionary containing the configurations for transposed convolutions.
    """

    def __init__(self, channels_in, channels_out, band_configs):
        super(SUlayer, self).__init__()

        # Initializing convolutional layers for each band
        self.convtrs = nn.ModuleList([
            nn.ConvTranspose2d(channels_in, channels_out, [config['kernel'], 1], [config['stride'], 1])
            for _, config in band_configs.items()
        ])

    def forward(self, x, lengths, origin_lengths):
        B, C, Fr, T = x.shape
        # Define splitting points based on input lengths
        splits = [
            (0, lengths[0]),
            (lengths[0], lengths[0] + lengths[1]),
            (lengths[0] + lengths[1], None)
        ]
        # Processing each band with the corresponding convolution
        outputs = []
        for idx, (convtr, (start, end)) in enumerate(zip(self.convtrs, splits)):
            out = convtr(x[:, :, start:end, :])
            # Calculate the distance to trim the output symmetrically to original length
            current_Fr_length = out.shape[2]
            dist = abs(origin_lengths[idx] - current_Fr_length) // 2

            # Trim the output to the original length symmetrically
            trimmed_out = out[:, :, dist:dist + origin_lengths[idx], :]

            outputs.append(trimmed_out)

        # Concatenate trimmed outputs along the frequency dimension to return the final tensor
        x = torch.cat(outputs, dim=2)

        return x


class SDblock(nn.Module):
    """
    Implements a simplified Sparse Down-sample block in encoder.

    Args:
    - channels_in (int): Number of input channels.
    - channels_out (int): Number of output channels.
    - band_config (dict): Configuration for the SDlayer specifying band splits and convolutions.
    - conv_config (dict): Configuration for convolution modules applied to each band.
    - depths (list of int): List specifying the convolution depths for low, mid, and high frequency bands.
    """

    def __init__(self, channels_in, channels_out, band_configs={}, conv_config={}, depths=[3, 2, 1], kernel_size=3):
        super(SDblock, self).__init__()
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)

        # Dynamically create convolution modules for each band based on depths
        self.conv_modules = nn.ModuleList([
            ConvolutionModule(channels_out, depth, **conv_config) for depth in depths
        ])
        # Set the kernel_size to an odd number.
        self.globalconv = nn.Conv2d(channels_out, channels_out, kernel_size, 1, (kernel_size - 1) // 2)

    def forward(self, x):
        bands, original_lengths = self.SDlayer(x)
        # B, C, f, T = band.shape
        bands = [
            F.gelu(
                conv(band.permute(0, 2, 1, 3).reshape(-1, band.shape[1], band.shape[3]))
                .view(band.shape[0], band.shape[2], band.shape[1], band.shape[3])
                .permute(0, 2, 1, 3)
            )
            for conv, band in zip(self.conv_modules, bands)

        ]
        lengths = [band.size(-2) for band in bands]
        full_band = torch.cat(bands, dim=2)
        skip = full_band

        output = self.globalconv(full_band)

        return output, skip, lengths, original_lengths


class SCNet(nn.Module):
    """
    The implementation of SCNet: Sparse Compression Network for Music Source Separation. 
    Paper: https://arxiv.org/abs/2401.13276.pdf

    Args:
    - sources (List[str]): List of sources to be separated.
    - audio_channels (int): Number of audio channels.
    - nfft (int): Number of FFTs to determine the frequency dimension of the input.
    - hop_size (int): Hop size for the STFT.
    - win_size (int): Window size for STFT.
    - normalized (bool): Whether to normalize the STFT.
    - dims (List[int]): List of channel dimensions for each block.
    - band_SR (List[float]): The proportion of each frequency band.
    - band_stride (List[int]): The down-sampling ratio of each frequency band.
    - band_kernel (List[int]): The kernel sizes for down-sampling convolution in each frequency band
    - conv_depths (List[int]): List specifying the number of convolution modules in each SD block.
    - compress (int): Compression factor for convolution module.
    - conv_kernel (int): Kernel size for convolution layer in convolution module.
    - num_dplayer (int): Number of dual-path layers.
    - expand (int): Expansion factor in the dual-path RNN, default is 1.
    """

    def __init__(self,
                 sources=['drums', 'bass', 'other', 'vocals'],
                 audio_channels=2,
                 # Main structure
                 dims=[4, 32, 64, 128],  # dims = [4, 64, 128, 256] in SCNet-large
                 # STFT
                 nfft=4096,
                 hop_size=1024,
                 win_size=4096,
                 normalized=True,
                 # SD/SU layer
                 band_SR=[0.175, 0.392, 0.433],
                 band_stride=[1, 4, 16],
                 band_kernel=[3, 4, 16],
                 # Convolution Module
                 conv_depths=[3, 2, 1],
                 compress=4,
                 conv_kernel=3,
                 # Dual-path RNN
                 num_dplayer=6,
                 expand=1,
                 ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.dims = dims
        band_keys = ['low', 'mid', 'high']
        self.band_configs = {band_keys[i]: {'SR': band_SR[i], 'stride': band_stride[i], 'kernel': band_kernel[i]} for i
                             in range(len(band_keys))}
        self.hop_length = hop_size
        self.conv_config = {
            'compress': compress,
            'kernel': conv_kernel,
        }

        self.stft_config = {
            'n_fft': nfft,
            'hop_length': hop_size,
            'win_length': win_size,
            'center': True,
            'normalized': normalized
        }

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(len(dims) - 1):
            enc = SDblock(
                channels_in=dims[index],
                channels_out=dims[index + 1],
                band_configs=self.band_configs,
                conv_config=self.conv_config,
                depths=conv_depths
            )
            self.encoder.append(enc)

            dec = nn.Sequential(
                FusionLayer(channels=dims[index + 1]),
                SUlayer(
                    channels_in=dims[index + 1],
                    channels_out=dims[index] if index != 0 else dims[index] * len(sources),
                    band_configs=self.band_configs,
                )
            )
            self.decoder.insert(0, dec)

        self.separation_net = SeparationNet(
            channels=dims[-1],
            expand=expand,
            num_layers=num_dplayer,
        )

    def forward(self, x):
        # B, C, L = x.shape
        B = x.shape[0]
        # In the initial padding, ensure that the number of frames after the STFT (the length of the T dimension) is even,
        # so that the RFFT operation can be used in the separation network.
        padding = self.hop_length - x.shape[-1] % self.hop_length
        if (x.shape[-1] + padding) // self.hop_length % 2 == 0:
            padding += self.hop_length
        x = F.pad(x, (0, padding))

        # STFT - use rectangular window like reference
        L = x.shape[-1]
        x = x.reshape(-1, L)
        x = torch.stft(x, **self.stft_config, return_complex=True)
        x = torch.view_as_real(x)
        # Fix tensor reshaping - use reference implementation logic
        x = x.permute(0, 3, 1, 2).reshape(x.shape[0] // self.audio_channels, x.shape[3] * self.audio_channels,
                                          x.shape[1], x.shape[2])

        B, C, Fr, T = x.shape

        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()
        # encoder
        for sd_layer in self.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # separation
        x = self.separation_net(x)

        # decoder
        for fusion_layer, su_layer in self.decoder:
            x = fusion_layer(x, save_skip.pop())
            x = su_layer(x, save_lengths.pop(), save_original_lengths.pop())

        # output
        n = self.dims[0]
        x = x.view(B, n, -1, Fr, T)
        x = x.reshape(-1, 2, Fr, T).permute(0, 2, 3, 1)
        x = torch.view_as_complex(x.contiguous())
        # Use same STFT config for ISTFT  
        x = torch.istft(x, **self.stft_config)
        x = x.reshape(B, len(self.sources), self.audio_channels, -1)

        x = x[:, :, :, :-padding]

        return x


class SCNetSeparator:
    """
    SCNet separator class compatible with UVR5 interface
    """
    DEFAULT_SR = 44100
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR

    def __init__(self, model_path: str, device="cpu", margin=DEFAULT_MARGIN_SIZE, chunks=15, **kwargs):
        self.device = device
        self.margin = margin
        self.chunks = chunks
        
        # Load model config and weights
        self.model = self._load_model(model_path)
        self.model.eval()
        
        if 'cuda' in str(device).lower() and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        print(f"✅ SCNet model loaded: {model_path}")

    def _load_model(self, model_path: str) -> SCNet:
        """Load SCNet model from checkpoint"""
        # Detect model configuration based on filename
        model_name = os.path.basename(model_path).lower()
        
        if "xl" in model_name:
            # XL IHF configuration - corrected based on actual checkpoint structure
            model_config = {
                'sources': ['drums', 'bass', 'other', 'vocals'],
                'audio_channels': 2,
                'dims': [4, 64, 128, 256],  # XL configuration
                'nfft': 4096,
                'hop_size': 1024,
                'win_size': 4096,
                'normalized': True,  # Back to reference setting
                'band_SR': [0.175, 0.392, 0.433],
                'band_stride': [1, 4, 4],  # Corrected based on checkpoint error
                'band_kernel': [3, 4, 4],  # Corrected based on checkpoint error  
                'conv_depths': [3, 2, 1],
                'compress': 4,
                'conv_kernel': 3,
                'num_dplayer': 6,
                'expand': 1,
            }
        else:
            # Default SCNet configuration
            model_config = {
                'sources': ['drums', 'bass', 'other', 'vocals'],
                'audio_channels': 2,
                'dims': [4, 32, 64, 128],  # Standard configuration
                'nfft': 4096,
                'hop_size': 1024,
                'win_size': 4096,
                'normalized': True,
                'band_SR': [0.175, 0.392, 0.433],
                'band_stride': [1, 4, 16],
                'band_kernel': [3, 4, 16],
                'conv_depths': [3, 2, 1],
                'compress': 4,
                'conv_kernel': 3,
                'num_dplayer': 6,
                'expand': 1,
            }
        
        # Create model
        model = SCNet(**model_config)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Clean up state dict keys if needed
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # Remove 'model.' prefix if present
            clean_key = key.replace('model.', '') if key.startswith('model.') else key
            cleaned_state_dict[clean_key] = value
            
        model.load_state_dict(cleaned_state_dict, strict=False)
        
        return model

    def run_inference(self, audio_path: str, format: str = "mp3"):
        """Run inference on audio file with chunked processing"""
        from rvc_audio import load_input_audio, save_input_audio
        import tempfile
        import os
        
        # Load audio
        audio_data, sample_rate = load_input_audio(audio_path, mono=False)
        
        # Convert to tensor
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data).float()
        else:
            audio_tensor = audio_data.clone().detach().float()
            
        # Ensure correct shape [batch, channels, samples] 
        if audio_tensor.dim() == 2:
            # Shape is [channels, samples] - add batch dimension
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.dim() == 1:
            # Shape is [samples] - assume mono, add batch and channel dimensions
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            # Duplicate for stereo
            audio_tensor = audio_tensor.repeat(1, 2, 1)
        
        # Chunked processing for large audio files to avoid OOM
        chunk_size = sample_rate * 30  # 30 seconds chunks
        total_samples = audio_tensor.shape[2]
        
        if total_samples > chunk_size:
            outputs = []
            
            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                chunk = audio_tensor[:, :, start:end]
                
                # Move to device
                if 'cuda' in str(self.device).lower() and torch.cuda.is_available():
                    chunk = chunk.cuda()
                    
                # Run separation on chunk
                with torch.no_grad():
                    separated = self.model(chunk)
                    outputs.append(separated.cpu())
                    
                    # Clear CUDA cache 
                    if 'cuda' in str(self.device).lower() and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Concatenate all chunks along time dimension
            full_output = torch.cat(outputs, dim=3)
            
            # Extract results
            vocals = full_output[0, 3].numpy()  # vocals
            instrumentals = full_output[0, 2].numpy()  # other (instrumentals)
            
        else:
            # Process entire audio if small enough
            # Move to device
            if 'cuda' in str(self.device).lower() and torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()
                
            # Run separation
            with torch.no_grad():
                separated = self.model(audio_tensor)
                
            # Extract results
            vocals = separated[0, 3].cpu().numpy()  # vocals
            instrumentals = separated[0, 2].cpu().numpy()  # other (instrumentals)
        
        # Convert back to expected format
        input_audio = (audio_data, sample_rate)
        vocals_audio = (vocals, sample_rate)
        instrumentals_audio = (instrumentals, sample_rate)
        
        return vocals_audio, instrumentals_audio, input_audio

    def __del__(self):
        """Clean up model"""
        try:
            if hasattr(self, 'model'):
                del self.model
            gc_collect()
        except:
            pass