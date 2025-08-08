"""
MDX23C: TFC-TDF v3 Architecture for Music Source Separation
Implementation based on ZFTurbo's Music-Source-Separation-Training repository

MIT License - Copyright (c) 2024 Roman Solovyev (ZFTurbo)
Adapted for ComfyUI TTS Audio Suite
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from .utils import gc_collect


class STFT:
    def __init__(self, n_fft=2048, hop_length=512, dim_f=1024):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.dim_f = dim_f

    def __call__(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        
        # Ensure input length is compatible with STFT for consistent output dimensions
        # Pad input to be a multiple of hop_length for deterministic STFT behavior
        pad_amount = (self.hop_length - (t % self.hop_length)) % self.hop_length
        if pad_amount > 0:
            x = torch.nn.functional.pad(x, (0, pad_amount))
        
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([*batch_dims, c, 2, -1, x.shape[-1]]).reshape([*batch_dims, c * 2, -1, x.shape[-1]])
        return x[..., :self.dim_f, :]

    def inverse(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]
        n = self.n_fft // 2 + 1
        f_pad = torch.zeros([*batch_dims, c, n - f, t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
        x = x.permute([0, 2, 3, 1])
        x = x[..., 0] + x[..., 1] * 1.j
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True)
        x = x.reshape([*batch_dims, 2, -1])
        return x


def get_norm(norm_type):
    def norm(c, norm_type=norm_type):
        if norm_type == 'BatchNorm':
            return nn.BatchNorm2d(c)
        elif norm_type == 'InstanceNorm':
            return nn.InstanceNorm2d(c)
        elif norm_type == 'GroupNorm':
            return nn.GroupNorm(1, c)
    return partial(norm)


def get_act(act_type):
    if act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'prelu':
        return nn.PReLU()
    elif act_type == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.2)


class Upscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, scale, scale, bias=False)
        self.norm = norm(out_c)
        self.act = act

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x


class Downscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, scale, scale, bias=False)
        self.norm = norm(out_c)
        self.act = act

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x


class TFC_TDF(nn.Module):
    def __init__(self, in_c, c, l, f, bn, norm, act):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(l):
            block = nn.Module()

            block.tfc1 = nn.Sequential(
                norm(in_c),
                act,
                nn.Conv2d(in_c, c, 3, 1, 1, bias=False),
            )
            block.tdf = nn.Sequential(
                norm(c),
                act,
                nn.Linear(f, f // bn, bias=False),
                norm(c),
                act,
                nn.Linear(f // bn, f, bias=False),
            )
            block.tfc2 = nn.Sequential(
                norm(c),
                act,
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
            )
            block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)

            self.blocks.append(block)
            in_c = c

    def forward(self, x):
        for block in self.blocks:
            s = block.shortcut(x)
            x = block.tfc1(x)
            x = x + block.tdf(x)
            x = block.tfc2(x)
            x = x + s
        return x


class TFC_TDF_net(nn.Module):
    def __init__(self, 
                 # Audio config
                 n_fft=2048,
                 hop_length=512, 
                 dim_f=1024,
                 num_channels=2,
                 # Model config
                 num_subbands=4,
                 num_scales=5,
                 scale=[2, 2],
                 num_blocks_per_scale=2,
                 num_channels_start=32,
                 growth=32,
                 bottleneck_factor=4,
                 norm_type='InstanceNorm',
                 act_type='gelu',
                 # Targets
                 target_instrument='vocals'
                 ):
        super().__init__()
        
        norm = get_norm(norm_type=norm_type)
        act = get_act(act_type=act_type)

        self.num_target_instruments = 2  # vocals + other (based on checkpoint dimensions)  
        self.num_subbands = num_subbands

        dim_c = self.num_subbands * num_channels * 2
        n = num_scales
        l = num_blocks_per_scale
        c = num_channels_start
        g = growth
        bn = bottleneck_factor
        f = dim_f // self.num_subbands

        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)

        self.encoder_blocks = nn.ModuleList()
        for i in range(n):
            block = nn.Module()
            block.tfc_tdf = TFC_TDF(c, c, l, f, bn, norm, act)
            block.downscale = Downscale(c, c + g, scale, norm, act)
            f = f // scale[1]
            c += g
            self.encoder_blocks.append(block)

        self.bottleneck_block = TFC_TDF(c, c, l, f, bn, norm, act)

        self.decoder_blocks = nn.ModuleList()
        for i in range(n):
            block = nn.Module()
            block.upscale = Upscale(c, c - g, scale, norm, act)
            f = f * scale[1]
            c -= g
            block.tfc_tdf = TFC_TDF(2 * c, c, l, f, bn, norm, act)
            self.decoder_blocks.append(block)

        self.final_conv = nn.Sequential(
            nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False),
            act,
            nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False)
        )

        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, dim_f=dim_f)

    def cac2cws(self, x):
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c, k, f // k, t)
        x = x.reshape(b, c * k, f // k, t)
        return x

    def cws2cac(self, x):
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c // k, k, f, t)
        x = x.reshape(b, c // k, f * k, t)
        return x

    def forward(self, x):

        x = self.stft(x)

        mix = x = self.cac2cws(x)

        first_conv_out = x = self.first_conv(x)

        x = x.transpose(-1, -2)

        encoder_outputs = []
        for block in self.encoder_blocks:
            x = block.tfc_tdf(x)
            encoder_outputs.append(x)
            x = block.downscale(x)

        x = self.bottleneck_block(x)

        for block in self.decoder_blocks:
            x = block.upscale(x)
            x = torch.cat([x, encoder_outputs.pop()], 1)
            x = block.tfc_tdf(x)

        x = x.transpose(-1, -2)

        x = x * first_conv_out  # reduce artifacts

        x = self.final_conv(torch.cat([mix, x], 1))

        x = self.cws2cac(x)

        if self.num_target_instruments > 1:
            b, c, f, t = x.shape
            x = x.reshape(b, self.num_target_instruments, -1, f, t)

        x = self.stft.inverse(x)

        return x


class MDX23CSeparator:
    """
    MDX23C separator class compatible with UVR5 interface
    """
    def __init__(self, model_path: str, device="cpu", **kwargs):
        self.device = device
        
        # Load model and weights
        self.model = self._load_model(model_path)
        self.model.eval()
        
        if 'cuda' in str(device).lower() and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        print(f"✅ MDX23C model loaded: {model_path}")

    def _load_model(self, model_path: str) -> TFC_TDF_net:
        """Load MDX23C model from checkpoint"""
        
        # MDX23C configuration matching ZFTurbo's official config_vocals_mdx23c.yaml  
        model_config = {
            'n_fft': 8192,              # Official: 8192
            'hop_length': 1024,         # Official: 1024  
            'dim_f': 4096,              # Official: 4096
            'num_channels': 2,
            'num_subbands': 4,          # Official: 4
            'num_scales': 5,            # Official: 5
            'scale': [2, 2],            # Official: [2, 2]
            'num_blocks_per_scale': 2,  # Official: 2
            'num_channels_start': 128,  # Official: 128
            'growth': 128,              # Official: 128
            'bottleneck_factor': 4,     # Official: 4
            'norm_type': 'InstanceNorm',# Official: InstanceNorm
            'act_type': 'gelu',         # Official: gelu  
            'target_instrument': 'vocals'
        }
        
        # Create model
        model = TFC_TDF_net(**model_config)
        
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
        """Run inference on audio file"""
        from rvc_audio import load_input_audio
        
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
        
        # Use overlap-add processing for large audio files to avoid OOM and tensor mismatches
        hop_length = 1024
        chunk_duration = 10  # seconds (smaller chunks for MDX23C)
        overlap_duration = 1  # 1 second overlap
        
        chunk_samples = chunk_duration * sample_rate
        overlap_samples = overlap_duration * sample_rate
        
        # Align to hop length
        chunk_size = (chunk_samples // hop_length) * hop_length
        overlap_size = (overlap_samples // hop_length) * hop_length
        step_size = chunk_size - overlap_size
        
        total_samples = audio_tensor.shape[2]
        
        # Temporarily disable chunking to avoid tensor dimension issues
        # MDX23C architecture has complex encoder-decoder skip connections that don't align well with chunking
        if False and total_samples > chunk_size:
            print(f"🔄 Processing {total_samples//sample_rate}s audio with overlap-add (10s chunks, 1s overlap)...")
            
            # Initialize output buffers
            vocals_output = torch.zeros(audio_tensor.shape[0], audio_tensor.shape[1], total_samples)
            instrumentals_output = torch.zeros(audio_tensor.shape[0], audio_tensor.shape[1], total_samples)
            weight_sum = torch.zeros(total_samples)
            
            for start in range(0, total_samples - overlap_size, step_size):
                end = min(start + chunk_size, total_samples)
                actual_chunk_size = end - start
                
                chunk = audio_tensor[:, :, start:end]
                
                # Ensure all chunks have exactly the same length by padding if necessary
                if actual_chunk_size < chunk_size:
                    padding_needed = chunk_size - actual_chunk_size
                    padding = torch.zeros(chunk.shape[0], chunk.shape[1], padding_needed, 
                                        device=chunk.device, dtype=chunk.dtype)
                    chunk = torch.cat([chunk, padding], dim=2)
                    print(f"🔧 Processing chunk {start//sample_rate}s-{end//sample_rate}s (padded to {chunk_size} samples)")
                else:
                    print(f"🔧 Processing chunk {start//sample_rate}s-{end//sample_rate}s")
                
                # Move to device
                if 'cuda' in str(self.device).lower() and torch.cuda.is_available():
                    chunk = chunk.cuda()
                    
                # Run separation on chunk (now guaranteed to have consistent dimensions)
                with torch.no_grad():
                    print(f"   Chunk input shape: {chunk.shape}")
                    try:
                        separated = self.model(chunk)
                        separated = separated.cpu()
                        print(f"   Separated output shape: {separated.shape}")
                    except Exception as e:
                        print(f"   Model forward failed: {e}")
                        print(f"   Chunk shape: {chunk.shape}")
                        raise
                    
                    # If we padded the chunk, remove padding from output
                    if actual_chunk_size < chunk_size:
                        separated = separated[:, :, :, :actual_chunk_size]
                    
                    # Create fade windows for overlap regions
                    fade_in = torch.linspace(0, 1, overlap_size) if start > 0 else torch.ones(actual_chunk_size)
                    fade_out = torch.linspace(1, 0, overlap_size) if end < total_samples else torch.ones(actual_chunk_size)
                    
                    # Apply fading only to overlap regions
                    if start > 0 and end < total_samples:
                        # Both fade in and fade out
                        window = torch.cat([fade_in, torch.ones(actual_chunk_size - 2*overlap_size), fade_out])
                    elif start > 0:
                        # Only fade in
                        window = torch.cat([fade_in, torch.ones(actual_chunk_size - overlap_size)])
                    elif end < total_samples:
                        # Only fade out  
                        window = torch.cat([torch.ones(actual_chunk_size - overlap_size), fade_out])
                    else:
                        # No fading
                        window = torch.ones(actual_chunk_size)
                    
                    # Apply window to separated audio
                    windowed_vocals = separated[0, 0] * window
                    windowed_instrumentals = separated[0, 1] * window
                    
                    # Add to output buffers
                    vocals_output[0, :, start:end] += windowed_vocals.unsqueeze(0)
                    instrumentals_output[0, :, start:end] += windowed_instrumentals.unsqueeze(0)
                    weight_sum[start:end] += window
                    
                    # Clear CUDA cache 
                    if 'cuda' in str(self.device).lower() and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Normalize by weight sum to handle overlaps
            weight_sum = torch.clamp(weight_sum, min=1e-8)  # Avoid division by zero
            vocals = (vocals_output[0] / weight_sum.unsqueeze(0)).numpy()
            instrumentals = (instrumentals_output[0] / weight_sum.unsqueeze(0)).numpy()
            
        else:
            # Process entire audio - optimized for memory usage
            print(f"🔄 Processing full {total_samples//sample_rate}s audio (no chunking)")
            
            # Move to device
            if 'cuda' in str(self.device).lower() and torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()
                
                # Enable memory efficient attention if available
                if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                    torch.backends.cuda.enable_flash_sdp(False)  # Disable flash attention to save memory
                    
                # Clear cache before processing
                torch.cuda.empty_cache()
                
            # Run separation with memory optimizations
            with torch.no_grad():
                # Use autocast for mixed precision to save memory
                if 'cuda' in str(self.device).lower() and torch.cuda.is_available():
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        separated = self.model(audio_tensor.half()).float()
                else:
                    separated = self.model(audio_tensor)
                    
                # Clear cache after processing
                if 'cuda' in str(self.device).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            # Extract results
            vocals = separated[0, 0].cpu().numpy()     # vocals (target 0)
            instrumentals = separated[0, 1].cpu().numpy()  # other/instrumentals (target 1)
        
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