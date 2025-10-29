"""
Voice Fixer Audio Restoration Node for ComfyUI
Restores degraded audio by removing noise, reverberation, clipping, and low-resolution artifacts
"""

import torch
import numpy as np
from typing import Tuple, Optional
import os

# Lazy import VoiceFixer - don't import until first use to avoid cache downloads
VOICEFIXER_AVAILABLE = True
VoiceFixer = None

# Model management
from utils.downloads.voicefixer_downloader import VoiceFixerDownloader
from utils.models.extra_paths import get_preferred_download_path

# Add bundled voicefixer to path
import sys
current_dir = os.path.dirname(__file__)
utils_dir = os.path.dirname(os.path.dirname(current_dir))
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)


class VoiceFixerNode:
    """
    Audio restoration node using VoiceFixer to clean degraded audio.

    Handles:
    - Noise removal
    - Reverberation removal
    - Clipping restoration (0.1-1.0 threshold)
    - Low-resolution audio upscaling (2kHz~44.1kHz)

    All in a single unified model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "restoration_mode": (["0 - Original (Default)", "1 - With High-Freq Removal", "2 - Train Mode (Seriously Degraded)"],),
                "use_cuda": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("restored_audio", "restoration_info")
    FUNCTION = "restore_audio"
    CATEGORY = "audio/restoration"

    def __init__(self):
        self.voicefixer = None
        self.downloader = VoiceFixerDownloader()

    @classmethod
    def NAME(cls):
        return "🎙️ Voice Fixer"

    def _ensure_analysis_module_path(self, voicefixer_dir):
        """Ensure analysis module path is configured"""
        # The analysis module is loaded directly by VoiceFixer base class
        # It looks in ~/.cache by default, but we've already ensured it's downloaded
        # to voicefixer_dir via the downloader
        pass

    def restore_audio(self, audio: torch.Tensor, restoration_mode: str, use_cuda: bool) -> Tuple[torch.Tensor, str]:
        """
        Restore degraded audio using VoiceFixer.

        Args:
            audio: Input audio tensor [batch, channels, samples] or [batch, samples]
            restoration_mode: Which restoration mode to use (0, 1, or 2)
            use_cuda: Whether to use CUDA acceleration

        Returns:
            Tuple of (restored_audio_tensor, info_string)
        """
        global VoiceFixer

        # Parse mode from dropdown string
        mode = int(restoration_mode.split(" - ")[0])

        # Ensure models are downloaded (lazy - only on first use)
        if not self.downloader.ensure_models_downloaded():
            raise RuntimeError("Failed to download VoiceFixer models")

        # Initialize VoiceFixer if not already done
        if self.voicefixer is None:
            # Import and patch ONLY after models are downloaded
            if VoiceFixer is None:
                import sys
                from io import StringIO

                voicefixer_dir = self.downloader.voicefixer_dir
                analysis_ckpt = os.path.join(voicefixer_dir, 'vf.ckpt')
                vocoder_ckpt = os.path.join(voicefixer_dir, 'model.ckpt-1490000_trimed.pt')

                # Verify the files exist before patching
                if not os.path.exists(analysis_ckpt):
                    raise RuntimeError(f"Analysis module checkpoint not found at {analysis_ckpt}")
                if not os.path.exists(vocoder_ckpt):
                    raise RuntimeError(f"Vocoder checkpoint not found at {vocoder_ckpt}")

                print(f"✅ Using VoiceFixer models from: {voicefixer_dir}")

                # Suppress stdout during import
                old_stdout = sys.stdout
                sys.stdout = StringIO()

                try:
                    # Patch vocoder config BEFORE any imports that use it
                    import voicefixer_bundled.vocoder.config as vocoder_config
                    vocoder_config.Config.ckpt = vocoder_ckpt

                    # Import from bundled version
                    from voicefixer_bundled.base import VoiceFixer as VoiceFixerClass
                    VoiceFixer = VoiceFixerClass
                finally:
                    sys.stdout = old_stdout

                # Now patch the instance to override analysis checkpoint path
                # Store the correct paths for use during initialization
                self._voicefixer_analysis_ckpt = analysis_ckpt
                self._voicefixer_vocoder_ckpt = vocoder_ckpt

                # Monkey-patch torch.load to intercept analysis checkpoint loading
                original_torch_load = torch.load

                def patched_load(path, *args, **kwargs):
                    # If loading the default cache path, redirect to our checkpoint
                    if "analysis_module/checkpoints/vf.ckpt" in str(path):
                        print(f"   Redirecting analysis checkpoint from {path} to {analysis_ckpt}")
                        return original_torch_load(analysis_ckpt, *args, **kwargs)
                    return original_torch_load(path, *args, **kwargs)

                torch.load = patched_load
                print("✅ VoiceFixer imported and patched successfully")

            print("Initializing VoiceFixer...")
            self.voicefixer = VoiceFixer()
            print("✅ VoiceFixer initialized successfully")

        # Handle device
        use_cuda = use_cuda and torch.cuda.is_available()
        if use_cuda and not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available - falling back to CPU")
            use_cuda = False

        # Convert ComfyUI tensor format to numpy
        # ComfyUI uses [batch, samples] or [batch, channels, samples]
        if audio.dim() == 2:
            # [batch, samples] - mono batch
            batch_size, num_samples = audio.shape
            if batch_size != 1:
                raise ValueError(f"Expected batch size 1, got {batch_size}")
            wav_numpy = audio[0].cpu().numpy().astype(np.float32)
        elif audio.dim() == 3:
            # [batch, channels, samples] - multichannel
            batch_size, channels, num_samples = audio.shape
            if batch_size != 1:
                raise ValueError(f"Expected batch size 1, got {batch_size}")
            if channels > 1:
                # Mix down to mono
                wav_numpy = audio[0].mean(dim=0).cpu().numpy().astype(np.float32)
            else:
                wav_numpy = audio[0, 0].cpu().numpy().astype(np.float32)
        else:
            raise ValueError(f"Unexpected audio tensor shape: {audio.shape}")

        # Resample to 44.1kHz if needed (VoiceFixer works at 44.1kHz internally)
        # Note: Users should handle resampling separately if they need specific sample rates
        # VoiceFixer internally resamples, but for best results use 44.1kHz input

        # Restore audio
        restored_wav = self.voicefixer.restore_inmem(
            wav_10k=wav_numpy,
            cuda=use_cuda,
            mode=mode,
            your_vocoder_func=None
        )

        # Convert back to ComfyUI tensor format
        restored_tensor = torch.from_numpy(restored_wav).unsqueeze(0).float()  # [1, samples]

        # Generate info string
        mode_names = ["Original", "High-Freq Removal", "Train Mode"]
        info = f"🎙️ VoiceFixer Mode {mode} ({mode_names[mode]}) | Input: {wav_numpy.shape[0]:,} samples | Output: {restored_wav.shape[0]:,} samples"

        return (restored_tensor, info)
