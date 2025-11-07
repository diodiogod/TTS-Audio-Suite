"""
PyTorch/TorchAudio Compatibility Patches

Handles compatibility issues with PyTorch and TorchAudio versions.
Critical: Global monkey-patching of torchaudio.save/load for PyTorch 2.9 TorchCodec DLL errors.
"""

import warnings
from typing import Optional


class PyTorchPatches:
    """Centralized PyTorch/TorchAudio compatibility patches manager"""

    _patches_applied = set()

    @classmethod
    def apply_all_patches(cls, verbose: bool = True):
        """Apply all necessary PyTorch compatibility patches"""
        cls.patch_torchaudio_torchcodec(verbose=verbose)

        if verbose and cls._patches_applied:
            print(f"✅ Applied {len(cls._patches_applied)} PyTorch compatibility patches")

    @classmethod
    def patch_torchaudio_torchcodec(cls, verbose: bool = True):
        """
        Patch torchaudio.save() and torchaudio.load() to use scipy instead of TorchCodec.

        Issues Fixed:
        1. PyTorch 2.9.0+cu128 TorchCodec incompatibility on Windows
           - Symptom: "Could not load libtorchcodec" errors during WAV file operations
           - Root cause: TorchCodec DLL doesn't support Windows
           - Fix: Globally monkey-patch to use scipy.io.wavfile (pure Python, no native deps)

        2. PyTorch 2.9 changed torchaudio.load() behavior
           - Returns raw int16 values (±32767) instead of normalized float [-1, 1]
           - Handled by safe_load_audio() utility function which normalizes audio after loading
           - This global patch ensures scipy fallback also normalizes consistently

        Implementation: Globally monkey-patches save/load for WAV files to use scipy,
        which is pure Python with no dependencies (no FFmpeg, no TorchCodec DLL).
        This fixes ALL direct torchaudio.save/load calls across the entire codebase without
        needing individual fixes in every file.
        """
        if "torchaudio_torchcodec" in cls._patches_applied:
            return

        try:
            import torch
            import torchaudio
            from scipy.io import wavfile as scipy_wavfile
            import numpy as np
            from io import BytesIO

            # Only apply patch on PyTorch 2.9+
            torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            if torch_version < (2, 9):
                return

            if verbose:
                print("🔧 Applying PyTorch 2.9 TorchCodec compatibility patches...")

            # Store original functions
            _original_torchaudio_save = torchaudio.save
            _original_torchaudio_load = torchaudio.load

            def _patched_torchaudio_save(uri, src, sample_rate, **kwargs):
                """
                Wrapper for torchaudio.save that uses scipy for WAV files to avoid TorchCodec DLL issues.
                Falls back to original torchaudio.save for non-WAV formats.
                """
                try:
                    # Only patch WAV files - use original for other formats
                    if isinstance(uri, str) and uri.lower().endswith('.wav'):
                        # Convert tensor to numpy
                        if hasattr(src, 'cpu'):  # torch.Tensor
                            src_np = src.cpu().detach().numpy()
                        else:
                            src_np = src

                        # Convert to float32 for scipy
                        src_np = src_np.astype(np.float32)

                        # scipy expects (samples, channels), transpose if needed
                        if src_np.ndim == 2 and src_np.shape[0] <= 2:
                            # Likely (channels, samples) - transpose to (samples, channels)
                            src_np = src_np.T

                        scipy_wavfile.write(uri, sample_rate, src_np)
                        return
                except Exception as e:
                    # Fallback to original torchaudio.save if scipy fails
                    pass

                # Use original for non-WAV or if patched version fails
                return _original_torchaudio_save(uri, src, sample_rate, **kwargs)

            def _patched_torchaudio_load(uri, *args, **kwargs):
                """
                Wrapper for torchaudio.load that uses scipy for WAV files to avoid TorchCodec DLL issues.
                Falls back to original torchaudio.load for non-WAV formats.
                Also normalizes int16 audio to float [-1, 1] range (PyTorch 2.9 change).
                """
                try:
                    # Handle both file paths (str) and file-like objects (BytesIO)
                    is_wav = False
                    if isinstance(uri, str):
                        is_wav = uri.lower().endswith('.wav')
                    elif isinstance(uri, BytesIO):
                        # For BytesIO, we'll try scipy and fall back to original
                        is_wav = True

                    if is_wav:
                        # Load with scipy
                        sample_rate, waveform_np = scipy_wavfile.read(uri)
                        waveform = torch.from_numpy(waveform_np.astype(np.float32))

                        # Normalize int16 range to float [-1, 1]
                        # PyTorch 2.9 changed behavior to return raw int16 values
                        max_val = torch.max(torch.abs(waveform))
                        if max_val > 1.0:
                            waveform = waveform / 32767.0

                        # Reshape to (channels, samples) format
                        if waveform.ndim == 1:
                            waveform = waveform.unsqueeze(0)
                        elif waveform.ndim == 2:
                            # Assume (samples, channels), transpose to (channels, samples)
                            waveform = waveform.T

                        return waveform, sample_rate
                except Exception as e:
                    # Fallback to original torchaudio.load if scipy fails
                    pass

                # Use original for non-WAV or if patched version fails
                return _original_torchaudio_load(uri, *args, **kwargs)

            # Apply the monkey-patches globally
            torchaudio.save = _patched_torchaudio_save
            torchaudio.load = _patched_torchaudio_load

            cls._patches_applied.add("torchaudio_torchcodec")

            if verbose:
                print("   🔧 torchaudio.save/load globally patched (scipy for WAV files, no TorchCodec)")

        except ImportError as e:
            warnings.warn(f"scipy not available for torchaudio patch: {e}")
        except Exception as e:
            warnings.warn(f"torchaudio TorchCodec patching failed: {e}")

    @classmethod
    def get_applied_patches(cls):
        """Get list of applied patches"""
        return list(cls._patches_applied)

    @classmethod
    def is_patch_applied(cls, patch_name: str) -> bool:
        """Check if a specific patch has been applied"""
        return patch_name in cls._patches_applied


# Convenience function for easy import
def apply_pytorch_patches(verbose: bool = True):
    """Apply all PyTorch compatibility patches"""
    PyTorchPatches.apply_all_patches(verbose=verbose)
