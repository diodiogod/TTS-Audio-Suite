"""
PyTorch/TorchAudio Compatibility Patches

Handles compatibility issues with PyTorch and TorchAudio versions.
"""

import warnings
from typing import Optional


class PyTorchPatches:
    """Centralized PyTorch/TorchAudio compatibility patches manager"""

    _patches_applied = set()

    @classmethod
    def apply_all_patches(cls, verbose: bool = True):
        """Apply all necessary PyTorch compatibility patches"""
        if verbose:
            print("🔧 Applying PyTorch compatibility patches...")

        cls.patch_torchaudio_torchcodec(verbose=verbose)

        if verbose:
            print(f"✅ Applied {len(cls._patches_applied)} PyTorch compatibility patches")

    @classmethod
    def patch_torchaudio_torchcodec(cls, verbose: bool = True):
        """
        Patch torchaudio.save() and torchaudio.load() to use soundfile instead of TorchCodec.

        Issue: PyTorch 2.9.0+cu128 has TorchCodec incompatibility on Windows
        causing "Could not load libtorchcodec" errors during WAV file operations.
        Affects: IndexTTS-2 engine inference that saves/loads WAV files
        Solution: Use soundfile library for WAV files instead of TorchCodec
        """
        if "torchaudio_torchcodec" in cls._patches_applied:
            return

        try:
            import torch
            import torchaudio

            # Only apply patch on PyTorch 2.9+
            torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            if torch_version < (2, 9):
                return

            import soundfile as sf

            # Store original functions
            _original_torchaudio_save = torchaudio.save
            _original_torchaudio_load = torchaudio.load

            def _patched_torchaudio_save(uri, src, sample_rate, **kwargs):
                """Wrapper for torchaudio.save that uses soundfile for WAV files to avoid TorchCodec issues."""
                try:
                    # Only patch WAV files - use original for other formats
                    if isinstance(uri, str) and uri.lower().endswith('.wav'):
                        # Convert tensor to numpy
                        if isinstance(src, torch.Tensor):
                            src_np = src.cpu().detach().numpy()
                        else:
                            src_np = src

                        # soundfile expects (samples, channels), transpose if needed
                        if src_np.ndim == 2 and src_np.shape[0] <= 2:
                            # Likely (channels, samples) - transpose to (samples, channels)
                            src_np = src_np.T

                        sf.write(uri, src_np, sample_rate)
                        return
                except Exception as e:
                    # Fallback to original torchaudio.save if soundfile fails
                    pass

                # Use original for non-WAV or if patched version fails
                return _original_torchaudio_save(uri, src, sample_rate, **kwargs)

            def _patched_torchaudio_load(uri, *args, **kwargs):
                """Wrapper for torchaudio.load that uses soundfile for WAV files to avoid TorchCodec issues."""
                try:
                    # Only patch WAV files - use original for other formats
                    if isinstance(uri, str) and uri.lower().endswith('.wav'):
                        # Load with soundfile
                        waveform_np, sample_rate = sf.read(uri, dtype='float32')

                        # Convert to tensor and reshape to (channels, samples)
                        waveform = torch.from_numpy(waveform_np).unsqueeze(0) if waveform_np.ndim == 1 else torch.from_numpy(waveform_np).T

                        return waveform, sample_rate
                except Exception as e:
                    # Fallback to original torchaudio.load if soundfile fails
                    pass

                # Use original for non-WAV or if patched version fails
                return _original_torchaudio_load(uri, *args, **kwargs)

            # Apply the monkey-patches
            torchaudio.save = _patched_torchaudio_save
            torchaudio.load = _patched_torchaudio_load

            cls._patches_applied.add("torchaudio_torchcodec")

            if verbose:
                print("   🔧 torchaudio.save/load patched (using soundfile for WAV files)")

        except ImportError as e:
            warnings.warn(f"soundfile not available for torchaudio patch: {e}")
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
