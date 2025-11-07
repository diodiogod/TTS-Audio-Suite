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
        cls.patch_s2mel_normalization(verbose=verbose)

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
    def patch_s2mel_normalization(cls, verbose: bool = True):
        """
        Patch S2Mel CFM solver for PyTorch 2.9 numerical instability.

        Issue: PyTorch 2.9 exhibits numerical instability in the CFM solver's
        Classifier-Free Guidance (CFG) scaling. The formula:
            dphi_dt = (1.0 + cfg_rate) * dphi_dt - cfg_rate * cfg_dphi_dt
        with cfg_rate=0.5 amplifies output by 1.5x, causing extreme values [-5, +3.26].

        Root cause: PyTorch 2.9's float32 precision in neural network operations
        combined with CFG amplification creates numerical instability.

        Solution: Monkey-patch solve_euler to clip velocity estimates before
        accumulation to prevent unbounded growth.
        """
        if "s2mel_normalization" in cls._patches_applied:
            return

        try:
            import torch
            import sys

            # Only apply patch on PyTorch 2.9+
            torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            if torch_version < (2, 9):
                return

            # Find and patch the BASECFM class's solve_euler method
            for module_name, module in list(sys.modules.items()):
                if 'flow_matching' in module_name.lower() and module is not None:
                    if hasattr(module, 'BASECFM'):
                        original_solve_euler = module.BASECFM.solve_euler

                        def patched_solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
                            """Patched solve_euler that clips velocity to prevent numerical instability in PyTorch 2.9"""
                            t, _, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]
                            sol = []
                            prompt_len = prompt.size(-1)
                            prompt_x = torch.zeros_like(x)
                            prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
                            x[..., :prompt_len] = 0
                            if hasattr(self, 'zero_prompt_speech_token') and self.zero_prompt_speech_token:
                                mu[..., :prompt_len] = 0

                            from tqdm import tqdm
                            for step in tqdm(range(1, len(t_span))):
                                dt = t_span[step] - t_span[step - 1]
                                if inference_cfg_rate > 0:
                                    stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                                    stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                                    stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                                    stacked_x = torch.cat([x, x], dim=0)
                                    stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)

                                    stacked_dphi_dt = self.estimator(
                                        stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu,
                                    )
                                    dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
                                    dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
                                else:
                                    dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)

                                # PYTORCH 2.9 FIX: Clip velocity to prevent extreme values
                                dphi_dt = torch.clamp(dphi_dt, -2.0, 2.0)

                                x = x + dt * dphi_dt
                                t = t + dt
                                sol.append(x)
                                if step < len(t_span) - 1:
                                    dt = t_span[step + 1] - t
                                x[:, :, :prompt_len] = 0

                            return sol[-1]

                        module.BASECFM.solve_euler = patched_solve_euler
                        break

            cls._patches_applied.add("s2mel_normalization")

            if verbose:
                print("   🔧 S2Mel CFM solver patched (velocity clipping for PyTorch 2.9)")

        except Exception as e:
            warnings.warn(f"S2Mel CFM solver patch failed: {e}")

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
