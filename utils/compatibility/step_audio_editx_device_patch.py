"""
Step Audio EditX Device Compatibility Patch

Patches hardcoded CUDA device calls to support MPS (Apple Silicon) and CPU.

Issue: Step Audio EditX bundled implementation (from Alibaba) has hardcoded
torch.device('cuda') calls that fail on Mac (MPS) and CPU-only systems.

Affected file:
- stepvocoder/cosyvoice2/transformer/upsample_encoder_v2.py
  Lines: 270, 272, 274, 306, 308, 310

Solution: Monkey-patch the _init_cuda_graph method to use the actual device
from self.linear_out (which is correctly set during model initialization).
"""

import warnings
from typing import Optional


class StepAudioEditXDevicePatches:
    """Centralized Step Audio EditX device compatibility patches manager"""

    _patches_applied = set()

    @classmethod
    def apply_all_patches(cls, verbose: bool = True):
        """Apply all Step Audio EditX device compatibility patches"""
        cls.patch_cuda_graph_device(verbose=verbose)

    @classmethod
    def patch_cuda_graph_device(cls, verbose: bool = True):
        """
        Patch hardcoded CUDA device in CUDA Graph initialization.

        Issue: upsample_encoder_v2.py hardcodes torch.device('cuda') in _init_cuda_graph
        This fails on Mac (MPS) and CPU-only systems with "Torch not compiled with CUDA enabled"

        Solution: Replace _init_cuda_graph method to dynamically detect device from model
        """
        if "cuda_graph_device" in cls._patches_applied:
            return

        try:
            # Import the module that needs patching
            import sys
            import os

            # Add Step Audio EditX impl to path if not already
            impl_path = None
            for path_item in sys.path:
                if 'step_audio_editx_impl' in path_item or path_item.endswith('step_audio_editx/impl'):
                    impl_path = path_item
                    break

            if impl_path is None:
                # Try to find it relative to this file
                compat_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(compat_dir))
                impl_path = os.path.join(project_root, 'engines', 'step_audio_editx', 'step_audio_editx_impl')
                if os.path.exists(impl_path) and impl_path not in sys.path:
                    sys.path.insert(0, impl_path)

            # Try to import the module
            try:
                from stepvocoder.cosyvoice2.transformer.upsample_encoder_v2 import UpsampleEncoderV2
            except ImportError:
                # Module not loaded yet, patch will be applied when it loads
                if verbose:
                    print("ℹ️ [Step Audio EditX Device Patch] Module not loaded yet, will patch on first use")
                return

            # Check if already patched
            if hasattr(UpsampleEncoderV2._init_cuda_graph, '_device_patched'):
                if verbose:
                    print("✓ [Step Audio EditX Device Patch] Already applied")
                cls._patches_applied.add("cuda_graph_device")
                return

            # Save original method
            original_init_cuda_graph = UpsampleEncoderV2._init_cuda_graph

            # Track if we've already logged for non-CUDA devices (to avoid spam)
            _logged_non_cuda = {'logged': False}

            # Create patched version
            def patched_init_cuda_graph(self):
                """
                Patched version that uses dynamic device instead of hardcoded 'cuda'.

                Detects device from model's parameters (self.linear_out) and uses that
                for CUDA Graph initialization. Falls back to disabling CUDA graphs if
                not on CUDA device.
                """
                import torch

                # Detect actual device from model
                if hasattr(self, 'linear_out') and hasattr(self.linear_out, 'weight'):
                    device = self.linear_out.weight.device
                else:
                    # Fallback: assume cuda if cuda is available
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # CUDA Graphs only work on CUDA devices
                if device.type != 'cuda':
                    # Log once per session for non-CUDA devices
                    if not _logged_non_cuda['logged']:
                        print(f"ℹ️ Step Audio EditX: CUDA Graphs disabled (running on {device.type})")
                        _logged_non_cuda['logged'] = True
                    self.enable_cuda_graph = False
                    return

                # Call original implementation (it will use CUDA correctly)
                original_init_cuda_graph(self)

            # Mark as patched
            patched_init_cuda_graph._device_patched = True

            # Apply patch
            UpsampleEncoderV2._init_cuda_graph = patched_init_cuda_graph

            if verbose:
                print("✓ [Step Audio EditX Device Patch] Applied CUDA Graph device compatibility patch")

            cls._patches_applied.add("cuda_graph_device")

        except ImportError as e:
            if verbose:
                print(f"⚠️ [Step Audio EditX Device Patch] Could not apply patch (module not available): {e}")
        except Exception as e:
            if verbose:
                print(f"⚠️ [Step Audio EditX Device Patch] Failed to apply patch: {e}")


# Auto-apply patches on import
def auto_apply_patches():
    """Automatically apply patches when this module is imported"""
    StepAudioEditXDevicePatches.apply_all_patches(verbose=False)


# Export
__all__ = ["StepAudioEditXDevicePatches", "auto_apply_patches"]
