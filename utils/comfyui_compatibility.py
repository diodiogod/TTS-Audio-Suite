"""
ComfyUI Compatibility Patches for TTS Audio Suite

This module applies necessary patches to resolve compatibility issues
between ComfyUI versions and the TTS Audio Suite.
"""

import sys
import logging

_cudnn_fix_applied = False

def ensure_python312_cudnn_fix():
    """
    Ensure Python 3.12 CUDNN benchmark fix is applied before TTS operations.

    This is called from TTS nodes before generation to prevent VRAM spikes.
    """
    global _cudnn_fix_applied

    if _cudnn_fix_applied or sys.version_info[:2] != (3, 12):
        return

    try:
        import torch
        from comfy.cli_args import args, PerformanceFeature

        # Check if CUDNN benchmark is enabled (the problematic setting)
        if (torch.cuda.is_available() and
            torch.backends.cudnn.is_available() and
            hasattr(args, 'fast') and
            args.fast and
            PerformanceFeature.AutoTune in args.fast):

            # Disable the problematic CUDNN benchmarking
            torch.backends.cudnn.benchmark = False
            print("🩹 TTS AUDIO SUITE CUDNN FIX APPLIED")
            print("   Disabled CUDNN benchmark on Python 3.12 to prevent VRAM spikes")
            print("   This fixes ComfyUI v0.3.57+ regression - VRAM spikes eliminated!")
            _cudnn_fix_applied = True

    except Exception as e:
        print(f"🩹 Warning: Could not apply CUDNN fix: {e}")

def apply_all_compatibility_patches():
    """Apply all necessary ComfyUI compatibility patches."""
    # Python 3.12 CUDNN fix
    if sys.version_info[:2] == (3, 12):
        print("🩹 TTS Audio Suite: Python 3.12 CUDNN fix ready (will apply before TTS generation)")

    # Apply PyTorch 2.9+ TorchCodec global patches - CRITICAL for Windows users
    # PyTorch 2.9 made torchaudio.save/load depend on TorchCodec which doesn't support Windows
    # This globally monkey-patches torchaudio to use scipy instead (pure Python, no dependencies)
    try:
        from utils.compatibility.pytorch_patches import apply_pytorch_patches
        apply_pytorch_patches(verbose=True)
    except Exception as e:
        print(f"⚠️ Warning: Could not apply PyTorch patches: {e}")