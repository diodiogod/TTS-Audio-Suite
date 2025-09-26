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
    # This is called at startup but the real fix is applied per-node
    if sys.version_info[:2] == (3, 12):
        print("🩹 TTS Audio Suite: Python 3.12 CUDNN fix ready (will apply before TTS generation)")
    pass