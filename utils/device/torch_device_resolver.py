"""
Torch Device Resolver - Intelligent device selection for TTS engines
Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU with automatic detection
"""

import torch
from typing import Literal

# Cache for "auto" device resolution to ensure consistency across calls
# Critical for cache key generation in model loading
_cached_auto_device = None


def resolve_torch_device(device: str) -> str:
    """
    Resolve device string to actual available device on current system.

    Handles intelligent device selection with fallback chain:
    - "auto": Detects best available (MPS > CUDA > XPU > CPU) and caches result
    - "mps": Apple Silicon GPU (macOS only)
    - "cuda": NVIDIA GPU
    - "xpu": Intel GPU
    - "cpu": CPU fallback

    Args:
        device: Device specification string ("auto", "cuda", "mps", "xpu", or "cpu")

    Returns:
        Resolved device string: "cuda", "mps", "xpu", or "cpu"

    Example:
        >>> device = resolve_torch_device("auto")  # Returns "cuda" on Linux with NVIDIA
        >>> device = resolve_torch_device("auto")  # Returns "mps" on macOS with Apple Silicon
        >>> device = resolve_torch_device("auto")  # Returns "xpu" on Linux with Intel GPU
        >>> device = resolve_torch_device("auto")  # Returns "cpu" on any system without GPU

    Note:
        The "auto" device resolution is cached after first call to ensure consistency.
        This prevents cache key mismatches in model loading when device availability changes mid-session.
    """
    global _cached_auto_device

    if device == "auto":
        # Return cached resolution if available to ensure consistency
        if _cached_auto_device is not None:
            return _cached_auto_device

        # Intelligent device detection with proper fallback chain
        # Check MPS first (Apple Silicon)
        if torch.backends.mps.is_available():
            _cached_auto_device = "mps"
        # Check CUDA second (NVIDIA GPUs)
        elif torch.cuda.is_available():
            _cached_auto_device = "cuda"
        # Check XPU third (Intel GPUs)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            _cached_auto_device = "xpu"
        # CPU fallback (always available)
        else:
            _cached_auto_device = "cpu"

        return _cached_auto_device

    # Explicit device selection - pass through as-is
    # Validation happens when torch.device() is created
    return device


def reset_device_cache() -> None:
    """
    Reset cached device resolution.

    Use this if device availability changes (e.g., after GPU reset or power state change).
    Normally should not be needed - cache persists for session consistency.
    """
    global _cached_auto_device
    _cached_auto_device = None


def validate_device_string(device: str) -> bool:
    """
    Validate that device string is valid.

    Args:
        device: Device string to validate

    Returns:
        True if device is valid, False otherwise
    """
    valid_devices = {"auto", "cuda", "mps", "xpu", "cpu"}
    return device.lower() in valid_devices
