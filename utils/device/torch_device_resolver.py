"""
Torch Device Resolver - Intelligent device selection for TTS engines
Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU with automatic detection
"""

import torch
from typing import Literal


def resolve_torch_device(device: str) -> str:
    """
    Resolve device string to actual available device on current system.

    Handles intelligent device selection with fallback chain:
    - "auto": Detects best available (MPS > CUDA > CPU)
    - "mps": Apple Silicon GPU (macOS only)
    - "cuda": NVIDIA GPU
    - "cpu": CPU fallback

    Args:
        device: Device specification string ("auto", "cuda", "mps", or "cpu")

    Returns:
        Resolved device string: "cuda", "mps", or "cpu"

    Example:
        >>> device = resolve_torch_device("auto")  # Returns "cuda" on Linux with NVIDIA
        >>> device = resolve_torch_device("auto")  # Returns "mps" on macOS with Apple Silicon
        >>> device = resolve_torch_device("auto")  # Returns "cpu" on any system without GPU
    """

    if device == "auto":
        # Intelligent device detection with proper fallback chain
        # Check MPS first (Apple Silicon)
        if torch.backends.mps.is_available():
            return "mps"
        # Check CUDA second (NVIDIA GPUs)
        elif torch.cuda.is_available():
            return "cuda"
        # CPU fallback (always available)
        else:
            return "cpu"

    # Explicit device selection - pass through as-is
    # Validation happens when torch.device() is created
    return device


def validate_device_string(device: str) -> bool:
    """
    Validate that device string is valid.

    Args:
        device: Device string to validate

    Returns:
        True if device is valid, False otherwise
    """
    valid_devices = {"auto", "cuda", "mps", "cpu"}
    return device.lower() in valid_devices
