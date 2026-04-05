"""
RVC Engine - Real-time Voice Conversion implementation for TTS Audio Suite
Provides RVC voice conversion capabilities with model management and inference
"""

from __future__ import annotations

__all__ = ["RVCEngine"]


def __getattr__(name: str):
    if name == "RVCEngine":
        from .rvc_engine import RVCEngine

        return RVCEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
