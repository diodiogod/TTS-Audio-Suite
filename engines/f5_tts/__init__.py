"""
F5-TTS (Bundled Version)

This is a bundled version of F5-TTS included with TTS Audio Suite to resolve dependency conflicts.

Original project: https://github.com/SWivid/F5-TTS
License: MIT License
"""

# Make the main API available at package level
try:
    from .api import F5TTS
    __all__ = ['F5TTS']
except ImportError:
    # Graceful degradation if dependencies are missing
    __all__ = []

__version__ = "1.1.7-bundled"