"""
IndexTTS-2 Engine for TTS Audio Suite

A breakthrough emotionally expressive and duration-controlled autoregressive zero-shot TTS engine.
Features emotion disentanglement, voice cloning, and precise duration control.
"""

from .index_tts import IndexTTSEngine
from .index_tts_downloader import IndexTTSDownloader

__all__ = ['IndexTTSEngine', 'IndexTTSDownloader']