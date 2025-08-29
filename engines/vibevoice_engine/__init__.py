"""
VibeVoice Engine Module for TTS Audio Suite
Microsoft VibeVoice TTS integration with multi-speaker and long-form capabilities
"""

from .vibevoice_engine import VibeVoiceEngine
from .vibevoice_downloader import VibeVoiceDownloader, VIBEVOICE_MODELS

__all__ = ['VibeVoiceEngine', 'VibeVoiceDownloader', 'VIBEVOICE_MODELS']