"""
Higgs Audio 2 Engine for TTS Audio Suite
High-quality text-to-speech with voice cloning capabilities
"""

from .higgs_audio import HiggsAudioEngine
from .higgs_audio_downloader import HiggsAudioDownloader

__all__ = ["HiggsAudioEngine", "HiggsAudioDownloader"]