"""
Step Audio EditX Engine for TTS Audio Suite

A powerful 3B-parameter LLM-based audio model specialized in expressive audio editing.
Features zero-shot voice cloning and advanced emotion/style/speed editing capabilities.
"""

from .step_audio_editx import StepAudioEditXEngine
from .step_audio_editx_downloader import StepAudioEditXDownloader

# Expose bundled step_audio_editx_impl module for direct access
from . import step_audio_editx_impl

__all__ = ['StepAudioEditXEngine', 'StepAudioEditXDownloader', 'step_audio_editx_impl']
