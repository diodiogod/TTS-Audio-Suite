"""
Qwen3-TTS Engine

Unified engine supporting 3 model variants:
- CustomVoice: 9 preset speakers with optional instruction control
- VoiceDesign: Text-to-voice design (UNIQUE FEATURE)
- Base: Zero-shot voice cloning from 3-second audio

Sample rate: 24kHz
Languages: 10 (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
"""

# PATCH: TTS Audio Suite - Fix PyTorch 2.8.0 duplicate registration bug
# Note: This patch is applied in the main __init__.py BEFORE any imports
# No fallback needed here - if the patch wasn't applied at extension load, it's too late now

from .qwen3_tts import Qwen3TTSEngine

__all__ = ["Qwen3TTSEngine"]
