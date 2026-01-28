"""
Qwen3-TTS Engine

Unified engine supporting 3 model variants:
- CustomVoice: 9 preset speakers with optional instruction control
- VoiceDesign: Text-to-voice design (UNIQUE FEATURE)
- Base: Zero-shot voice cloning from 3-second audio

Sample rate: 24kHz
Languages: 10 (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
"""

from .qwen3_tts import Qwen3TTSEngine

__all__ = ["Qwen3TTSEngine"]
