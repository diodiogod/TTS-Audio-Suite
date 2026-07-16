"""
Engine-specific handlers for ComfyUI model wrapper system
"""

from .base_handler import BaseEngineHandler
from .vibevoice_handler import VibeVoiceHandler
from .higgs_audio_handler import HiggsAudioHandler
from .generic_handler import GenericHandler
from .step_audio_editx_handler import StepAudioEditXHandler
from .cosyvoice_handler import CosyVoiceHandler
from .qwen3_tts_handler import Qwen3TTSHandler
from .moss_tts_handler import MossTTSHandler
from .moss_soundeffect_v2_handler import MossSoundEffectV2Handler


def get_engine_handler(engine: str) -> BaseEngineHandler:
    """
    Get the appropriate engine handler for an engine.

    Args:
        engine: Engine name ("chatterbox", "f5tts", "higgs_audio", "stateless_tts", "vibevoice", "step_audio_editx", "cosyvoice", "qwen3_tts", etc.)

    Returns:
        Engine-specific handler instance
    """
    if engine == "vibevoice":
        return VibeVoiceHandler()
    elif engine == "higgs_audio" or engine == "stateless_tts":
        # Both higgs_audio and stateless_tts (higgs audio stateless wrapper) use HiggsAudioHandler
        return HiggsAudioHandler()
    elif engine == "step_audio_editx":
        # Step Audio EditX handler for bitsandbytes int8/int4 support
        return StepAudioEditXHandler()
    elif engine == "cosyvoice":
        # CosyVoice handler for proper component-level device management
        return CosyVoiceHandler()
    elif engine == "qwen3_tts":
        # Qwen3-TTS handler for CUDA graph cleanup
        return Qwen3TTSHandler()
    elif engine == "moss_tts":
        # Large MOSS checkpoints must be released, not copied into system RAM.
        return MossTTSHandler()
    elif engine == "moss_soundeffect_v2":
        # The v2 diffusion stack is also too large to copy to RAM on removal.
        return MossSoundEffectV2Handler()
    else:
        # Generic handler for chatterbox, f5tts, rvc, etc.
        return GenericHandler()


__all__ = [
    'BaseEngineHandler',
    'VibeVoiceHandler',
    'HiggsAudioHandler',
    'GenericHandler',
    'StepAudioEditXHandler',
    'CosyVoiceHandler',
    'Qwen3TTSHandler',
    'MossTTSHandler',
    'MossSoundEffectV2Handler',
    'get_engine_handler'
]
