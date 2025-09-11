"""
Engine-specific handlers for ComfyUI model wrapper system
"""

from .base_handler import BaseEngineHandler
from .vibevoice_handler import VibeVoiceHandler
from .higgs_audio_handler import HiggsAudioHandler
from .generic_handler import GenericHandler


def get_engine_handler(engine: str) -> BaseEngineHandler:
    """
    Get the appropriate engine handler for an engine.
    
    Args:
        engine: Engine name ("chatterbox", "f5tts", "higgs_audio", "stateless_tts", "vibevoice", etc.)
        
    Returns:
        Engine-specific handler instance
    """
    if engine == "vibevoice":
        return VibeVoiceHandler()
    elif engine == "higgs_audio" or engine == "stateless_tts":
        # Both higgs_audio and stateless_tts (higgs audio stateless wrapper) use HiggsAudioHandler
        return HiggsAudioHandler()
    else:
        # Generic handler for chatterbox, f5tts, rvc, etc.
        return GenericHandler()


__all__ = [
    'BaseEngineHandler',
    'VibeVoiceHandler', 
    'HiggsAudioHandler',
    'GenericHandler',
    'get_engine_handler'
]