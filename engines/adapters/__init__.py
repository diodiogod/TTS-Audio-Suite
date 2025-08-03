# Engine Adapters Package

# Import adapters with error handling
try:
    from .chatterbox_adapter import ChatterBoxEngineAdapter
    CHATTERBOX_ADAPTER_AVAILABLE = True
except ImportError as e:
    CHATTERBOX_ADAPTER_AVAILABLE = False
    # Create dummy class for compatibility
    class ChatterBoxEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"ChatterBox adapter not available: {e}")

try:
    from .f5tts_adapter import F5TTSEngineAdapter
    F5TTS_ADAPTER_AVAILABLE = True
except ImportError as e:
    F5TTS_ADAPTER_AVAILABLE = False
    # Create dummy class for compatibility
    class F5TTSEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"F5-TTS adapter not available: {e}")

__all__ = ['ChatterBoxEngineAdapter', 'F5TTSEngineAdapter', 'CHATTERBOX_ADAPTER_AVAILABLE', 'F5TTS_ADAPTER_AVAILABLE']