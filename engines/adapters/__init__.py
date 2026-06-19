# Engine Adapters Package

# Import adapters with error handling
try:
    from .chatterbox_adapter import ChatterBoxEngineAdapter
    CHATTERBOX_ADAPTER_AVAILABLE = True
except Exception as e:
    CHATTERBOX_ADAPTER_AVAILABLE = False
    # Create dummy class for compatibility
    class ChatterBoxEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"ChatterBox adapter not available: {e}")

try:
    from .f5tts_adapter import F5TTSEngineAdapter
    F5TTS_ADAPTER_AVAILABLE = True
except Exception as e:
    F5TTS_ADAPTER_AVAILABLE = False
    # Create dummy class for compatibility
    class F5TTSEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"F5-TTS adapter not available: {e}")

try:
    from .cosyvoice_adapter import CosyVoiceAdapter
    COSYVOICE_ADAPTER_AVAILABLE = True
except Exception as e:
    COSYVOICE_ADAPTER_AVAILABLE = False
    # Create dummy class for compatibility
    class CosyVoiceAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CosyVoice adapter not available: {e}")

try:
    from .echo_tts_adapter import EchoTTSEngineAdapter
    ECHO_TTS_ADAPTER_AVAILABLE = True
except Exception as e:
    ECHO_TTS_ADAPTER_AVAILABLE = False
    class EchoTTSEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Echo-TTS adapter not available: {e}")

try:
    from .dots_tts_adapter import DotsTTSEngineAdapter
    DOTS_TTS_ADAPTER_AVAILABLE = True
except Exception as e:
    DOTS_TTS_ADAPTER_AVAILABLE = False
    class DotsTTSEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Dots TTS adapter not available: {e}")

try:
    from .omnivoice_adapter import OmniVoiceEngineAdapter
    OMNIVOICE_ADAPTER_AVAILABLE = True
except Exception as e:
    OMNIVOICE_ADAPTER_AVAILABLE = False
    class OmniVoiceEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"OmniVoice adapter not available: {e}")

try:
    from .moss_tts_adapter import MossTTSEngineAdapter
    MOSS_TTS_ADAPTER_AVAILABLE = True
except Exception as e:
    MOSS_TTS_ADAPTER_AVAILABLE = False
    class MossTTSEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"MOSS-TTS adapter not available: {e}")

try:
    from .higgs_audio_v3_adapter import HiggsAudioV3EngineAdapter
    HIGGS_AUDIO_V3_ADAPTER_AVAILABLE = True
except Exception as e:
    HIGGS_AUDIO_V3_ADAPTER_AVAILABLE = False
    class HiggsAudioV3EngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Higgs Audio v3 adapter not available: {e}")

__all__ = [
    'ChatterBoxEngineAdapter', 'F5TTSEngineAdapter', 'CosyVoiceAdapter', 'EchoTTSEngineAdapter',
    'DotsTTSEngineAdapter', 'OmniVoiceEngineAdapter',
    'MossTTSEngineAdapter', 'HiggsAudioV3EngineAdapter',
    'CHATTERBOX_ADAPTER_AVAILABLE', 'F5TTS_ADAPTER_AVAILABLE', 'COSYVOICE_ADAPTER_AVAILABLE',
    'ECHO_TTS_ADAPTER_AVAILABLE', 'DOTS_TTS_ADAPTER_AVAILABLE', 'OMNIVOICE_ADAPTER_AVAILABLE',
    'MOSS_TTS_ADAPTER_AVAILABLE', 'HIGGS_AUDIO_V3_ADAPTER_AVAILABLE'
]
