# Engine Adapters Package

# Import adapters with error handling
try:
    from .chatterbox_adapter import ChatterBoxEngineAdapter
    CHATTERBOX_ADAPTER_AVAILABLE = True
except Exception as e:
    _CHATTERBOX_IMPORT_ERROR = e  # ``e`` is unbound once this block exits
    CHATTERBOX_ADAPTER_AVAILABLE = False
    # Create dummy class for compatibility
    class ChatterBoxEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"ChatterBox adapter not available: {_CHATTERBOX_IMPORT_ERROR}")

try:
    from .f5tts_adapter import F5TTSEngineAdapter
    F5TTS_ADAPTER_AVAILABLE = True
except Exception as e:
    _F5TTS_IMPORT_ERROR = e  # ``e`` is unbound once this block exits
    F5TTS_ADAPTER_AVAILABLE = False
    # Create dummy class for compatibility
    class F5TTSEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"F5-TTS adapter not available: {_F5TTS_IMPORT_ERROR}")

try:
    from .cosyvoice_adapter import CosyVoiceAdapter
    COSYVOICE_ADAPTER_AVAILABLE = True
except Exception as e:
    _COSYVOICE_IMPORT_ERROR = e  # ``e`` is unbound once this block exits
    COSYVOICE_ADAPTER_AVAILABLE = False
    # Create dummy class for compatibility
    class CosyVoiceAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"CosyVoice adapter not available: {_COSYVOICE_IMPORT_ERROR}")

try:
    from .echo_tts_adapter import EchoTTSEngineAdapter
    ECHO_TTS_ADAPTER_AVAILABLE = True
except Exception as e:
    _ECHO_TTS_IMPORT_ERROR = e  # ``e`` is unbound once this block exits
    ECHO_TTS_ADAPTER_AVAILABLE = False
    class EchoTTSEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Echo-TTS adapter not available: {_ECHO_TTS_IMPORT_ERROR}")

try:
    from .dots_tts_adapter import DotsTTSEngineAdapter
    DOTS_TTS_ADAPTER_AVAILABLE = True
except Exception as e:
    _DOTS_TTS_IMPORT_ERROR = e  # ``e`` is unbound once this block exits
    DOTS_TTS_ADAPTER_AVAILABLE = False
    class DotsTTSEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Dots TTS adapter not available: {_DOTS_TTS_IMPORT_ERROR}")

try:
    from .dramabox_adapter import DramaBoxEngineAdapter
    DRAMABOX_ADAPTER_AVAILABLE = True
except Exception as e:
    _DRAMABOX_IMPORT_ERROR = e  # ``e`` is unbound once this block exits
    DRAMABOX_ADAPTER_AVAILABLE = False
    class DramaBoxEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"DramaBox adapter not available: {_DRAMABOX_IMPORT_ERROR}")

try:
    from .fish_audio_s2_adapter import FishAudioS2Adapter
    FISH_AUDIO_S2_ADAPTER_AVAILABLE = True
except Exception as e:
    _FISH_AUDIO_S2_IMPORT_ERROR = e  # ``e`` is unbound once this block exits
    FISH_AUDIO_S2_ADAPTER_AVAILABLE = False
    class FishAudioS2Adapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Fish Audio S2 adapter not available: {_FISH_AUDIO_S2_IMPORT_ERROR}")

try:
    from .omnivoice_adapter import OmniVoiceEngineAdapter
    OMNIVOICE_ADAPTER_AVAILABLE = True
except Exception as e:
    _OMNIVOICE_IMPORT_ERROR = e  # ``e`` is unbound once this block exits
    OMNIVOICE_ADAPTER_AVAILABLE = False
    class OmniVoiceEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"OmniVoice adapter not available: {_OMNIVOICE_IMPORT_ERROR}")

try:
    from .zonos3_mini_adapter import Zonos3MiniEngineAdapter
    ZONOS3_MINI_ADAPTER_AVAILABLE = True
except Exception as e:
    _ZONOS3_MINI_IMPORT_ERROR = e  # ``e`` is unbound once this block exits
    ZONOS3_MINI_ADAPTER_AVAILABLE = False
    class Zonos3MiniEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"ZONOS3-Mini adapter not available: {_ZONOS3_MINI_IMPORT_ERROR}")

try:
    from .moss_tts_adapter import MossTTSEngineAdapter
    MOSS_TTS_ADAPTER_AVAILABLE = True
except Exception as e:
    _MOSS_TTS_IMPORT_ERROR = e  # ``e`` is unbound once this block exits
    MOSS_TTS_ADAPTER_AVAILABLE = False
    class MossTTSEngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"MOSS-TTS adapter not available: {_MOSS_TTS_IMPORT_ERROR}")

try:
    from .higgs_audio_v3_adapter import HiggsAudioV3EngineAdapter
    HIGGS_AUDIO_V3_ADAPTER_AVAILABLE = True
except Exception as e:
    _HIGGS_AUDIO_V3_IMPORT_ERROR = e  # ``e`` is unbound once this block exits
    HIGGS_AUDIO_V3_ADAPTER_AVAILABLE = False
    class HiggsAudioV3EngineAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"Higgs Audio v3 adapter not available: {_HIGGS_AUDIO_V3_IMPORT_ERROR}")

__all__ = [
    'ChatterBoxEngineAdapter', 'F5TTSEngineAdapter', 'CosyVoiceAdapter', 'EchoTTSEngineAdapter',
    'DotsTTSEngineAdapter', 'DramaBoxEngineAdapter', 'OmniVoiceEngineAdapter',
    'MossTTSEngineAdapter', 'HiggsAudioV3EngineAdapter', 'Zonos3MiniEngineAdapter',
    'CHATTERBOX_ADAPTER_AVAILABLE', 'F5TTS_ADAPTER_AVAILABLE', 'COSYVOICE_ADAPTER_AVAILABLE',
    'ECHO_TTS_ADAPTER_AVAILABLE', 'DOTS_TTS_ADAPTER_AVAILABLE',
    'DRAMABOX_ADAPTER_AVAILABLE', 'OMNIVOICE_ADAPTER_AVAILABLE',
    'MOSS_TTS_ADAPTER_AVAILABLE', 'HIGGS_AUDIO_V3_ADAPTER_AVAILABLE', 'ZONOS3_MINI_ADAPTER_AVAILABLE',
    'MossSoundEffectV2Adapter'
]

from .moss_soundeffect_v2_adapter import MossSoundEffectV2Adapter
