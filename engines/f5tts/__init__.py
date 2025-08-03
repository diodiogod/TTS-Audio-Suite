"""
F5-TTS Integration Package for ChatterBox Voice
Provides F5-TTS functionality following ChatterBox patterns
"""

# Set up global warning filters for F5-TTS
import warnings

# Filter out specific warning messages that may come from F5-TTS dependencies
warnings.filterwarnings("ignore", message=".*torch.load.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Import F5-TTS modules with error handling
F5TTS_IMPORT_ERROR = None
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from .f5tts import ChatterBoxF5TTS, get_f5tts_models
    F5TTS_AVAILABLE = True
except ImportError as e:
    F5TTS_AVAILABLE = False
    F5TTS_IMPORT_ERROR = str(e)
    # Create dummy class for compatibility
    class ChatterBoxF5TTS:
        @classmethod
        def from_pretrained(cls, device):
            raise ImportError(f"F5-TTS not available: {F5TTS_IMPORT_ERROR}")
        @classmethod
        def from_local(cls, path, device, model_name):
            raise ImportError(f"F5-TTS not available: {F5TTS_IMPORT_ERROR}")
    
    # Create dummy function for compatibility
    def get_f5tts_models():
        return ["F5TTS_Base", "F5TTS_v1_Base", "E2TTS_Base"]

__all__ = ['ChatterBoxF5TTS', 'get_f5tts_models', 'F5TTS_AVAILABLE', 'F5TTS_IMPORT_ERROR']