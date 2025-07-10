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
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from .f5tts import ChatterBoxF5TTS
    F5TTS_AVAILABLE = True
except ImportError:
    F5TTS_AVAILABLE = False
    # Create dummy class for compatibility
    class ChatterBoxF5TTS:
        @classmethod
        def from_pretrained(cls, device):
            raise ImportError("F5-TTS not available - missing dependencies")
        @classmethod
        def from_local(cls, path, device, model_name):
            raise ImportError("F5-TTS not available - missing dependencies")

__all__ = ['ChatterBoxF5TTS', 'F5TTS_AVAILABLE']