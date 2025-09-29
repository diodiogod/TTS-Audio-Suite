"""
VibeVoice Engine Module for TTS Audio Suite
Microsoft VibeVoice TTS integration with multi-speaker and long-form capabilities
"""

import warnings

# Check VibeVoice package availability
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Test the specific imports we need
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        
        # Apply transformers compatibility patches for newer versions
        from .transformers_compatibility import apply_all_compatibility_patches
        apply_all_compatibility_patches()
        
        # If imports work, load our engine components
        from .vibevoice_engine import VibeVoiceEngine
        from .vibevoice_downloader import VibeVoiceDownloader, VIBEVOICE_MODELS
    VIBEVOICE_AVAILABLE = True
except ImportError as e:
    VIBEVOICE_AVAILABLE = False
    VIBEVOICE_IMPORT_ERROR = str(e)
    # Create dummy classes for compatibility
    class VibeVoiceEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"VibeVoice not available: {VIBEVOICE_IMPORT_ERROR}")
    
    class VibeVoiceDownloader:
        @classmethod
        def get_available_models(cls):
            return {}

        def download_huggingface_model(self, *args, **kwargs):
            raise ImportError(f"VibeVoice not available: {VIBEVOICE_IMPORT_ERROR}")

        def get_model_path(self, model_name):
            raise ImportError(f"VibeVoice not available: {VIBEVOICE_IMPORT_ERROR}")

        def ensure_vibevoice_package(self):
            return False
    
    VIBEVOICE_MODELS = {}

__all__ = ['VibeVoiceEngine', 'VibeVoiceDownloader', 'VIBEVOICE_MODELS', 'VIBEVOICE_AVAILABLE']