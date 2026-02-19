# Set up global warning filters
import warnings

# Filter out specific warning messages
warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear.*")
warnings.filterwarnings("ignore", message=".*PerthNet.*")
warnings.filterwarnings("ignore", message=".*requires authentication.*")

# --- LAZY ENGINE IMPORTS ---
# TTS/VC/F5TTS classes are NOT imported at module level to avoid pulling in
# transformers (~5s), diffusers (~3s), and torch._dynamo (~3s) at plugin startup.
# Instead they are loaded on first access via module-level __getattr__.
# This breaks the import chain:
#   language_mapper -> engines.chatterbox.language_models -> (this __init__.py)
#   which previously triggered: .tts -> models.t3 -> transformers/diffusers
#
# Consumers that only need language_models (CHATTERBOX_MODELS, get_available_languages)
# no longer pay the ~8s cost of importing the model definition code.

_TTS_AVAILABLE = None
_VC_AVAILABLE = None
_F5TTS_SUPPORT_AVAILABLE = None
_ChatterboxTTS = None
_ChatterboxVC = None
_ChatterBoxF5TTS = None


def _ensure_tts_loaded():
    """Lazily load ChatterboxTTS on first use."""
    global _TTS_AVAILABLE, _ChatterboxTTS
    if _TTS_AVAILABLE is not None:
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from .tts import ChatterboxTTS as _cls
        _ChatterboxTTS = _cls
        _TTS_AVAILABLE = True
    except ImportError:
        _TTS_AVAILABLE = False
        class _DummyTTS:
            @classmethod
            def from_pretrained(cls, device):
                raise ImportError("ChatterboxTTS not available - missing dependencies")
            @classmethod
            def from_local(cls, path, device):
                raise ImportError("ChatterboxTTS not available - missing dependencies")
        _ChatterboxTTS = _DummyTTS


def _ensure_vc_loaded():
    """Lazily load ChatterboxVC on first use."""
    global _VC_AVAILABLE, _ChatterboxVC
    if _VC_AVAILABLE is not None:
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from .vc import ChatterboxVC as _cls
        _ChatterboxVC = _cls
        _VC_AVAILABLE = True
    except ImportError:
        _VC_AVAILABLE = False
        class _DummyVC:
            @classmethod
            def from_pretrained(cls, device):
                raise ImportError("ChatterboxVC not available - missing dependencies")
            @classmethod
            def from_local(cls, path, device):
                raise ImportError("ChatterboxVC not available - missing dependencies")
        _ChatterboxVC = _DummyVC


def _ensure_f5tts_loaded():
    """Lazily load ChatterBoxF5TTS on first use."""
    global _F5TTS_SUPPORT_AVAILABLE, _ChatterBoxF5TTS
    if _F5TTS_SUPPORT_AVAILABLE is not None:
        return
    try:
        from .f5tts import ChatterBoxF5TTS as _cls, F5TTS_AVAILABLE
        _ChatterBoxF5TTS = _cls
        _F5TTS_SUPPORT_AVAILABLE = F5TTS_AVAILABLE
    except ImportError:
        _F5TTS_SUPPORT_AVAILABLE = False
        class _DummyF5TTS:
            @classmethod
            def from_pretrained(cls, device, model_name):
                raise ImportError("F5-TTS not available - missing dependencies")
            @classmethod
            def from_local(cls, path, device, model_name):
                raise ImportError("F5-TTS not available - missing dependencies")
        _ChatterBoxF5TTS = _DummyF5TTS


def __getattr__(name):
    """Module-level lazy attribute access for TTS/VC/F5TTS classes and availability flags."""
    if name == "ChatterboxTTS":
        _ensure_tts_loaded()
        return _ChatterboxTTS
    elif name == "TTS_AVAILABLE":
        _ensure_tts_loaded()
        return _TTS_AVAILABLE
    elif name == "ChatterboxVC":
        _ensure_vc_loaded()
        return _ChatterboxVC
    elif name == "VC_AVAILABLE":
        _ensure_vc_loaded()
        return _VC_AVAILABLE
    elif name == "ChatterBoxF5TTS":
        _ensure_f5tts_loaded()
        return _ChatterBoxF5TTS
    elif name == "F5TTS_SUPPORT_AVAILABLE":
        _ensure_f5tts_loaded()
        return _F5TTS_SUPPORT_AVAILABLE
    raise AttributeError(f"module 'engines.chatterbox' has no attribute {name!r}")

# Language models support
try:
    from .language_models import (
        get_chatterbox_models, get_model_config, get_model_files_for_language,
        find_local_model_path, detect_model_format, get_available_languages,
        is_model_incomplete, get_model_requirements, validate_model_completeness,
        get_tokenizer_filename
    )
    LANGUAGE_MODELS_AVAILABLE = True
except ImportError:
    LANGUAGE_MODELS_AVAILABLE = False
    # Create dummy functions for compatibility
    def get_available_languages():
        return ["English"]
    def find_local_model_path(language):
        return None
    def get_chatterbox_models():
        return ["English"]
    def get_model_config(language):
        return None
    def get_model_files_for_language(language):
        return ("pt", "ResembleAI/chatterbox")
    def detect_model_format(model_path):
        return "pt"
    def is_model_incomplete(language):
        return False
    def get_model_requirements(language):
        return []
    def validate_model_completeness(model_path, language):
        return True, []
    def get_tokenizer_filename(language):
        return "tokenizer.json"

# SRT subtitle support modules - import independently
try:
    from .srt_parser import SRTParser, SRTSubtitle, SRTParseError, validate_srt_timing_compatibility
    from .audio_timing import (
        AudioTimingUtils, PhaseVocoderTimeStretcher, TimedAudioAssembler,
        calculate_timing_adjustments, AudioTimingError
    )
    SRT_AVAILABLE = True
    
    __all__ = [
        'ChatterboxTTS', 'ChatterboxVC', 'ChatterBoxF5TTS',
        'get_chatterbox_models', 'get_model_config', 'get_model_files_for_language',
        'find_local_model_path', 'detect_model_format', 'get_available_languages',
        'is_model_incomplete', 'get_model_requirements', 'validate_model_completeness',
        'get_tokenizer_filename',
        'SRTParser', 'SRTSubtitle', 'SRTParseError', 'validate_srt_timing_compatibility',
        'AudioTimingUtils', 'PhaseVocoderTimeStretcher', 'TimedAudioAssembler',
        'calculate_timing_adjustments', 'AudioTimingError'
    ]
except ImportError:
    SRT_AVAILABLE = False
    # SRT support not available - only export main modules and language functions
    __all__ = [
        'ChatterboxTTS', 'ChatterboxVC', 'ChatterBoxF5TTS',
        'get_chatterbox_models', 'get_model_config', 'get_model_files_for_language',
        'find_local_model_path', 'detect_model_format', 'get_available_languages',
        'is_model_incomplete', 'get_model_requirements', 'validate_model_completeness',
        'get_tokenizer_filename'
    ]
