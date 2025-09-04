# Set up global warning filters
import warnings

# Filter out specific warning messages
warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear.*")
warnings.filterwarnings("ignore", message=".*PerthNet.*")
warnings.filterwarnings("ignore", message=".*requires authentication.*")

# Import main TTS/VC modules with error handling
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from .tts import ChatterboxTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    # Create dummy class
    class ChatterboxTTS:
        @classmethod
        def from_pretrained(cls, device):
            raise ImportError("ChatterboxTTS not available - missing dependencies")
        @classmethod
        def from_local(cls, path, device):
            raise ImportError("ChatterboxTTS not available - missing dependencies")

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from .vc import ChatterboxVC
    VC_AVAILABLE = True
except ImportError:
    VC_AVAILABLE = False
    # Create dummy class
    class ChatterboxVC:
        @classmethod
        def from_pretrained(cls, device):
            raise ImportError("ChatterboxVC not available - missing dependencies")
        @classmethod
        def from_local(cls, path, device):
            raise ImportError("ChatterboxVC not available - missing dependencies")

# F5-TTS support modules - import independently
try:
    from .f5tts import ChatterBoxF5TTS, F5TTS_AVAILABLE
    F5TTS_SUPPORT_AVAILABLE = F5TTS_AVAILABLE
except ImportError:
    F5TTS_SUPPORT_AVAILABLE = False
    # Create dummy class
    class ChatterBoxF5TTS:
        @classmethod
        def from_pretrained(cls, device, model_name):
            raise ImportError("F5-TTS not available - missing dependencies")
        @classmethod
        def from_local(cls, path, device, model_name):
            raise ImportError("F5-TTS not available - missing dependencies")

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
