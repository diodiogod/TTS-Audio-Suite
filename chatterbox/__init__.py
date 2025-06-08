# Import main TTS/VC modules with error handling
try:
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

# SRT subtitle support modules - import independently
try:
    from .srt_parser import SRTParser, SRTSubtitle, SRTParseError, validate_srt_timing_compatibility
    from .audio_timing import (
        AudioTimingUtils, PhaseVocoderTimeStretcher, TimedAudioAssembler,
        calculate_timing_adjustments, AudioTimingError
    )
    SRT_AVAILABLE = True
    
    __all__ = [
        'ChatterboxTTS', 'ChatterboxVC',
        'SRTParser', 'SRTSubtitle', 'SRTParseError', 'validate_srt_timing_compatibility',
        'AudioTimingUtils', 'PhaseVocoderTimeStretcher', 'TimedAudioAssembler',
        'calculate_timing_adjustments', 'AudioTimingError'
    ]
except ImportError:
    SRT_AVAILABLE = False
    # SRT support not available - only export main modules
    __all__ = ['ChatterboxTTS', 'ChatterboxVC']
