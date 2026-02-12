"""KugelAudio - Open Source Text-to-Speech Model

KugelAudio is a state-of-the-art neural text-to-speech model that generates
natural, expressive speech from text using pre-encoded voices.

Note:
    Voice cloning from raw audio is not supported. Use pre-encoded voices
    from the voices.json registry instead.

Example:
    >>> from engines.vibevoice_engine.kugelaudio_impl import KugelAudioForConditionalGenerationInference, KugelAudioProcessor
    >>> model = KugelAudioForConditionalGenerationInference.from_pretrained("kugelaudio/kugelaudio-0-open")
    >>> model.model.strip_encoders()  # Free VRAM
    >>> processor = KugelAudioProcessor.from_pretrained("kugelaudio/kugelaudio-0-open")
    >>> inputs = processor(text="Hello world!", voice="default", return_tensors="pt")
"""

__version__ = "0.1.0"

from .configs import (
    KugelAudioAcousticTokenizerConfig,
    KugelAudioConfig,
    KugelAudioDiffusionHeadConfig,
    KugelAudioSemanticTokenizerConfig,
)
from .models import (
    KugelAudioAcousticTokenizerModel,
    KugelAudioDiffusionHead,
    KugelAudioForConditionalGeneration,
    KugelAudioForConditionalGenerationInference,
    KugelAudioModel,
    KugelAudioPreTrainedModel,
    KugelAudioSemanticTokenizerModel,
)
from .processors import KugelAudioProcessor
from .schedule import DPMSolverMultistepScheduler
from .watermark import AudioWatermark


# Lazy imports for optional components
def launch_ui(*args, **kwargs):
    """Launch the Gradio web interface."""
    try:
        from .ui import launch_ui as _launch_ui

        return _launch_ui(*args, **kwargs)
    except ImportError:
        raise ImportError(
            "Gradio is required for the web interface. " "Install it with: pip install gradio"
        )


__all__ = [
    # Version
    "__version__",
    # Configs
    "KugelAudioConfig",
    "KugelAudioAcousticTokenizerConfig",
    "KugelAudioSemanticTokenizerConfig",
    "KugelAudioDiffusionHeadConfig",
    # Models
    "KugelAudioModel",
    "KugelAudioPreTrainedModel",
    "KugelAudioForConditionalGeneration",
    "KugelAudioForConditionalGenerationInference",
    "KugelAudioAcousticTokenizerModel",
    "KugelAudioSemanticTokenizerModel",
    "KugelAudioDiffusionHead",
    # Scheduler
    "DPMSolverMultistepScheduler",
    # Processors
    "KugelAudioProcessor",
    # Watermark
    "AudioWatermark",
    # UI
    "launch_ui",
]
