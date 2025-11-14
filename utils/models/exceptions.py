"""
TTS Model Loading Exception Hierarchy

Standardized exceptions for all model loading operations across all TTS engines.
Provides consistent error handling and clear error messages to users and developers.
"""


class TTSModelLoadingError(Exception):
    """
    Base exception for all TTS model loading errors.

    All model loading failures inherit from this class, allowing callers to
    catch all model-related errors with a single except clause.
    """
    pass


class TTSModelNotFoundError(TTSModelLoadingError):
    """
    Model file or repository not found.

    Raised when:
    - Local model files don't exist
    - HuggingFace repository doesn't exist
    - Model path is invalid
    """
    pass


class TTSModelDownloadError(TTSModelLoadingError):
    """
    Error downloading model from HuggingFace or remote source.

    Raised when:
    - Network request fails
    - Repository is gated (requires authentication)
    - Download is interrupted
    - Insufficient disk space
    """
    pass


class TTSDeviceError(TTSModelLoadingError):
    """
    Device-related errors (CUDA, MPS, CPU issues).

    Raised when:
    - Device is unavailable
    - Device VRAM is insufficient
    - Device type is invalid
    - Device-specific operations fail
    """
    pass


class TTSModelFormatError(TTSModelLoadingError):
    """
    Model format compatibility or loading issues.

    Raised when:
    - Model file format is unsupported
    - Required model components are missing
    - Model architecture is incompatible
    - Weights can't be loaded into model
    """
    pass


class TTSModelInitializationError(TTSModelLoadingError):
    """
    Error initializing model after loading weights.

    Raised when:
    - Model initialization fails
    - Required dependencies missing
    - Configuration is invalid
    - Post-load setup fails
    """
    pass


class TTSLanguageNotSupportedError(TTSModelLoadingError):
    """
    Language is not supported by this model or engine.

    Raised when:
    - Language model files don't exist
    - Engine doesn't support requested language
    - Language code is invalid
    """
    pass
