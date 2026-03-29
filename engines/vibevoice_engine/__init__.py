"""
VibeVoice engine package.

Keep package import lightweight. Importing this package during node discovery
must not eagerly import the external `vibevoice`/`diffusers` stack, otherwise a
transient startup-time import failure poisons the module for the whole process.
"""

from .vibevoice_downloader import VibeVoiceDownloader, VIBEVOICE_MODELS

# Package import being available does not guarantee the external vibevoice stack
# is healthy. Real availability is checked lazily when VibeVoiceEngine is used.
VIBEVOICE_AVAILABLE = True


def __getattr__(name):
    if name == "VibeVoiceEngine":
        try:
            from .vibevoice_engine import VibeVoiceEngine
            return VibeVoiceEngine
        except Exception as e:
            raise ImportError(f"VibeVoice not available: {e}") from e

    if name == "VIBEVOICE_IMPORT_ERROR":
        return None

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["VibeVoiceEngine", "VibeVoiceDownloader", "VIBEVOICE_MODELS", "VIBEVOICE_AVAILABLE"]
