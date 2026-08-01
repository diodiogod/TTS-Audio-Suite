"""Hume TADA integration with lazy exports safe for the worker runtime."""

from .languages import (
    TADA_LANGUAGE_OPTIONS,
    format_tada_language_display,
    normalize_tada_language,
    validate_tada_model_language,
)


def __getattr__(name):
    if name in {"TadaDownloader", "tada_downloader"}:
        from .tada_downloader import TadaDownloader, tada_downloader

        return {"TadaDownloader": TadaDownloader, "tada_downloader": tada_downloader}[name]
    if name in {"TADA_SAMPLE_RATE", "TadaEngine"}:
        from .tada_engine import TADA_SAMPLE_RATE, TadaEngine

        return {"TADA_SAMPLE_RATE": TADA_SAMPLE_RATE, "TadaEngine": TadaEngine}[name]
    raise AttributeError(name)


__all__ = [
    "TADA_LANGUAGE_OPTIONS",
    "TADA_SAMPLE_RATE",
    "TadaDownloader",
    "TadaEngine",
    "format_tada_language_display",
    "normalize_tada_language",
    "tada_downloader",
    "validate_tada_model_language",
]
