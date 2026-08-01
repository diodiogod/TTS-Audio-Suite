"""Language metadata and validation for the official Hume TADA checkpoints."""

from __future__ import annotations

from typing import Optional


_LANGUAGE_DISPLAY_TO_CODE = {
    "English": None,
    "Arabic": "ar",
    "Chinese": "ch",
    "German": "de",
    "Spanish": "es",
    "French": "fr",
    "Italian": "it",
    "Japanese": "ja",
    "Polish": "pl",
    "Portuguese": "pt",
}

TADA_LANGUAGE_OPTIONS = tuple(_LANGUAGE_DISPLAY_TO_CODE)

_LANGUAGE_ALIASES = {
    "": None,
    "auto": None,
    "default": None,
    "en": None,
    "eng": None,
    "english": None,
    "ar": "ar",
    "arabic": "ar",
    "ch": "ch",
    "zh": "ch",
    "zh-cn": "ch",
    "chinese": "ch",
    "de": "de",
    "german": "de",
    "es": "es",
    "spanish": "es",
    "fr": "fr",
    "french": "fr",
    "it": "it",
    "italian": "it",
    "ja": "ja",
    "jp": "ja",
    "japanese": "ja",
    "pl": "pl",
    "polish": "pl",
    "pt": "pt",
    "pt-br": "pt",
    "portuguese": "pt",
}


def normalize_tada_language(value: object) -> Optional[str]:
    """Return the official aligner code, using ``None`` for English."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized not in _LANGUAGE_ALIASES:
        supported = ", ".join(TADA_LANGUAGE_OPTIONS)
        raise ValueError(f"Unsupported TADA language '{value}'. Supported languages: {supported}")
    return _LANGUAGE_ALIASES[normalized]


def format_tada_language_display(value: object) -> str:
    """Return the canonical UI label for a language value or backend code."""
    code = normalize_tada_language(value)
    for display, candidate in _LANGUAGE_DISPLAY_TO_CODE.items():
        if candidate == code:
            return display
    return "English"


def validate_tada_model_language(model_name: str, language: object) -> Optional[str]:
    """Validate checkpoint language support and return its backend aligner code."""
    code = normalize_tada_language(language)
    normalized_model = str(model_name or "").lower().replace("_", "-")
    if code is not None and ("tada-1b" in normalized_model or normalized_model == "1b"):
        raise ValueError(
            "TADA-1B is English-only. Select TADA-3B-ML for multilingual synthesis."
        )
    return code


__all__ = [
    "TADA_LANGUAGE_OPTIONS",
    "format_tada_language_display",
    "normalize_tada_language",
    "validate_tada_model_language",
]
