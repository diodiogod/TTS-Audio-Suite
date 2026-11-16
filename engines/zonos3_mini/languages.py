"""ZONOS3-Mini language codes: node display names, the official request codes and the suite's language codes."""

from typing import Dict

# display name -> official request code (zonos3/registries/languages.json, same order)
LANGUAGES: Dict[str, str] = {
    "English (US)": "en_us",
    "English (UK)": "en_gb",
    "French": "fr_fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt_br",
    "Japanese": "ja",
    "Mandarin": "cmn",
    "Korean": "ko",
    "Arabic": "ar",
    "Bengali": "bn",
    "Bulgarian": "bg",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "Finnish": "fi",
    "Georgian": "ka",
    "Greek": "el",
    "Gujarati": "gu",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Kannada": "kn",
    "Malay": "ms",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Norwegian": "no",
    "Odia": "or",
    "Polish": "pl",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
}
DEFAULT_LANGUAGE = "English (US)"

_CODES = frozenset(LANGUAGES.values())
_BY_DISPLAY = {name.lower(): code for name, code in LANGUAGES.items()}
_ALIASES = {
    "en": "en_us",
    "fr": "fr_fr",
    "pt": "pt_br",
    "pt_pt": "pt_br",
    "zh": "cmn",
    "zh_cn": "cmn",
    "zh_tw": "cmn",
    "nb": "no",
}
_TO_SUITE = {"en_us": "en", "en_gb": "en-gb", "fr_fr": "fr", "pt_br": "pt-br", "cmn": "zh"}


def to_zonos_language(value) -> str:
    """A node display name, official code or suite language code -> the official ZONOS3 request code."""
    raw = str(value or "").strip()
    if raw.lower() in _BY_DISPLAY:
        return _BY_DISPLAY[raw.lower()]
    code = raw.lower().replace("-", "_")
    code = _ALIASES.get(code, code)
    if code not in _CODES:
        raise ValueError(f"ZONOS3-Mini does not support language '{value}'")
    return code


def to_suite_language(code: str) -> str:
    """An official ZONOS3 code -> the language code the suite's character parser uses."""
    return _TO_SUITE.get(code, code)
