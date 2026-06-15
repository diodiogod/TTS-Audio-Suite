"""
Language helpers for Dots TTS.
"""

from typing import Optional


DOTS_LANGUAGE_CODE_BY_NAME = {
    "arabic": "AR",
    "czech": "CS",
    "german": "DE",
    "greek": "EL",
    "english": "EN",
    "spanish": "ES",
    "finnish": "FI",
    "french": "FR",
    "hindi": "HI",
    "indonesian": "ID",
    "italian": "IT",
    "japanese": "JA",
    "korean": "KO",
    "dutch": "NL",
    "polish": "PL",
    "portuguese": "PT",
    "pt": "PT",
    "pt-br": "PT",
    "pt-pt": "PT",
    "po": "PT",
    "romanian": "RO",
    "russian": "RU",
    "thai": "TH",
    "turkish": "TR",
    "ukrainian": "UK",
    "uk": "UK",
    "vietnamese": "VI",
    "cantonese": "YUE",
    "yue": "YUE",
    "chinese": "ZH",
    "mandarin": "ZH",
    "zh": "ZH",
    "zh-cn": "ZH",
    "zh-tw": "ZH",
}

DOTS_LANGUAGE_DISPLAY_BY_CODE = {
    "AR": "Arabic",
    "CS": "Czech",
    "DE": "German",
    "EL": "Greek",
    "EN": "English",
    "ES": "Spanish",
    "FI": "Finnish",
    "FR": "French",
    "HI": "Hindi",
    "ID": "Indonesian",
    "IT": "Italian",
    "JA": "Japanese",
    "KO": "Korean",
    "NL": "Dutch",
    "PL": "Polish",
    "PT": "Portuguese",
    "RO": "Romanian",
    "RU": "Russian",
    "TH": "Thai",
    "TR": "Turkish",
    "UK": "Ukrainian",
    "VI": "Vietnamese",
    "YUE": "Cantonese",
    "ZH": "Chinese",
}

DOTS_LANGUAGE_OPTIONS = [
    "Auto",
    "None",
    "Arabic",
    "Czech",
    "German",
    "Greek",
    "English",
    "Spanish",
    "Finnish",
    "French",
    "Hindi",
    "Indonesian",
    "Italian",
    "Japanese",
    "Korean",
    "Dutch",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Thai",
    "Turkish",
    "Ukrainian",
    "Vietnamese",
    "Cantonese",
    "Chinese",
]


def normalize_dots_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None

    normalized = str(language).strip()
    if not normalized:
        return None

    lowered = normalized.lower()
    if lowered in {"auto", "auto detect", "auto_detect"}:
        return "auto_detect"
    if lowered in {"none", "off", "no tag", "no language tag"}:
        return None

    mapped = DOTS_LANGUAGE_CODE_BY_NAME.get(lowered)
    if mapped:
        return mapped

    upper_value = normalized.upper()
    if upper_value in DOTS_LANGUAGE_DISPLAY_BY_CODE:
        return upper_value

    if lowered == "yue":
        return "YUE"

    return upper_value


def format_dots_language_display(language: Optional[str]) -> str:
    normalized = normalize_dots_language(language)
    if normalized == "auto_detect":
        return "Auto"
    if normalized is None:
        return "None"

    return DOTS_LANGUAGE_DISPLAY_BY_CODE.get(normalized, str(language).strip() or "Auto")
