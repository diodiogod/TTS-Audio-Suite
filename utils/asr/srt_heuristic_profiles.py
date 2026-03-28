"""
Language-aware heuristic defaults for subtitle segmentation.
"""

from typing import Dict, Optional


DEFAULT_HEURISTIC_PROFILE_LABEL = "Auto"
HEURISTIC_PROFILE_OPTIONS = [
    "Auto",
    "English",
    "Portuguese (Brazil)",
    "Custom",
]

ENGLISH_DANGLING_TAIL_ALLOWLIST = "a,an,the,to,of,and,or,im,i'm,you,you're,we,they,he,she,it"
ENGLISH_INCOMPLETE_KEYWORDS = "what,why,how,where,who,which,when"

PORTUGUESE_BR_DANGLING_TAIL_ALLOWLIST = (
    "o,a,os,as,um,uma,uns,umas,de,do,da,dos,das,e,ou,mas,se,que,como,quando,"
    "onde,quem,para,pra,por,com,sem,em,no,na,nos,nas,ao,aos,pelo,pela,pelos,pelas"
)
PORTUGUESE_BR_INCOMPLETE_KEYWORDS = (
    "o que,por que,porque,como,onde,quem,qual,quais,quando"
)

_PROFILE_DEFAULTS: Dict[str, Dict[str, str]] = {
    "english": {
        "merge_dangling_tail_allowlist": ENGLISH_DANGLING_TAIL_ALLOWLIST,
        "merge_incomplete_keywords": ENGLISH_INCOMPLETE_KEYWORDS,
    },
    "pt-br": {
        "merge_dangling_tail_allowlist": PORTUGUESE_BR_DANGLING_TAIL_ALLOWLIST,
        "merge_incomplete_keywords": PORTUGUESE_BR_INCOMPLETE_KEYWORDS,
    },
}

_PROFILE_LABEL_TO_KEY = {
    "auto": "auto",
    "english": "english",
    "portuguese (brazil)": "pt-br",
    "custom": "custom",
}

_LANGUAGE_TO_PROFILE_KEY = {
    "en": "english",
    "en-us": "english",
    "en-gb": "english",
    "english": "english",
    "pt": "pt-br",
    "pt-br": "pt-br",
    "pt_br": "pt-br",
    "portuguese": "pt-br",
    "portuguese (brazil)": "pt-br",
    "brazilian portuguese": "pt-br",
}


def normalize_profile_selection(selection: Optional[str]) -> str:
    if not selection:
        return "auto"
    return _PROFILE_LABEL_TO_KEY.get(str(selection).strip().lower(), "auto")


def resolve_profile_from_language(language: Optional[str]) -> str:
    if not language:
        return "english"

    normalized = str(language).strip().lower().replace("_", "-")
    if normalized in _LANGUAGE_TO_PROFILE_KEY:
        return _LANGUAGE_TO_PROFILE_KEY[normalized]

    if normalized.startswith("pt"):
        return "pt-br"
    if normalized.startswith("en"):
        return "english"
    return "english"


def get_profile_defaults(profile_key: str) -> Dict[str, str]:
    return dict(_PROFILE_DEFAULTS.get(profile_key, _PROFILE_DEFAULTS["english"]))


def resolve_profile_defaults(selection: Optional[str], language: Optional[str] = None) -> Dict[str, str]:
    normalized = normalize_profile_selection(selection)
    if normalized == "custom":
        return {}
    if normalized == "auto":
        normalized = resolve_profile_from_language(language)
    return get_profile_defaults(normalized)
