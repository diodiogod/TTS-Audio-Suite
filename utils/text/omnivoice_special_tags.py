"""
OmniVoice native tag helpers.

TTS Audio Suite accepts suite-default `<>` aliases for OmniVoice non-verbal tags
and converts them to the official square-bracket syntax expected by OmniVoice.
"""

from __future__ import annotations

import re
from typing import Set


OMNIVOICE_NON_VERBAL_TAGS: Set[str] = {
    "laughter",
    "sigh",
    "confirmation-en",
    "question-en",
    "question-ah",
    "question-oh",
    "question-ei",
    "question-yi",
    "surprise-ah",
    "surprise-oh",
    "surprise-wa",
    "surprise-yo",
    "dissatisfaction-hnn",
}


def convert_omnivoice_special_tags(text: str) -> str:
    """
    Convert suite-default OmniVoice aliases like `<laughter>` to `[laughter]`.

    Unknown angle tags are left untouched so other engine-native syntaxes or user
    content are not damaged.
    """

    def replace_tag(match):
        tag = match.group(1).lower()
        if tag in OMNIVOICE_NON_VERBAL_TAGS:
            return f"[{tag}]"
        return match.group(0)

    return re.sub(r"<([a-zA-Z_-]+)>", replace_tag, text or "")


def get_supported_omnivoice_tags() -> Set[str]:
    """Expose the supported OmniVoice non-verbal tag names."""
    return OMNIVOICE_NON_VERBAL_TAGS.copy()
