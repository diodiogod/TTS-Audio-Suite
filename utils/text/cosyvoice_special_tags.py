"""
CosyVoice3 Special Tags Handler

Converts user-friendly SSML-style tags <tag> to CosyVoice3's expected [tag] format.
This prevents conflicts with the character switching system [CharacterName].

Usage:
    User writes: "Hello <breath> nice to meet you <laughter>"
    Handler converts to: "Hello [breath] nice to meet you [laughter]"
    CosyVoice3 processes: [breath] and [laughter] as paralinguistic tokens
"""

import re
from typing import Set

# CosyVoice3 paralinguistic tags (single tags, not wrappers)
# Users should write these in <angle> brackets, we convert to [square] brackets
# Note: <strong>...</strong> and <laughter>...</laughter> wrappers are NOT converted
COSYVOICE_PARALINGUISTIC_TAGS: Set[str] = {
    # Breathing sounds
    'breath', 'quick_breath',

    # Vocal expressions
    'laughter', 'cough', 'sigh', 'gasp',

    # Background sounds
    'noise', 'hissing', 'vocalized-noise',

    # Speech artifacts
    'lipsmack', 'mn', 'clucking', 'accent'
}


def convert_cosyvoice_special_tags(text: str) -> str:
    """
    Convert SSML-style <tag> to CosyVoice3 [tag] format for single tags.

    This allows users to use <breath>, <laughter>, etc. without conflicting with
    the character switching system that uses [CharacterName].

    Wrapper tags like <strong>...</strong> are left unchanged as CosyVoice supports them.

    Args:
        text: Input text with <tag> tags

    Returns:
        Text with <tag> converted to [tag] for known CosyVoice tags

    Examples:
        >>> convert_cosyvoice_special_tags("Hello <breath> world")
        "Hello [breath] world"

        >>> convert_cosyvoice_special_tags("This is <strong>important</strong>")
        "This is <strong>important</strong>"  # Wrapper tags left as-is

        >>> convert_cosyvoice_special_tags("Text with <unknown> tag")
        "Text with <unknown> tag"  # Unknown tags left as-is
    """
    def replace_tag(match):
        tag = match.group(1).lower()
        # Only convert known CosyVoice single tags
        if tag in COSYVOICE_PARALINGUISTIC_TAGS:
            return f'[{match.group(1)}]'  # Preserve original case
        else:
            # Leave unknown tags as-is (might be wrappers like <strong>, HTML, or other markup)
            return match.group(0)

    # Pattern matches <tag> where tag is alphanumeric + underscore/hyphen
    # This won't match </tag> closing tags or <tag>...</tag> wrapper structures
    pattern = r'<([a-zA-Z_-]+)(?!\s*>)>'
    return re.sub(pattern, replace_tag, text)


def has_cosyvoice_special_tags(text: str) -> bool:
    """Check if text contains any CosyVoice special tags in <> format."""
    pattern = r'<([a-zA-Z_-]+)>'
    matches = re.findall(pattern, text)
    return any(match.lower() in COSYVOICE_PARALINGUISTIC_TAGS for match in matches)


def get_supported_cosyvoice_tags() -> Set[str]:
    """Get set of supported CosyVoice special tags."""
    return COSYVOICE_PARALINGUISTIC_TAGS.copy()
