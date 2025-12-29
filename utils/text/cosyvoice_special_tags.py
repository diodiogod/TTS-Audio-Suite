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
    Convert user-friendly tags to CosyVoice3 format.

    Single tags: <breath>, <laughter> → [breath], [laughter]
    Wrapper tags: <laughing>text</laughing> → <laughter>text</laughter>

    This allows users to use <breath>, <laughter>, etc. without conflicting with
    the character switching system that uses [CharacterName].

    Args:
        text: Input text with <tag> tags

    Returns:
        Text with tags converted to CosyVoice3 format

    Examples:
        >>> convert_cosyvoice_special_tags("Hello <breath> world")
        "Hello [breath] world"

        >>> convert_cosyvoice_special_tags("This is <laughing>funny</laughing>")
        "This is <laughter>funny</laughter>"

        >>> convert_cosyvoice_special_tags("This is <strong>important</strong>")
        "This is <strong>important</strong>"  # Already correct format
    """
    # First, convert wrapper tags: <laughing>text</laughing> → <laughter>text</laughter>
    text = re.sub(r'<laughing>', '<laughter>', text)
    text = re.sub(r'</laughing>', '</laughter>', text)

    # Then convert single tags to [tag] format
    def replace_tag(match):
        tag = match.group(1).lower()
        # Only convert known CosyVoice single tags
        if tag in COSYVOICE_PARALINGUISTIC_TAGS:
            return f'[{match.group(1)}]'  # Preserve original case
        else:
            # Leave unknown tags as-is (might be wrappers like <strong>, HTML, or other markup)
            return match.group(0)

    # Pattern matches single <tag> - must NOT be part of a wrapper pair
    # Check that it's not followed immediately by text (which would make it a wrapper opening)
    # and not preceded by / (which would make it a closing tag)
    pattern = r'(?<!</)(?<!\w)<([a-zA-Z_-]+)>(?!\w)'
    return re.sub(pattern, replace_tag, text)


def has_cosyvoice_special_tags(text: str) -> bool:
    """Check if text contains any CosyVoice special tags in <> format."""
    pattern = r'<([a-zA-Z_-]+)>'
    matches = re.findall(pattern, text)
    return any(match.lower() in COSYVOICE_PARALINGUISTIC_TAGS for match in matches)


def get_supported_cosyvoice_tags() -> Set[str]:
    """Get set of supported CosyVoice special tags."""
    return COSYVOICE_PARALINGUISTIC_TAGS.copy()
