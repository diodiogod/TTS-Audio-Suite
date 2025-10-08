"""
ChatterBox Official 23-Lang v2 Special Tags Handler

Converts user-friendly SSML-style tags <emotion> to ChatterBox v2's expected [emotion] format.
This prevents conflicts with the character switching system [CharacterName].

Usage:
    User writes: "Hello <giggle> nice to meet you"
    Handler converts to: "Hello [giggle] nice to meet you"
    ChatterBox v2 processes: [giggle] as special token
"""

import re
from typing import Set

# ChatterBox v2 special tokens (emotions, sounds, vocal effects)
# Users should write these in <angle> brackets, we convert to [square] brackets
CHATTERBOX_V2_SPECIAL_TOKENS: Set[str] = {
    # Emotional expressions
    'giggle', 'laughter', 'guffaw', 'sigh', 'cry', 'gasp', 'groan',

    # Breathing & speech modifiers
    'inhale', 'exhale', 'whisper', 'mumble', 'UH', 'UM',

    # Vocal performances
    'singing', 'music', 'humming', 'whistle',

    # Body sounds
    'cough', 'sneeze', 'sniff', 'snore', 'clear_throat', 'chew', 'sip', 'kiss',

    # Animal sounds
    'bark', 'howl', 'meow',

    # Other
    'shhh', 'gibberish'
}


def convert_v2_special_tags(text: str) -> str:
    """
    Convert SSML-style <emotion> tags to ChatterBox v2 [emotion] format.

    This allows users to use <giggle>, <sigh>, etc. without conflicting with
    the character switching system that uses [CharacterName].

    Args:
        text: Input text with <emotion> tags

    Returns:
        Text with <emotion> converted to [emotion] for known v2 tokens

    Examples:
        >>> convert_v2_special_tags("Hello <giggle> world")
        "Hello [giggle] world"

        >>> convert_v2_special_tags("Text with <unknown> tag")
        "Text with <unknown> tag"  # Unknown tags left as-is
    """
    def replace_tag(match):
        tag = match.group(1).lower()
        # Only convert known v2 tokens
        if tag in CHATTERBOX_V2_SPECIAL_TOKENS:
            return f'[{match.group(1)}]'  # Preserve original case
        else:
            # Leave unknown tags as-is (might be HTML or other markup)
            return match.group(0)

    # Pattern matches <tag> where tag is alphanumeric + underscore
    pattern = r'<([a-zA-Z_]+)>'
    return re.sub(pattern, replace_tag, text)


def has_v2_special_tags(text: str) -> bool:
    """Check if text contains any v2 special tags in <> format."""
    pattern = r'<([a-zA-Z_]+)>'
    matches = re.findall(pattern, text)
    return any(match.lower() in CHATTERBOX_V2_SPECIAL_TOKENS for match in matches)


def get_supported_v2_tags() -> Set[str]:
    """Get set of supported v2 special tags."""
    return CHATTERBOX_V2_SPECIAL_TOKENS.copy()
