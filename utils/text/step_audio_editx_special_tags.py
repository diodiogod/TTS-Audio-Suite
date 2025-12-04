"""
Step Audio EditX Special Tags Handler

Converts user-friendly SSML-style tags <effect> to Step Audio EditX's expected [effect] format.
This prevents conflicts with the character switching system [CharacterName].

Paralinguistic tags are used for inserting non-verbal sounds into audio during editing.
They are processed during the edit phase, NOT during initial TTS generation.

Usage:
    User writes: "Hello <Laughter> nice to meet you"
    Handler converts to: "Hello [Laughter] nice to meet you"
    Step Audio EditX processes: [Laughter] as paralinguistic insertion point
"""

import re
from typing import Set, Tuple, Optional

# Step Audio EditX paralinguistic tokens
# Users should write these in <angle> brackets, we convert to [square] brackets
STEP_AUDIO_EDITX_PARALINGUISTIC_TOKENS: Set[str] = {
    # Breathing and vocalizations
    'breathing',
    'laughter',
    'sigh',
    'uhm',

    # Surprise expressions
    'surprise-oh',
    'surprise-ah',
    'surprise-wa',

    # Other expressions
    'confirmation-en',
    'question-ei',
    'dissatisfaction-hnn',
}

# Canonical format mapping (lowercase -> proper case for engine)
PARALINGUISTIC_CANONICAL_FORMAT = {
    'breathing': 'Breathing',
    'laughter': 'Laughter',
    'sigh': 'Sigh',
    'uhm': 'Uhm',
    'surprise-oh': 'Surprise-oh',
    'surprise-ah': 'Surprise-ah',
    'surprise-wa': 'Surprise-wa',
    'confirmation-en': 'Confirmation-en',
    'question-ei': 'Question-ei',
    'dissatisfaction-hnn': 'Dissatisfaction-hnn',
}


def convert_step_audio_editx_tags(text: str) -> str:
    """
    Convert SSML-style <effect> tags to Step Audio EditX [Effect] format.

    This allows users to use <Laughter>, <Sigh>, etc. without conflicting with
    the character switching system that uses [CharacterName].

    Args:
        text: Input text with <effect> tags

    Returns:
        Text with <effect> converted to [Effect] for known paralinguistic tokens

    Examples:
        >>> convert_step_audio_editx_tags("Hello <Laughter> world")
        "Hello [Laughter] world"

        >>> convert_step_audio_editx_tags("Text with <unknown> tag")
        "Text with <unknown> tag"  # Unknown tags left as-is
    """
    def replace_tag(match):
        tag = match.group(1).lower()
        # Only convert known paralinguistic tokens
        if tag in STEP_AUDIO_EDITX_PARALINGUISTIC_TOKENS:
            # Use canonical format (proper case)
            canonical = PARALINGUISTIC_CANONICAL_FORMAT.get(tag, tag.title())
            return f'[{canonical}]'
        else:
            # Leave unknown tags as-is
            return match.group(0)

    # Pattern matches <tag> where tag is alphanumeric + hyphen + underscore
    pattern = r'<([a-zA-Z_-]+)>'
    return re.sub(pattern, replace_tag, text)


def has_step_audio_editx_tags(text: str) -> bool:
    """Check if text contains any Step Audio EditX paralinguistic tags in <> format."""
    pattern = r'<([a-zA-Z_-]+)>'
    matches = re.findall(pattern, text)
    return any(match.lower() in STEP_AUDIO_EDITX_PARALINGUISTIC_TOKENS for match in matches)


def extract_paralinguistic_tags(text: str) -> Tuple[str, list]:
    """
    Extract paralinguistic tags from text.

    Returns:
        Tuple of (clean_text_without_tags, list_of_tags_found)

    Examples:
        >>> extract_paralinguistic_tags("Hello <Laughter> world <Sigh>")
        ("Hello  world ", ["Laughter", "Sigh"])
    """
    tags_found = []

    def extract_and_remove(match):
        tag = match.group(1).lower()
        if tag in STEP_AUDIO_EDITX_PARALINGUISTIC_TOKENS:
            canonical = PARALINGUISTIC_CANONICAL_FORMAT.get(tag, tag.title())
            tags_found.append(canonical)
            return ''  # Remove tag from text
        return match.group(0)  # Keep unknown tags

    pattern = r'<([a-zA-Z_-]+)>'
    clean_text = re.sub(pattern, extract_and_remove, text)

    return clean_text, tags_found


def strip_paralinguistic_tags(text: str) -> str:
    """
    Strip all paralinguistic tags from text, returning clean text.

    This is useful for deriving audio_text (transcript) from text that
    contains paralinguistic insertion markers.

    Args:
        text: Text possibly containing <Laughter>, <Sigh>, etc.

    Returns:
        Clean text with all paralinguistic tags removed

    Examples:
        >>> strip_paralinguistic_tags("Hello <Laughter> how are you?")
        "Hello  how are you?"
    """
    clean_text, _ = extract_paralinguistic_tags(text)
    # Clean up double spaces left by tag removal
    clean_text = re.sub(r'  +', ' ', clean_text)
    return clean_text.strip()


def get_supported_paralinguistic_tags() -> Set[str]:
    """Get set of supported paralinguistic tags (in canonical format)."""
    return set(PARALINGUISTIC_CANONICAL_FORMAT.values())


def get_paralinguistic_options_for_ui() -> list:
    """Get list of paralinguistic options for ComfyUI dropdown."""
    return sorted(PARALINGUISTIC_CANONICAL_FORMAT.values())
