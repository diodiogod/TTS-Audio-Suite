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

Extended Inline Edit Tag Syntax (for post-processing any TTS engine):
    <Laughter>           - paralinguistic, 1 iteration
    <Laughter:3>         - paralinguistic, 3 iterations
    <emotion:happy>      - emotion edit, 1 iteration
    <emotion:happy:2>    - emotion edit, 2 iterations
    <style:whisper:3>    - style edit, 3 iterations
    <speed:faster>       - speed edit, 1 iteration

Multiple tags can be combined:
    <Laughter:2|style:whisper:1>  - pipe-separated in single tag
    <Laughter:2><style:whisper>   - separate tags (equivalent)
"""

import re
from typing import Set, Tuple, Optional, List
from dataclasses import dataclass, field

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


# =============================================================================
# EXTENDED INLINE EDIT TAG SYSTEM
# =============================================================================
# Supports <Laughter:2>, <emotion:happy:1>, <style:whisper:3>, <speed:faster>
# for post-processing TTS output with Step Audio EditX

@dataclass
class EditTag:
    """Represents a parsed edit tag with type, value, and iteration count."""
    edit_type: str  # "paralinguistic", "emotion", "style", "speed", "denoise", "vad"
    value: str      # The specific effect (e.g., "Laughter", "happy", "whisper", "faster")
    iterations: int = 1  # Number of edit iterations (1-5)
    position: Optional[int] = None  # Character position in text (for paralinguistic only)

    def __repr__(self):
        pos_str = f", pos={self.position}" if self.position is not None else ""
        return f"EditTag({self.edit_type}:{self.value}:{self.iterations}{pos_str})"


# Valid edit type values from Step Audio EditX edit_config.py
VALID_EMOTIONS = {
    'happy', 'angry', 'sad', 'humour', 'confusion', 'disgusted',
    'empathy', 'embarrass', 'fear', 'surprised', 'excited',
    'depressed', 'coldness', 'admiration', 'remove'
}

VALID_STYLES = {
    'serious', 'arrogant', 'child', 'older', 'girl', 'pure',
    'sister', 'sweet', 'ethereal', 'whisper', 'gentle', 'recite',
    'generous', 'act_coy', 'warm', 'shy', 'comfort', 'authority',
    'chat', 'radio', 'soulful', 'story', 'vivid', 'program',
    'news', 'advertising', 'roar', 'murmur', 'shout', 'deeply', 'loudly',
    'remove', 'exaggerated'
}

VALID_SPEEDS = {'faster', 'slower', 'more_faster', 'more_slower', 'more faster', 'more slower'}

# For quick lookup
EDIT_TYPE_VALUES = {
    'emotion': VALID_EMOTIONS,
    'style': VALID_STYLES,
    'speed': VALID_SPEEDS,
    'paralinguistic': set(PARALINGUISTIC_CANONICAL_FORMAT.values()),
    'denoise': {'denoise'},
    'vad': {'vad'},
    'restore': {'restore'}  # Voice restoration via ChatterBox VC
}


def _parse_single_tag_part(part: str, current_position: int) -> Optional[EditTag]:
    """
    Parse a single tag part (e.g., "Laughter:2", "emotion:happy:1", "style:whisper").

    Args:
        part: Single tag part without angle brackets
        current_position: Current character position in original text

    Returns:
        EditTag if valid, None otherwise
    """
    part = part.strip()
    if not part:
        return None

    # Split by colon
    components = part.split(':')

    # Case 1: Paralinguistic tag - <Laughter> or <Laughter:2>
    first_lower = components[0].lower()
    if first_lower in STEP_AUDIO_EDITX_PARALINGUISTIC_TOKENS:
        canonical = PARALINGUISTIC_CANONICAL_FORMAT.get(first_lower, components[0].title())
        iterations = 1
        if len(components) >= 2:
            try:
                iterations = max(1, min(5, int(components[1])))
            except ValueError:
                pass
        return EditTag(
            edit_type="paralinguistic",
            value=canonical,
            iterations=iterations,
            position=current_position
        )

    # Case 1.5: Restore tag - <restore> or <restore:2>
    if first_lower == 'restore':
        iterations = 1
        if len(components) >= 2:
            try:
                iterations = max(1, min(5, int(components[1])))
            except ValueError:
                pass
        return EditTag(
            edit_type="restore",
            value="restore",
            iterations=iterations,
            position=None  # Restore doesn't need position
        )

    # Case 2: Typed tag - <emotion:happy:2>, <style:whisper>, <speed:faster:1>
    if len(components) >= 2:
        type_name = components[0].lower()
        value = components[1].lower()

        # Validate type
        if type_name not in EDIT_TYPE_VALUES:
            return None

        # Validate value
        valid_values = EDIT_TYPE_VALUES[type_name]
        if value not in valid_values:
            # Try with underscores replaced by spaces for speed
            if type_name == 'speed':
                value_with_spaces = value.replace('_', ' ')
                if value_with_spaces not in valid_values:
                    return None
                value = value_with_spaces
            else:
                return None

        # Parse iterations
        iterations = 1
        if len(components) >= 3:
            try:
                iterations = max(1, min(5, int(components[2])))
            except ValueError:
                pass

        # Position only matters for paralinguistic
        position = current_position if type_name == "paralinguistic" else None

        return EditTag(
            edit_type=type_name,
            value=value,
            iterations=iterations,
            position=position
        )

    return None


def parse_edit_tags_with_iterations(text: str) -> Tuple[str, List[EditTag]]:
    """
    Parse all edit tags from text, return clean text and tag list.

    Supports:
    - <Laughter> → paralinguistic, 1 iter
    - <Laughter:3> → paralinguistic, 3 iter
    - <style:whisper:2> → style, 2 iter
    - <emotion:happy> → emotion, 1 iter
    - <Laughter:2|style:whisper:1> → multiple in one tag (pipe-separated)
    - <speed:faster>, <speed:more_faster>
    - <denoise>, <vad>

    NOTE: Paralinguistic tags REQUIRE space before them. This function auto-inserts
    space if tag follows a word character (e.g., "word<Laughter>" → "word [Laughter]")

    Args:
        text: Input text with potential edit tags

    Returns:
        Tuple of (clean_text, list_of_EditTag)
    """
    edit_tags: List[EditTag] = []

    # Pattern: <content> where content can include pipes, colons, alphanumeric, underscore, hyphen, space
    # More permissive to capture all potential tags
    pattern = r'<([a-zA-Z][a-zA-Z0-9_\-\s:|\d]*)>'

    # Track position offset as we remove tags
    offset = 0
    clean_parts = []
    last_end = 0

    for match in re.finditer(pattern, text):
        tag_content = match.group(1)
        tag_start = match.start()
        tag_end = match.end()

        # Calculate position in clean text (where tag would be inserted)
        clean_position = tag_start - offset

        # Add text before this tag
        text_before = text[last_end:tag_start]

        # Check if we need to add space before paralinguistic tag
        # (when tag follows word character without space)
        needs_space = False
        if text_before and text_before[-1].isalnum():
            # Check if this is a paralinguistic tag
            parts = tag_content.split('|')
            for part in parts:
                first_word = part.split(':')[0].lower()
                if first_word in STEP_AUDIO_EDITX_PARALINGUISTIC_TOKENS:
                    needs_space = True
                    break

        if needs_space:
            text_before += ' '
            offset -= 1  # Compensate for added space

        clean_parts.append(text_before)

        # Parse tag content (may have pipes for multiple effects)
        parts = tag_content.split('|')
        tag_found = False
        tags_from_this_match = []

        for part in parts:
            edit_tag = _parse_single_tag_part(part, clean_position)
            if edit_tag:
                edit_tags.append(edit_tag)
                tags_from_this_match.append(edit_tag)
                tag_found = True

        # If any valid tag was found, update offset (tag is removed)
        if tag_found:
            offset += (tag_end - tag_start)

            # Check if we need to add space AFTER paralinguistic tag
            # (when tag is followed by word character without space)
            text_after = text[tag_end:] if tag_end < len(text) else ""
            if text_after and text_after[0].isalnum():
                # Check if any of the tags we JUST PARSED FROM THIS MATCH was paralinguistic
                for tag in tags_from_this_match:
                    if tag.edit_type == 'paralinguistic':
                        clean_parts.append(' ')
                        offset -= 1  # Compensate for added space
                        break
        else:
            # Unknown tag - keep it in text
            clean_parts.append(match.group(0))

        last_end = tag_end

    # Add remaining text
    clean_parts.append(text[last_end:])

    # Join clean parts
    clean_text_before_collapse = ''.join(clean_parts)

    # Collapse multiple spaces to single space and track position shifts
    # We need to adjust tag positions after collapsing spaces
    clean_text = clean_text_before_collapse
    position_adjustments = {}  # Maps original position to adjustment amount

    # Find all multi-space sequences and calculate adjustments
    adjustment = 0
    i = 0
    while i < len(clean_text):
        if clean_text[i:i+2] == '  ':  # Found double space
            # Count consecutive spaces
            space_count = 0
            while i + space_count < len(clean_text) and clean_text[i + space_count] == ' ':
                space_count += 1

            # Collapse to single space means we remove (space_count - 1) characters
            spaces_removed = space_count - 1

            # All positions after this point shift left by spaces_removed
            adjustment += spaces_removed

            # Track cumulative adjustment for positions after this multi-space
            for tag in edit_tags:
                if tag.position is not None and tag.position > i:
                    if tag.position not in position_adjustments:
                        position_adjustments[tag.position] = 0
                    position_adjustments[tag.position] += spaces_removed

            i += space_count
        else:
            i += 1

    # Apply space collapse
    clean_text = re.sub(r'  +', ' ', clean_text_before_collapse)

    # Apply position adjustments from space collapsing
    for tag in edit_tags:
        if tag.position is not None and tag.position in position_adjustments:
            tag.position = max(0, tag.position - position_adjustments[tag.position])

    # Adjust positions if we strip leading whitespace
    leading_spaces = len(clean_text) - len(clean_text.lstrip())
    if leading_spaces > 0:
        # Adjust all positions by subtracting leading spaces removed
        for tag in edit_tags:
            if tag.position is not None:
                tag.position = max(0, tag.position - leading_spaces)

    clean_text = clean_text.strip()

    return clean_text, edit_tags


def has_edit_tags(text: str) -> bool:
    """
    Quick check if text contains any valid edit tags.

    Args:
        text: Text to check

    Returns:
        True if text contains valid edit tags
    """
    _, tags = parse_edit_tags_with_iterations(text)
    return len(tags) > 0


def get_edit_tags_for_segment(text: str) -> Tuple[str, List[EditTag]]:
    """
    Convenience function for segment processing.
    Extracts edit tags and returns clean text for TTS.

    This is the main entry point for segment_parameters.py integration.

    Args:
        text: Segment text potentially containing edit tags

    Returns:
        Tuple of (clean_text_for_tts, edit_tags_list)
    """
    return parse_edit_tags_with_iterations(text)


def sort_edit_tags_for_processing(tags: List[EditTag]) -> List[EditTag]:
    """
    Sort edit tags in optimal processing order.

    Processing order:
    1. Non-paralinguistic edits first (emotion → style → speed → denoise/vad)
    2. Paralinguistic edits last (position matters for sound insertion)

    Args:
        tags: List of EditTag objects

    Returns:
        Sorted list of EditTag objects
    """
    # Define priority order (lower = processed first)
    type_priority = {
        'emotion': 1,
        'style': 2,
        'speed': 3,
        'denoise': 4,
        'vad': 5,
        'paralinguistic': 10,  # Position-sensitive sound insertion
        'restore': 20  # LAST - voice restoration after all edits
    }

    return sorted(tags, key=lambda t: type_priority.get(t.edit_type, 99))
