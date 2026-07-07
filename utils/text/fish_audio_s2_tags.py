"""Fish Audio S2 Pro inline-tag translation at the engine boundary."""

import re

from utils.models.language_mapper import resolve_language_alias
from utils.text.character_parser.language_resolver import LanguageResolver


# Common examples published in the official model card. This is documentation,
# not an allowlist: S2 officially accepts free-form natural-language tags.
FISH_S2_OFFICIAL_TAGS = frozenset({
    "pause", "emphasis", "laughing", "inhale", "chuckle", "tsk", "singing",
    "excited", "laughing tone", "interrupting", "chuckling", "excited tone",
    "volume up", "echo", "angry", "low volume", "sigh", "low voice", "whisper",
    "screaming", "shouting", "loud", "surprised", "short pause", "exhale",
    "delight", "panting", "audience laughter", "with strong accent", "volume down",
    "clearing throat", "sad", "moaning", "shocked",
})

_ANGLE_TAG_RE = re.compile(r"<\s*([^<>]+?)\s*>")

_LANGUAGE_RESOLVER = LanguageResolver("en")


def translate_fish_s2_inline_tags(text: str) -> str:
    """Translate suite ``<instruction>`` syntax to Fish's native brackets."""

    def replace(match: re.Match) -> str:
        tag = " ".join(match.group(1).split())
        return f"[{tag}]"

    return _ANGLE_TAG_RE.sub(replace, text or "")


def get_fish_language_instruction(language: str | None, explicit: bool = False) -> str | None:
    """Map a resolved suite language code to a natural Fish instruction tag."""
    if not language:
        return None
    canonical = resolve_language_alias(str(language))
    if canonical == "en" and not explicit:
        return None
    return _LANGUAGE_RESOLVER.get_language_display_name(canonical)
