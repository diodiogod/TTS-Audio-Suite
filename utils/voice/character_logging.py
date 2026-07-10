"""Shared display helpers for resolved TTS character voices."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def resolved_voice_name(voice_ref: Any, fallback: str = "narrator") -> str:
    """Return the user-facing name of the voice selected for a character."""
    if isinstance(voice_ref, dict):
        for key in ("character_name", "voice_name", "name"):
            value = voice_ref.get(key)
            if value:
                return str(value)
        for key in ("audio_path", "prompt_audio_path", "path"):
            value = voice_ref.get(key)
            if value:
                return Path(str(value)).stem
        for key in ("audio", "waveform"):
            value = voice_ref.get(key)
            if isinstance(value, (dict, list, tuple, str, os.PathLike)):
                resolved = resolved_voice_name(value, fallback="")
                if resolved:
                    return resolved
    elif isinstance(voice_ref, (list, tuple)):
        for value in voice_ref:
            resolved = resolved_voice_name(value, fallback="")
            if resolved:
                return resolved
    elif isinstance(voice_ref, (str, os.PathLike)):
        return Path(str(voice_ref)).stem
    else:
        for key in ("character_name", "voice_name", "name", "audio_path", "path"):
            value = getattr(voice_ref, key, None)
            if value:
                return Path(str(value)).stem if key in {"audio_path", "path"} else str(value)
    return str(fallback or "narrator")


def resolved_character_label(character: str, voice_ref: Any) -> str:
    """Use the resolved voice name while retaining a character fallback."""
    return resolved_voice_name(voice_ref, fallback=character or "narrator")


def format_resolved_character_text(character: str, text: str, voice_ref: Any) -> str:
    """Format a log line showing which resolved voice will speak the text."""
    return f"[{resolved_character_label(character, voice_ref)}] {text}"


def format_resolved_character_block(
    character: str, text: str, voice_ref: Any, width: int = 60
) -> str:
    """Format the standard boxed prompt preview used in engine logs."""
    return "\n".join(
        ("=" * width, format_resolved_character_text(character, text, voice_ref), "=" * width)
    )


def format_character_override_warning(
    engine_label: str, speaker_label: str, character: str
) -> str:
    """Format the standard warning used when an input replaces a character alias."""
    return (
        f"⚠️ {engine_label} priority: {speaker_label} input overrides "
        f"['{character}'] alias"
    )
