"""The language registry (``registries/languages.json``): every server's and the studio's language tables.

``SERVER`` are the TTS language codes requests use, ``SPOKEN`` the languages speech recognition, forced alignment and
other references understand (by Whisper-style code), ``ESPEAK`` the espeak-ng voices that guess unknown words of a
pronunciation lexicon language. The ggml server embeds the same file at build time; the studio and the in-browser
server import it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent / "registries" / "languages.json"


@dataclass(frozen=True)
class ServerLanguage:
    code: str  # the request code (en_us, cmn, ...)
    label: str  # the studio's name
    nemo: str  # NeMo text-normalization package
    lexicon: str  # pronunciation lexicon language (ISO 639-3)
    speech: str  # its SPOKEN code


@dataclass(frozen=True)
class SpokenLanguage:
    code: str  # Whisper-style code (en, zh, yue, ...)
    name: str  # Qwen3-ASR language name
    tts: str | None = None  # the server language that speaks it
    unspaced: bool = False  # written without spaces between words


_data = json.loads(REGISTRY.read_text(encoding="utf-8"))
SERVER: tuple[ServerLanguage, ...] = tuple(ServerLanguage(**row) for row in _data["server"])
SPOKEN: tuple[SpokenLanguage, ...] = tuple(SpokenLanguage(**row) for row in _data["spoken"])
ESPEAK: dict[str, str] = dict(_data["espeak"])
del _data

CODES: tuple[str, ...] = tuple(lang.code for lang in SERVER)
BY_CODE: dict[str, ServerLanguage] = {lang.code: lang for lang in SERVER}
SPOKEN_BY_CODE: dict[str, SpokenLanguage] = {lang.code: lang for lang in SPOKEN}
# Spoken codes of the languages written without spaces between words (ja, zh, yue).
UNSPACED: frozenset[str] = frozenset(lang.code for lang in SPOKEN if lang.unspaced)


def normalize(language: str) -> str:
    """A request's ``language`` as a server code (case-insensitive, ``-`` for ``_``); ValueError for another one."""
    normalized = str(language or "").strip().lower().replace("-", "_")
    if normalized not in BY_CODE:
        raise ValueError(f"Unsupported language code: {language!r}. Supported: {', '.join(CODES)}.")
    return normalized
