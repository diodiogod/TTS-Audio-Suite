"""TTS text normalization: zyphra_textnorm (Rust ports of the NeMo grammars, github.com/Zyphra/zyphra-textnorm).

Converts written text to its spoken form ("$5.32" -> "five dollars thirty two cents") before tokenization, in the
grammar of the request language's ``nemo`` package (registries/languages.json), with pronunciation groups and the
start / pause markers kept as typed. Disable globally with ZONOS3_TTS_NORM=0, or
per request via the text_normalization flag.
"""

from __future__ import annotations

import os
import re

import zyphra_textnorm

from zonos3 import languages

collapse_line_breaks = zyphra_textnorm.collapse_line_breaks

# The start / pause markers (U+E002..U+E00A) go through the grammar wrapped as one-character pronunciation groups
# (U+E000 ... U+E001), which the normalizer keeps verbatim with the whitespace around them.
_TEXT_MARKER = re.compile("[\ue002-\ue00a]")
_WRAPPED_MARKER = re.compile("\ue000([\ue002-\ue00a])\ue001")


def normalization_enabled() -> bool:
    return os.environ.get("ZONOS3_TTS_NORM", "1") != "0"


class TTSTextNormalizer:
    """The Rust normalizer for server language codes, memoized inside. ``normalize``
    never raises: unsupported languages come back with line breaks collapsed."""

    def __init__(self) -> None:
        self._tn = zyphra_textnorm.Normalizer()

    def normalize(self, text: str, language: str) -> str:
        lang = languages.BY_CODE.get(language)
        if lang is None:
            return collapse_line_breaks(text)
        out = self._tn.normalize(_TEXT_MARKER.sub(lambda m: "\ue000" + m.group() + "\ue001", text), lang.nemo)
        return _WRAPPED_MARKER.sub(r"\1", out)


def load_normalizer() -> TTSTextNormalizer | None:
    """The server's normalizer (None with ZONOS3_TTS_NORM=0)."""
    return TTSTextNormalizer() if normalization_enabled() else None


def normalize_request_text(text: str, language: str, normalizer) -> str:
    """A request's text as the prompt builder gets it: line breaks collapsed (also when normalization is off), then
    ``normalizer.normalize`` (anything with that method; None skips it)."""
    text = collapse_line_breaks(text)
    return text if normalizer is None else normalizer.normalize(text, language)
