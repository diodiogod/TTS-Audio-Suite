"""Turn ZONOS3 emphasis markup into spans over the text sent to inference."""

import re
from bisect import bisect_left

_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
_MARK = re.compile(r"(?<!\{)\{([^{}]+):[ \t]*(" + _NUMBER + r")[ \t]*\}(?!\})")
_BOUNDARY = re.compile(r"\x00emphasis(?::(" + _NUMBER + r"))?\x00")
_MAX_STRENGTH = 6.0


class EmphasisParser:
    """Carry an open mark across fragments separated by Suite tags and pauses."""

    def __init__(self):
        self._strength = None

    @staticmethod
    def prepare(text: str) -> str:
        """Protect compact marks before the Suite splits text at speaker and pause tags."""
        def encode(match: re.Match) -> str:
            phrase, raw_strength = match.groups()
            strength = float(raw_strength)
            if not 0 <= strength <= _MAX_STRENGTH:
                raise ValueError("ZONOS3-Mini: word emphasis strength must be between 0 and 6.")
            if not phrase.strip():
                raise ValueError("ZONOS3-Mini: put words before the strength in {words:strength}.")
            if not strength:
                return phrase
            return f"\x00emphasis:{raw_strength}\x00{phrase}\x00emphasis\x00"

        return _MARK.sub(encode, text)

    def strip_tags(self, text: str) -> tuple[str, list[tuple[int, int, float]]]:
        parts = []
        spans = []
        previous = offset = start = 0
        for mark in _BOUNDARY.finditer(text):
            fragment = text[previous:mark.start()]
            parts.append(fragment)
            offset += len(fragment)
            previous = mark.end()
            raw_strength = mark.group(1)
            if raw_strength is None:
                if self._strength and start < offset:
                    spans.append((start, offset, self._strength))
                self._strength = None
            else:
                self._strength = float(raw_strength)
                start = offset
        remainder = text[previous:]
        parts.append(remainder)
        end = offset + len(remainder)
        if self._strength and start < end:
            spans.append((start, end, self._strength))
        return "".join(parts), spans

    def finish(self) -> None:
        if self._strength is not None:
            raise ValueError("ZONOS3-Mini: could not align a word emphasis mark. Keep {words:strength} in the spoken text.")


def project_emphasis_spans(text: str, chunks: list[str], spans: list[tuple[int, int, float]]) -> list[list[dict]]:
    """Clip spans to each chunk and account for whitespace removed by the chunker."""
    if not spans:
        return [[] for _ in chunks]

    source_positions = [match.start() for match in re.finditer(r"\S", text)]
    source = "".join(text[position] for position in source_positions)
    # The chunker preserves non-whitespace characters, including punctuation and Unicode.
    # Match them in order so repeated words and whitespace normalization cannot shift a span.
    ranges = [
        (bisect_left(source_positions, start), bisect_left(source_positions, end), strength)
        for start, end, strength in spans
    ]
    result = []
    consumed = 0
    for chunk in chunks:
        positions = [match.start() for match in re.finditer(r"\S", chunk)]
        chunk_end = consumed + len(positions)
        if "".join(chunk[position] for position in positions) != source[consumed:chunk_end]:
            raise ValueError("ZONOS3-Mini: text changed during chunking, so word emphasis could not be aligned.")
        chunk_spans = []
        for start, end, strength in ranges:
            start, end = max(start, consumed), min(end, chunk_end)
            if start < end:
                chunk_spans.append({
                    "start": positions[start - consumed],
                    "end": positions[end - consumed - 1] + 1,
                    # Zyphra Studio's word emphasis maps its strength to these two native directions.
                    "controls": {"emphasis_nv_nodur.emphasis": 4 * strength, "mood.arousal": strength},
                })
        result.append(chunk_spans)
        consumed = chunk_end
    if consumed != len(source):
        raise ValueError("ZONOS3-Mini: text changed during chunking, so word emphasis could not be aligned.")
    return result
