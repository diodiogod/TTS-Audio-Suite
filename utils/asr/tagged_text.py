"""
Tagged text helpers for subtitle generation.

Control tags are preserved for rendering, but they do not count as spoken text
for subtitle heuristics like chars-per-line, cps, and word counts.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import re

from utils.asr.types import ASRSegment, ASRWord


PAUSE_TAG_PATTERN = re.compile(
    r"^\[(?:pause|wait|stop|Pause|Wait|Stop|PAUSE|WAIT|STOP):(\d+(?:\.\d+)?)(s|ms)?\]$"
)
# Event-like bracket tags that should be preserved but never re-emitted as
# persistent speaker state when a line/cue wraps.
KNOWN_EVENT_SQUARE_TAGS = {
    "giggle", "laughter", "guffaw", "sigh", "cry", "gasp", "groan",
    "inhale", "exhale", "whisper", "mumble", "uh", "um",
    "singing", "music", "humming", "whistle",
    "cough", "sneeze", "sniff", "snore", "clear_throat", "chew", "sip", "kiss",
    "bark", "howl", "meow", "shhh", "gibberish",
    "breathing", "surprise-oh", "surprise-ah", "surprise-wa",
    "confirmation-en", "question-ei", "dissatisfaction-hnn",
}


@dataclass
class ControlTag:
    text: str
    kind: str  # "state", "pause", "event"
    duration: float = 0.0


@dataclass
class TaggedTextProfile:
    original_text: str
    spoken_text: str
    word_count: int
    controls_by_anchor: Dict[int, List[ControlTag]]
    active_state_by_word: List[Optional[str]]
    pause_durations_by_anchor: Dict[int, float]

    @property
    def has_control_tags(self) -> bool:
        return bool(self.controls_by_anchor)

    def controls_for_anchor(self, anchor: int) -> List[ControlTag]:
        return list(self.controls_by_anchor.get(anchor, []))

    def active_state_for_word(self, word_index: int) -> Optional[str]:
        if 0 <= word_index < len(self.active_state_by_word):
            return self.active_state_by_word[word_index]
        return self.active_state_by_word[-1] if self.active_state_by_word else None

    @property
    def has_pause_tags(self) -> bool:
        return bool(self.pause_durations_by_anchor)


def _count_words(text: str) -> int:
    if not text:
        return 0
    count = 0
    for token in text.split():
        if re.sub(r"[^\w]+", "", token, flags=re.UNICODE):
            count += 1
    return count


def _normalize_pause_duration(value: str, unit: Optional[str]) -> float:
    duration = float(value)
    if unit == "ms":
        return duration / 1000.0
    return duration


def _classify_square_tag(tag_text: str) -> Optional[ControlTag]:
    pause_match = PAUSE_TAG_PATTERN.match(tag_text)
    if pause_match:
        return ControlTag(
            text=tag_text,
            kind="pause",
            duration=_normalize_pause_duration(pause_match.group(1), pause_match.group(2)),
        )

    inner = tag_text[1:-1].strip()
    if not inner:
        return None

    # Treat obvious prose-like bracket content as normal text instead of a
    # control tag. This keeps the special path focused on project syntax.
    if any(ch in inner for ch in ".!?"):
        return None

    if inner.lower() in KNOWN_EVENT_SQUARE_TAGS:
        return ControlTag(text=tag_text, kind="event")

    return ControlTag(text=tag_text, kind="state")


def _classify_angle_tag(tag_text: str) -> Optional[ControlTag]:
    inner = tag_text[1:-1].strip()
    if not inner:
        return None
    return ControlTag(text=tag_text, kind="event")


def _extract_control_tag(text: str, start_idx: int):
    opener = text[start_idx]
    closer = "]" if opener == "[" else ">"
    end_idx = text.find(closer, start_idx + 1)
    if end_idx < 0:
        return None

    tag_text = text[start_idx:end_idx + 1]
    if opener == "[":
        control = _classify_square_tag(tag_text)
    else:
        control = _classify_angle_tag(tag_text)

    if control is None:
        return None
    return control, end_idx + 1


def parse_tagged_text(text: str) -> TaggedTextProfile:
    if not text or ("[" not in text and "<" not in text):
        return TaggedTextProfile(
            original_text=text,
            spoken_text=text,
            word_count=_count_words(text),
            controls_by_anchor={},
            active_state_by_word=[None] * _count_words(text),
            pause_durations_by_anchor={},
        )

    controls_by_anchor: Dict[int, List[ControlTag]] = {}
    pause_durations_by_anchor: Dict[int, float] = {}
    spoken_parts: List[str] = []

    idx = 0
    chunk_start = 0
    word_count = 0
    found_controls = False

    while idx < len(text):
        char = text[idx]
        if char not in "[<":
            idx += 1
            continue

        extracted = _extract_control_tag(text, idx)
        if not extracted:
            idx += 1
            continue

        control, next_idx = extracted
        found_controls = True

        plain_chunk = text[chunk_start:idx]
        if plain_chunk:
            spoken_parts.append(plain_chunk)
            word_count += _count_words(plain_chunk)

        controls_by_anchor.setdefault(word_count, []).append(control)
        if control.kind == "pause":
            pause_durations_by_anchor[word_count] = (
                pause_durations_by_anchor.get(word_count, 0.0) + control.duration
            )

        # Keep neighboring words from collapsing together after tags are removed.
        spoken_parts.append(" ")
        idx = next_idx
        chunk_start = idx

    tail_chunk = text[chunk_start:]
    if tail_chunk:
        spoken_parts.append(tail_chunk)
        word_count += _count_words(tail_chunk)

    if not found_controls:
        return TaggedTextProfile(
            original_text=text,
            spoken_text=text,
            word_count=_count_words(text),
            controls_by_anchor={},
            active_state_by_word=[None] * _count_words(text),
            pause_durations_by_anchor={},
        )

    active_state_by_word: List[Optional[str]] = []
    active_state: Optional[str] = None
    for word_index in range(word_count):
        for control in controls_by_anchor.get(word_index, []):
            if control.kind == "state":
                active_state = control.text
        active_state_by_word.append(active_state)

    return TaggedTextProfile(
        original_text=text,
        spoken_text="".join(spoken_parts),
        word_count=word_count,
        controls_by_anchor=controls_by_anchor,
        active_state_by_word=active_state_by_word,
        pause_durations_by_anchor=pause_durations_by_anchor,
    )


def apply_pause_offsets_to_words(words: List[ASRWord], profile: TaggedTextProfile) -> List[ASRWord]:
    if not words or not profile.has_pause_tags:
        return words

    cumulative_shift = 0.0
    for word_index, word in enumerate(words):
        cumulative_shift += profile.pause_durations_by_anchor.get(word_index, 0.0)
        word.start += cumulative_shift
        word.end += cumulative_shift

    trailing_pause = profile.pause_durations_by_anchor.get(len(words), 0.0)
    if trailing_pause > 0.0:
        words[-1].end += trailing_pause

    return words


def apply_pause_offsets_to_segments(segments: List[ASRSegment], profile: TaggedTextProfile) -> List[ASRSegment]:
    if not segments or not profile.has_pause_tags:
        return segments

    flat_words = [word for segment in segments for word in (segment.words or [])]
    if not flat_words:
        return segments

    apply_pause_offsets_to_words(flat_words, profile)
    for segment in segments:
        if segment.words:
            segment.start = segment.words[0].start
            segment.end = segment.words[-1].end

    return segments
