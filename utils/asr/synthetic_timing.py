"""
Synthetic ASR timing estimation for text-only subtitle generation.
"""

import re
from typing import List

from utils.asr.srt_heuristic_profiles import (
    ENGLISH_DANGLING_TAIL_ALLOWLIST,
    ENGLISH_INCOMPLETE_KEYWORDS,
)
from utils.asr.srt_builder import segments_from_words
from utils.asr.tagged_text import apply_pause_offsets_to_words, parse_tagged_text
from utils.asr.types import ASRResult, ASRSegment, ASRWord


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+", flags=re.UNICODE)
CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+", flags=re.UNICODE)
SENTENCE_END_CHARS = {".", "!", "?", "…"}
SOFT_BREAK_CHARS = {",", ";", ":"}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _split_overlong_part(text: str, max_chars: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    current_words: List[str] = []
    for word in words:
        candidate = " ".join(current_words + [word]).strip()
        if current_words and len(candidate) > max_chars:
            chunks.append(" ".join(current_words).strip())
            current_words = [word]
            continue
        current_words.append(word)

    if current_words:
        chunks.append(" ".join(current_words).strip())
    return [chunk for chunk in chunks if chunk]


def _split_sentence(sentence: str, max_chars: int) -> List[str]:
    sentence = sentence.strip()
    if not sentence:
        return []
    if len(sentence) <= max_chars:
        return [sentence]

    parts = [part.strip() for part in CLAUSE_SPLIT_RE.split(sentence) if part.strip()]
    if len(parts) <= 1:
        return _split_overlong_part(sentence, max_chars)

    chunks: List[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if current and len(candidate) > max_chars:
            chunks.extend(_split_overlong_part(current, max_chars))
            current = part
            continue
        current = candidate

    if current:
        chunks.extend(_split_overlong_part(current, max_chars))
    return [chunk for chunk in chunks if chunk]


def _split_paragraph(paragraph: str, max_chars: int) -> List[str]:
    paragraph = _normalize_whitespace(paragraph)
    if not paragraph:
        return []

    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(paragraph) if sentence.strip()]
    if not sentences:
        return _split_overlong_part(paragraph, max_chars)

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        sentence_parts = _split_sentence(sentence, max_chars)
        for part in sentence_parts:
            if not current:
                current = part
                continue

            candidate = f"{current} {part}".strip()
            if len(candidate) > max_chars:
                chunks.append(current)
                current = part
                continue
            current = candidate

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk]


def _split_sentence_tts_ready(sentence: str, target_chars: int) -> List[str]:
    sentence = sentence.strip()
    if not sentence:
        return []

    soft_limit = max(24, target_chars)
    hard_limit = max(soft_limit + 18, int(soft_limit * 1.35))
    if len(sentence) <= hard_limit:
        return [sentence]

    parts = [part.strip() for part in CLAUSE_SPLIT_RE.split(sentence) if part.strip()]
    if len(parts) <= 1:
        return _split_overlong_part(sentence, hard_limit)

    chunks: List[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if current and len(candidate) > hard_limit:
            chunks.append(current)
            current = part
            continue
        current = candidate

    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def _split_paragraph_tts_ready(paragraph: str, target_chars: int) -> List[str]:
    paragraph = _normalize_whitespace(paragraph)
    if not paragraph:
        return []

    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(paragraph) if sentence.strip()]
    if not sentences:
        return _split_sentence_tts_ready(paragraph, target_chars)

    soft_limit = max(24, target_chars)
    hard_limit = max(soft_limit + 18, int(soft_limit * 1.35))
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        for part in _split_sentence_tts_ready(sentence, target_chars):
            if not current:
                current = part
                continue

            candidate = f"{current} {part}".strip()
            if len(candidate) <= soft_limit:
                current = candidate
                continue

            if _ends_sentence_like(current) and len(candidate) <= hard_limit:
                current = candidate
                continue

            chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk]


def _ends_sentence_like(text: str) -> bool:
    text = text.rstrip()
    return bool(text) and text[-1:] in SENTENCE_END_CHARS


def _word_weight(word: str) -> float:
    alnum = re.sub(r"[^\w]+", "", word, flags=re.UNICODE)
    return float(max(1, len(alnum)))


def _estimate_gap(cue_text: str, min_gap: float, paragraph_break: bool) -> float:
    if paragraph_break:
        return max(min_gap * 1.5, 0.9)

    tail = cue_text.rstrip()[-1:] if cue_text.rstrip() else ""
    if tail in SENTENCE_END_CHARS:
        return max(min_gap, 0.6)
    if tail in SOFT_BREAK_CHARS:
        return max(0.2, min_gap * 0.55)
    return max(0.1, min_gap * 0.35)


def _estimate_word_timings(cue_text: str, start_time: float, end_time: float) -> List[ASRWord]:
    word_tokens = cue_text.split()
    if not word_tokens:
        return []

    weights = [_word_weight(token) for token in word_tokens]
    total_weight = sum(weights) or float(len(word_tokens))
    duration = max(0.001, end_time - start_time)

    words: List[ASRWord] = []
    cursor = start_time
    for idx, token in enumerate(word_tokens):
        if idx == len(word_tokens) - 1:
            word_end = end_time
        else:
            word_duration = duration * (weights[idx] / total_weight)
            word_end = min(end_time, cursor + word_duration)
        words.append(ASRWord(start=cursor, end=word_end, text=token))
        cursor = word_end
    return words


def estimate_asr_result_from_text(
    text: str,
    *,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    max_duration: float = 6.0,
    min_duration: float = 1.0,
    min_gap: float = 0.6,
    max_cps: float = 20.0,
    tts_ready_mode: bool = False,
    punctuation_grace_chars: int = 12,
    min_words_per_segment: int = 2,
    min_segment_seconds: float = 0.4,
    merge_trailing_punct_word: bool = True,
    merge_trailing_punct_max_gap: float = 1.0,
    merge_leading_short_phrase: bool = True,
    merge_leading_short_max_words: int = 2,
    merge_leading_short_max_gap: float = 2.0,
    merge_dangling_tail: bool = True,
    merge_dangling_tail_max_words: int = 3,
    merge_dangling_tail_max_gap: float = 3.0,
    merge_dangling_tail_allowlist: str = ENGLISH_DANGLING_TAIL_ALLOWLIST,
    merge_leading_short_no_punct: bool = True,
    merge_leading_short_no_punct_max_words: int = 2,
    merge_leading_short_no_punct_max_gap: float = 1.5,
    merge_incomplete_sentence: bool = True,
    merge_incomplete_max_gap: float = 1.2,
    merge_incomplete_keywords: str = ENGLISH_INCOMPLETE_KEYWORDS,
    merge_incomplete_split_next: bool = True,
    merge_allow_overlong: bool = False,
) -> ASRResult:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Cannot estimate subtitle timings from empty text.")

    tagged_profile = parse_tagged_text(cleaned)
    spoken_text = tagged_profile.spoken_text if tagged_profile.has_control_tags else cleaned

    if tts_ready_mode:
        target_chars = max(24, int(max(max_cps, 0.1) * max(max_duration, 0.2)))
    else:
        target_chars = max(
            12,
            min(
                max(12, max_chars_per_line * max(1, max_lines)),
                max(12, int(max(max_cps, 0.1) * max(max_duration, 0.2))),
            ),
        )

    paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n+", spoken_text) if paragraph.strip()]
    all_words: List[ASRWord] = []
    timeline = 0.0

    for paragraph_index, paragraph in enumerate(paragraphs):
        cues = _split_paragraph_tts_ready(paragraph, target_chars) if tts_ready_mode else _split_paragraph(paragraph, target_chars)
        for cue_index, cue_text in enumerate(cues):
            cue_duration = len(cue_text) / max(max_cps, 0.1)
            cue_duration = min(max_duration, max(min_duration, cue_duration))
            cue_start = timeline
            cue_end = cue_start + cue_duration
            cue_words = _estimate_word_timings(cue_text, cue_start, cue_end)
            all_words.extend(cue_words)

            timeline = cue_end
            is_last_cue = cue_index == len(cues) - 1
            is_last_paragraph = paragraph_index == len(paragraphs) - 1
            if not (is_last_cue and is_last_paragraph):
                timeline += _estimate_gap(
                    cue_text,
                    min_gap=min_gap,
                    paragraph_break=is_last_cue and not is_last_paragraph,
                )

    if not all_words:
        raise ValueError("Could not estimate subtitle timings from the provided text.")

    apply_pause_offsets_to_words(all_words, tagged_profile)

    segments: List[ASRSegment] = segments_from_words(
        all_words,
        min_gap=min_gap,
        max_duration=max_duration,
        min_duration=min_duration,
        max_chars=max_chars_per_line * max(1, max_lines),
        max_cps=max_cps,
        tts_ready_mode=tts_ready_mode,
        punctuation_grace_chars=punctuation_grace_chars,
        min_words_per_segment=min_words_per_segment,
        min_segment_seconds=min_segment_seconds,
        merge_trailing_punct_word=merge_trailing_punct_word,
        merge_trailing_punct_max_gap=merge_trailing_punct_max_gap,
        merge_leading_short_phrase=merge_leading_short_phrase,
        merge_leading_short_max_words=merge_leading_short_max_words,
        merge_leading_short_max_gap=merge_leading_short_max_gap,
        merge_dangling_tail=merge_dangling_tail,
        merge_dangling_tail_max_words=merge_dangling_tail_max_words,
        merge_dangling_tail_max_gap=merge_dangling_tail_max_gap,
        merge_dangling_tail_allowlist=merge_dangling_tail_allowlist,
        merge_leading_short_no_punct=merge_leading_short_no_punct,
        merge_leading_short_no_punct_max_words=merge_leading_short_no_punct_max_words,
        merge_leading_short_no_punct_max_gap=merge_leading_short_no_punct_max_gap,
        merge_incomplete_sentence=merge_incomplete_sentence,
        merge_incomplete_max_gap=merge_incomplete_max_gap,
        merge_incomplete_keywords=merge_incomplete_keywords,
        merge_incomplete_split_next=merge_incomplete_split_next,
        merge_allow_overlong=merge_allow_overlong,
    )

    return ASRResult(
        text=cleaned,
        segments=segments,
        raw={
            "notes": [
                "Timings were estimated from plain text because asr_timing_data was not connected.",
                "Estimated timings and cue boundaries were guided by the current SRT advanced options.",
                "Estimated timings are approximate and should be reviewed before final use.",
            ]
        },
    )
