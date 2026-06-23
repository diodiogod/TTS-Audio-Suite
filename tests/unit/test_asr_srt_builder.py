"""
Unit tests for ASR SRT builder language-aware rendering.
"""

import os
import sys
from pathlib import Path

custom_node_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(custom_node_root))
os.environ.setdefault("COMFYUI_TESTING", "1")

from utils.asr.srt_builder import build_srt
from utils.asr.pipeline import format_asr_output
from utils.asr.synthetic_timing import estimate_asr_result_from_text
from utils.asr.tagged_text import parse_tagged_text
from utils.asr.types import ASRSegment, ASRWord


def _words_from_chars(text: str, step: float = 0.2):
    words = []
    cursor = 0.0
    for char in text:
        words.append(ASRWord(start=cursor, end=cursor + step, text=char))
        cursor += step
    return words


def test_chinese_punctuation_attaches_to_previous_character_without_spaces():
    spoken_chars = "唉人气太高也是一种苦恼谁让我这么受欢迎呢"
    words = _words_from_chars(spoken_chars)
    segments = [ASRSegment(start=0.0, end=words[-1].end, text=spoken_chars, words=words)]

    srt, stats = build_srt(
        segments,
        mode="smart",
        full_text="唉，人气太高也是一种苦恼。谁让我这么受欢迎呢？",
        max_chars_per_line=100,
        max_lines=2,
        min_duration=0.0,
        min_gap=10.0,
        min_words_per_segment=1,
        min_segment_seconds=0.0,
        dedupe_overlaps=False,
        return_stats=True,
    )

    assert "唉，人气太高也是一种苦恼。谁让我这么受欢迎呢？" in srt
    assert "唉，。？" not in srt
    assert "人 气" not in srt
    assert stats["matched"] == stats["total"]


def test_forced_aligner_punctuation_words_do_not_leak_into_output():
    spoken_chars = "唉人气太高也是一种苦恼谁让我这么受欢迎呢"
    words = [
        ASRWord(start=0.0, end=0.01, text="？"),
        ASRWord(start=0.01, end=0.02, text="。"),
        ASRWord(start=0.02, end=0.03, text="，"),
    ]
    cursor = 0.03
    for char in spoken_chars:
        words.append(ASRWord(start=cursor, end=cursor + 0.2, text=char))
        cursor += 0.2
    segments = [ASRSegment(start=0.0, end=words[-1].end, text=spoken_chars, words=words)]

    srt = build_srt(
        segments,
        mode="smart",
        full_text="唉，人气太高也是一种苦恼。谁让我这么受欢迎呢？",
        max_chars_per_line=100,
        max_lines=2,
        min_duration=0.0,
        min_gap=10.0,
        min_words_per_segment=1,
        min_segment_seconds=0.0,
        dedupe_overlaps=False,
    )

    assert "？。，唉" not in srt
    assert "唉，人气太高也是一种苦恼。谁让我这么受欢迎呢？" in srt


def test_plain_text_builder_estimation_handles_chinese_punctuation():
    text = "唉，人气太高也是一种苦恼。谁让我这么受欢迎呢？"
    result = estimate_asr_result_from_text(
        text,
        max_chars_per_line=100,
        max_lines=2,
        min_duration=0.0,
        min_gap=10.0,
        min_words_per_segment=1,
        min_segment_seconds=0.0,
    )

    formatted = format_asr_output(
        result,
        max_chars_per_line=100,
        max_lines=2,
        min_duration=0.0,
        min_gap=10.0,
        min_words_per_segment=1,
        min_segment_seconds=0.0,
        dedupe_overlaps=False,
    )

    assert "？。，唉" not in formatted["srt"]
    assert text in formatted["srt"]


def test_chinese_curly_quotes_do_not_add_inner_spaces():
    text = "母亲发来的消息：“吃饭了吗？别总是熬夜。”他说：“我很好。”"
    result = estimate_asr_result_from_text(
        text,
        max_chars_per_line=100,
        max_lines=2,
        min_duration=0.0,
        min_gap=10.0,
        min_words_per_segment=1,
        min_segment_seconds=0.0,
    )

    formatted = format_asr_output(
        result,
        max_chars_per_line=100,
        max_lines=2,
        min_duration=0.0,
        min_gap=10.0,
        min_words_per_segment=1,
        min_segment_seconds=0.0,
        dedupe_overlaps=False,
    )

    assert "：“ 吃饭" not in formatted["srt"]
    assert "“ 我很好" not in formatted["srt"]
    assert text in formatted["srt"]


def test_english_spacing_is_preserved():
    words = [
        ASRWord(start=0.0, end=0.2, text="You"),
        ASRWord(start=0.2, end=0.4, text="know"),
        ASRWord(start=0.4, end=0.6, text="after"),
    ]
    segments = [ASRSegment(start=0.0, end=0.6, text="You know after", words=words)]

    srt = build_srt(
        segments,
        mode="smart",
        full_text="You know, after.",
        max_chars_per_line=100,
        max_lines=2,
        min_duration=0.0,
        min_gap=10.0,
        min_words_per_segment=1,
        min_segment_seconds=0.0,
        dedupe_overlaps=False,
    )

    assert "You know, after." in srt


def test_punctuation_only_word_stream_does_not_crash_srt_builder():
    words = [
        ASRWord(start=0.0, end=0.01, text=","),
        ASRWord(start=0.01, end=0.02, text="."),
        ASRWord(start=0.02, end=0.03, text="?"),
    ]
    segments = [ASRSegment(start=0.0, end=0.03, text="", words=words)]

    srt, stats = build_srt(
        segments,
        mode="smart",
        full_text="Hello world.",
        max_chars_per_line=100,
        max_lines=2,
        min_duration=0.0,
        min_gap=10.0,
        min_words_per_segment=1,
        min_segment_seconds=0.0,
        dedupe_overlaps=False,
        return_stats=True,
    )

    assert isinstance(srt, str)
    assert stats["matched"] == 0
    assert stats["total"] == 0
    assert stats["punct"] == []


def test_speaker_state_tags_consume_trailing_colon_for_alignment():
    full_text = "[Speaker 1]: Jane. [Speaker 2]: Mark."
    profile = parse_tagged_text(full_text)

    assert profile.spoken_text == " Jane.   Mark."

    words = [
        ASRWord(start=0.0, end=0.4, text="Jane"),
        ASRWord(start=0.4, end=0.8, text="Mark"),
    ]
    segments = [
        ASRSegment(start=0.0, end=0.4, text="[Speaker 1]: Jane.", speaker="Speaker 1", words=[words[0]]),
        ASRSegment(start=0.4, end=0.8, text="[Speaker 2]: Mark.", speaker="Speaker 2", words=[words[1]]),
    ]

    srt, stats = build_srt(
        segments,
        mode="smart",
        full_text=full_text,
        max_chars_per_line=100,
        max_lines=2,
        min_duration=0.0,
        min_gap=10.0,
        min_words_per_segment=1,
        min_segment_seconds=0.0,
        dedupe_overlaps=False,
        return_stats=True,
    )

    assert "[Speaker 1]: Jane." in srt
    assert "[Speaker 2]: Mark." in srt
    assert "Jane.:" not in srt
    assert "Mark.:" not in srt
    assert all(evt["punct"] != ":" for evt in stats["punct"])
