"""
Text to SRT Builder Node - Build subtitle files from timed text data.
"""

import os
import sys
import importlib.util
from typing import Any, Dict, List

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load base_node module directly
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

BaseChatterBoxNode = base_module.BaseChatterBoxNode

from utils.asr.pipeline import format_asr_output, append_info_items
from utils.asr.synthetic_timing import estimate_asr_result_from_text
from utils.asr.tagged_text import apply_pause_offsets_to_segments, parse_tagged_text
from utils.asr.types import ASRResult, ASRSegment, ASRWord


def _coerce_word(word: Any) -> ASRWord:
    if isinstance(word, ASRWord):
        return word
    if isinstance(word, dict):
        return ASRWord(
            start=float(word.get("start", 0.0)),
            end=float(word.get("end", 0.0)),
            text=str(word.get("text", "")),
        )
    raise ValueError("Unsupported word timing item in timing_data")


def _coerce_segment(segment: Any) -> ASRSegment:
    if isinstance(segment, ASRSegment):
        return segment
    if isinstance(segment, dict):
        words = [_coerce_word(word) for word in (segment.get("words") or [])]
        return ASRSegment(
            start=float(segment.get("start", 0.0)),
            end=float(segment.get("end", 0.0)),
            text=str(segment.get("text", "")),
            speaker=segment.get("speaker"),
            words=words,
        )
    raise ValueError("Unsupported segment item in timing_data")


def _coerce_asr_result(timing_data: Any, text: str) -> ASRResult:
    if timing_data is None:
        raise ValueError("No timing_data provided.")

    if isinstance(timing_data, ASRResult):
        return ASRResult(
            text=text,
            language=timing_data.language,
            segments=[_coerce_segment(segment) for segment in (timing_data.segments or [])],
            raw=timing_data.raw,
        )

    if isinstance(timing_data, dict):
        segments: List[ASRSegment] = [_coerce_segment(seg) for seg in (timing_data.get("segments") or [])]
        return ASRResult(
            text=text,
            language=timing_data.get("language"),
            segments=segments,
            raw=timing_data.get("raw"),
        )

    raise ValueError(
        "timing_data must come from ✏️ ASR Transcribe or use a compatible ASR timing structure."
    )


class TextToSRTBuilderNode(BaseChatterBoxNode):
    @classmethod
    def NAME(cls):
        return "📺 Text to SRT Builder"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text to use in the subtitle cues.\nUse the transcript directly, feed in cleaned text from a post-process node like ASR Punctuation / Truecase, or write plain text and let the builder estimate timings."
                }),
            },
            "optional": {
                "timing_data": ("TIMING_DATA", {
                    "tooltip": "Optional timed segment/word data.\nWhen connected, the builder uses these timings (for example from ✏️ ASR Transcribe).\nWhen omitted, it estimates subtitle timings from the text using the SRT options."
                }),
                "srt_options": ("SRT_OPTIONS", {
                    "tooltip": "Optional subtitle-building policy.\nIf omitted, the builder uses the Broadcast-style defaults."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("srt", "timestamps", "info")
    FUNCTION = "build"
    CATEGORY = "TTS Audio Suite/📺 Subtitles"

    def build(
        self,
        text: str,
        timing_data: Any = None,
        srt_options: Dict[str, Any] = None,
    ):
        if not text or not text.strip():
            raise ValueError("Text to SRT Builder requires non-empty text.")

        opts = srt_options or {}
        text_value = text.strip()

        if timing_data is None:
            result = estimate_asr_result_from_text(
                text_value,
                max_chars_per_line=opts.get("srt_max_chars_per_line", 42),
                max_lines=opts.get("srt_max_lines", 2),
                max_duration=opts.get("srt_max_duration", 6.0),
                min_duration=opts.get("srt_min_duration", 1.0),
                min_gap=opts.get("srt_min_gap", 0.6),
                max_cps=opts.get("srt_max_cps", 17.0),
                punctuation_grace_chars=opts.get("punctuation_grace_chars", 12),
                min_words_per_segment=opts.get("min_words_per_segment", 2),
                min_segment_seconds=opts.get("min_segment_seconds", 0.4),
                merge_trailing_punct_word=opts.get("merge_trailing_punct_word", True),
                merge_trailing_punct_max_gap=opts.get("merge_trailing_punct_max_gap", 1.0),
                merge_leading_short_phrase=opts.get("merge_leading_short_phrase", True),
                merge_leading_short_max_words=opts.get("merge_leading_short_max_words", 2),
                merge_leading_short_max_gap=opts.get("merge_leading_short_max_gap", 2.0),
                merge_dangling_tail=opts.get("merge_dangling_tail", True),
                merge_dangling_tail_max_words=opts.get("merge_dangling_tail_max_words", 3),
                merge_dangling_tail_max_gap=opts.get("merge_dangling_tail_max_gap", 3.0),
                merge_dangling_tail_allowlist=opts.get(
                    "merge_dangling_tail_allowlist",
                    "a,an,the,to,of,and,or,im,i'm,you,you're,we,they,he,she,it",
                ),
                merge_leading_short_no_punct=opts.get("merge_leading_short_no_punct", True),
                merge_leading_short_no_punct_max_words=opts.get("merge_leading_short_no_punct_max_words", 2),
                merge_leading_short_no_punct_max_gap=opts.get("merge_leading_short_no_punct_max_gap", 1.5),
                merge_incomplete_sentence=opts.get("merge_incomplete_sentence", True),
                merge_incomplete_max_gap=opts.get("merge_incomplete_max_gap", 1.2),
                merge_incomplete_keywords=opts.get("merge_incomplete_keywords", "what,why,how,where,who,which,when"),
                merge_incomplete_split_next=opts.get("merge_incomplete_split_next", True),
                merge_allow_overlong=opts.get("merge_allow_overlong", True),
            )
        else:
            result = _coerce_asr_result(timing_data, text_value)
            if not result.segments:
                raise ValueError(
                    "timing_data contains no timed segments. Run ✏️ ASR Transcribe with timestamps=word first."
                )
            tagged_profile = parse_tagged_text(text_value)
            apply_pause_offsets_to_segments(result.segments, tagged_profile)

        formatted = format_asr_output(
            result,
            srt_mode=opts.get("srt_mode", "smart"),
            max_chars_per_line=opts.get("srt_max_chars_per_line", 42),
            max_lines=opts.get("srt_max_lines", 2),
            max_duration=opts.get("srt_max_duration", 6.0),
            min_duration=opts.get("srt_min_duration", 1.0),
            min_gap=opts.get("srt_min_gap", 0.6),
            max_cps=opts.get("srt_max_cps", 17.0),
            dedupe_overlaps=opts.get("dedupe_overlaps", True),
            dedupe_window_ms=opts.get("dedupe_window_ms", 1500),
            dedupe_min_words=opts.get("dedupe_min_words", 2),
            dedupe_overlap_ratio=opts.get("dedupe_overlap_ratio", 0.6),
            punctuation_grace_chars=opts.get("punctuation_grace_chars", 12),
            min_words_per_segment=opts.get("min_words_per_segment", 2),
            min_segment_seconds=opts.get("min_segment_seconds", 0.4),
            merge_trailing_punct_word=opts.get("merge_trailing_punct_word", True),
            merge_trailing_punct_max_gap=opts.get("merge_trailing_punct_max_gap", 1.0),
            merge_leading_short_phrase=opts.get("merge_leading_short_phrase", True),
            merge_leading_short_max_words=opts.get("merge_leading_short_max_words", 2),
            merge_leading_short_max_gap=opts.get("merge_leading_short_max_gap", 2.0),
            merge_dangling_tail=opts.get("merge_dangling_tail", True),
            merge_dangling_tail_max_words=opts.get("merge_dangling_tail_max_words", 3),
            merge_dangling_tail_max_gap=opts.get("merge_dangling_tail_max_gap", 3.0),
            merge_dangling_tail_allowlist=opts.get(
                "merge_dangling_tail_allowlist",
                "a,an,the,to,of,and,or,im,i'm,you,you're,we,they,he,she,it",
            ),
            merge_leading_short_no_punct=opts.get("merge_leading_short_no_punct", True),
            merge_leading_short_no_punct_max_words=opts.get("merge_leading_short_no_punct_max_words", 2),
            merge_leading_short_no_punct_max_gap=opts.get("merge_leading_short_no_punct_max_gap", 1.5),
            merge_incomplete_sentence=opts.get("merge_incomplete_sentence", True),
            merge_incomplete_max_gap=opts.get("merge_incomplete_max_gap", 1.2),
            merge_incomplete_keywords=opts.get("merge_incomplete_keywords", "what,why,how,where,who,which,when"),
            merge_incomplete_split_next=opts.get("merge_incomplete_split_next", True),
            merge_allow_overlong=opts.get("merge_allow_overlong", True),
            normalize_cue_end_punctuation=opts.get("normalize_cue_end_punctuation", False),
        )

        if timing_data is None:
            info = append_info_items(
                formatted["info"],
                "NOTES",
                "Subtitle text and timings both came from the Text to SRT Builder input. Timings were estimated from the SRT options.",
            )
        else:
            info = append_info_items(
                formatted["info"],
                "NOTES",
                "Subtitle text came from the Text to SRT Builder input. Timings came from timing_data.",
            )
        return (formatted["srt"], formatted["timestamps"], info)


NODE_CLASS_MAPPINGS = {
    "TextToSRTBuilderNode": TextToSRTBuilderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextToSRTBuilderNode": "📺 Text to SRT Builder"
}
