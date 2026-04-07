"""
Text to SRT Builder Node - Build subtitle files from timed text data.
"""

import json
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
from utils.asr.srt_heuristic_profiles import (
    ENGLISH_DANGLING_TAIL_ALLOWLIST,
    ENGLISH_INCOMPLETE_KEYWORDS,
    resolve_profile_defaults,
)
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
    raise ValueError("Unsupported word timing item in asr_timing_data")


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
    raise ValueError("Unsupported segment item in asr_timing_data")


def _coerce_asr_result(asr_timing_data: Any, text: str) -> ASRResult:
    if asr_timing_data is None:
        raise ValueError("No asr_timing_data provided.")

    if isinstance(asr_timing_data, str):
        timing_text = asr_timing_data.strip()
        if not timing_text:
            raise ValueError("No asr_timing_data provided.")
        try:
            parsed = json.loads(timing_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "asr_timing_data must be valid JSON exported by ✏️ ASR Transcribe."
            ) from exc
        return _coerce_asr_result(parsed, text)

    if isinstance(asr_timing_data, ASRResult):
        return ASRResult(
            text=text,
            language=asr_timing_data.language,
            segments=[_coerce_segment(segment) for segment in (asr_timing_data.segments or [])],
            raw=asr_timing_data.raw,
        )

    if isinstance(asr_timing_data, dict):
        segments: List[ASRSegment] = [_coerce_segment(seg) for seg in (asr_timing_data.get("segments") or [])]
        return ASRResult(
            text=text,
            language=asr_timing_data.get("language"),
            segments=segments,
            raw=asr_timing_data.get("raw"),
        )

    raise ValueError(
        "asr_timing_data must be JSON exported by ✏️ ASR Transcribe or a compatible ASR timing structure."
    )


def _resolve_srt_options(opts: Dict[str, Any], language: str = None) -> Dict[str, Any]:
    resolved = dict(opts or {})
    profile_defaults = resolve_profile_defaults(
        resolved.get("heuristic_language_profile"),
        language=language,
    )
    for key, value in profile_defaults.items():
        resolved[key] = value
    return resolved


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
                "asr_timing_data": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional ASR timing JSON.\nConnect it from ✏️ ASR Transcribe or paste a saved JSON blob here.\nWhen omitted, the builder estimates subtitle timings from the text using the SRT options."
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
        asr_timing_data: Any = None,
        srt_options: Dict[str, Any] = None,
    ):
        if not text or not text.strip():
            raise ValueError("Text to SRT Builder requires non-empty text.")

        opts = srt_options or {}
        text_value = text.strip()
        has_asr_timing_data = bool(asr_timing_data and (not isinstance(asr_timing_data, str) or asr_timing_data.strip()))

        if not has_asr_timing_data:
            resolved_opts = _resolve_srt_options(opts)
            result = estimate_asr_result_from_text(
                text_value,
                max_chars_per_line=resolved_opts.get("srt_max_chars_per_line", 42),
                max_lines=resolved_opts.get("srt_max_lines", 2),
                max_duration=resolved_opts.get("srt_max_duration", 6.0),
                min_duration=resolved_opts.get("srt_min_duration", 1.0),
                min_gap=resolved_opts.get("srt_min_gap", 0.6),
                max_cps=resolved_opts.get("srt_max_cps", 17.0),
                tts_ready_mode=resolved_opts.get("tts_ready_mode", False),
                tts_ready_paragraph_mode=resolved_opts.get("tts_ready_paragraph_mode", False),
                punctuation_grace_chars=resolved_opts.get("punctuation_grace_chars", 12),
                min_words_per_segment=resolved_opts.get("min_words_per_segment", 2),
                min_segment_seconds=resolved_opts.get("min_segment_seconds", 0.4),
                merge_trailing_punct_word=resolved_opts.get("merge_trailing_punct_word", True),
                merge_trailing_punct_max_gap=resolved_opts.get("merge_trailing_punct_max_gap", 1.0),
                merge_leading_short_phrase=resolved_opts.get("merge_leading_short_phrase", True),
                merge_leading_short_max_words=resolved_opts.get("merge_leading_short_max_words", 2),
                merge_leading_short_max_gap=resolved_opts.get("merge_leading_short_max_gap", 2.0),
                merge_dangling_tail=resolved_opts.get("merge_dangling_tail", True),
                merge_dangling_tail_max_words=resolved_opts.get("merge_dangling_tail_max_words", 3),
                merge_dangling_tail_max_gap=resolved_opts.get("merge_dangling_tail_max_gap", 3.0),
                merge_dangling_tail_allowlist=resolved_opts.get(
                    "merge_dangling_tail_allowlist",
                    ENGLISH_DANGLING_TAIL_ALLOWLIST,
                ),
                merge_leading_short_no_punct=resolved_opts.get("merge_leading_short_no_punct", True),
                merge_leading_short_no_punct_max_words=resolved_opts.get("merge_leading_short_no_punct_max_words", 2),
                merge_leading_short_no_punct_max_gap=resolved_opts.get("merge_leading_short_no_punct_max_gap", 1.5),
                merge_incomplete_sentence=resolved_opts.get("merge_incomplete_sentence", True),
                merge_incomplete_max_gap=resolved_opts.get("merge_incomplete_max_gap", 1.2),
                merge_incomplete_keywords=resolved_opts.get("merge_incomplete_keywords", ENGLISH_INCOMPLETE_KEYWORDS),
                merge_incomplete_split_next=resolved_opts.get("merge_incomplete_split_next", True),
                merge_allow_overlong=resolved_opts.get("merge_allow_overlong", True),
            )
        else:
            result = _coerce_asr_result(asr_timing_data, text_value)
            if not result.segments:
                raise ValueError(
                    "asr_timing_data contains no timed segments. Run ✏️ ASR Transcribe with timestamps=word first."
                )
            tagged_profile = parse_tagged_text(text_value)
            apply_pause_offsets_to_segments(result.segments, tagged_profile)
            resolved_opts = _resolve_srt_options(opts, language=result.language)

        formatted = format_asr_output(
            result,
            srt_mode=resolved_opts.get("srt_mode", "smart"),
            max_chars_per_line=resolved_opts.get("srt_max_chars_per_line", 42),
            max_lines=resolved_opts.get("srt_max_lines", 2),
            max_duration=resolved_opts.get("srt_max_duration", 6.0),
            min_duration=resolved_opts.get("srt_min_duration", 1.0),
            min_gap=resolved_opts.get("srt_min_gap", 0.6),
            max_cps=resolved_opts.get("srt_max_cps", 17.0),
            tts_ready_mode=resolved_opts.get("tts_ready_mode", False),
            tts_ready_paragraph_mode=resolved_opts.get("tts_ready_paragraph_mode", False),
            dedupe_overlaps=resolved_opts.get("dedupe_overlaps", True),
            dedupe_window_ms=resolved_opts.get("dedupe_window_ms", 1500),
            dedupe_min_words=resolved_opts.get("dedupe_min_words", 2),
            dedupe_overlap_ratio=resolved_opts.get("dedupe_overlap_ratio", 0.6),
            punctuation_grace_chars=resolved_opts.get("punctuation_grace_chars", 12),
            min_words_per_segment=resolved_opts.get("min_words_per_segment", 2),
            min_segment_seconds=resolved_opts.get("min_segment_seconds", 0.4),
            merge_trailing_punct_word=resolved_opts.get("merge_trailing_punct_word", True),
            merge_trailing_punct_max_gap=resolved_opts.get("merge_trailing_punct_max_gap", 1.0),
            merge_leading_short_phrase=resolved_opts.get("merge_leading_short_phrase", True),
            merge_leading_short_max_words=resolved_opts.get("merge_leading_short_max_words", 2),
            merge_leading_short_max_gap=resolved_opts.get("merge_leading_short_max_gap", 2.0),
            merge_dangling_tail=resolved_opts.get("merge_dangling_tail", True),
            merge_dangling_tail_max_words=resolved_opts.get("merge_dangling_tail_max_words", 3),
            merge_dangling_tail_max_gap=resolved_opts.get("merge_dangling_tail_max_gap", 3.0),
            merge_dangling_tail_allowlist=resolved_opts.get(
                "merge_dangling_tail_allowlist",
                ENGLISH_DANGLING_TAIL_ALLOWLIST,
            ),
            merge_leading_short_no_punct=resolved_opts.get("merge_leading_short_no_punct", True),
            merge_leading_short_no_punct_max_words=resolved_opts.get("merge_leading_short_no_punct_max_words", 2),
            merge_leading_short_no_punct_max_gap=resolved_opts.get("merge_leading_short_no_punct_max_gap", 1.5),
            merge_incomplete_sentence=resolved_opts.get("merge_incomplete_sentence", True),
            merge_incomplete_max_gap=resolved_opts.get("merge_incomplete_max_gap", 1.2),
            merge_incomplete_keywords=resolved_opts.get("merge_incomplete_keywords", ENGLISH_INCOMPLETE_KEYWORDS),
            merge_incomplete_split_next=resolved_opts.get("merge_incomplete_split_next", True),
            merge_allow_overlong=resolved_opts.get("merge_allow_overlong", True),
            normalize_cue_end_punctuation=resolved_opts.get("normalize_cue_end_punctuation", False),
        )

        if not has_asr_timing_data:
            info = append_info_items(
                formatted["info"],
                "NOTES",
                "Subtitle text and timings both came from the Text to SRT Builder input. Timings were estimated from the SRT options.",
            )
        else:
            info = append_info_items(
                formatted["info"],
                "NOTES",
                "Subtitle text came from the Text to SRT Builder input. Timings came from asr_timing_data.",
            )
        return (formatted["srt"], formatted["timestamps"], info)


NODE_CLASS_MAPPINGS = {
    "TextToSRTBuilderNode": TextToSRTBuilderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextToSRTBuilderNode": "📺 Text to SRT Builder"
}
