"""
Unified ASR pipeline: adapter selection + result normalization.
"""

from typing import Dict, Any

from utils.asr.types import ASRRequest, ASRResult, ASRSegment, ASRWord
from utils.asr.srt_builder import build_srt
from utils.asr.adapter_registry import get_asr_adapter_class


def _format_timestamps(result: ASRResult) -> str:
    if not result.segments:
        return "No timestamps generated."
    lines = []
    for seg in result.segments:
        if seg.words:
            for w in seg.words:
                lines.append(f"[{w.start:.2f} - {w.end:.2f}] {w.text}")
        else:
            lines.append(f"[{seg.start:.2f} - {seg.end:.2f}] {seg.text}")
    return "\n".join(lines) if lines else "No timestamps generated."


def run_asr(engine_data: Dict[str, Any], request: ASRRequest) -> ASRResult:
    engine_type = engine_data.get("engine_type")
    adapter_cls = get_asr_adapter_class(engine_type)
    adapter = adapter_cls(engine_data)
    return adapter.transcribe(request)


def format_asr_output(result: ASRResult,
                      srt_mode: str = "smart",
                      max_chars_per_line: int = 42,
                      max_lines: int = 2,
                      max_duration: float = 6.0,
                      min_duration: float = 1.0,
                      min_gap: float = 0.6,
                      max_cps: float = 20.0,
                      dedupe_overlaps: bool = True,
                      dedupe_window_ms: int = 1500,
                      dedupe_min_words: int = 2,
                      dedupe_overlap_ratio: float = 0.6,
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
                      merge_dangling_tail_allowlist: str = "a,an,the,to,of,and,or,im,i'm,you,you're,we,they,he,she,it",
                      merge_leading_short_no_punct: bool = True,
                      merge_leading_short_no_punct_max_words: int = 2,
                      merge_leading_short_no_punct_max_gap: float = 1.5,
                      merge_incomplete_sentence: bool = True,
                      merge_incomplete_max_gap: float = 1.2,
                      merge_incomplete_keywords: str = "what,why,how,where,who,which,when",
                      merge_incomplete_split_next: bool = True,
                      merge_allow_overlong: bool = False) -> Dict[str, Any]:
    srt, stats = build_srt(
        result.segments,
        mode=srt_mode,
        full_text=result.text or "",
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
        max_duration=max_duration,
        min_duration=min_duration,
        min_gap=min_gap,
        max_cps=max_cps,
        dedupe_overlaps=dedupe_overlaps,
        dedupe_window_ms=dedupe_window_ms,
        dedupe_min_words=dedupe_min_words,
        dedupe_overlap_ratio=dedupe_overlap_ratio,
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
        return_stats=True,
    )
    align_info = ""
    if stats.get("total", 0) > 0:
        match_ratio = stats["matched"] / max(1, stats["total"])
        if match_ratio < 0.6:
            print(f"⚠️ ASR punctuation alignment matched {stats['matched']}/{stats['total']} words "
                  f"({match_ratio:.0%}). SRT punctuation may be imperfect.")
        align_info = f"punct_align={match_ratio:.0%} ({stats['matched']}/{stats['total']})"
        if stats.get("missed"):
            align_info += f" | missed={' '.join(stats['missed'])}"
    info_parts = [f"language={result.language or 'unknown'}", f"segments={len(result.segments)}"]
    if stats.get("deduped", 0) > 0:
        info_parts.append(f"deduped={stats['deduped']}")
    info = " | ".join(info_parts)
    if align_info:
        info += f"\n{align_info}"
    if stats.get("punct"):
        issues = [p for p in stats["punct"] if p.get("fallback")]
        if issues:
            info += "\nPUNCTUATION (fallbacks):"
            for evt in issues:
                punct = evt.get("punct", "")
                target = evt.get("attached_to", "")
                if evt.get("start") is not None:
                    timing = f"{evt['start']:.2f}-{evt['end']:.2f}"
                else:
                    timing = "n/a"
                info += f"\n  {punct:<2} -> '{target}'".ljust(34) + f" | {timing:>11}s"
    if stats.get("dedupe_events"):
        info += "\nDEDUPE (removed overlaps):"
        for evt in stats["dedupe_events"]:
            phrase = evt.get("phrase", "")
            timing = f"{evt.get('start', 0.0):.2f}-{evt.get('end', 0.0):.2f}"
            ratio = evt.get("overlap_ratio", 0.0)
            count = evt.get("count", 0)
            info += f"\n  {phrase[:60]}".ljust(34) + f" | {timing:>11}s | {count}w | {ratio:.0%}"
    return {
        "text": result.text or "",
        "timestamps": _format_timestamps(result),
        "srt": srt,
        "info": info,
        "language": result.language or "",
    }
