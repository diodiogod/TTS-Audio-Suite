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
                      max_cps: float = 20.0) -> Dict[str, Any]:
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
    return {
        "text": result.text or "",
        "timestamps": _format_timestamps(result),
        "srt": srt,
        "info": info,
        "language": result.language or "",
    }
