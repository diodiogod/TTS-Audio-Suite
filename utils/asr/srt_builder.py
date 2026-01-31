"""
SRT builder for ASR outputs with smart readability defaults.
"""

from typing import List, Tuple
import re

from utils.asr.types import ASRSegment, ASRWord


def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
        if secs == 60:
            minutes += 1
            secs = 0
            if minutes == 60:
                hours += 1
                minutes = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _wrap_text(text: str, max_chars_per_line: int, max_lines: int) -> str:
    words = text.split()
    if not words:
        return text

    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars_per_line:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)

    return "\n".join(lines[:max_lines])


def _segments_from_words(words: List[ASRWord],
                         min_gap: float,
                         max_duration: float,
                         min_duration: float,
                         max_chars: int,
                         max_cps: float) -> List[ASRSegment]:
    segments: List[ASRSegment] = []
    if not words:
        return segments

    current_words: List[ASRWord] = []
    current_text = ""
    seg_start = words[0].start

    def flush_segment(end_time: float):
        nonlocal current_words, current_text, seg_start
        if not current_words:
            return
        segments.append(ASRSegment(start=seg_start, end=end_time, text=current_text.strip(), words=current_words))
        current_words = []
        current_text = ""

    for idx, word in enumerate(words):
        gap = 0.0
        if current_words:
            gap = max(0.0, word.start - current_words[-1].end)

        if not current_words:
            seg_start = word.start

        projected_text = (current_text + " " + word.text).strip() if current_text else word.text
        projected_end = word.end
        projected_duration = max(0.001, projected_end - seg_start)
        projected_cps = len(projected_text) / projected_duration

        split_on_gap = gap >= min_gap
        split_on_duration = projected_duration > max_duration
        split_on_chars = len(projected_text) > max_chars
        split_on_cps = projected_cps > max_cps and projected_duration >= min_duration

        if current_words and (split_on_gap or split_on_duration or split_on_chars or split_on_cps):
            flush_segment(current_words[-1].end)
            seg_start = word.start
            projected_text = word.text

        current_words.append(word)
        current_text = (current_text + " " + word.text).strip() if current_text else word.text

        # Punctuation-based split
        if current_text:
            last_char = current_text[-1:]
            if last_char in [".", "!", "?", "…"]:
                if (word.end - seg_start) >= min_duration:
                    flush_segment(word.end)
            elif last_char in [",", ";", ":"]:
                if (word.end - seg_start) >= min_duration and len(current_text) >= int(max_chars * 0.5):
                    flush_segment(word.end)

    if current_words:
        flush_segment(current_words[-1].end)

    return segments


def _apply_punctuation(words: List[ASRWord], full_text: str):
    if not words or not full_text:
        return words, 0, 0, []

    # Tokenize text into words and punctuation
    # Split into word/punct tokens (keeps apostrophes for contractions)
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", full_text, flags=re.UNICODE)
    word_idx = 0
    last_word_idx = -1

    def normalize(w: str) -> str:
        return re.sub(r"[^\w]+", "", w, flags=re.UNICODE).lower()

    def expand_contraction(token: str) -> List[str]:
        # Split common apostrophe contractions into word tokens
        if "'" in token:
            parts = token.split("'")
            if len(parts) == 2 and parts[0] and parts[1]:
                return [parts[0], parts[1]]
        return [token]

    total_word_tokens = 0
    matched_word_tokens = 0

    missed_tokens = []
    punct_events = []
    for tok in tokens:
        # Word token
        if re.match(r"[A-Za-z0-9]", tok, flags=re.UNICODE):
            sub_tokens = expand_contraction(tok)
            total_word_tokens += len(sub_tokens)
            if word_idx >= len(words):
                if len(missed_tokens) < 3:
                    missed_tokens.append(tok)
                break
            # Align each sub-token with a small lookahead window
            for sub in sub_tokens:
                norm_tok = normalize(sub)
                matched = False
                lookahead = 4
                for j in range(word_idx, min(len(words), word_idx + lookahead)):
                    if normalize(words[j].text) == norm_tok:
                        words[j].text = sub
                        last_word_idx = j
                        matched_word_tokens += 1
                        word_idx = j + 1
                        matched = True
                        break
                if not matched and len(missed_tokens) < 3:
                    missed_tokens.append(sub)
        else:
            # punctuation: attach to previous word, or next word if none
            attached = False
            if last_word_idx >= 0:
                words[last_word_idx].text = f"{words[last_word_idx].text}{tok}"
                punct_events.append({
                    "punct": tok,
                    "attached_to": words[last_word_idx].text,
                    "start": words[last_word_idx].start,
                    "end": words[last_word_idx].end,
                    "fallback": False,
                })
                attached = True
            else:
                # attach to next available word (fallback to avoid dropping)
                for j in range(word_idx, len(words)):
                    if words[j].text:
                        words[j].text = f"{tok}{words[j].text}"
                        punct_events.append({
                            "punct": tok,
                            "attached_to": words[j].text,
                            "start": words[j].start,
                            "end": words[j].end,
                            "fallback": True,
                        })
                        attached = True
                        break
            if not attached:
                punct_events.append({
                    "punct": tok,
                    "attached_to": "[dropped]",
                    "start": None,
                    "end": None,
                    "fallback": True,
                })

    return words, matched_word_tokens, total_word_tokens, missed_tokens, punct_events


def build_srt(segments: List[ASRSegment],
              mode: str = "smart",
              full_text: str = "",
              max_chars_per_line: int = 42,
              max_lines: int = 2,
              max_duration: float = 6.0,
              min_duration: float = 1.0,
              min_gap: float = 0.6,
              max_cps: float = 20.0,
              return_stats: bool = False):
    if not segments:
        return ("", {"matched": 0, "total": 0}) if return_stats else ""

    if mode == "words":
        flat_words = [w for s in segments for w in (s.words or [])]
        flat_words, matched, total, missed, punct_events = _apply_punctuation(flat_words, full_text)
        segments_to_use = [ASRSegment(start=w.start, end=w.end, text=w.text, words=[w]) for w in flat_words]
    elif mode == "engine_segments":
        segments_to_use = segments
    else:
        flat_words = [w for s in segments for w in (s.words or [])]
        flat_words, matched, total, missed, punct_events = _apply_punctuation(flat_words, full_text)
        if flat_words:
            segments_to_use = _segments_from_words(
                flat_words,
                min_gap=min_gap,
                max_duration=max_duration,
                min_duration=min_duration,
                max_chars=max_chars_per_line * max_lines,
                max_cps=max_cps,
            )
        else:
            segments_to_use = segments
    stats = {
        "matched": matched if 'matched' in locals() else 0,
        "total": total if 'total' in locals() else 0,
        "missed": missed if 'missed' in locals() else [],
        "punct": punct_events if 'punct_events' in locals() else []
    }

    # Heuristic: add sentence-ending punctuation when gaps + capitalization suggest a new sentence
    for i in range(len(segments_to_use) - 1):
        cur = segments_to_use[i]
        nxt = segments_to_use[i + 1]
        if not cur.text:
            continue
        if cur.text[-1:] in [".", "!", "?", "…"]:
            continue
        if nxt.text and nxt.text[:1].isupper() and (nxt.start - cur.end) >= min_gap:
            cur.text = f"{cur.text}."

    lines: List[str] = []
    for idx, seg in enumerate(segments_to_use, start=1):
        start_ts = _format_timestamp(seg.start)
        end_ts = _format_timestamp(seg.end)
        text = _wrap_text(seg.text, max_chars_per_line, max_lines)
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")

    srt = "\n".join(lines).strip()
    return (srt, stats) if return_stats else srt
