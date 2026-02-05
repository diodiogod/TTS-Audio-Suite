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
                         max_cps: float,
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
                         merge_allow_overlong: bool = False) -> List[ASRSegment]:
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
        last_char_projected = projected_text[-1:] if projected_text else ""
        is_sentence_end = last_char_projected in [".", "!", "?", "…"]

        split_on_gap = gap >= min_gap
        split_on_duration = projected_duration > max_duration
        split_on_chars = len(projected_text) > max_chars
        if split_on_chars and is_sentence_end and punctuation_grace_chars > 0:
            split_on_chars = len(projected_text) > (max_chars + punctuation_grace_chars)
        split_on_cps = projected_cps > max_cps and projected_duration >= min_duration

        if split_on_gap and merge_trailing_punct_word:
            if gap <= merge_trailing_punct_max_gap and is_sentence_end:
                split_on_gap = False

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

    # Merge micro-segments (very short or too few words)
    merged: List[ASRSegment] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        word_count = len(seg.text.split())
        duration = seg.end - seg.start
        if (word_count < min_words_per_segment or duration <= min_segment_seconds) and i + 1 < len(segments):
            nxt = segments[i + 1]
            merged_text = f"{seg.text} {nxt.text}".strip()
            merged_seg = ASRSegment(start=seg.start, end=nxt.end, text=merged_text, words=(seg.words + nxt.words))
            merged.append(merged_seg)
            i += 2
            continue
        merged.append(seg)
        i += 1

    max_merge_duration = max_duration * 1.2
    def can_merge_by_duration(start: float, end: float) -> bool:
        if merge_allow_overlong:
            return True
        return (end - start) <= max_merge_duration

    # Merge a very short leading phrase into the previous segment if it follows punctuation
    if merge_leading_short_phrase and len(merged) > 1:
        stitched: List[ASRSegment] = [merged[0]]
        for seg in merged[1:]:
            prev = stitched[-1]
            gap = seg.start - prev.end
            if gap <= merge_leading_short_max_gap:
                if prev.text and prev.text[-1:] in [".", "!", "?", "…"]:
                    if len(seg.text.split()) <= max(1, merge_leading_short_max_words):
                        if can_merge_by_duration(prev.start, seg.end):
                            merged_text = f"{prev.text} {seg.text}".strip()
                            stitched[-1] = ASRSegment(
                                start=prev.start,
                                end=seg.end,
                                text=merged_text,
                                words=(prev.words + seg.words),
                            )
                            continue
            stitched.append(seg)
        merged = stitched

    if merge_leading_short_no_punct and len(merged) > 1:
        stitched: List[ASRSegment] = [merged[0]]
        for seg in merged[1:]:
            prev = stitched[-1]
            gap = seg.start - prev.end
            if gap <= merge_leading_short_no_punct_max_gap:
                if prev.text and prev.text[-1:] not in [".", "!", "?", "…"]:
                    if len(seg.text.split()) <= max(1, merge_leading_short_no_punct_max_words):
                        if can_merge_by_duration(prev.start, seg.end):
                            merged_text = f"{prev.text} {seg.text}".strip()
                            stitched[-1] = ASRSegment(
                                start=prev.start,
                                end=seg.end,
                                text=merged_text,
                                words=(prev.words + seg.words),
                            )
                            continue
            stitched.append(seg)
        merged = stitched

    if merge_dangling_tail and len(merged) > 1:
        allow = {w.strip().lower() for w in merge_dangling_tail_allowlist.split(",") if w.strip()}
        stitched: List[ASRSegment] = [merged[0]]
        for seg in merged[1:]:
            prev = stitched[-1]
            gap = seg.start - prev.end
            if gap <= merge_dangling_tail_max_gap and prev.text:
                # Only look at the last clause after sentence-ending punctuation
                clauses = re.split(r"[.!?…]\s*", prev.text)
                last_clause = clauses[-1].strip() if clauses else prev.text.strip()
                clause_words = last_clause.split()
                tail = clause_words[-1].strip(".,!?…:;").lower() if clause_words else ""
                next_has_sentence_end = bool(seg.text and re.search(r"[.!?…]", seg.text))
                if tail in allow and next_has_sentence_end:
                    if len(clause_words) <= max(1, merge_dangling_tail_max_words):
                        # Prefer merging only the first sentence from next segment
                        split_idx = -1
                        for i, w in enumerate(seg.words):
                            if w.text and w.text[-1:] in [".", "!", "?", "…"]:
                                split_idx = i
                                break
                        if split_idx >= 0:
                            head_words = seg.words[:split_idx + 1]
                            tail_words = seg.words[split_idx + 1:]
                            head_text = " ".join([w.text for w in head_words]).strip()
                            tail_text = " ".join([w.text for w in tail_words]).strip()
                            combined = f"{prev.text} {head_text}".strip()
                            if len(combined) <= (max_chars + punctuation_grace_chars) and can_merge_by_duration(prev.start, head_words[-1].end):
                                stitched[-1] = ASRSegment(
                                    start=prev.start,
                                    end=head_words[-1].end,
                                    text=combined,
                                    words=(prev.words + head_words),
                                )
                                if tail_words:
                                    stitched.append(ASRSegment(
                                        start=tail_words[0].start,
                                        end=tail_words[-1].end,
                                        text=tail_text,
                                        words=tail_words,
                                    ))
                                continue
                        if can_merge_by_duration(prev.start, seg.end):
                            merged_text = f"{prev.text} {seg.text}".strip()
                            stitched[-1] = ASRSegment(
                                start=prev.start,
                                end=seg.end,
                                text=merged_text,
                                words=(prev.words + seg.words),
                            )
                            continue
            stitched.append(seg)
        merged = stitched

    if merge_incomplete_sentence and len(merged) > 1:
        keywords = {w.strip().lower() for w in merge_incomplete_keywords.split(",") if w.strip()}
        stitched: List[ASRSegment] = [merged[0]]
        for seg in merged[1:]:
            prev = stitched[-1]
            gap = seg.start - prev.end
            if gap <= merge_incomplete_max_gap:
                if prev.text and prev.text[-1:] not in [".", "!", "?", "…"]:
                    prev_lower = prev.text.lower()
                    if any(k in prev_lower.split() for k in keywords):
                        combined = f"{prev.text} {seg.text}".strip()
                        if len(combined) <= (max_chars + punctuation_grace_chars) and can_merge_by_duration(prev.start, seg.end):
                            stitched[-1] = ASRSegment(
                                start=prev.start,
                                end=seg.end,
                                text=combined,
                                words=(prev.words + seg.words),
                            )
                            continue
            stitched.append(seg)
        merged = stitched

    if merge_incomplete_split_next and len(merged) > 1:
        keywords = {w.strip().lower() for w in merge_incomplete_keywords.split(",") if w.strip()}
        stitched: List[ASRSegment] = [merged[0]]
        for seg in merged[1:]:
            prev = stitched[-1]
            gap = seg.start - prev.end
            if gap <= merge_incomplete_max_gap:
                if prev.text and prev.text[-1:] not in [".", "!", "?", "…"]:
                    prev_lower = prev.text.lower()
                    if any(k in prev_lower.split() for k in keywords):
                        # Split next segment at first sentence-ending punctuation
                        split_idx = -1
                        for i, w in enumerate(seg.words):
                            if w.text and w.text[-1:] in [".", "!", "?", "…"]:
                                split_idx = i
                                break
                        if split_idx >= 0:
                            head_words = seg.words[:split_idx + 1]
                            tail_words = seg.words[split_idx + 1:]
                            head_text = " ".join([w.text for w in head_words]).strip()
                            tail_text = " ".join([w.text for w in tail_words]).strip()
                            combined = f"{prev.text} {head_text}".strip()
                            if len(combined) <= (max_chars + punctuation_grace_chars) and can_merge_by_duration(prev.start, head_words[-1].end):
                                stitched[-1] = ASRSegment(
                                    start=prev.start,
                                    end=head_words[-1].end,
                                    text=combined,
                                    words=(prev.words + head_words),
                                )
                                if tail_words:
                                    stitched.append(ASRSegment(
                                        start=tail_words[0].start,
                                        end=tail_words[-1].end,
                                        text=tail_text,
                                        words=tail_words,
                                    ))
                                continue
            stitched.append(seg)
        merged = stitched

    return merged


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


def _dedupe_overlaps(words: List[ASRWord],
                     window_ms: int = 1500,
                     min_match_words: int = 2,
                     overlap_ratio: float = 0.6):
    if not words:
        return words, 0, []

    window_s = max(0.0, window_ms / 1000.0)
    out: List[ASRWord] = []
    removed = 0
    events = []

    # Normalize text for matching
    def norm(w: str) -> str:
        return re.sub(r"[^\w]+", "", w, flags=re.UNICODE).lower()

    i = 0
    while i < len(words):
        w = words[i]
        # Look back for a potential overlap window
        j = len(out) - 1
        while j >= 0 and (w.start - out[j].end) <= window_s:
            # Check overlap in time
            if w.start <= out[j].end:
                # Try to match sequences from out[j:] to words[i:]
                max_k = min(6, len(words) - i, len(out) - j)
                for k in range(max_k, min_match_words - 1, -1):
                    seq_a = [norm(out[j + t].text) for t in range(k)]
                    seq_b = [norm(words[i + t].text) for t in range(k)]
                    if seq_a == seq_b:
                        # Overlap ratio check using time span
                        span_a = out[j + k - 1].end - out[j].start
                        span_b = words[i + k - 1].end - words[i].start
                        if span_b <= 0:
                            span_b = 0.001
                        overlap = max(0.0, out[j + k - 1].end - words[i].start)
                        if overlap / span_b >= overlap_ratio:
                            # Skip duplicate sequence in current stream
                            removed += k
                            phrase = " ".join([out[j + t].text for t in range(k)])
                            events.append({
                                "phrase": phrase,
                                "start": words[i].start,
                                "end": words[i + k - 1].end,
                                "overlap_ratio": overlap / span_b,
                                "count": k,
                            })
                            i += k
                            break
                else:
                    j -= 1
                    continue
                break
            j -= 1
        else:
            out.append(w)
            i += 1
            continue
        # If we skipped, don't append current word
        if i < len(words) and (len(out) == 0 or out[-1] is not words[i]):
            continue

    return out, removed, events


def build_srt(segments: List[ASRSegment],
              mode: str = "smart",
              full_text: str = "",
              max_chars_per_line: int = 42,
              max_lines: int = 2,
              max_duration: float = 6.0,
              min_duration: float = 1.0,
              min_gap: float = 0.6,
              max_cps: float = 20.0,
              return_stats: bool = False,
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
              merge_allow_overlong: bool = False):
    if not segments:
        return ("", {"matched": 0, "total": 0}) if return_stats else ""

    if mode == "words":
        flat_words = [w for s in segments for w in (s.words or [])]
        if dedupe_overlaps:
            flat_words, removed, dedupe_events = _dedupe_overlaps(
                flat_words,
                window_ms=dedupe_window_ms,
                min_match_words=dedupe_min_words,
                overlap_ratio=dedupe_overlap_ratio,
            )
        else:
            removed = 0
            dedupe_events = []
        flat_words, matched, total, missed, punct_events = _apply_punctuation(flat_words, full_text)
        segments_to_use = [ASRSegment(start=w.start, end=w.end, text=w.text, words=[w]) for w in flat_words]
    elif mode == "engine_segments":
        segments_to_use = segments
    else:
        flat_words = [w for s in segments for w in (s.words or [])]
        if dedupe_overlaps:
            flat_words, removed, dedupe_events = _dedupe_overlaps(
                flat_words,
                window_ms=dedupe_window_ms,
                min_match_words=dedupe_min_words,
                overlap_ratio=dedupe_overlap_ratio,
            )
        else:
            removed = 0
            dedupe_events = []
        flat_words, matched, total, missed, punct_events = _apply_punctuation(flat_words, full_text)
        if flat_words:
            segments_to_use = _segments_from_words(
                flat_words,
                min_gap=min_gap,
                max_duration=max_duration,
                min_duration=min_duration,
                max_chars=max_chars_per_line * max_lines,
                max_cps=max_cps,
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
        else:
            segments_to_use = segments
    stats = {
        "matched": matched if 'matched' in locals() else 0,
        "total": total if 'total' in locals() else 0,
        "missed": missed if 'missed' in locals() else [],
        "punct": punct_events if 'punct_events' in locals() else [],
        "deduped": removed if 'removed' in locals() else 0,
        "dedupe_events": dedupe_events if 'dedupe_events' in locals() else [],
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
