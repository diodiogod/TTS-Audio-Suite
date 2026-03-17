"""
SRT builder for ASR outputs with smart readability defaults.
"""

from functools import lru_cache
from typing import List, Tuple
import re

from utils.asr.types import ASRSegment, ASRWord


SENTENCE_END_CHARS = {".", "!", "?", "…"}
SOFT_BREAK_CHARS = {",", ";", ":"}
DEFAULT_CONNECTOR_ALLOWLIST = "a,an,the,to,of,and,or,im,i'm,you,you're,we,they,he,she,it"


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
    del max_lines  # Overflow is handled by cue splitting, not silent truncation.
    words = text.split()
    if not words:
        return text

    line_groups = _group_items_into_lines(words, max_chars_per_line)
    return "\n".join(" ".join(str(word) for word in line).strip() for line in line_groups)


def _trim_trailing_soft_punctuation(text: str) -> str:
    return re.sub(r"[.,;:]+$", "", text).rstrip()


def _capitalize_segment_start(text: str) -> str:
    for idx, char in enumerate(text):
        if char.isalpha():
            return text[:idx] + char.upper() + text[idx + 1:]
    return text


def _text_of_item(item) -> str:
    return item.text if hasattr(item, "text") else str(item)


def _normalize_token(text: str) -> str:
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE).lower()


def _strip_edge_punctuation(text: str) -> str:
    return text.strip(".,!?…:;\"'()[]{}")


def _ends_sentence(text: str) -> bool:
    return bool(text) and text[-1:] in SENTENCE_END_CHARS


def _ends_soft_break(text: str) -> bool:
    return bool(text) and text[-1:] in SOFT_BREAK_CHARS


def _is_capitalizedish(text: str) -> bool:
    stripped = _strip_edge_punctuation(text)
    if not stripped:
        return False
    return stripped[0].isupper() or stripped.isupper()


def _group_items_into_lines(items, max_chars_per_line: int):
    lines: List[str] = []
    current_line = []
    for item in items:
        candidate_items = current_line + [item]
        candidate_text = " ".join(_text_of_item(x) for x in candidate_items).strip()
        if current_line and len(candidate_text) > max_chars_per_line:
            lines.append(current_line)
            current_line = [item]
            continue
        current_line = candidate_items
    if current_line:
        lines.append(current_line)
    return lines


def _items_text(items) -> str:
    return " ".join(_text_of_item(item) for item in items).strip()


def _items_duration(items, total_duration: float = 0.0, total_items: int = 0) -> float:
    if not items:
        return 0.0
    if hasattr(items[0], "start") and hasattr(items[-1], "end"):
        return max(0.0, items[-1].end - items[0].start)
    if total_items <= 0:
        return 0.0
    return total_duration * (len(items) / max(1, total_items))


def _looks_phrase_bridge(left_items, right_items) -> bool:
    if not left_items or not right_items:
        return False

    left_last = _strip_edge_punctuation(_text_of_item(left_items[-1]))
    right_first = _strip_edge_punctuation(_text_of_item(right_items[0]))
    right_second = _strip_edge_punctuation(_text_of_item(right_items[1])) if len(right_items) > 1 else ""

    if not right_first:
        return False

    if right_first.isupper() and 1 <= len(right_first) <= 4:
        return True

    if _is_capitalizedish(left_last) and _is_capitalizedish(right_first):
        return True

    if _is_capitalizedish(right_first) and right_second and _is_capitalizedish(right_second):
        return True

    return False


def _boundary_score(left_items,
                    right_items,
                    *,
                    target_chars: int,
                    min_words_per_segment: int,
                    min_segment_seconds: float,
                    connector_words,
                    sentence_bonus: float = 120.0,
                    soft_break_bonus: float = 60.0,
                    phrase_bridge_penalty: float = 90.0,
                    total_duration: float = 0.0,
                    total_items: int = 0) -> float:
    if not left_items:
        return float("-inf")

    left_text = _items_text(left_items)
    right_text = _items_text(right_items)
    left_duration = _items_duration(left_items, total_duration=total_duration, total_items=total_items)
    right_duration = _items_duration(right_items, total_duration=total_duration, total_items=total_items)
    left_count = len(left_items)
    right_count = len(right_items)

    score = 0.0

    fill_target = max(8.0, target_chars * 0.72)
    score -= abs(len(left_text) - fill_target) / 4.0

    if _ends_sentence(left_text):
        score += sentence_bonus
    elif _ends_soft_break(left_text):
        score += soft_break_bonus

    left_tail = _normalize_token(_text_of_item(left_items[-1]))
    right_head = _normalize_token(_text_of_item(right_items[0])) if right_items else ""
    if left_tail in connector_words:
        score -= 95.0
    if right_head in connector_words:
        score -= 30.0

    if left_count < max(1, min_words_per_segment):
        score -= 120.0
    if right_items and right_count < max(1, min_words_per_segment):
        score -= 110.0

    if left_duration <= min_segment_seconds:
        score -= 90.0
    if right_items and right_duration <= min_segment_seconds:
        score -= 80.0

    if right_items and not _ends_sentence(left_text) and _looks_phrase_bridge(left_items, right_items):
        score -= phrase_bridge_penalty

    if right_items:
        if len(right_text) <= max(10, int(target_chars * 0.4)):
            score -= 35.0
        if right_count <= max(2, min_words_per_segment):
            score -= 50.0

    return score


def _choose_best_overflow_split(candidate_items,
                                *,
                                max_duration: float,
                                min_duration: float,
                                max_chars: int,
                                max_cps: float,
                                punctuation_grace_chars: int,
                                min_words_per_segment: int,
                                min_segment_seconds: float,
                                minimum_score: float,
                                connector_words):
    if len(candidate_items) < 2:
        return None

    best_idx = None
    best_score = float("-inf")
    total_duration = _items_duration(candidate_items)
    total_items = len(candidate_items)

    for split_idx in range(1, len(candidate_items)):
        left_items = candidate_items[:split_idx]
        right_items = candidate_items[split_idx:]
        left_text = _items_text(left_items)
        left_duration = _items_duration(left_items)
        left_cps = len(left_text) / max(0.001, left_duration)
        hard_boundary = _ends_sentence(left_text) or _ends_soft_break(left_text)

        score = _boundary_score(
            left_items,
            right_items,
            target_chars=max_chars,
            min_words_per_segment=min_words_per_segment,
            min_segment_seconds=min_segment_seconds,
            connector_words=connector_words,
            sentence_bonus=140.0,
            soft_break_bonus=95.0,
            phrase_bridge_penalty=110.0,
            total_duration=total_duration,
            total_items=total_items,
        )

        allowed_chars = max_chars + (punctuation_grace_chars if _ends_sentence(left_text) else 0)
        if len(left_text) > allowed_chars:
            score -= 220.0 + ((len(left_text) - allowed_chars) * 8.0)
        if left_duration > max_duration:
            score -= 240.0 + ((left_duration - max_duration) * 60.0)
        if left_duration >= min_duration and left_cps > max_cps:
            if hard_boundary:
                score -= 55.0 + ((left_cps - max_cps) * 8.0)
            else:
                score -= 180.0 + ((left_cps - max_cps) * 25.0)
        if left_duration < min_duration:
            score -= 35.0 + ((min_duration - left_duration) * 40.0)

        if score > best_score:
            best_score = score
            best_idx = split_idx

    if best_idx is None or best_score < minimum_score:
        return None
    return best_idx


def _choose_display_split_groups(source_items,
                                 *,
                                 max_chars_per_line: int,
                                 max_lines: int,
                                 min_words_per_segment: int,
                                 min_segment_seconds: float,
                                 total_duration: float = 0.0,
                                 connector_words=None):
    if not source_items:
        return []

    connector_words = connector_words or {
        w.strip().lower() for w in DEFAULT_CONNECTOR_ALLOWLIST.split(",") if w.strip()
    }
    total_items = len(source_items)
    max_chars_total = max_chars_per_line * max(1, max_lines)

    @lru_cache(None)
    def solve(start_idx: int):
        if start_idx >= total_items:
            return 0.0, ()

        best_score = float("-inf")
        best_cuts = None

        for end_idx in range(start_idx + 1, total_items + 1):
            left_items = source_items[start_idx:end_idx]
            if len(_group_items_into_lines(left_items, max_chars_per_line)) > max_lines:
                break

            right_items = source_items[end_idx:]
            local_score = _boundary_score(
                left_items,
                right_items,
                target_chars=max_chars_total,
                min_words_per_segment=min_words_per_segment,
                min_segment_seconds=min_segment_seconds,
                connector_words=connector_words,
                sentence_bonus=110.0,
                soft_break_bonus=35.0,
                phrase_bridge_penalty=95.0,
                total_duration=total_duration,
                total_items=total_items,
            )
            if right_items:
                local_score -= 22.0
            next_score, next_cuts = solve(end_idx)
            total_score = local_score + next_score
            if total_score > best_score:
                best_score = total_score
                best_cuts = (end_idx,) + next_cuts

        return best_score, best_cuts or ()

    _, cuts = solve(0)
    groups = []
    start_idx = 0
    for end_idx in cuts:
        if end_idx > start_idx:
            groups.append(source_items[start_idx:end_idx])
        start_idx = end_idx
    return groups or [source_items]


def _split_segment_for_display(seg: ASRSegment,
                               max_chars_per_line: int,
                               max_lines: int,
                               min_words_per_segment: int = 1,
                               min_segment_seconds: float = 0.0) -> List[ASRSegment]:
    if not seg.text or max_lines <= 0:
        return [seg]

    source_items = seg.words if seg.words else seg.text.split()
    if not source_items:
        return [seg]

    if len(_group_items_into_lines(source_items, max_chars_per_line)) <= max_lines:
        return [seg]

    total_items = len(source_items)
    total_duration = max(0.0, seg.end - seg.start)
    connector_words = {
        w.strip().lower() for w in DEFAULT_CONNECTOR_ALLOWLIST.split(",") if w.strip()
    }
    cue_item_groups = _choose_display_split_groups(
        source_items,
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
        min_words_per_segment=min_words_per_segment,
        min_segment_seconds=min_segment_seconds,
        total_duration=total_duration,
        connector_words=connector_words,
    )

    split_segments: List[ASRSegment] = []
    consumed_items = 0
    for group_index, cue_items in enumerate(cue_item_groups):
        cue_text = " ".join(_text_of_item(item) for item in cue_items).strip()
        if not cue_text:
            consumed_items += len(cue_items)
            continue

        if seg.words:
            cue_words = cue_items
            cue_start = cue_words[0].start
            cue_end = cue_words[-1].end
        else:
            cue_words = []
            cue_item_count = len(cue_items)
            cue_start = seg.start + (total_duration * (consumed_items / max(1, total_items)))
            cue_end = seg.start + (total_duration * ((consumed_items + cue_item_count) / max(1, total_items)))
            if group_index == len(cue_item_groups) - 1:
                cue_end = seg.end

        split_segments.append(
            ASRSegment(
                start=cue_start,
                end=cue_end,
                text=cue_text,
                speaker=seg.speaker,
                words=cue_words,
            )
        )
        consumed_items += len(cue_items)

    return split_segments or [seg]


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
    connector_words = {
        w.strip().lower() for w in merge_dangling_tail_allowlist.split(",") if w.strip()
    }

    def flush_segment(end_time: float, words_override=None, text_override: str = None):
        nonlocal current_words, current_text, seg_start
        words_to_flush = words_override if words_override is not None else current_words
        text_to_flush = text_override if text_override is not None else current_text.strip()
        if not words_to_flush:
            return
        segments.append(ASRSegment(start=words_to_flush[0].start, end=end_time, text=text_to_flush, words=words_to_flush))
        if words_override is None:
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
        split_on_cps = (
            projected_cps > max_cps
            and projected_duration >= min_duration
            and len(projected_text) >= int(max_chars * 0.65)
        )

        if split_on_gap and merge_trailing_punct_word:
            if gap <= merge_trailing_punct_max_gap and is_sentence_end:
                split_on_gap = False

        if current_words and (split_on_gap or split_on_duration or split_on_chars or split_on_cps):
            soft_pressure_only = split_on_cps and not (split_on_duration or split_on_chars or split_on_gap)
            if split_on_duration or split_on_chars or split_on_cps:
                candidate_words = current_words + [word]
                split_idx = _choose_best_overflow_split(
                    candidate_words,
                    max_duration=max_duration,
                    min_duration=min_duration,
                    max_chars=max_chars,
                    max_cps=max_cps,
                    punctuation_grace_chars=punctuation_grace_chars,
                    min_words_per_segment=min_words_per_segment,
                    min_segment_seconds=min_segment_seconds,
                    minimum_score=25.0 if split_on_cps and not (split_on_duration or split_on_chars) else 0.0,
                    connector_words=connector_words,
                )
                if split_idx is not None and 0 < split_idx < len(candidate_words):
                    left_words = candidate_words[:split_idx]
                    right_words = candidate_words[split_idx:]
                    flush_segment(
                        left_words[-1].end,
                        words_override=left_words,
                        text_override=_items_text(left_words),
                    )
                    current_words = right_words
                    current_text = _items_text(right_words)
                    seg_start = right_words[0].start
                    continue
                if soft_pressure_only:
                    split_idx = None

            if not soft_pressure_only:
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
        too_short_for_reading = duration < min_duration and not _ends_sentence(seg.text)
        if (word_count < min_words_per_segment or duration <= min_segment_seconds or too_short_for_reading) and i + 1 < len(segments):
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

    def try_match(norm_tok: str, replacement: str, credit: int) -> bool:
        nonlocal word_idx, last_word_idx, matched_word_tokens
        lookahead = 4
        for j in range(word_idx, min(len(words), word_idx + lookahead)):
            if normalize(words[j].text) == norm_tok:
                words[j].text = replacement
                last_word_idx = j
                matched_word_tokens += credit
                word_idx = j + 1
                return True
        return False

    for tok in tokens:
        # Word token
        if re.match(r"[A-Za-z0-9]", tok, flags=re.UNICODE):
            sub_tokens = expand_contraction(tok)
            total_word_tokens += len(sub_tokens)
            if word_idx >= len(words):
                if len(missed_tokens) < 3:
                    missed_tokens.append(tok)
                break
            # Prefer matching the full contraction first (e.g. "I'll" -> "i'll").
            if "'" in tok and try_match(normalize(tok), tok, len(sub_tokens)):
                continue

            # Fall back to aligning each contraction part separately when the
            # timing stream splits them into multiple words.
            for sub in sub_tokens:
                if not try_match(normalize(sub), sub, 1) and len(missed_tokens) < 3:
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
              merge_allow_overlong: bool = False,
              normalize_cue_end_punctuation: bool = False):
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

    display_segments: List[ASRSegment] = []
    for seg in segments_to_use:
        display_segments.extend(
            _split_segment_for_display(
                seg,
                max_chars_per_line=max_chars_per_line,
                max_lines=max_lines,
                min_words_per_segment=min_words_per_segment,
                min_segment_seconds=min_segment_seconds,
            )
        )

    if normalize_cue_end_punctuation:
        for idx, seg in enumerate(display_segments):
            if seg.text:
                trimmed = _trim_trailing_soft_punctuation(seg.text)
                stripped_soft_punct = trimmed != seg.text
                seg.text = trimmed
                if stripped_soft_punct and idx + 1 < len(display_segments):
                    next_seg = display_segments[idx + 1]
                    if next_seg.text:
                        next_seg.text = _capitalize_segment_start(next_seg.text)

    lines: List[str] = []
    for idx, seg in enumerate(display_segments, start=1):
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
