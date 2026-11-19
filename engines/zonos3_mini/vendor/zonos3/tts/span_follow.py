"""Span steering that follows the model's reading position (``control_span_follow: "reading"``).

A request's ``control_spans`` (zonos3.tts.steering) are normally placed by character position at a frames-per-character
rate. Here they follow the alignment probe (zonos3_sgl.tts.align_probe, the LACI heads of the request's language) instead:

- The tokenizer maps every span's characters (of the request text) to the text tokens that spell them: through text
  normalization (``map_offsets``) to the normalized text, then to the SuperBPE tokens (or phoneme-group ids, or bytes)
  covering them (``span_token_ranges``), as text-span positions [a, b) (0 is BOS, ``text_len - 1`` EOS).
- While the request decodes, its probe reads the alignment heads' argmax text position once per step; the
  ``SpanFollower`` smooths it (median of the last ``smooth`` readouts, held monotone, at most +1 token per readout).
  Span k ramps in (``ramp`` frames, linear) on the first frame decided with the smoothed position at or past
  ``a - lead`` and ramps out on the first frame decided at or past ``b + lag``. Every decision is recorded per
  generated frame, so the speaker-CFG twin and a LACI rollback (``truncate``) see the same weights for the frames
  they share.

The readout of a step reaches the host with its sampled tokens, so the frame decided now sees the readouts up to one or
two steps back (overlap scheduling); ``lead`` absorbs that and the ramp. A request without probe heads for its
language keeps the character-rate schedule.
"""

from __future__ import annotations

import bisect
import difflib
import statistics
from dataclasses import dataclass
from typing import Sequence

import torch

from .steering import RAMP_FRAMES


@dataclass(frozen=True)
class FollowConfig:
    lead: float = 1.0  # tokens: ramp in once the reading reaches a span's first token - lead (README: dev-split pick)
    lag: float = 0.0  # tokens: ramp out once the reading reaches the token after the span + lag
    smooth: int = 3  # readouts in the running median
    ramp: int = RAMP_FRAMES


# --------------------------------------------------------------------------------------------------- token mapping


def map_offsets(original: str, final: str, offsets: Sequence[int], *, end: bool = False) -> list[int]:
    """Character offsets of ``original`` mapped into ``final`` (``original`` after normalization): exact inside
    unchanged stretches, proportional inside rewritten ones (rounded outward: down for starts, up when ``end``)."""
    if original == final:
        return [int(o) for o in offsets]
    # the blocks holding original characters partition [0, len(original)); insertions hold none
    blocks = [
        op for op in difflib.SequenceMatcher(None, original, final, autojunk=False).get_opcodes() if op[2] > op[1]
    ]
    starts = [op[1] for op in blocks]
    out = []
    for o in offsets:
        o = int(o)
        if o >= len(original):
            out.append(len(final))
            continue
        tag, i1, i2, j1, j2 = blocks[bisect.bisect_right(starts, o) - 1]
        if tag == "equal":
            out.append(j1 + o - i1)
        else:
            x = j1 + (o - i1) * (j2 - j1) / (i2 - i1)
            out.append(int(-(-x // 1)) if end else int(x))
    return out


def token_char_ranges(text: str, spec: dict | None) -> list[tuple[int, int]]:
    """Character range of ``text`` behind each id of its text span (``TTSPromptBuilder.text_ids``): BOS and EOS are
    the empty ranges at 0 and ``len(text)``; SuperBPE tokens their offsets, a phoneme group's open id its open marker,
    each phone id its code point, the close id the close marker, the start and pause ids their marker; byte ids the
    character they encode."""
    if spec is None:
        ranges = [(i, i + 1) for i, ch in enumerate(text) for _ in ch.encode("utf-8")]
    else:
        from .superbpe import PAUSE_MARKER_RE, START_MARKER, tokenizer

        bpe = tokenizer(spec["path"], spec["sha256"], spec["vocab_size"])

        def stretch(s: str, base: int) -> list[tuple[int, int]]:
            return [(base + a, base + b) for a, b in bpe.encode(s, add_special_tokens=False).offsets] if s else []

        def payload(s: str, base: int) -> list[tuple[int, int]]:  # superbpe.payload_ids validated the text
            if spec.get("start") is not None and START_MARKER in s:
                return [(base, base + 1), *payload(s[1:], base + 1)]
            if spec.get("pauses") is not None and PAUSE_MARKER_RE.search(s):
                out, pos = [], 0
                for match in PAUSE_MARKER_RE.finditer(s):
                    if match.start() > pos:
                        out += payload(s[pos : match.start()], base + pos)
                    out.append((base + match.start(), base + match.end()))
                    pos = match.end()
                if pos < len(s):
                    out += payload(s[pos:], base + pos)
                return out
            phonemes = spec.get("phonemes")
            if phonemes is None or (phonemes["open_marker"] not in s and phonemes["close_marker"] not in s):
                return stretch(s, base)
            open_, close = phonemes["open_marker"], phonemes["close_marker"]
            out, pos = [], 0
            while pos < len(s):
                start = s.find(open_, pos)
                out += stretch(s[pos:] if start < 0 else s[pos:start], base + pos)
                if start < 0:
                    break
                end = s.find(close, start + 1)
                out += [(base + i, base + i + 1) for i in range(start, end + 1)]
                pos = end + 1
            return out

        ranges = payload(text, 0)
    return [(0, 0), *ranges, (len(text), len(text))]


def span_token_ranges(
    text: str, final: str, spans: Sequence[tuple[int, int]], spec: dict | None
) -> list[tuple[int, int]]:
    """Text-span positions [a, b) of the tokens spelling each character span [start, end) of ``text``, once the text
    is ``final`` (normalized): the tokens whose characters, outer whitespace trimmed, overlap the span's trimmed
    characters. A span that maps onto no token keeps the token at its start (a zero-width span never steers)."""
    ranges = token_char_ranges(final, spec)
    trimmed = []
    for a, b in ranges:
        while a < b and final[a].isspace():
            a += 1
        while b > a and final[b - 1].isspace():
            b -= 1
        trimmed.append((a, b))
    out = []
    for start, end in spans:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        s, e = map_offsets(text, final, [start])[0], map_offsets(text, final, [end], end=True)[0]
        hit = [i for i, (a, b) in enumerate(trimmed) if a < b and a < e and s < b]
        if hit:
            out.append((hit[0], hit[-1] + 1))
        else:
            first = next((i for i, (a, b) in enumerate(trimmed) if a < b and b > s), len(ranges) - 1)
            out.append((first, first))
    return out


# --------------------------------------------------------------------------------------------------------- follower


class SpanFollower:
    """The span gate of one request (see the module docstring). ``observe(frame, pos)`` takes the readout of the step
    that generated ``frame`` (frames count the stream's generated rows: a LACI continuation's start at its
    history_rows); ``delta(frame)`` decides and returns the summed span delta for the frame generated now (None when
    no span is on); ``truncate(frame)`` forgets the readouts and decisions of frames >= ``frame`` (a rollback).

    As a probe observer (``on_step``) it maps step s of the probe to frame ``origin + s``."""

    def __init__(
        self,
        ranges: Sequence[tuple[int, int]],
        deltas: torch.Tensor,
        cfg: FollowConfig = FollowConfig(),
        origin: int = 0,
    ):
        if deltas.dim() != 2 or deltas.shape[0] != len(ranges):
            raise ValueError(f"span follower: {len(ranges)} token ranges for deltas of shape {tuple(deltas.shape)}")
        if cfg.smooth < 1 or cfg.ramp < 1:
            raise ValueError("span follower: smooth and ramp must be >= 1")
        self.ranges = [(int(a), int(b)) for a, b in ranges]
        self.deltas = deltas.to(torch.float32)
        self.cfg = cfg
        self.origin = int(origin)
        self._frames: list[int] = []  # observed frames (increasing) and their readouts
        self._pos: list[int] = []
        self._best = 0.0  # smoothed, monotone reading position (0 = BOS before any readout)
        k = len(self.ranges)
        self._enter = [None] * k  # (decision frame, ramp anchor) per span
        self._exit = [None] * k
        self._weights: dict[int, list[float]] = {}  # frame -> per-span weights decided for it

    # -- readout -------------------------------------------------------------------------------------------------

    def on_step(self, probe, step: int, pos: int, m: float) -> None:
        self.observe(self.origin + step, pos)

    def observe(self, frame: int, pos: int) -> None:
        if self._frames and frame <= self._frames[-1]:
            raise ValueError(f"span follower: frames must increase, got {frame} after {self._frames[-1]}")
        self._frames.append(int(frame))
        self._pos.append(int(pos))
        self._advance(len(self._pos))

    def _advance(self, n: int) -> None:
        med = statistics.median(self._pos[max(0, n - self.cfg.smooth) : n])
        self._best = min(max(self._best, med), self._best + 1.0)

    @property
    def position(self) -> float:
        return self._best

    # -- decisions -----------------------------------------------------------------------------------------------

    def weights(self, frame: int) -> list[float]:
        frame = int(frame)
        w = self._weights.get(frame)
        if w is not None:
            return w
        cfg, s = self.cfg, self._best
        for k, (a, b) in enumerate(self.ranges):
            if a >= b:
                continue
            if self._enter[k] is None and s >= a - cfg.lead:
                # on before any readout (a span at the start of the text): full weight at once, like the schedule
                self._enter[k] = (frame, frame - cfg.ramp + 1 if not self._frames else frame)
            if self._enter[k] is not None and self._exit[k] is None and s >= b + cfg.lag:
                self._exit[k] = (frame, frame)
        w = []
        for k in range(len(self.ranges)):
            if self._enter[k] is None:
                w.append(0.0)
                continue
            up = (frame - self._enter[k][1] + 1) / cfg.ramp
            down = 1.0 if self._exit[k] is None else (self._exit[k][1] + cfg.ramp - 1 - frame) / cfg.ramp
            w.append(min(1.0, max(0.0, min(up, down))))
        self._weights[frame] = w
        return w

    def delta(self, frame: int) -> torch.Tensor | None:
        w = self.weights(frame)
        if not any(x > 0 for x in w):
            return None
        return (torch.tensor(w, dtype=torch.float32)[:, None] * self.deltas).sum(0)

    def truncate(self, frame: int) -> None:
        frame = int(frame)
        keep = bisect.bisect_left(self._frames, frame)
        del self._frames[keep:], self._pos[keep:]
        self._best = 0.0
        for n in range(1, keep + 1):
            self._advance(n)
        for events in (self._enter, self._exit):
            for k, e in enumerate(events):
                if e is not None and e[0] >= frame:
                    events[k] = None
        self._weights = {f: w for f, w in self._weights.items() if f < frame}
