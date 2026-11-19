"""LACI -- Localized Attention-Constrained Inference (arXiv 2609.16989) for ZONOS3 Mini.

While a request decodes, the alignment heads' attention over the text (zonos3_sgl.tts.align_probe) is reduced to a reading
position per step and fed to a ``Detector``. A skip (the reading jumps past a token bin that got far fewer steps than
the bins before it), a hallucination (the reading stops advancing) or a truncation (EOS with two or more bins unread)
rolls the request back to the failure onset; it is regenerated from there with a new seed under an ACI hard attention
mask (arXiv 2404.19723) until the reading passes the detection point, at most ``max_retries`` times. The text is done
once the last bin has been read for its share and the reading has reached the last text token; generation stops
``tail_frames`` later, and ``stop_boost_delay`` rows after the text is done the sampler raises the stop id's logit by
``stop_boost`` (``LaciSession.stop_boost``) so a take that would run on past the text ends at its next likely stop.

The scheduler runs this for requests whose ``TTSUserMsg.laci`` names an alignment-head spec (``LaciHeads``), offline
(``TTSLLM.generate(laci=...)``) and in the server (``laci: true`` on the request) alike: a ``LaciSession`` per request
owns the stream, a rollback becomes a continuation request (the prompt plus the kept rows, re-prefilled) under the same
uid, and only rows no later rollback can change are handed downstream (``LaciSession.drain``), so the detokenizer and the
streamed response see one uninterrupted stream; a request decoded whole at the end (``stream=false`` or ``stream_decode``
whole) holds every row until it is final (``hold``). Constraints: no CFG twin (speaker and controls CFG scale 1), the FA
attention backend, tensor parallel size 1.

A short text (zonos3.tts.short_text) runs under LACI as the text, a pause marker and a filler sentence: the session
(``short_text``: the filler's first text token) holds the take, ends it at the cut once the reading reaches the
filler (``end`` "short_text"), regenerates a take that ran into the filler as a fresh one (failure "short_run_on")
and, once those retries are spent, cuts where the reading reached the filler ("short_text_forced").

Vocabulary
  row       stream index: the request's prefill rows (codec silence) are 0..P-1, generated rows follow.
  pos       text-span index 0..text_len-1; 0 is BOS, text_len-1 is EOS.
  bin       b(pos) = pos // bin_tokens; nb = ceil(text_len / bin_tokens).
  share(b)  frames bin b is expected to take: pace_tok() * tokens_b, where pace_tok() is the median frames/token of the
            last pace_window_bins bins below the frontier (the prior until warmup_bins exist).
  frontier  highest bin that has been read for at least cover_frac * share.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import torch

Head = tuple[int, int]  # (layer_id, q_head)


@dataclass(frozen=True)
class LaciConfig:
    bin_tokens: int = 8
    cover_frac: float = 0.5
    skip_frac: float = 0.15
    stall_factor: float = 3.0
    min_stall_frames: int = 45
    tail_frames: int = 15
    rollback_margin_frames: int = 8
    max_retries: int = 8
    pace_window_bins: int = 16
    warmup_bins: int = 2
    # Once the text is done and stop_boost_delay more rows were sampled, the sampler adds stop_boost to the stop id's
    # logit in every codebook (before the sampling filters): a take that would run on past the text (extra words,
    # mostly with speaker embeddings, whose stop probability there is ~0.02 instead of ~1) ends at its next likely
    # stop. 0 turns it off.
    stop_boost: float = 3.0
    stop_boost_delay: int = 6


@dataclass(frozen=True)
class Failure:
    kind: str  # "skip" | "hallucination" | "truncation"
    detect_row: int
    onset_row: int
    rollback_row: int
    bin: int  # skip: first skipped bin; hallucination/truncation: first unread bin
    frontier: int  # skip: the bin the reading jumped to; otherwise the frontier


class Detector:
    """Streaming skip / hallucination / truncation detector.

    Feed one (row, pos) per decode step with strictly increasing rows. After a failure the caller rolls back with
    truncate(rollback_row) and re-observes from that row. A returned failure leaves the detector state unchanged except
    for the recorded observation; callers that ignore failures (retry budget spent) will see the same failure again on
    later rows.
    """

    def __init__(self, text_len, frames_per_token_prior, cfg, min_rollback_row):
        if text_len < 1:
            raise ValueError(f"text_len must be >= 1, got {text_len}")
        self.text_len = int(text_len)
        self.prior = float(frames_per_token_prior)
        self.cfg = cfg
        self.min_rollback_row = int(min_rollback_row)
        self.nb = -(-self.text_len // cfg.bin_tokens)
        self._obs_rows = []
        self._obs_pos = []
        self._bumps = []  # rows of the truncate() calls that still lie inside the kept observations
        self._reset()

    # -- state ---------------------------------------------------------------

    def _reset(self):
        self._counts = [0] * self.nb
        self._first_row = [None] * self.nb
        self._frontier = 0
        self._last_advance_row = None
        self._done_row = None
        self._read_row = None  # first row the reading was on the last text token (EOS)

    def _tokens(self, b):
        return min(self.cfg.bin_tokens, self.text_len - b * self.cfg.bin_tokens)

    @property
    def frontier(self):
        return self._frontier

    @property
    def counts(self):
        return list(self._counts)

    @property
    def last_advance_row(self):
        return self._last_advance_row

    @property
    def done_row(self):
        return self._done_row

    def pace_tok(self):
        """Median frames/token over the last pace_window_bins bins below the frontier."""
        lo = max(0, self._frontier - self.cfg.pace_window_bins)
        rates = [self._counts[x] / self._tokens(x) for x in range(lo, self._frontier)]
        if len(rates) < self.cfg.warmup_bins:
            return self.prior
        return statistics.median(rates)

    def _share(self, b):
        return self.pace_tok() * self._tokens(b)

    def last_pos_before(self, row):
        """Last observed pos at a row < `row`; 0 if there is none."""
        i = bisect.bisect_left(self._obs_rows, row)
        return self._obs_pos[i - 1] if i else 0

    def stop_row(self):
        if self._done_row is None:
            return None
        return self._done_row + self.cfg.tail_frames

    def commit_row(self, next_row):
        """Every skip or hallucination this detector can still report (observing rows >= ``next_row``) rolls back to a
        row >= the returned one. Truncation (``on_eos``) is bounded by the caller, which knows the EOS row.

        Hallucination onsets are the stall clock ``last_advance_row``, which never decreases. Skip onsets are the first
        row seen in a bin above the skipped one, and a skipped bin lies above the frontier, so the onset is the first
        row of a bin >= frontier + 2 (seen already, or a future row). Once the text is done neither can occur."""
        if self._done_row is not None:
            return math.inf
        onsets = [next_row if self._last_advance_row is None else self._last_advance_row, next_row]
        onsets += [r for r in self._first_row[self._frontier + 2 :] if r is not None]
        return self._rollback(min(onsets))

    # -- observation ---------------------------------------------------------

    def _rollback(self, onset):
        return max(onset - self.cfg.rollback_margin_frames, self.min_rollback_row)

    def observe(self, row, pos):
        if not 0 <= pos < self.text_len:
            raise ValueError(f"pos {pos} outside text span [0, {self.text_len})")
        if self._obs_rows and row <= self._obs_rows[-1]:
            raise ValueError(f"rows must increase: got {row} after {self._obs_rows[-1]}")
        self._obs_rows.append(row)
        self._obs_pos.append(pos)
        return self._step(row, pos)

    def _step(self, row, pos):
        cfg = self.cfg
        # 1. count
        b = pos // cfg.bin_tokens
        self._counts[b] += 1
        if self._first_row[b] is None:
            self._first_row[b] = row
        if self._last_advance_row is None:
            self._last_advance_row = row

        # 2. frontier / skip
        if b > self._frontier and self._counts[b] >= cfg.cover_frac * self._share(b):
            for x in range(self._frontier + 1, b):
                if self._counts[x] < cfg.skip_frac * self._share(x):
                    onset = min(self._first_row[y] for y in range(x + 1, self.nb) if self._first_row[y] is not None)
                    return Failure(
                        "skip", detect_row=row, onset_row=onset, rollback_row=self._rollback(onset), bin=x, frontier=b
                    )
            self._frontier = b
            self._last_advance_row = row

        # 3. stall
        if self._done_row is None:
            limit = max(cfg.stall_factor * self.pace_tok() * cfg.bin_tokens, cfg.min_stall_frames)
            if row - self._last_advance_row > limit:
                onset = self._last_advance_row
                return Failure(
                    "hallucination",
                    detect_row=row,
                    onset_row=onset,
                    rollback_row=self._rollback(onset),
                    bin=self._frontier + 1,
                    frontier=self._frontier,
                )

        # 4. text done: the last bin read for its share, and the reading has reached the last text token (with two
        # bins the share alone, a pace prior, can call the text read a second before the end is spoken)
        if self._read_row is None and pos >= self.text_len - 1:
            self._read_row = row
        if (
            self._done_row is None
            and self._read_row is not None
            and self._frontier >= self.nb - 2
            and self._counts[self.nb - 1] >= self._share(self.nb - 1)
        ):
            self._done_row = row
        return None

    def on_eos(self, row):
        """Model emitted EOS at `row`: truncation iff two or more bins are unread."""
        if self._frontier < self.nb - 2:
            return Failure(
                "truncation",
                detect_row=row,
                onset_row=row,
                rollback_row=self._rollback(row),
                bin=self._frontier + 1,
                frontier=self._frontier,
            )
        return None

    def truncate(self, row):
        """Drop observations at rows >= row and rebuild all derived state.

        The stall clock of what survives is moved up to `row`, so the regeneration gets a full stall budget. That
        bump is part of the detector's history: each earlier truncate() whose row is still inside the kept
        observations is re-applied at the same point of the replay, so repeated truncations replay without a stall
        the live run never saw."""
        keep = bisect.bisect_left(self._obs_rows, row)
        rows, poss = self._obs_rows[:keep], self._obs_pos[:keep]
        bumps = [r for r in self._bumps if r < row]
        self._obs_rows, self._obs_pos = [], []
        self._reset()
        pending = list(bumps)
        for r, p in zip(rows, poss):
            while pending and r > pending[0]:
                if self._last_advance_row is not None:
                    self._last_advance_row = max(self._last_advance_row, pending[0])
                pending.pop(0)
            self._obs_rows.append(r)
            self._obs_pos.append(p)
            failure = self._step(r, p)
            assert failure is None, f"replay raised {failure}"
        if self._last_advance_row is not None:
            self._last_advance_row = max(self._last_advance_row, row)
        self._bumps = bumps + [row]


class AciCenter:
    """ACI dynamic-programming attention centre (2404.19723, Eq. 2/5).

    d_t[l] = min(d_{t-1}[l], d_{t-1}[l-1]) + (m_t - l)^2 over the monotone paths that advance 0 or 1 text index per
    step, starting at c0 with cost 0. The support grows by one index per step, capped at text_len-1. The centre is
    argmin d_t (ties to the smaller index), held monotone and limited to +1 per update so the mask window never
    retreats or jumps.
    """

    def __init__(self, c0, text_len):
        self.text_len = int(text_len)
        self.c0 = min(max(int(c0), 0), self.text_len - 1)
        self._d = [0.0]  # d[i] is the cost of index c0 + i
        self.center = self.c0

    def update(self, m):
        d = self._d
        if self.c0 + len(d) < self.text_len:
            d.append(math.inf)
        for i in range(len(d) - 1, 0, -1):
            d[i] = min(d[i], d[i - 1]) + (m - (self.c0 + i)) ** 2
        d[0] += (m - self.c0) ** 2
        best = 0
        for i in range(1, len(d)):
            if d[i] < d[best]:
                best = i
        self.center = min(max(self.c0 + best, self.center), self.center + 1)
        return self.center


# ------------------------------------------------------------------------------------------------- request driver


@dataclass(frozen=True)
class LaciHeads:
    """Alignment heads of one model and language (the JSON files in assets/laci_heads/)."""

    heads: tuple[Head, ...]
    radius: dict[Head, int]  # ACI mask half-width per head
    frames_per_token: float  # pace prior until the reading has covered warmup_bins bins

    @classmethod
    def load(cls, path: str | Path) -> "LaciHeads":
        spec = json.loads(Path(path).read_text())
        if not spec.get("heads"):
            raise ValueError(f"{path}: no alignment heads")
        heads = tuple((int(h["layer"]), int(h["head"])) for h in spec["heads"])
        radius = {(int(h["layer"]), int(h["head"])): int(h["radius"]) for h in spec["heads"]}
        return cls(heads, radius, float(spec["frames_per_token"]))


# ------------------------------------------------------------------------------------------------- duration targets
#
# A LACI request may also carry a ``DurationTarget``: the take's length is then held to it (a duration-target take,
# zonos3.server.plugins.LaciTakeRequest, fits a segment this way). The reading position reaches the last text token
# ~0.2 s before the speech ends, so (row of that moment - first generated row) / target estimates the take's duration
# ratio; mid-text it is too uneven per token to judge (calibrated on 124 en->fr duration-target takes, 2026-10-28). A
# take that reads the text too early, or has not read it PACE_TOLERANCE * target + PACE_SLACK_FRAMES frames after its
# target, is rolled back to its first row and regenerated in the speaking-rate bucket the ratio asks for (the
# continuation's condition rows), at most PACE_MAX_RETRIES times; with ``adjust_rate`` off a miss is only recorded.
#
# Two content checks also follow the target. A truncation (EOS with text unread) of a rate-adjustable take regenerates
# one speaking-rate bucket slower: in the fastest bucket the model often ends a short text within a few frames, and
# reseeding in the same bucket repeated that up to the retry cap (2026-10-29 take log: 511 of 575 truncations in the
# first text bin; bucket 9 looped on 22 of 40 seeded takes where buckets 5-8 never did). And the PaceMonitor flags a
# stall once the reading has not advanced for stall_factor text bins at the target pace (at least
# DURATION_MIN_STALL_FRAMES): the detector's limit (stall_factor bins at the observed pace, at least min_stall_frames
# = 3 s) outlasts most duration-target segments, so a stall inside one only surfaced as a late pace_slow retake in a
# faster bucket instead of a rollback to its onset. It watches the smoothed position itself, not the detector's bins:
# until two bins are read (all of a short text) the detector's bins follow the heads' average pace, which
# duration-target takes read ~2x faster, so a shorter bin-based limit fired on clean takes (40 of 40 stalls in a test
# run), and a target-pace prior for the bins moved LACI's text-done stop early enough to cut last words.
PACE_TOLERANCE = 0.10
PACE_SLACK_FRAMES = 3
PACE_SMOOTH_STEPS = 5
PACE_MAX_RETRIES = 2
PACE_RATIO_RANGE = (0.5, 2.0)  # clamp of the duration-ratio estimate a retry bucket is computed from
PACE_KINDS = ("pace_slow", "pace_fast")
DURATION_MIN_STALL_FRAMES = 15


@dataclass(frozen=True)
class DurationTarget:
    frames: float  # target length of the take, in generated rows
    adjust_rate: bool = True  # on a miss, regenerate in the speaking-rate bucket the measured ratio asks for


@dataclass(frozen=True)
class RateBuckets:
    """A speaking-rate scale: the checkpoint's speaking-rate condition (its condition-row index and ``n_buckets`` row
    buckets; id ``n_buckets`` is "unknown") and the scale's bucket edges (UTF-8 bytes / s): bucket i, ``first`` <= i <=
    ``last``, is [edges[i - first - 1], edges[i - first]). The row's own scale is first = 0, last = n_buckets - 1; an
    extended scale (a duration-target take's, LaciTakeRequest.rate_edges) adds buckets below 0 / above n_buckets - 1
    that are the rate row unknown plus a pace-steering dose (the take's per-bucket steering delta)."""

    index: int
    edges: tuple[float, ...]
    n_buckets: int
    first: int = 0

    @property
    def last(self) -> int:
        return self.first + len(self.edges)

    def row(self, bucket: int) -> int:
        """Condition-row id of a bucket: itself for a row bucket, unknown for a steered one."""
        return bucket if 0 <= bucket < self.n_buckets else self.n_buckets

    def rate(self, bucket: int) -> float:
        """Representative rate of a bucket: its midpoint; the open end buckets extend by half a neighbour."""
        e, i = self.edges, bucket - self.first
        if i <= 0:
            return e[0] - (e[1] - e[0]) / 2
        if i >= len(e):
            return e[-1] + (e[-1] - e[-2]) / 2
        return (e[i - 1] + e[i]) / 2

    def retry_bucket(self, bucket: int, ratio: float) -> int | None:
        """Bucket of a pace retry: the rate that would have made the take's duration ratio 1 (``ratio`` > 1: faster),
        at least one bucket away from ``bucket``; None when that leaves ``[first, last]``."""
        want = self.first + bisect.bisect_right(self.edges, self.rate(bucket) * ratio)
        want = max(want, bucket + 1) if ratio > 1 else min(want, bucket - 1)
        return want if self.first <= want <= self.last else None


@dataclass(frozen=True)
class PaceFailure:
    """A duration miss; carries the attributes the failure records and rollbacks read."""

    kind: str  # "pace_slow" (text not read by the deadline) | "pace_fast" (text read too early)
    detect_row: int
    onset_row: int
    rollback_row: int
    bin: int
    frontier: int
    ratio: float  # estimated take duration / target duration


class PaceMonitor:
    """Duration check over the reading position (see PACE_TOLERANCE).

    ``observe`` takes one readout per decode step. The text counts as read once the median of the last
    PACE_SMOOTH_STEPS positions reaches the last text token (``text_len - 2``; ``text_len - 1`` is EOS) or the detector
    says so. Reading it before ``target * (1 - PACE_TOLERANCE) - PACE_SLACK_FRAMES`` frames is ``pace_fast``; not having
    read it ``target * (1 + PACE_TOLERANCE) + PACE_SLACK_FRAMES`` frames in is ``pace_slow``, whose ratio is
    extrapolated from the share of the text read so far. Both roll back to ``min_rollback_row`` (a fresh take).

    It also watches for stalls (see DURATION_MIN_STALL_FRAMES): before the text is read, a smoothed position that has
    not advanced for ``stall_frames`` is a ``hallucination`` (a detector ``Failure``) with its onset at the last
    advance, rolled back by ``cfg.rollback_margin_frames`` like the detector's own.
    """

    def __init__(self, target_frames: float, origin_row: int, min_rollback_row: int, cfg: LaciConfig = LaciConfig()):
        if target_frames <= 0:
            raise ValueError(f"target_frames must be > 0, got {target_frames}")
        self.target = float(target_frames)
        self.origin = int(origin_row)
        self.min_rollback_row = int(min_rollback_row)
        self.margin = PACE_TOLERANCE * self.target + PACE_SLACK_FRAMES
        self.cfg = cfg
        self.ignored: set[str] = set()  # kinds already recorded as ignored
        self.read_row: int | None = None  # row at which the text counted as read
        self._rows: list[int] = []
        self._pos: list[int] = []
        self._best: float | None = None  # furthest smoothed position so far, and the row it was first reached
        self._best_row: int | None = None

    def stall_frames(self, text_len: int) -> float:
        """Rows without an advance that make a stall: stall_factor text bins at the target pace, at least
        DURATION_MIN_STALL_FRAMES."""
        per_token = self.target / max(1, text_len)
        return max(self.cfg.stall_factor * self.cfg.bin_tokens * per_token, DURATION_MIN_STALL_FRAMES)

    def observe(self, row: int, pos: int, text_len: int, text_done: bool = False, frontier: int = 0):
        if self._rows and row <= self._rows[-1]:
            raise ValueError(f"rows must increase: got {row} after {self._rows[-1]}")
        self._rows.append(int(row))
        self._pos.append(int(pos))
        if self.read_row is not None:
            return None  # the tail
        smooth = statistics.median(self._pos[-PACE_SMOOTH_STEPS:])
        if self._best is None or smooth > self._best:
            self._best, self._best_row = smooth, int(row)
        last = text_len - 2
        elapsed = row - self.origin
        if smooth >= last or text_done:
            self.read_row = int(row)
            if elapsed < self.target - self.margin:
                return self._failure("pace_fast", row, frontier, elapsed / self.target)
            return None
        if row - self._best_row > self.stall_frames(text_len):
            onset = self._best_row
            return Failure(
                "hallucination",
                detect_row=int(row),
                onset_row=onset,
                rollback_row=max(onset - self.cfg.rollback_margin_frames, self.min_rollback_row),
                bin=int(frontier) + 1,
                frontier=int(frontier),
            )
        if elapsed > self.target + self.margin:
            share = (smooth - self._pos[0]) / max(1, last - self._pos[0])  # of the text, read so far
            lower = elapsed / self.target  # the take is at least this long
            ratio = lower / share if share > 0 else PACE_RATIO_RANGE[1]
            return self._failure("pace_slow", row, frontier, max(ratio, lower))
        return None

    def _failure(self, kind, row, frontier, ratio):
        lo, hi = PACE_RATIO_RANGE
        return PaceFailure(
            kind,
            detect_row=int(row),
            onset_row=self.min_rollback_row,
            rollback_row=self.min_rollback_row,
            bin=int(frontier),
            frontier=int(frontier),
            ratio=float(min(hi, max(lo, ratio))),
        )

    def truncate(self, row: int) -> None:
        """Drop the observations at rows >= ``row`` (the stream was rolled back there)."""
        keep = bisect.bisect_left(self._rows, row)
        del self._rows[keep:], self._pos[keep:]
        if self.read_row is not None and self.read_row >= row:
            self.read_row = None
        self._best = self._best_row = None
        for k, r in enumerate(self._rows):
            smooth = statistics.median(self._pos[max(0, k - PACE_SMOOTH_STEPS + 1) : k + 1])
            if self._best is None or smooth > self._best:
                self._best, self._best_row = smooth, r
        if self._best_row is not None:
            self._best_row = max(self._best_row, int(row))  # the regeneration gets a full stall budget


# ---------------------------------------------------------------------------------------------------------- session


@dataclass(frozen=True)
class Continuation:
    """A rolled-back request's regeneration: the original prompt (without its prefill rows) plus ``rows``."""

    rows: torch.Tensor  # stream rows [0, resume_row): the prefill rows and the kept generated rows
    history_rows: int  # rows of ``rows`` past the prefill rows (repetition-penalty history)
    seed: int | None
    max_tokens: int
    condition_values: list[int] | None = None  # the condition-row ids to regenerate under (None: the prompt's)
    # the steering delta to regenerate under (the bucket's, on an extended rate scale); only when ``swap_delta``
    steering_delta: torch.Tensor | None = None
    swap_delta: bool = False


def _failure_record(
    f, retry_seed, ignored: bool, resume_row: int | None = None, retry_bucket: int | None = None
) -> dict:
    """``rollback_row`` is where the regeneration resumed: the failure's own rollback row, or the first unsent row when
    that lies above it (``clamped``). ``retry_bucket``: the speaking-rate bucket the regeneration switched to."""
    rollback = f.rollback_row if resume_row is None else int(resume_row)
    return {
        "kind": f.kind,
        "detect_row": f.detect_row,
        "onset_row": f.onset_row,
        "rollback_row": rollback,
        "clamped": rollback != f.rollback_row,
        "bin": f.bin,
        "frontier": f.frontier,
        "retry_seed": retry_seed,
        "ignored": ignored,
        "retry_bucket": retry_bucket,
    }


@dataclass(eq=False)
class LaciSession:
    """One request under LACI: its stream, detector, retries and ACI probation (see the module docstring).

    The scheduler calls, per attempt: ``probe_mask()`` (the probe windows) when it prefills the attempt's request;
    the probe's observer (``on_step``) after every decode step; ``append(row, eos_frame)`` for every row the request
    emits and then ``drain()`` for the rows that are final; ``finish_attempt()`` when the request finishes, which
    returns a ``Continuation`` to regenerate or None when the result is final (then ``drain(final=True)``,
    ``eos_frame()``, ``record()``).

    With a ``duration`` target a ``PaceMonitor`` watches the same readout (see PACE_TOLERANCE): ``rates`` and
    ``condition_values`` (the request's condition-row ids) let a miss regenerate in another speaking-rate bucket. On an
    extended scale (``rates.first`` < 0 or ``rates.last`` >= ``rates.n_buckets``), ``bucket`` is the take's bucket
    (its row id is unknown in a steered one) and ``rate_deltas[b - rates.first]`` the request's steering delta in
    bucket b, which a regeneration in another bucket switches to."""

    heads: LaciHeads
    cfg: LaciConfig
    prefill_rows: torch.Tensor  # [P, frame width] rows that open the stream (codec silence); may be empty
    seed: int | None
    max_tokens: int
    n_codebooks: int
    # a whole decode: nothing goes downstream before the result is final, so no rollback is ever clamped
    hold: bool = False
    stream: list = field(default_factory=list)  # 1D int32 frames: the prefill rows, then the generated rows
    attempt_start: int = 0  # stream row of the current attempt's first generated row
    emitted: int = 0  # generated rows handed downstream (final)
    detector: Detector | None = None
    text_len: int = 0
    attempts: int = 0
    failures: list = field(default_factory=list)
    probation: tuple | None = None  # (Failure, AciCenter) while the ACI mask is on
    pending_failure: Failure | None = None
    text_done: bool = False
    eos_row: int | None = None  # current attempt's stream EOS row, once the model emitted EOS
    generated_total: int = 0
    end: tuple | None = None  # (kind, stream eos row or None) once final
    _ignored: set = field(default_factory=set)
    duration: DurationTarget | None = None
    rates: RateBuckets | None = None
    condition_values: list[int] | None = None  # the current attempt's condition-row ids
    bucket: int | None = None  # the current attempt's speaking-rate bucket (default: its condition-row id)
    rate_deltas: list | None = None  # per bucket of ``rates``: the request's steering delta in that bucket
    pace: PaceMonitor | None = None
    pace_retries: int = 0
    # span steering by reading position (zonos3.tts.span_follow.SpanFollower): sees every readout, rolled back with
    # the stream
    follower: object | None = None
    # a short text (zonos3.tts.short_text): the filler's first text-span token; None = an ordinary request
    short_text: int | None = None
    short: object | None = None  # its ShortCut
    short_retries: int = 0

    def __post_init__(self):
        self.stream = list(self.prefill_rows.to(torch.int32))
        self.attempt_start = self.n_prefill
        if self.duration is not None:
            self.pace = PaceMonitor(self.duration.frames, self.n_prefill, self.n_prefill, self.cfg)
            if self.duration.adjust_rate and (self.rates is None or self.condition_values is None):
                raise ValueError("a duration target with adjust_rate needs the checkpoint's speaking-rate condition")
        if self.bucket is None and self.rates is not None and self.condition_values is not None:
            self.bucket = int(self.condition_values[self.rates.index])
        if self.rate_deltas is not None and len(self.rate_deltas) != self.rates.last - self.rates.first + 1:
            raise ValueError(
                f"rate_deltas: {len(self.rate_deltas)} for {self.rates.last - self.rates.first + 1} buckets"
            )
        if self.short_text is not None:
            from .short_text import ShortCut

            self.short = ShortCut(int(self.short_text), self.n_codebooks)

    @property
    def rate_bucket(self) -> int | None:
        if self.rates is None or self.condition_values is None:
            return None
        return self.bucket

    @property
    def n_prefill(self) -> int:
        return int(self.prefill_rows.shape[0])

    # -- per attempt -----------------------------------------------------------------------------------------------

    def window(self, center: int) -> dict:
        return {h: (center - r + 1, center + r - 1) for h, r in self.heads.radius.items()}

    def probe_mask(self) -> dict[Head, tuple[int, int]] | None:
        """Text windows of the next attempt's probe: the ACI window around the probation center, else None."""
        return self.window(self.probation[1].center) if self.probation is not None else None

    def stop_boost(self) -> float:
        """The logit the sampler adds to the stop id (every codebook) for the next row (stream row ``len(stream)``):
        ``cfg.stop_boost`` once the text is done and ``cfg.stop_boost_delay`` rows followed, else 0."""
        det = self.detector
        if det is None or det.done_row is None or len(self.stream) < det.done_row + self.cfg.stop_boost_delay:
            return 0.0
        return float(self.cfg.stop_boost)

    def on_step(self, probe, step, pos, m) -> None:
        """Observer: one decode step of the current attempt, the reading position ``pos`` and centroid ``m``."""
        row = self.attempt_start + step
        if self.follower is not None:
            self.follower.observe(row - self.n_prefill, pos)
        if self.detector is None:
            self.text_len = probe.text_len
            self.detector = Detector(probe.text_len, self.heads.frames_per_token, self.cfg, self.n_prefill)
        det = self.detector
        if self.probation is not None:
            failure, aci = self.probation
            probe.mask = self.window(aci.update(m))
            if det.frontier > failure.frontier:
                self.probation, probe.mask = None, None
        f = det.observe(row, pos)
        if f is not None:
            if self.attempts < self.cfg.max_retries:
                self.pending_failure, probe.stop = f, True
                return
            self._ignore(f)
        if self.short is not None:
            # the cut, decided over the rows so far and this reading (one row late when a row decided it)
            self.short.observe_reading(row - self.n_prefill, pos)
            if self.short.decided is not None:
                probe.stop = True
                return
        stop_row = det.stop_row()
        # keep the rows the delayed codebooks of the last kept frame need (stop_row + n_codebooks - 1), with a margin
        if stop_row is not None and row >= stop_row + self.n_codebooks + 2:
            self.text_done, probe.stop = True, True
        if self.pace is not None:
            self._pace_step(probe, row, pos)

    def _rate_adjustable(self) -> bool:
        return self.duration is not None and self.duration.adjust_rate

    def _pace_step(self, probe, row: int, pos: int) -> None:
        det = self.detector
        f = self.pace.observe(row, pos, probe.text_len, text_done=det.done_row is not None, frontier=det.frontier)
        if f is None or probe.stop:  # in time, or the attempt already ends (text read)
            return
        if not isinstance(f, PaceFailure):  # a stall: a content failure like the detector's
            if self.attempts < self.cfg.max_retries:
                self.pending_failure, probe.stop = f, True
            else:
                self._ignore(f)
        elif self._pace_retry_bucket(f) is not None:
            self.pending_failure, probe.stop = f, True
        elif f.kind not in self.pace.ignored:
            self.pace.ignored.add(f.kind)
            self.failures.append(_failure_record(f, None, True))

    def _pace_retry_bucket(self, f: PaceFailure) -> int | None:
        """Speaking-rate bucket a pace failure would regenerate in; None when it is only recorded."""
        if not self.duration.adjust_rate or self.pace_retries >= PACE_MAX_RETRIES:
            return None
        if self.attempts >= self.cfg.max_retries:
            return None
        return self.rates.retry_bucket(self.rate_bucket, f.ratio)

    def _pace_open(self) -> bool:
        """Whether a pace failure can still roll the whole attempt back (then every generated row is held)."""
        return (
            self.pace is not None
            and self.duration.adjust_rate
            and self.pace.read_row is None
            and self.pace_retries < PACE_MAX_RETRIES
            and self.attempts < self.cfg.max_retries
        )

    def _ignore(self, f: Failure) -> None:
        key = (f.kind, f.bin, f.onset_row)
        if key not in self._ignored:
            self._ignored.add(key)
            self.failures.append(_failure_record(f, None, True))

    def append(self, row: torch.Tensor, eos_frame: int | None) -> None:
        """One row the current attempt's request emitted; ``eos_frame`` is the request's aligned EOS frame (counted in
        its own generated rows) once the model emitted EOS."""
        self.stream.append(row.to(torch.int32))
        self.generated_total += 1
        if eos_frame is not None:
            self.eos_row = self.attempt_start + int(eos_frame)
        if self.short is not None:
            self.short.observe_row(row)

    def commit_row(self) -> int:
        """Stream rows below this go downstream: they are outside every rollback the current detector state can still
        report (``finish_attempt`` keeps any deeper rollback above them). The last row is always held back so the
        final message has a row to carry. With ``hold`` nothing goes downstream before the result is final."""
        n = len(self.stream)
        if self.end is not None:
            return n
        if self.hold or self.short is not None:  # a short text is held until it is cut
            return self.n_prefill
        if self.attempts >= self.cfg.max_retries:
            row = n
        elif self._pace_open():
            row = self.n_prefill
        else:
            # truncation: EOS at a stream row >= the EOS-bearing row - (n_codebooks - 1); not yet seen -> row >= n
            eos = self.eos_row if self.eos_row is not None else n - (self.n_codebooks - 1)
            row = max(eos - self.cfg.rollback_margin_frames, self.n_prefill)
            if self.detector is not None:
                row = min(row, self.detector.commit_row(n))
            else:
                row = self.n_prefill
        return max(self.n_prefill, min(row, n - 1))

    def drain(self, final: bool = False) -> list[torch.Tensor]:
        """Generated rows that became final since the last drain (all remaining ones when ``final``)."""
        upto = len(self.stream) if final else self.commit_row()
        start = self.n_prefill + self.emitted
        out = self.stream[start:upto]
        self.emitted += len(out)
        return out

    def finish_attempt(self) -> Continuation | None:
        """The current attempt's request finished: roll back (a ``Continuation``) or make the result final (None)."""
        f = self.pending_failure
        # a short text's filler is there to be cut: one that ends before it keeps that end (no truncation to retry)
        if f is None and self.eos_row is not None and self.detector is not None and self.short is None:
            f = self.detector.on_eos(self.eos_row)
            if f is not None and self.attempts >= self.cfg.max_retries:
                self._ignore(f)
                f = None
        if f is None and self.short is not None:
            f = self._short_run_on()
        # Rows already handed downstream are final: a rollback whose row lies below them restarts at the first unsent
        # row. commit_row() makes this rare: it holds back what the current detector state can still roll back, but
        # a regeneration's own rollbacks (the margin again, or a bin the replayed state now counts as skipped) can
        # reach below it.
        resume = None if f is None else max(f.rollback_row, self.n_prefill + self.emitted)
        if f is not None and self.max_tokens - (resume - self.n_prefill) <= 0:
            self._ignore(f)  # no budget left to regenerate from the rollback row: keep what was generated
            f = None
        if f is None:
            if self.short is not None and self.short.enter is not None:
                self.end = self._short_end()
            elif self.text_done:
                self.end = ("text_done", self.detector.stop_row())
            elif self.eos_row is not None:
                self.end = ("eos", self.eos_row)
            else:
                self.end = ("max_tokens", None)
            return None
        # the bucket a regeneration switches to, decided before the attempt counts
        if isinstance(f, PaceFailure):
            bucket = self._pace_retry_bucket(f)
        elif f.kind == "truncation" and self._rate_adjustable() and self.rate_bucket > self.rates.first:
            bucket = self.rate_bucket - 1  # ended too early: one bucket slower
        else:
            bucket = None
        self.attempts += 1
        seed = None if self.seed is None else self.seed + 1000 * self.attempts
        self.failures.append(_failure_record(f, seed, False, resume, bucket))
        del self.stream[resume:]
        self.detector.truncate(resume)
        if self.follower is not None:
            self.follower.truncate(resume - self.n_prefill)
        if self.short is not None:
            self.short.truncate(resume - self.n_prefill)
        if self.pace is not None:
            self.pace.truncate(resume)
        if bucket is not None:
            # a steered bucket's row is unknown: its pace comes from its steering delta (rate_deltas)
            self.bucket = bucket
            self.condition_values = list(self.condition_values)
            self.condition_values[self.rates.index] = self.rates.row(bucket)
        if isinstance(f, PaceFailure) or f.kind == "short_run_on":
            # a fresh take (rollback_row is the first row): in the bucket the measured duration asks for, or the
            # retake of a short text that ran into its filler
            if f.kind == "short_run_on":
                self.short_retries += 1
            else:
                self.pace_retries += 1
            self.probation = None
        else:
            self.probation = (f, AciCenter(self.detector.last_pos_before(resume), self.text_len))
        self.pending_failure, self.text_done, self.eos_row = None, False, None
        self.attempt_start = resume
        return Continuation(
            torch.stack(self.stream),
            resume - self.n_prefill,
            seed,
            self.max_tokens - (resume - self.n_prefill),
            None if self.condition_values is None else list(self.condition_values),
            steering_delta=(None if self.rate_deltas is None else self.rate_deltas[self.bucket - self.rates.first]),
            swap_delta=self.rate_deltas is not None,
        )

    def _short_run_on(self) -> Failure | None:
        """A run-on to regenerate as a fresh take (retries left), else None (an exhausted one is recorded, ignored)."""
        decided = self.short.decided
        if decided is None or decided[0] != "run_on":
            return None
        det = self.detector
        f = Failure(
            "short_run_on",
            detect_row=len(self.stream) - 1,
            onset_row=self.n_prefill + decided[1],
            rollback_row=self.n_prefill,
            bin=det.frontier,
            frontier=det.frontier,
        )
        if self.short_retries < self.short.cfg.max_retries and self.attempts < self.cfg.max_retries:
            return f
        self._ignore(f)
        return None

    def _short_end(self) -> tuple[str, int]:
        """The end of a short-text take whose reading reached the filler: the cut, else (a run-on, or a take that ended
        first) the frame where the reading reached the filler, within the complete frames and before any EOS."""
        decided = self.short.decided
        if decided is not None and decided[0] == "cut":
            return ("short_text", self.n_prefill + decided[1])
        frames = min(self.short.enter, len(self.stream) - self.n_prefill - (self.n_codebooks - 1))
        if self.eos_row is not None:
            frames = min(frames, self.eos_row - self.n_prefill)
        return ("short_text_forced", self.n_prefill + max(frames, 0))

    # -- final -----------------------------------------------------------------------------------------------------

    def eos_frame(self) -> int | None:
        """The final result's EOS as an aligned frame counted in generated rows (None: no EOS, cap hit)."""
        eos = self.end[1]
        return None if eos is None else int(eos) - self.n_prefill

    def record(self) -> dict:
        det = self.detector
        kind, eos = self.end
        final_rows = (len(self.stream) if eos is None else int(eos)) - self.n_prefill
        return {
            "attempts": self.attempts,
            "capped": self.attempts >= self.cfg.max_retries,
            "end": kind,
            "text_len": self.text_len,
            "n_bins": det.nb if det else 0,
            "final_rows": final_rows,
            "final_eos_row": None if eos is None else int(eos),
            "generated_total": self.generated_total,
            "pace_tok": det.pace_tok() if det else None,
            "heads": [list(h) for h in self.heads.heads],
            "failures": list(self.failures),
            "pace_retries": self.pace_retries,
            "rate_bucket": self.rate_bucket,
        }


# ----------------------------------------------------------------------------------------------------- head files


def parse_laci_heads(entries) -> dict[str, str]:
    """``--laci-heads LANG=PATH ...`` -> {request code: path}: the code lowercased with ``-`` as ``_``, the path
    user-expanded and an existing file; ValueError otherwise."""
    heads = {}
    for entry in entries or ():
        lang, sep, path = entry.partition("=")
        if not sep or not lang or not path:
            raise ValueError(f"--laci-heads takes LANG=PATH entries, got {entry!r}")
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            raise ValueError(f"--laci-heads {lang}: no file {path}")
        heads[lang.strip().lower().replace("-", "_")] = path
    return heads


def discover_laci_heads(directory: str | Path) -> dict[str, str]:
    """The head files of a directory for every server language (in ``zonos3.languages.CODES`` order): exactly one
    ``*_heads.json`` is the default; a ``*_heads_<code>.json`` serves the server languages whose speech code is
    ``<code>``. ValueError for zero or several default files, or a ``<code>`` no server language speaks."""
    from zonos3 import languages

    root = Path(directory).expanduser().resolve()
    defaults = sorted(root.glob("*_heads.json"))
    if len(defaults) != 1:
        raise ValueError(f"{root}: needs exactly one *_heads.json default head file, found {len(defaults)}")
    spoken = {lang.speech for lang in languages.SERVER}
    by_speech: dict[str, Path] = {}
    for path in sorted(root.glob("*_heads_*.json")):
        code = path.stem.rpartition("_heads_")[2]
        if code not in spoken:
            raise ValueError(f"{path}: no server language speaks {code!r}")
        by_speech[code] = path
    return {lang.code: str(by_speech.get(lang.speech, defaults[0])) for lang in languages.SERVER}


def main(argv=None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m zonos3.tts.laci", description="LANG=PATH --laci-heads arguments of a head directory"
    )
    parser.add_argument(
        "--all",
        required=True,
        metavar="DIR",
        help="every server language: the directory's *_heads.json, or its *_heads_<speech code>.json",
    )
    args = parser.parse_args(argv)
    try:
        heads = discover_laci_heads(args.all)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(" ".join(f"{lang}={path}" for lang, path in heads.items()))
    return 0
