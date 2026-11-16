"""Short texts: a text of a few words generated with a filler sentence after it, cut where the filler starts.

Alone, one or two words often come out wrong or run on into words the text does not have: the model was trained on
sentences. A short text is generated under LACI as ``<text><pause marker><filler>`` (``filler_text``: the 0.5 s pause
token of the release text rows, then the language's filler sentence; ``registries/short_text.json``) and cut once the
reading position reaches the filler (``ShortCut``): at the first frame a codes-only silence classifier calls silent,
searched from ``lookback`` frames before the reading reached the filler, with ``min_voiced`` non-silent frames up to
it. A take that reaches no such frame within ``max_frames`` frames of the reading reaching the filler ran into it (a
run-on): it is regenerated as a fresh take at most ``max_retries`` times, then cut where the reading reached the
filler. The LACI session holds the take until the cut is decided, so nothing past the cut reaches the decoder.

The request field ``short_text`` (``auto``, the default, or ``off``) turns this on for a request the rules allow
(``zonos3.tts.plan.Rules.short_text``: no CFG, control spans, speech edit or LACI duration target, a language with
alignment heads) whose normalized text qualifies (``prompt``): a checkpoint with the pause marker's token, a language
with a filler, a text without start, pause or phoneme markers, and a short text (``is_short``). The ggml server
(zonos3-short-text.cpp) and the in-browser server (servers/web/src/request/shortText.ts) port this module;
servers/ggml/tests/fixtures/laci-replay pins the three.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

REGISTRY = Path(__file__).resolve().parents[1] / "registries" / "short_text.json"


@dataclass(frozen=True)
class ShortTextConfig:
    pause_marker: str
    max_words: int
    max_ratio: tuple[int, int]  # short: text tokens * max_ratio[1] <= filler tokens * max_ratio[0]
    lookback: int
    min_voiced: int
    max_frames: int
    max_retries: int
    codebook_size: int
    threshold: int
    weights: np.ndarray  # int32 [n_codebooks, codebook_size]
    fillers: dict[str, str]  # per server language, with the space that joins it to the pause marker


@lru_cache(maxsize=1)
def registry() -> ShortTextConfig:
    d = json.loads(REGISTRY.read_text(encoding="utf-8"))
    s, cut = d["silence"], d["cut"]
    n, size = int(s["n_codebooks"]), int(s["codebook_size"])
    weights = np.frombuffer(base64.b64decode(s["weights"]), dtype=np.int8).reshape(n, size).astype(np.int32)
    return ShortTextConfig(
        pause_marker=d["pause_marker"],
        max_words=int(d["max_words"]),
        max_ratio=(int(d["max_ratio"][0]), int(d["max_ratio"][1])),
        lookback=int(cut["lookback"]),
        min_voiced=int(cut["min_voiced"]),
        max_frames=int(cut["max_frames"]),
        max_retries=int(cut["max_retries"]),
        codebook_size=size,
        threshold=int(s["threshold"]),
        weights=weights,
        fillers=dict(d["fillers"]),
    )


def is_short(text: str, language: str, spec: dict) -> bool:
    """Whether a normalized text is short: at most ``max_words`` words and few tokens next to its language's filler."""
    from .superbpe import payload_ids

    cfg = registry()
    if len(text.split()) > cfg.max_words:
        return False
    n = len(payload_ids(text, spec))
    return 0 < n and n * cfg.max_ratio[1] <= len(payload_ids(cfg.fillers[language], spec)) * cfg.max_ratio[0]


def prompt(text: str, language: str, spec: dict | None) -> tuple[str, int] | None:
    """A short text's prompt text (the text, the pause marker, the language's filler) and the text-span index (0 =
    BOS) of the filler's first token: the first past the marker that is not all whitespace. None when the normalized
    ``text`` does not qualify (see the module docstring)."""
    from .superbpe import PAUSE_MARKER_RE, PHONEME_CLOSE, PHONEME_OPEN, START_MARKER

    cfg = registry()
    if spec is None or spec.get("pauses") is None or cfg.pause_marker not in spec["pauses"]["markers"]:
        return None
    if language not in cfg.fillers or PAUSE_MARKER_RE.search(text):
        return None
    if any(m in text for m in (START_MARKER, PHONEME_OPEN, PHONEME_CLOSE)) or not is_short(text, language, spec):
        return None
    from .span_follow import token_char_ranges

    full = text + cfg.pause_marker + cfg.fillers[language]
    for i, (a, b) in enumerate(token_char_ranges(full, spec)):
        if a > len(text) and not full[a:b].isspace():
            return full, i
    raise ValueError("short text: the filler has no token")


class ShortCut:
    """The cut of a short-text take, over its generated frames (frame f = codebook k of generated row f + k).

    ``observe_reading(frame, pos)`` once per decode step (``frame``: the generated row of the step's readout),
    ``observe_row(row)`` per generated row; ``decided`` turns to ``("cut", frames)`` (keep generated frames
    [0, frames)) or ``("run_on", enter)`` (``enter``: the frame where the reading reached the filler) once it can be
    told. ``truncate(n)`` keeps generated rows [0, n) (a LACI rollback)."""

    def __init__(self, filler_token: int, n_codebooks: int):
        self.filler_token, self.n_codebooks = filler_token, n_codebooks
        self.cfg = registry()
        self.enter: int | None = None
        self.decided: tuple[str, int] | None = None
        self._rows: list[list[int]] = []
        self._silent: list[bool] = []  # per complete frame

    def truncate(self, n: int) -> None:
        del self._rows[n:]
        del self._silent[max(0, n - self.n_codebooks + 1) :]
        if self.enter is not None and self.enter >= n:
            self.enter = None
        self.decided = None
        self._decide()

    def observe_reading(self, frame: int, pos: int) -> None:
        if self.enter is None and pos >= self.filler_token:
            self.enter = frame
            self._decide()

    def observe_row(self, row) -> None:
        self._rows.append([int(x) for x in row[: self.n_codebooks]])
        f = len(self._rows) - self.n_codebooks  # the frame this row completes
        if f >= 0:
            self._silent.append(self._is_silent(f))
            self._decide()

    def _is_silent(self, f: int) -> bool:
        cfg = self.cfg
        codes = [self._rows[f + k][k] for k in range(self.n_codebooks)]
        if any(not 0 <= c < cfg.codebook_size for c in codes):
            return False
        return sum(int(cfg.weights[k, c]) for k, c in enumerate(codes)) > cfg.threshold

    def _decide(self) -> None:
        if self.decided is not None or self.enter is None:
            return
        cfg = self.cfg
        voiced = 0
        for f, silent in enumerate(self._silent):
            voiced += not silent
            if f < self.enter - cfg.lookback:
                continue
            if f + 1 - self.enter > cfg.max_frames:
                self.decided = ("run_on", self.enter)
                return
            if silent and voiced >= cfg.min_voiced:
                self.decided = ("cut", f + 1)
                return
