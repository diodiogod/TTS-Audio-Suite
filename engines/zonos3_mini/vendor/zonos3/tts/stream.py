"""Streaming decode of the generated rows: one request's TTSVocoderManager (zonos3_sgl.tokenizer.vocoder).

Rows arrive delayed (codebook j of frame f in row f + j). The buffer opens with the silence-prefill rows (decoded,
never emitted). ``whole`` mode decodes every pre-EOS frame once at end of stream, re-decoding the last
``overlap_frames`` already-counted rows as left context; ``prefix`` mode decodes incrementally on a stateful codec
stream (``PrefixDecoder``): chunks ramp from ``first_chunk`` frames (1, 2, 4, 8, ...) up to ``chunk``, behind
``hold_back`` frames, and at EOS the same stream continues to the end and is flushed; nothing is decoded twice.

A codec stream is any object with ``frames: int``, ``decode(codes [n, n_codebooks] int64) -> float32 samples`` that
are final and ``flush() -> float32 samples`` (zonos3.codec.istft_head_a_20261002.stream.ISTFTStream, zonos3_mlx's
CodecStream). ggml's ``prefix_submit_ready`` and the web server's ``prefixReady`` follow the same rules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from zonos3.tts.delay import undelay_window

PREFIX_ERROR = (
    "prefix stream settings: chunk_frames >= 1, first_chunk_frames >= 1, hold_back_frames >= 0, "
    "context_frames >= chunk_frames"
)


@dataclass(frozen=True)
class StreamSettings:
    """Prefix streaming: chunks ramp from ``first_chunk`` frames, doubling, up to ``chunk``, decoded behind
    ``hold_back`` complete frames (``prefix_need``). ``context`` is validated (>= chunk) but no longer used: the stream
    is stateful, nothing is re-decoded."""

    chunk: int = 15
    first_chunk: int = 1
    hold_back: int = 0
    context: int = 450

    def __post_init__(self):
        if self.chunk < 1 or self.first_chunk < 1 or self.hold_back < 0 or self.context < self.chunk:
            raise ValueError(PREFIX_ERROR)

    @classmethod
    def from_env(cls) -> StreamSettings:
        """ZONOS3_PREFIX_STREAM_{CHUNK,FIRST_CHUNK,HOLD_BACK,CONTEXT}_FRAMES over the defaults; ValueError if invalid."""
        default = cls()
        env = os.environ.get
        try:
            return cls(
                chunk=int(env("ZONOS3_PREFIX_STREAM_CHUNK_FRAMES", default.chunk)),
                first_chunk=int(env("ZONOS3_PREFIX_STREAM_FIRST_CHUNK_FRAMES", default.first_chunk)),
                hold_back=int(env("ZONOS3_PREFIX_STREAM_HOLD_BACK_FRAMES", default.hold_back)),
                context=int(env("ZONOS3_PREFIX_STREAM_CONTEXT_FRAMES", default.context)),
            )
        except (TypeError, ValueError):
            raise ValueError(PREFIX_ERROR) from None


def prefix_need(emitted: int, prefill: int, s: StreamSettings) -> int:
    """Frames the next prefix chunk waits for: ``first_chunk`` for the first (nothing past the ``prefill`` silence rows
    emitted yet), then ``first_chunk`` + the frames emitted so far, capped at ``chunk``: 1, 2, 4, 8, 15, 15, ... by
    default (the next need stays geometric when more frames arrive at once)."""
    return min(s.chunk, s.first_chunk + emitted - prefill)


def prefix_chunks(prefill: int, s: StreamSettings, frames: int) -> list[int]:
    """The frames of each codec-stream decode of a primed prefix stream (the ``prefill`` frames decoded on their own)
    whose rows keep arriving one at a time, until ``frames`` frames are decoded: 3, then 1, 2, 4, 8, 15, 15, ... by
    default."""
    chunks, start = [prefill] if prefill else [], prefill
    while start < frames:
        chunks.append(prefix_need(start, prefill, s))
        start += chunks[-1]
    return chunks


def _host(x) -> np.ndarray:
    """A numpy view of rows or codes (torch tensors on any device, arrays, lists)."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


_EMPTY = np.zeros(0, dtype=np.float32)


class PrefixDecoder:
    """One request's prefix-mode decode on a stateful codec ``stream``. The stream decodes [decode-context frames |
    silence-prefill frames | generated frames] from frame 0; the context and prefill samples are dropped. A ``stream``
    that already decoded the prefill frames (``stream.frames == prefill``, no decode context; ``primed_samples`` of
    them came out) starts at the generated frames. ``step`` emits the next chunk once ``prefix_need`` frames are
    decodable behind the hold-back; ``finish`` decodes up to the target and flushes the stream (no samples past the
    target's end, or the last frame emitted).

    Early partial-frame decode (``codebooks`` < n_codebooks; ``first_only``: the first chunk only): a frame is
    decodable once its first ``codebooks`` codebooks arrived (``ready``). The missing residual codes (audio padding
    after the undelay) reach the codec as -1, left out of the frame (a coarser RVQ decode); a frame is never decoded
    again."""

    def __init__(
        self,
        stream,
        *,
        n_codebooks: int,
        audio_pad_id: int,
        hop: int,
        prefill: int,
        context_codes=None,
        settings: StreamSettings = StreamSettings(),
        primed_samples: int = 0,
        codebooks: int | None = None,
        first_only: bool = False,
    ):
        self.stream = stream
        self.n_codebooks, self.pad, self.hop = n_codebooks, audio_pad_id, hop
        self.codebooks = n_codebooks if codebooks is None else int(codebooks)
        if not 1 <= self.codebooks <= n_codebooks:
            raise ValueError(f"stream_codebooks must be 1..{n_codebooks}")
        self.first_only = first_only
        self.prefill = int(prefill)
        self.settings = settings
        self.context = None
        if context_codes is not None and len(context_codes):
            self.context = _host(context_codes).astype(np.int64).reshape(-1, n_codebooks)
        ctx = 0 if self.context is None else len(self.context)
        self.ctx = ctx
        self.emitted = self.prefill
        if stream.frames and (stream.frames != self.prefill or ctx):
            raise ValueError("a primed codec stream must have decoded exactly the prefill frames, without context")
        self.produced = int(primed_samples) if stream.frames else 0  # raw stream samples
        self.skip = (ctx + self.prefill) * hop - self.produced  # raw stream samples still to drop
        self.done = False

    def ready(self, buffer, complete: int, eos: int | None = None) -> int:
        """Frames decodable now: the ``complete`` ones (undelayable, pre-EOS), or with early decode every frame whose
        first ``codebooks`` codebooks arrived (codebook j of frame f rides row f + j), capped at ``eos`` and before any
        frame whose semantic code is not a code (padding / EOA)."""
        if self.codebooks >= self.n_codebooks or (self.first_only and self.emitted > self.prefill):
            return complete
        ready = len(buffer) - (self.codebooks - 1)
        if eos is not None:
            ready = min(ready, eos)
        while ready > max(complete, self.emitted) and int(buffer[ready - 1][0]) >= self.pad - 1:
            ready -= 1
        return ready

    def step(self, buffer, complete: int, eos: int | None = None) -> np.ndarray:
        """The delayed rows so far, the complete (undelayable, pre-EOS) frame count and the EOS frame when known -> the
        next chunk, or empty."""
        decodable = self.ready(buffer, complete, eos) - self.settings.hold_back
        if decodable - self.emitted < prefix_need(self.emitted, self.prefill, self.settings):
            return _EMPTY
        return self._decode(buffer, decodable)

    def finish(self, buffer, target: int) -> np.ndarray:
        """End of stream: frames up to ``target`` (complete frames capped at EOS), then the stream's flush."""
        if self.done:
            raise RuntimeError("stream already finished")
        self.done = True
        parts = [self._decode(buffer, target)] if target > self.emitted else []
        tail = self.stream.flush()
        cap = max(0, (self.ctx + max(target, self.emitted)) * self.hop - self.produced)
        parts.append(self._out(tail[:cap]))
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _decode(self, buffer, end: int) -> np.ndarray:
        start = 0 if self.stream.frames == 0 else self.emitted  # the first call decodes the prefill rows too
        codes = undelay_window(buffer, start, end, self.n_codebooks, self.pad).numpy()
        if self.codebooks < self.n_codebooks:
            codes[:, 1:][codes[:, 1:] == self.pad] = -1
        if start == 0 and self.context is not None:
            codes = np.concatenate([self.context, codes], axis=0)
        self.emitted = end
        return self._out(self.stream.decode(codes))

    def _out(self, raw) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        self.produced += len(raw)
        drop = min(len(raw), self.skip)
        self.skip -= drop
        return raw[drop:]


class StreamDecoder:
    """One request's rows as they come -> float32 PCM. ``decode(codes)`` runs the codec on undelayed int64 codes
    [frames, n_codebooks] (whole mode; the reference tail prepended at the stream start, its samples dropped here);
    prefix mode drives ``stream`` (a codec stream) through a ``PrefixDecoder`` (``codebooks`` / ``first_only``: its
    early partial-frame decode, the request's stream_codebooks / stream_codebooks_scope == "first")."""

    def __init__(
        self,
        decode: Callable[[np.ndarray], np.ndarray],
        *,
        stream=None,
        n_codebooks: int,
        audio_pad_id: int,
        hop: int,
        prefill_rows=None,
        context_codes=None,
        mode: str = "whole",
        overlap_frames: int = 4,
        settings: StreamSettings = StreamSettings(),
        codebooks: int | None = None,
        first_only: bool = False,
    ):
        if mode not in ("whole", "prefix"):
            raise ValueError(f"stream decode mode must be whole or prefix, got {mode!r}")
        if mode == "prefix" and stream is None:
            raise ValueError("prefix stream decode needs a codec stream")
        self.decode = decode
        self.n_codebooks, self.pad, self.eoa, self.hop = n_codebooks, audio_pad_id, audio_pad_id - 1, hop
        self.mode, self.overlap, self.settings = mode, overlap_frames, settings
        self.buffer: list[list[int]] = (
            []
            if prefill_rows is None
            else [[int(v) for v in row[:n_codebooks]] for row in _host(prefill_rows).tolist()]
        )
        self.prefill = len(self.buffer)
        self.context = None
        if context_codes is not None and len(context_codes):
            self.context = _host(context_codes).astype(np.int64).reshape(-1, n_codebooks)
        self.decoded = self.prefill  # whole mode: frames already counted
        self.prefix = None
        if mode == "prefix":
            self.prefix = PrefixDecoder(
                stream,
                n_codebooks=n_codebooks,
                audio_pad_id=audio_pad_id,
                hop=hop,
                prefill=self.prefill,
                context_codes=self.context,
                settings=settings,
                codebooks=codebooks,
                first_only=first_only,
            )
        self.eos: int | None = None
        self.done = False

    def _decode(self, start: int, frames_end: int) -> np.ndarray:
        """Output frames [start, frames_end) from the delayed rows, with the reference tail as left context when the
        slice starts at the stream start (its samples dropped)."""
        codes = undelay_window(self.buffer, start, frames_end, self.n_codebooks, self.pad).numpy()
        if start == 0 and self.context is not None:
            audio = self.decode(np.concatenate([self.context, codes], axis=0))
            return np.asarray(audio, dtype=np.float32)[len(self.context) * self.hop :]
        return np.asarray(self.decode(codes), dtype=np.float32)

    def _target(self) -> int:
        target = len(self.buffer) - (self.n_codebooks - 1)
        return target if self.eos is None else min(target, self.eos)

    def push(self, row, eos_frame: int | None, finished: bool = False) -> np.ndarray:
        """One generated row (``eos_frame``: the scheduler's aligned EOS frame in generated rows); the float32 PCM it
        completes (maybe none). ``finished`` marks the stream's last row (the same as pushing it and calling
        ``finish``)."""
        if self.done:
            raise RuntimeError("stream already finished")
        head = row[: self.n_codebooks]
        codes = [int(v) for v in (head.tolist() if hasattr(head, "tolist") else head)]
        self.buffer.append(codes)
        candidates = []
        if eos_frame is not None:
            candidates.append(max(0, int(eos_frame)) + self.prefill)
        cols = [i for i, tok in enumerate(codes) if tok == self.eoa]
        if cols:
            candidates.append(max(0, len(self.buffer) - 1 - max(cols)))
        if candidates:
            observed = min(candidates)
            self.eos = observed if self.eos is None else min(self.eos, observed)
        if finished:
            return self.finish()
        return self._prefix_chunk() if self.mode == "prefix" else _EMPTY

    def finish(self) -> np.ndarray:
        """End of stream: the rest of the audio (the whole decode, or the prefix stream continued and flushed)."""
        if self.done:
            raise RuntimeError("stream already finished")
        self.done = True
        return self._prefix_final() if self.mode == "prefix" else self._whole_final()

    def _whole_final(self) -> np.ndarray:
        """Every complete pre-EOS frame in one decode; the last ``overlap`` counted rows (the silence prefill) are
        re-decoded and emitted with the rest, as TTSVocoderManager._decode_final does."""
        target = self._target()
        if target <= self.decoded:
            return _EMPTY
        start = self.decoded - min(self.overlap, self.decoded)
        audio = self._decode(start, target)
        self.decoded = target
        return audio

    def _prefix_chunk(self) -> np.ndarray:
        return self.prefix.step(self.buffer, self._target(), self.eos)

    def _prefix_final(self) -> np.ndarray:
        return self.prefix.finish(self.buffer, self._target())
