"""Fixed-shape batching for encoders that must keep files independent."""

from __future__ import annotations

import concurrent.futures
import math
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Iterator, Optional

import numpy as np


@dataclass
class _Piece:
    waveform: object
    context_frames: int
    output_frames: int
    trailing_context_frames: int
    input_offset_frames: int = 0
    output_start_frame: int = 0
    fixed_shape: bool = True
    output: Optional[np.ndarray] = None


@dataclass
class _Record:
    sample: object
    pieces: list[_Piece]

    @property
    def complete(self) -> bool:
        return self.sample.error is not None or all(piece.output is not None for piece in self.pieces)


@dataclass
class _PreparedBatch:
    records: list[_Record]
    pieces: list[_Piece]
    waveform: object


@dataclass
class _EncodedBatch:
    pieces: list[_Piece]
    host_output: object
    device_output: object
    ready: object
    compute_started: object
    compute_finished: object


class IndependentBatcher:
    """Batch one causal sequence per lane without mixing file boundaries.

    Files longer than a lane are split on hop-aligned boundaries. Continuation
    pieces carry real left context, and non-causal encoders may request real
    right context as well. Initial pieces still begin at the native sequence
    boundary rather than borrowing state from another file.
    """

    def __init__(
        self,
        *,
        torch,
        device,
        sample_rate: int,
        hop_length: int,
        feature_dim: int,
        prepare: Callable[[np.ndarray, int], object],
        run: Callable[[object], object],
        batch_size: int,
        chunk_seconds: float,
        continuation_context_seconds: float,
        window: int,
        output_dtype: np.dtype,
        label: str,
        trailing_context_seconds: float = 0.0,
        preparation_workers: int = 1,
        input_dtype=None,
        input_channels: int = 1,
    ):
        if (
            min(
                sample_rate,
                hop_length,
                feature_dim,
                batch_size,
                window,
                preparation_workers,
                input_channels,
            )
            <= 0
        ):
            raise ValueError("batcher dimensions and window must be positive")
        if chunk_seconds <= 0 or continuation_context_seconds < 0 or trailing_context_seconds < 0:
            raise ValueError("batcher durations are invalid")

        self.torch = torch
        self.device = device
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.feature_dim = int(feature_dim)
        self.prepare = prepare
        self.run = run
        self.batch_size = int(batch_size)
        self.window = int(window)
        self.preparation_workers = int(preparation_workers)
        self.output_dtype = np.dtype(output_dtype)
        self.label = label
        self.input_dtype = input_dtype or torch.float32
        self.input_channels = int(input_channels)

        frames_per_second = self.sample_rate / self.hop_length
        self.chunk_frames = max(1, math.floor(chunk_seconds * frames_per_second))
        self.context_frames = max(0, math.ceil(continuation_context_seconds * frames_per_second))
        self.trailing_context_frames = max(0, math.ceil(trailing_context_seconds * frames_per_second))
        if self.context_frames + self.trailing_context_frames >= self.chunk_frames:
            raise ValueError("combined context must be shorter than a chunk")
        self.chunk_samples = self.chunk_frames * self.hop_length
        self.context_samples = self.context_frames * self.hop_length
        self.trailing_context_samples = self.trailing_context_frames * self.hop_length
        self.continuation_samples = self.chunk_samples - self.context_samples
        self.continuation_samples -= self.trailing_context_samples

    def _record(self, sample) -> _Record:
        if sample.error is not None:
            return _Record(sample, [])
        try:
            waveform = self.prepare(sample.audio, sample.sample_rate)
            if (
                waveform.ndim != 3
                or waveform.shape[0] != 1
                or waveform.shape[1] != self.input_channels
                or waveform.shape[2] <= 0
            ):
                raise ValueError(
                    f"prepared audio must be non-empty [1, {self.input_channels}, time], got {tuple(waveform.shape)}"
                )
            waveform = waveform.contiguous()
        except Exception as exc:
            failed = replace(sample, error=f"{type(exc).__name__}: {exc}")
            return _Record(failed, [])

        if self.trailing_context_frames:
            pieces = self._bidirectional_pieces(waveform)
            return _Record(sample, pieces)

        pieces = []
        total_samples = int(waveform.shape[-1])
        start = 0
        while start < total_samples:
            first = start == 0
            context_samples = 0 if first else self.context_samples
            capacity = self.chunk_samples - context_samples - self.trailing_context_samples
            end = min(total_samples, start + capacity)
            history_start = start - context_samples
            trailing_samples = min(self.trailing_context_samples, total_samples - end)
            pieces.append(
                _Piece(
                    waveform=waveform[..., history_start : end + trailing_samples],
                    context_frames=0 if first else self.context_frames,
                    output_frames=math.ceil((end - start) / self.hop_length),
                    trailing_context_frames=math.ceil(trailing_samples / self.hop_length),
                    output_start_frame=0 if first else self.context_frames,
                )
            )
            start = end
        return _Record(sample, pieces)

    def _bidirectional_pieces(self, waveform) -> list[_Piece]:
        """Split while retaining the sequence boundaries seen by the model.

        A fixed lane padded after a short file is not equivalent to running a
        multilayer convolutional encoder on that file: biases create nonzero
        activations past the real boundary. Head pieces are therefore anchored
        at the lane start and final pieces at its end. Short files are stitched
        from both anchors. Files too short to contain both context regions use
        one exact, dynamic-shape invocation.
        """

        total_samples = int(waveform.shape[-1])
        if total_samples % self.hop_length:
            raise ValueError(
                "bidirectional batching requires hop-aligned prepared audio, got "
                f"{total_samples} samples for hop {self.hop_length}"
            )
        total_frames = total_samples // self.hop_length
        if total_frames > self.chunk_frames:
            pieces = []
            head_frames = self.chunk_frames - self.trailing_context_frames
            pieces.append(
                _Piece(
                    waveform=waveform[..., : self.chunk_samples],
                    context_frames=0,
                    output_frames=head_frames,
                    trailing_context_frames=self.trailing_context_frames,
                )
            )
            start = head_frames
            interior_frames = self.chunk_frames - self.context_frames - self.trailing_context_frames
            while start < total_frames:
                remaining = total_frames - start
                if remaining + self.context_frames <= self.chunk_frames:
                    source_start = start - self.context_frames
                    source = waveform[..., source_start * self.hop_length :]
                    source_frames = int(source.shape[-1]) // self.hop_length
                    input_offset = self.chunk_frames - source_frames
                    pieces.append(
                        _Piece(
                            waveform=source,
                            context_frames=self.context_frames,
                            output_frames=remaining,
                            trailing_context_frames=0,
                            input_offset_frames=input_offset,
                            output_start_frame=input_offset + self.context_frames,
                        )
                    )
                    break

                end = start + interior_frames
                source_start = start - self.context_frames
                source_end = end + self.trailing_context_frames
                pieces.append(
                    _Piece(
                        waveform=waveform[
                            ...,
                            source_start * self.hop_length : source_end * self.hop_length,
                        ],
                        context_frames=self.context_frames,
                        output_frames=interior_frames,
                        trailing_context_frames=self.trailing_context_frames,
                        output_start_frame=self.context_frames,
                    )
                )
                start = end
            return pieces

        if total_frames == self.chunk_frames:
            return [
                _Piece(
                    waveform=waveform,
                    context_frames=0,
                    output_frames=total_frames,
                    trailing_context_frames=0,
                )
            ]

        if total_frames < self.context_frames + self.trailing_context_frames:
            return [
                _Piece(
                    waveform=waveform,
                    context_frames=0,
                    output_frames=total_frames,
                    trailing_context_frames=0,
                    fixed_shape=False,
                )
            ]

        earliest_tail = self.context_frames
        latest_head = total_frames - self.trailing_context_frames
        split = min(latest_head, max(earliest_tail, total_frames // 2))
        input_offset = self.chunk_frames - total_frames
        return [
            _Piece(
                waveform=waveform,
                context_frames=0,
                output_frames=split,
                trailing_context_frames=self.trailing_context_frames,
            ),
            _Piece(
                waveform=waveform,
                context_frames=self.context_frames,
                output_frames=total_frames - split,
                trailing_context_frames=0,
                input_offset_frames=input_offset,
                output_start_frame=input_offset + split,
            ),
        ]

    def _prepare_batch(self, records: list[_Record], pieces: list[_Piece]) -> _PreparedBatch:
        batch = self.torch.zeros(
            (self.batch_size, self.input_channels, self.chunk_samples),
            device="cpu",
            dtype=self.input_dtype,
            pin_memory=self.device.type == "cuda",
        )
        for lane, piece in enumerate(pieces):
            start = piece.input_offset_frames * self.hop_length
            end = start + piece.waveform.shape[-1]
            batch[lane, :, start:end].copy_(piece.waveform[0])
        return _PreparedBatch(records, pieces, batch)

    def _prepare_direct(self, records: list[_Record], piece: _Piece) -> _PreparedBatch:
        waveform = piece.waveform.to(
            device="cpu",
            dtype=self.input_dtype,
            copy=True,
        )
        if self.device.type == "cuda":
            waveform = waveform.pin_memory()
        return _PreparedBatch(records, [piece], waveform)

    def _stage(self, prepared: _PreparedBatch, transfer_stream):
        """Enqueue one pinned-host batch without delaying the current encode."""

        if transfer_stream is None:
            return prepared.waveform.to(self.device), None
        with self.torch.cuda.stream(transfer_stream):
            batch = prepared.waveform.to(self.device, non_blocking=True)
            ready = self.torch.cuda.Event()
            ready.record(transfer_stream)
        return batch, ready

    def _encode(self, prepared: _PreparedBatch, batch, ready, transfer_stream) -> tuple[int, int, int, _EncodedBatch]:
        pieces = prepared.pieces
        if ready is not None:
            self.torch.cuda.current_stream(self.device).wait_event(ready)

        compute_started = None
        compute_finished = None
        if self.device.type == "cuda":
            compute_started = self.torch.cuda.Event(enable_timing=True)
            compute_finished = self.torch.cuda.Event(enable_timing=True)
            compute_started.record(self.torch.cuda.current_stream(self.device))
        with self.torch.inference_mode():
            outputs = self.run(batch)
        if compute_finished is not None:
            compute_finished.record(self.torch.cuda.current_stream(self.device))
        if (
            outputs.ndim != 3
            or outputs.shape[0] != batch.shape[0]
            or outputs.shape[1] < max(piece.output_start_frame + piece.output_frames for piece in pieces)
            or outputs.shape[2] != self.feature_dim
        ):
            raise ValueError(f"encoder must return [batch, time, feature], got {tuple(outputs.shape)}")

        if self.output_dtype == np.dtype(np.float32):
            output_torch_dtype = self.torch.float32
        elif self.output_dtype == np.dtype(np.float16):
            output_torch_dtype = self.torch.float16
        elif self.output_dtype == np.dtype(np.int16):
            output_torch_dtype = self.torch.int16
        else:
            raise TypeError(f"unsupported encoder output dtype: {self.output_dtype}")

        # The compiled callable may reuse its output storage. Copy into a
        # private device buffer before allowing the next invocation to start.
        device_output = outputs.to(dtype=output_torch_dtype, copy=True)
        if transfer_stream is None:
            host_output = device_output
            output_ready = None
        else:
            host_output = self.torch.empty(
                device_output.shape,
                device="cpu",
                dtype=output_torch_dtype,
                pin_memory=True,
            )
            compute_ready = self.torch.cuda.Event()
            compute_ready.record(self.torch.cuda.current_stream(self.device))
            with self.torch.cuda.stream(transfer_stream):
                transfer_stream.wait_event(compute_ready)
                host_output.copy_(device_output, non_blocking=True)
                output_ready = self.torch.cuda.Event()
                output_ready.record(transfer_stream)

        content_frames = 0
        work_frames = 0
        for lane, piece in enumerate(pieces):
            content_frames += piece.output_frames
            work_frames += piece.context_frames + piece.output_frames + piece.trailing_context_frames
        encoded = _EncodedBatch(
            pieces,
            host_output,
            device_output,
            output_ready,
            compute_started,
            compute_finished,
        )
        scheduled_frames = int(batch.shape[0] * outputs.shape[1])
        return content_frames, work_frames, scheduled_frames, encoded

    def _collect(self, encoded: _EncodedBatch) -> float:
        if encoded.ready is not None:
            encoded.ready.synchronize()
        host = encoded.host_output.numpy()
        for lane, piece in enumerate(encoded.pieces):
            start = piece.output_start_frame
            end = start + piece.output_frames
            piece.output = np.array(
                host[lane, start:end],
                dtype=self.output_dtype,
                order="C",
                copy=True,
            )
        if encoded.compute_started is None:
            return 0.0
        return encoded.compute_started.elapsed_time(encoded.compute_finished) / 1000.0

    def encode_stream(self, samples: Iterable[object]) -> Iterator[tuple[object, Optional[np.ndarray]]]:
        if self.device.type == "cuda":
            self.torch.cuda.set_device(self.device)

        end = object()
        prepared: queue.Queue = queue.Queue(maxsize=min(4, self.window))
        stop = threading.Event()

        def put(item) -> bool:
            while not stop.is_set():
                try:
                    prepared.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False

        def produce() -> None:
            try:
                records = []
                pieces = []

                def prepared_records():
                    if self.preparation_workers == 1:
                        for sample in samples:
                            if stop.is_set():
                                return
                            yield self._record(sample)
                        return

                    pending = deque()
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.preparation_workers,
                        thread_name_prefix=f"{self.label}-prepare",
                    ) as pool:
                        for sample in samples:
                            if stop.is_set():
                                return
                            pending.append(pool.submit(self._record, sample))
                            if len(pending) >= self.window:
                                yield pending.popleft().result()
                        while pending and not stop.is_set():
                            yield pending.popleft().result()

                for record in prepared_records():
                    records.append(record)
                    for piece in record.pieces:
                        if not piece.fixed_shape:
                            if pieces:
                                if not put(self._prepare_batch(records, pieces)):
                                    return
                                records = []
                                pieces = []
                            if not put(self._prepare_direct(records, piece)):
                                return
                            records = []
                            continue
                        pieces.append(piece)
                        if len(pieces) == self.batch_size:
                            if not put(self._prepare_batch(records, pieces)):
                                return
                            records = []
                            pieces = []
                if records or pieces:
                    if not put(self._prepare_batch(records, pieces)):
                        return
            except BaseException as exc:
                put(exc)
            finally:
                put(end)

        producer = threading.Thread(target=produce, name=f"{self.label}-input-preparation", daemon=True)
        producer.start()
        transfer_stream = self.torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        records: list[_Record] = []
        pack_count = 0
        content_frames = 0
        work_frames = 0
        scheduled_frames = 0
        input_wait = 0.0
        encode_time = 0.0
        output_wait = 0.0
        gpu_compute_time = 0.0
        downstream_time = 0.0
        pending_outputs: list[_EncodedBatch] = []

        def receive():
            nonlocal input_wait
            started = time.perf_counter()
            item = prepared.get()
            input_wait += time.perf_counter() - started
            if isinstance(item, BaseException):
                raise item
            return item

        def complete_records():
            nonlocal downstream_time
            while records and records[0].complete:
                record = records.pop(0)
                output = None
                if record.sample.error is None:
                    output = np.concatenate([piece.output for piece in record.pieces], axis=0)
                started = time.perf_counter()
                yield record.sample, output
                downstream_time += time.perf_counter() - started

        try:
            item = receive()
            staged = None
            while item is not end:
                if isinstance(item, BaseException):
                    raise item
                records.extend(item.records)

                if item.pieces:
                    if staged is None:
                        staged = self._stage(item, transfer_stream)

                    # Keep one H2D transfer in flight during the current model
                    # call. Never wait for prefetch: an empty producer queue
                    # should not postpone work that is already GPU-ready.
                    try:
                        next_item = prepared.get_nowait()
                    except queue.Empty:
                        next_item = None
                    next_staged = None
                    if isinstance(next_item, _PreparedBatch) and next_item.pieces:
                        next_staged = self._stage(next_item, transfer_stream)

                    started = time.perf_counter()
                    content, work, scheduled, encoded = self._encode(item, *staged, transfer_stream)
                    encode_time += time.perf_counter() - started
                    pending_outputs.append(encoded)
                    pack_count += 1
                    content_frames += content
                    work_frames += work
                    scheduled_frames += scheduled
                else:
                    next_item = None
                    next_staged = None
                # One batch of slack lets output transfer overlap the next
                # model call while bounding temporary device and pinned memory.
                if len(pending_outputs) >= 2:
                    started = time.perf_counter()
                    gpu_compute_time += self._collect(pending_outputs.pop(0))
                    output_wait += time.perf_counter() - started
                yield from complete_records()

                if next_item is None:
                    item = receive()
                    staged = None
                else:
                    item = next_item
                    staged = next_staged
            while pending_outputs:
                started = time.perf_counter()
                gpu_compute_time += self._collect(pending_outputs.pop(0))
                output_wait += time.perf_counter() - started
                yield from complete_records()
            yield from complete_records()
            if records:
                raise RuntimeError("batcher retained incomplete records")
        finally:
            stop.set()
            producer.join(timeout=5)

        rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))
        work_fill = work_frames / scheduled_frames if scheduled_frames else 0.0
        content_fill = content_frames / scheduled_frames if scheduled_frames else 0.0
        print(
            f"[rank {rank}] {self.label} stages: batches={pack_count} "
            f"work_fill={work_fill:.4f} content_fill={content_fill:.4f} "
            f"input_wait={input_wait:.3f}s encode={encode_time:.3f}s "
            f"output_wait={output_wait:.3f}s gpu_compute={gpu_compute_time:.3f}s "
            f"downstream={downstream_time:.3f}s",
            flush=True,
        )


__all__ = ["IndependentBatcher"]
