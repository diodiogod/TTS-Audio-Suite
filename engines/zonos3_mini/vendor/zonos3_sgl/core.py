from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch
from zonos3.generation import SAMPLING

if TYPE_CHECKING:
    from zonos3.tts.span_follow import SpanFollower

    from zonos3_sgl.attention import BaseAttnBackend, BaseAttnMetadata
    from zonos3_sgl.kvcache import BaseCacheHandle
    from zonos3_sgl.models.gdn import PrefillInputs
    from zonos3_sgl.tts.align_probe import AlignProbe, ProbeInputs


@dataclass
class Context:
    page_size: int
    attn_backend: BaseAttnBackend
    _batch: TTSBatch | None = field(default=None, init=False)

    @property
    def batch(self) -> TTSBatch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: TTSBatch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX


# =============================================================================
# TTS-specific data structures
# =============================================================================


@dataclass
class TTSSamplingParams:
    """Sampling parameters for TTS generation (defaults: the generation registry's sampling defaults)."""

    temperature: float = SAMPLING["temperature"]
    topk: int = SAMPLING["topk"]
    top_p: float = SAMPLING["top_p"]
    min_p: float = SAMPLING["min_p"]
    max_tokens: int = 1024
    ignore_eos: bool = False
    repetition_window: int = SAMPLING["repetition_window"]
    repetition_penalty: float = SAMPLING["repetition_penalty"]
    repetition_codebooks: int = SAMPLING["repetition_codebooks"]
    seed: int | None = None


@dataclass(eq=False)
class TTSReq:
    """Request class for TTS generation with 2D token format.

    Tokens are in unpacked format: [cb0, cb1, ..., cb8, text_token] per frame.
    """

    input_ids: torch.Tensor  # 2D CPU tensor (seq_len, frame_width)
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: TTSSamplingParams
    cache_handle: BaseCacheHandle
    n_codebooks: int = 9
    # Audio padding id (marks text/marker frames) and the end token that
    # finishes this request (EOA, or span-end for edit spans). Required keywords.
    audio_pad_id: int = field(kw_only=True)
    stop_id: int = field(kw_only=True)
    # "any": a stop token in any codebook starts the stop countdown; "codebook0": only
    # one in codebook 0 does (lingua generate --eoa-policy codebook0).
    eoa_policy: str = "any"
    eos_frame: int = -1  # Aligned frame where EOS first appeared (-1 = not seen)
    eos_countdown: int = -1  # Steps remaining after EOS (-1 = not in countdown)
    total_generated: int = 0  # Total frames generated (for logging)
    processed_generated: int = field(default=0, init=False)  # generated frames whose results the host processed
    has_audio_prefix: bool = field(default=False, init=False)
    rng: torch.Generator | None = None  # Per-request RNG for deterministic sampling
    speaker_embedding: torch.Tensor | None = None  # 1D CPU float32 tensor
    speaker_token_position: int = -1  # Injection position within the prompt sequence
    # Condition rows (cond3 finetunes): bucket id per table (n_buckets = unknown) and the prompt position of the
    # first of their contiguous all-pad rows (right before BOS, after the reference prefix / speaker row).
    condition_values: list[int] | None = None
    condition_token_start: int = -1
    # Decode context / silence prefill (2026-10-17), handed to the detokenizer with the first generated row.
    decode_context_codes: torch.Tensor | None = None
    prefill_rows: torch.Tensor | None = None
    context_sent: bool = False  # set once the detokenizer got them (first emitted row)
    # Hidden-space steering delta of the request's controls, added on every generated frame: one hidden-size slot per
    # steered Full AttnRes bank source, flattened; 1D CPU float32 tensor.
    steering_delta: torch.Tensor | None = None
    # Span steering (zonos3.tts.steering): [K, 3 + delta] rows of (start frame, end frame, ramp frames, delta in the
    # steering_delta layout), added on the generated frames they cover.
    steer_schedule: torch.Tensor | None = None
    # Steer then hold (zonos3.tts.steering.release_weight): the released request-level hidden controls (layout of
    # steering_delta) and (onset frames, ramp frames, hold); the delta is added at its frame's weight.
    steer_release_delta: torch.Tensor | None = None
    steer_release: tuple[int, int, float] | None = None
    # Span steering by reading position (zonos3.tts.span_follow.SpanFollower; the speaker-CFG twin shares its
    # request's): replaces the schedule's frame placement of the same span deltas.
    span_follower: "SpanFollower | None" = None
    # Classifier-free guidance. On the conditional request, cfg_scale != 1.0 and
    # cfg_twin points at its paired twin request with its own prompt and cache:
    # controls CFG drops the steering delta and schedule, speaker CFG the speaker conditioning.
    # is_cfg_uncond marks the twin
    # (internal, emits no output). guided = twin + cfg_scale*(cond - twin).
    cfg_scale: float = 1.0
    is_cfg_uncond: bool = False
    cfg_twin: "TTSReq | None" = None
    # Accent adapter slot (0 = base model) and strength alpha: the request is
    # served as base + alpha * adapter (zonos3_sgl.models.accent_adapters).
    accent_slot: int = 0
    accent_scale: float = 0.0
    # Alignment probe (zonos3_sgl.tts.align_probe): reads/masks text attention during decode and may stop the request.
    align_probe: "AlignProbe | None" = None
    # Trailing prompt rows that count as repetition-penalty history (continuation from generated rows).
    history_rows: int = 0

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        assert self.input_ids.dim() == 2, "TTS input_ids must be 2D (seq_len, frame_width)"
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
        # A separate speaker reference before the text is completed context.
        # Only audio in the final prompt frame means the target stream is
        # already underway. A text-ending prompt needs its own delay triangle.
        audio_boundary = self.input_ids[-1, : self.n_codebooks]
        self.has_audio_prefix = bool(torch.any(audio_boundary != self.audio_pad_id))

        if self.speaker_embedding is not None:
            emb = self.speaker_embedding
            if emb.dim() == 2 and emb.shape[0] == 1:
                emb = emb.squeeze(0)
            if emb.dim() != 1:
                raise ValueError(f"speaker_embedding must be 1D or (1, D), got shape {tuple(emb.shape)}")
            self.speaker_embedding = emb.to(dtype=torch.float32, device="cpu")

        if self.steering_delta is not None:
            d = self.steering_delta
            if d.dim() == 2 and d.shape[0] == 1:
                d = d.squeeze(0)
            if d.dim() != 1:
                raise ValueError(f"steering_delta must be 1D or (1, D), got shape {tuple(d.shape)}")
            self.steering_delta = d.to(dtype=torch.float32, device="cpu")

        if self.steer_schedule is not None:
            s = self.steer_schedule
            if s.dim() != 2 or s.shape[1] < 4:
                raise ValueError(f"steer_schedule must be [K, 3 + delta], got shape {tuple(s.shape)}")
            self.steer_schedule = s.to(dtype=torch.float32, device="cpu")

        if self.steer_release_delta is not None:
            if self.steer_release is None:
                raise ValueError("steer_release_delta needs steer_release (onset frames, ramp frames, hold)")
            self.steer_release_delta = self.steer_release_delta.reshape(-1).to(dtype=torch.float32, device="cpu")
            onset, ramp, hold = self.steer_release
            self.steer_release = (int(onset), int(ramp), float(hold))

        if self.speaker_token_position < 0:
            # Training convention: reserved speaker slot is at prompt position 0.
            self.speaker_token_position = 0
        if self.speaker_token_position >= self.device_len:
            self.speaker_token_position = 0

    @property
    def frame_width(self) -> int:
        """Number of elements per frame (n_codebooks + extras)."""
        return self.input_ids.shape[-1]

    def steer_delta(self) -> torch.Tensor | None:
        """The hidden steering delta for the frame this request generates now: its constant delta, the released
        controls at that frame's weight (steer then hold) and the span deltas active at that frame (the schedule's, or
        the span follower's). Frames count the stream's generated frames: a LACI continuation resumes after its
        history_rows."""
        frame = self.total_generated + self.history_rows
        base = self.steering_delta
        if self.steer_release_delta is not None:
            from zonos3.tts.steering import released_delta

            base = released_delta(base, self.steer_release_delta, self.steer_release, frame)
        if self.span_follower is not None:
            span = self.span_follower.delta(frame)
        elif self.steer_schedule is None:
            return base
        else:
            from zonos3.tts.steering import schedule_delta

            span = schedule_delta(self.steer_schedule, frame)
        if span is None:
            return base
        return span if base is None else base + span

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    @property
    def num_completion_tokens(self) -> int:
        """Number of generated tokens (frames)."""
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1
        self.total_generated += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        """Append a single frame (unpacked token) to input_ids."""
        assert next_token.dim() == 1, "next_token must be 1D (frame_width,)"
        self.input_ids = torch.cat([self.input_ids, next_token.unsqueeze(0)], dim=0)

    def can_decode(self) -> bool:
        return self.remain_len > 0 and self.eos_countdown != 0

    def check_eos(self, audio_codes: List[int]) -> bool:
        """Check for EOS and update countdown state.

        Args:
            audio_codes: List of audio codebook values for one frame

        Returns:
            True if sequence is finished (countdown reached 0)
        """
        step = self.processed_generated
        self.processed_generated += 1
        if self.sampling_params.ignore_eos:
            return False

        # Match Zonos3 reference inference: any sampled stop codebook starts the
        # delayed stop countdown (eoa_policy "codebook0": codebook 0's only). The
        # aligned frame is shifted back by the highest stop codebook index and
        # clamped at zero.
        # Host result processing may lag submitted GPU work by one frame:
        # count processed results rather than reading total_generated here.
        if self.eos_frame < 0:
            eos_cols = [c == self.stop_id for c in audio_codes[: self.n_codebooks]]
            if self.eoa_policy == "codebook0":
                eos_cols = eos_cols[:1]
            if any(eos_cols):
                # First EOS: compute aligned frame
                max_eos_cb = max(i for i, is_eos in enumerate(eos_cols) if is_eos)
                self.eos_frame = max(0, step - max_eos_cb)
                self.eos_countdown = self.n_codebooks + 1

        # Decrement countdown
        if self.eos_countdown > 0:
            self.eos_countdown -= 1
            if self.eos_countdown == 0:
                return True

        return False

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len}, eos_frame={self.eos_frame})"
        )


@dataclass
class TTSBatch:
    """Batch of TTS requests with 2D token format."""

    reqs: List[TTSReq]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)  # (total_tokens, frame_width)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[TTSReq] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)
    # Optional per-batch speaker conditioning data (set by TTS scheduler).
    speaker_emb_values: torch.Tensor | None = field(default=None, init=False)
    speaker_token_positions: torch.Tensor | None = field(default=None, init=False)
    # Condition rows inside this extend span, per table: (bucket ids, packed positions) or None (set by the scheduler).
    condition_rows: list | None = field(default=None, init=False)
    # Per-token accent slots / strengths (set by the scheduler when adapters are
    # loaded) and the [tokens, K] adapter mask the model builds from them.
    accent_slots: torch.Tensor | None = field(default=None, init=False)
    accent_scales: torch.Tensor | None = field(default=None, init=False)
    accent_mask: torch.Tensor | None = field(default=None, init=False)
    # Alignment probes of a decode batch (zonos3_sgl.tts.align_probe.prepare, set by the engine): batched readout inputs,
    # and the rows needing the eager per-request path (masked heads, all-heads probes).
    probe: "ProbeInputs | None" = field(default=None, init=False)
    probe_legacy: list[int] | None = field(default=None, init=False)
    # A prefill CUDA graph's GDN inputs (zonos3_sgl.engine.prefill_graph): padded layout, slots, real row counts.
    gdn_prefill: "PrefillInputs | None" = field(default=None, init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)
