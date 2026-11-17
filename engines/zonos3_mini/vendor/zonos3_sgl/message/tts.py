from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from zonos3.generation import (
    DECODE_CONTEXT_FRAMES,
    SAMPLE_RATE,
    SAMPLING,
    STREAM_CODEBOOKS,
    STREAM_CODEBOOKS_SCOPE,
)

from .utils import deserialize_type, serialize_type


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
    # Which end token finishes generation: "utterance" stops on EOA and never
    # samples span-end; "span" (edit infill, needs a checkpoint with span_end_id)
    # stops on span-end and never samples EOA.
    end_token: str = "utterance"
    # The stop token is masked until the generation has min_frames aligned frames (an edit
    # fill's lower bound; 0 = off).
    min_frames: int = 0
    # Controls classifier-free guidance scale. 1.0 disables guidance. Otherwise
    # the scheduler runs a paired unconditional twin (same prompt + speaker but
    # no hidden steering delta / schedule) and combines logits: guided = uncond + scale*(cond-uncond),
    # amplifying the hidden-space controls in output space.
    controls_cfg_scale: float = 1.0
    speaker_cfg_scale: float = 1.0
    # Progressive decode for the matched long-context codecs: "prefix" streams
    # chunks of an incremental decode on a stateful codec stream, flushed at EOS
    # (equal to the whole decode); None = one exact chunk at EOS (default).
    stream_decode: str | None = None
    # Prefix decode only: decode a frame once this many of its codebooks arrived (n_codebooks = complete frames), the
    # newest frames' later residual codebooks left out (a coarser RVQ decode, first audio up to n_codebooks - 1 rows
    # earlier); for the "first" chunk only (later chunks wait for complete frames) or "all" (registry defaults).
    stream_codebooks: int = STREAM_CODEBOOKS
    stream_codebooks_scope: str = STREAM_CODEBOOKS_SCOPE
    # The audio decoder (a served decoder id, checked by the frontend; None = the server default), read by the vocoder.
    decoder: str | None = None


@dataclass
class BaseTTSTokenizerMsg:
    """Base class for TTS tokenizer messages."""

    @staticmethod
    def encoder(msg: BaseTTSTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTTSTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTTSTokenizerMsg(BaseTTSTokenizerMsg):
    """Batch of TTS tokenizer messages."""

    data: List[BaseTTSTokenizerMsg]


@dataclass
class TTSTokenizeMsg(BaseTTSTokenizerMsg):
    """Request to build TTS prompt frames from text (Frontend -> Tokenizer)."""

    uid: int
    text: str
    sampling_params: TTSSamplingParams
    # Language code used for text normalization (e.g. "en_us").
    language: str = "en_us"
    # Run written->spoken text normalization before tokenization.
    text_normalization: bool = True
    # Silence prefill (2026-10-17): on a matched codec, this many codec-matched silence frames (15 fps -> 3 = 0.2 s)
    # open the audio stream as sheared prompt rows; the LM continues after them. None = the server default, 0 = off.
    silence_prefill_frames: int | None = None
    # The prompt text opens with the start marker (start_immediately, zonos3.tts.plan.Rules.start): the tokenizer
    # worker prepends it to the normalized text unless the text already starts with it.
    start_immediately: bool = False
    # Decode context (2026-10-17): decode the generated frames behind the last decode_context_frames of the
    # request's reference prompt codes (their audio dropped), so the first generated frame is not the decoder's
    # position 0.
    decode_context: bool = False
    decode_context_frames: int = DECODE_CONTEXT_FRAMES
    # Optional speaker embedding for voice cloning; shape: (speaker_embedding_dim,)
    speaker_embedding: torch.Tensor | None = None
    # The raw reference prefix codes [frames, n_codebooks] (int) of speaker-reference-prefix checkpoints: inline in the
    # request or encoded by the frontend. Mutually exclusive with the vector speaker_embedding path.
    speaker_reference_codes: torch.Tensor | None = None
    # Token position in the prompt sequence where speaker embedding is injected.
    speaker_token_position: int = -1
    # Hidden-space steering delta of the request's controls (zonos3.tts.steering.resolve_controls): one hidden-size
    # slot per steered Full AttnRes bank source (the engine's steer_sources), flattened, added on every generated frame.
    steering_delta: torch.Tensor | None = None
    # Span steering [K, 3 + delta] (zonos3.tts.steering.resolve_controls), same layout as steering_delta.
    steer_schedule: torch.Tensor | None = None
    # Steer then hold (zonos3.tts.steering.release_weight): the released request-level hidden controls (layout of
    # steering_delta) and [onset frames, ramp frames, hold]. None: every control is in steering_delta.
    steer_release_delta: torch.Tensor | None = None
    steer_release: List[float] | None = None
    # Span steering that follows the reading position (zonos3.tts.span_follow; control_span_follow "reading"): the
    # character range [start, end) of ``text`` of every steer_schedule row and the gate's [lead, lag] in text tokens.
    # None: the rows keep the character-rate schedule.
    steer_follow_chars: List[List[int]] | None = None
    steer_follow_gate: List[float] | None = None
    # Condition rows (cond3 finetunes): one bucket id per table, n_buckets = unknown; None = every row unknown.
    condition_buckets: List[int] | None = None
    # Optional accent adapter (by name) served at strength accent_strength:
    # base + accent_strength * adapter. None serves the base model.
    accent: str | None = None
    accent_strength: float = 1.0
    # Speech-edit infill (all three set together; sampling_params.end_token must
    # be "span"). The tokenizer replaces the plain text prompt with the edit
    # layout ending on the marker of span len(edit_fills): edit_codes are the
    # original codec frames [frames, n_codebooks], edit_spans ascending
    # [start_frame, length_frames] pairs, edit_fills the already generated fills.
    edit_codes: torch.Tensor | None = None
    edit_spans: List[List[int]] | None = None
    edit_fills: List[torch.Tensor] | None = None
    # LACI (zonos3.tts.laci): guard the generation against skips / hallucinations / truncation (the tokenizer keys the
    # alignment-head spec by ``language``).
    laci: bool = False
    # LACI duration target (zonos3.tts.laci.DurationTarget, needs laci): the take's length in generated rows, and
    # whether a miss is retaken in the speaking-rate bucket the measured ratio asks for (else only recorded).
    laci_target_frames: float | None = None
    laci_adjust_rate: bool = True
    # LACI over an extended speaking-rate scale (zonos3.tts.laci.RateBuckets; a LACI take whose scale has tempo-steered
    # buckets, README "Extended speaking-rate buckets"): the scale's edges and first bucket, the take's bucket on it,
    # and the request's steering_delta in every bucket of the scale (the take's own steering, plus the bucket's
    # tempo dose in a steered one), which a retake in another bucket switches to. None: the checkpoint's row buckets.
    laci_rate_edges: List[float] | None = None
    laci_rate_first: int = 0
    laci_rate_bucket: int | None = None
    laci_rate_deltas: List[torch.Tensor | None] | None = None
    # A short text (zonos3.tts.plan.Rules.short_text): the tokenizer worker generates a normalized text that qualifies
    # with its pause marker and filler under LACI (zonos3.tts.short_text.prompt).
    short_text: bool = False


@dataclass
class TTSNormalizeMsg(BaseTTSTokenizerMsg):
    """A request's text to normalize ahead of its TTSTokenizeMsg (Frontend -> Tokenizer, no reply): the tokenizer
    worker runs the (memoized) normalizer while the frontend still resolves the request's rules and speaker."""

    text: str
    language: str


@dataclass
class TTSDecodeCodesMsg(BaseTTSTokenizerMsg):
    """Decode complete codec frames [frames, n_codebooks] (Frontend -> Tokenizer -> TTSAudioReply)."""

    uid: int
    codes: torch.Tensor
    decoder: str | None = None  # a served decoder id; None = the server default


@dataclass
class TTSDetokenizeMsg(BaseTTSTokenizerMsg):
    """TTS output frame to vocoder (Scheduler -> Tokenizer).

    Contains audio codes for one generated frame.
    """

    uid: int
    audio_codes: List[int]  # [cb0, cb1, ..., cb8] for one frame
    finished: bool
    eos_frame: int | None = None
    # First row of a request only (2026-10-17): the vocoder's decode context (reference tail) and prefilled rows.
    decode_context_codes: torch.Tensor | None = None
    prefill_rows: torch.Tensor | None = None
    # Request's progressive-decode mode (TTSSamplingParams.stream_decode), read by the vocoder.
    stream_decode: str | None = None
    # Request's early partial-frame decode (TTSSamplingParams.stream_codebooks / _scope), read by the vocoder.
    stream_codebooks: int = STREAM_CODEBOOKS
    stream_codebooks_scope: str = STREAM_CODEBOOKS_SCOPE
    # Request's audio decoder (TTSSamplingParams.decoder), read by the vocoder.
    decoder: str | None = None
    # The request's sampling_params.end_token. "span" fills are returned as
    # delayed codec rows (TTSCodesReply), never vocoded on their own.
    end_token: str = "utterance"
    # Final row of a LACI request only: its record (attempts, end, failures, ...; zonos3.tts.laci.LaciSession.record).
    laci: Dict | None = None


@dataclass
class BaseTTSBackendMsg:
    """Base class for TTS backend messages."""

    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseTTSBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTTSBackendMsg(BaseTTSBackendMsg):
    """Batch of TTS backend messages."""

    data: List[BaseTTSBackendMsg]


@dataclass
class TTSUserMsg(BaseTTSBackendMsg):
    """Prompt-tokenized TTS request to scheduler (Tokenizer -> Scheduler).

    Contains 2D token tensor in unpacked format.
    """

    uid: int
    input_ids: torch.Tensor  # 2D tensor (seq_len, frame_width)
    sampling_params: TTSSamplingParams
    # Exact number of delayed reference rows before the text, including its pad tail.
    speaker_reference_prefix_length: int = 0
    # Optional speaker embedding for voice cloning; shape: (speaker_embedding_dim,)
    speaker_embedding: torch.Tensor | None = None
    # Token position in the prompt sequence where speaker embedding is injected.
    speaker_token_position: int = -1
    # Hidden-space steering delta (see TTSTokenizeMsg).
    steering_delta: torch.Tensor | None = None
    # Span steering schedule (see TTSTokenizeMsg).
    steer_schedule: torch.Tensor | None = None
    # Steer then hold (see TTSTokenizeMsg).
    steer_release_delta: torch.Tensor | None = None
    steer_release: List[float] | None = None
    # Span steering that follows the reading position (zonos3.tts.span_follow): the probe's head-spec key (the
    # request's language in the scheduler's LACI heads), each steer_schedule row's text-span positions [a, b) and the
    # gate's [lead, lag]. None: the character-rate schedule.
    steer_follow: str | None = None
    steer_follow_tokens: List[List[int]] | None = None
    steer_follow_gate: List[float] | None = None
    # Condition rows (cond3 finetunes): bucket ids per table and the prompt position of the first row; None = none.
    condition_values: List[int] | None = None
    condition_token_start: int = -1
    # Decode context / silence prefill (2026-10-17): what precedes the generated rows for the vocoder, relayed by
    # the scheduler to the request's first detokenize message (the detokenizer is its own process).
    decode_context_codes: torch.Tensor | None = None  # undelayed [K, n_codebooks] int32 reference tail
    prefill_rows: torch.Tensor | None = None  # [T, n_codebooks + 1] int32 sheared silence rows
    # Accent adapter name and strength (see TTSTokenizeMsg).
    accent: str | None = None
    accent_strength: float = 1.0
    # Offline-only (TTSLLM.generate hands Python objects to the in-process scheduler): alignment probe.
    align_probe: object | None = None
    # Trailing prompt rows that count as repetition-penalty history (continuations from generated rows).
    history_rows: int = 0
    # LACI (zonos3.tts.laci): the alignment-head spec key (language code, e.g. "en_us") or None = off.
    laci: str | None = None
    # LACI duration target (see TTSTokenizeMsg): the scheduler builds the session's DurationTarget from these.
    laci_target_frames: float | None = None
    laci_adjust_rate: bool = True
    # LACI extended speaking-rate scale (see TTSTokenizeMsg).
    laci_rate_edges: List[float] | None = None
    laci_rate_first: int = 0
    laci_rate_bucket: int | None = None
    laci_rate_deltas: List[torch.Tensor | None] | None = None
    # A short text (zonos3.tts.short_text): its filler's first text-span token (needs laci); None = an ordinary request.
    laci_short_text: int | None = None


@dataclass
class BaseTTSFrontendMsg:
    """Base class for TTS frontend messages."""

    @staticmethod
    def encoder(msg: BaseTTSFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTTSFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTTSFrontendMsg(BaseTTSFrontendMsg):
    """Batch of TTS frontend messages."""

    data: List[BaseTTSFrontendMsg]


@dataclass
class TTSAudioReply(BaseTTSFrontendMsg):
    """Audio chunk reply to frontend (Tokenizer -> Frontend).

    Contains PCM audio data for streaming playback.
    """

    uid: int
    audio_data: bytes  # PCM audio chunk (float32, 44.1kHz)
    finished: bool
    sample_rate: int = SAMPLE_RATE
    # Final chunk of a LACI request only: the scheduler's LACI record (TTSDetokenizeMsg.laci).
    laci: Dict | None = None


@dataclass
class TTSCodesReply(BaseTTSFrontendMsg):
    """Codec frames for the frontend (Tokenizer -> Frontend): a "span" generation's delayed rows (excluding the
    finishing row, as the offline engine does) plus eos_frame."""

    uid: int
    codes: torch.Tensor
    finished: bool = True
    eos_frame: int | None = None
