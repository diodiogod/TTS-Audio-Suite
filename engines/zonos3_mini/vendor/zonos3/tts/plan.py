"""The v2 GenerateRequest rules and prompt plan of every Python server (the reference, the MLX and the sgl server).

One request goes through three steps, in the order the sgl frontend has always checked them (servers/api/openapi.yaml; the
rules R1-R17 the ggml and in-browser servers port natively and ``model/tests/fixtures/request_errors.json`` pins):

1. ``request_head``: the speaker mode (R2) and the condition rows (R3) -- everything the server needs
   before it resolves the request's speaker inputs (uploads, voices, inline codes) into a ``Speaker``.
2. ``request_rules``: language (R4), the max_tokens budget (R5), the silence prefill (R6), the accent (R9), steering
   controls (R10), the span-follow gate (R11), CFG (R12), text markers: pronunciation groups, the start marker, pause
   markers (R13), LACI (R14), the seed (R16) and short texts (R17).
   The sgl frontend stops here (its tokenizer worker builds the rows in another process).
3. ``build_prompt``: the prompt rows -- condition rows, the reference prefix (R7) or speaker row, the text and silence
   rows, the decode context (R8) and the CFG twin -- and the budget shrunk to the context left (R5). A short text
   (R17, zonos3.tts.short_text) gets its pause marker and filler sentence here, after normalization.

``plan_prompt`` runs the three for a server whose speaker resolution is synchronous. A server supplies what the rules
read through ``RulesContext`` and, for the rows, ``PromptContext``; every refusal is a ValueError whose message is the
HTTP 400 detail. The defaults and limits come from the generation registry (``zonos3.generation``).
"""

from __future__ import annotations

import bisect
import math
import os
import secrets
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import torch

from zonos3 import languages
from zonos3.generation import (
    DECODE_CONTEXT_FRAMES,
    SAMPLING,
    SILENCE_PREFILL_DEFAULT_FRAMES,
    SILENCE_PREFILL_MAX_FRAMES,
    STREAM_CODEBOOKS,
    STREAM_CODEBOOKS_SCOPE,
)
from zonos3.speaker_cfg import speaker_cfg_twin
from zonos3.tts.prompt import TTSPromptBuilder, silence_rows_from_codes
from zonos3.tts.span_follow import FollowConfig
from zonos3.tts.steering import SteeringBundle, no_steering_message, request_release, resolve_controls
from zonos3.tts.superbpe import (
    NO_PAUSE_TOKENS,
    NO_START_TOKEN,
    PAUSE_MARKER_RE,
    PHONEME_CLOSE,
    PHONEME_OPEN,
    START_MARKER,
    START_NOT_FIRST,
    payload_ids,
)

# v2 GenerateRequest (servers/api/openapi.yaml): every field and its default.
GENERATE_DEFAULTS: dict = {
    "text": None,
    "language": "en_us",
    "text_normalization": True,
    "silence_prefill_frames": None,
    "start_immediately": False,
    "decode_context": False,
    "decode_context_frames": DECODE_CONTEXT_FRAMES,
    **SAMPLING,
    "max_tokens": None,
    "fade_out_ms": 0.0,
    "seed": None,
    "stream": True,
    "stream_decode": None,
    "stream_codebooks": STREAM_CODEBOOKS,
    "stream_codebooks_scope": STREAM_CODEBOOKS_SCOPE,
    "format": "pcm",
    "speaker_mode": None,
    "condition_buckets": None,
    "condition_values": None,
    "headroom_db": None,
    "decoder": None,
    "speaker_audio_base64": None,
    "speaker_audio_name": None,
    "speaker_reference_codes": None,
    "speaker_embedding_base64": None,
    "speaker_embedding_name": None,
    "speaker_embedding_id": None,
    "speaker_blend_embedding_id_a": None,
    "speaker_blend_embedding_id_b": None,
    "speaker_blend_t": None,
    "controls_cfg_scale": 1.0,
    "speaker_cfg_scale": 1.0,
    "accent": None,
    "accent_strength": 1.0,
    "controls": None,
    "control_spans": None,
    "control_span_rate": None,
    "control_span_follow": "rate",
    "control_span_lead": None,
    "control_span_lag": None,
    "steer_hold": None,
    "steer_onset_seconds": None,
    "steer_ramp_seconds": None,
    "laci": False,
    "short_text": "auto",
}
NO_CONDITION_ROWS = "This checkpoint has no condition rows (loudness / fmax / speaking rate)"


class Request(dict):
    """A v2 GenerateRequest with its defaults filled in (attribute access)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def parse_request(request: Mapping) -> Request:
    """``request`` (a JSON body or a schema model's ``model_dump()``) with the defaults of the fields it leaves out."""
    if isinstance(request, Request):
        return request
    unknown = sorted(set(request) - set(GENERATE_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown GenerateRequest field(s) {unknown}")
    if not isinstance(request.get("text"), str):
        raise ValueError("GenerateRequest.text is required")
    return Request({**GENERATE_DEFAULTS, **request})


# ----------------------------------------------------------------------------------------------------- contexts


class RulesContext(Protocol):
    """What the request rules read from a server: the checkpoint, the loaded companions and the server limits."""

    n_codebooks: int
    codebook_size: int
    hidden_size: int  # the hidden-delta width per steered source
    condition_names: Sequence[str]  # the condition rows in prompt order; empty without condition rows
    condition_bucket_counts: Sequence[int]  # per condition row: ids 0..n-1, n = unknown
    condition_edges: Mapping[str, Sequence[float]]  # raw value -> bucket edges per name
    condition_flags: Mapping[str, Mapping]  # flag rows: name -> {"values": [...]} (default [0, 1])
    speaker_modes: Sequence[str]  # "reference" (the prefix) and / or "embedding" (the speaker row), default first
    text_tokenizer_spec: dict | None  # SuperBPE spec (phoneme tokens); None = byte text
    accent_names: Sequence[str]  # the loaded accent adapters
    steer_bundle: SteeringBundle | None
    steer_hint: str  # how this server loads a steering bundle (in the refusal of `controls` without one)
    steering_sources: tuple[int, ...]  # the Full AttnRes bank sources hidden deltas steer, ascending
    laci_languages: Collection[str]  # languages with LACI alignment heads
    frame_rate: float  # codec frames per second (steer_release onset / ramp frames)
    max_frames: int  # the server's frame budget (R5)
    reference_max_frames: int  # longer reference codes keep their first frames (R7)
    silence_available: bool  # the codec has codec-matched silence frames (R6)


class PromptContext(RulesContext, Protocol):
    """What the prompt rows read besides the rules."""

    audio_pad_id: int
    text_vocab: int
    span_end_id: int | None
    max_seqlen: int  # the context capacity (R5)

    def prompt_builder(self) -> TTSPromptBuilder: ...

    def normalize_text(self, text: str, language: str, enabled: bool) -> str: ...

    def silence_frame(self) -> torch.Tensor | None:
        """One codec-matched silence frame [n_codebooks] (called only when ``silence_available``)."""
        ...


@dataclass
class Speaker:
    """A request's speaker inputs as the server resolved them for its mode: the speaker-row embedding or the reference
    prefix codes (``reference_codes``: inline ones already checked and cut to the reference limit, or the server's
    encode of a reference clip)."""

    mode: str | None
    embedding: torch.Tensor | None = None
    reference_codes: torch.Tensor | None = None


# ----------------------------------------------------------------------------------------------------- the rules


def resolve_speaker_mode(modes: Sequence[str], value: str | None) -> str | None:
    """R2: the request's conditioning path. None = the checkpoint's first mode (None without any); an explicit value
    must be one of ``modes`` exactly."""
    modes = list(modes)
    if value is None:
        return modes[0] if modes else None
    if value not in modes:
        raise ValueError(f"speaker_mode={value!r} is not available for this checkpoint (supported: {modes})")
    return value


def resolve_condition_buckets(ctx: RulesContext, buckets, values) -> list[int] | None:
    """R3: the bucket id of every condition row. Explicit ids win, raw values map through the edges (bucket i =
    [edge_{i-1}, edge_i), the training loader's bisect_right) or, on a flag row, to ``1 if value >= 0.5 else 0``;
    anything else is unknown (the row's bucket count, ``condition_bucket_counts[i]``). None for a checkpoint without
    condition rows, which refuses a request that names any."""
    names = list(ctx.condition_names)
    if not names:
        if buckets not in (None, {}, []) or values not in (None, {}, []):
            raise ValueError(NO_CONDITION_ROWS)
        return None
    flags = ctx.condition_flags

    def as_map(raw, what):
        if raw is None:
            return {}
        if isinstance(raw, Mapping):
            unknown = [str(k) for k in raw if str(k) not in names]
            if unknown:
                raise ValueError(f"{what}: unknown condition name(s) {unknown}; expected {names}")
            return {str(k): v for k, v in raw.items()}
        raw = list(raw)
        if len(raw) != len(names):
            raise ValueError(f"{what}: expected {len(names)} entries in the order {names}")
        return dict(zip(names, raw))

    ids_by_name = as_map(buckets, "condition_buckets")
    values_by_name = as_map(values, "condition_values")
    out = []
    for i, name in enumerate(names):
        n_buckets = int(ctx.condition_bucket_counts[i])
        if ids_by_name.get(name) is not None:
            try:
                bucket = int(ids_by_name[name])
            except (TypeError, ValueError):
                raise ValueError(f"condition_buckets[{name}] must be an integer") from None
            flag = flags.get(name)
            if flag is not None and bucket != n_buckets:
                allowed = list(flag.get("values") or [0, 1])
                if bucket not in [int(v) for v in allowed]:
                    raise ValueError(
                        f"condition_buckets[{name}] is a flag: one of {allowed} or {n_buckets} (= unknown)"
                    )
            if not 0 <= bucket <= n_buckets:
                raise ValueError(
                    f"condition_buckets[{name}] must be in [0, {n_buckets - 1}] or {n_buckets} (= unknown)"
                )
        elif values_by_name.get(name) is not None:
            try:
                value = float(values_by_name[name])
            except (TypeError, ValueError):
                raise ValueError(f"condition_values[{name}] must be a number") from None
            if not math.isfinite(value):
                raise ValueError(f"condition_values[{name}] must be finite")
            if name in flags:
                bucket = 1 if value >= 0.5 else 0  # a flag row: any truthy value means "yes"
            else:
                edges = ctx.condition_edges.get(name)
                if not edges:
                    raise ValueError(
                        f"the checkpoint carries no edges for {name}; send condition_buckets[{name}] instead"
                    )
                bucket = int(bisect.bisect_right(list(edges), value))
        else:
            bucket = n_buckets
        out.append(bucket)
    return out


def max_tokens_budget(requested, server_max: int) -> int:
    """R5 before the prompt exists: the request's positive budget, at most the server's (its frame budget)."""
    if requested is None:
        return int(server_max)
    if int(requested) <= 0:
        raise ValueError("max_tokens must be positive.")
    return min(int(requested), int(server_max))


def fit_max_tokens(budget: int, prompt_len: int, capacity: int) -> int:
    """R5 once the prompt exists: the budget shrunk to the context the prompt leaves (a prompt that fills the context
    is refused)."""
    if prompt_len >= capacity:
        raise ValueError(f"TTS input len {prompt_len} exceeds {capacity}")
    return min(int(budget), int(capacity) - int(prompt_len))


@dataclass(frozen=True)
class SilenceDefault:
    """The server's silence prefill for a request without silence_prefill_frames: ZONOS3_SILENCE_PREFILL_FRAMES frames
    (the registry default, 3 = 0.2 s, when unset or not an integer), off with ZONOS3_SILENCE_PREFILL_DEFAULT=0 / false /
    no / off or 0 frames."""

    frames: int
    on: bool

    @property
    def length(self) -> int:
        return self.frames if self.on else 0


def silence_prefill_default() -> SilenceDefault:
    try:
        frames = max(0, int(os.environ.get("ZONOS3_SILENCE_PREFILL_FRAMES", str(SILENCE_PREFILL_DEFAULT_FRAMES))))
    except ValueError:
        frames = SILENCE_PREFILL_DEFAULT_FRAMES
    on = os.environ.get("ZONOS3_SILENCE_PREFILL_DEFAULT", "1").strip().lower() not in ("0", "false", "no", "off")
    return SilenceDefault(frames, on and frames > 0)


def silence_prefill_frames(requested, *, available: bool = True) -> int:
    """R6: the silence prefill length of a request. An explicit silence_prefill_frames must be an integer in
    0..registry max (never clamped); None = the server default (``silence_prefill_default``); 0 on a codec without
    silence frames."""
    if requested is not None:
        try:
            frames = int(requested)
            valid = not isinstance(requested, bool) and frames == requested
        except (TypeError, ValueError):
            valid = False
        if not valid or not 0 <= frames <= SILENCE_PREFILL_MAX_FRAMES:
            raise ValueError(f"silence_prefill_frames must be 0..{SILENCE_PREFILL_MAX_FRAMES}")
    if not available:
        return 0
    return silence_prefill_default().length if requested is None else frames


def reference_frame_limit(max_seconds: float, hop: int, sample_rate: int) -> int:
    """R7 on a torch checkpoint: the reference prefix frames its ``speaker_reference_max_seconds`` holds."""
    return int(float(max_seconds) * int(sample_rate) // int(hop))


def reference_codes(ctx: RulesContext, codes) -> torch.Tensor | None:
    """R7: inline ``speaker_reference_codes`` ([frames, n_codebooks] codec ids) as int32, cut to their first
    ``ctx.reference_max_frames`` frames; None when the request has none."""
    if codes is None:
        return None
    try:
        tensor = torch.as_tensor(codes, dtype=torch.int32)
    except (TypeError, ValueError, RuntimeError):
        raise ValueError("speaker_reference_codes must be a non-empty [frames, codebooks] list") from None
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] != int(ctx.n_codebooks):
        raise ValueError("speaker_reference_codes must be a non-empty [frames, codebooks] list")
    if bool((tensor < 0).any()):
        raise ValueError("speaker_reference_codes must be non-negative")
    if bool((tensor >= int(ctx.codebook_size)).any()):
        raise ValueError(f"speaker_reference_codes must be codec ids in [0, {int(ctx.codebook_size) - 1}]")
    return truncate_reference(tensor, ctx.reference_max_frames)


def truncate_reference(codes: torch.Tensor, limit: int) -> torch.Tensor:
    """R7: reference codes [frames, n_codebooks] cut to their first ``limit`` frames (never refused)."""
    return codes[: int(limit)].contiguous()


def decode_context_codes(codes: torch.Tensor | None, decode_context: bool, frames) -> torch.Tensor | None:
    """R8: the reference tail decoded as left context: the last ``frames`` (None or 0 = the registry's) frames of the
    reference codes, at most all of them; None without decode_context or reference codes."""
    if not decode_context or codes is None:
        return None
    k = max(0, min(int(frames or DECODE_CONTEXT_FRAMES), int(codes.shape[0])))
    return codes[int(codes.shape[0]) - k :] if k > 0 else None


def resolve_accent(names: Sequence[str], accent, strength) -> tuple[str, float] | None:
    """R9: (adapter, strength) of a request's accent; None for the base model."""
    if accent is None or accent == "" or accent == "none":
        return None
    if accent not in names:
        loaded = ", ".join(names) or "none (load accent adapters)"
        raise ValueError(f"Unknown accent {accent!r}; loaded accents: {loaded}.")
    strength = float(strength)
    if not math.isfinite(strength):
        raise ValueError(f"accent_strength must be finite, got {strength!r}.")
    return accent, strength


SHORT_TEXT_MODES = ("auto", "off")
START_NEEDS_TOKEN = "start_immediately needs a checkpoint with a start token"
EDIT_NO_MARKERS = "Speech editing takes no start or pause markers: remove U+E002 / U+E003-U+E00A from target_text"


def validate_text_markers(spec: dict | None, text: str, *, edit: bool = False) -> None:
    """R13: the text's pronunciation groups (U+E000 phonemes U+E001) need phoneme tokens and must parse; the start
    marker (U+E002) needs a start token and must open the text, once; pause markers (U+E003 + bucket) need pause tokens
    and a bucket the checkpoint has. ``edit``: a speech-edit target text, which takes no start or pause markers."""
    spec = spec or {}
    phonemes, start, pauses = spec.get("phonemes"), spec.get("start"), spec.get("pauses")
    has_start = START_MARKER in text
    pause = PAUSE_MARKER_RE.search(text)
    if edit and (has_start or pause):
        raise ValueError(EDIT_NO_MARKERS)
    open_, close = (
        (PHONEME_OPEN, PHONEME_CLOSE) if phonemes is None else (phonemes["open_marker"], phonemes["close_marker"])
    )
    has_groups = open_ in text or close in text
    if has_groups and phonemes is None:
        raise ValueError("This checkpoint has no phoneme tokens: remove the U+E000 / U+E001 pronunciation groups")
    if has_start and start is None:
        raise ValueError(NO_START_TOKEN)
    if pause and pauses is None:
        raise ValueError(NO_PAUSE_TOKENS)
    if has_start and (not text.startswith(START_MARKER) or START_MARKER in text[1:]):
        raise ValueError(START_NOT_FIRST)
    if pause:
        n = len(pauses["ids"])
        for match in PAUSE_MARKER_RE.finditer(text):
            if ord(match.group()) - 0xE003 >= n:
                raise ValueError(f"Pause marker U+{ord(match.group()):04X} outside the {n} pause buckets")
    if has_groups:
        try:
            payload_ids(text, spec)
        except ValueError as exc:
            raise ValueError(f"Pronunciation groups: {exc}") from exc


def start_silence(requested, start: bool):
    """R6 with a start marker: no silence prefill unless the request asks for one (an explicit value wins)."""
    return 0 if start and requested is None else requested


def with_start_marker(text: str) -> str:
    """``text`` opening with the start marker once (start_immediately on a text that may already have typed it)."""
    return text if text.startswith(START_MARKER) else START_MARKER + text


def check_laci(ctx: RulesContext, req: Request, language: str, *, duration_target: bool = False) -> None:
    """R14: a laci=true request needs alignment heads for its language and no CFG; a LACI duration target (a
    duration-target take, zonos3.server.plugins.LaciTakeRequest) needs laci=true."""
    if not req.laci:
        if duration_target:
            raise ValueError("A LACI duration target needs laci=true")
        return
    heads = ctx.laci_languages
    if not heads:
        raise ValueError("LACI is not enabled on this server (start it with --laci-heads LANG=PATH)")
    if language not in heads:
        raise ValueError(f"No LACI alignment heads for language {language!r} (configured: {', '.join(sorted(heads))})")
    if float(req.speaker_cfg_scale) != 1.0 or float(req.controls_cfg_scale) != 1.0:
        raise ValueError("LACI needs speaker_cfg_scale = controls_cfg_scale = 1")


@dataclass(frozen=True)
class RequestHead:
    """What the server needs before it resolves the speaker inputs: the speaker mode and the condition bucket ids."""

    mode: str | None
    condition_ids: list[int] | None


def request_head(ctx: RulesContext, req: Request) -> RequestHead:
    """R2 the speaker mode, R3 the condition rows."""
    mode = resolve_speaker_mode(ctx.speaker_modes, req.speaker_mode)
    return RequestHead(mode, resolve_condition_buckets(ctx, req.condition_buckets, req.condition_values))


@dataclass
class Rules:
    """A request that passed every rule, with what the rules resolved (everything but the prompt rows)."""

    request: Request
    speaker: Speaker
    language: str
    condition_ids: list[int] | None
    max_tokens: int  # the budget before the prompt shrinks it (R5)
    silence_frames: int  # 0 = no silence prefill (R6)
    start: bool  # the prompt text opens with the start marker (start_immediately, or typed first)
    accent: tuple[str, float] | None
    embedding: torch.Tensor | None  # the speaker row's vector after the raw-space controls
    sources: tuple[int, ...]  # steered Full AttnRes bank sources
    hidden_delta: torch.Tensor | None  # constant hidden delta, flattened per source
    release_delta: torch.Tensor | None  # released hidden delta (steer then hold), same layout
    release: tuple[int, int, float] | None  # (onset frames, ramp frames, hold) of release_delta
    schedule: torch.Tensor | None  # span schedule [K, 3 + len(sources) * dim]
    follow_chars: list[list[int]] | None  # character range of every schedule row (control_span_follow "reading")
    follow_gate: list[float] | None  # [lead, lag]
    cfg_kind: str | None  # "speaker" | "controls"
    cfg_scale: float
    seed: int
    # R17: the request may run as a short text (zonos3.tts.short_text; the normalized text decides, build_prompt)
    short_text: bool = False


def request_rules(
    ctx: RulesContext,
    req: Request,
    head: RequestHead,
    speaker: Speaker,
    *,
    hidden_delta=None,
    release_delta=None,
    edit: bool = False,
    duration_target: bool = False,
) -> Rules:
    """R4-R17 for the speaker inputs the server resolved for ``head.mode``. ``hidden_delta``: a server-side steering
    delta added to the request's (a speech-edit region's); ``release_delta``: a server-side released delta (a
    speech-edit region's controls not exempt from the request's steer_release) added to the request's; ``edit``: a
    speech-edit region (no silence prefill, no span follow); ``duration_target``: the request carries a LACI duration
    target (a duration-target take)."""
    language = languages.normalize(req.language)
    max_tokens = max_tokens_budget(req.max_tokens, ctx.max_frames)
    start = bool(req.start_immediately) or req.text.startswith(START_MARKER)
    silence = (
        0
        if edit
        else silence_prefill_frames(start_silence(req.silence_prefill_frames, start), available=ctx.silence_available)
    )
    if not 1 <= int(req.stream_codebooks) <= int(ctx.n_codebooks):
        raise ValueError(f"stream_codebooks must be 1..{int(ctx.n_codebooks)}")
    accent = resolve_accent(ctx.accent_names, req.accent, req.accent_strength)
    embedding = speaker.embedding
    sources = tuple(ctx.steering_sources)
    delta = schedule = span_chars = released = None
    # R10: steer then hold (the bundle's steer_release defaults; hold 1 = constant controls), then the controls
    release = request_release(
        ctx.steer_bundle, req.steer_hold, req.steer_onset_seconds, req.steer_ramp_seconds, hint=ctx.steer_hint
    )
    if req.controls or req.control_spans:
        if ctx.steer_bundle is None:
            raise ValueError(no_steering_message(ctx.steer_hint, field="controls"))
        resolved = resolve_controls(
            ctx.steer_bundle,
            req.controls,
            list(req.control_spans or []),
            text=req.text,
            speaker_embedding=embedding,
            sources=sources,
            hidden_size=int(ctx.hidden_size),
            frames_per_char=req.control_span_rate,
            release_exempt=None if release is None else release.exempt,
        )
        embedding, delta, released, schedule, span_chars = (
            resolved.speaker_embedding,
            resolved.hidden_delta,
            resolved.release_delta,
            resolved.schedule,
            resolved.span_chars,
        )
    if hidden_delta is not None:
        delta = hidden_delta if delta is None else delta + hidden_delta
    if release_delta is not None:
        released = release_delta if released is None else released + release_delta
    release_frames = None if released is None or release is None else release.frames(float(ctx.frame_rate))
    if release_frames is None:
        released = None
    # R11: spans that follow the reading position need the language's alignment heads (else the character rate)
    follow_chars = follow_gate = None
    if req.control_span_follow == "reading" and span_chars and language in ctx.laci_languages and not edit:
        default = FollowConfig()
        follow_gate = [
            default.lead if req.control_span_lead is None else float(req.control_span_lead),
            default.lag if req.control_span_lag is None else float(req.control_span_lag),
        ]
        follow_chars = [[int(a), int(b)] for a, b in span_chars]
    # R12: one CFG twin per request; controls CFG needs a hidden delta and a speaker row
    cfg_kind, cfg_scale = None, 1.0
    if float(req.speaker_cfg_scale) != 1.0:
        has_reference = speaker.mode == "reference" and speaker.reference_codes is not None
        has_embedding = speaker.mode == "embedding" and embedding is not None
        if not (has_reference or has_embedding):
            raise ValueError(
                "Speaker CFG needs a speaker prompt: speaker_mode=reference with reference audio or codes, "
                "or speaker_mode=embedding with a clip or an embedding"
            )
        if float(req.controls_cfg_scale) != 1.0:
            raise ValueError("Speaker and controls CFG cannot be combined")
        cfg_kind, cfg_scale = "speaker", float(req.speaker_cfg_scale)
    elif (delta is not None or released is not None) and embedding is not None and float(req.controls_cfg_scale) != 1.0:
        cfg_kind, cfg_scale = "controls", float(req.controls_cfg_scale)
    validate_text_markers(ctx.text_tokenizer_spec, req.text, edit=edit)
    if req.start_immediately and (ctx.text_tokenizer_spec or {}).get("start") is None:
        raise ValueError(START_NEEDS_TOKEN)
    check_laci(ctx, req, language, duration_target=duration_target)
    seed = int(req.seed) if req.seed is not None else secrets.randbelow(2**31)
    # R17: a short text runs under LACI on the language's heads, alone (no CFG twin) and unsteered by spans
    if req.short_text not in SHORT_TEXT_MODES:
        raise ValueError("short_text must be 'auto' or 'off'")
    short = (
        req.short_text == "auto"
        and language in ctx.laci_languages
        and not (edit or duration_target or req.control_spans)
        and float(req.speaker_cfg_scale) == 1.0
        and float(req.controls_cfg_scale) == 1.0
    )
    return Rules(
        request=req,
        speaker=speaker,
        language=language,
        condition_ids=head.condition_ids,
        max_tokens=max_tokens,
        silence_frames=silence,
        start=start,
        accent=accent,
        embedding=embedding,
        sources=sources,
        hidden_delta=delta,
        release_delta=released,
        release=release_frames,
        schedule=schedule,
        follow_chars=follow_chars,
        follow_gate=follow_gate,
        cfg_kind=cfg_kind,
        cfg_scale=cfg_scale,
        seed=seed,
        short_text=short,
    )


# ----------------------------------------------------------------------------------------------------- the rows


@dataclass
class Branch:
    """One LM input: the prompt rows, the speaker row (embedding replacing row ``speaker_row``; -1 = none), the
    condition rows (bucket ids from ``condition_token_start``), and whether the request's hidden steering (constant
    delta, span schedule / follower) applies to its generated frames."""

    input_ids: torch.Tensor
    speaker_embedding: torch.Tensor | None = None
    speaker_row: int = -1
    condition_values: list[int] | None = None
    condition_token_start: int = -1
    steered: bool = True

    def speaker_rows(self) -> dict | None:
        return None if self.speaker_embedding is None else {self.speaker_row: self.speaker_embedding}

    def condition_rows(self) -> dict | None:
        if self.condition_values is None or self.condition_token_start < 0:
            return None
        return {self.condition_token_start + i: int(v) for i, v in enumerate(self.condition_values)}


@dataclass
class TwinMsg:
    """The fields zonos3.speaker_cfg.speaker_cfg_twin reads from a request."""

    input_ids: torch.Tensor
    speaker_reference_prefix_length: int
    speaker_embedding: torch.Tensor | None
    speaker_token_position: int


@dataclass
class PromptPlan:
    request: Request
    text: str  # normalized text (a short text: with its pause marker and filler)
    language: str
    cond: Branch
    twin: Branch | None
    cfg_kind: str | None  # "speaker" | "controls"
    cfg_scale: float
    reference_prefix_length: int
    prefill_rows: torch.Tensor | None  # [P, n_codebooks + 1] silence rows ending the prompt
    context_codes: torch.Tensor | None  # [K, n_codebooks] reference tail decoded as left context
    sources: tuple[int, ...]  # steered Full AttnRes bank sources
    hidden_delta: torch.Tensor | None  # constant hidden delta, flattened per source
    release_delta: torch.Tensor | None  # released hidden delta (steer then hold), same layout
    release: tuple[int, int, float] | None  # (onset frames, ramp frames, hold) of release_delta
    schedule: torch.Tensor | None  # span schedule [K, 3 + len(sources) * dim]
    follow_tokens: list[list[int]] | None  # token ranges of the schedule rows (control_span_follow "reading")
    follow_gate: list[float] | None
    accent: tuple[str, float] | None  # adapter accent and strength
    max_tokens: int
    seed: int
    laci: str | None  # LACI head key (the language)
    short_text: int | None = None  # a short text: its filler's first text-span token (zonos3.tts.short_text)

    @property
    def speaker_rows(self) -> int:
        """Leading rows before the condition rows: the reference prefix, or the speaker row."""
        return self.reference_prefix_length + (1 if self.cond.speaker_embedding is not None else 0)

    def cond_json(self, branch: Branch | None = None) -> dict:
        """The ``.cond.json`` sidecar of ``zonos3-cli --build-prompt`` (servers/ggml/src/zonos3-cond-io.h)."""
        b = self.cond if branch is None else branch
        doc: dict = {"speaker_row": b.speaker_row if b.speaker_embedding is not None else -1}
        if doc["speaker_row"] >= 0:
            doc["speaker_embedding"] = [float(v) for v in b.speaker_embedding.reshape(-1).tolist()]
        doc["condition_row"] = b.condition_token_start if b.condition_values is not None else -1
        doc["condition_buckets"] = [int(v) for v in (b.condition_values or [])]
        # the speaker-CFG twin drops the reference prefix / speaker row; the controls twin keeps them
        doc["speaker_rows"] = 0 if b is not self.cond and self.cfg_kind == "speaker" else self.speaker_rows
        doc["prefill_frames"] = 0 if self.prefill_rows is None else int(len(self.prefill_rows))
        return doc


def text_rows(
    builder: TTSPromptBuilder,
    text: str,
    *,
    silence: torch.Tensor | None = None,
    edit=None,
    span_end_id: int | None = None,
    quality_buckets=None,
) -> torch.Tensor:
    """The text rows of a prompt and what follows them: a speech-edit region's layout (``edit``: its window
    ``codes``, ``spans`` and the ``fills`` so far; needs ``span_end_id``), else the silence prefill rows."""
    prompt = builder.build(text, quality_buckets=quality_buckets)
    if edit is not None:
        from zonos3.tts.edit import edit_prompt_tokens

        if span_end_id is None:
            raise ValueError("Speech editing needs a checkpoint with a span-end audio token")
        return edit_prompt_tokens(prompt, edit.codes, edit.spans, edit.fills, builder.config, span_end_id=span_end_id)
    return prompt if silence is None else torch.cat((prompt, silence.to(prompt.dtype)), dim=0)


@dataclass
class PromptRows:
    """A prompt in the training layout ``[reference prefix | speaker row][condition rows] text rows``."""

    input_ids: torch.Tensor
    reference_prefix_length: int
    speaker_row: int  # 0 with a speaker row, else -1
    condition_token_start: int  # the first condition row; -1 without condition rows


def prompt_rows(
    builder: TTSPromptBuilder,
    text_rows: torch.Tensor,
    *,
    condition_count: int,
    reference_codes: torch.Tensor | None,
    speaker_row: bool,
) -> PromptRows:
    """The rows in front of ``text_rows``: ``condition_count`` condition rows, the reference prefix of
    ``reference_codes`` and the speaker row (``speaker_row``), in that order towards the start."""
    t = text_rows
    if condition_count:
        t = torch.cat((builder.condition_rows(condition_count, dtype=t.dtype, device=t.device), t), dim=0)
    prefix_length = 0
    if reference_codes is not None:
        prefix = builder.speaker_reference_prefix(reference_codes)
        prefix_length = len(prefix)
        t = torch.cat((prefix.to(t.dtype), t), dim=0)
    if speaker_row:
        t = torch.cat((builder.speaker_slot(dtype=t.dtype, device=t.device), t), dim=0)
    start = prefix_length + (1 if speaker_row else 0) if condition_count else -1
    return PromptRows(t, prefix_length, 0 if speaker_row else -1, start)


def build_prompt(ctx: PromptContext, rules: Rules, *, edit=None) -> PromptPlan:
    """The prompt rows, conditioning and CFG twin of a request that passed ``request_rules``. ``edit``: a speech-edit
    region (``text_rows``); its rules must have been resolved with ``edit=True``."""
    req, speaker = rules.request, rules.speaker
    builder = ctx.prompt_builder()
    text = ctx.normalize_text(req.text, rules.language, bool(req.text_normalization))
    if rules.start:
        text = with_start_marker(text)
    short = None
    if rules.short_text:
        from zonos3.tts import short_text

        short = short_text.prompt(text, rules.language, builder.config.text_tokenizer_spec)
        if short is not None:
            text = short[0]
    silence = None
    if rules.silence_frames > 0:
        silence = silence_rows_from_codes(ctx.silence_frame(), rules.silence_frames, builder.config).to(torch.int32)

    follow_tokens = None
    if rules.follow_chars is not None:
        from zonos3.tts.span_follow import span_token_ranges

        follow_tokens = [
            list(r)
            for r in span_token_ranges(
                req.text, text, [tuple(x) for x in rules.follow_chars], builder.config.text_tokenizer_spec
            )
        ]
    codes = speaker.reference_codes
    embedding = rules.embedding
    rows = prompt_rows(
        builder,
        text_rows(builder, text, silence=silence, edit=edit, span_end_id=ctx.span_end_id),
        condition_count=len(rules.condition_ids or ()),
        reference_codes=codes,
        speaker_row=embedding is not None,
    )
    t = rows.input_ids
    cond = Branch(t.to(torch.int32), embedding, rows.speaker_row, rules.condition_ids, rows.condition_token_start)
    twin = None
    if rules.cfg_kind == "speaker":
        msg = TwinMsg(t, rows.reference_prefix_length, embedding, max(rows.speaker_row, 0))
        spec = speaker_cfg_twin(
            msg,
            t,
            n_codebooks=ctx.n_codebooks,
            audio_pad_id=ctx.audio_pad_id,
            text_vocab=ctx.text_vocab,
            speaker_token_position=max(rows.speaker_row, 0),
            condition_values=rules.condition_ids,
            condition_token_start=rows.condition_token_start,
            condition_unknowns=tuple(ctx.condition_bucket_counts),
        )
        twin = Branch(spec.input_ids.to(torch.int32), None, -1, spec.condition_values, spec.condition_token_start)
    elif rules.cfg_kind == "controls":
        twin = replace(cond, steered=False)
    prompt_len = max(len(cond.input_ids), 0 if twin is None else len(twin.input_ids))
    return PromptPlan(
        request=req,
        text=text,
        language=rules.language,
        cond=cond,
        twin=twin,
        cfg_kind=rules.cfg_kind,
        cfg_scale=rules.cfg_scale,
        reference_prefix_length=rows.reference_prefix_length,
        prefill_rows=silence,
        context_codes=decode_context_codes(codes, bool(req.decode_context), req.decode_context_frames),
        sources=rules.sources,
        hidden_delta=rules.hidden_delta,
        release_delta=rules.release_delta,
        release=rules.release,
        schedule=rules.schedule,
        follow_tokens=follow_tokens,
        follow_gate=rules.follow_gate,
        accent=rules.accent,
        max_tokens=fit_max_tokens(rules.max_tokens, prompt_len, ctx.max_seqlen),
        seed=rules.seed,
        laci=rules.language if req.laci or short is not None else None,
        short_text=None if short is None else short[1],
    )


def plan_prompt(
    ctx: PromptContext,
    request: Mapping,
    resolve_speaker: Callable[[Request, str | None], Speaker],
    *,
    hidden_delta=None,
    release_delta=None,
    edit=None,
    duration_target: bool = False,
) -> PromptPlan:
    """``request_head``, the server's ``resolve_speaker(request, mode)``, ``request_rules`` and ``build_prompt``."""
    req = parse_request(request)
    head = request_head(ctx, req)
    speaker = resolve_speaker(req, head.mode)
    rules = request_rules(
        ctx,
        req,
        head,
        speaker,
        hidden_delta=hidden_delta,
        release_delta=release_delta,
        edit=edit is not None,
        duration_target=duration_target,
    )
    return build_prompt(ctx, rules, edit=edit)
