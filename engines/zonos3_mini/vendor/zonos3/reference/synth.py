"""Reference synthesis of one v2 ``GenerateRequest``: the sgl server's request path in plain PyTorch.

``plan_prompt`` resolves the request's speaker inputs (an uploaded embedding file or clip, inline reference codes, a
clip encoded with the dual codec) and runs the shared request rules and prompt plan of every Python server
(``zonos3.tts.plan``) on the LM's config and the ``ReferenceAssets``. ``synthesize`` then generates with the TTS sampler
of zonos3_sgl.engine.sample (seeded per request), LACI and span-follow on the LM's alignment readout, and decodes the
whole utterance with the matched iSTFT codec as zonos3_sgl.tokenizer.vocoder does at end of stream. Speaker ids
(default voices, the session cache) are server state and are refused here.

v2 hard-wires the former defaults of the dropped legacy fields (no speaking-rate token, no quality / background /
accurate-mode rows): ``Zonos3LMConfig.from_params`` only accepts checkpoints without those rows, so the prompt simply has
none. ``format`` is ignored. With ``on_audio`` a take is decoded progressively (prefix-streamed as it is generated)
unless ``stream_decode`` is whole; a LACI take follows the request's resolved decode instead (``stream`` and
``stream_decode``): the progressive one streams the rows no rollback can change, the whole one holds every row until the
result is final.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from zonos3.generation import SAMPLE_RATE, SAMPLING
from zonos3.models.accent_format import AccentAdapter, list_accent_adapters, load_adapter
from zonos3.reference.accents import AccentState
from zonos3.reference.lm import LMCache, ProbeSpec, Zonos3LM
from zonos3.server.speech import prefix_stream_supported, resolve_stream_decode
from zonos3.speaker_cfg import guided_logits
from zonos3.tokenizer.textnorm import load_normalizer, normalize_request_text
from zonos3.tts import plan as P
from zonos3.tts.audio import output_pcm
from zonos3.tts.delay import apply_cold_start_delay
from zonos3.tts.laci import LaciConfig, LaciHeads, LaciSession, RateBuckets
from zonos3.tts.probe import locate_text_span
from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig
from zonos3.tts.sampler import active_repetition_window, clamp_sampling, sample_tts
from zonos3.tts.span_follow import FollowConfig, SpanFollower
from zonos3.tts.steering import SteeringBundle, load_bundle, released_delta, schedule_delta
from zonos3.tts.stream import StreamDecoder, StreamSettings

# ----------------------------------------------------------------------------------------------------- assets

_UNLOADED = object()


@dataclass
class ReferenceAssets:
    """Everything a request may use besides the LM: the steering bundle, accent adapters, LACI heads per language; the
    codec decoder, the speaker encoder and the text normalizer are loaded on first use."""

    steer_bundle: SteeringBundle | None = None
    accent_adapters: dict[str, AccentAdapter] = field(default_factory=dict)
    laci_heads: dict[str, LaciHeads] = field(default_factory=dict)
    _decoders: dict = field(default_factory=dict, repr=False)
    _speaker_encoders: dict = field(default_factory=dict, repr=False)
    _normalizer: object = field(default=_UNLOADED, repr=False)

    @classmethod
    def load(cls, *, steer_bundle=None, accent_adapters=None, laci_heads: dict | None = None) -> "ReferenceAssets":
        adapters = {}
        if accent_adapters:
            for name in list_accent_adapters(accent_adapters):
                adapters[name] = load_adapter(Path(accent_adapters) / f"{name}.pt")
        return cls(
            steer_bundle=load_bundle(steer_bundle) if steer_bundle else None,
            accent_adapters=adapters,
            laci_heads={key: LaciHeads.load(path) for key, path in (laci_heads or {}).items()},
        )

    def steering_sources(self) -> tuple[int, ...]:
        """The Full AttnRes bank sources hidden deltas steer, ascending (sgl Engine._init_steering)."""
        return () if self.steer_bundle is None else tuple(self.steer_bundle.hidden_sources)

    def grow(self, lm: Zonos3LM) -> Zonos3LM:
        """``lm`` grown to the vocabulary its ``/3`` adapters need (zonos3.models.accent_format), before any request."""
        for adapter in self.accent_adapters.values():
            if adapter.grown is not None:
                lm.grow(adapter.grown)
        return lm

    def accent_names(self) -> list[str]:
        return list(self.accent_adapters)

    def normalizer(self):
        """The text normalizer (None with ZONOS3_TTS_NORM=0), loaded on first use."""
        if self._normalizer is _UNLOADED:
            self._normalizer = load_normalizer()
        return self._normalizer

    def decoder(self, codec: str, device: torch.device, decoder: str | None = None):
        """A matched codec's iSTFT decoder (``decoder``: its id, None = the codec's default) on ``device``
        (zonos3.tokenizer.matched_codec)."""
        from zonos3.tokenizer import matched_codec

        key = (codec, matched_codec.decoder_id(codec, decoder), str(device))
        if key not in self._decoders:
            from zonos3.utils.hashing import sha256_file
            from zonos3.utils.release import resolve_file

            entry = matched_codec.entry(codec)
            actual = sha256_file(resolve_file(entry["checkpoint"]))
            if actual != entry["checkpoint_sha256"]:
                raise ValueError(f"codec generator checkpoint sha256 {actual} != {entry['checkpoint_sha256']}")
            self._decoders[key] = matched_codec.load_model(codec, device, decoder=key[1])
        return self._decoders[key]

    def speaker_encoder(self, codec: str | None, dim: int, device: torch.device):
        key = (codec, dim, str(device))
        if key not in self._speaker_encoders:
            from zonos3.models.speaker_cloning import create_speaker_embedder

            self._speaker_encoders[key] = create_speaker_embedder(codec, dim, device=str(device))
        return self._speaker_encoders[key]


# ----------------------------------------------------------------------------------------------------- the plan


def decode_base64_blob(data: str, field_name: str) -> bytes:
    payload = data.strip()
    if payload.startswith("data:"):
        if "," not in payload:
            raise ValueError(f"Invalid data URL for {field_name}.")
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError(f"{field_name} must be valid base64-encoded data.") from exc


def load_embedding_vector(embedding_bytes: bytes, *, expected_dim: int, file_name: str | None) -> torch.Tensor:
    try:
        loaded = np.load(io.BytesIO(embedding_bytes), allow_pickle=False)
    except Exception as exc:
        raise ValueError(
            f"{file_name or 'Speaker embedding file'} must be a valid .npy or .npz file saved without pickle."
        ) from exc
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "emb" in loaded.files:
                arr = loaded["emb"]
            elif len(loaded.files) == 1:
                arr = loaded[loaded.files[0]]
            else:
                raise ValueError("Embedding archive must contain an 'emb' array or exactly one array.")
        finally:
            loaded.close()
    else:
        arr = loaded
    if isinstance(arr, np.ndarray) and arr.dtype.names:
        if "emb" in arr.dtype.names:
            arr = arr["emb"]
        elif len(arr.dtype.names) == 1:
            arr = arr[arr.dtype.names[0]]
        else:
            raise ValueError("Structured embedding arrays must contain an 'emb' field.")
    arr = np.squeeze(np.asarray(arr, dtype=np.float32))
    if arr.ndim == 1:
        vector = arr
    elif arr.ndim == 2:
        if arr.shape[0] == 0:
            raise ValueError("Speaker embedding file is empty.")
        vector = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
    else:
        raise ValueError(f"Unsupported embedding array shape {tuple(arr.shape)}. Expected (D,) or (N, D).")
    vector = np.ascontiguousarray(vector, dtype=np.float32)
    if vector.shape[-1] != expected_dim:
        raise ValueError(
            f"Reference embedding dimension mismatch. Model expects {expected_dim}, "
            f"but loaded file has {vector.shape[-1]}."
        )
    return torch.from_numpy(vector).to(dtype=torch.float32, device="cpu")


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation of two speaker embeddings (``speaker_blend_t``), on the CPU in float32."""
    v0 = v0.detach().to(dtype=torch.float32, device="cpu").contiguous().view(-1)
    v1 = v1.detach().to(dtype=torch.float32, device="cpu").contiguous().view(-1)
    if v0.shape != v1.shape:
        raise ValueError(
            f"Cannot blend speaker embeddings with different shapes {tuple(v0.shape)} and {tuple(v1.shape)}."
        )
    blend = float(t)
    if blend < 0.0 or blend > 1.0:
        raise ValueError("speaker_blend_t must be between 0.0 and 1.0.")
    v0_n = v0 / (torch.linalg.vector_norm(v0) + 1e-8)
    v1_n = v1 / (torch.linalg.vector_norm(v1) + 1e-8)
    omega = torch.acos(torch.clamp(torch.dot(v0_n, v1_n), -1.0, 1.0))
    if float(omega) < 1e-6:
        return ((1.0 - blend) * v0 + blend * v1).contiguous()
    sin_omega = torch.sin(omega)
    mixed = (torch.sin((1.0 - blend) * omega) / sin_omega) * v0 + (torch.sin(blend * omega) / sin_omega) * v1
    return mixed.to(dtype=torch.float32, device="cpu").contiguous()


def silence_frame(codec: str, device) -> torch.Tensor:
    """The codec-matched silence frame (registry ``silence_frame``, else the dual encoder's steady state of 2 s of
    silence, as TTSVocoderManager.silence_frame_codes)."""
    from zonos3.tokenizer.matched_codec import entry

    frame = entry(codec).get("silence_frame")
    if frame is not None:
        return torch.tensor([int(v) for v in frame], dtype=torch.int32)
    from zonos3.tokenizer.dual_reference import encode_dual_reference

    codes = encode_dual_reference(torch.zeros(1, 2 * SAMPLE_RATE), SAMPLE_RATE, max_seconds=5.0, device=str(device))
    return codes[-2].to(torch.int32).clone()


def prompt_builder(lm: Zonos3LM) -> TTSPromptBuilder:
    """The LM's prompt builder: its checkpoint's tokenizer spec (path resolved), with the phonemes / start / pauses of
    its config (rows appended at load, ``Zonos3LM.append_text_rows``)."""
    from zonos3.tts.superbpe import load_spec

    c = lm.config
    spec = load_spec(lm.root) if lm.root is not None else None
    extra = {k: v for k in ("phonemes", "start", "pauses") if (v := (c.text_tokenizer_spec or {}).get(k)) is not None}
    if spec is not None and extra:
        spec = {**spec, **extra}
    return TTSPromptBuilder(
        TTSPromptConfig(
            n_codebooks=c.n_codebooks,
            codebook_sizes=c.codebook_sizes,
            audio_pad_id=c.audio_pad_id,
            text_vocab=c.text_vocab,
            text_tokenizer_spec=spec,
        )
    )


class ReferenceContext:
    """The reference as ``zonos3.tts.plan.PromptContext``: the LM's config, the assets and ``device`` (where an uploaded
    clip is encoded). The frame budget is the checkpoint's max_seqlen, the reference limit its
    ``speaker_reference_max_seconds``."""

    steer_hint = "start it with --steer-bundle"

    def __init__(self, lm: Zonos3LM, assets: ReferenceAssets, device: torch.device):
        from zonos3.tokenizer.matched_codec import entry

        c = lm.config
        self.config, self.assets, self.device = c, assets, device
        self.n_codebooks, self.codebook_size = c.n_codebooks, c.codebook_size
        self.audio_pad_id, self.text_vocab, self.span_end_id = c.audio_pad_id, c.text_vocab, c.span_end_id
        self.hidden_size = c.dim
        self.max_seqlen = self.max_frames = c.max_seqlen
        self.condition_names, self.condition_bucket_counts = c.condition_names, c.condition_bucket_counts
        self.condition_edges, self.condition_flags = c.condition_edges, {}
        # the reference's architecture always has the speaker row
        self.speaker_modes = (["reference"] if c.speaker_reference_prefix_enabled else []) + ["embedding"]
        self._builder = prompt_builder(lm)
        self.text_tokenizer_spec = self._builder.config.text_tokenizer_spec
        self.accent_names = assets.accent_names()
        self.steer_bundle, self.steering_sources = assets.steer_bundle, assets.steering_sources()
        self.laci_languages = tuple(assets.laci_heads)
        codec = entry(c.audio_codec)
        self.reference_max_frames = P.reference_frame_limit(c.speaker_reference_max_seconds, codec["hop"], SAMPLE_RATE)
        self.frame_rate = SAMPLE_RATE / int(codec["hop"])
        self.silence_available = codec.get("silence_frame") is not None or codec.get("family") == "dual"

    def prompt_builder(self) -> TTSPromptBuilder:
        return self._builder

    def normalize_text(self, text: str, language: str, enabled: bool) -> str:
        return normalize_request_text(text, language, self.assets.normalizer() if enabled else None)

    def silence_frame(self) -> torch.Tensor:
        return silence_frame(self.config.audio_codec, self.device)


def _speaker_embedding(lm: Zonos3LM, req: P.Request, assets: ReferenceAssets, device) -> torch.Tensor | None:
    """speaker_mode=embedding: an uploaded embedding file or clip (MECT encoder) -> the speaker row's vector."""
    c = lm.config
    if req.speaker_audio_base64 and req.speaker_embedding_base64:
        raise ValueError("Provide either a reference audio file or a saved embedding file, not both.")
    if req.speaker_embedding_base64:
        return load_embedding_vector(
            decode_base64_blob(req.speaker_embedding_base64, "speaker_embedding_base64"),
            expected_dim=c.speaker_embedding_dim,
            file_name=req.speaker_embedding_name,
        )
    if not req.speaker_audio_base64:
        return None
    from zonos3.tts.audio import _decode_audio_bytes

    wav, sample_rate = _decode_audio_bytes(decode_base64_blob(req.speaker_audio_base64, "speaker_audio_base64"))
    encoder = assets.speaker_encoder(c.speaker_embedding_codec, c.speaker_embedding_dim, device)
    with torch.inference_mode():
        output = encoder(wav, sample_rate)
    candidates = output if isinstance(output, tuple) else (output,)
    for candidate in candidates:
        candidate = candidate.squeeze(0).to(dtype=torch.float32, device="cpu")
        if candidate.numel() == c.speaker_embedding_dim:
            return candidate
    raise ValueError(f"Reference embedding dimension mismatch. Model expects {c.speaker_embedding_dim}.")


def _reference_codes(ctx: ReferenceContext, req: P.Request, mode: str | None) -> torch.Tensor | None:
    """speaker_mode=reference: inline prefix codes, or an uploaded clip encoded with the dual codec."""
    if req.speaker_reference_codes is not None:
        return P.reference_codes(ctx, req.speaker_reference_codes)
    if mode != "reference" or not req.speaker_audio_base64:
        return None
    device = torch.device(ctx.device)
    if device.type != "cuda":
        raise ValueError("encoding reference audio with the dual codec needs a CUDA device")
    from zonos3.tokenizer.dual_reference import encode_dual_reference
    from zonos3.tts.audio import _decode_audio_bytes

    wav, sample_rate = _decode_audio_bytes(decode_base64_blob(req.speaker_audio_base64, "speaker_audio_base64"))
    index = device.index if device.index is not None else torch.cuda.current_device()
    return encode_dual_reference(
        wav, sample_rate, max_seconds=ctx.config.speaker_reference_max_seconds, device=f"cuda:{index}"
    )


def _resolve_speaker(lm: Zonos3LM, ctx: ReferenceContext, req: P.Request, mode: str | None) -> P.Speaker:
    """The request's speaker inputs for ``mode``; speaker ids need a server's voices."""
    if req.speaker_embedding_id or req.speaker_blend_embedding_id_a or req.speaker_blend_embedding_id_b:
        raise ValueError("speaker ids need a server")
    if mode == "reference":
        if req.speaker_embedding_base64:
            raise ValueError(
                "This checkpoint conditions on raw reference audio and does not accept "
                "speaker embedding files or embedding blends."
            )
        embedding = None
    else:
        embedding = _speaker_embedding(lm, req, ctx.assets, ctx.device)
    return P.Speaker(mode, embedding, _reference_codes(ctx, req, mode))


@torch.inference_mode()
def plan_prompt(lm: Zonos3LM, request: dict, *, assets: ReferenceAssets, device=None) -> P.PromptPlan:
    """The prompt rows, conditioning and CFG twin the sgl server builds for ``request`` (no generation)."""
    ctx = ReferenceContext(lm, assets, torch.device(device if device is not None else lm.device))
    return P.plan_prompt(ctx, request, lambda req, mode: _resolve_speaker(lm, ctx, req, mode))


@dataclass
class EditLayout:
    """A speech-edit region's prompt inputs (``zonos3.tts.plan.text_rows``): the window's codes, its spans and the fills
    generated so far."""

    codes: torch.Tensor
    spans: list
    fills: list


@torch.inference_mode()
def edit_region(
    lm: Zonos3LM,
    request: dict,
    *,
    assets: ReferenceAssets,
    layout: EditLayout,
    min_frames: int,
    speaker_embedding: torch.Tensor | None = None,
    hidden_delta=None,
    release_delta=None,
    accents: AccentState | None = None,
    device=None,
) -> tuple[list[torch.Tensor], int | None]:
    """One speech-edit region (``zonos3.tts.edit_runtime.FillJob``): ``request`` (text, language, normalization,
    sampling, seed, max_tokens, condition buckets, steer-then-hold fields) planned over the edit layout behind the
    speaker row (``speaker_embedding``, already steered) and generated until the span-end token, masked for
    ``min_frames`` rows; ``hidden_delta``: the region's constant steering delta on its generated frames,
    ``release_delta``: its released one (steer then hold, from the fill's first frame). Returns the generated rows and
    the aligned span-end frame (None: the budget ran out)."""
    c = lm.config
    if c.span_end_id is None:
        raise ValueError("Speech editing needs a checkpoint with a span-end audio token (span_end_id)")
    device = torch.device(device if device is not None else lm.device)
    ctx = ReferenceContext(lm, assets, device)
    plan = P.plan_prompt(
        ctx,
        request,
        lambda req, mode: P.Speaker(mode, speaker_embedding, None),
        hidden_delta=hidden_delta,
        release_delta=release_delta,
        edit=layout,
    )
    sampling = replace(Sampling.from_request(plan.request), min_frames=int(min_frames))
    attempt = generate_rows(
        lm,
        plan.cond,
        sampling=sampling,
        seed=plan.seed,
        max_tokens=plan.max_tokens,
        stop_id=c.span_end_id,
        twin=plan.twin,
        cfg_scale=plan.cfg_scale,
        accents=accents,
        steer=steer_fn(plan, None),
    )
    return attempt.rows, attempt.eos_frame


# ----------------------------------------------------------------------------------------------------- generation


@dataclass
class Sampling:
    """Per-request sampler settings (zonos3_sgl.engine.sample.TTSSampler.prepare); the defaults are the registry's."""

    temperature: float = SAMPLING["temperature"]
    topk: int = SAMPLING["topk"]
    top_p: float = SAMPLING["top_p"]
    min_p: float = SAMPLING["min_p"]
    repetition_window: int = SAMPLING["repetition_window"]
    repetition_penalty: float = SAMPLING["repetition_penalty"]
    repetition_codebooks: int = SAMPLING["repetition_codebooks"]
    min_frames: int = 0

    @classmethod
    def from_request(cls, req: P.Request) -> "Sampling":
        return cls(
            float(req.temperature),
            int(req.topk),
            float(req.top_p),
            float(req.min_p),
            int(req.repetition_window),
            float(req.repetition_penalty),
            int(req.repetition_codebooks),
        )


class ProbeState:
    """The per-attempt probe state an observer reads and sets (sgl AlignProbe)."""

    def __init__(self, heads, text_start: int, text_len: int, mask=None):
        self.heads, self.text_start, self.text_len, self.mask = heads, text_start, text_len, mask
        self.steps, self.stop = 0, False

    def spec(self) -> ProbeSpec:
        return ProbeSpec(tuple(self.heads), self.text_start, self.text_len, self.mask)


@dataclass
class Attempt:
    rows: list[torch.Tensor]  # generated rows [n_codebooks + 1] (int32), the finishing row included
    eos_frame: int | None  # aligned EOS frame in generated rows


def sample_row(
    config,
    logits,
    sampling: Sampling,
    history: torch.Tensor,
    total_generated: int,
    history_rows: int,
    stop_id: int,
    generator: torch.Generator,
    stop_bias: float = 0.0,
) -> torch.Tensor:
    """One sampled row; ``config`` supplies n_codebooks, codebook_size, codebook_sizes, audio_pad_id and text_vocab.
    ``stop_bias``: added to the stop id's logit in every codebook before the filters (LACI's stop boost)."""
    c = config
    n_cb = c.n_codebooks
    device = logits.device
    s = clamp_sampling(sampling, n_codebooks=n_cb, codebook_size=c.codebook_size)
    active = active_repetition_window(s, min(total_generated + history_rows, len(history)))
    repetition_ids = repetition_penalties = None
    if active > 0:
        recent = history[len(history) - active :, :n_cb].to(device=device, dtype=torch.int64).transpose(0, 1)
        valid = (recent >= 0) & (recent < c.codebook_size)
        valid[s.repetition_codebooks :] = False
        repetition_ids = torch.where(valid, recent, torch.full_like(recent, -1)).unsqueeze(0)
        repetition_penalties = torch.tensor([s.repetition_penalty], dtype=torch.float32, device=device)
    stop_blocked = None
    if sampling.min_frames > 0:
        stop_blocked = torch.tensor([[total_generated < sampling.min_frames + cb for cb in range(n_cb)]], device=device)
    # the per-request scalars stay on the CPU: sample_tts reads them on the host
    return sample_tts(
        logits.float().unsqueeze(0),
        torch.tensor([s.temperature], dtype=torch.float32),
        torch.tensor([s.topk], dtype=torch.int32),
        torch.tensor([s.top_p], dtype=torch.float32),
        torch.tensor([s.min_p], dtype=torch.float32),
        codebook_size=c.codebook_size,
        audio_pad_id=c.audio_pad_id,
        stop_token_ids=torch.tensor([stop_id], dtype=torch.int64),
        repetition_token_ids=repetition_ids,
        repetition_penalties=repetition_penalties,
        codebook_sizes=c.codebook_sizes,
        text_vocab=c.text_vocab,
        generators=[generator],
        stop_blocked=stop_blocked,
        stop_bias=torch.tensor([stop_bias], dtype=torch.float32) if stop_bias else None,
    )[0].cpu()


def generate_rows(
    lm: Zonos3LM,
    cond: P.Branch,
    *,
    sampling: Sampling,
    seed: int,
    max_tokens: int,
    stop_id: int,
    twin: P.Branch | None = None,
    cfg_scale: float = 1.0,
    history_rows: int = 0,
    accents: AccentState | None = None,
    steer=None,
    probe: ProbeState | None = None,
    observer=None,
    on_row=None,
) -> Attempt:
    """One request's autoregressive generation (TTSScheduler prefill + decode steps, non-overlapped): ``steer(frame)``
    is the hidden delta (flattened per source) for generated frame ``frame`` or None; ``observer.on_step`` gets the probe
    readout of every decode step (LACI, span follow) and, when it has ``stop_boost()`` (a LaciSession), biases the stop
    id of the next row; ``on_row(row, eos_frame)`` sees every generated row."""
    c = lm.config
    n_cb, pad = c.n_codebooks, c.audio_pad_id
    device = lm.device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    prompt = cond.input_ids.to(torch.int32)
    has_audio_prefix = bool((prompt[-1, :n_cb] != pad).any())
    branches = [(cond, LMCache())] + ([] if twin is None else [(twin, LMCache())])

    def forward(branch: P.Branch, cache: LMCache, ids: torch.Tensor, frame: int | None, with_probe: bool):
        hidden = None
        if frame is not None and branch.steered and steer is not None:
            delta = steer(frame)
            if delta is not None:
                hidden = delta
        return lm(
            ids,
            cache,
            speaker_rows=branch.speaker_rows(),
            condition_rows=branch.condition_rows(),
            hidden_deltas=hidden,
            accents=accents,
            probe=probe.spec() if with_probe and probe is not None else None,
        )

    outs = [forward(b, cache, b.input_ids, None, False) for b, cache in branches]
    rows: list[torch.Tensor] = []
    history = prompt[:, :n_cb].clone()
    eos_frame, countdown = -1, -1
    remain = max_tokens
    step = 0
    while True:
        logits = outs[0].logits[-1].float()
        if twin is not None:
            logits = guided_logits(logits, outs[1].logits[-1].float(), cfg_scale)
        total_generated = len(rows)
        stop_bias = observer.stop_boost() if hasattr(observer, "stop_boost") else 0.0
        token = sample_row(
            lm.config, logits, sampling, history, total_generated, history_rows, stop_id, generator, stop_bias
        )
        apply_cold_start_delay(token, total_generated, n_codebooks=n_cb, pad_id=pad, has_audio_prefix=has_audio_prefix)
        if step > 0 and probe is not None and observer is not None:
            probe.steps += 1
            if not probe.stop:
                pos, m = outs[0].probe[-1]
                observer.on_step(probe, probe.steps, int(pos), float(m))
            if probe.stop:
                remain = 0
        rows.append(token)
        history = torch.cat((history, token[None, :n_cb]), dim=0)
        remain -= 1
        finished = remain <= 0  # a probe stop zeroed the remaining budget
        # EOS (TTSReq.check_eos): any stop token starts the delayed countdown
        if eos_frame < 0:
            cols = [i for i, tok in enumerate(token[:n_cb].tolist()) if tok == stop_id]
            if cols:
                eos_frame = max(0, total_generated - max(cols))
                countdown = n_cb + 1
        if countdown > 0:
            countdown -= 1
            if countdown == 0:
                finished = True
        if on_row is not None:
            on_row(token, eos_frame if eos_frame >= 0 else None, finished)
        if finished:
            return Attempt(rows, eos_frame if eos_frame >= 0 else None)
        step += 1
        frame = len(rows) + history_rows
        ids = token[None]
        outs = [forward(b, cache, ids, frame, i == 0) for i, (b, cache) in enumerate(branches)]


# ----------------------------------------------------------------------------------------------------- decoding


def _codec_decode(lm: Zonos3LM, assets: ReferenceAssets, device, decoder: str | None):
    """The checkpoint's matched codec through its decoder ``decoder`` (None = the default) on undelayed codes
    [frames, n_codebooks] -> float32 PCM (ValueError for codes outside the codebook)."""
    from zonos3.tokenizer.matched_codec import decode_model, entry

    c = lm.config
    codec = entry(c.audio_codec)
    model = assets.decoder(c.audio_codec, torch.device(device), decoder)

    def decode(codes: np.ndarray) -> np.ndarray:
        codes = torch.as_tensor(codes, dtype=torch.int64)
        if bool(((codes < 0) | (codes >= c.codebook_size)).any()):
            raise ValueError("decode rows hold non-codec tokens")
        return decode_model(model, codes.unsqueeze(0), codec)[0].numpy().astype(np.float32)

    return decode


def _codec_hop(lm: Zonos3LM) -> int:
    from zonos3.tokenizer.matched_codec import entry

    return int(entry(lm.config.audio_codec)["hop"])


def decode_rows(
    lm: Zonos3LM,
    assets: ReferenceAssets,
    rows: list,
    *,
    prefill_rows: torch.Tensor | None,
    eos_frame: int | None,
    context_codes: torch.Tensor | None,
    device,
    decoder: str | None = None,
) -> np.ndarray:
    """End-of-stream decode of TTSVocoderManager.decode_frames / _decode_final (a whole-mode StreamDecoder) through the
    decoder ``decoder`` (None = the default): the prefill rows open the stream, the EOS boundary is the earliest of the
    aligned EOS frame and every row's EOA column, the last overlap rows of the prefill are re-decoded with the rest, and
    the reference tail is decoded as left context at the stream start."""
    c = lm.config
    whole = StreamDecoder(
        _codec_decode(lm, assets, device, decoder),
        n_codebooks=c.n_codebooks,
        audio_pad_id=c.audio_pad_id,
        hop=_codec_hop(lm),
        prefill_rows=prefill_rows,
        context_codes=context_codes,
        mode="whole",
    )
    for row in rows:
        whole.push(row, eos_frame)
    return whole.finish()


class _Emit:
    """Streamed PCM out: through the output stage (gain + soft limiter), the last ``fade_out_ms`` held back and faded at
    the end of the take."""

    def __init__(self, on_audio, gain: float, fade_out_ms: float):
        self.on_audio, self.gain, self.fade_out_ms = on_audio, np.float32(gain), fade_out_ms
        self.hold = int(SAMPLE_RATE * max(0.0, fade_out_ms) / 1000.0)
        self.held = np.zeros(0, dtype=np.float32)
        self.parts: list[np.ndarray] = []

    def __call__(self, chunk: np.ndarray, last: bool) -> None:
        data = np.concatenate((self.held, output_pcm(chunk, float(self.gain)))).astype(np.float32)
        if last:
            data, self.held = fade_out(data, self.fade_out_ms), data[:0]
        elif self.hold:
            data, self.held = data[: -self.hold], data[-self.hold :]
        else:
            self.held = data[:0]
        if len(data):
            self.parts.append(data)
            self.on_audio(data)

    def pcm(self) -> np.ndarray:
        return np.concatenate(self.parts) if self.parts else np.zeros(0, dtype=np.float32)


def output_gain(codec: str, headroom_db: float | None, decoder: str | None = None) -> float:
    """The output gain of ``headroom_db``, None = the headroom of the codec's decoder ``decoder`` (None = the default;
    0 dB without a matched codec)."""
    if headroom_db is None:
        from zonos3.tokenizer.matched_codec import codec_names, decoder_headroom_db

        headroom_db = decoder_headroom_db(codec, decoder) if codec in codec_names() else 0.0
    return float(10.0 ** (-float(headroom_db) / 20.0))


def fade_out(samples: np.ndarray, fade_out_ms: float) -> np.ndarray:
    if fade_out_ms <= 0 or not len(samples):
        return samples
    n = min(len(samples), int(SAMPLE_RATE * fade_out_ms / 1000.0))
    if n <= 0:
        return samples
    samples = samples.copy()
    samples[-n:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32)))
    return samples


# ----------------------------------------------------------------------------------------------------- synthesize


@dataclass
class SynthResult:
    pcm: np.ndarray  # float32 mono 44100 Hz
    seed: int
    laci: dict | None  # LACI record (LaciSession.record) of a laci request
    rows: list = field(default_factory=list)  # generated rows (int32 [n_codebooks + 1])
    eos_frame: int | None = None
    plan: P.PromptPlan | None = None


def steer_fn(plan: P.PromptPlan, follower: SpanFollower | None, base_delta=None):
    """Hidden delta of generated frame f: the request's constant delta, its released controls at f's weight (steer
    then hold) and the span deltas active at f (the follower's or the character-rate schedule's) --
    TTSReq.steer_delta; None outside hidden mode."""
    if not plan.sources:
        return None
    delta0 = plan.hidden_delta if base_delta is None else base_delta

    def steer(frame: int):
        base = released_delta(delta0, plan.release_delta, plan.release, frame)
        if follower is not None:
            span = follower.delta(frame)
        elif plan.schedule is not None:
            span = schedule_delta(plan.schedule, frame)
        else:
            span = None
        delta = base if span is None else (span if base is None else base + span)
        if delta is None:
            return None
        per_source = delta.reshape(len(plan.sources), -1)
        return {source: per_source[j] for j, source in enumerate(plan.sources)}

    return steer


@torch.inference_mode()
def synthesize(
    lm: Zonos3LM,
    request: dict,
    *,
    assets: ReferenceAssets,
    device=None,
    dtype: torch.dtype | None = None,
    on_plan=None,
    on_audio=None,
    stream: StreamSettings = StreamSettings(),
) -> SynthResult:
    """Generate and decode one v2 GenerateRequest. ``on_plan(plan)`` sees the prompt plan before generation starts;
    ``on_audio(pcm)`` gets the output (gained, faded) as it is decoded: prefix-streamed chunks (``stream``; under LACI
    the committed rows of a progressive request), or the whole take at the end (``stream_decode: whole``, and a LACI
    take decoded whole)."""
    device = torch.device(device if device is not None else lm.device)
    if lm.device != device:
        lm.to(device)
    if dtype is not None and lm.dtype != dtype:
        lm.cast(dtype)
    plan = plan_prompt(lm, request, assets=assets, device=device)
    if on_plan is not None:
        on_plan(plan)
    c = lm.config
    req = plan.request
    sampling = Sampling.from_request(req)
    accents = None
    if plan.accent is not None:
        accents = AccentState.for_model(lm, assets.accent_adapters[plan.accent[0]], plan.accent[1])
    follower = None
    if plan.follow_tokens is not None and plan.schedule is not None:
        lead, lag = plan.follow_gate
        follower = SpanFollower(
            plan.follow_tokens, plan.schedule[:, 3:], FollowConfig(lead=float(lead), lag=float(lag))
        )
    text_start, text_len = locate_text_span(plan.cond.input_ids, c.n_codebooks)
    gain = output_gain(c.audio_codec, req.headroom_db, req.decoder)
    prefix = emit = on_row = None
    streamed = _progressive(req, c.audio_codec) if plan.laci is not None else req.stream_decode != "whole"
    if on_audio is not None and streamed:
        from zonos3.codec.istft_head_a_20261002.stream import ISTFTStream

        prefix = StreamDecoder(
            _codec_decode(lm, assets, device, req.decoder),
            stream=ISTFTStream(assets.decoder(c.audio_codec, torch.device(device), req.decoder)),
            n_codebooks=c.n_codebooks,
            audio_pad_id=c.audio_pad_id,
            hop=_codec_hop(lm),
            prefill_rows=plan.prefill_rows,
            context_codes=plan.context_codes,
            mode="prefix",
            settings=stream,
            codebooks=int(req.stream_codebooks),
            first_only=req.stream_codebooks_scope == "first",
        )
        emit = _Emit(on_audio, gain, float(req.fade_out_ms))

        def on_row(row, eos_frame, finished):
            emit(prefix.push(row, eos_frame, finished), finished)

    record = None
    if plan.laci is None:
        probe = None
        if follower is not None:
            probe = ProbeState(assets.laci_heads[plan.language].heads, text_start, text_len)
        attempt = generate_rows(
            lm,
            plan.cond,
            sampling=sampling,
            seed=plan.seed,
            max_tokens=plan.max_tokens,
            stop_id=c.eoa_id,
            twin=plan.twin,
            cfg_scale=plan.cfg_scale,
            accents=accents,
            steer=steer_fn(plan, follower),
            probe=probe,
            observer=follower,
            on_row=on_row,
        )
        rows, eos_frame = attempt.rows, attempt.eos_frame
    else:
        rows, eos_frame, record = _laci(
            lm, plan, assets, sampling, accents, follower, text_start, text_len, emit_row=on_row
        )
    if emit is not None:
        return SynthResult(emit.pcm(), plan.seed, record, rows, eos_frame, plan)
    pcm = decode_rows(
        lm,
        assets,
        rows,
        prefill_rows=plan.prefill_rows,
        eos_frame=eos_frame,
        context_codes=plan.context_codes,
        device=device,
        decoder=req.decoder,
    )
    pcm = fade_out(output_pcm(pcm, gain), float(req.fade_out_ms)).astype(np.float32)
    if on_audio is not None:
        on_audio(pcm)
    return SynthResult(pcm, plan.seed, record, rows, eos_frame, plan)


def _progressive(req: P.Request, audio_codec: str) -> bool:
    """Whether the request's resolved decode is the progressive (prefix) one (zonos3.server.speech)."""
    try:
        mode = resolve_stream_decode(
            req.stream_decode, streaming=bool(req.stream), prefix_supported=prefix_stream_supported(audio_codec)
        )
    except ValueError:  # the servers refuse prefix on a codec without it; the reference decodes whole
        return False
    return mode == "prefix"


def _laci(lm, plan: P.PromptPlan, assets, sampling, accents, follower, text_start, text_len, emit_row=None):
    """LACI (zonos3.tts.laci): attempts until the session's result is final; a rollback re-prefills the prompt plus the
    kept rows (TTSScheduler._laci_rows) under seed + 1000 * attempt. ``emit_row(row, eos_frame, finished)`` sees every
    row the session hands downstream, as it does (the final EOS on the last one)."""
    c = lm.config
    names = list(c.condition_names)
    edges = c.condition_edges.get("speaking_rate")
    rates = (
        RateBuckets(names.index("speaking_rate"), tuple(edges), c.condition_bucket_counts[names.index("speaking_rate")])
        if c.condition_count and "speaking_rate" in names and edges
        else None
    )
    width = c.n_codebooks + 1
    prefill = plan.prefill_rows if plan.prefill_rows is not None else torch.zeros((0, width), dtype=torch.int32)
    session = LaciSession(
        heads=assets.laci_heads[plan.laci],
        cfg=LaciConfig(),
        prefill_rows=prefill,
        seed=plan.seed,
        max_tokens=plan.max_tokens,
        n_codebooks=c.n_codebooks,
        hold=not _progressive(plan.request, c.audio_codec),
        rates=rates,
        condition_values=None if plan.cond.condition_values is None else list(plan.cond.condition_values),
        follower=follower,
        short_text=plan.short_text,
    )
    template = plan.cond.input_ids
    prompt = template[: len(template) - session.n_prefill]
    branch, seed, max_tokens, history_rows, delta = plan.cond, plan.seed, plan.max_tokens, 0, plan.hidden_delta
    while True:
        probe = ProbeState(session.heads.heads, text_start, text_len, mask=session.probe_mask())
        state = {}

        def on_row(row, eos, finished):
            session.append(row, eos)
            final = False
            if finished:
                state["continuation"] = session.finish_attempt()
                final = state["continuation"] is None
                out = session.drain(final=final)
            else:
                out = session.drain()
            if emit_row is not None:
                eos_frame = session.eos_frame() if final else None
                for k, r in enumerate(out):
                    last = final and k == len(out) - 1
                    emit_row(r, eos_frame if last else None, last)

        generate_rows(
            lm,
            branch,
            sampling=sampling,
            seed=seed,
            max_tokens=max_tokens,
            stop_id=c.eoa_id,
            history_rows=history_rows,
            accents=accents,
            steer=steer_fn(plan, follower, delta),
            probe=probe,
            observer=session,
            on_row=on_row,
        )
        cont = state["continuation"]
        if cont is None:
            break
        branch = replace(
            branch,
            input_ids=torch.cat((prompt, cont.rows.to(prompt.dtype)), dim=0),
            condition_values=cont.condition_values if cont.condition_values is not None else branch.condition_values,
        )
        seed, max_tokens, history_rows = cont.seed, cont.max_tokens, cont.history_rows
        if cont.swap_delta:
            delta = cont.steering_delta
    rows = session.stream[session.n_prefill :]
    return rows, session.eos_frame(), session.record()
