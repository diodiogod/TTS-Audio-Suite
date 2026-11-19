"""Plain-PyTorch ZONOS3 Mini LM (GDN/global backbone with Full AttnRes): the numerical oracle.

Parameter names are the release checkpoint's flat state dict (``model.pth``; ``tools/convert_gguf.py lm`` maps them
through ``LM_NAMES``), so ``Zonos3LM.load`` is a strict ``load_state_dict``. The forward is the golden CPU oracle
(formerly ``tools/dump-golden.py`` ``Reference``; ``tools/dump_golden.py lm`` runs this module) with the serving-time
inputs of the sgl server's ``Zonos3ForCausalLM._forward_model``: speaker and condition rows replace their summed
embeddings before the weightless embedding norm, and hidden steering deltas add to Full AttnRes bank sources (0 =
embedding, 2i+1 = attention/GDN output of layer i, 2i+2 = FFN/MoE output of layer i). No custom kernels: the GDN
recurrence runs token by token in float32, MoE experts run per (token, expert) pair.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from zonos3.condition_buckets import bucket_counts
from zonos3.tts.probe import Head, clip_window, probe_weight, readout

GDN_HEADS, GDN_HEAD_DIM, GDN_KERNEL = 4, 128, 4
_GDN_PATTERN = re.compile(r"gdn(\d+)_global1")

# Architecture keys of the release LM (the converter's validate_config): params.json must carry them
# with exactly these values.
_REQUIRED = {
    "residual_type": "full_attnres",
    "use_xsa": False,
    "speaker_enabled": True,
    "condition_tokens_enabled": True,
    "moe_router_type": "linear",
    "moe_router_use_eda": False,
    "moe_balancing_strategy": "quantile",
    "moe_router_topk": 2,
    "moe_start_from_layer": 1,
    "moe_end_from_layer": 0,
}
# Keys the converter checks in config.yaml; params.json may carry them, and then with these values. The legacy prompt
# tokens (speaking-rate / quality / background / accurate-mode rows) and the speaker LDA are not part of this
# architecture.
_IF_PRESENT = {
    "embedding_norm": True,
    "qknorm": True,
    "qwen_gating_style": "headwise",
    "flash_msa_every_n_layers": 0,
    "sliding_window": -1,
    "ffn_out_norm": False,
    "attn_out_norm": False,
    "moe_routed_outnorm": False,
    "use_eda": None,
    "rope_head_dim": None,
    "speaker_lda_dim": None,
    "speaking_rate_num_buckets": 0,
    "quality_num_buckets": 0,
    "speaker_background_token_enabled": False,
    "accurate_mode_token_enabled": False,
}


def _unsupported(key, value):
    raise ValueError(f"unsupported checkpoint: {key}={value!r} (the reference implements the ZONOS3 Mini architecture)")


def _round_up(value: float, multiple: int) -> int:
    return multiple * ((int(value) + multiple - 1) // multiple)


@dataclass(frozen=True)
class Zonos3LMConfig:
    n_layers: int
    dim: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    rope_theta: float
    max_seqlen: int
    loss_softcap: float
    gdn_period: int  # every gdn_period-th layer (layer + 1 divisible) is a global attention layer
    ffn_hidden: int  # dense FFN (layer 0) width
    moe_hidden: int  # expert width
    moe_routed_dim: int  # latent MoE width (dim * moe_alpha)
    n_experts: int
    moe_topk: int
    n_codebooks: int
    codebook_size: int
    eoa_id: int
    audio_pad_id: int
    span_end_id: int | None
    text_vocab: int
    text_extra_tokens: int
    text_tokenizer_spec: dict | None
    speaker_embedding_dim: int
    speaker_embedding_codec: str | None
    speaker_reference_prefix_enabled: bool
    speaker_reference_max_seconds: float
    condition_count: int
    condition_bucket_counts: tuple[int, ...]  # per condition table: ids 0..n-1, n = unknown
    condition_names: tuple[str, ...]
    condition_edges: dict[str, tuple[float, ...]]
    condition_units: dict[str, str]
    audio_codec: str

    @property
    def audio_vocab(self) -> int:
        """Per-codebook vocabulary: codes, EOA, pad and the optional span-end."""
        return self.codebook_size + 2 + (0 if self.span_end_id is None else 1)

    @property
    def text_rows(self) -> int:
        """Text embedding rows: the vocabulary, the padding row and the extra rows (edit markers, phoneme / start /
        pause tokens)."""
        return self.text_vocab + 1 + self.text_extra_tokens

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return (self.codebook_size,) * self.n_codebooks

    @property
    def n_sources(self) -> int:
        """Full AttnRes bank sources: the embedding plus two per layer."""
        return 2 * self.n_layers + 1

    @property
    def gdn_inner(self) -> int:
        return GDN_HEADS * GDN_HEAD_DIM

    def uses_gdn(self, layer: int) -> bool:
        return (layer + 1) % self.gdn_period != 0

    @property
    def attention_layers(self) -> tuple[int, ...]:
        return tuple(layer for layer in range(self.n_layers) if not self.uses_gdn(layer))

    @classmethod
    def from_params(cls, params: dict) -> "Zonos3LMConfig":
        """Validate a ``params.json`` (the whole document or its ``model`` table) and derive the module sizes."""
        p = params.get("model", params)
        for key, wanted in _REQUIRED.items():
            if p.get(key) != wanted:
                _unsupported(key, p.get(key))
        for key, wanted in _IF_PRESENT.items():
            # absent-equivalent None is accepted where the expected value is falsy
            if key in p and p[key] != wanted and not (p[key] is None and not wanted):
                _unsupported(key, p[key])
        pattern = _GDN_PATTERN.fullmatch(str(p.get("attention_pattern")))
        if pattern is None or int(pattern.group(1)) < 1:
            _unsupported("attention_pattern", p.get("attention_pattern"))
        n_layers, dim, head_dim = int(p["n_layers"]), int(p["dim"]), int(p["head_dim"])
        n_heads, n_kv_heads = int(p["n_heads"]), int(p["n_kv_heads"])
        if n_layers < 1 or dim < 1 or head_dim % 2 or n_heads % n_kv_heads:
            _unsupported("n_layers/dim/head_dim/n_heads/n_kv_heads", (n_layers, dim, head_dim, n_heads, n_kv_heads))
        n_codebooks, codebook_size = int(p["n_codebooks"]), int(p["codebook_size"])
        sizes = p.get("codebook_sizes") or [codebook_size] * n_codebooks
        if list(sizes) != [codebook_size] * n_codebooks:
            _unsupported("codebook_sizes", sizes)
        if int(p["eoa_id"]) != codebook_size or int(p["audio_pad_id"]) != codebook_size + 1:
            _unsupported("eoa_id/audio_pad_id", (p["eoa_id"], p["audio_pad_id"]))
        span_end_id = p.get("span_end_id")
        if span_end_id is not None and int(span_end_id) != codebook_size + 2:
            _unsupported("span_end_id", span_end_id)
        text_vocab = int(p["text_vocab"])
        extra = int(p.get("text_extra_tokens", 0) or 0)
        if extra < 0:
            _unsupported("text_extra_tokens", extra)
        spec = p.get("text_tokenizer_spec")
        if spec is not None:
            if (spec.get("kind"), spec.get("offset"), spec.get("bos"), spec.get("eos")) != ("superbpe", 4, 2, 3) or int(
                spec.get("vocab_size", -1)
            ) + 4 != text_vocab:
                _unsupported(
                    "text_tokenizer_spec", {k: spec.get(k) for k in ("kind", "offset", "bos", "eos", "vocab_size")}
                )
            phonemes = spec.get("phonemes")
            if phonemes is not None and (
                int(phonemes["open_id"]) < text_vocab + 1 or int(phonemes["close_id"]) != int(phonemes["open_id"]) + 1
            ):
                _unsupported("text_tokenizer_spec.phonemes ids", (phonemes["open_id"], phonemes["close_id"]))
            from zonos3.tts.superbpe import extra_token_count, rows_before_extra_tokens

            rows = extra_token_count(spec)
            if rows and rows_before_extra_tokens(spec, text_vocab) + rows != extra:
                _unsupported("text_tokenizer_spec tokens vs text_extra_tokens", extra)
        alpha = float(p["moe_alpha"])
        if not 0.0 < alpha <= 1.0:
            _unsupported("moe_alpha", alpha)
        n_experts = int(p["moe_n_experts"])
        if n_experts < 2:
            _unsupported("moe_n_experts", n_experts)
        multiple_of = int(p.get("multiple_of", 256))
        moe_multiple_of = int(p.get("moe_multiple_of") or multiple_of)
        ffn_mult = float(p["ffn_dim_multiplier"])
        moe_mult = float(p.get("moe_ffn_dim_multiplier") or ffn_mult)
        names = tuple(str(n) for n in p.get("condition_names") or ())
        count = int(p["condition_count"])
        if len(names) != count:
            _unsupported("condition_names", names)
        try:
            counts = bucket_counts(count, int(p["condition_n_buckets"]), p.get("condition_bucket_counts"))
        except ValueError:
            _unsupported("condition_bucket_counts", p.get("condition_bucket_counts"))
        edges = {str(k): tuple(float(v) for v in vs) for k, vs in (p.get("condition_edges") or {}).items()}
        for name, values in edges.items():
            if (
                name not in names
                or len(values) != counts[names.index(name)] - 1
                or not all(math.isfinite(v) for v in values)
                or any(b <= a for a, b in zip(values, values[1:]))
            ):
                _unsupported(f"condition_edges.{name}", values)
        return cls(
            n_layers=n_layers,
            dim=dim,
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            norm_eps=float(p["norm_eps"]),
            rope_theta=float(p["rope_theta"]),
            max_seqlen=int(p["max_seqlen"]),
            loss_softcap=float(p["loss_softcap"]),
            gdn_period=int(pattern.group(1)) + 1,
            ffn_hidden=_round_up(ffn_mult * dim, multiple_of),
            moe_hidden=_round_up(moe_mult * dim, moe_multiple_of),
            moe_routed_dim=max(1, int(round(dim * alpha))),
            n_experts=n_experts,
            moe_topk=int(p["moe_router_topk"]),
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            eoa_id=int(p["eoa_id"]),
            audio_pad_id=int(p["audio_pad_id"]),
            span_end_id=None if span_end_id is None else int(span_end_id),
            text_vocab=text_vocab,
            text_extra_tokens=extra,
            text_tokenizer_spec=spec,
            speaker_embedding_dim=int(p["speaker_embedding_dim"]),
            speaker_embedding_codec=p.get("speaker_embedding_codec"),
            speaker_reference_prefix_enabled=bool(p.get("speaker_reference_prefix_enabled", False)),
            speaker_reference_max_seconds=float(p.get("speaker_reference_max_seconds", 30.0)),
            condition_count=count,
            condition_bucket_counts=counts,
            condition_names=names,
            condition_edges=edges,
            condition_units={str(k): str(v) for k, v in (p.get("condition_units") or {}).items()},
            audio_codec=str(p.get("audio_codec", "")),
        )


# ----------------------------------------------------------------------------------------------------- module tree


def _param(*shape) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape), requires_grad=False)


class _Weight(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.weight = _param(*shape)


def _linear(n_in: int, n_out: int, bias: bool = False) -> nn.Linear:
    return nn.Linear(n_in, n_out, bias=bias).requires_grad_(False)


class _GDN(nn.Module):
    def __init__(self, c: Zonos3LMConfig):
        super().__init__()
        inner = c.gdn_inner
        self.in_proj = _linear(c.dim, 4 * inner + 2 * GDN_HEADS)
        self.conv = _Weight(3 * inner, 1, GDN_KERNEL)
        self.out_weight = _param(GDN_HEAD_DIM)
        self.A_log = _param(GDN_HEADS)
        self.dt_bias = _param(GDN_HEADS)
        self.wo = _linear(inner, c.dim)


class _Attention(nn.Module):
    def __init__(self, c: Zonos3LMConfig):
        super().__init__()
        self.wq = _linear(c.dim, c.n_heads * c.head_dim)
        self.wkv = _Weight(2, c.n_kv_heads * c.head_dim, c.dim)
        self.wo = _linear(c.n_heads * c.head_dim, c.dim)
        self.temp = _param(1, c.n_heads, 1)
        self.gater = _linear(c.dim, c.n_heads)


class _Dense(nn.Module):
    def __init__(self, c: Zonos3LMConfig):
        super().__init__()
        self.w_in = _Weight(2, c.ffn_hidden, c.dim)
        self.w_out = _linear(c.ffn_hidden, c.dim)


class _Experts(nn.Module):
    def __init__(self, c: Zonos3LMConfig):
        super().__init__()
        # gate_up_proj rows [0, moe_hidden) are the gate, [moe_hidden, 2 moe_hidden) the up projection.
        self.gate_up_proj = _param(c.n_experts, 2 * c.moe_hidden, c.moe_routed_dim)
        self.down_proj = _param(c.n_experts, c.moe_routed_dim, c.moe_hidden)


class _Router(nn.Module):
    def __init__(self, c: Zonos3LMConfig):
        super().__init__()
        self.router_linear = _linear(c.dim, c.n_experts)
        self.rmsnorm_eda = _Weight(c.dim)
        self.balancing_biases = _param(c.n_experts)


class _MoE(nn.Module):
    def __init__(self, c: Zonos3LMConfig):
        super().__init__()
        self.experts = _Experts(c)
        self.latent_down = _linear(c.dim, c.moe_routed_dim)
        self.latent_up = _linear(c.moe_routed_dim, c.dim)
        self.router = _Router(c)


class _Layer(nn.Module):
    def __init__(self, c: Zonos3LMConfig, layer: int):
        super().__init__()
        self.attention_norm = _Weight(c.dim)
        self.ffn_norm = _Weight(c.dim)
        self.attention = _GDN(c) if c.uses_gdn(layer) else _Attention(c)
        self.feed_forward = _Dense(c) if layer == 0 else _MoE(c)


class _MultiEmbedder(nn.Module):
    def __init__(self, c: Zonos3LMConfig):
        super().__init__()
        rows = [c.audio_vocab] * c.n_codebooks + [c.text_rows]
        self.embedders = nn.ModuleList(nn.Embedding(n, c.dim).requires_grad_(False) for n in rows)


# ----------------------------------------------------------------------------------------------------- state


@dataclass
class LMCache:
    """Decoding state: the absolute position of the next row, the global layers' KV ([rows, kv heads, head dim]) and
    each GDN layer's recurrent state ([heads, value, key], float32) and causal-conv history ([channels, kernel - 1])."""

    position: int = 0
    kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    gdn_state: dict[int, torch.Tensor] = field(default_factory=dict)
    conv_state: dict[int, torch.Tensor] = field(default_factory=dict)

    def fork(self) -> "LMCache":
        return LMCache(
            self.position,
            {k: (a.clone(), b.clone()) for k, (a, b) in self.kv.items()},
            {k: v.clone() for k, v in self.gdn_state.items()},
            {k: v.clone() for k, v in self.conv_state.items()},
        )


@dataclass(frozen=True)
class ProbeSpec:
    """Alignment probe of one forward (zonos3.tts.probe): the probed heads, the BOS..EOS text rows of the prompt
    (absolute start, length) and the ACI windows of the masked heads (None = unmasked). Masked heads have their
    attention output recomputed with the text keys outside their window at -inf (non-text keys are never masked)."""

    heads: tuple[Head, ...]
    text_start: int
    text_len: int
    windows: dict[Head, tuple[int, int]] | None = None


class LMOutput(NamedTuple):
    logits: torch.Tensor  # [rows, n_codebooks, audio_vocab], softcapped
    probe: list[tuple[int, float]] | None  # (argmax, centroid) per row when a probe was given


# ----------------------------------------------------------------------------------------------------- math


def full_attnres_route(sources, query, eps):
    bank = torch.stack(sources)
    keys = F.rms_norm(bank, (bank.shape[-1],), eps=eps)
    scores = (keys * query).sum(-1, dtype=torch.float32)
    weights = scores.softmax(0).to(bank.dtype)
    return (bank * weights.unsqueeze(-1)).sum(0)


def rope_interleaved(x, positions, theta):
    hd = x.shape[-1]
    frequencies = theta ** (-torch.arange(0, hd, 2, dtype=torch.float32, device=x.device) / hd)
    angles = positions.float()[:, None] * frequencies[None, :]
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
    pairs = x.float().reshape(*x.shape[:-1], hd // 2, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2).to(x.dtype)


class Zonos3LM(nn.Module):
    def __init__(self, config: Zonos3LMConfig):
        super().__init__()
        c = self.config = config
        self.multi_embedder = _MultiEmbedder(c)
        self.layers = nn.ModuleList(_Layer(c, layer) for layer in range(c.n_layers))
        self.out_norm = _Weight(c.dim)
        self.multi_output = _linear(c.dim, c.n_codebooks * c.audio_vocab)
        self.block_attnres_queries = _param(c.n_sources, c.dim)
        self.speaker_projection = _linear(c.speaker_embedding_dim, c.dim, bias=True)
        self.condition_embeddings = nn.ModuleList(
            nn.Embedding(n + 1, c.dim).requires_grad_(False) for n in c.condition_bucket_counts
        )
        self.root: Path | None = None  # checkpoint directory (set by load)
        self._capture: dict | None = None
        self._accents = None

    # -- loading ----------------------------------------------------------------------------------------------------

    @classmethod
    def load(cls, path_or_hf, *, device="cpu", dtype: torch.dtype = torch.float32) -> "Zonos3LM":
        """Load ``params.json`` + ``model.pth`` of a checkpoint directory or ``hf://`` folder (strict)."""
        from zonos3.utils.release import resolve_dir_or_local

        root = Path(resolve_dir_or_local(str(path_or_hf)))
        config = Zonos3LMConfig.from_params(json.loads((root / "params.json").read_text()))
        with torch.device("meta"):
            model = cls(config)
        model.to_empty(device=device)
        state = torch.load(root / "model.pth", map_location="cpu", mmap=True, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.root = root
        return model.cast(dtype)

    def grow(self, grown) -> "Zonos3LM":
        """This model with a grown vocabulary (``zonos3.models.accent_format.GrownVocabulary`` of a ``/3`` adapter: a
        span-end audio id, extra text rows and the rows they add). The existing rows keep their values, so a request
        that never reads or samples the new ids runs exactly as before (the sampler masks every special id but the
        request's stop id). Growing again to the same vocabulary is a no-op."""
        c = self.config
        if (c.span_end_id, c.text_extra_tokens) == (grown.span_end_id, grown.text_extra_tokens):
            return self
        if c.span_end_id is not None or c.text_extra_tokens:
            raise ValueError(
                f"grow: the model already has span_end_id {c.span_end_id} and {c.text_extra_tokens} extra "
                f"text rows; the adapter wants {grown.span_end_id} and {grown.text_extra_tokens}"
            )
        new = replace(c, span_end_id=int(grown.span_end_id), text_extra_tokens=int(grown.text_extra_tokens))
        if new.audio_vocab != c.audio_vocab + 1 or new.span_end_id != c.codebook_size + 2:
            raise ValueError("grow: the span-end id must be the one row after the audio pad id")
        embedders = self.multi_embedder.embedders
        for i in range(c.n_codebooks + 1):
            key = f"multi_embedder.embedders.{i}.weight"
            rows, width = (new.audio_vocab, c.audio_vocab) if i < c.n_codebooks else (new.text_rows, c.text_rows)
            extra = grown.rows[key].to(embedders[i].weight)
            if tuple(extra.shape) != (rows - width, c.dim):
                raise ValueError(f"grow: {key} new rows {tuple(extra.shape)} != {(rows - width, c.dim)}")
            embedders[i].weight = nn.Parameter(torch.cat((embedders[i].weight, extra)), requires_grad=False)
        head = self.multi_output.weight
        extra = grown.rows["multi_output.weight"].to(head)
        if tuple(extra.shape) != (c.n_codebooks, c.dim):
            raise ValueError(f"grow: multi_output.weight new rows {tuple(extra.shape)} != {(c.n_codebooks, c.dim)}")
        blocks = head.view(c.n_codebooks, c.audio_vocab, c.dim)
        self.multi_output.weight = nn.Parameter(
            torch.cat((blocks, extra[:, None]), 1).reshape(-1, c.dim), requires_grad=False
        )
        self.config = new
        return self

    def append_text_rows(self, extra) -> "Zonos3LM":
        """This model with ``extra`` (a ``zonos3.models.text_extra.TextExtraRows``, e.g. phoneme, start and pause
        tokens) after its text table's rows, the extra rows it already has included; ``config.text_tokenizer_spec``
        gains their ``phonemes`` / ``start`` / ``pauses``. Text that never reads them runs exactly as before."""
        c = self.config
        spec = extra.spec(c.text_tokenizer_spec, text_vocab=c.text_vocab, existing=c.text_extra_tokens)
        table = self.multi_embedder.embedders[c.n_codebooks]
        table.weight = nn.Parameter(extra.append(table.weight), requires_grad=False)
        self.config = replace(c, text_extra_tokens=c.text_extra_tokens + extra.count, text_tokenizer_spec=spec)
        return self

    def cast(self, dtype: torch.dtype) -> "Zonos3LM":
        """Weights in ``dtype``; the routers' balancing biases stay float32 (as in the checkpoint)."""
        for name, param in self.named_parameters():
            param.data = param.data.to(torch.float32 if name.endswith("balancing_biases") else dtype)
        return self

    @property
    def dtype(self) -> torch.dtype:
        return self.multi_output.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.multi_output.weight.device

    # -- helpers -----------------------------------------------------------------------------------------------------

    def _save(self, name, tensor):
        if self._capture is not None:
            self._capture[name] = tensor.detach().float().cpu().numpy().copy()

    def _rms(self, x, weight=None, eps=None):
        return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=self.config.norm_eps if eps is None else eps)

    def _lin(self, module, x, weight, bias=None):
        y = F.linear(x, weight, bias)
        return y if self._accents is None else self._accents.linear(module, x, y)

    # -- sublayers ---------------------------------------------------------------------------------------------------

    def _global_attention(self, x, layer, cache: LMCache, probe: ProbeSpec | None, queries: dict):
        c, a, p = self.config, self.layers[layer].attention, f"layers.{layer}.attention."
        n, nh, nkv, hd = len(x), c.n_heads, c.n_kv_heads, c.head_dim
        gate = self._lin(p + "gater", x, a.gater.weight).sigmoid()
        q = self._lin(p + "wq", x, a.wq.weight).view(n, nh, hd)
        kv = self._lin(p + "wkv", x, a.wkv.weight.reshape(2 * nkv * hd, c.dim))
        k, v = kv[:, : nkv * hd].reshape(n, nkv, hd), kv[:, nkv * hd :].reshape(n, nkv, hd)
        q = self._rms(q, eps=1e-6) * a.temp.abs().view(1, nh, 1)
        k = self._rms(k, eps=1e-6)
        positions = torch.arange(cache.position, cache.position + n, device=x.device)
        q = rope_interleaved(q, positions, c.rope_theta)
        k = rope_interleaved(k, positions, c.rope_theta)
        old_k, old_v = cache.kv.get(layer, (k[:0], v[:0]))
        k, v = torch.cat((old_k, k), 0), torch.cat((old_v, v), 0)
        cache.kv[layer] = (k, v)
        rep = nh // nkv
        key = k.to(q.dtype).repeat_interleave(rep, 1).transpose(0, 1)
        value = v.to(q.dtype).repeat_interleave(rep, 1).transpose(0, 1)
        mask = torch.arange(len(k), device=x.device)[None, :] <= positions[:, None]
        out = F.scaled_dot_product_attention(q.transpose(0, 1), key, value, attn_mask=mask, scale=hd**-0.5).transpose(
            0, 1
        )
        if probe is not None:
            queries[layer] = q
            if probe.windows:
                out = self._masked_heads(layer, q, k, v, out, mask, probe)
        return self._lin(p + "wo", (out * gate.unsqueeze(-1)).reshape(n, nh * hd), a.wo.weight)

    def _masked_heads(self, layer, q, k, v, out, causal, probe: ProbeSpec):
        """ACI hard mask: the masked heads' output over all keys with the text keys outside their window at -inf
        (sgl align_probe._masked)."""
        rep = self.config.n_heads // self.config.n_kv_heads
        tidx = torch.arange(len(k), device=q.device) - probe.text_start
        in_text = (tidx >= 0) & (tidx < probe.text_len)
        out = out.clone()
        for (head_layer, head), window in probe.windows.items():
            if head_layer != layer:
                continue
            lo, hi = clip_window(window, probe.text_len)
            outside = in_text & ((tidx < lo) | (tidx > hi))
            logits = (q[:, head].float() @ k[:, head // rep].float().T) * self.config.head_dim**-0.5
            weights = torch.softmax(logits.masked_fill(outside[None, :] | ~causal, float("-inf")), dim=-1)
            out[:, head] = (weights @ v[:, head // rep].float()).to(out.dtype)
        return out

    def _gdn(self, x, layer, cache: LMCache):
        a, p = self.layers[layer].attention, f"layers.{layer}.attention."
        heads, hd, kernel = GDN_HEADS, GDN_HEAD_DIM, GDN_KERNEL
        inner = heads * hd
        channels = 3 * inner
        projection = self._lin(p + "in_proj", x, a.in_proj.weight)
        qkv, z, aa, b = projection.split((3 * inner, inner, heads, heads), -1)
        history = cache.conv_state.get(layer, torch.zeros(channels, kernel - 1, dtype=x.dtype, device=x.device))
        sequence = torch.cat((history, qkv.T), -1)
        # Depthwise causal cross-correlation: weight[0] is oldest, weight[-1] current.
        conv = F.conv1d(sequence.unsqueeze(0), a.conv.weight, groups=channels)
        conv = F.silu(conv.squeeze(0).T)
        cache.conv_state[layer] = sequence[:, -(kernel - 1) :].clone()
        q, k, v = [part.reshape(len(x), heads, hd).float() for part in conv.split(inner, -1)]
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        decay = torch.exp(-a.A_log.float().exp() * F.softplus(aa.float() + a.dt_bias.float()))
        beta = b.float().sigmoid()
        state = cache.gdn_state.get(layer, torch.zeros(heads, hd, hd, dtype=torch.float32, device=x.device))
        outputs = []
        for token in range(len(x)):
            # FLA state_v_first=True: [heads, value, key], all accumulation in F32.
            state = state * decay[token, :, None, None]
            prediction = (state * k[token, :, None, :]).sum(-1)
            delta = beta[token, :, None] * (v[token] - prediction)
            state = state + delta[:, :, None] * k[token, :, None, :]
            outputs.append((state * (q[token] * hd**-0.5)[:, None, :]).sum(-1))
        cache.gdn_state[layer] = state
        recurrent = torch.stack(outputs).to(x.dtype)
        # FLA's rms_norm_gated has fixed eps=1e-5 and swish gate, per head.
        normalized = self._rms(recurrent, a.out_weight, eps=1e-5)
        gated = (normalized.float() * F.silu(z.reshape(len(x), heads, hd).float())).to(x.dtype).reshape(len(x), inner)
        if self._capture is not None:
            for name, tensor in (
                ("gdn_conv", conv),
                ("gdn_q", q),
                ("gdn_k", k),
                ("gdn_v", v),
                ("gdn_decay", decay),
                ("gdn_beta", beta),
                ("gdn_recurrent", recurrent),
                ("gdn_gated", gated),
                ("gdn_state", state),
                ("gdn_conv_state", cache.conv_state[layer]),
            ):
                self._save(f"{name}_{layer}", tensor)
        return self._lin(p + "wo", gated, a.wo.weight)

    def _feed_forward(self, x, layer):
        f, p = self.layers[layer].feed_forward, f"layers.{layer}.feed_forward."
        if layer == 0:
            hidden = self.config.ffn_hidden
            flat = self._lin(p + "w_in", x, f.w_in.weight.reshape(2 * hidden, -1))
            return self._lin(p + "w_out", flat[:, :hidden] * F.silu(flat[:, hidden:]), f.w_out.weight)
        router_input = self._rms(x, f.router.rmsnorm_eda.weight)
        logits = self._lin(p + "router.router_linear", router_input, f.router.router_linear.weight)
        self._save(f"router_logits_{layer}", logits)
        bias = f.router.balancing_biases.float()
        if self._accents is not None:
            bias = self._accents.router_bias(p + "router", bias)
        choices = torch.topk(logits.float() - bias, self.config.moe_topk, -1).indices
        weights = logits.float().gather(-1, choices).softmax(-1)
        latent = self._lin(p + "latent_down", x, f.latent_down.weight)
        gate_up_proj, w2 = f.experts.gate_up_proj, f.experts.down_proj
        half = gate_up_proj.shape[1] // 2
        experts = p + "experts"
        output = torch.zeros_like(latent)
        for token in range(len(x)):
            for rank in range(self.config.moe_topk):
                expert = int(choices[token, rank])
                gate_up = F.linear(latent[token], gate_up_proj[expert])
                gate, up = gate_up[:half], gate_up[half:]
                if self._accents is not None:
                    shift = self._accents.expert_shift(experts, "w13", expert, latent[token])
                    if shift is not None:  # SonicExperts layout: gate/up rows interleaved
                        gate, up = gate + shift[0::2], up + shift[1::2]
                hidden = F.silu(gate) * up
                expert_out = F.linear(hidden, w2[expert])
                if self._accents is not None:
                    shift = self._accents.expert_shift(experts, "w2", expert, hidden)
                    if shift is not None:
                        expert_out = expert_out + shift
                output[token] += (expert_out.float() * weights[token, rank]).to(x.dtype)
        return self._lin(p + "latent_up", output, f.latent_up.weight)

    def _replace_rows(self, x, lo, speaker_rows, condition_rows):
        """Speaker / condition rows (absolute positions) replace their summed embeddings (sgl index_copy)."""
        hi = lo + len(x)
        if not speaker_rows and not condition_rows:
            return x
        x = x.clone()
        for row, vec in (speaker_rows or {}).items():
            if lo <= row < hi:
                w = self.speaker_projection
                projected = F.linear(vec.to(x.device, w.weight.dtype).reshape(-1), w.weight, w.bias)
                x[row - lo] = projected.to(x.dtype)
        # condition table i serves the i-th condition row (rows ascending)
        for table, row in enumerate(sorted(condition_rows or {})):
            if lo <= row < hi:
                x[row - lo] = self.condition_embeddings[table].weight[int(condition_rows[row])]
        return x

    # -- forward -----------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        ids: torch.Tensor,
        cache: LMCache,
        *,
        speaker_rows: dict[int, torch.Tensor] | None = None,
        condition_rows: dict[int, int] | None = None,
        hidden_deltas: dict[int, torch.Tensor] | None = None,
        accents=None,
        probe: ProbeSpec | None = None,
        capture: dict | None = None,
    ) -> LMOutput:
        """Logits of ``ids`` [rows, n_codebooks + 1] continuing ``cache`` (updated in place).

        Rows are absolute positions in ``speaker_rows`` (speaker embedding replacing that row through the speaker
        projection) and ``condition_rows`` (row -> bucket id; the i-th row in ascending order reads condition table i).
        ``hidden_deltas`` maps a Full AttnRes bank source to a delta ([rows, dim] or [dim]) added to that source.
        ``accents`` is a zonos3.reference.accents.AccentState; ``probe`` adds the alignment readout of every row;
        ``capture`` receives the named intermediates ``tools/dump_golden.py lm`` writes (float32 numpy)."""
        c = self.config
        ids = ids.to(self.device, torch.long)
        lo = cache.position
        if lo + len(ids) > c.max_seqlen:
            raise ValueError(f"{lo + len(ids)} rows exceed the model context {c.max_seqlen}")
        self._capture, self._accents = capture, accents
        try:
            embedders = self.multi_embedder.embedders
            x = sum(F.embedding(ids[:, cb], embedders[cb].weight) for cb in range(c.n_codebooks + 1))
            self._save("emb_sum", x)
            x = self._replace_rows(x, lo, speaker_rows, condition_rows)
            self._save("emb_cond", x)
            x = self._rms(x)
            self._save("emb_norm", x)

            def source(index, tensor):
                delta = (hidden_deltas or {}).get(index)
                return tensor if delta is None else tensor + delta.to(tensor.device, tensor.dtype)

            sources = [source(0, x)]
            queries = self.block_attnres_queries
            probe_queries: dict[int, torch.Tensor] = {}
            for layer in range(c.n_layers):
                block = self.layers[layer]
                mixed = full_attnres_route(sources, queries[2 * layer], c.norm_eps)
                self._save(f"attnres_attn_{layer}", mixed)
                attn_in = self._rms(mixed, block.attention_norm.weight)
                self._save(f"attn_in_{layer}", attn_in)
                if c.uses_gdn(layer):
                    attention = self._gdn(attn_in, layer, cache)
                else:
                    attention = self._global_attention(attn_in, layer, cache, probe, probe_queries)
                self._save(f"attn_out_{layer}", attention)
                sources.append(source(2 * layer + 1, attention))
                mixed = full_attnres_route(sources, queries[2 * layer + 1], c.norm_eps)
                self._save(f"attnres_ffn_{layer}", mixed)
                ffn_in = self._rms(mixed, block.ffn_norm.weight)
                self._save(f"ffn_in_{layer}", ffn_in)
                x = self._feed_forward(ffn_in, layer)
                self._save(f"ffn_out_{layer}", x)
                self._save(f"layer_x_{layer}", x)
                sources.append(source(2 * layer + 2, x))
            mixed = full_attnres_route(sources, queries[-1], c.norm_eps)
            self._save("attnres_final", mixed)
            normalized = self._rms(mixed, self.out_norm.weight)
            self._save("out_norm", normalized)
            logits = self._lin("multi_output", normalized, self.multi_output.weight).view(
                len(ids), c.n_codebooks, c.audio_vocab
            )
            logits = c.loss_softcap * torch.tanh(logits / c.loss_softcap)
            self._save("logits", logits)
            cache.position = lo + len(ids)
            readouts = None if probe is None else self._readout(probe, probe_queries, cache)
            return LMOutput(logits, readouts)
        finally:
            self._capture = self._accents = None

    def _readout(self, probe: ProbeSpec, probe_queries: dict, cache: LMCache) -> list[tuple[int, float]]:
        layers = self.config.attention_layers
        weights = probe_weight(probe.heads, layers, self.config.n_heads)
        text = slice(probe.text_start, probe.text_start + probe.text_len)
        keys = torch.stack([cache.kv[layer][0][text] for layer in layers])
        rows = len(probe_queries[layers[0]])
        return [
            readout(
                torch.stack([probe_queries[layer][r] for layer in layers]),
                keys,
                layers,
                weights,
                probe.windows,
                self.config.head_dim**-0.5,
            )
            for r in range(rows)
        ]
