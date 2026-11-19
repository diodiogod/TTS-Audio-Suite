from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from zonos3.condition_buckets import bucket_counts
from zonos3.utils.moe_topk import normalize_special_topk_layers, resolve_layer_topk


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


def normalize_moe_balancing_strategy(strategy: str) -> str:
    normalized = strategy.strip().lower().replace("-", "_")
    aliases = {
        "current": "quantile",
        "quantile": "quantile",
        "qbalancing": "quantile",
        "megatron": "quantile",
        "megatron_qbalancing": "quantile",
        "megatron_quantile": "quantile",
        "quantile_balancing": "quantile",
        "old": "legacy",
        "legacy": "legacy",
        "aux": "legacy",
        "aux_loss": "legacy",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "Unsupported model.moe_balancing_strategy="
            f"{strategy!r}. Expected one of: quantile, qbalancing, legacy, aux_loss."
        ) from exc


def normalize_moe_router_type(router_type: str) -> str:
    normalized = router_type.strip().lower().replace("-", "_")
    aliases = {"mlp": "mlp", "linear": "linear", "single_linear": "linear"}
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported model.moe_router_type={router_type!r}. Expected 'mlp' or 'linear'.") from exc


def get_routed_moe_dim(dim: int, alpha: float) -> int:
    if not (0 < alpha <= 1.0):
        raise ValueError(f"moe_alpha must be in (0, 1], got {alpha}.")
    return max(1, int(round(dim * alpha)))


def compute_audio_vocab_size(codebook_size: int, span_end_id: int | None) -> int:
    """Per-codebook audio vocabulary: codes 0..codebook_size-1, EOA, pad, then the optional span-end."""
    return int(codebook_size) + 2 + (0 if span_end_id is None else 1)


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    special_topk_layers: dict[int, int] | None = None
    # Zonos3-specific fields
    n_codebooks: int = 9
    codebook_size: int = 1024
    codebook_sizes: tuple[int, ...] | None = None
    audio_codec: str = "opus_dual_3n_600k_sched600k_r4096_step600000"
    text_vocab: int | None = None
    # Text rows after the padding row: speech-edit marker rows, then phoneme / start / pause tokens (text_tokenizer_spec).
    text_extra_tokens: int = 0
    moe_n_experts: int = 1
    moe_start_from_layer: int = 0
    moe_end_from_layer: int = 0
    moe_router_dim: int = 256  # Router hidden dimension
    moe_router_type: str = "mlp"
    moe_router_use_eda: bool = True
    moe_alpha: float = 1.0
    moe_impl: str = "grouped"
    moe_balancing_strategy: str = "legacy"
    use_xsa: bool = False
    attention_pattern: str = "global"
    residual_type: str = "standard"
    block_attnres_block_size: int = 4
    # TTS-specific fields
    eoa_id: int = 1024  # End-of-audio token ID
    audio_pad_id: int = 1025  # Audio padding token ID
    span_end_id: int | None = None  # Edit span-end token ID (audio_pad_id + 1) or None
    loss_softcap: float = 15.0  # Logit soft capping value
    speaker_enabled: bool = False
    speaker_reference_prefix_enabled: bool = False
    speaker_reference_max_seconds: float = 30.0
    speaker_embedding_dim: int = 128
    speaker_embedding_codec: str | None = None
    speaker_lda_dim: int | None = None
    speaker_background_token_enabled: bool = False
    accurate_mode_token_enabled: bool = False
    speaking_rate_num_buckets: int = 0
    speaking_rate_buckets: tuple[str, ...] = ()
    quality_num_buckets: int = 0
    quality_features: tuple[str, ...] = ()
    quality_buckets: Dict[str, tuple[str, ...]] | None = None
    # Condition rows (loudness / fmax / speaking-rate buckets) of the cond3 finetunes; see utils/hf.py. One bucket
    # count per table (ids 0..n-1, n = unknown); () without condition rows.
    condition_tokens_enabled: bool = False
    condition_bucket_counts: tuple[int, ...] = ()
    condition_count: int = 3
    condition_names: tuple[str, ...] = ()
    condition_edges: Dict[str, tuple[float, ...]] | None = None
    condition_units: Dict[str, str] | None = None
    condition_flags: Dict[str, dict] | None = None
    quality_dropout: Dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.attention_pattern not in {"global", "gdn3_global1"}:
            raise ValueError(f"Unsupported attention pattern: {self.attention_pattern}")
        object.__setattr__(
            self,
            "special_topk_layers",
            normalize_special_topk_layers(self.special_topk_layers),
        )
        object.__setattr__(
            self,
            "moe_router_type",
            normalize_moe_router_type(self.moe_router_type),
        )
        object.__setattr__(
            self,
            "moe_balancing_strategy",
            normalize_moe_balancing_strategy(self.moe_balancing_strategy),
        )
        get_routed_moe_dim(self.hidden_size, float(self.moe_alpha))
        residual_type = str(self.residual_type).strip().lower().replace("-", "_")
        if residual_type not in {"standard", "block_attnres", "full_attnres"}:
            raise ValueError(
                f"residual_type must be 'standard', 'block_attnres' or 'full_attnres', got {residual_type!r}."
            )
        object.__setattr__(self, "residual_type", residual_type)
        if self.block_attnres_block_size <= 0 or self.block_attnres_block_size % 2:
            raise ValueError("block_attnres_block_size must be a positive even number.")
        if self.speaker_reference_max_seconds <= 0:
            raise ValueError("speaker_reference_max_seconds must be positive.")
        codebook_sizes = self.codebook_sizes
        if codebook_sizes is None:
            codebook_sizes = (int(self.codebook_size),) * int(self.n_codebooks)
        else:
            codebook_sizes = tuple(int(item) for item in codebook_sizes)
        if len(codebook_sizes) != int(self.n_codebooks):
            raise ValueError("codebook_sizes must contain one size per codebook.")
        if any(size <= 0 or size > int(self.codebook_size) for size in codebook_sizes):
            raise ValueError("Invalid per-codebook vocabulary size.")
        object.__setattr__(self, "codebook_sizes", codebook_sizes)
        if self.span_end_id is not None and int(self.span_end_id) != int(self.audio_pad_id) + 1:
            raise ValueError(
                f"span_end_id must equal audio_pad_id + 1 ({int(self.audio_pad_id) + 1}), got {self.span_end_id}."
            )

    @property
    def audio_vocab_size(self) -> int:
        """Per-codebook audio vocabulary: ordinary codes, EOA, pad and the optional span-end."""
        return compute_audio_vocab_size(self.codebook_size, self.span_end_id)

    def get_num_experts_per_tok(self, layer_idx: int) -> int:
        default_topk = self.num_experts_per_tok if self.num_experts_per_tok > 0 else 1
        return resolve_layer_topk(default_topk, self.special_topk_layers, layer_idx)

    @classmethod
    def from_checkpoint_config(cls, config) -> ModelConfig:
        return cls.from_zonos3_config(config)

    @classmethod
    def from_zonos3_config(cls, config) -> ModelConfig:
        """Create ModelConfig from Zonos3Config (training checkpoint format).

        Zonos3Config uses different field names than HuggingFace:
        - n_layers -> num_layers
        - dim -> hidden_size
        - head_dim -> head_dim
        - n_heads -> num_qo_heads
        - n_kv_heads -> num_kv_heads
        - ffn_dim_multiplier * dim -> intermediate_size
        - moe_ffn_dim_multiplier * dim -> moe_intermediate_size (when set)
        """
        # Calculate dimensions
        head_dim = config.head_dim
        dim = config.dim
        n_heads = config.n_heads if config.n_heads is not None else dim // head_dim
        n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else n_heads

        # Calculate intermediate size from ffn_dim_multiplier
        ffn_dim = int(config.ffn_dim_multiplier * dim)
        # Round to multiple_of
        multiple_of = getattr(config, "multiple_of", 256)
        intermediate_size = multiple_of * ((ffn_dim + multiple_of - 1) // multiple_of)

        # Expert FFNs can be intentionally finer or coarser than the dense
        # FFNs while preserving total/active MoE capacity. Older checkpoints
        # omit these fields and therefore retain the historical shared size.
        moe_ffn_dim_multiplier = getattr(config, "moe_ffn_dim_multiplier", None)
        if moe_ffn_dim_multiplier is None:
            moe_ffn_dim_multiplier = config.ffn_dim_multiplier
        moe_multiple_of = getattr(config, "moe_multiple_of", None)
        if moe_multiple_of is None:
            moe_multiple_of = multiple_of
        moe_ffn_dim = int(moe_ffn_dim_multiplier * dim)
        moe_intermediate_size = moe_multiple_of * ((moe_ffn_dim + moe_multiple_of - 1) // moe_multiple_of)

        # Total vocab = n_codebooks * audio_vocab + text_vocab (if present)
        span_end_id = getattr(config, "span_end_id", None)
        vocab_size = config.n_codebooks * compute_audio_vocab_size(config.codebook_size, span_end_id)
        if config.text_vocab is not None:
            vocab_size += config.text_vocab + 1 + int(getattr(config, "text_extra_tokens", 0))

        # MoE settings
        moe_n_experts = getattr(config, "moe_n_experts", 1)
        moe_router_topk = getattr(config, "moe_router_topk", 1)
        special_topk_layers = getattr(config, "special_topk_layers", None)
        moe_router_dim = getattr(config, "moe_router_dim", 256)
        moe_router_type = normalize_moe_router_type(getattr(config, "moe_router_type", "mlp"))
        moe_router_use_eda = bool(
            getattr(
                config,
                "moe_router_use_eda",
                getattr(config, "use_eda", True),
            )
        )
        moe_alpha = float(getattr(config, "moe_alpha", 1.0))
        get_routed_moe_dim(dim, moe_alpha)
        moe_impl = getattr(config, "moe_impl", "grouped")
        moe_balancing_strategy = normalize_moe_balancing_strategy(getattr(config, "moe_balancing_strategy", "legacy"))

        return cls(
            attention_pattern=getattr(config, "attention_pattern", "global"),
            num_layers=config.n_layers,
            num_qo_heads=n_heads,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_size=dim,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",  # Zonos3 uses SiLU
            rms_norm_eps=config.norm_eps,
            tie_word_embeddings=False,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_seqlen,
                base=config.rope_theta,
                scaling=None,
            ),
            num_experts=moe_n_experts if moe_n_experts > 1 else 0,
            num_experts_per_tok=moe_router_topk if moe_n_experts > 1 else 0,
            special_topk_layers=special_topk_layers,
            moe_intermediate_size=moe_intermediate_size if moe_n_experts > 1 else 0,
            norm_topk_prob=bool(getattr(config, "norm_topk_prob", False)),
            # Zonos3 fields
            n_codebooks=config.n_codebooks,
            codebook_size=config.codebook_size,
            codebook_sizes=tuple(
                getattr(config, "codebook_sizes", None) or [config.codebook_size] * config.n_codebooks
            ),
            audio_codec=str(getattr(config, "audio_codec", "opus_dual_3n_600k_sched600k_r4096_step600000")),
            text_vocab=config.text_vocab,
            text_extra_tokens=int(getattr(config, "text_extra_tokens", 0)),
            moe_n_experts=moe_n_experts,
            moe_start_from_layer=getattr(config, "moe_start_from_layer", 0),
            moe_end_from_layer=getattr(config, "moe_end_from_layer", 0),
            moe_router_dim=moe_router_dim,
            moe_router_type=moe_router_type,
            moe_router_use_eda=moe_router_use_eda,
            moe_alpha=moe_alpha,
            moe_impl=moe_impl,
            moe_balancing_strategy=moe_balancing_strategy,
            use_xsa=bool(getattr(config, "use_xsa", False)),
            residual_type=getattr(config, "residual_type", "standard"),
            block_attnres_block_size=int(getattr(config, "block_attnres_block_size", 4)),
            # TTS fields
            eoa_id=config.eoa_id,
            audio_pad_id=config.audio_pad_id,
            span_end_id=None if span_end_id is None else int(span_end_id),
            loss_softcap=config.loss_softcap,
            condition_tokens_enabled=bool(getattr(config, "condition_tokens_enabled", False)),
            condition_bucket_counts=(
                bucket_counts(
                    int(getattr(config, "condition_count", 3) or 3),
                    int(getattr(config, "condition_n_buckets", 10) or 10),
                    getattr(config, "condition_bucket_counts", None),
                )
                if getattr(config, "condition_tokens_enabled", False)
                else ()
            ),
            condition_count=int(getattr(config, "condition_count", 3) or 3),
            condition_names=tuple(str(item) for item in (getattr(config, "condition_names", None) or ())),
            condition_edges={
                str(name): tuple(float(x) for x in (edges or ()))
                for name, edges in (getattr(config, "condition_edges", None) or {}).items()
            }
            or None,
            condition_units=dict(getattr(config, "condition_units", None) or {}) or None,
            condition_flags={str(k): dict(v or {}) for k, v in (getattr(config, "condition_flags", None) or {}).items()}
            or None,
            speaker_enabled=getattr(config, "speaker_enabled", False),
            speaker_reference_prefix_enabled=bool(getattr(config, "speaker_reference_prefix_enabled", False)),
            speaker_reference_max_seconds=float(getattr(config, "speaker_reference_max_seconds", 30.0)),
            speaker_embedding_dim=getattr(config, "speaker_embedding_dim", 128),
            speaker_embedding_codec=getattr(config, "speaker_embedding_codec", None),
            speaker_lda_dim=getattr(config, "speaker_lda_dim", None),
            speaker_background_token_enabled=bool(getattr(config, "speaker_background_token_enabled", False)),
            accurate_mode_token_enabled=bool(getattr(config, "accurate_mode_token_enabled", False)),
            speaking_rate_num_buckets=getattr(config, "speaking_rate_num_buckets", 0),
            speaking_rate_buckets=tuple(getattr(config, "speaking_rate_buckets", None) or ()),
            quality_num_buckets=int(getattr(config, "quality_num_buckets", 0) or 0),
            quality_features=tuple(getattr(config, "quality_features", None) or ()),
            quality_buckets={
                str(feature): tuple(str(item) for item in (buckets or ()))
                for feature, buckets in (getattr(config, "quality_buckets", None) or {}).items()
            }
            or None,
            quality_dropout=dict(getattr(config, "quality_dropout", None) or {}) or None,
        )
