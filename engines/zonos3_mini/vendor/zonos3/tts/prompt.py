from __future__ import annotations

from dataclasses import dataclass

import torch

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKEN_IDS = (PAD_ID, UNK_ID, BOS_ID, EOS_ID)

# 192 legacy symbol IDs followed by 256 byte IDs.
# With 21 speaking-rate buckets, text_vocab is 469:
# bytes occupy 192..447, rate buckets occupy 448..468, and 469 is text padding.
LEGACY_SYMBOL_VOCAB_SIZE = 192
BYTE_VOCAB_SIZE = 256
BYTE_TEXT_VOCAB_SIZE = LEGACY_SYMBOL_VOCAB_SIZE + BYTE_VOCAB_SIZE


@dataclass(frozen=True)
class TTSPromptConfig:
    n_codebooks: int = 9
    codebook_sizes: tuple[int, ...] | None = None
    audio_pad_id: int = 1025
    text_vocab: int = BYTE_TEXT_VOCAB_SIZE
    text_tokenizer_spec: dict | None = None
    speaking_rate_num_buckets: int = 0
    quality_bucket_counts: tuple[int, ...] = ()
    speaker_background_num_buckets: int = 0
    accurate_mode_num_buckets: int = 0

    def __post_init__(self) -> None:
        n_codebooks = int(self.n_codebooks)
        audio_pad_id = int(self.audio_pad_id)
        if self.text_vocab is None:
            raise ValueError("text_vocab is required for TTS prompt construction.")
        text_vocab = int(self.text_vocab)
        speaking_rate_num_buckets = int(self.speaking_rate_num_buckets)
        quality_bucket_counts = _normalize_quality_bucket_counts(self.quality_bucket_counts)
        speaker_background_num_buckets = int(self.speaker_background_num_buckets)
        accurate_mode_num_buckets = int(self.accurate_mode_num_buckets)

        if n_codebooks <= 0:
            raise ValueError("n_codebooks must be positive.")
        if audio_pad_id < 0:
            raise ValueError("audio_pad_id must be non-negative.")
        if speaking_rate_num_buckets < 0:
            raise ValueError("speaking_rate_num_buckets must be non-negative.")
        if speaker_background_num_buckets < 0:
            raise ValueError("speaker_background_num_buckets must be non-negative.")
        if accurate_mode_num_buckets < 0:
            raise ValueError("accurate_mode_num_buckets must be non-negative.")

        if text_vocab < BYTE_TEXT_VOCAB_SIZE:
            raise ValueError(
                "text_vocab must include the byte vocabulary; "
                f"expected at least {BYTE_TEXT_VOCAB_SIZE}, got {text_vocab}."
            )
        # Validates that all conditioning buckets fit inside text_vocab.
        _conditioning_base_text_vocab(
            text_vocab,
            speaking_rate_num_buckets,
            quality_bucket_counts,
            speaker_background_num_buckets,
            accurate_mode_num_buckets,
            context="TTS prompt construction",
        )

        object.__setattr__(self, "n_codebooks", n_codebooks)
        object.__setattr__(self, "audio_pad_id", audio_pad_id)
        object.__setattr__(self, "text_vocab", text_vocab)
        object.__setattr__(self, "speaking_rate_num_buckets", speaking_rate_num_buckets)
        object.__setattr__(self, "quality_bucket_counts", quality_bucket_counts)
        object.__setattr__(self, "speaker_background_num_buckets", speaker_background_num_buckets)
        object.__setattr__(self, "accurate_mode_num_buckets", accurate_mode_num_buckets)


def byte_text_vocab_size() -> int:
    return BYTE_TEXT_VOCAB_SIZE


def conditioned_text_vocab_size(
    speaking_rate_num_buckets: int = 0,
    quality_num_buckets: int = 0,
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> int:
    counts = (
        int(speaking_rate_num_buckets),
        int(quality_num_buckets),
        int(speaker_background_num_buckets),
        int(accurate_mode_num_buckets),
    )
    if any(count < 0 for count in counts):
        raise ValueError("conditioning bucket counts must be non-negative.")
    return BYTE_TEXT_VOCAB_SIZE + sum(counts)


def text_to_byte_ids(text: str) -> list[int]:
    return [BOS_ID, *(byte + LEGACY_SYMBOL_VOCAB_SIZE for byte in text.encode("utf-8")), EOS_ID]


def _normalize_quality_bucket_counts(quality_bucket_counts) -> tuple[int, ...]:
    counts = tuple(int(count) for count in (quality_bucket_counts or ()))
    if any(count < 0 for count in counts):
        raise ValueError("quality_bucket_counts must be non-negative.")
    return counts


def _conditioning_base_text_vocab(
    text_vocab: int | None,
    speaking_rate_num_buckets: int,
    quality_bucket_counts=(),
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
    *,
    context: str,
) -> int:
    """First conditioning token id; everything below it is normal text vocabulary.

    Conditioning tokens occupy the tail of the text vocabulary in this order:
    speaking-rate buckets (legacy checkpoints; never emitted, but they still
    shift the ids after them), quality buckets (per feature), speaker-background
    markers (clean, noisy), accurate-mode marker. text_vocab itself is padding.
    """
    if text_vocab is None:
        raise ValueError(f"text_vocab is required for {context}.")

    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    base_text_vocab = (
        int(text_vocab)
        - int(speaking_rate_num_buckets)
        - sum(counts)
        - int(speaker_background_num_buckets)
        - int(accurate_mode_num_buckets)
    )
    if base_text_vocab < 0:
        raise ValueError(
            "text_vocab is smaller than the configured conditioning buckets; cannot locate conditioning tokens."
        )
    return base_text_vocab


def quality_token_id(
    text_vocab: int | None,
    speaking_rate_num_buckets: int,
    quality_bucket_counts,
    feature_idx: int,
    quality_bucket: int,
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> int:
    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    if not counts:
        raise ValueError("Current model does not define quality buckets.")

    feature = int(feature_idx)
    if feature < 0 or feature >= len(counts):
        raise ValueError(f"quality feature index must be in [0, {len(counts) - 1}], got {feature}.")

    num_buckets = counts[feature]
    if num_buckets <= 0:
        raise ValueError(f"quality feature {feature} does not define buckets.")

    bucket = int(quality_bucket)
    if bucket < 0 or bucket >= num_buckets:
        raise ValueError(f"quality bucket for feature {feature} must be in [0, {num_buckets - 1}], got {bucket}.")

    base_text_vocab = _conditioning_base_text_vocab(
        text_vocab,
        speaking_rate_num_buckets,
        counts,
        speaker_background_num_buckets,
        accurate_mode_num_buckets,
        context="quality conditioning",
    )
    return base_text_vocab + int(speaking_rate_num_buckets) + sum(counts[:feature]) + bucket


def speaker_background_token_id(
    text_vocab: int | None,
    speaking_rate_num_buckets: int,
    quality_bucket_counts,
    clean: bool,
    speaker_background_num_buckets: int = 2,
    accurate_mode_num_buckets: int = 0,
) -> int:
    """Token id for the clean/noisy speaker-background marker."""
    num_buckets = int(speaker_background_num_buckets)
    if num_buckets < 2:
        raise ValueError("speaker_background_num_buckets must be at least 2 for background tokens.")

    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    base_text_vocab = _conditioning_base_text_vocab(
        text_vocab,
        speaking_rate_num_buckets,
        counts,
        num_buckets,
        accurate_mode_num_buckets,
        context="speaker-background conditioning",
    )
    return base_text_vocab + int(speaking_rate_num_buckets) + sum(counts) + (0 if bool(clean) else 1)


def accurate_mode_token_id(
    text_vocab: int | None,
    speaking_rate_num_buckets: int,
    quality_bucket_counts,
    speaker_background_num_buckets: int = 2,
    accurate_mode_num_buckets: int = 1,
) -> int:
    """Token id for the accurate-mode marker (absent = expressive mode)."""
    accurate_count = int(accurate_mode_num_buckets)
    if accurate_count <= 0:
        raise ValueError("accurate_mode_num_buckets must be positive.")
    background_count = int(speaker_background_num_buckets)
    if background_count < 2:
        raise ValueError("speaker_background_num_buckets must be at least 2 for accurate-mode tokens.")

    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    base_text_vocab = _conditioning_base_text_vocab(
        text_vocab,
        speaking_rate_num_buckets,
        counts,
        background_count,
        accurate_count,
        context="accurate-mode conditioning",
    )
    return base_text_vocab + int(speaking_rate_num_buckets) + sum(counts) + background_count


def tokens_to_prompt_tokens(
    tokens: list[int],
    n_codebooks: int = 9,
    audio_pad_id: int = 1025,
    text_vocab: int | None = None,
    speaking_rate_num_buckets: int = 0,
    quality_bucket_counts=(),
    quality_buckets=None,
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> list[list[int]]:
    quality_bucket_counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    if text_vocab is None:
        text_vocab = conditioned_text_vocab_size(
            speaking_rate_num_buckets,
            sum(quality_bucket_counts),
            speaker_background_num_buckets,
            accurate_mode_num_buckets,
        )
    config = TTSPromptConfig(
        n_codebooks=n_codebooks,
        audio_pad_id=audio_pad_id,
        text_vocab=text_vocab,
        speaking_rate_num_buckets=speaking_rate_num_buckets,
        quality_bucket_counts=quality_bucket_counts,
        speaker_background_num_buckets=speaker_background_num_buckets,
        accurate_mode_num_buckets=accurate_mode_num_buckets,
    )
    return _text_rows(tokens, config, quality_buckets=quality_buckets)


def text_to_prompt_tokens(
    text: str,
    n_codebooks: int = 9,
    audio_pad_id: int = 1025,
    text_vocab: int | None = None,
    speaking_rate_num_buckets: int = 0,
    quality_bucket_counts=(),
    quality_buckets=None,
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> list[list[int]]:
    return tokens_to_prompt_tokens(
        text_to_byte_ids(text),
        n_codebooks=n_codebooks,
        audio_pad_id=audio_pad_id,
        text_vocab=text_vocab,
        speaking_rate_num_buckets=speaking_rate_num_buckets,
        quality_bucket_counts=quality_bucket_counts,
        quality_buckets=quality_buckets,
        speaker_background_num_buckets=speaker_background_num_buckets,
        accurate_mode_num_buckets=accurate_mode_num_buckets,
    )


def shear(x: torch.Tensor, pad: int) -> torch.Tensor:
    T, C = x.shape
    padded = x.new_full((C - 1 + T, C), pad)
    padded[C - 1 :] = x
    row_idx = (C - 1) + torch.arange(T, device=x.device).unsqueeze(1) - torch.arange(C, device=x.device)
    return padded.gather(0, row_idx)


def silence_rows_from_codes(frame: torch.Tensor, frames: int, config: TTSPromptConfig) -> torch.Tensor:
    """Codec-matched silence prefill (2026-10-17): `frames` copies of one silence frame as the first rows of the audio
    stream in the training layout (sheared, WITHOUT the flushed tail, so the LM continues the delay pattern from the
    next row), text column = the text pad."""
    frames = int(frames)
    if frames < 1:
        raise ValueError("silence_prefill_frames must be >= 1")
    codes = frame.to(torch.int32).reshape(1, -1)
    if codes.shape[1] != config.n_codebooks:
        raise ValueError(f"silence frame has {codes.shape[1]} codebooks, the model {config.n_codebooks}")
    sheared = shear(codes.repeat(frames, 1), config.audio_pad_id)[:frames]
    text_col = torch.full((frames, 1), config.text_vocab, dtype=torch.int32)
    return torch.cat([sheared, text_col], dim=1)


def speaker_reference_prefix_tokens(
    codes: torch.Tensor,
    config: TTSPromptConfig,
) -> torch.Tensor:
    """Encode raw codec codes exactly like the training reference prefix.

    The reference is unsupervised context before the text prompt. Each codebook
    is delayed by its index and the tail is flushed with ordinary audio padding;
    there is intentionally no separator or learned boundary token.
    """

    if codes.ndim != 2 or codes.shape[1] != config.n_codebooks:
        raise ValueError(
            f"Speaker-reference codes must have shape [time, {config.n_codebooks}], got {tuple(codes.shape)}."
        )
    if codes.shape[0] == 0:
        raise ValueError("Speaker-reference codes must not be empty.")
    sizes = config.codebook_sizes or (1024,) * config.n_codebooks
    if len(sizes) != config.n_codebooks or any(v <= 0 for v in sizes):
        raise ValueError("Invalid speaker-reference codebook sizes.")
    limits = torch.tensor(sizes, device=codes.device)[None, :]
    if bool(torch.any(codes < 0)) or bool(torch.any(codes >= limits)):
        raise ValueError("Speaker-reference codes exceed their codebook vocabulary.")

    length = codes.shape[0] + config.n_codebooks - 1
    audio_columns = torch.full(
        (length, config.n_codebooks),
        config.audio_pad_id,
        dtype=torch.int32,
        device=codes.device,
    )
    audio_columns[: codes.shape[0]] = codes.to(dtype=torch.int32)
    delayed = shear(audio_columns, config.audio_pad_id)
    text_column = torch.full(
        (length, 1),
        config.text_vocab,
        dtype=torch.int32,
        device=codes.device,
    )
    return torch.cat((delayed, text_column), dim=1)


def make_speaker_slot(
    config: TTSPromptConfig,
    *,
    dtype: torch.dtype = torch.int32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    slot = torch.full((1, config.n_codebooks + 1), config.audio_pad_id, dtype=dtype, device=device)
    slot[:, config.n_codebooks] = config.text_vocab
    return slot


def _text_rows(
    tokens: list[int],
    config: TTSPromptConfig,
    *,
    quality_buckets=None,
) -> list[list[int]]:
    rows: list[list[int]] = []
    if quality_buckets is not None:
        for feature_idx, bucket in enumerate(quality_buckets):
            if bucket is None:
                continue
            rows.append(
                [config.audio_pad_id] * config.n_codebooks
                + [
                    quality_token_id(
                        config.text_vocab,
                        config.speaking_rate_num_buckets,
                        config.quality_bucket_counts,
                        feature_idx,
                        bucket,
                        config.speaker_background_num_buckets,
                        config.accurate_mode_num_buckets,
                    )
                ]
            )

    rows.extend([config.audio_pad_id] * config.n_codebooks + [token] for token in tokens)
    return rows


class TTSPromptBuilder:
    def __init__(self, config: TTSPromptConfig):
        self.config = config
        if config.text_tokenizer_spec is not None:
            from .superbpe import validate_spec

            validate_spec(config.text_tokenizer_spec, config.text_vocab)

    def text_ids(self, text):
        if self.config.text_tokenizer_spec is not None:
            from .superbpe import text_ids

            return text_ids(text, self.config.text_tokenizer_spec)
        return text_to_byte_ids(text)

    def build_text_prompt(
        self,
        text: str,
        *,
        quality_buckets=None,
    ) -> torch.Tensor:
        rows = _text_rows(self.text_ids(text), self.config, quality_buckets=quality_buckets)
        return torch.tensor(rows, dtype=torch.int32)

    def build(
        self,
        text: str,
        *,
        quality_buckets=None,
    ) -> torch.Tensor:
        return self.build_text_prompt(text, quality_buckets=quality_buckets)

    def condition_rows(
        self,
        count: int,
        *,
        dtype: torch.dtype = torch.int32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """``count`` all-pad rows (audio pad on every codebook, text pad) for the loudness / fmax / speaking-rate
        buckets of the cond3 finetunes; the model replaces their embeddings by the bucket tables."""
        rows = torch.full(
            (int(count), self.config.n_codebooks + 1), self.config.audio_pad_id, dtype=dtype, device=device
        )
        rows[:, self.config.n_codebooks] = self.config.text_vocab
        return rows

    def speaker_slot(
        self,
        *,
        dtype: torch.dtype = torch.int32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return make_speaker_slot(self.config, dtype=dtype, device=device)

    def speaker_reference_prefix(self, codes: torch.Tensor) -> torch.Tensor:
        return speaker_reference_prefix_tokens(codes, self.config)
