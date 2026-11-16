"""Codec edit prompts, generated-fill extraction and exact unmasked-context splicing."""

import torch

from zonos3.tts.delay import shear_up

from .prompt import TTSPromptConfig, shear


def edit_marker_rows(text_extra_tokens: int, text_tokenizer_spec: dict | None) -> int:
    """Edit marker rows of a checkpoint: the extra text rows before the phoneme / start / pause rows (all of them
    without those tokens). Markers sit at text_vocab + 1 + j, where edit training put them; superbpe.load_spec checks
    the phoneme / start / pause rows fill the rest of text_extra_tokens."""
    from .superbpe import extra_token_count, rows_before_extra_tokens

    rows = extra_token_count(text_tokenizer_spec)
    if rows == 0:
        return int(text_extra_tokens)
    markers = rows_before_extra_tokens(text_tokenizer_spec, int(text_tokenizer_spec["vocab_size"]) + 4)
    if markers + rows != int(text_extra_tokens):
        raise ValueError("Phoneme / start / pause tokens must be the last extra text rows")
    return markers


def edit_marker_id(config: TTSPromptConfig, index: int) -> int:
    return config.text_vocab + 1 + index


def validate_edit_spans(spans, total_frames: int, max_spans: int) -> None:
    if not spans or len(spans) > max_spans:
        raise ValueError("Edit requires between one and max_spans regions")
    previous_end = -1
    for start, length in spans:
        if start < 0 or length < 1 or start + length > total_frames or start <= previous_end:
            raise ValueError("Edit spans must be ascending, separated and within the audio")
        previous_end = start + length


def sheared_block(frames, config: TTSPromptConfig, *, end_id: int | None):
    """Delay-stack frames (+ one all-``end_id`` row when given) followed by K-1 pad rows."""
    frames = torch.as_tensor(frames, dtype=torch.int32, device="cpu")
    k = config.n_codebooks
    if frames.ndim != 2 or frames.shape[1] != k:
        raise ValueError("Invalid codec frame shape")
    has_end = end_id is not None
    audio = torch.full((len(frames) + int(has_end) + k - 1, k), config.audio_pad_id, dtype=torch.int32)
    audio[: len(frames)] = frames
    if has_end:
        audio[len(frames)] = end_id
    return torch.cat(
        (shear(audio, config.audio_pad_id), torch.full((len(audio), 1), config.text_vocab, dtype=torch.int32)), dim=1
    )


def edit_prompt_tokens(text_rows, codes, spans, fills, config: TTSPromptConfig, *, span_end_id: int):
    """Training edit layout up to the next infill marker.

    The unmasked context ends with EOA (audio_pad_id - 1); each already generated
    fill ends with the span-end token. The prompt ends on marker ``len(fills)``.
    """
    if span_end_id != config.audio_pad_id + 1:
        raise ValueError("span_end_id must equal audio_pad_id + 1")
    codes = torch.as_tensor(codes, dtype=torch.int32, device="cpu")
    validate_edit_spans(spans, len(codes), len(spans))
    if len(fills) >= len(spans):
        raise ValueError("Prompt needs an ungenerated span")

    def marker(j):
        row = torch.full((1, config.n_codebooks + 1), config.audio_pad_id, dtype=torch.int32)
        row[0, -1] = edit_marker_id(config, j)
        return row

    parts = [torch.as_tensor(text_rows, dtype=torch.int32, device="cpu")]
    cursor = 0
    for j, (start, length) in enumerate(spans):
        if start > cursor:
            parts.append(sheared_block(codes[cursor:start], config, end_id=None))
        parts.append(marker(j))
        cursor = start + length
    parts.append(sheared_block(codes[cursor:], config, end_id=config.audio_pad_id - 1))
    for j, fill in enumerate(fills):
        parts.extend((marker(j), sheared_block(fill, config, end_id=span_end_id)))
    parts.append(marker(len(fills)))
    return torch.cat(parts)


def extract_fill(delayed_rows, eos_frame: int | None, *, audio_pad_id: int, codebook_sizes):
    """Codec frames of one generated span and whether it ended on span-end.

    ``delayed_rows`` are the generated delay-stacked rows without the finishing
    row. Without an end token the incomplete K-1 delay tail is dropped; the fill
    is cut before the first frame holding a non-codec token.
    """
    limits = torch.tensor(tuple(codebook_sizes), dtype=torch.int64)
    frames = shear_up(torch.as_tensor(delayed_rows, dtype=torch.int64).reshape(-1, len(limits)), audio_pad_id)
    fill = frames[: max(0, eos_frame if eos_frame is not None else len(frames) - (len(limits) - 1))]
    invalid = ((fill < 0) | (fill >= limits)).any(dim=1)
    truncated = bool(invalid.any())
    if truncated:
        fill = fill[: int(invalid.nonzero()[0, 0])]
    return fill, eos_frame is not None and not truncated


def splice_edit(codes, spans, fills):
    codes = torch.as_tensor(codes, dtype=torch.int64, device="cpu")
    validate_edit_spans(spans, len(codes), len(spans))
    if len(fills) != len(spans):
        raise ValueError("One fill is required per span")
    parts, cursor = [], 0
    for (start, length), fill in zip(spans, fills):
        fill = torch.as_tensor(fill, dtype=torch.int64, device="cpu")
        if fill.ndim != 2 or fill.shape[1] != codes.shape[1]:
            raise ValueError("Invalid fill shape")
        parts.extend((codes[cursor:start], fill))
        cursor = start + length
    parts.append(codes[cursor:])
    return torch.cat(parts)
