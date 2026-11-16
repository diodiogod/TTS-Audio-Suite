"""Classifier-free guidance on the speaker conditioning: the raw codec reference prefix or the reserved speaker-embedding
row. The unconditional twin is the same prompt with that conditioning removed (and, on condition-row models, all-unknown
condition rows): the training-side evaluation generator's twin. guided = uncond + scale * (cond - uncond)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


def is_reserved_speaker_slot(frame: torch.Tensor, *, n_codebooks: int, audio_pad_id: int, text_vocab: int) -> bool:
    """The all-pad row (audio pad on every codebook, text pad) whose embedding the speaker projection replaces."""
    return (
        frame.numel() > n_codebooks
        and bool(torch.all(frame[:n_codebooks] == int(audio_pad_id)))
        and int(frame[n_codebooks]) == int(text_vocab)
    )


def unconditional_prompt(
    msg,
    *,
    n_codebooks: int,
    audio_pad_id: int,
    input_ids: torch.Tensor | None = None,
    speaker_token_position: int | None = None,
    text_vocab: int | None = None,
) -> torch.Tensor:
    """The prompt with its speaker conditioning removed, never padded in place: the recorded reference prefix (raw
    reference path) or the reserved speaker row at position 0 (embedding path; `input_ids` / `speaker_token_position`
    are the scheduler's final prompt when it added the row itself)."""
    tokens = msg.input_ids if input_ids is None else input_ids
    # A controls delta rides along: hidden-space (attnres_source) deltas steer both branches (the scheduler copies
    # them to the twin).
    if len(tokens) == 0:
        raise ValueError("Speaker CFG requires a text-ending target prompt")
    # text / condition / speaker rows are all-pad audio rows; rows after the text may be a prefilled audio opening
    # (codec silence, 2026-10-17) that both branches keep
    if not bool(torch.all(tokens[:, :n_codebooks] == audio_pad_id, dim=1).any()):
        raise ValueError("Speaker CFG requires a text-ending target prompt")
    prefix = int(msg.speaker_reference_prefix_length)
    if msg.speaker_embedding is not None:
        if prefix:
            raise ValueError("Speaker CFG: a speaker embedding and a reference prefix are mutually exclusive")
        position = msg.speaker_token_position if speaker_token_position is None else speaker_token_position
        if (
            position != 0
            or text_vocab is None
            or len(tokens) < 2
            or not is_reserved_speaker_slot(
                tokens[0], n_codebooks=n_codebooks, audio_pad_id=audio_pad_id, text_vocab=text_vocab
            )
        ):
            raise ValueError("Speaker CFG on the embedding path needs the reserved speaker row at position 0")
        return tokens[1:].clone()
    if not 0 < prefix < len(tokens):
        raise ValueError("Speaker CFG requires a nonempty reference prefix and target prompt")

    return tokens[prefix:].clone()


@dataclass
class SpeakerCfgTwin:
    input_ids: torch.Tensor
    removed_rows: int  # rows the twin lacks in front: the reference prefix or the speaker row
    condition_values: list[int] | None  # all unknown on condition-row models
    condition_token_start: int  # shifted back by removed_rows


def speaker_cfg_twin(
    msg,
    input_ids: torch.Tensor,
    *,
    n_codebooks: int,
    audio_pad_id: int,
    text_vocab: int,
    speaker_token_position: int,
    condition_values: list[int] | None,
    condition_token_start: int,
    condition_unknowns: Sequence[int],
) -> SpeakerCfgTwin:
    """Everything the scheduler needs for the speaker-CFG twin of a request: the prompt without its speaker conditioning,
    no speaker embedding (the caller drops it), and on condition-row models all-unknown rows shifted back by the removed
    rows. Identical for both paths to the evaluation generator's twin."""
    twin_ids = unconditional_prompt(
        msg,
        n_codebooks=n_codebooks,
        audio_pad_id=audio_pad_id,
        input_ids=input_ids,
        speaker_token_position=speaker_token_position,
        text_vocab=text_vocab,
    )
    removed = int(len(input_ids) - len(twin_ids))
    if condition_values is not None:
        if len(condition_unknowns) != len(condition_values):
            raise ValueError("Speaker CFG twin: one unknown id per condition row")
        values: list[int] | None = [int(u) for u in condition_unknowns]
        start = int(condition_token_start) - removed
        if start < 0:
            raise ValueError("Speaker CFG twin: the condition rows would start before the prompt")
    else:
        values, start = None, int(condition_token_start)
    return SpeakerCfgTwin(
        input_ids=twin_ids, removed_rows=removed, condition_values=values, condition_token_start=start
    )


def guided_logits(conditional: torch.Tensor, unconditional: torch.Tensor, scale: float) -> torch.Tensor:
    """Mix raw logits in FP32 before any sampler transforms or invalid-code masks."""
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Speaker CFG scale must be finite and nonnegative")
    if conditional.shape != unconditional.shape:
        raise ValueError("CFG branches must have identical logit shapes")
    if scale == 1.0:
        return conditional.float()
    if scale == 0.0:
        return unconditional.float()
    return unconditional.float() + scale * (conditional.float() - unconditional.float())
