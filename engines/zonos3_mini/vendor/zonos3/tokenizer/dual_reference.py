"""Encode reference audio with the frozen fixed-600k dual codec (the family "dual" entry of the codec registry).

This is the encoder side of the matched codec that decodes the ZONOS3 LM's tokens: the MossAutoencoder adapter of
zonos3.codec (native SSL teacher -> semantic Q1, acoustic residual RVQ, 10 x 4096 codes at 15 Hz, hop 2940), built
from the registry entry's generator checkpoint and SSL teacher. Preprocessing mirrors the codec evaluation's prompt
encoding: mono, resample to 44.1 kHz with torchaudio, peak normalize only above 1.0, keep whole codec frames at the
reference limit (the training selector's rule).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from zonos3.generation import UPLOAD_MAX_SECONDS
from zonos3.tokenizer import matched_codec
from zonos3.utils import release

CODEC_RATE = 44100
HOP = 2940
CODEBOOKS = 10
_ENCODERS: dict = {}


def _dual_entry() -> dict:
    entries = [e for e in matched_codec.registry().values() if e.get("family") == "dual"]
    if len(entries) != 1:
        raise RuntimeError(f"expected exactly one dual-family codec in {matched_codec.REGISTRY}, found {len(entries)}")
    return entries[0]


def get_encoder(device: str | None = None):
    """The codec adapter's MossAutoencoder, one per device, loaded on first use."""
    device = device or os.environ.get("ZONOS3_AUDIO_CODEC_DEVICE", "cuda:0")
    if device in _ENCODERS:
        return _ENCODERS[device]
    # The codec's Transformers teacher must not pick up the TTS runtime's pinned `kernels` module.
    saved_kernels = sys.modules.get("kernels")
    sys.modules["kernels"] = None  # type: ignore[assignment]
    try:
        from transformers import Wav2Vec2BertModel  # noqa: F401  (import under the shim)
    finally:
        if saved_kernels is None:
            sys.modules.pop("kernels", None)
        else:
            sys.modules["kernels"] = saved_kernels
    from zonos3.codec.codec_adapter import MossAutoencoder

    e = _dual_entry()
    index = torch.device(device).index or 0
    previous = torch.cuda.current_device()
    torch.cuda.set_device(index)
    try:
        encoder = MossAutoencoder(
            SimpleNamespace(
                hybrid_batch_size=1,
                hybrid_chunk_seconds=20.0,
                hybrid_context_seconds=10.0,
                hybrid_cuda_graph=os.environ.get("ZONOS3_DUAL_CODEC_CUDA_GRAPH", "1") == "1",
            ),
            generator_path=release.resolve_file(e["checkpoint"]),
            teacher_dir=release.resolve_dir_or_local(e["teacher"]),
            step=int(e["codec_step"]),
            codebook_sizes=e["codebook_sizes"],
            name=e["codec"],
            checkpoint_sha256=e["checkpoint_sha256"],
            device=device,
        )
    finally:
        torch.cuda.set_device(previous)
    _ENCODERS[device] = encoder
    return encoder


def prepare_reference(waveform: torch.Tensor, sample_rate: int, max_seconds: float) -> torch.Tensor:
    if not 0 < max_seconds <= UPLOAD_MAX_SECONDS:
        raise ValueError(f"Codec input limit must be in (0, {UPLOAD_MAX_SECONDS:g}] seconds")
    if waveform.ndim == 1:
        waveform = waveform[None]
    if waveform.ndim != 2 or not waveform.shape[-1] or not torch.isfinite(waveform).all():
        raise ValueError("Expected a finite, nonempty reference waveform")
    waveform = waveform.detach().cpu().float().mean(0, keepdim=True)
    if sample_rate != CODEC_RATE:
        import torchaudio.functional as AF

        waveform = AF.resample(waveform, int(sample_rate), CODEC_RATE)
    if waveform.shape[-1] < CODEC_RATE // 2:
        raise ValueError("Reference shorter than 0.5 s")
    peak = float(waveform.abs().amax())
    if peak > 1.0:  # as encode_prompts.py / the interactive codec service
        waveform = waveform / peak
    maximum = math.floor(max_seconds * CODEC_RATE / HOP) * HOP
    return waveform[:, :maximum].contiguous()


@torch.inference_mode()
def encode_dual_reference(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    max_seconds: float = 30.0,
    device: str | None = None,
) -> torch.Tensor:
    """Waveform -> int32 codes [frames, 10] for the reference prefix."""
    waveform = prepare_reference(waveform, sample_rate, max_seconds)
    encoder = get_encoder(device)
    index = torch.device(device or os.environ.get("ZONOS3_AUDIO_CODEC_DEVICE", "cuda:0")).index or 0
    previous_device = torch.cuda.current_device()
    previous_stream = torch.cuda.current_stream(previous_device)
    torch.cuda.set_device(index)
    try:
        codes = np.asarray(encoder.encode(waveform.numpy(), CODEC_RATE))
    finally:
        torch.cuda.set_device(previous_device)
        torch.cuda.set_stream(previous_stream)
    expected = math.ceil(waveform.shape[-1] / HOP)
    if codes.ndim != 2 or codes.shape != (expected, CODEBOOKS):
        raise RuntimeError(f"dual codec returned codes of shape {codes.shape}; expected {(expected, CODEBOOKS)}")
    if codes.min() < 0 or codes.max() >= 4096:
        raise RuntimeError("dual codec returned out-of-range codes")
    return torch.from_numpy(codes.astype(np.int32)).contiguous()


def encode_dual_reference_file(
    path: str | Path, *, max_seconds: float = 30.0, device: str | None = None
) -> torch.Tensor:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return encode_dual_reference(torch.from_numpy(wav.T.copy()), int(sr), max_seconds=max_seconds, device=device)
