"""Reference-audio decoding (ffmpeg -> PCM WAV -> mono float tensor), the served PCM's output stage (gain + soft
limiter) and the one PCM16 WAV encoder.

Shared by the HTTP server and the offline :class:`~zonos3_sgl.tts.llm.TTSLLM`; has no
FastAPI / server dependency.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import wave

import numpy as np
import torch

from zonos3.generation import OUTPUT_LIMIT_DBFS

logger = logging.getLogger(__name__)

# The soft limiter's knee (generation registry output_limit_dbfs) as a linear amplitude.
OUTPUT_LIMIT_KNEE = np.float32(10.0 ** (OUTPUT_LIMIT_DBFS / 20.0))


def output_pcm(samples: np.ndarray, gain: float) -> np.ndarray:
    """Decoder audio -> the served float32 PCM: times the output ``gain`` (the decoder's headroom), then the soft
    limiter. Samples at or below the knee k (OUTPUT_LIMIT_KNEE) pass unchanged; louder ones become
    ``sign(x) * (k + (1 - k) * tanh((|x| - k) / (1 - k)))``: slope 1 at the knee, bounded by full scale (float32 tanh
    reaches it only past |x| ~ 1.9). Per sample, so streamed chunks equal the whole audio."""
    x = np.asarray(samples, dtype=np.float32)
    if gain != 1.0:
        x = x * np.float32(gain)
    over = np.abs(x) > OUTPUT_LIMIT_KNEE
    if not over.any():
        return x
    width = np.float32(1.0) - OUTPUT_LIMIT_KNEE
    y = x.copy()
    peak = x[over]
    y[over] = np.sign(peak) * (OUTPUT_LIMIT_KNEE + width * np.tanh((np.abs(peak) - OUTPUT_LIMIT_KNEE) / width))
    return y


def pcm16_samples(pcm: np.ndarray) -> np.ndarray:
    """Float PCM -> little-endian int16: NaN -> 0, clip to [-1, 1], x 32767, round half to even (libsndfile's
    ``lrintf(x * 32767)``)."""
    x = np.nan_to_num(np.asarray(pcm, dtype=np.float32), nan=0.0)
    return np.rint(np.clip(x, -1.0, 1.0) * np.float32(32767)).astype("<i2")


def pcm16_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Mono 16-bit WAV of float PCM (:func:`pcm16_samples`)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm16_samples(pcm).tobytes())
    return buf.getvalue()


def write_pcm16_wav(path: str | os.PathLike, pcm: np.ndarray, sample_rate: int) -> None:
    with open(path, "wb") as f:
        f.write(pcm16_wav_bytes(pcm, sample_rate))


def _decode_wav_bytes(wav_bytes: bytes) -> tuple[torch.Tensor, int]:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            pcm = wav_file.readframes(n_frames)
    except Exception as exc:  # pragma: no cover - branch depends on invalid user payload
        raise ValueError("Reference audio must be a valid PCM WAV file.") from exc

    if len(pcm) == 0:
        raise ValueError("Reference audio is empty.")

    pcm_view = memoryview(bytearray(pcm))

    if sample_width == 1:
        audio = torch.frombuffer(pcm_view, dtype=torch.uint8).to(torch.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = torch.frombuffer(pcm_view, dtype=torch.int16).to(torch.float32)
        audio = audio / 32768.0
    elif sample_width == 4:
        audio = torch.frombuffer(pcm_view, dtype=torch.int32).to(torch.float32)
        audio = audio / 2147483648.0
    else:
        raise ValueError("Unsupported WAV bit depth. Use 8/16/32-bit PCM WAV.")

    if channels > 1:
        audio = audio.view(-1, channels).transpose(0, 1).contiguous()
    else:
        audio = audio.view(1, -1)

    return audio, sample_rate


def _transcode_audio_bytes_to_wav(audio_bytes: bytes) -> bytes:
    if not audio_bytes:
        raise ValueError("Reference file is empty.")

    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                "pipe:1",
            ],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ValueError("Reference file must contain an audio stream supported by ffmpeg.") from exc

    if proc.returncode != 0 or not proc.stdout:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        suffix = f" ffmpeg said: {stderr}" if stderr else ""
        raise ValueError("Reference file must contain an audio stream supported by ffmpeg." + suffix)

    logger.debug("Converted reference media to PCM WAV via ffmpeg for speaker embedding.")
    return proc.stdout


def _decode_audio_bytes(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    return _decode_wav_bytes(_transcode_audio_bytes_to_wav(audio_bytes))
