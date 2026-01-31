# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import io
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf

AudioLike = Union[
    str,                      # wav path / URL / base64
    Tuple[np.ndarray, int],   # (waveform, sr)
]
MaybeList = Union[Any, List[Any]]

SAMPLE_RATE = 16000
MAX_ASR_INPUT_SECONDS = 1200
MAX_FORCE_ALIGN_INPUT_SECONDS = 180
MIN_ASR_INPUT_SECONDS = 0.5
SUPPORTED_LANGUAGES: List[str] = [
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian"
]
_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def normalize_language_name(language: str) -> str:
    """
    Normalize language name to the canonical format used by Qwen3-ASR:
    first letter uppercase, the rest lowercase (e.g., 'cHINese' -> 'Chinese').

    Args:
        language (str): Input language name.

    Returns:
        str: Normalized language name.

    Raises:
        ValueError: If language is empty.
    """
    if language is None:
        raise ValueError("language is None")
    s = str(language).strip()
    if not s:
        raise ValueError("language is empty")
    return s[:1].upper() + s[1:].lower()


def validate_language(language: str) -> None:
    """
    Validate the language is supported.

    Args:
        language (str): Canonical language name.

    Raises:
        ValueError: If unsupported.
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")


def ensure_list(x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]


def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def is_probably_base64(s: str) -> bool:
    if s.startswith("data:audio"):
        return True
    if ("/" not in s and "\\" not in s) and len(s) > 256:
        return True
    return False


def decode_base64_bytes(b64: str) -> bytes:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def load_audio_any(x: str) -> Tuple[np.ndarray, int]:
    if is_url(x):
        with urllib.request.urlopen(x) as resp:
            audio_bytes = resp.read()
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    elif is_probably_base64(x):
        audio_bytes = decode_base64_bytes(x)
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    else:
        audio, sr = librosa.load(x, sr=None, mono=False)

    audio = np.asarray(audio, dtype=np.float32)
    sr = int(sr)
    return audio, sr


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    # soundfile can return shape (T, C); some pipelines use (C, T)
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        return np.mean(audio, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported audio ndim={audio.ndim}")


def float_range_normalize(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    # If decoded audio is int-like scaled or out-of-range, normalize conservatively.
    if peak > 1.0:
        audio = audio / peak
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def normalize_audio_input(a: AudioLike) -> np.ndarray:
    """
    Normalize one audio input to mono 16k float32 waveform in [-1, 1].

    Supported inputs:
        - str: local file path / https URL / base64 audio string
        - (np.ndarray, sr): waveform and sampling rate

    Returns:
        np.ndarray:
            Mono 16k float32 waveform in [-1, 1].
    """
    if isinstance(a, str):
        audio, sr = load_audio_any(a)
    elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
        audio, sr = a[0], int(a[1])
    else:
        raise TypeError(f"Unsupported audio input type: {type(a)}")

    audio = to_mono(np.asarray(audio))
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
    audio = float_range_normalize(audio)
    return audio


def normalize_audios(audios: Union[AudioLike, List[AudioLike]]) -> List[np.ndarray]:
    items = ensure_list(audios)
    return [normalize_audio_input(a) for a in items]


def chunk_list(xs: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    """
    Yield chunks of a list.

    Args:
        xs (List[Any]): Input list.
        chunk_size (int): Chunk size.

    Yields:
        List[Any]: Slices of xs.
    """
    if chunk_size <= 0:
        yield xs
        return
    for i in range(0, len(xs), chunk_size):
        yield xs[i : i + chunk_size]


@dataclass(frozen=True)
class AudioChunk:
    """
    One chunk cut from an original audio.

    Attributes:
        orig_index: Index of the original sample in the input batch.
        chunk_index: Index of this chunk within the original sample.
        wav: Mono float32 waveform.
        sr: Sampling rate.
        offset_sec: Start offset of this chunk in the original audio, in seconds.
    """
    orig_index: int
    chunk_index: int
    wav: np.ndarray
    sr: int
    offset_sec: float


def split_audio_into_chunks(
    wav: np.ndarray,
    sr: int,
    max_chunk_sec: float,
    search_expand_sec: float = 5.0,
    min_window_ms: float = 100.0,
) -> List[Tuple[np.ndarray, float]]:
    """
    Split a long audio into chunks close to max_chunk_sec, using a low-energy boundary.

    This implementation guarantees:
      - Concatenating all returned chunks reproduces the original audio exactly
        (total number of samples is identical, no overlaps, no gaps).

    Args:
        wav: Mono waveform float32.
        sr: Sampling rate.
        max_chunk_sec: Target max chunk duration in seconds.
        search_expand_sec: Boundary search half-window in seconds.
        min_window_ms: Sliding window in milliseconds for energy estimation.

    Returns:
        List[Tuple[np.ndarray, float]]: List of (chunk_wav, offset_sec).
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)

    total_len = int(wav.shape[0])
    total_sec = total_len / float(sr)
    if total_sec <= max_chunk_sec:
        return [(wav, 0.0)]

    max_len = int(max_chunk_sec * sr)
    expand = int(search_expand_sec * sr)
    win = max(4, int((min_window_ms / 1000.0) * sr))

    chunks: List[Tuple[np.ndarray, float]] = []

    start = 0
    offset_sec = 0.0

    while (total_len - start) > max_len:
        cut = start + max_len

        left = max(start, cut - expand)
        right = min(total_len, cut + expand)

        if right - left <= win:
            boundary = cut
        else:
            seg = wav[left:right]
            seg_abs = np.abs(seg)

            window_sums = np.convolve(seg_abs, np.ones(win, dtype=np.float32), mode="valid")

            min_pos = int(np.argmin(window_sums))

            wstart = min_pos
            wend = min_pos + win
            local = seg_abs[wstart:wend]
            inner = int(np.argmin(local))
            boundary = left + wstart + inner

        boundary = int(max(boundary, start + 1))
        boundary = int(min(boundary, total_len))

        chunk = wav[start:boundary]
        chunks.append((chunk, offset_sec))

        offset_sec += (boundary - start) / float(sr)
        start = boundary

    tail = wav[start:total_len]
    chunks.append((tail, offset_sec))

    # Pad too-short chunks to at least MIN_ASR_INPUT_SECONDS (zero-padding at tail)
    min_len = int(MIN_ASR_INPUT_SECONDS * sr)
    padded: List[Tuple[np.ndarray, float]] = []
    for c, off in chunks:
        if c.shape[0] < min_len:
            pad = min_len - int(c.shape[0])
            c = np.pad(c, (0, pad), mode="constant", constant_values=0.0).astype(np.float32)
        padded.append((c, off))
    chunks = padded

    return chunks


def detect_and_fix_repetitions(text, threshold=20):
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1

            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i+count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s
            
        i = 0
        result = []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                    
                pattern = s[i:i+k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx+k] != pattern:
                        valid = False
                        break
                
                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index+k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            
            if found:
                break
            else:
                result.append(s[i])
                i += 1

        if not found:
            result.append(s[i:])
        return ''.join(result)
    
    text_raw = text
    text = fix_char_repeats(text_raw, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text


def parse_asr_output(
    raw: str,
    user_language: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Parse Qwen3-ASR raw output into (language, text).

    Cases:
      - With tag: "language Chinese<asr_text>...."
      - With newlines: "language Chinese\\n...\\n<asr_text>...."
      - No tag: treat whole string as text.
      - "language None<asr_text>": treat as empty audio -> ("", "")

    If user_language is provided, language is forced to user_language and raw is treated as text-only
    (the model is expected to output plain transcription without metadata).

    Args:
        raw: Raw decoded string.
        user_language: Canonical language name if user forced language.

    Returns:
        Tuple[str, str]: (language, text)
    """
    if raw is None:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""

    s = detect_and_fix_repetitions(s)

    if user_language:
        # user explicitly forced language => model output is treated as pure text
        return user_language, s

    meta_part = s
    text_part = ""
    has_tag = _ASR_TEXT_TAG in s
    if has_tag:
        meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    else:
        # no tag => pure text
        return "", s.strip()

    meta_lower = meta_part.lower()

    # empty audio heuristic
    if "language none" in meta_lower:
        t = text_part.strip()
        if not t:
            return "", ""
        # if model still returned something, keep it but language unknown
        return "", t

    # extract "language xxx" from meta
    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX):].strip()
            if val:
                lang = normalize_language_name(val)
            break

    return lang, text_part.strip()


def merge_languages(langs: List[str]) -> str:
    """
    Merge per-chunk languages into a compact comma-separated string,
    keeping order and removing consecutive duplicates and empty entries.

    Example:
      ["Chinese", "English", "English"] -> "Chinese,English"

    Args:
        langs: List of canonical language names.

    Returns:
        str: Merged language string.
    """
    out: List[str] = []
    prev = None
    for x in langs:
        x = (x or "").strip()
        if not x:
            continue
        if x == prev:
            continue
        out.append(x)
        prev = x
    return ",".join(out)
