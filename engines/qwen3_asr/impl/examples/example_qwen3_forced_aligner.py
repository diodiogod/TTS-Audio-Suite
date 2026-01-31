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
"""
Examples for Qwen3ForcedAligner.

Covers:
  - single-sample inference (URL audio)
  - batch inference (URL audio)
  - base64 audio input (data:audio/wav;base64,...)
  - numpy waveform input as (np.ndarray, sr) using urllib request
"""

import base64
import io
import urllib.request
from typing import Tuple

import numpy as np
import soundfile as sf
import torch

from qwen_asr import Qwen3ForcedAligner


MODEL_PATH = "Qwen/Qwen3-ForcedAligner-0.6B"

URL_ZH = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"
URL_EN = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"

TEXT_ZH = "甚至出现交易几乎停滞的情况。"
TEXT_EN = (
    "Mm. Oh, yeah, yeah. He wasn't even that big when I started listening to him, "
    "but and his solo music didn't do overly well, but he did very well when he "
    "started writing for other people."
)


def _download_audio_bytes(url: str, timeout: int = 30) -> bytes:
    """
    Download audio bytes from a URL.

    Args:
        url (str): Audio URL.
        timeout (int): Request timeout in seconds.

    Returns:
        bytes: Raw response bytes.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _read_wav_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Decode audio bytes into waveform and sampling rate.

    Args:
        audio_bytes (bytes): Encoded audio bytes (wav/flac/ogg supported by libsndfile).

    Returns:
        Tuple[np.ndarray, int]: (waveform, sr). Waveform may be mono or multi-channel.
    """
    with io.BytesIO(audio_bytes) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)


def _to_data_url_base64(audio_bytes: bytes, mime: str = "audio/wav") -> str:
    """
    Convert audio bytes into a base64 data URL string.

    Args:
        audio_bytes (bytes): Encoded audio bytes.
        mime (str): MIME type.

    Returns:
        str: data:{mime};base64,... string.
    """
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _print_result(title: str, results) -> None:
    """
    Print a compact summary for debugging.

    Args:
        title (str): Case name.
        results (List[ForcedAlignResult]): Outputs from aligner.align(...).
    """
    print(f"\n===== {title} =====")
    for i, r in enumerate(results):
        n = len(r)
        head = r[0] if n > 0 else None
        tail = r[-1] if n > 0 else None
        print(f"[sample {i}] item={n}")
        if head is not None:
            print(f"  first: {head.text!r} {head.start_time}->{head.end_time} s")
            print(f"  last : {tail.text!r} {tail.start_time}->{tail.end_time} s")


def test_single_url(aligner: Qwen3ForcedAligner) -> None:
    """
    Single-sample alignment using HTTPS URL audio input.
    """
    results = aligner.align(
        audio=URL_ZH,
        text=TEXT_ZH,
        language="Chinese",
    )
    assert isinstance(results, list) and len(results) == 1
    assert len(results[0]) > 0
    _print_result("single-url", results)


def test_batch_url(aligner: Qwen3ForcedAligner) -> None:
    """
    Batch alignment using HTTPS URL audio input.
    """
    results = aligner.align(
        audio=[URL_ZH, URL_EN],
        text=[TEXT_ZH, TEXT_EN],
        language=["Chinese", "English"],
    )
    assert len(results) == 2
    assert len(results[0]) > 0 and len(results[1]) > 0
    _print_result("batch-url", results)


def test_base64_data_url(aligner: Qwen3ForcedAligner) -> None:
    """
    Single-sample alignment using base64 data URL audio input.
    """
    audio_bytes = _download_audio_bytes(URL_ZH)
    b64 = _to_data_url_base64(audio_bytes, mime="audio/wav")

    results = aligner.align(
        audio=b64,
        text=TEXT_ZH,
        language="Chinese",
    )
    assert len(results) == 1
    assert len(results[0]) > 0
    _print_result("single-base64-data-url", results)


def test_numpy_tuple_from_request(aligner: Qwen3ForcedAligner) -> None:
    """
    Single-sample alignment using (np.ndarray, sr) input where waveform is obtained by HTTP request.
    """
    audio_bytes = _download_audio_bytes(URL_EN)
    wav, sr = _read_wav_from_bytes(audio_bytes)

    results = aligner.align(
        audio=(wav, sr),
        text=TEXT_EN,
        language="English",
    )
    assert len(results) == 1
    assert len(results[0]) > 0
    _print_result("single-numpy-tuple-from-request", results)


def test_batch_mixed_inputs(aligner: Qwen3ForcedAligner) -> None:
    """
    Batch alignment mixing URL, base64, and (np.ndarray, sr) inputs.
    """
    zh_bytes = _download_audio_bytes(URL_ZH)
    en_bytes = _download_audio_bytes(URL_EN)

    zh_b64 = _to_data_url_base64(zh_bytes, mime="audio/wav")
    en_wav, en_sr = _read_wav_from_bytes(en_bytes)

    results = aligner.align(
        audio=[URL_ZH, zh_b64, (en_wav, en_sr)],
        text=[TEXT_ZH, TEXT_ZH, TEXT_EN],
        language=["Chinese", "Chinese", "English"],
    )
    assert len(results) == 3
    assert all(len(r) > 0 for r in results)
    _print_result("batch-mixed-inputs", results)


def main() -> None:
    aligner = Qwen3ForcedAligner.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        # attn_implementation="flash_attention_2",
    )

    test_single_url(aligner)
    test_batch_url(aligner)
    test_base64_data_url(aligner)
    test_numpy_tuple_from_request(aligner)
    test_batch_mixed_inputs(aligner)


if __name__ == "__main__":
    main()