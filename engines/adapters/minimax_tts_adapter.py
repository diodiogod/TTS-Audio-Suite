"""
MiniMax Cloud TTS Engine Adapter

Provides cloud-based text-to-speech via the MiniMax T2A v2 API.
Unlike local TTS engines, this adapter requires a MINIMAX_API_KEY
environment variable and makes HTTP requests to the MiniMax API.

Supported models: speech-2.8-hd, speech-2.8-turbo
Supported voices: 12 built-in English/multilingual voices
"""

import io
import os
import struct
import torch
from typing import Dict, Any, Optional, Tuple

from utils.audio.cache import get_audio_cache
from utils.text.chunking import ImprovedChatterBoxChunker
from utils.audio.chunk_timing import ChunkTimingHelper


# Verified voice IDs grouped by category
MINIMAX_VOICES = {
    "English_Graceful_Lady": "Graceful Lady (English)",
    "English_Insightful_Speaker": "Insightful Speaker (English)",
    "English_radiant_girl": "Radiant Girl (English)",
    "English_Persuasive_Man": "Persuasive Man (English)",
    "English_Lucky_Robot": "Lucky Robot (English)",
    "Wise_Woman": "Wise Woman",
    "cute_boy": "Cute Boy",
    "lovely_girl": "Lovely Girl",
    "Friendly_Person": "Friendly Person",
    "Inspirational_girl": "Inspirational Girl",
    "Deep_Voice_Man": "Deep Voice Man",
    "sweet_girl": "Sweet Girl",
}

MINIMAX_VOICE_IDS = list(MINIMAX_VOICES.keys())

MINIMAX_MODELS = ["speech-2.8-hd", "speech-2.8-turbo"]

# MiniMax TTS outputs MP3 at 32000 Hz by default
MINIMAX_TTS_SAMPLE_RATE = 32000


class MiniMaxTTSAdapter:
    """Adapter for MiniMax Cloud TTS API (T2A v2)."""

    SAMPLE_RATE = MINIMAX_TTS_SAMPLE_RATE

    def __init__(self, config: Dict[str, Any]):
        self.config = config.copy() if config else {}
        self.audio_cache = get_audio_cache()

    def update_config(self, new_config: Dict[str, Any]):
        """Update adapter configuration."""
        self.config = new_config.copy() if new_config else {}

    def _get_api_key(self) -> str:
        """Get MiniMax API key from config or environment."""
        api_key = self.config.get("api_key") or os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            raise ValueError(
                "MiniMax API key not found. Set the MINIMAX_API_KEY environment variable "
                "or provide it in the engine configuration."
            )
        return api_key

    def _call_tts_api(self, text: str, voice_id: str, model: str, speed: float) -> bytes:
        """
        Call MiniMax T2A v2 API and return raw audio bytes (MP3).

        Args:
            text: Text to synthesize
            voice_id: MiniMax voice ID
            model: Model name (speech-2.8-hd or speech-2.8-turbo)
            speed: Speech speed multiplier (0.5-2.0)

        Returns:
            Raw MP3 audio bytes
        """
        import urllib.request
        import json

        api_key = self._get_api_key()
        url = "https://api.minimax.io/v1/t2a_v2"

        payload = {
            "model": model,
            "text": text,
            "voice_setting": {
                "voice_id": voice_id,
                "speed": max(0.5, min(2.0, speed)),
            },
            "audio_setting": {
                "format": "mp3",
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
        except Exception as e:
            raise RuntimeError(f"MiniMax TTS API request failed: {e}")

        result = json.loads(body)

        # Check for API errors
        if result.get("base_resp", {}).get("status_code", 0) != 0:
            status_msg = result.get("base_resp", {}).get("status_msg", "Unknown error")
            raise RuntimeError(f"MiniMax TTS API error: {status_msg}")

        # Extract hex-encoded audio from response
        audio_hex = result.get("data", {}).get("audio", "")
        if not audio_hex:
            raise RuntimeError("MiniMax TTS API returned empty audio data")

        return bytes.fromhex(audio_hex)

    @staticmethod
    def _decode_mp3_to_tensor(mp3_bytes: bytes) -> Tuple[torch.Tensor, int]:
        """
        Decode MP3 bytes to a torch tensor using torchaudio.

        Returns:
            Tuple of (audio_tensor [1D float32], sample_rate)
        """
        import torchaudio

        # Write MP3 to a temporary buffer and decode
        buffer = io.BytesIO(mp3_bytes)
        waveform, sample_rate = torchaudio.load(buffer, format="mp3")

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Flatten to 1D
        audio = waveform.squeeze(0)

        # Normalize to [-1, 1]
        max_val = audio.abs().max()
        if max_val > 0 and max_val > 1.0:
            audio = audio / max_val

        return audio, sample_rate

    def generate_single(
        self, text: str, enable_audio_cache: bool = True, **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate audio for a single text segment.

        Args:
            text: Text to synthesize
            enable_audio_cache: Whether to use caching
            **kwargs: Additional parameters (voice_id, model, speed)

        Returns:
            Tuple of (audio_tensor [1D float32], sample_rate)
        """
        voice_id = kwargs.get("voice_id") or self.config.get("voice_id", "English_Graceful_Lady")
        model = kwargs.get("model") or self.config.get("model", "speech-2.8-hd")
        speed = float(kwargs.get("speed") or self.config.get("speed", 1.0))

        # Check cache
        cache_key = None
        if enable_audio_cache:
            cache_key = self.audio_cache.generate_cache_key(
                "minimax_tts",
                text=text,
                voice_id=voice_id,
                model=model,
                speed=speed,
            )
            cached_audio = self.audio_cache.get_cached_audio(cache_key)
            if cached_audio:
                print(f"\U0001f4be Using cached MiniMax TTS audio: '{text[:30]}...'")
                return cached_audio[0], self.SAMPLE_RATE

        # Call API
        mp3_bytes = self._call_tts_api(text, voice_id, model, speed)

        # Decode
        audio_tensor, sample_rate = self._decode_mp3_to_tensor(mp3_bytes)

        # Cache
        if enable_audio_cache and cache_key:
            duration = audio_tensor.shape[-1] / float(sample_rate)
            self.audio_cache.cache_audio(cache_key, audio_tensor, duration)

        self.SAMPLE_RATE = sample_rate
        return audio_tensor, sample_rate

    def process_text(
        self,
        text: str,
        enable_chunking: bool = True,
        max_chars_per_chunk: int = 400,
        chunk_combination_method: str = "auto",
        silence_between_chunks_ms: int = 100,
        enable_audio_cache: bool = True,
        return_info: bool = False,
    ):
        """
        Process text with optional chunking and combine audio segments.

        Args:
            text: Text to synthesize
            enable_chunking: Whether to split text into chunks
            max_chars_per_chunk: Maximum characters per chunk
            chunk_combination_method: How to combine chunks
            silence_between_chunks_ms: Silence padding between chunks
            enable_audio_cache: Whether to cache audio
            return_info: Whether to return chunk timing info

        Returns:
            audio_tensor or (audio_tensor, chunk_info) if return_info=True
        """
        if enable_chunking:
            max_chars = ImprovedChatterBoxChunker.validate_chunking_params(max_chars_per_chunk)
            text_chunks = ImprovedChatterBoxChunker.split_into_chunks(text, max_chars=max_chars)
        else:
            text_chunks = [text]

        audio_segments = []
        sample_rate = self.SAMPLE_RATE

        for chunk in text_chunks:
            audio_tensor, sr = self.generate_single(chunk, enable_audio_cache=enable_audio_cache)
            sample_rate = sr
            audio_segments.append(audio_tensor)

        combined_audio, chunk_info = ChunkTimingHelper.combine_audio_with_timing(
            audio_segments=audio_segments,
            combination_method=chunk_combination_method,
            silence_ms=silence_between_chunks_ms,
            crossfade_duration=0.1,
            sample_rate=sample_rate,
            text_length=len(text),
            original_text=text,
            text_chunks=text_chunks,
        )

        if return_info:
            return combined_audio, chunk_info
        return combined_audio
