"""Small stdlib HTTP client for the audio.cpp server API."""

from __future__ import annotations

import array
import base64
import binascii
import io
import json
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch


class AudioCppClientError(RuntimeError):
    """Base error for audio.cpp transport failures."""


class AudioCppConnectionError(AudioCppClientError):
    """The audio.cpp endpoint could not be reached."""


class AudioCppTimeoutError(AudioCppClientError):
    """The audio.cpp endpoint did not respond before the configured timeout."""


class AudioCppProtocolError(AudioCppClientError):
    """The server returned a response that does not match its API contract."""


class AudioCppHTTPError(AudioCppClientError):
    """Structured non-success response from audio.cpp."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        error_type: Optional[str] = None,
        path: str = "",
        response_body: str = "",
    ) -> None:
        self.status = int(status)
        self.error_type = error_type
        self.path = path
        self.response_body = response_body
        label = f"audio.cpp HTTP {self.status}"
        if error_type:
            label += f" ({error_type})"
        if path:
            label += f" for {path}"
        super().__init__(f"{label}: {message}")


@dataclass(frozen=True)
class AudioCppAudio:
    """Decoded PCM audio returned by audio.cpp."""

    waveform: torch.Tensor
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class AudioCppTaskResult:
    """Decoded result from ``POST /v1/tasks/run``."""

    waveform: Optional[torch.Tensor] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    named_audio: Dict[str, AudioCppAudio] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


def _decode_pcm_wav(wav_bytes: bytes, *, context: str = "audio") -> AudioCppAudio:
    if not isinstance(wav_bytes, (bytes, bytearray)) or not wav_bytes:
        raise AudioCppProtocolError(f"audio.cpp returned empty {context} WAV data")

    try:
        with wave.open(io.BytesIO(bytes(wav_bytes)), "rb") as wav_file:
            channels = int(wav_file.getnchannels())
            sample_rate = int(wav_file.getframerate())
            sample_width = int(wav_file.getsampwidth())
            frame_count = int(wav_file.getnframes())
            compression = wav_file.getcomptype()
            frames = wav_file.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise AudioCppProtocolError(f"audio.cpp returned an invalid {context} WAV: {exc}") from exc

    if compression != "NONE":
        raise AudioCppProtocolError(
            f"audio.cpp returned unsupported compressed {context} WAV data ({compression})"
        )
    if channels <= 0 or sample_rate <= 0:
        raise AudioCppProtocolError(
            f"audio.cpp returned invalid {context} WAV metadata "
            f"(sample_rate={sample_rate}, channels={channels})"
        )
    if sample_width not in (1, 2, 3, 4):
        raise AudioCppProtocolError(
            f"audio.cpp returned unsupported {context} WAV sample width: {sample_width} bytes"
        )

    expected_samples = frame_count * channels
    expected_bytes = expected_samples * sample_width
    if len(frames) != expected_bytes:
        raise AudioCppProtocolError(
            f"audio.cpp returned truncated {context} WAV data "
            f"({len(frames)} bytes, expected {expected_bytes})"
        )

    if sample_width == 1:
        values = torch.tensor(list(frames), dtype=torch.float32)
        values = (values - 128.0) / 128.0
    elif sample_width == 2:
        pcm = array.array("h")
        pcm.frombytes(frames)
        if sys.byteorder != "little":
            pcm.byteswap()
        values = torch.tensor(pcm, dtype=torch.float32) / 32768.0
    elif sample_width == 3:
        decoded = []
        for offset in range(0, len(frames), 3):
            sample = int.from_bytes(frames[offset : offset + 3], "little", signed=False)
            if sample & 0x800000:
                sample -= 0x1000000
            decoded.append(sample)
        values = torch.tensor(decoded, dtype=torch.float32) / 8388608.0
    else:
        pcm = array.array("i")
        pcm.frombytes(frames)
        if sys.byteorder != "little":
            pcm.byteswap()
        values = torch.tensor(pcm, dtype=torch.float32) / 2147483648.0

    if expected_samples == 0:
        waveform = torch.empty((channels, 0), dtype=torch.float32)
    else:
        waveform = values.reshape(frame_count, channels).transpose(0, 1).contiguous()
    return AudioCppAudio(
        waveform=waveform.cpu(),
        sample_rate=sample_rate,
        channels=channels,
    )


def _decode_base64_wav(value: Any, *, context: str) -> AudioCppAudio:
    if not isinstance(value, str) or not value:
        raise AudioCppProtocolError(f"audio.cpp response field '{context}' must be base64 WAV text")
    try:
        wav_bytes = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AudioCppProtocolError(
            f"audio.cpp response field '{context}' is not valid base64"
        ) from exc
    return _decode_pcm_wav(wav_bytes, context=context)


class AudioCppClient:
    """Synchronous audio.cpp HTTP client using only Python's standard library."""

    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout: float = 5.0,
        request_timeout: float = 600.0,
        max_response_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        parsed = urllib.parse.urlsplit(str(base_url).strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Invalid audio.cpp server URL: {base_url!r}")
        if parsed.query or parsed.fragment:
            raise ValueError("audio.cpp server URL must not contain a query or fragment")
        self.base_url = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
        )
        self.connect_timeout = max(0.01, float(connect_timeout))
        self.request_timeout = max(0.01, float(request_timeout))
        self.max_response_bytes = max(1, int(max_response_bytes))

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    @staticmethod
    def _error_details(body: bytes, fallback: str) -> tuple[str, Optional[str], str]:
        text = body.decode("utf-8", errors="replace")
        message = fallback
        error_type = None
        try:
            payload = json.loads(text)
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                message = str(error.get("message") or fallback)
                error_type = str(error.get("type")) if error.get("type") else None
            elif error:
                message = str(error)
        except (TypeError, ValueError):
            if text.strip():
                message = text.strip()[:1000]
        return message, error_type, text

    @staticmethod
    def _is_timeout_error(exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, socket.timeout)):
            return True
        if isinstance(exc, urllib.error.URLError):
            return isinstance(exc.reason, (TimeoutError, socket.timeout))
        return False

    def _request_bytes(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> tuple[bytes, Mapping[str, str]]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            try:
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"audio.cpp request is not JSON serializable: {exc}") from exc
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            self._url(path),
            data=body,
            headers=headers,
            method=method.upper(),
        )
        effective_timeout = self.request_timeout if timeout is None else max(0.01, float(timeout))
        try:
            with urllib.request.urlopen(request, timeout=effective_timeout) as response:
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        if int(content_length) > self.max_response_bytes:
                            raise AudioCppProtocolError(
                                f"audio.cpp response exceeds {self.max_response_bytes} bytes"
                            )
                    except ValueError:
                        pass
                response_body = response.read(self.max_response_bytes + 1)
                if len(response_body) > self.max_response_bytes:
                    raise AudioCppProtocolError(
                        f"audio.cpp response exceeds {self.max_response_bytes} bytes"
                    )
                return response_body, response.headers
        except urllib.error.HTTPError as exc:
            error_body = exc.read(65536)
            message, error_type, response_text = self._error_details(error_body, str(exc.reason))
            raise AudioCppHTTPError(
                exc.code,
                message,
                error_type=error_type,
                path=path,
                response_body=response_text,
            ) from exc
        except AudioCppClientError:
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            if self._is_timeout_error(exc):
                raise AudioCppTimeoutError(
                    f"audio.cpp request to {path} timed out after {effective_timeout:.2f}s"
                ) from exc
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            raise AudioCppConnectionError(
                f"Could not connect to audio.cpp at {self.base_url}: {reason}"
            ) from exc

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        response_body, _ = self._request_bytes(method, path, payload=payload, timeout=timeout)
        try:
            decoded = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise AudioCppProtocolError(
                f"audio.cpp returned invalid JSON for {path}: {exc}"
            ) from exc
        if not isinstance(decoded, dict):
            raise AudioCppProtocolError(
                f"audio.cpp returned {type(decoded).__name__}, expected an object for {path}"
            )
        return decoded

    def health(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._request_json(
            "GET",
            "/health",
            timeout=self.connect_timeout if timeout is None else timeout,
        )

    def models(self, *, timeout: Optional[float] = None) -> list[Dict[str, Any]]:
        payload = self._request_json("GET", "/v1/models", timeout=timeout)
        models = payload.get("data")
        if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
            raise AudioCppProtocolError("audio.cpp /v1/models response is missing a valid data list")
        return list(models)

    def voices(self, model_id: str, *, timeout: Optional[float] = None) -> list[str]:
        query = urllib.parse.urlencode({"model": str(model_id)})
        payload = self._request_json("GET", f"/v1/audio/voices?{query}", timeout=timeout)
        voices = payload.get("voices")
        if not isinstance(voices, list) or any(not isinstance(item, str) for item in voices):
            raise AudioCppProtocolError("audio.cpp voices response is missing a valid voices list")
        return list(voices)

    def features(self, *, timeout: Optional[float] = None) -> set[str]:
        """Read optional future feature metadata without assuming release-0.5.1 has it."""
        payload = self.health(timeout=timeout)
        raw = payload.get("features", payload.get("capabilities", []))
        if isinstance(raw, dict):
            return {str(name) for name, enabled in raw.items() if enabled}
        if isinstance(raw, list):
            return {str(item) for item in raw}
        return set()

    def supports_feature(self, name: str, *, timeout: Optional[float] = None) -> bool:
        return str(name) in self.features(timeout=timeout)

    def run_task(
        self,
        model_id: str,
        request: Mapping[str, Any],
        *,
        timeout: Optional[float] = None,
    ) -> AudioCppTaskResult:
        if not isinstance(request, Mapping):
            raise TypeError("audio.cpp task request must be a mapping")
        payload = self._request_json(
            "POST",
            "/v1/tasks/run",
            payload={"model": str(model_id), "request": dict(request)},
            timeout=timeout,
        )

        named_audio: Dict[str, AudioCppAudio] = {}
        raw_named = payload.get("named_audio_outputs", [])
        if raw_named is None:
            raw_named = []
        if not isinstance(raw_named, list):
            raise AudioCppProtocolError("audio.cpp named_audio_outputs must be a list")
        for index, item in enumerate(raw_named):
            if not isinstance(item, dict):
                raise AudioCppProtocolError(
                    f"audio.cpp named_audio_outputs[{index}] must be an object"
                )
            output_id = item.get("id")
            if not isinstance(output_id, str) or not output_id:
                raise AudioCppProtocolError(
                    f"audio.cpp named_audio_outputs[{index}] is missing a non-empty id"
                )
            if output_id in named_audio:
                raise AudioCppProtocolError(f"audio.cpp returned duplicate named audio id: {output_id}")
            decoded = _decode_base64_wav(
                item.get("audio"), context=f"named_audio_outputs[{index}].audio"
            )
            self._validate_declared_audio_metadata(item, decoded, f"named_audio_outputs[{index}]")
            named_audio[output_id] = decoded

        primary: Optional[AudioCppAudio] = None
        if "audio" in payload and payload.get("audio") is not None:
            primary = _decode_base64_wav(payload.get("audio"), context="audio")
            self._validate_declared_audio_metadata(payload, primary, "audio")
        elif len(named_audio) == 1:
            primary = next(iter(named_audio.values()))
        elif len(named_audio) > 1:
            raise AudioCppProtocolError(
                "audio.cpp task result contains multiple named audio outputs but no primary audio"
            )
        elif not named_audio and not any(
            key in payload for key in ("text", "segments", "speaker_turns", "words")
        ):
            raise AudioCppProtocolError("audio.cpp task result did not contain task output")

        return AudioCppTaskResult(
            waveform=primary.waveform if primary is not None else None,
            sample_rate=primary.sample_rate if primary is not None else None,
            channels=primary.channels if primary is not None else None,
            named_audio=named_audio,
            raw=payload,
        )

    @staticmethod
    def _validate_declared_audio_metadata(
        payload: Mapping[str, Any],
        decoded: AudioCppAudio,
        context: str,
    ) -> None:
        declared_rate = payload.get("sample_rate")
        if declared_rate is not None and int(declared_rate) != decoded.sample_rate:
            raise AudioCppProtocolError(
                f"audio.cpp {context} sample rate metadata ({declared_rate}) "
                f"does not match its WAV ({decoded.sample_rate})"
            )
        declared_channels = payload.get("channels")
        if declared_channels is not None and int(declared_channels) != decoded.channels:
            raise AudioCppProtocolError(
                f"audio.cpp {context} channel metadata ({declared_channels}) "
                f"does not match its WAV ({decoded.channels})"
            )


__all__ = [
    "AudioCppAudio",
    "AudioCppClient",
    "AudioCppClientError",
    "AudioCppConnectionError",
    "AudioCppHTTPError",
    "AudioCppProtocolError",
    "AudioCppTaskResult",
    "AudioCppTimeoutError",
]
