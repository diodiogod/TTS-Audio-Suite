"""Adapter between the suite's TTS processors and an audio.cpp session."""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from utils.audio.audio_hash import generate_stable_audio_component
from utils.audio.cache import get_audio_cache
from utils.audio.processing import AudioProcessingUtils
from utils.voice.reference import effective_voice_audio


_CACHE_SAMPLE_RATES: Dict[str, int] = {}
_CACHE_SAMPLE_RATES_LOCK = threading.Lock()


def _get_session(config: Mapping[str, Any]):
    """Import lazily so the node can still be discovered before optional setup."""
    from utils.audio_cpp.session import get_audio_cpp_session

    return get_audio_cpp_session(dict(config))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class AudioCppEngineAdapter:
    """Build generic ``/v1/tasks/run`` requests and retain their real sample rate."""

    _COMMON_REQUEST_FIELDS = (
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_tokens",
        "max_steps",
        "num_inference_steps",
        "guidance_scale",
        "speaking_rate",
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self.audio_cache = get_audio_cache()
        self._last_sample_rate: Optional[int] = None
        self._reference_files: Dict[str, str] = {}
        self._reference_lock = threading.RLock()

    @property
    def sample_rate(self) -> Optional[int]:
        return self._last_sample_rate

    def update_config(self, new_config: Optional[Dict[str, Any]]) -> None:
        self.config = dict(new_config or {})

    @staticmethod
    def _reference_text(voice_ref: Any) -> str:
        if not isinstance(voice_ref, Mapping):
            return ""
        return str(
            voice_ref.get("reference_text")
            or voice_ref.get("prompt_text")
            or voice_ref.get("text")
            or ""
        ).strip()

    def _materialize_reference(self, voice_ref: Any) -> Tuple[Optional[str], str, str, Optional[str]]:
        """Return path, transcript, stable hash, and the path that must be removed."""
        reference_text = AudioCppEngineAdapter._reference_text(voice_ref)
        if not isinstance(voice_ref, Mapping):
            return None, reference_text, "default_voice", None

        audio = effective_voice_audio(voice_ref)
        if audio is None:
            return None, reference_text, "default_voice", None

        if isinstance(audio, (str, os.PathLike)):
            path = os.path.abspath(os.path.expanduser(os.fspath(audio)))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"audio.cpp reference audio not found: {path}")
            component = generate_stable_audio_component(audio_file_path=path)
            return path, reference_text, component, None

        if isinstance(audio, Mapping):
            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate")
            audio_dict = dict(audio)
        elif torch.is_tensor(audio):
            waveform = audio
            sample_rate = voice_ref.get("sample_rate")
            audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
        else:
            raise TypeError(f"Unsupported audio.cpp voice reference type: {type(audio).__name__}")

        if not torch.is_tensor(waveform):
            raise TypeError("audio.cpp reference audio must contain a waveform tensor")
        if sample_rate is None or int(sample_rate) <= 0:
            raise ValueError("audio.cpp reference audio must contain a positive sample_rate")

        audio_dict["sample_rate"] = int(sample_rate)
        component = generate_stable_audio_component(reference_audio=audio_dict)
        if component not in {"ref_audio_error", "ref_audio_error_not_tensor"}:
            with self._reference_lock:
                cached_path = self._reference_files.get(component)
                if cached_path and os.path.isfile(cached_path):
                    return cached_path, reference_text, component, None
                temp_path = os.path.abspath(
                    AudioProcessingUtils.save_audio_to_temp_file(waveform, int(sample_rate))
                )
                self._reference_files[component] = temp_path
                return temp_path, reference_text, component, None

        # Hash failures must not make unrelated references share one file.
        temp_path = os.path.abspath(
            AudioProcessingUtils.save_audio_to_temp_file(waveform, int(sample_rate))
        )
        return temp_path, reference_text, component, temp_path

    def close(self) -> None:
        with self._reference_lock:
            paths = list(self._reference_files.values())
            self._reference_files.clear()
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _advanced_options(self) -> Dict[str, Any]:
        value = self.config.get(
            "advanced_options",
            self.config.get("request_options", self.config.get("advanced_json", {})),
        )
        if value in (None, ""):
            return {}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid audio.cpp advanced JSON: {exc.msg}") from exc
        if not isinstance(value, Mapping):
            raise ValueError("audio.cpp advanced options must be a JSON object")
        return dict(value)

    def _resolved_task(self, session: Any) -> str:
        requested = str(self.config.get("task", self.config.get("requested_task", "auto"))).lower()
        for source in (session, getattr(session, "config", None)):
            if source is None:
                continue
            value = source.get("task") if isinstance(source, Mapping) else getattr(source, "task", None)
            if str(value).lower() in {"tts", "clon", "vdes"}:
                return str(value).lower()

        if requested in {"tts", "clon", "vdes"}:
            return requested
        if str(self.config.get("connection_mode", "auto")).lower() == "external_server":
            return "auto"
        try:
            from utils.audio_cpp.catalog import resolve_task

            return str(
                resolve_task(
                    self.config.get("family", ""),
                    self.config.get("package_id", ""),
                    requested="auto",
                )
            ).lower()
        except (ImportError, KeyError, TypeError, ValueError):
            return "tts"

    def _build_request(
        self,
        text: str,
        voice_path: Optional[str],
        reference_text: str,
        seed: int,
        advanced: Dict[str, Any],
        task: str,
    ) -> Dict[str, Any]:
        request: Dict[str, Any] = {"text": text, "seed": str(int(seed)), "options": advanced}
        del task  # The persistent session owns its one configured model/task.

        language = str(self.config.get("language", "")).strip()
        if language and language.lower() not in {"auto", "none"}:
            request["language"] = language
        voice_id = str(self.config.get("voice_id", self.config.get("voice", ""))).strip()
        if voice_id:
            request["voice_id"] = voice_id
        if voice_path:
            request["voice_ref"] = voice_path
        if reference_text:
            request["reference_text"] = reference_text
        instruct = str(self.config.get("instruct", "")).strip()
        if instruct:
            request["instruct"] = instruct

        for key in self._COMMON_REQUEST_FIELDS:
            value = self.config.get(key)
            if value is not None and value != "":
                request[key] = value
        return request

    def _cache_key(
        self,
        text: str,
        audio_component: str,
        reference_text: str,
        seed: int,
        task: str,
        advanced: Dict[str, Any],
        character_name: Optional[str],
        session: Any,
    ) -> str:
        session_config = getattr(session, "config", {})
        if not isinstance(session_config, Mapping):
            session_config = {}
        session_family = getattr(session, "family", None) or session_config.get(
            "family", self.config.get("family", "")
        )
        session_model_id = getattr(session, "model_id", None) or session_config.get(
            "model_id", self.config.get("model_id", "")
        )
        # Owned servers use a random loopback port on every restart; that port is
        # transport state, not model identity. External endpoints are stable and
        # must participate in the cache key.
        if bool(getattr(session, "owned", False)):
            session_endpoint = ""
        else:
            session_endpoint = getattr(session, "endpoint", None) or self.config.get(
                "server_url", self.config.get("external_server_url", "")
            )
        extra_identity = {
            "options": advanced,
            "speaking_rate": self.config.get("speaking_rate"),
            "connection_mode": self.config.get("connection_mode", "auto"),
            "server_url": session_endpoint,
            "binary_path": session_config.get("binary_path", self.config.get("binary_path", "")),
            "backend": session_config.get("backend", self.config.get("backend", "")),
            "device": session_config.get("device", self.config.get("device", "")),
            "load_options": session_config.get("load_options", self.config.get("load_options", {})),
            "session_options": session_config.get(
                "session_options", self.config.get("session_options", {})
            ),
            "default_request_options": session_config.get(
                "default_request_options", self.config.get("default_request_options", {})
            ),
        }
        return self.audio_cache.generate_cache_key(
            "audio_cpp",
            text=text,
            audio_component=audio_component,
            reference_text=reference_text,
            family=session_family,
            package_id=session_config.get("package_id", self.config.get("package_id", "")),
            model_path=session_config.get("model_path", self.config.get("model_path", "")),
            model_id=session_model_id,
            task=task,
            language=self.config.get("language", ""),
            voice_id=self.config.get("voice_id", self.config.get("voice", "")),
            instruct=self.config.get("instruct", ""),
            temperature=self.config.get("temperature"),
            top_p=self.config.get("top_p"),
            top_k=self.config.get("top_k"),
            repetition_penalty=self.config.get("repetition_penalty"),
            max_tokens=self.config.get("max_tokens"),
            max_steps=self.config.get("max_steps"),
            num_inference_steps=self.config.get("num_inference_steps"),
            guidance_scale=self.config.get("guidance_scale"),
            seed=int(seed),
            request_options=_canonical_json(extra_identity),
            character=character_name or "narrator",
        )

    @staticmethod
    def _normalize_result(result: Any) -> Tuple[torch.Tensor, int]:
        waveform = result.get("waveform") if isinstance(result, Mapping) else getattr(result, "waveform", None)
        sample_rate = result.get("sample_rate") if isinstance(result, Mapping) else getattr(result, "sample_rate", None)

        if waveform is None:
            named = result.get("named_audio", {}) if isinstance(result, Mapping) else getattr(result, "named_audio", {})
            values = list(named.values()) if isinstance(named, Mapping) else list(named or [])
            if len(values) == 1:
                item = values[0]
                waveform = item.get("waveform") if isinstance(item, Mapping) else getattr(item, "waveform", None)
                sample_rate = sample_rate or (item.get("sample_rate") if isinstance(item, Mapping) else getattr(item, "sample_rate", None))

        if waveform is None:
            raise RuntimeError("audio.cpp returned no primary audio output")
        if not torch.is_tensor(waveform):
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
        waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        if waveform.dim() != 2:
            raise ValueError(f"audio.cpp waveform must be [channels, samples], got {tuple(waveform.shape)}")
        if sample_rate is None or int(sample_rate) <= 0:
            raise ValueError("audio.cpp returned an invalid sample rate")
        return waveform.contiguous(), int(sample_rate)

    def generate_single(
        self,
        text: str,
        voice_ref: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        enable_audio_cache: bool = True,
        character_name: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int]:
        stripped = str(text or "").strip()
        if not stripped:
            if self._last_sample_rate is None:
                raise ValueError("audio.cpp cannot determine a sample rate for empty text")
            return torch.zeros(1, 0, dtype=torch.float32), self._last_sample_rate

        session = _get_session(self.config)
        task = self._resolved_task(session)
        advanced = self._advanced_options()
        cleanup_path: Optional[str] = None
        try:
            voice_path, reference_text, audio_component, cleanup_path = self._materialize_reference(voice_ref)
            cache_key = self._cache_key(
                stripped,
                audio_component,
                reference_text,
                seed,
                task,
                advanced,
                character_name,
                session,
            )
            if enable_audio_cache:
                cached = self.audio_cache.get_cached_audio(cache_key)
                with _CACHE_SAMPLE_RATES_LOCK:
                    cached_rate = _CACHE_SAMPLE_RATES.get(cache_key)
                if cached is not None and cached_rate is not None:
                    self._last_sample_rate = cached_rate
                    return cached[0].clone(), cached_rate

            request = self._build_request(stripped, voice_path, reference_text, seed, advanced, task)
            waveform, sample_rate = self._normalize_result(session.run(request))
            self._last_sample_rate = sample_rate
            if enable_audio_cache:
                duration = waveform.shape[-1] / sample_rate
                self.audio_cache.cache_audio(cache_key, waveform, duration)
                with _CACHE_SAMPLE_RATES_LOCK:
                    _CACHE_SAMPLE_RATES[cache_key] = sample_rate
            return waveform, sample_rate
        finally:
            if cleanup_path:
                try:
                    os.remove(cleanup_path)
                except FileNotFoundError:
                    pass


# Short alias for callers that do not use the older ``EngineAdapter`` suffix.
AudioCppAdapter = AudioCppEngineAdapter
