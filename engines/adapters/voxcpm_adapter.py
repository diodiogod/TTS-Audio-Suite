"""Adapter between TTS Audio Suite generation and official VoxCPM inference."""

from __future__ import annotations

import os
import re
import secrets
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

import torch

from utils.audio.audio_hash import generate_stable_audio_component
from utils.audio.cache import get_audio_cache
from utils.audio.processing import AudioProcessingUtils
from utils.models.factory_config import ModelLoadConfig
from utils.voice.reference import effective_voice_audio


class VoxCPMEngineAdapter:
    """Translate suite voice/config/cache data into official VoxCPM calls."""

    DEFAULT_MODEL = "VoxCPM2"
    SAMPLE_RATE = 48000
    SUPPORTED_ARCHITECTURES = frozenset({"voxcpm", "voxcpm2"})

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config.copy() if config else {}
        self.audio_cache = get_audio_cache()
        self._last_config: Optional[ModelLoadConfig] = None
        self._load_signature: Optional[Tuple[Any, ...]] = None
        self._model_spec: Optional[Dict[str, Any]] = None
        self._resolved_model_path: Optional[str] = None
        self._model_spec_selection: Optional[str] = None
        self.sample_rate = self.SAMPLE_RATE
        self.architecture = "voxcpm2"

    def update_config(self, new_config: Dict[str, Any]):
        previous_selection = self._model_selection()
        previous_lora = self._lora_adapter()
        self.config = new_config.copy() if new_config else {}
        if (
            self._model_selection() != previous_selection
            or self._lora_adapter() != previous_lora
        ):
            self._model_spec = None
            self._resolved_model_path = None
            self._model_spec_selection = None

    def _model_selection(self) -> str:
        return str(
            self.config.get("model_variant")
            or self.config.get("model_name")
            or self.DEFAULT_MODEL
        )

    def _lora_adapter(self) -> Optional[str]:
        value = str(self.config.get("lora_adapter") or "").strip()
        return os.path.abspath(os.path.expanduser(value)) if value else None

    def _lora_signature(self) -> Optional[Tuple[Any, ...]]:
        path = self._lora_adapter()
        if not path:
            return None
        values = [path]
        for filename in ("lora_config.json", "lora_weights.safetensors"):
            target = os.path.join(path, filename)
            try:
                stat = os.stat(target)
                values.extend((stat.st_size, stat.st_mtime_ns))
            except OSError:
                values.extend((None, None))
        return tuple(values)

    def _build_load_signature(self) -> Tuple[Any, ...]:
        return (
            self._model_selection(),
            self.config.get("device", "auto"),
            bool(self.config.get("optimize", False)),
            self.config.get("runtime_mode", "main_environment"),
            self.config.get("runtime_profile"),
            self._lora_signature(),
        )

    @classmethod
    def _validate_spec(cls, spec: Dict[str, Any]) -> Dict[str, Any]:
        architecture = str(spec.get("architecture") or "").lower()
        if architecture not in cls.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported VoxCPM checkpoint architecture '{architecture or 'missing'}'. "
                f"Expected one of: {', '.join(sorted(cls.SUPPORTED_ARCHITECTURES))}."
            )
        sample_rate = int(spec.get("sample_rate") or 0)
        if sample_rate <= 0:
            raise ValueError("VoxCPM checkpoint spec has no valid output sample rate.")

        validated = dict(spec)
        validated["architecture"] = architecture
        validated["sample_rate"] = sample_rate
        return validated

    def _resolve_model(self, selection: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        from engines.voxcpm.voxcpm_downloader import VoxCPMDownloader

        chosen = str(selection or self._model_selection())
        downloader = VoxCPMDownloader()
        resolved_path = downloader.resolve_model_path(chosen)
        spec = self._validate_spec(
            downloader.get_model_spec(chosen, resolved_path=resolved_path)
        )
        return resolved_path, spec

    def load_model(
        self,
        model_variant: Optional[str] = None,
        device: str = "auto",
        optimize: bool = False,
        runtime_mode: str = "main_environment",
        runtime_profile: Optional[str] = None,
    ):
        """Load one verified checkpoint through the unified model interface."""
        from utils.models.unified_model_interface import unified_model_interface

        selection = str(model_variant or self._model_selection())
        self.config.update(
            {
                "model_variant": selection,
                "device": device,
                "optimize": bool(optimize),
                "runtime_mode": runtime_mode,
                "runtime_profile": runtime_profile,
            }
        )
        resolved_path, spec = self._resolve_model(selection)
        canonical = str(spec.get("canonical") or selection)

        load_config = ModelLoadConfig(
            engine_name="voxcpm",
            model_type="tts",
            model_name=canonical,
            model_path=resolved_path,
            repo_id=spec.get("repo_id"),
            device=device,
            runtime_mode=runtime_mode,
            runtime_profile=runtime_profile,
            additional_params={
                "architecture": spec["architecture"],
                "sample_rate": spec["sample_rate"],
                "optimize": bool(optimize),
                "load_denoiser": False,
                "lora_adapter": self._lora_adapter(),
                "lora_signature": self._lora_signature(),
            },
        )
        self._last_config = load_config
        self._model_spec = spec
        self._resolved_model_path = resolved_path
        self._model_spec_selection = selection
        self.architecture = spec["architecture"]
        self.sample_rate = spec["sample_rate"]
        self.SAMPLE_RATE = self.sample_rate
        self._load_signature = (
            selection,
            device,
            bool(optimize),
            runtime_mode,
            runtime_profile,
            self._lora_signature(),
        )
        return unified_model_interface.load_model(load_config)

    def _ensure_model_loaded(self):
        signature = self._build_load_signature()
        if signature == self._load_signature and self._last_config is not None:
            return
        self.load_model(
            model_variant=self._model_selection(),
            device=self.config.get("device", "auto"),
            optimize=bool(self.config.get("optimize", False)),
            runtime_mode=self.config.get("runtime_mode", "main_environment"),
            runtime_profile=self.config.get("runtime_profile"),
        )

    def _get_engine(self):
        self._ensure_model_loaded()
        from utils.models.unified_model_interface import unified_model_interface

        return unified_model_interface.load_model(self._last_config)

    def get_model_spec(self) -> Dict[str, Any]:
        selection = self._model_selection()
        if (
            self._model_spec is None
            or self._resolved_model_path is None
            or self._model_spec_selection != selection
        ):
            resolved_path, spec = self._resolve_model(selection)
            self._model_spec = spec
            self._resolved_model_path = resolved_path
            self._model_spec_selection = selection
            self.architecture = spec["architecture"]
            self.sample_rate = spec["sample_rate"]
            self.SAMPLE_RATE = self.sample_rate
        return dict(self._model_spec)

    def get_sample_rate(self) -> int:
        return int(self.get_model_spec()["sample_rate"])

    def get_architecture(self) -> str:
        return str(self.get_model_spec()["architecture"])

    def supports_voice_cloning(self) -> bool:
        return True

    def supports_reference_only(self) -> bool:
        return self.get_architecture() == "voxcpm2"

    def supports_voice_instruction(self) -> bool:
        return self.get_architecture() == "voxcpm2"

    def supports_voice_design(self) -> bool:
        return self.get_architecture() == "voxcpm2"

    @staticmethod
    def _reference_text(voice_ref: Optional[Dict[str, Any]]) -> str:
        if not isinstance(voice_ref, dict):
            return ""
        return str(
            voice_ref.get("reference_text")
            or voice_ref.get("prompt_text")
            or voice_ref.get("text")
            or ""
        ).strip()

    def _extract_reference(
        self, voice_ref: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[torch.Tensor], Optional[int], str, str]:
        """Return waveform, rate, transcript, and a content-based cache identity."""
        transcript = self._reference_text(voice_ref)
        audio = effective_voice_audio(voice_ref)
        if audio is None or (isinstance(audio, str) and audio.strip().lower() in {"", "none"}):
            return None, None, "", "default_voice"

        if isinstance(audio, str):
            if not os.path.isfile(audio):
                raise FileNotFoundError(f"VoxCPM reference audio does not exist: {audio}")
            audio_component = generate_stable_audio_component(audio_file_path=audio)
            waveform, sample_rate = AudioProcessingUtils.safe_load_audio(audio)
            return waveform, int(sample_rate), transcript, audio_component

        if isinstance(audio, dict) and "waveform" in audio:
            waveform = audio["waveform"]
            if not torch.is_tensor(waveform):
                raise TypeError(
                    f"VoxCPM reference waveform must be a torch.Tensor, got {type(waveform)}"
                )
            sample_rate = int(audio.get("sample_rate") or self.get_sample_rate())
            return (
                waveform,
                sample_rate,
                transcript,
                generate_stable_audio_component(reference_audio=audio),
            )

        if torch.is_tensor(audio):
            sample_rate = int(
                (voice_ref or {}).get("sample_rate") or self.get_sample_rate()
            )
            audio_dict = {"waveform": audio, "sample_rate": sample_rate}
            return (
                audio,
                sample_rate,
                transcript,
                generate_stable_audio_component(reference_audio=audio_dict),
            )

        raise TypeError(f"Unsupported VoxCPM voice reference type: {type(audio)}")

    @staticmethod
    @contextmanager
    def _temporary_reference_wav(
        waveform: Optional[torch.Tensor], sample_rate: Optional[int]
    ) -> Iterator[Optional[str]]:
        """Materialize one mono WAV and remove it even if saving/inference fails."""
        if waveform is None:
            yield None
            return

        if not sample_rate or int(sample_rate) <= 0:
            raise ValueError("VoxCPM reference audio has no valid sample rate.")

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            audio = waveform.detach().float().cpu()
            if audio.dim() == 3:
                if audio.shape[0] != 1:
                    raise ValueError(
                        "VoxCPM reference audio must contain exactly one batch item."
                    )
                audio = audio.squeeze(0)
            if audio.dim() == 2:
                audio = audio.mean(dim=0)
            elif audio.dim() != 1:
                raise ValueError(
                    f"Unsupported VoxCPM reference shape: {tuple(audio.shape)}"
                )

            try:
                import soundfile as sf

                sf.write(
                    temp_path,
                    audio.contiguous().numpy(),
                    int(sample_rate),
                    format="WAV",
                    subtype="FLOAT",
                )
            except ImportError:
                # Keep the bridge usable in minimal environments without
                # soundfile/TorchCodec by writing a standard mono PCM16 WAV.
                import wave

                pcm16 = (
                    audio.clamp(-1.0, 1.0)
                    .mul(32767.0)
                    .round()
                    .to(torch.int16)
                    .contiguous()
                    .numpy()
                )
                with wave.open(temp_path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(int(sample_rate))
                    wav_file.writeframes(pcm16.tobytes())
            yield temp_path
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    @staticmethod
    def _sanitize_instruction(instruction: Any) -> str:
        cleaned = re.sub(r"[()（）]", "", str(instruction or ""))
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _effective_seed(seed: Optional[int]) -> int:
        requested = int(seed or 0)
        return requested if requested != 0 else secrets.randbits(32)

    def _generation_parameters(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        min_len = int(self.config.get("min_len", 2))
        max_len = int(self.config.get("max_len", 4096))
        steps = int(self.config.get("inference_timesteps", 10))
        cfg_value = float(self.config.get("cfg_value", 2.0))
        retry_max = int(self.config.get("retry_badcase_max_times", 3))
        retry_ratio = float(self.config.get("retry_badcase_ratio_threshold", 6.0))

        if min_len < 0 or max_len < min_len:
            raise ValueError("VoxCPM requires 0 <= min_len <= max_len.")
        if not 0.1 <= cfg_value <= 10.0:
            raise ValueError("VoxCPM cfg_value must be between 0.1 and 10.0.")
        if not 1 <= steps <= 100:
            raise ValueError(
                "VoxCPM inference_timesteps must be between 1 and 100."
            )
        if retry_max < 1 or retry_ratio <= 0:
            raise ValueError("VoxCPM bad-case retry settings must be positive.")
        if str(spec.get("canonical")) == "VoxCPM-0.5B":
            max_len = min(max_len, 4096)

        return {
            "cfg_value": cfg_value,
            "inference_timesteps": steps,
            "min_len": min_len,
            "max_len": max_len,
            "normalize": bool(self.config.get("normalize_text", False)),
            "retry_badcase": bool(self.config.get("retry_badcase", True)),
            "retry_badcase_max_times": retry_max,
            "retry_badcase_ratio_threshold": retry_ratio,
        }

    def generate_single(
        self,
        text: str,
        voice_ref: Optional[Dict[str, Any]],
        seed: int = 0,
        enable_audio_cache: bool = True,
        character_name: Optional[str] = None,
    ) -> torch.Tensor:
        target_text = str(text or "").strip()
        if not target_text:
            return torch.zeros(1, 0, dtype=torch.float32)

        spec = self.get_model_spec()
        params = self._generation_parameters(spec)
        waveform, reference_sample_rate, transcript, audio_component = (
            self._extract_reference(voice_ref)
        )
        instruction = self._sanitize_instruction(
            self.config.get("voice_instruction", "")
        )
        architecture = str(spec["architecture"])
        is_v2 = architecture == "voxcpm2"

        if instruction and not is_v2:
            raise ValueError(
                f"{spec.get('canonical', self._model_selection())} does not support "
                "voice instructions. Use VoxCPM2 or clear voice_instruction."
            )
        if self.config.get("model_role") == "voice_design" and not is_v2:
            raise ValueError("Voice design is supported only by VoxCPM2 checkpoints.")
        if waveform is not None and not is_v2 and not transcript:
            raise ValueError(
                "Legacy VoxCPM voice cloning requires the exact transcript of the "
                "reference audio in reference_text."
            )

        conditioned_text = f"({instruction}){target_text}" if instruction else target_text
        effective_seed = self._effective_seed(seed)
        model_variant = self._model_selection()

        cache_key = None
        if enable_audio_cache:
            cache_key = self.audio_cache.generate_cache_key(
                "voxcpm",
                text=target_text,
                audio_component=audio_component,
                reference_text=transcript,
                model_variant=model_variant,
                architecture=architecture,
                voice_instruction=instruction,
                cfg_value=params["cfg_value"],
                inference_timesteps=params["inference_timesteps"],
                min_len=params["min_len"],
                max_len=params["max_len"],
                normalize_text=params["normalize"],
                retry_badcase=params["retry_badcase"],
                retry_badcase_max_times=params["retry_badcase_max_times"],
                retry_badcase_ratio_threshold=params[
                    "retry_badcase_ratio_threshold"
                ],
                seed=effective_seed,
                device=self.config.get("device", "auto"),
                optimize=bool(self.config.get("optimize", False)),
                runtime_mode=self.config.get("runtime_mode", "main_environment"),
                runtime_profile=self.config.get("runtime_profile"),
                model_path=self._resolved_model_path,
                lora_adapter=self._lora_signature(),
                load_denoiser=False,
                sample_rate=spec["sample_rate"],
                character=character_name or "narrator",
            )
            cached = self.audio_cache.get_cached_audio(cache_key)
            if cached:
                print(
                    f"💾 Using cached VoxCPM audio for "
                    f"'{character_name or 'narrator'}': '{target_text[:30]}...'"
                )
                return cached[0]

        engine = self._get_engine()
        engine_architecture = str(engine.architecture).lower()
        engine_sample_rate = int(engine.sample_rate)
        if engine_architecture != architecture:
            raise RuntimeError(
                f"VoxCPM checkpoint architecture changed after load: spec={architecture}, "
                f"runtime={engine_architecture}."
            )
        if engine_sample_rate != int(spec["sample_rate"]):
            raise RuntimeError(
                f"VoxCPM sample-rate mismatch: spec={spec['sample_rate']}, "
                f"runtime={engine_sample_rate}."
            )

        with self._temporary_reference_wav(
            waveform, reference_sample_rate
        ) as reference_path:
            prompt_wav_path = None
            prompt_text = None
            reference_wav_path = None
            if reference_path:
                if is_v2:
                    reference_wav_path = reference_path
                    if transcript:
                        # VoxCPM2 ultimate mode uses the same audio for both caches.
                        prompt_wav_path = reference_path
                        prompt_text = transcript
                else:
                    prompt_wav_path = reference_path
                    prompt_text = transcript

            audio = engine.generate(
                text=conditioned_text,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                reference_wav_path=reference_wav_path,
                cfg_value=params["cfg_value"],
                inference_timesteps=params["inference_timesteps"],
                min_len=params["min_len"],
                max_len=params["max_len"],
                normalize=params["normalize"],
                retry_badcase=params["retry_badcase"],
                retry_badcase_max_times=params["retry_badcase_max_times"],
                retry_badcase_ratio_threshold=params[
                    "retry_badcase_ratio_threshold"
                ],
                seed=effective_seed,
            )

        if not torch.is_tensor(audio):
            audio = torch.as_tensor(audio, dtype=torch.float32)
        audio = audio.detach().float().cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        if audio.dim() != 2:
            raise ValueError(
                f"VoxCPM returned unsupported audio shape: {tuple(audio.shape)}"
            )

        self.sample_rate = engine_sample_rate
        self.SAMPLE_RATE = engine_sample_rate
        if enable_audio_cache and cache_key:
            duration = self.audio_cache._calculate_duration(
                audio, "voxcpm", sample_rate=self.sample_rate
            )
            self.audio_cache.cache_audio(cache_key, audio, duration)
        return audio
