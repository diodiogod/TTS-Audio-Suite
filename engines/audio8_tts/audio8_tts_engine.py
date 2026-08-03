"""Wrapper for the official Audio8 TTS remote-code model."""

from __future__ import annotations

import os
import random
import warnings
from typing import Any, Dict, Optional, Tuple

import torch

from engines.audio8_tts.downloader import Audio8TTSDownloader
from engines.audio8_tts.progress import build_audio8_stopping_criteria
from utils.device import resolve_torch_device


class Audio8TTSEngine:
    """ComfyUI-friendly wrapper around the official Audio8 model."""

    SAMPLE_RATE = 44100

    def __init__(
        self,
        model_name: str = Audio8TTSDownloader.MODEL_NAME,
        device: str = "auto",
        dtype: str = "auto",
        model_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = resolve_torch_device(device)
        self.dtype_name = str(dtype or "auto").lower()
        self.dtype = self._resolve_dtype(self.dtype_name, self.device)
        self.model_dir = model_dir or Audio8TTSDownloader().resolve_model_path(
            model_name
        )
        self._processor = None
        self._model = None
        self._codec = None

    @staticmethod
    def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
        normalized = str(dtype or "auto").lower()
        valid = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if normalized not in {"auto", *valid}:
            raise ValueError(
                "Audio8 TTS dtype must be auto, bfloat16, float16, or float32"
            )
        if str(device).startswith("cpu"):
            return torch.float32
        if normalized in valid:
            return valid[normalized]
        if str(device).startswith("cuda") and torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability(torch.device(device))
            return torch.bfloat16 if major >= 8 else torch.float16
        if str(device).startswith("xpu"):
            return torch.bfloat16
        return torch.float16

    def _ensure_model_loaded(self):
        """Load the processor, model, and bundled codec strictly from local files."""
        if (
            self._processor is not None
            and self._model is not None
            and self._codec is not None
        ):
            return self

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Audio8 TTS local model directory does not exist: {self.model_dir}"
            )

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Audio8 TTS requires the suite's shared Transformers 4 runtime"
            ) from exc

        print(
            f"📦 Loading Audio8 TTS from {self.model_dir} "
            f"on {self.device} ({self.dtype})"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"`torch\.nn\.utils\.weight_norm` is deprecated in favor of "
                    r"`torch\.nn\.utils\.parametrizations\.weight_norm`\."
                ),
                category=FutureWarning,
            )
            self._processor = AutoProcessor.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                local_files_only=True,
            )
            self._model = AutoModel.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                local_files_only=True,
                dtype=self.dtype,
            )
        self._model = self._model.eval().to(self.device)

        # The official model stores the codec outside nn.Module registration.
        # Hold and move it explicitly so model offload/reload cannot strand it.
        self._codec = self._model.load_codec(
            device=self.device,
            dtype=self.dtype,
        )
        self._codec.eval()

        sample_rate = int(
            getattr(self._model.config, "codec_sample_rate", self.SAMPLE_RATE)
        )
        if sample_rate != self.SAMPLE_RATE:
            raise RuntimeError(
                f"Audio8 TTS checkpoint reports unexpected sample rate "
                f"{sample_rate}; expected {self.SAMPLE_RATE}"
            )
        print("✅ Audio8 TTS model and codec loaded")
        return self

    # Compatibility alias for callers using the shorter established name.
    _ensure_loaded = _ensure_model_loaded

    def _clear_kv_caches(self) -> None:
        """Drop every static slow/fast attention cache before device movement."""
        if self._model is None:
            return
        for collection_name in ("layers", "fast_layers"):
            for layer in getattr(self._model, collection_name, ()) or ():
                attention = getattr(layer, "attention", None)
                if attention is not None and hasattr(attention, "kv_cache"):
                    attention.kv_cache = None

    def to(self, device):
        """Move all weights for ComfyUI Clear VRAM and reload handling."""
        target = resolve_torch_device(
            str(device) if not isinstance(device, str) else device
        )
        self._clear_kv_caches()
        self.device = target
        if self._model is not None:
            self._model = self._model.to(target).eval()
        if self._codec is not None:
            codec_dtype = torch.float32 if str(target).startswith("cpu") else self.dtype
            self._codec = self._codec.to(
                device=target,
                dtype=codec_dtype,
            ).eval()
        return self

    def parameters(self):
        """Expose all weight tensors to ComfyUI memory accounting."""
        if self._model is not None:
            yield from self._model.parameters()
        if self._codec is not None:
            yield from self._codec.parameters()

    @staticmethod
    def _mono_audio(waveform: Any) -> torch.Tensor:
        audio = torch.as_tensor(waveform, dtype=torch.float32).detach().cpu()
        if audio.ndim == 3:
            if audio.shape[0] != 1:
                raise ValueError(
                    "Audio8 TTS reference audio must contain one batch item"
                )
            audio = audio[0]
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        if audio.ndim != 1 or audio.numel() == 0:
            raise ValueError(
                "Audio8 TTS reference audio must be non-empty mono or "
                "channels-first audio"
            )
        return audio.contiguous()

    def _normalize_reference_audio(
        self,
        reference_audio: Any,
    ) -> Tuple[Any, Optional[int]]:
        if isinstance(reference_audio, (str, os.PathLike)):
            return os.fspath(reference_audio), None
        if isinstance(reference_audio, dict):
            waveform = reference_audio.get(
                "waveform",
                reference_audio.get("array"),
            )
            sample_rate = reference_audio.get(
                "sample_rate",
                reference_audio.get("sampling_rate"),
            )
            if waveform is None or sample_rate is None:
                raise ValueError(
                    "Audio8 TTS in-memory reference audio requires waveform "
                    "and sample_rate"
                )
            return {
                "array": self._mono_audio(waveform),
                "sampling_rate": int(sample_rate),
            }, int(sample_rate)
        if isinstance(reference_audio, (tuple, list)) and len(reference_audio) == 2:
            waveform, sample_rate = reference_audio
            if not isinstance(sample_rate, (int, float)):
                raise ValueError(
                    "Audio8 TTS reference tuple must be (waveform, sample_rate)"
                )
            return {
                "array": self._mono_audio(waveform),
                "sampling_rate": int(sample_rate),
            }, int(sample_rate)
        raise TypeError(
            f"Unsupported Audio8 TTS reference audio type: {type(reference_audio)}"
        )

    def _prepare_inputs(
        self,
        text: str,
        reference_audio: Any,
        reference_text: Optional[str],
    ) -> Dict[str, torch.Tensor]:
        processor_kwargs: Dict[str, Any] = {
            "text": [text],
            "return_tensors": "pt",
        }
        if reference_audio is not None:
            normalized_audio, sample_rate = self._normalize_reference_audio(
                reference_audio
            )
            processor_kwargs.update(
                reference_audio=[normalized_audio],
                reference_text=[reference_text],
            )
            if sample_rate is not None:
                processor_kwargs["sampling_rate"] = [sample_rate]

        inputs = self._processor(**processor_kwargs)
        return {name: value.to(self.device) for name, value in inputs.items()}

    def _make_generator(self, seed: int):
        seed = int(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            return torch.Generator(device=torch.device(self.device)).manual_seed(seed)
        except Exception:
            return None

    def _generate_once(
        self,
        inputs: Dict[str, torch.Tensor],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        generator,
    ):
        criteria, tracker = build_audio8_stopping_criteria(max_new_tokens)
        try:
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                generator=generator,
                stopping_criteria=criteria,
                return_dict_in_generate=True,
            )
        except BaseException:
            tracker.abort()
            raise
        tracker.close()
        return output

    def generate(
        self,
        text: str,
        reference_audio: Any = None,
        reference_text: Optional[str] = None,
        *,
        max_new_tokens: int = 1024,
        retry_max_new_tokens: int = 2000,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        seed: int = 42,
    ) -> torch.Tensor:
        """Generate one utterance as a CPU float tensor ``[channels, samples]``."""
        text = str(text or "").strip()
        if not text:
            return torch.zeros(1, 0, dtype=torch.float32)

        reference_text = str(reference_text or "").strip()
        if reference_audio is not None and not reference_text:
            raise ValueError(
                "Audio8 TTS voice cloning requires the exact transcript of "
                "the reference audio"
            )
        if reference_audio is None and reference_text:
            raise ValueError(
                "Audio8 TTS reference_text requires matching reference audio"
            )

        max_new_tokens = int(max_new_tokens)
        retry_max_new_tokens = int(retry_max_new_tokens)
        temperature = float(temperature)
        top_p = float(top_p)
        top_k = int(top_k)
        if max_new_tokens < 1:
            raise ValueError("Audio8 TTS max_new_tokens must be positive")
        if retry_max_new_tokens < max_new_tokens:
            raise ValueError(
                "Audio8 TTS retry_max_new_tokens must be >= max_new_tokens"
            )
        if temperature <= 0:
            raise ValueError("Audio8 TTS temperature must be positive")
        if not 0 < top_p <= 1:
            raise ValueError("Audio8 TTS top_p must be in (0, 1]")
        if top_k < 1:
            raise ValueError("Audio8 TTS top_k must be at least 1")

        self._ensure_model_loaded()
        inputs = self._prepare_inputs(text, reference_audio, reference_text)
        generator = self._make_generator(seed)

        output = self._generate_once(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=bool(do_sample),
            generator=generator,
        )
        finished = bool(output.finished[0].item())

        if not finished and retry_max_new_tokens > max_new_tokens:
            print(
                "⚠️ Audio8 TTS reached max_new_tokens without EOS; "
                f"retrying with {retry_max_new_tokens}"
            )
            output = self._generate_once(
                inputs,
                max_new_tokens=retry_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=bool(do_sample),
                generator=generator,
            )
            finished = bool(output.finished[0].item())

        if not finished:
            print(
                "⚠️ Audio8 TTS generation ended without EOS; returning the "
                "valid decoded frames"
            )

        waveforms, waveform_lengths = self._model.decode_audio(output.codes)
        waveform_length = int(waveform_lengths[0].item())
        waveform = waveforms[0, :waveform_length].detach().float().cpu()
        return waveform.unsqueeze(0).contiguous()
