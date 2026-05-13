"""
MOSS-TTS engine wrapper.

Wraps the official OpenMOSS Hugging Face implementation while keeping model
loading under TTS Audio Suite's unified model manager.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from typing import Any, Dict, Optional, Tuple

import torch


class MossTTSEngine:
    """Official MOSS-TTS inference wrapper."""

    SAMPLE_RATE = 24000

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "MOSS-TTS-Local-Transformer": {
            "repo_id": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
            "architecture": "local",
            "display": "MOSS-TTS Local 1.7B",
            "audio_temperature": 1.0,
            "audio_top_p": 0.95,
            "audio_top_k": 50,
            "audio_repetition_penalty": 1.1,
            "max_new_tokens": 4096,
        },
        "MOSS-TTS": {
            "repo_id": "OpenMOSS-Team/MOSS-TTS",
            "architecture": "delay",
            "display": "MOSS-TTS Delay 8B",
            "audio_temperature": 1.7,
            "audio_top_p": 0.8,
            "audio_top_k": 25,
            "audio_repetition_penalty": 1.0,
            "max_new_tokens": 4096,
        },
    }

    SUPPORTED_LANGUAGES = {
        "auto": None,
        "zh": "zh",
        "en": "en",
        "de": "de",
        "es": "es",
        "fr": "fr",
        "ja": "ja",
        "it": "it",
        "hu": "hu",
        "ko": "ko",
        "ru": "ru",
        "fa": "fa",
        "ar": "ar",
        "pl": "pl",
        "pt": "pt",
        "cs": "cs",
        "da": "da",
        "sv": "sv",
        "el": "el",
        "tr": "tr",
    }

    def __init__(
        self,
        model_path: str,
        codec_path: str,
        model_variant: str = "MOSS-TTS-Local-Transformer",
        device: str = "auto",
        dtype: str = "auto",
        attn_implementation: str = "auto",
    ):
        self.model_path = model_path
        self.codec_path = codec_path
        self.model_variant = model_variant
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.attn_implementation = self._resolve_attn_implementation(attn_implementation)
        self._model = None
        self._processor = None

        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _resolve_dtype(self, dtype: str):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype in dtype_map:
            return dtype_map[dtype]
        if str(self.device).startswith("cuda"):
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _resolve_attn_implementation(self, requested: str) -> str:
        normalized = (requested or "auto").strip().lower()
        if normalized not in {"", "auto"}:
            return normalized

        if (
            str(self.device).startswith("cuda")
            and self.dtype in {torch.float16, torch.bfloat16}
            and importlib.util.find_spec("flash_attn") is not None
        ):
            try:
                importlib.metadata.version("flash_attn")
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    return "flash_attention_2"
            except Exception:
                pass
        return "sdpa" if str(self.device).startswith("cuda") else "eager"

    def _check_transformers_version(self) -> None:
        try:
            import transformers
            from packaging import version
        except Exception as e:
            raise ImportError(f"MOSS-TTS requires transformers to load official remote code: {e}") from e

        current = version.parse(transformers.__version__)
        minimum = version.parse("4.57.0")
        if current < minimum:
            raise RuntimeError(
                f"MOSS-TTS requires transformers>=4.57.0 for the official remote code; "
                f"found {transformers.__version__}. Do not upgrade to Transformers 5 unless "
                "you have validated Qwen3-TTS compatibility in this suite."
            )

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        self._check_transformers_version()
        from transformers import AutoModel, AutoProcessor

        print(f"🔄 Loading MOSS-TTS: {self.model_variant}")
        print(f"   Model: {self.model_path}")
        print(f"   Codec: {self.codec_path}")
        print(f"   Device: {self.device} | Dtype: {self.dtype} | Attention: {self.attn_implementation}")

        processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            codec_path=self.codec_path,
        )
        if hasattr(processor, "audio_tokenizer") and processor.audio_tokenizer is not None:
            processor.audio_tokenizer = processor.audio_tokenizer.to(self.device)
            if hasattr(processor.audio_tokenizer, "eval"):
                processor.audio_tokenizer.eval()

        model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.dtype,
        ).to(self.device)
        model.eval()

        self._processor = processor
        self._model = model
        print(f"✅ MOSS-TTS model loaded: {self.model_variant}")

    def parameters(self):
        """Expose parameters for ComfyUI memory estimation."""
        if self._model is not None and hasattr(self._model, "parameters"):
            yield from self._model.parameters()
        audio_tokenizer = getattr(self._processor, "audio_tokenizer", None)
        if audio_tokenizer is not None and hasattr(audio_tokenizer, "parameters"):
            yield from audio_tokenizer.parameters()

    def to(self, device):
        """Move loaded MOSS components for ComfyUI Clear VRAM integration."""
        self.device = str(device)
        if self._model is not None and hasattr(self._model, "to"):
            self._model = self._model.to(device)
        audio_tokenizer = getattr(self._processor, "audio_tokenizer", None)
        if audio_tokenizer is not None and hasattr(audio_tokenizer, "to"):
            self._processor.audio_tokenizer = audio_tokenizer.to(device)
        return self

    def _ensure_runtime_device(self):
        self._ensure_model_loaded()
        target_device = "cuda" if torch.cuda.is_available() and self.device != "cpu" else self.device
        try:
            first_param = next(self._model.parameters())
        except StopIteration:
            return
        if str(first_param.device) != str(target_device):
            print(f"🔄 MOSS-TTS: moving model from {first_param.device} to {target_device}")
            self.to(target_device)

    def _waveform_to_codes(self, waveform, sample_rate: int):
        if not torch.is_tensor(waveform):
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
        waveform = waveform.detach().float().cpu()
        if waveform.dim() == 3:
            waveform = waveform[0]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(f"Unsupported MOSS-TTS reference waveform shape: {tuple(waveform.shape)}")
        return self._processor.encode_audios_from_wav(
            wav_list=[waveform],
            sampling_rate=int(sample_rate),
        )[0]

    def _prepare_reference(self, reference_audio, reference_sample_rate: Optional[int] = None):
        if reference_audio is None:
            return None
        if isinstance(reference_audio, (str, os.PathLike)):
            return str(reference_audio)
        if isinstance(reference_audio, dict) and "waveform" in reference_audio:
            return self._waveform_to_codes(
                reference_audio["waveform"],
                int(reference_audio.get("sample_rate", reference_sample_rate or self.SAMPLE_RATE)),
            )
        if isinstance(reference_audio, tuple) and len(reference_audio) == 2:
            waveform, sample_rate = reference_audio
            return self._waveform_to_codes(waveform, int(sample_rate))
        if torch.is_tensor(reference_audio):
            return self._waveform_to_codes(reference_audio, int(reference_sample_rate or self.SAMPLE_RATE))
        raise TypeError(f"Unsupported MOSS-TTS reference audio type: {type(reference_audio).__name__}")

    def _run_generation(
        self,
        input_ids,
        attention_mask,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        audio_repetition_penalty: float,
        max_new_tokens: int,
        n_vq_for_inference: Optional[int] = None,
    ):
        architecture = self.MODEL_VARIANTS.get(self.model_variant, {}).get("architecture", "local")
        if architecture == "local":
            from transformers import GenerationConfig

            class _MossLocalGenerationConfig(GenerationConfig):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.layers = kwargs.get("layers", [{} for _ in range(32)])
                    self.do_samples = kwargs.get("do_samples", None)
                    self.n_vq_for_inference = kwargs.get("n_vq_for_inference", 32)

            gen_config = _MossLocalGenerationConfig.from_pretrained(self.model_path)
            channels = int(getattr(self._model, "channels", 33))
            gen_config.pad_token_id = self._processor.tokenizer.pad_token_id
            gen_config.eos_token_id = 151653
            gen_config.max_new_tokens = int(max_new_tokens)
            gen_config.use_cache = True
            gen_config.do_sample = False
            gen_config.n_vq_for_inference = int(n_vq_for_inference or max(1, channels - 1))
            gen_config.do_samples = [True] * channels
            gen_config.layers = [
                {
                    "repetition_penalty": 1.0,
                    "temperature": 1.5,
                    "top_p": 1.0,
                    "top_k": int(audio_top_k),
                }
            ] + [
                {
                    "repetition_penalty": float(audio_repetition_penalty),
                    "temperature": float(audio_temperature),
                    "top_p": float(audio_top_p),
                    "top_k": int(audio_top_k),
                }
            ] * (channels - 1)
            return self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )

        return self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            audio_temperature=float(audio_temperature),
            audio_top_p=float(audio_top_p),
            audio_top_k=int(audio_top_k),
            audio_repetition_penalty=float(audio_repetition_penalty),
        )

    def generate(
        self,
        text: str,
        reference_audio=None,
        reference_sample_rate: Optional[int] = None,
        language: Optional[str] = None,
        duration_tokens: Optional[int] = None,
        seed: int = 0,
        audio_temperature: float = 1.0,
        audio_top_p: float = 0.95,
        audio_top_k: int = 50,
        audio_repetition_penalty: float = 1.1,
        max_new_tokens: int = 4096,
        n_vq_for_inference: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Generate a single MOSS-TTS utterance."""
        if not str(text or "").strip():
            raise ValueError("MOSS-TTS requires non-empty text")

        self._ensure_runtime_device()

        if seed and seed > 0:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        reference_item = self._prepare_reference(reference_audio, reference_sample_rate)
        references = [reference_item] if reference_item is not None else None
        language_value = self._normalize_language(language)

        user_message = self._processor.build_user_message(
            text=str(text),
            reference=references,
            tokens=int(duration_tokens) if duration_tokens else None,
            language=language_value,
        )
        batch = self._processor([[user_message]], mode="generation")
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self._run_generation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_temperature=audio_temperature,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
                max_new_tokens=max_new_tokens,
                n_vq_for_inference=n_vq_for_inference,
            )

        messages = self._processor.decode(outputs)
        if not messages or messages[0] is None or not messages[0].audio_codes_list:
            raise RuntimeError("MOSS-TTS returned no decodable audio")

        audio = messages[0].audio_codes_list[0]
        if not torch.is_tensor(audio):
            audio = torch.as_tensor(audio, dtype=torch.float32)
        audio = audio.detach().float().cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3:
            audio = audio.squeeze(0)
        return audio, int(getattr(self._processor.model_config, "sampling_rate", self.SAMPLE_RATE))

    def _normalize_language(self, language: Optional[str]) -> Optional[str]:
        if language is None:
            return None
        value = str(language).strip()
        if not value or value.lower() == "auto":
            return None
        return self.SUPPORTED_LANGUAGES.get(value.lower(), value.lower())
