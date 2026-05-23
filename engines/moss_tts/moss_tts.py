"""
MOSS-TTS engine wrapper.

Wraps the official OpenMOSS Hugging Face implementation while keeping model
loading under TTS Audio Suite's unified model manager.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import importlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


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
        "MOSS-TTSD-v1.0": {
            "repo_id": "OpenMOSS-Team/MOSS-TTSD-v1.0",
            "architecture": "ttsd",
            "display": "MOSS-TTSD v1.0 Dialogue 8B",
            "audio_temperature": 1.1,
            "audio_top_p": 0.9,
            "audio_top_k": 50,
            "audio_repetition_penalty": 1.1,
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
        lora_adapter: Optional[str] = None,
    ):
        self.model_path = model_path
        self.codec_path = codec_path
        self.model_variant = model_variant
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.attn_implementation = self._resolve_attn_implementation(attn_implementation)
        self.lora_adapter = str(lora_adapter or "").strip() or None
        self._model = None
        self._processor = None
        self._audio_tokenizer_device = "cpu"

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

    @staticmethod
    def _normalize_model_identity(value: Optional[str]) -> str:
        text = str(value or "").strip().replace("\\", "/").rstrip("/")
        if not text:
            return ""
        if "://" in text:
            text = text.split("://", 1)[1]
        return text.lower()

    def _validate_lora_compatibility(self) -> None:
        if not self.lora_adapter:
            return

        adapter_path = str(self.lora_adapter).strip()
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.isfile(config_path):
            raise RuntimeError(
                f"Invalid MOSS LoRA adapter: missing adapter_config.json in '{adapter_path}'. "
                "Expected a PEFT adapter folder, not a bare weights file."
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read MOSS LoRA adapter config '{config_path}': {e}"
            ) from e

        expected_repo_id = str(
            self.MODEL_VARIANTS.get(self.model_variant, {}).get("repo_id", "")
        ).strip()
        configured_base = str(adapter_config.get("base_model_name_or_path", "")).strip()

        if not configured_base or not expected_repo_id:
            return

        normalized_expected = self._normalize_model_identity(expected_repo_id)
        normalized_configured = self._normalize_model_identity(configured_base)
        configured_name = os.path.basename(normalized_configured)
        expected_name = os.path.basename(normalized_expected)

        if normalized_configured == normalized_expected:
            return
        if configured_name and configured_name == expected_name:
            return

        raise RuntimeError(
            "MOSS LoRA/base model mismatch. "
            f"Selected model '{self.model_variant}' expects base '{expected_repo_id}', "
            f"but adapter '{os.path.basename(adapter_path)}' was saved for '{configured_base}'. "
            "Use the matching MOSS variant or a LoRA trained for this model."
        )

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        self._check_transformers_version()
        self._validate_lora_compatibility()
        from transformers import AutoTokenizer
        from engines.moss_tts.impl.audio_tokenizer.modeling_moss_audio_tokenizer import (
            MossAudioTokenizerModel,
        )

        print(f"🔄 Loading MOSS-TTS: {self.model_variant}")
        print(f"   Model: {self.model_path}")
        print(f"   Codec: {self.codec_path}")
        print(f"   Device: {self.device} | Dtype: {self.dtype} | Attention: {self.attn_implementation}")
        if self.lora_adapter:
            print(f"   LoRA: {self.lora_adapter}")

        architecture = self.MODEL_VARIANTS.get(self.model_variant, {}).get("architecture", "local")
        if architecture == "local":
            package_base = "engines.moss_tts.impl.local_transformer"
        elif architecture == "ttsd":
            package_base = "engines.moss_tts.impl.ttsd"
        else:
            package_base = "engines.moss_tts.impl.delay"

        # TTS Audio Suite Patch: use bundled MOSS classes instead of trust_remote_code.
        config_module = importlib.import_module(f"{package_base}.configuration_moss_tts")
        processing_module = importlib.import_module(f"{package_base}.processing_moss_tts")
        modeling_module = importlib.import_module(f"{package_base}.modeling_moss_tts")

        MossConfig = getattr(config_module, "MossTTSDelayConfig")
        MossProcessor = getattr(processing_module, "MossTTSDelayProcessor")
        MossModel = getattr(modeling_module, "MossTTSDelayModel")

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        audio_tokenizer = MossAudioTokenizerModel.from_pretrained(self.codec_path)
        processor = MossProcessor(
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer,
            model_config=MossConfig.from_pretrained(self.model_path),
        )
        # TTS Audio Suite Patch: guard against partial configs where pad_token_id is unset.
        model_cfg = getattr(processor, "model_config", None)
        tokenizer_pad_id = getattr(tokenizer, "pad_token_id", None)
        tokenizer_eos_id = getattr(tokenizer, "eos_token_id", None)
        if model_cfg is not None and getattr(model_cfg, "pad_token_id", None) is None:
            if tokenizer_pad_id is not None:
                model_cfg.pad_token_id = tokenizer_pad_id
            elif tokenizer_eos_id is not None:
                model_cfg.pad_token_id = tokenizer_eos_id
            else:
                model_cfg.pad_token_id = 0
        # TTS Audio Suite Patch: transformers cache utils expects num_hidden_layers on decoder config.
        if model_cfg is not None and getattr(model_cfg, "num_hidden_layers", None) is None:
            language_cfg = getattr(model_cfg, "language_config", None)
            language_layers = getattr(language_cfg, "num_hidden_layers", None)
            local_layers = getattr(model_cfg, "local_num_layers", None)
            if language_layers is not None:
                model_cfg.num_hidden_layers = int(language_layers)
            elif local_layers is not None:
                model_cfg.num_hidden_layers = int(local_layers)
            else:
                model_cfg.num_hidden_layers = 1
        if hasattr(processor, "audio_tokenizer") and processor.audio_tokenizer is not None:
            # TTS Audio Suite Patch: keep the shared MOSS audio tokenizer on CPU by default.
            # The tokenizer itself is very large, and eager GPU placement makes the 8B
            # models fail to load on 24GB cards before generation even begins.
            processor.audio_tokenizer = processor.audio_tokenizer.to(self._audio_tokenizer_device)
            if hasattr(processor.audio_tokenizer, "eval"):
                processor.audio_tokenizer.eval()

        model = MossModel.from_pretrained(
            self.model_path,
            attn_implementation=self.attn_implementation,
            dtype=self.dtype,
        )
        if self.lora_adapter:
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ImportError(
                    "MOSS LoRA adapter loading requires the 'peft' package. Install it and restart ComfyUI."
                ) from e
            # TTS Audio Suite Patch: allow loading MOSS PEFT/LoRA adapters for inference.
            model = PeftModel.from_pretrained(model, self.lora_adapter)
        model = model.to(self.device)
        model_cfg_runtime = getattr(model, "config", None)
        resolved_pad_token_id = tokenizer_pad_id
        if resolved_pad_token_id is None:
            resolved_pad_token_id = tokenizer_eos_id if tokenizer_eos_id is not None else 0
        if model_cfg_runtime is not None and getattr(model_cfg_runtime, "pad_token_id", None) is None:
            model_cfg_runtime.pad_token_id = int(resolved_pad_token_id)
        if model_cfg_runtime is not None and getattr(model_cfg_runtime, "num_hidden_layers", None) is None:
            language_cfg = getattr(model_cfg_runtime, "language_config", None)
            language_layers = getattr(language_cfg, "num_hidden_layers", None)
            local_layers = getattr(model_cfg_runtime, "local_num_layers", None)
            if language_layers is not None:
                model_cfg_runtime.num_hidden_layers = int(language_layers)
            elif local_layers is not None:
                model_cfg_runtime.num_hidden_layers = int(local_layers)
            else:
                model_cfg_runtime.num_hidden_layers = 1
        generation_cfg_runtime = getattr(model, "generation_config", None)
        if generation_cfg_runtime is not None:
            if getattr(generation_cfg_runtime, "pad_token_id", None) is None:
                generation_cfg_runtime.pad_token_id = int(resolved_pad_token_id)
            if getattr(generation_cfg_runtime, "eos_token_id", None) is None:
                generation_cfg_runtime.eos_token_id = int(getattr(model_cfg_runtime, "audio_end_token_id", 151653))
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
            self._processor.audio_tokenizer = audio_tokenizer.to(self._audio_tokenizer_device)
        return self

    def _ensure_runtime_device(self):
        self._ensure_model_loaded()
        target_device = "cuda" if torch.cuda.is_available() and self.device != "cpu" else self.device
        try:
            first_param = next(self._model.parameters())
        except StopIteration:
            return
        current_device = first_param.device
        normalized_target = torch.device(target_device) if not isinstance(target_device, torch.device) else target_device
        same_device = current_device.type == normalized_target.type
        if same_device and current_device.type == "cuda":
            current_index = 0 if current_device.index is None else current_device.index
            target_index = torch.cuda.current_device() if normalized_target.index is None else normalized_target.index
            same_device = current_index == target_index
        if not same_device:
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

    def _reference_to_wav(self, reference_audio, reference_sample_rate: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        if reference_audio is None:
            raise ValueError("MOSS-TTSD reference audio is missing")

        if isinstance(reference_audio, (str, os.PathLike)):
            from utils.audio.processing import AudioProcessingUtils

            waveform, sample_rate = AudioProcessingUtils.safe_load_audio(str(reference_audio))
        elif isinstance(reference_audio, dict) and "waveform" in reference_audio:
            waveform = reference_audio["waveform"]
            sample_rate = int(reference_audio.get("sample_rate", reference_sample_rate or self.SAMPLE_RATE))
        elif isinstance(reference_audio, tuple) and len(reference_audio) == 2:
            waveform, sample_rate = reference_audio
            sample_rate = int(sample_rate)
        elif torch.is_tensor(reference_audio):
            waveform = reference_audio
            sample_rate = int(reference_sample_rate or self.SAMPLE_RATE)
        else:
            raise TypeError(f"Unsupported MOSS-TTSD reference audio type: {type(reference_audio).__name__}")

        if not torch.is_tensor(waveform):
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
        waveform = waveform.detach().float().cpu()
        if waveform.dim() == 3:
            waveform = waveform[0]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(f"Unsupported MOSS-TTSD reference waveform shape: {tuple(waveform.shape)}")
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target_sr = int(getattr(self._processor.model_config, "sampling_rate", self.SAMPLE_RATE))
        if int(sample_rate) != target_sr:
            new_num_samples = int(round(waveform.shape[-1] * float(target_sr) / float(sample_rate)))
            if new_num_samples <= 0:
                raise ValueError(f"Invalid MOSS-TTSD resample length from {sample_rate}Hz to {target_sr}Hz")
            waveform = F.interpolate(
                waveform.unsqueeze(0),
                size=new_num_samples,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
            sample_rate = target_sr

        return waveform, int(sample_rate)

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

    @staticmethod
    def _normalize_prompt_text(prompt_text: str, speaker_id: int) -> str:
        text = str(prompt_text or "").strip()
        expected = f"[S{speaker_id}]"
        if not text:
            raise ValueError(f"MOSS-TTSD S{speaker_id} reference transcript is missing")
        if not text.lstrip().startswith(expected):
            text = f"{expected} {text}"
        return text

    @staticmethod
    def _merge_consecutive_speaker_tags(text: str) -> str:
        segments = re.split(r"(?=\[S\d+\])", str(text or ""))
        merged_parts = []
        current_tag = None
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            match = re.match(r"^(\[S\d+\])\s*(.*)", segment, re.DOTALL)
            if not match:
                merged_parts.append(segment)
                continue
            tag, content = match.groups()
            if tag == current_tag:
                merged_parts.append(content)
            else:
                current_tag = tag
                merged_parts.append(f"{tag}{content}")
        return "".join(merged_parts)

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
            return self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                audio_temperature=float(audio_temperature),
                audio_top_p=float(audio_top_p),
                audio_top_k=int(audio_top_k),
                audio_repetition_penalty=float(audio_repetition_penalty),
                n_vq_for_inference=n_vq_for_inference,
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
        instruction: Optional[str] = None,
        quality: Optional[str] = None,
        sound_event: Optional[str] = None,
        ambient_sound: Optional[str] = None,
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
            instruction=str(instruction) if instruction else None,
            quality=str(quality) if quality else None,
            sound_event=str(sound_event) if sound_event else None,
            ambient_sound=str(ambient_sound) if ambient_sound else None,
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

    def generate_dialogue(
        self,
        dialogue_text: str,
        speaker_references: List[Optional[Dict[str, Any]]],
        language: Optional[str] = None,
        duration_tokens: Optional[int] = None,
        instruction: Optional[str] = None,
        quality: Optional[str] = None,
        sound_event: Optional[str] = None,
        ambient_sound: Optional[str] = None,
        seed: int = 0,
        audio_temperature: float = 1.1,
        audio_top_p: float = 0.9,
        audio_top_k: int = 50,
        audio_repetition_penalty: float = 1.1,
        max_new_tokens: int = 4096,
    ) -> Tuple[torch.Tensor, int]:
        """Generate native MOSS-TTSD dialogue from [S1]...[S5] text."""
        if not str(dialogue_text or "").strip():
            raise ValueError("MOSS-TTSD requires non-empty dialogue text")

        self._ensure_runtime_device()

        if seed and seed > 0:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        language_value = self._normalize_language(language)
        target_sr = int(getattr(self._processor.model_config, "sampling_rate", self.SAMPLE_RATE))
        cloned_speakers: List[int] = []
        clone_wavs: List[torch.Tensor] = []
        prompt_texts: Dict[int, str] = {}

        for idx, speaker_ref in enumerate(speaker_references or []):
            speaker_id = idx + 1
            if not speaker_ref:
                continue

            reference_audio = speaker_ref.get("audio")
            if reference_audio is None:
                reference_audio = speaker_ref.get("waveform")
            if reference_audio is None:
                reference_audio = speaker_ref.get("audio_path") or speaker_ref.get("prompt_audio_path")

            reference_text = (
                speaker_ref.get("reference_text")
                or speaker_ref.get("text")
                or speaker_ref.get("prompt_text")
                or ""
            )
            if reference_audio is None or not str(reference_text).strip():
                print(
                    f"⚠️ MOSS-TTSD: S{speaker_id} needs both reference audio and transcript; "
                    "using unconditioned voice for that speaker"
                )
                continue

            waveform, sample_rate = self._reference_to_wav(
                reference_audio,
                speaker_ref.get("sample_rate"),
            )
            if sample_rate != target_sr:
                raise RuntimeError(f"MOSS-TTSD reference resample failed for S{speaker_id}: {sample_rate} != {target_sr}")
            cloned_speakers.append(speaker_id)
            clone_wavs.append(waveform)
            prompt_texts[speaker_id] = self._normalize_prompt_text(reference_text, speaker_id)

        if not cloned_speakers:
            user_message = self._processor.build_user_message(
                text=str(dialogue_text),
                tokens=int(duration_tokens) if duration_tokens else None,
                instruction=str(instruction) if instruction else None,
                quality=str(quality) if quality else None,
                sound_event=str(sound_event) if sound_event else None,
                ambient_sound=str(ambient_sound) if ambient_sound else None,
                language=language_value,
            )
            conversations = [[user_message]]
            mode = "generation"
        else:
            prompt_prefix = "".join(prompt_texts[speaker_id] for speaker_id in cloned_speakers)
            conversation_text = self._merge_consecutive_speaker_tags(prompt_prefix + str(dialogue_text))

            encoded_refs = self._processor.encode_audios_from_wav(
                wav_list=clone_wavs,
                sampling_rate=target_sr,
            )
            reference_audio_codes = [None for _ in range(len(speaker_references))]
            for speaker_id, audio_codes in zip(cloned_speakers, encoded_refs):
                reference_audio_codes[speaker_id - 1] = audio_codes

            concat_prompt_wav = torch.cat(clone_wavs, dim=-1)
            prompt_audio = self._processor.encode_audios_from_wav(
                wav_list=[concat_prompt_wav],
                sampling_rate=target_sr,
            )[0]

            user_message = self._processor.build_user_message(
                text=conversation_text,
                reference=reference_audio_codes,
                tokens=int(duration_tokens) if duration_tokens else None,
                instruction=str(instruction) if instruction else None,
                quality=str(quality) if quality else None,
                sound_event=str(sound_event) if sound_event else None,
                ambient_sound=str(ambient_sound) if ambient_sound else None,
                language=language_value,
            )
            assistant_message = self._processor.build_assistant_message(
                audio_codes_list=[prompt_audio],
            )
            conversations = [[user_message, assistant_message]]
            mode = "continuation"

        batch = self._processor(conversations, mode=mode)
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
            )

        messages = self._processor.decode(outputs)
        if not messages or messages[0] is None or not messages[0].audio_codes_list:
            raise RuntimeError("MOSS-TTSD returned no decodable audio")

        audio = messages[0].audio_codes_list[0]
        if not torch.is_tensor(audio):
            audio = torch.as_tensor(audio, dtype=torch.float32)
        audio = audio.detach().float().cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3:
            audio = audio.squeeze(0)
        return audio, target_sr

    def _normalize_language(self, language: Optional[str]) -> Optional[str]:
        if language is None:
            return None
        value = str(language).strip()
        if not value or value.lower() == "auto":
            return None
        return self.SUPPORTED_LANGUAGES.get(value.lower(), value.lower())
