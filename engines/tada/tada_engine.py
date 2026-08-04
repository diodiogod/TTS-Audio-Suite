"""Local-only wrapper around the official ``hume-tada`` inference classes."""

from __future__ import annotations

import gc
import hashlib
import logging
import os
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import patch

import torch

from .languages import normalize_tada_language, validate_tada_model_language


TADA_SAMPLE_RATE = 24000


class TadaEngine:
    """Official TADA inference with every model dependency resolved locally."""

    _CFG_SCHEDULES = {"constant", "linear", "cosine"}
    _TIME_SCHEDULES = {"uniform", "cosine", "logsnr"}
    _NEGATIVE_SOURCES = {"negative_step_output", "prompt", "zero"}

    def __init__(
        self,
        model_name: str,
        model_path: str,
        codec_path: str,
        tokenizer_path: str,
        device: str = "auto",
        dtype: str = "auto",
        attn_implementation: str = "sdpa",
        use_torch_compile: bool = False,
        prompt_cache_size: int = 8,
    ):
        self.model_name = str(model_name)
        self.model_path = os.path.abspath(str(model_path))
        self.codec_path = os.path.abspath(str(codec_path))
        self.tokenizer_path = os.path.abspath(str(tokenizer_path))
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.attn_implementation = str(attn_implementation or "sdpa")
        self.use_torch_compile = bool(use_torch_compile)
        self.prompt_cache_size = max(0, int(prompt_cache_size))
        self._prompt_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.model = None

        self._validate_local_assets()
        self._load_model()

    @staticmethod
    def _resolve_device(device: object) -> torch.device:
        value = str(device or "auto")
        if value == "auto":
            value = "cuda" if torch.cuda.is_available() else "cpu"
        if value.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("TADA was configured for CUDA, but CUDA is not available in its shared runtime.")
        return torch.device(value)

    def _resolve_dtype(self, dtype: object) -> torch.dtype:
        value = str(dtype or "auto").lower()
        if value == "auto":
            if self.device.type == "cuda":
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if value not in mapping:
            raise ValueError(f"Unsupported TADA dtype '{dtype}'")
        if self.device.type == "cpu" and mapping[value] == torch.float16:
            return torch.float32
        return mapping[value]

    @staticmethod
    def _require_files(root: str, relative_paths: Tuple[str, ...], label: str) -> None:
        missing = [name for name in relative_paths if not os.path.isfile(os.path.join(root, name))]
        if missing:
            raise RuntimeError(f"Local {label} assets are incomplete at {root}. Missing: {missing}")

    def _validate_local_assets(self) -> None:
        model_files = ("config.json",)
        tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        codec_files = (
            "encoder/config.json",
            "encoder/model.safetensors",
            "decoder/config.json",
            "decoder/model.safetensors",
            "wav2vec2-large/config.json",
        )
        self._require_files(self.model_path, model_files, "TADA checkpoint")
        if not any(Path(self.model_path).glob("*.safetensors")):
            raise RuntimeError(f"Local TADA checkpoint has no safetensors weights: {self.model_path}")
        self._require_files(self.tokenizer_path, tokenizer_files, "Llama tokenizer")
        self._require_files(self.codec_path, codec_files, "TADA codec")

    def _load_model(self) -> None:
        try:
            import transformers
            from tada.modules.decoder import Decoder
            from tada.modules.tada import TadaForCausalLM
            from transformers import AutoTokenizer, LlamaForCausalLM
        except Exception as exc:
            raise RuntimeError(
                "TADA requires the shared Transformers 4 runtime with hume-tada installed. "
                "Select the Shared Runtime option and retry. "
                f"Import error: {exc}"
            ) from exc

        major = int(str(transformers.__version__).split(".", 1)[0])
        if major >= 5:
            raise RuntimeError(
                f"TADA is not compatible with Transformers {transformers.__version__} in the main runtime. "
                "Use the vibevoice_transformers4_shared runtime profile."
            )

        load_kwargs = {
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "dtype": self.dtype,
        }
        if self.attn_implementation not in {"", "auto"}:
            load_kwargs["attn_implementation"] = self.attn_implementation

        try:
            # Calling the parent classmethod directly bypasses TADA's override,
            # which otherwise downloads its decoder and tokenizer by repository ID.
            class _ExpectedBundledDecoderWarning(logging.Filter):
                def filter(self, record):
                    message = record.getMessage()
                    return not (
                        "were not used when initializing TadaForCausalLM" in message
                        and "_decoder." in message
                    )

            warning_filter = _ExpectedBundledDecoderWarning()
            model_logger = logging.getLogger("transformers.modeling_utils")
            model_logger.addFilter(warning_filter)
            try:
                model, loading_info = LlamaForCausalLM.from_pretrained.__func__(
                    TadaForCausalLM,
                    self.model_path,
                    output_loading_info=True,
                    **load_kwargs,
                )
            finally:
                model_logger.removeFilter(warning_filter)

            unexpected = loading_info.get("unexpected_keys", [])
            non_decoder = [key for key in unexpected if not key.startswith("_decoder.")]
            if non_decoder:
                raise RuntimeError(
                    "TADA checkpoint contains unexpected non-decoder weights: "
                    f"{non_decoder[:10]}"
                )
            if not unexpected:
                raise RuntimeError(
                    "TADA checkpoint did not expose the expected bundled decoder keys. "
                    "Its layout may be incompatible with this hume-tada runtime."
                )
            # This matches hume-tada's official from_pretrained contract: the
            # checkpoint's bundled _decoder keys are intentionally ignored and
            # the released tada-codec decoder is attached instead.
            decoder = Decoder.from_pretrained(
                self.codec_path,
                subfolder="decoder",
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load TADA entirely from local assets: {exc}") from exc

        model._decoder = decoder
        model._tokenizer = tokenizer
        model.to(self.device)
        model.eval()
        if self.use_torch_compile:
            self._enable_torch_compile(model)
        self.model = model

    def _enable_torch_compile(self, model) -> None:
        if self.device.type != "cuda":
            print("⚠️ TADA: torch.compile is only enabled on CUDA; using standard inference.")
            self.use_torch_compile = False
            return
        try:
            import triton  # noqa: F401
        except Exception as exc:
            print(f"⚠️ TADA: torch.compile requires a working Triton install; using standard inference. ({exc})")
            self.use_torch_compile = False
            return

        original_forward = model.prediction_head.forward
        try:
            compiled_forward = torch.compile(original_forward, mode="default")
        except Exception as exc:
            print(f"⚠️ TADA: could not prepare torch.compile; using standard inference. ({exc})")
            self.use_torch_compile = False
            return

        failed = False

        def compile_with_fallback(*args, **kwargs):
            nonlocal failed
            try:
                return compiled_forward(*args, **kwargs)
            except Exception as exc:
                if not failed:
                    print(f"⚠️ TADA: compiled refinement failed; falling back to standard inference. ({exc})")
                    failed = True
                model.prediction_head.forward = original_forward
                self.use_torch_compile = False
                return original_forward(*args, **kwargs)

        model.prediction_head.forward = compile_with_fallback
        print("✅ TADA: torch.compile enabled for audio refinement (first generation will compile).")

    def _required_aligner_subfolder(self, language: object) -> str:
        code = validate_tada_model_language(self.model_name, language)
        return "aligner" if code is None else f"aligner-{code}"

    def _load_encoder(self, language: object):
        from tada.modules import aligner as aligner_module
        from tada.modules.aligner import Aligner, AlignerConfig
        from tada.modules.encoder import Encoder, EncoderConfig
        from transformers import PreTrainedModel, Wav2Vec2Config

        aligner_subfolder = self._required_aligner_subfolder(language)
        self._require_files(
            self.codec_path,
            (
                f"{aligner_subfolder}/config.json",
                f"{aligner_subfolder}/model.safetensors",
            ),
            f"TADA {aligner_subfolder}",
        )

        wav2vec_config = Wav2Vec2Config.from_pretrained(
            os.path.join(self.codec_path, "wav2vec2-large"),
            local_files_only=True,
        )

        # Aligner.__init__ normally resolves both names over the network. Supply
        # the already-loaded local tokenizer and explicit local Wav2Vec config.
        with patch.object(
            aligner_module.AutoTokenizer,
            "from_pretrained",
            return_value=self.model.tokenizer,
        ), patch.object(
            aligner_module.AutoConfig,
            "from_pretrained",
            return_value=wav2vec_config,
        ):
            aligner = Aligner.from_pretrained(
                self.codec_path,
                subfolder=aligner_subfolder,
                config=AlignerConfig(),
                local_files_only=True,
                low_cpu_mem_usage=True,
            )

        # Encoder overrides from_pretrained and would construct another remotely
        # resolved aligner. Invoke the Transformers base loader and attach ours.
        encoder = PreTrainedModel.from_pretrained.__func__(
            Encoder,
            self.codec_path,
            subfolder="encoder",
            config=EncoderConfig(),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        encoder._aligner = aligner
        encoder.to(self.device)
        encoder.eval()
        return encoder

    @staticmethod
    def _normalize_reference(reference_audio: object) -> Tuple[torch.Tensor, int]:
        import torchaudio

        sample_rate = TADA_SAMPLE_RATE
        audio = reference_audio

        if isinstance(audio, (str, os.PathLike)):
            waveform, sample_rate = torchaudio.load(str(audio))
            audio = waveform
        elif isinstance(audio, (tuple, list)) and len(audio) == 2:
            audio, sample_rate = audio
        elif isinstance(audio, dict):
            if "audio" in audio and isinstance(audio["audio"], dict):
                audio = audio["audio"]
            if isinstance(audio, dict):
                path = audio.get("audio_path") or audio.get("path")
                if path:
                    waveform, sample_rate = torchaudio.load(str(path))
                    audio = waveform
                else:
                    sample_rate = audio.get("sample_rate", audio.get("sampling_rate", sample_rate))
                    audio = audio.get("waveform", audio.get("samples"))

        if audio is None:
            raise ValueError("TADA requires reference audio; no waveform or audio path was provided.")
        waveform = audio.detach().cpu() if isinstance(audio, torch.Tensor) else torch.as_tensor(audio)
        waveform = waveform.float()
        if waveform.ndim == 0 or waveform.numel() == 0:
            raise ValueError("TADA reference audio is empty.")
        if waveform.ndim >= 3:
            waveform = waveform[0]
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        elif waveform.ndim > 2:
            waveform = waveform.reshape(-1, waveform.shape[-1]).mean(dim=0)
        waveform = waveform.reshape(1, -1).contiguous()
        if not torch.isfinite(waveform).all():
            raise ValueError("TADA reference audio contains NaN or infinite values.")
        peak = waveform.abs().max()
        if float(peak) <= 0.0:
            raise ValueError("TADA reference audio is silent.")
        waveform = waveform / peak
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"Invalid TADA reference sample rate: {sample_rate}")
        return waveform, sample_rate

    @staticmethod
    def _prompt_state_to_device(state: Dict[str, Any], device: torch.device):
        from tada.modules.encoder import EncoderOutput

        restored = {}
        for name, value in state.items():
            restored[name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return EncoderOutput(**restored)

    @staticmethod
    def _prompt_state_to_cpu(prompt: object) -> Dict[str, Any]:
        state = {}
        for field in fields(prompt):
            value = getattr(prompt, field.name)
            state[field.name] = value.detach().cpu() if isinstance(value, torch.Tensor) else value
        return state

    @staticmethod
    def _prompt_cache_key(
        waveform: torch.Tensor,
        sample_rate: int,
        transcript: str,
        language: Optional[str],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(waveform.contiguous().numpy().tobytes())
        digest.update(str(sample_rate).encode("ascii"))
        digest.update(transcript.encode("utf-8"))
        digest.update(str(language).encode("ascii"))
        return digest.hexdigest()

    def _get_prompt(
        self,
        reference_audio: object,
        reference_text: str,
        language: object,
    ):
        waveform, sample_rate = self._normalize_reference(reference_audio)
        language_code = normalize_tada_language(language)
        key = self._prompt_cache_key(waveform, sample_rate, reference_text, language_code)
        cached = self._prompt_cache.pop(key, None)
        if cached is not None:
            self._prompt_cache[key] = cached
            return self._prompt_state_to_device(cached, self.device)

        fork_devices = []
        if self.device.type == "cuda":
            fork_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=fork_devices):
            # Prompt sampling is intentionally stable and independent from the
            # user generation seed. A cold and warm prompt cache therefore
            # produce identical generation RNG state.
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            encoder = self._load_encoder(language_code)
            try:
                prompt = encoder(
                    waveform.to(self.device),
                    text=[reference_text],
                    sample_rate=sample_rate,
                )
                state = self._prompt_state_to_cpu(prompt)
            finally:
                del encoder
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if self.prompt_cache_size > 0:
            self._prompt_cache[key] = state
            while len(self._prompt_cache) > self.prompt_cache_size:
                self._prompt_cache.popitem(last=False)
        return self._prompt_state_to_device(state, self.device)

    @contextmanager
    def _generation_progress(
        self,
        text: str,
        num_flow_matching_steps: int,
        num_transition_steps: int,
        two_pass: bool,
    ):
        """Estimate total work from TADA's tokenizer and count real flow iterations."""
        from tqdm.auto import tqdm

        original_compute_velocity = self.model._compute_velocity
        target_tokens = len(self.model.tokenizer.encode(text, add_special_tokens=False))
        # The official generator pre-fills the reference prompt, leaving target
        # text, transition, EOS, and a few structural positions to synthesize.
        estimated_generation_steps = max(
            1,
            target_tokens + int(num_transition_steps) + int(self.model.num_eos_tokens) + 4,
        )
        estimated_total = estimated_generation_steps * int(num_flow_matching_steps)
        if two_pass:
            estimated_total *= 2
        progress = tqdm(
            total=estimated_total,
            desc="TADA flow",
            unit="it",
            dynamic_ncols=True,
            mininterval=0.5,
            leave=True,
        )

        def tracked_compute_velocity(*args, **kwargs):
            result = original_compute_velocity(*args, **kwargs)
            progress.update(1)
            return result

        try:
            with patch.object(self.model, "_compute_velocity", tracked_compute_velocity):
                yield
        finally:
            # EOS can make the estimate short or long. Finish at the real count
            # instead of leaving a successful generation below/above 100%.
            progress.total = progress.n
            progress.close()

    def generate_speech(
        self,
        text: str,
        reference_audio: object,
        reference_text: str,
        seed: int = 42,
        language: object = "English",
        acoustic_cfg_scale: float = 1.6,
        duration_cfg_scale: float = 1.0,
        cfg_schedule: str = "cosine",
        time_schedule: str = "logsnr",
        num_flow_matching_steps: int = 10,
        noise_temperature: float = 0.9,
        speed_up_factor: Optional[float] = None,
        num_transition_steps: int = 5,
        negative_condition_source: str = "negative_step_output",
    ) -> Tuple[torch.Tensor, int]:
        if self.model is None:
            raise RuntimeError("TADA is not initialized.")
        if not str(text or "").strip():
            raise ValueError("TADA synthesis text cannot be empty.")
        if not str(reference_text or "").strip():
            raise ValueError(
                "TADA requires the exact transcript of the reference audio. "
                "Automatic transcription is intentionally disabled."
            )
        cfg_schedule = str(cfg_schedule).lower()
        time_schedule = str(time_schedule).lower()
        negative_condition_source = str(negative_condition_source).lower()
        if cfg_schedule not in self._CFG_SCHEDULES:
            raise ValueError(f"Invalid TADA CFG schedule: {cfg_schedule}")
        if time_schedule not in self._TIME_SCHEDULES:
            raise ValueError(f"Invalid TADA time schedule: {time_schedule}")
        if negative_condition_source not in self._NEGATIVE_SOURCES:
            raise ValueError(f"Invalid TADA negative condition source: {negative_condition_source}")
        if int(num_flow_matching_steps) < 1:
            raise ValueError("TADA flow matching steps must be at least 1.")
        if int(num_transition_steps) < 0:
            raise ValueError("TADA transition steps cannot be negative.")
        if speed_up_factor is not None and float(speed_up_factor) <= 0:
            raise ValueError("TADA speed_up_factor must be positive or None.")

        from tada.modules.tada import InferenceOptions

        seed = int(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        prompt = self._get_prompt(reference_audio, str(reference_text), language)
        options = InferenceOptions(
            acoustic_cfg_scale=float(acoustic_cfg_scale),
            duration_cfg_scale=float(duration_cfg_scale),
            cfg_schedule=cfg_schedule,
            noise_temperature=float(noise_temperature),
            num_flow_matching_steps=int(num_flow_matching_steps),
            time_schedule=time_schedule,
            speed_up_factor=None if speed_up_factor is None else float(speed_up_factor),
            negative_condition_source=negative_condition_source,
        )
        with self._generation_progress(
            text=str(text),
            num_flow_matching_steps=int(num_flow_matching_steps),
            num_transition_steps=int(num_transition_steps),
            two_pass=speed_up_factor is not None,
        ):
            output = self.model.generate(
                prompt=prompt,
                text=str(text),
                num_transition_steps=int(num_transition_steps),
                inference_options=options,
            )
        audio = output.audio[0] if output.audio else None
        if audio is None:
            raise RuntimeError("TADA generation completed without a decodable audio result.")
        waveform = audio.detach().float().cpu() if isinstance(audio, torch.Tensor) else torch.as_tensor(audio).float()
        waveform = waveform.squeeze()
        if waveform.ndim > 1:
            waveform = waveform.reshape(-1, waveform.shape[-1]).mean(dim=0)
        waveform = waveform.contiguous()
        if waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise RuntimeError("TADA produced an empty or invalid waveform.")
        return waveform, TADA_SAMPLE_RATE

    def cleanup(self) -> None:
        self._prompt_cache.clear()
        model = self.model
        self.model = None
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["TADA_SAMPLE_RATE", "TadaEngine"]
