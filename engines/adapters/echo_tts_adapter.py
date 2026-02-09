"""
Echo-TTS Engine Adapter

Provides a unified interface for Echo-TTS integration with TTS Audio Suite.
"""

import os
import hashlib
import inspect
import sys
from typing import Dict, Any, List, Optional, Tuple
from functools import partial

import torch

# Add project root to path
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio.processing import AudioProcessingUtils
from utils.audio.cache import get_audio_cache
from utils.text.chunking import ImprovedChatterBoxChunker
from utils.audio.chunk_timing import ChunkTimingHelper
from utils.models.extra_paths import get_preferred_download_path


class EchoTTSEngineAdapter:
    """Adapter for Echo-TTS model loading and inference."""

    SAMPLE_RATE = 44100
    MAX_REF_SECONDS = 300  # 5 minutes

    def __init__(self, config: Dict[str, Any]):
        self.config = config.copy() if config else {}
        self.model = None
        self.ae = None
        self.pca_state = None
        self._sample_pipeline = None
        self._loaded_key = None
        self._wrapper = None
        self.audio_cache = get_audio_cache()

    def update_config(self, new_config: Dict[str, Any]):
        self.config = new_config.copy() if new_config else {}
        # If critical load params changed, force reload on next call
        new_key = self._get_load_key()
        if self._loaded_key != new_key:
            self.model = None
            self.ae = None
            self.pca_state = None
            self._sample_pipeline = None
            self._loaded_key = None

    def _get_load_key(self) -> str:
        model_id = self.config.get("model", "jordand/echo-tts-base")
        device = self._resolve_device(self.config.get("device", "auto"))
        return f"{model_id}::{device}"

    @staticmethod
    def _filter_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in signature.parameters}

    def _resolve_device(self, device: str) -> str:
        device = (device or "auto").lower()
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: Echo-TTS: CUDA requested but not available. Falling back to CPU.")
            return "cpu"

        from utils.device import resolve_torch_device
        resolved = resolve_torch_device(device)

        if resolved == "cpu":
            print("WARNING: Echo-TTS running on CPU will be very slow.")
        return resolved

    @staticmethod
    def _normalize_prompt_text(text: str) -> str:
        """
        Ensure Echo-TTS prompt format uses [S1] prefix for narrator.

        If [S1] is already present, remove it and re-add once to normalize.
        This avoids character tag parsing conflicts elsewhere in the pipeline.
        """
        if not text:
            return "[S1]"

        cleaned = text.strip()
        if cleaned.lower().startswith("[s1]"):
            cleaned = cleaned[4:].lstrip()

        return f"[S1] {cleaned}".strip()

    def _get_local_repo_dir(self, repo_id: str, base_dir: Optional[str] = None) -> str:
        base_dir = base_dir or get_preferred_download_path("TTS")
        repo_name = repo_id.split("/")[-1]
        repo_dir = os.path.join(base_dir, repo_name)
        os.makedirs(repo_dir, exist_ok=True)
        return repo_dir

    def _hf_download(self, repo_id: str, filename: str, base_dir: Optional[str] = None) -> str:
        from huggingface_hub import hf_hub_download
        repo_dir = self._get_local_repo_dir(repo_id, base_dir=base_dir)
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=repo_dir,
            local_dir_use_symlinks=False,
        )

    def _ensure_model_loaded(self):
        load_key = self._get_load_key()
        if self.model is not None and self._loaded_key == load_key and self._wrapper is not None:
            return

        from utils.models.unified_model_interface import unified_model_interface, ModelLoadConfig
        from utils.device import resolve_torch_device

        device = resolve_torch_device(self.config.get("device", "auto"))
        model_id = self.config.get("model", "jordand/echo-tts-base")

        config = ModelLoadConfig(
            engine_name="echo_tts",
            model_type="tts",
            model_name=model_id,
            device=device,
        )

        self._wrapper = unified_model_interface.load_model(config)
        bundle = self._wrapper.model
        self.model = bundle.model
        self.ae = bundle.ae
        self.pca_state = bundle.pca_state
        self._sample_pipeline = bundle.sample_pipeline
        self._loaded_key = load_key

    def _prepare_reference_audio(self, speaker_audio: Any) -> Tuple[torch.Tensor, int]:
        if speaker_audio is None:
            raise ValueError("Echo-TTS requires reference audio.")

        if isinstance(speaker_audio, dict) and "waveform" in speaker_audio:
            waveform = speaker_audio["waveform"]
            sample_rate = int(speaker_audio.get("sample_rate", self.SAMPLE_RATE))
        elif isinstance(speaker_audio, str):
            waveform, sample_rate = AudioProcessingUtils.safe_load_audio(speaker_audio)
        else:
            waveform = speaker_audio
            sample_rate = self.SAMPLE_RATE

        if isinstance(waveform, torch.Tensor):
            audio = waveform
        else:
            audio = torch.tensor(waveform, dtype=torch.float32)

        if audio.dim() == 3:
            audio = audio[0]
        if audio.dim() == 2 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if audio.dim() == 2 and audio.shape[0] == 1:
            audio = audio.squeeze(0)

        # Resample to Echo-TTS sample rate
        if sample_rate != self.SAMPLE_RATE:
            import torchaudio
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.SAMPLE_RATE)
            audio = resampler(audio)
            sample_rate = self.SAMPLE_RATE

        # Normalize to [-1, 1]
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val

        # Ensure shape (1, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Trim to 5 minutes max
        max_samples = int(self.MAX_REF_SECONDS * sample_rate)
        if audio.shape[-1] > max_samples:
            audio = audio[:, :max_samples]

        return audio, sample_rate

    def _parse_languages(self) -> List[str]:
        langs = self.config.get("languages", "en")
        if isinstance(langs, list):
            return langs
        if not isinstance(langs, str):
            return ["en"]
        parts = [part.strip() for part in langs.split(",") if part.strip()]
        return parts or ["en"]

    def _generate_audio_for_text(
        self,
        text: str,
        ref_audio: torch.Tensor,
        ref_text: str,
        enable_audio_cache: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        self._ensure_model_loaded()

        # Seed for reproducibility if provided
        seed = int(self.config.get("seed", 0))
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        params = {
            "num_steps": int(self.config.get("num_steps", 40)),
            "cfg_scale_text": float(self.config.get("cfg_scale_text", 3.0)),
            "cfg_scale_speaker": float(self.config.get("cfg_scale_speaker", 8.0)),
            "cfg_min_t": float(self.config.get("cfg_min_t", 0.5)),
            "cfg_max_t": float(self.config.get("cfg_max_t", 1.0)),
            "truncation_factor": self.config.get("truncation_factor", None),
            "rescale_k": self.config.get("rescale_k", None),
            "rescale_sigma": self.config.get("rescale_sigma", None),
            "speaker_kv_scale": self.config.get("speaker_kv_scale", None),
            "speaker_kv_max_layers": self.config.get("speaker_kv_max_layers", None),
            "speaker_kv_min_t": self.config.get("speaker_kv_min_t", None),
            "sequence_length": int(self.config.get("sequence_length", 640)),
        }

        prompt_text = self._normalize_prompt_text(text)

        cache_key = None
        if enable_audio_cache:
            try:
                ref_audio_hash = hashlib.md5(ref_audio.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                audio_component = f"ref_audio_{ref_audio_hash}_{self.SAMPLE_RATE}"
            except Exception as e:
                print(f"⚠️ Echo-TTS: Failed to hash reference audio for cache key: {e}")
                audio_component = "ref_audio_error"

            resolved_device = self.config.get("device", "auto")
            if self._loaded_key and "::" in self._loaded_key:
                _, resolved_device = self._loaded_key.split("::", 1)

            cache_key = self.audio_cache.generate_cache_key(
                'echo_tts',
                text=prompt_text,
                audio_component=audio_component,
                reference_text=ref_text,
                model=self.config.get("model", "jordand/echo-tts-base"),
                device=resolved_device,
                num_steps=params["num_steps"],
                cfg_scale_text=params["cfg_scale_text"],
                cfg_scale_speaker=params["cfg_scale_speaker"],
                cfg_min_t=params["cfg_min_t"],
                cfg_max_t=params["cfg_max_t"],
                truncation_factor=params["truncation_factor"],
                rescale_k=params["rescale_k"],
                rescale_sigma=params["rescale_sigma"],
                speaker_kv_scale=params["speaker_kv_scale"],
                speaker_kv_max_layers=params["speaker_kv_max_layers"],
                speaker_kv_min_t=params["speaker_kv_min_t"],
                sequence_length=params["sequence_length"],
                seed=seed,
            )

            cached_audio = self.audio_cache.get_cached_audio(cache_key)
            if cached_audio:
                print(f"💾 Using cached Echo-TTS audio: '{text[:30]}...'")
                return cached_audio[0], self.SAMPLE_RATE

        # Resolve sampler
        sampler_kwargs = {
            "num_steps": params["num_steps"],
            "cfg_scale_text": params["cfg_scale_text"],
            "cfg_scale_speaker": params["cfg_scale_speaker"],
            "cfg_min_t": params["cfg_min_t"],
            "cfg_max_t": params["cfg_max_t"],
            "truncation_factor": params["truncation_factor"],
            "rescale_k": params["rescale_k"],
            "rescale_sigma": params["rescale_sigma"],
            "speaker_kv_scale": params["speaker_kv_scale"],
            "speaker_kv_max_layers": params["speaker_kv_max_layers"],
            "speaker_kv_min_t": params["speaker_kv_min_t"],
            "sequence_length": params["sequence_length"],
            # Alternate parameter names used by some sampler variants
            "cfg_scale": params["cfg_scale_text"],
            "speaker_k_scale": params["speaker_kv_scale"],
            "speaker_k_max_layers": params["speaker_kv_max_layers"],
            "speaker_k_min_t": params["speaker_kv_min_t"],
            "block_size": params["sequence_length"],
        }
        try:
            from echo_tts.samplers import sample_euler_cfg_any, GuidanceMode
            sample_fn = partial(
                sample_euler_cfg_any,
                **self._filter_kwargs(
                    sample_euler_cfg_any,
                    {
                        **sampler_kwargs,
                        "guidance_mode": GuidanceMode.INDEPENDENT,
                        "apg_eta_text": None,
                        "apg_eta_speaker": None,
                        "apg_momentum_text": None,
                        "apg_momentum_speaker": None,
                        "apg_norm_text": None,
                        "apg_norm_speaker": None,
                    },
                ),
            )
        except Exception:
            try:
                from echo_tts.inference import sample_euler_cfg_independent_guidances
                sample_fn = partial(
                    sample_euler_cfg_independent_guidances,
                    **self._filter_kwargs(sample_euler_cfg_independent_guidances, sampler_kwargs),
                )
            except Exception:
                from echo_tts.samplers import sample_euler_cfg
                sample_fn = partial(
                    sample_euler_cfg,
                    **self._filter_kwargs(sample_euler_cfg, sampler_kwargs),
                )

        with torch.no_grad():
            result = self._sample_pipeline(
                model=self.model,
                fish_ae=self.ae,
                pca_state=self.pca_state,
                sample_fn=sample_fn,
                text_prompt=prompt_text,
                speaker_audio=ref_audio,
                rng_seed=seed,
                pad_to_max_speaker_latent_length=2560,
                pad_to_max_text_length=768,
                normalize_text=True,
            )

        audio = None
        sample_rate = self.SAMPLE_RATE

        if isinstance(result, dict):
            audio = result.get("wav") or result.get("audio") or result.get("waveform")
            sample_rate = int(result.get("sr", result.get("sample_rate", self.SAMPLE_RATE)))
        elif isinstance(result, tuple) and len(result) >= 1:
            audio = result[0]
            if len(result) >= 2 and isinstance(result[1], (int, float)):
                sample_rate = int(result[1])
        else:
            audio = result

        if isinstance(audio, torch.Tensor):
            audio_tensor = audio
        else:
            audio_tensor = torch.tensor(audio, dtype=torch.float32)

        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()

        if sample_rate != self.SAMPLE_RATE:
            import torchaudio
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.SAMPLE_RATE)
            audio_tensor = resampler(audio_tensor)
            sample_rate = self.SAMPLE_RATE

        if enable_audio_cache and cache_key:
            duration = self.audio_cache._calculate_duration(audio_tensor, 'echo_tts')
            self.audio_cache.cache_audio(cache_key, audio_tensor, duration)

        return audio_tensor, sample_rate

    def process_text(self, text: str, speaker_audio: Any, reference_text: str, seed: int = 0,
                     enable_chunking: bool = True, max_chars_per_chunk: int = 400,
                     chunk_combination_method: str = "auto", silence_between_chunks_ms: int = 100,
                     enable_audio_cache: bool = True, return_info: bool = False):
        if seed is not None:
            self.config["seed"] = seed

        ref_audio, _ = self._prepare_reference_audio(speaker_audio)
        ref_text = reference_text or ""

        if enable_chunking:
            max_chars = ImprovedChatterBoxChunker.validate_chunking_params(max_chars_per_chunk)
            text_chunks = ImprovedChatterBoxChunker.split_into_chunks(text, max_chars=max_chars)
        else:
            text_chunks = [text]

        audio_segments = []
        for chunk in text_chunks:
            audio_tensor, _ = self._generate_audio_for_text(
                chunk,
                ref_audio,
                ref_text,
                enable_audio_cache=enable_audio_cache,
            )
            audio_segments.append(audio_tensor)

        combined_audio, chunk_info = ChunkTimingHelper.combine_audio_with_timing(
            audio_segments=audio_segments,
            combination_method=chunk_combination_method,
            silence_ms=silence_between_chunks_ms,
            crossfade_duration=0.1,
            sample_rate=self.SAMPLE_RATE,
            text_length=len(text),
            original_text=text,
            text_chunks=text_chunks,
        )

        if return_info:
            return combined_audio, chunk_info
        return combined_audio

    def process_srt_content(self, srt_content: str, speaker_audio: Any, reference_text: str,
                            seed: int = 0, timing_mode: str = "pad_with_silence",
                            timing_params: Optional[Dict[str, Any]] = None,
                            enable_audio_cache: bool = True):
        from utils.system.import_manager import import_manager
        from utils.timing.assembly import AudioAssemblyEngine

        success, modules, _ = import_manager.import_srt_modules()
        if not success:
            raise RuntimeError("SRT modules are not available")

        SRTParser = modules.get("SRTParser")
        if SRTParser is None:
            raise RuntimeError("SRT parser not available")

        srt_parser = SRTParser()
        subtitles = srt_parser.parse_srt_content(srt_content, allow_overlaps=True)

        # Echo-TTS SRT support is minimal; fall back to pad_with_silence if needed
        if timing_mode not in ["pad_with_silence", "concatenate"]:
            print(f"WARNING: Echo-TTS SRT: timing_mode '{timing_mode}' not supported, using pad_with_silence")
            timing_mode = "pad_with_silence"

        if seed is not None:
            self.config["seed"] = seed
        ref_audio, _ = self._prepare_reference_audio(speaker_audio)
        ref_text = reference_text or ""

        audio_segments = []
        for subtitle in subtitles:
            audio_tensor, _ = self._generate_audio_for_text(
                subtitle.text,
                ref_audio,
                ref_text,
                enable_audio_cache=enable_audio_cache,
            )
            audio_segments.append(audio_tensor.cpu())

        assembler = AudioAssemblyEngine(sample_rate=self.SAMPLE_RATE)
        final_audio = assembler.assemble_by_timing_mode(
            audio_segments=audio_segments,
            subtitles=subtitles,
            timing_mode=timing_mode,
            device="cpu",
            adjustments=None,
            processed_segments=None,
            fade_duration=0.01
        )

        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0).unsqueeze(0)
        elif final_audio.dim() == 2:
            final_audio = final_audio.unsqueeze(0)

        audio_output = {"waveform": final_audio, "sample_rate": self.SAMPLE_RATE}

        timing_info = srt_parser.get_timing_info(subtitles)
        timing_report = (
            f"Echo-TTS SRT timing\n"
            f"Subtitles: {timing_info.get('subtitle_count', 0)}\n"
            f"Total duration: {timing_info.get('total_duration', 0):.2f}s\n"
            f"Mode: {timing_mode}"
        )

        total_duration = final_audio.shape[-1] / float(self.SAMPLE_RATE)
        info = f"Generated {total_duration:.1f}s Echo-TTS SRT audio using {timing_mode} mode"

        return audio_output, info, timing_report, srt_content
