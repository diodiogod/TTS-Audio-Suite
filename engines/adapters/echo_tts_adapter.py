"""
Echo-TTS Engine Adapter

Provides a unified interface for Echo-TTS integration with TTS Audio Suite.
"""

import os
import hashlib
import inspect
import sys
from typing import Dict, Any, Optional, Tuple
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
        self.job_tracker = None

    def start_job(self, total_blocks: int, block_texts: list):
        """Initialize job tracker for weighted multi-block progress estimation."""
        import time

        normalized_texts = [max(1, int(v)) for v in (block_texts or [])]
        self.job_tracker = {
            "start_time": time.time(),
            "total_blocks": max(0, int(total_blocks)),
            "blocks_completed": 0,
            "block_texts": normalized_texts,
            "total_text": sum(normalized_texts),
            "text_completed": 0,
            "current_block_text": 0,
            "current_block_steps": 0,
            "current_block_total_steps": 0,
            "total_steps_completed": 0,
        }

    def set_current_block(self, block_idx: int):
        """Set current block metadata for weighted ETA calculation."""
        if not self.job_tracker:
            return
        block_texts = self.job_tracker.get("block_texts", [])
        if block_idx < 0 or block_idx >= len(block_texts):
            return
        self.job_tracker["current_block_text"] = block_texts[block_idx]
        self.job_tracker["current_block_steps"] = 0
        self.job_tracker["current_block_total_steps"] = 0

    def complete_block(self):
        """Mark current block as completed in the job tracker."""
        if not self.job_tracker:
            return
        self.job_tracker["blocks_completed"] += 1
        self.job_tracker["text_completed"] += self.job_tracker.get("current_block_text", 0)
        self.job_tracker["total_steps_completed"] += self.job_tracker.get("current_block_steps", 0)

    def end_job(self):
        """Clear active job tracker."""
        self.job_tracker = None

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

    def _create_progress_bar(self, total_steps: int, text: str = ""):
        """
        Create a ComfyUI progress bar with optional weighted job ETA support.

        Args:
            total_steps: Number of denoising steps expected for this generation call
            text: Input text for this generation call (kept for API parity)
        """
        del text  # Progress total is step-accurate for Echo-TTS.
        try:
            import time
            import comfy.utils

            class TimedProgressBar:
                """ComfyUI progress wrapper that also provides job-level timing helpers."""

                def __init__(self, total: int, tracker: Optional[Dict[str, Any]]):
                    self.total = max(1, int(total))
                    self.current = 0
                    self.start_time = time.time()
                    self.tracker = tracker
                    self.wrapped = comfy.utils.ProgressBar(self.total)
                    if self.tracker is not None:
                        self.tracker["current_block_total_steps"] = self.total
                        self.tracker["current_block_steps"] = 0

                def update(self, delta: int = 1):
                    if delta <= 0:
                        return
                    remaining = max(self.total - self.current, 0)
                    step = min(int(delta), remaining)
                    if step <= 0:
                        return
                    self.current += step
                    self.wrapped.update(step)
                    if self.tracker is not None:
                        self.tracker["current_block_steps"] = self.current

                def get_job_elapsed(self):
                    if self.tracker is not None:
                        return time.time() - self.tracker["start_time"]
                    return time.time() - self.start_time

                def get_job_remaining_str(self):
                    if self.tracker is None:
                        return None
                    total_text = self.tracker.get("total_text", 0)
                    if total_text <= 0:
                        return None

                    elapsed = time.time() - self.tracker["start_time"]
                    if elapsed <= 0:
                        return None

                    current_block_text = self.tracker.get("current_block_text", 0)
                    current_total_steps = max(self.tracker.get("current_block_total_steps", self.total), 1)
                    current_ratio = min(self.current / current_total_steps, 1.0)

                    effective_text_done = self.tracker.get("text_completed", 0) + (current_block_text * current_ratio)
                    if effective_text_done <= 0:
                        return None

                    remaining_text = max(total_text - effective_text_done, 0)
                    if remaining_text <= 0:
                        return None

                    seconds_per_text = elapsed / effective_text_done
                    remaining_seconds = remaining_text * seconds_per_text

                    if remaining_seconds < 60:
                        return f"~{remaining_seconds:.0f}s left"
                    return f"~{remaining_seconds / 60:.1f}m left"

            return TimedProgressBar(total_steps, self.job_tracker)
        except (ImportError, AttributeError):
            return None

    @torch.inference_mode()
    def _sample_euler_cfg_independent_guidances_with_progress(
        self,
        model,
        speaker_latent: torch.Tensor,
        speaker_mask: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        rng_seed: int,
        num_steps: int,
        cfg_scale_text: float,
        cfg_scale_speaker: float,
        cfg_min_t: float,
        cfg_max_t: float,
        truncation_factor: Optional[float],
        rescale_k: Optional[float],
        rescale_sigma: Optional[float],
        speaker_kv_scale: Optional[float],
        speaker_kv_max_layers: Optional[int],
        speaker_kv_min_t: Optional[float],
        sequence_length: Optional[int] = None,
        progress_bar=None,
    ) -> torch.Tensor:
        """Echo-TTS sampler equivalent with ComfyUI progress updates per denoise step."""
        from echo_tts.inference import _multiply_kv_cache, _concat_kv_caches, _temporal_score_rescale

        try:
            import comfy.model_management as model_management
        except Exception:
            model_management = None

        if sequence_length is None:
            sequence_length = 640

        init_scale = 0.999
        num_steps = max(1, int(num_steps))
        device, dtype = model.device, model.dtype
        batch_size = text_input_ids.shape[0]
        rng = torch.Generator(device=device).manual_seed(rng_seed)
        t_schedule = torch.linspace(1.0, 0.0, num_steps + 1, device=device) * init_scale

        text_mask_uncond = torch.zeros_like(text_mask)
        speaker_mask_uncond = torch.zeros_like(speaker_mask)

        kv_text_cond = model.get_kv_cache_text(text_input_ids, text_mask)
        kv_speaker_cond = model.get_kv_cache_speaker(speaker_latent.to(dtype))

        if speaker_kv_scale is not None:
            _multiply_kv_cache(kv_speaker_cond, speaker_kv_scale, speaker_kv_max_layers)

        kv_text_full = _concat_kv_caches(kv_text_cond, kv_text_cond, kv_text_cond)
        kv_speaker_full = _concat_kv_caches(kv_speaker_cond, kv_speaker_cond, kv_speaker_cond)

        full_text_mask = torch.cat([text_mask, text_mask_uncond, text_mask], dim=0)
        full_speaker_mask = torch.cat([speaker_mask, speaker_mask, speaker_mask_uncond], dim=0)

        x_t = torch.randn((batch_size, sequence_length, 80), device=device, dtype=torch.float32, generator=rng)
        if truncation_factor is not None:
            x_t = x_t * truncation_factor

        for i in range(num_steps):
            if model_management is not None and getattr(model_management, "interrupt_processing", False):
                raise InterruptedError("Echo-TTS generation interrupted by user")

            t, t_next = t_schedule[i], t_schedule[i + 1]
            has_cfg = ((t >= cfg_min_t) * (t <= cfg_max_t)).item()

            if has_cfg:
                v_cond, v_uncond_text, v_uncond_speaker = model(
                    x=torch.cat([x_t, x_t, x_t], dim=0).to(dtype),
                    t=(torch.ones((batch_size * 3,), device=device) * t).to(dtype),
                    text_mask=full_text_mask,
                    speaker_mask=full_speaker_mask,
                    kv_cache_text=kv_text_full,
                    kv_cache_speaker=kv_speaker_full,
                ).float().chunk(3, dim=0)
                v_pred = v_cond + cfg_scale_text * (v_cond - v_uncond_text) + cfg_scale_speaker * (v_cond - v_uncond_speaker)
            else:
                v_pred = model(
                    x=x_t.to(dtype),
                    t=(torch.ones((batch_size,), device=device) * t).to(dtype),
                    text_mask=text_mask,
                    speaker_mask=speaker_mask,
                    kv_cache_text=kv_text_cond,
                    kv_cache_speaker=kv_speaker_cond,
                ).float()

            if rescale_k is not None and rescale_sigma is not None:
                v_pred = _temporal_score_rescale(v_pred, x_t, t, rescale_k, rescale_sigma)

            if (
                speaker_kv_scale is not None
                and speaker_kv_min_t is not None
                and t_next < speaker_kv_min_t
                and t >= speaker_kv_min_t
            ):
                _multiply_kv_cache(kv_speaker_cond, 1.0 / speaker_kv_scale, speaker_kv_max_layers)
                kv_speaker_full = _concat_kv_caches(kv_speaker_cond, kv_speaker_cond, kv_speaker_cond)

            x_t = x_t + v_pred * (t_next - t)

            if progress_bar is not None:
                try:
                    progress_bar.update(1)
                except Exception:
                    pass

        return x_t

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

        progress_bar = self._create_progress_bar(params["num_steps"], text)

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
            from echo_tts.inference import sample_euler_cfg_independent_guidances
            sample_fn = partial(
                self._sample_euler_cfg_independent_guidances_with_progress,
                progress_bar=progress_bar,
                **self._filter_kwargs(sample_euler_cfg_independent_guidances, sampler_kwargs),
            )
        except Exception:
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

        if progress_bar is not None and hasattr(progress_bar, "total") and hasattr(progress_bar, "current"):
            remaining = progress_bar.total - progress_bar.current
            if remaining > 0:
                progress_bar.update(remaining)

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
