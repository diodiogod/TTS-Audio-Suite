"""
Qwen3-ASR adapter for unified ASR pipeline.
Uses qwen_asr package (transformers backend) and optional forced aligner.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import comfy.model_management as model_management
import comfy.utils
import time

from utils.asr.types import ASRRequest, ASRResult, ASRSegment, ASRWord


class Qwen3ASREngineAdapter:
    def __init__(self, engine_data: Dict[str, Any]):
        self.engine_data = engine_data
        self.config = engine_data.get("config", engine_data)

    def _resolve_precision(self, precision: str) -> torch.dtype:
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        if precision == "fp32":
            return torch.float32
        # auto
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
            return torch.bfloat16 if major >= 8 else torch.float16
        return torch.float32

    def _get_model(self, req: ASRRequest):
        from utils.models.unified_model_interface import unified_model_interface, ModelLoadConfig
        from utils.device import resolve_torch_device

        device = self.config.get("device", "auto")
        if model_management.cpu_mode():
            device = "cpu"
        elif device != "cpu":
            device = resolve_torch_device(device)

        # Infer ASR model variant from existing Qwen3 model_size setting
        model_size = self.config.get("model_size", "1.7B")
        model_name = "Qwen3-ASR-0.6B" if model_size == "0.6B" else "Qwen3-ASR-1.7B"
        model_id = model_name

        precision = self.config.get("dtype", "auto")

        attn = self.config.get("attn_implementation", "auto")
        if attn == "auto":
            flash_attn = False
        else:
            flash_attn = attn == "flash_attention_2"

        forced_aligner_enabled = self.config.get("asr_use_forced_aligner", False)

        forced_aligner_model = "Qwen3-ForcedAligner-0.6B" if forced_aligner_enabled else ""

        additional_params = {
            "precision": precision,
            "attn_implementation": "flash_attention_2" if flash_attn else "sdpa",
            "max_new_tokens": int(self.config.get("max_new_tokens", 256)),
        }

        if forced_aligner_enabled:
            additional_params["forced_aligner"] = forced_aligner_model
            additional_params["forced_aligner_kwargs"] = None

        config = ModelLoadConfig(
            engine_name="qwen3_asr",
            model_type="asr",
            model_name=model_name,
            model_path=model_id,
            device=device,
            additional_params=additional_params,
        )

        return unified_model_interface.load_model(config)

    def _prepare_audio(self, audio: Dict[str, Any]) -> torch.Tensor:
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if waveform.ndim == 3:
            waveform = waveform[0]
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform[0]

        if sample_rate != 16000:
            w = waveform.view(1, 1, -1)
            w = F.interpolate(w, size=int(w.shape[-1] * 16000 / sample_rate), mode="linear", align_corners=False)
            waveform = w.view(-1)
        return waveform

    def transcribe(self, req: ASRRequest) -> ASRResult:
        model_management.throw_exception_if_processing_interrupted()

        model = self._get_model(req)
        waveform = self._prepare_audio(req.audio)
        sample_rate = 16000

        language = None if not req.language or req.language.lower() == "auto" else req.language
        return_ts = req.timestamps == "word" or req.use_forced_aligner or self.config.get("asr_use_forced_aligner", False)

        chunk_size = int(req.chunk_size)
        overlap = int(req.overlap)

        segments: List[ASRSegment] = []
        full_text_parts: List[str] = []
        wav_np = waveform.cpu().numpy()
        total_samples = len(wav_np)
        progress_bar = None

        if chunk_size > 0:
            chunk_samples = chunk_size * sample_rate
            overlap_samples = overlap * sample_rate
            step_samples = max(1, chunk_samples - overlap_samples)
            num_chunks = max(1, int((total_samples - overlap_samples + step_samples - 1) / step_samples))

            if getattr(model, "backend", None) == "transformers":
                try:
                    from utils.asr.progress_callback import ASRProgressStreamer
                    total_tokens = int(self.config.get("max_new_tokens", 256)) * num_chunks
                    progress_bar = comfy.utils.ProgressBar(total_tokens)
                    model._streamer_factory = lambda max_new_tokens, pb: ASRProgressStreamer(
                        max_new_tokens or int(self.config.get("max_new_tokens", 256)),
                        pb,
                        label="ASR",
                    )
                    model._progress_bar = progress_bar
                except Exception:
                    pass

            # Chunk-level fallback progress (matches TTS-style console line)
            chunk_progress_bar = comfy.utils.ProgressBar(num_chunks)
            start_time = time.time()
            last_print = 0.0

            for i in range(num_chunks):
                model_management.throw_exception_if_processing_interrupted()
                start = i * step_samples
                end = min(start + chunk_samples, total_samples)
                chunk_np = wav_np[start:end]
                results = model.transcribe(
                    audio=[(chunk_np, sample_rate)],
                    language=language,
                    return_time_stamps=return_ts,
                )
                res = results[0]
                full_text_parts.append(res.text)
                chunk_offset = start / sample_rate

                if return_ts and hasattr(res, "time_stamps") and res.time_stamps:
                    for ts in res.time_stamps:
                        word = ASRWord(
                            start=ts.start_time + chunk_offset,
                            end=ts.end_time + chunk_offset,
                            text=ts.text,
                        )
                        seg = ASRSegment(
                            start=word.start,
                            end=word.end,
                            text=word.text,
                            words=[word],
                        )
                        segments.append(seg)

                try:
                    chunk_progress_bar.update(1)
                    now = time.time()
                    if now - last_print >= 0.5 or (i + 1) == num_chunks:
                        elapsed = now - start_time
                        its = (i + 1) / elapsed if elapsed > 0 else 0.0
                        eta = (num_chunks - (i + 1)) / its if its > 0 else 0.0
                        bar_width = 12
                        filled = int(((i + 1) / num_chunks) * bar_width)
                        bar = f"[{'█' * filled}{'░' * (bar_width - filled)}]"
                        print(f"\r   Progress: {bar} {i+1}/{num_chunks} | {its:.1f} it/s | {elapsed:.0f}s | ETA {eta:.0f}s", end="", flush=True)
                        last_print = now
                except Exception:
                    pass

            try:
                elapsed = time.time() - start_time
                avg = (num_chunks / elapsed) if elapsed > 0 else 0.0
                print(f"\r   Complete: {num_chunks} chunks in {elapsed:.1f}s (avg {avg:.1f} it/s)" + " " * 20)
            except Exception:
                pass

        else:
            if getattr(model, "backend", None) == "transformers":
                try:
                    from utils.asr.progress_callback import ASRProgressStreamer
                    total_tokens = int(self.config.get("max_new_tokens", 256))
                    progress_bar = comfy.utils.ProgressBar(total_tokens)
                    model._streamer_factory = lambda max_new_tokens, pb: ASRProgressStreamer(
                        max_new_tokens or total_tokens,
                        pb,
                        label="ASR",
                    )
                    model._progress_bar = progress_bar
                except Exception:
                    pass
            results = model.transcribe(
                audio=[(wav_np, sample_rate)],
                language=language,
                return_time_stamps=return_ts,
            )
            res = results[0]
            full_text_parts.append(res.text)
            if return_ts and hasattr(res, "time_stamps") and res.time_stamps:
                for ts in res.time_stamps:
                    word = ASRWord(start=ts.start_time, end=ts.end_time, text=ts.text)
                    segments.append(ASRSegment(start=word.start, end=word.end, text=word.text, words=[word]))

        text = " ".join([t for t in full_text_parts if t]).strip()

        return ASRResult(text=text, language=getattr(results[0], "language", None), segments=segments, raw=None)
