"""
Granite ASR adapter for unified ASR pipeline.
Uses Granite speech for transcription/translation and optionally Qwen forced aligner for timestamps.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import comfy.model_management as model_management
import comfy.utils
import time

from engines.granite_asr.prompting import DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE
from utils.asr.types import ASRRequest, ASRResult, ASRSegment, ASRWord


class GraniteASREngineAdapter:
    GRANITE_LANGUAGES = {
        "english": "English",
        "french": "French",
        "german": "German",
        "spanish": "Spanish",
        "portuguese": "Portuguese",
        "japanese": "Japanese",
    }

    QWEN_ALIGNER_LANGUAGES = {
        "Chinese",
        "English",
        "Cantonese",
        "French",
        "German",
        "Italian",
        "Japanese",
        "Korean",
        "Portuguese",
        "Russian",
        "Spanish",
    }

    GENERIC_SPACE_ALIGNER_LANGUAGE = "English"

    def __init__(self, engine_data: Dict[str, Any]):
        self.engine_data = engine_data
        self.config = engine_data.get("config", engine_data)

    def _normalize_language(self, language: Optional[str]) -> Optional[str]:
        if not language or str(language).lower() == "auto":
            return None
        return self.GRANITE_LANGUAGES.get(str(language).strip().lower())

    def _get_model(self):
        from utils.models.unified_model_interface import unified_model_interface, ModelLoadConfig
        from utils.device import resolve_torch_device

        device = self.config.get("device", "auto")
        if model_management.cpu_mode():
            device = "cpu"
        elif device != "cpu":
            device = resolve_torch_device(device)

        config = ModelLoadConfig(
            engine_name="granite_asr",
            model_type="asr",
            model_name=self.config.get("model_name", "granite-4.0-1b-speech"),
            model_path=self.config.get("model_name", "granite-4.0-1b-speech"),
            device=device,
            additional_params={
                "precision": self.config.get("dtype", "auto"),
                "attn_implementation": self.config.get("attn_implementation", "auto"),
            },
        )
        return unified_model_interface.load_model(config)

    def _get_forced_aligner(self):
        from utils.models.unified_model_interface import unified_model_interface, ModelLoadConfig
        from utils.device import resolve_torch_device

        device = self.config.get("device", "auto")
        if model_management.cpu_mode():
            device = "cpu"
        elif device != "cpu":
            device = resolve_torch_device(device)

        config = ModelLoadConfig(
            engine_name="qwen3_asr",
            model_type="aligner",
            model_name="Qwen3-ForcedAligner-0.6B",
            model_path="Qwen3-ForcedAligner-0.6B",
            device=device,
            additional_params={
                "precision": self.config.get("dtype", "auto"),
                "attn_implementation": self.config.get("attn_implementation", "auto"),
            },
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

    def _alignment_requested(self, req: ASRRequest) -> bool:
        return req.timestamps == "word" or req.use_forced_aligner

    def _can_use_aligner(self, req: ASRRequest, normalized_language: Optional[str], warnings: List[str]) -> bool:
        if not self._alignment_requested(req):
            return False

        if req.task != "transcribe":
            warnings.append("Granite timestamps skipped: Qwen forced aligner only works for transcription, not translation.")
            return False

        if not self.config.get("asr_use_forced_aligner", False):
            warnings.append("Granite timestamps skipped: Qwen forced aligner is disabled on the Granite engine node.")
            return False

        if normalized_language and normalized_language not in self.QWEN_ALIGNER_LANGUAGES:
            warnings.append(f"Granite timestamps skipped: Qwen forced aligner does not support '{normalized_language}'.")
            return False

        return True

    def _append_once(self, messages: List[str], message: str) -> None:
        if message not in messages:
            messages.append(message)

    def _contains_japanese_script(self, text: str) -> bool:
        for ch in text:
            code = ord(ch)
            if 0x3040 <= code <= 0x309F:  # Hiragana
                return True
            if 0x30A0 <= code <= 0x30FF:  # Katakana
                return True
            if 0x31F0 <= code <= 0x31FF:  # Katakana Phonetic Extensions
                return True
            if 0xFF66 <= code <= 0xFF9D:  # Halfwidth Katakana
                return True
            if 0x3400 <= code <= 0x4DBF:  # CJK Unified Ideographs Extension A
                return True
            if 0x4E00 <= code <= 0x9FFF:  # CJK Unified Ideographs
                return True
            if 0xF900 <= code <= 0xFAFF:  # CJK Compatibility Ideographs
                return True
        return False

    def _resolve_aligner_language(
        self,
        normalized_language: Optional[str],
        transcript: str,
        notes: List[str],
    ) -> str:
        if normalized_language:
            return normalized_language

        if self._contains_japanese_script(transcript):
            self._append_once(
                notes,
                "Granite Auto alignment used a Japanese-script heuristic, so Qwen forced aligner ran in Japanese mode. This is tokenizer routing for alignment, not Granite native language detection.",
            )
            return "Japanese"

        self._append_once(
            notes,
            "Granite Auto alignment used Qwen's generic space-delimited tokenizer path. 'English' here is only the aligner mode for non-Japanese Granite languages, not detected spoken language.",
        )
        return self.GENERIC_SPACE_ALIGNER_LANGUAGE

    def _align_words(
        self,
        aligner,
        chunk_audio,
        transcript: str,
        language: str,
        chunk_offset: float,
        warnings: List[str],
    ) -> List[ASRSegment]:
        if not transcript.strip():
            return []

        try:
            results = aligner.align(audio=[(chunk_audio, 16000)], text=[transcript], language=[language])
            if not results:
                return []

            segments: List[ASRSegment] = []
            for item in results[0]:
                word = ASRWord(
                    start=float(item.start_time) + chunk_offset,
                    end=float(item.end_time) + chunk_offset,
                    text=str(item.text),
                )
                segments.append(
                    ASRSegment(
                        start=word.start,
                        end=word.end,
                        text=word.text,
                        words=[word],
                    )
                )
            return segments
        except Exception as e:
            warnings.append(f"Granite timestamps skipped: Qwen forced aligner failed ({e}).")
            return []

    def _run_chunk(
        self,
        runtime,
        chunk_np,
        req: ASRRequest,
        normalized_language: Optional[str],
        target_language: Optional[str],
        translate_instruction_override: Optional[str],
    ) -> Tuple[str, Optional[str]]:
        generation_kwargs = {
            "max_new_tokens": int(self.config.get("max_new_tokens", req.max_new_tokens or 200)),
            "do_sample": bool(self.config.get("do_sample", False)),
            "num_beams": int(self.config.get("num_beams", 1)),
            "temperature": float(self.config.get("temperature", 1.0)),
            "top_k": int(self.config.get("top_k", 50)),
            "top_p": float(self.config.get("top_p", 1.0)),
            "repetition_penalty": float(self.config.get("repetition_penalty", 1.0)),
            "length_penalty": float(self.config.get("length_penalty", 1.0)),
            "no_repeat_ngram_size": int(self.config.get("no_repeat_ngram_size", 0)),
            "early_stopping": bool(self.config.get("early_stopping", False)),
        }
        result = runtime.transcribe(
            chunk_np,
            task=req.task,
            language=normalized_language,
            target_language=target_language,
            translate_instruction_override=translate_instruction_override,
            generation_kwargs=generation_kwargs,
        )
        return result.get("text", ""), result.get("language")

    def transcribe(self, req: ASRRequest) -> ASRResult:
        model_management.throw_exception_if_processing_interrupted()

        runtime = self._get_model()
        waveform = self._prepare_audio(req.audio)
        wav_np = waveform.cpu().numpy()
        sample_rate = 16000

        warnings: List[str] = []
        notes: List[str] = []
        normalized_language = self._normalize_language(req.language)
        target_language = self.config.get("asr_translate_target_language", "English")
        translate_instruction_override = str(
            self.config.get("asr_translate_instruction_override", DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE)
            or DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE
        ).strip()
        if req.language and str(req.language).lower() != "auto" and not normalized_language:
            warnings.append(
                f"Granite received unsupported language hint '{req.language}'. Falling back to Granite auto language handling."
            )
        if req.task == "translate":
            self._append_once(
                notes,
                f"Granite translation target is '{target_language}'. This target is prompt-driven by the Granite engine node, not a native model target-language parameter.",
            )
            if translate_instruction_override != DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE:
                self._append_once(
                    notes,
                    "Granite custom translation instruction override is active on the engine node. This is still wrapped in Granite's chat format, but malformed or weak instructions can fall back to plain transcription.",
                )

        use_aligner = self._can_use_aligner(req, normalized_language, warnings)
        aligner = self._get_forced_aligner() if use_aligner else None

        chunk_size = int(req.chunk_size)
        overlap = int(req.overlap)
        total_samples = len(wav_np)

        segments: List[ASRSegment] = []
        full_text_parts: List[str] = []
        output_language = normalized_language

        if chunk_size > 0:
            chunk_samples = chunk_size * sample_rate
            overlap_samples = overlap * sample_rate
            step_samples = max(1, chunk_samples - overlap_samples)
            num_chunks = max(1, int((total_samples - overlap_samples + step_samples - 1) / step_samples))

            chunk_progress_bar = comfy.utils.ProgressBar(num_chunks)
            start_time = time.time()
            last_print = 0.0

            for i in range(num_chunks):
                model_management.throw_exception_if_processing_interrupted()
                start = i * step_samples
                end = min(start + chunk_samples, total_samples)
                chunk_np = wav_np[start:end]
                text, detected_language = self._run_chunk(
                    runtime,
                    chunk_np,
                    req,
                    normalized_language,
                    target_language,
                    translate_instruction_override,
                )
                if text:
                    full_text_parts.append(text)
                if detected_language and not output_language:
                    output_language = detected_language

                if aligner is not None and text:
                    aligner_language = self._resolve_aligner_language(normalized_language, text, notes)
                    chunk_offset = start / sample_rate
                    segments.extend(
                        self._align_words(
                            aligner=aligner,
                            chunk_audio=chunk_np,
                            transcript=text,
                            language=aligner_language,
                            chunk_offset=chunk_offset,
                            warnings=warnings,
                        )
                    )

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
            text, detected_language = self._run_chunk(
                runtime,
                wav_np,
                req,
                normalized_language,
                target_language,
                translate_instruction_override,
            )
            if text:
                full_text_parts.append(text)
            output_language = detected_language or output_language

            if aligner is not None and text:
                aligner_language = self._resolve_aligner_language(normalized_language, text, notes)
                segments.extend(
                    self._align_words(
                        aligner=aligner,
                        chunk_audio=wav_np,
                        transcript=text,
                        language=aligner_language,
                        chunk_offset=0.0,
                        warnings=warnings,
                    )
                )

        text = " ".join(part for part in full_text_parts if part).strip()
        raw = None
        if warnings or notes:
            raw = {}
            if warnings:
                raw["warnings"] = warnings
            if notes:
                raw["notes"] = notes
        return ASRResult(text=text, language=output_language, segments=segments, raw=raw)
