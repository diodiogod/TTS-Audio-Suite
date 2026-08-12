"""SRT timing orchestration for audio.cpp with a response-defined sample rate."""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from utils.system.import_manager import import_manager
from utils.timing.assembly import AudioAssemblyEngine
from utils.timing.engine import TimingEngine
from utils.timing.overlap_detection import SRTOverlapHandler
from utils.timing.reporting import SRTReportGenerator


def _processor_class():
    """Load by path because this project also has a top-level ``nodes.py`` module."""
    path = os.path.join(os.path.dirname(__file__), "audio_cpp_processor.py")
    spec = importlib.util.spec_from_file_location("audio_cpp_processor_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load audio.cpp processor from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AudioCppProcessor


def _adapter_class():
    """Load directly so unrelated optional adapters are not imported eagerly."""
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "engines", "adapters", "audio_cpp_adapter.py")
    )
    spec = importlib.util.spec_from_file_location("audio_cpp_adapter_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load audio.cpp adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AudioCppEngineAdapter


class AudioCppSRTProcessor:
    """Generate one subtitle cue at a time and assemble it on the SRT timeline."""

    def __init__(self, node_instance: Any, config: Optional[Dict[str, Any]] = None):
        self.node_instance = node_instance
        self.config = dict(config or {})
        self.adapter = _adapter_class()(self.config)
        self._processor = _processor_class()(self.adapter, self.config)
        success, modules, message = import_manager.import_srt_modules()
        if not success or modules.get("SRTParser") is None:
            raise ImportError(f"audio.cpp SRT unavailable: {message}")
        self.SRTParser = modules["SRTParser"]

    @property
    def processor(self) -> Any:
        return self._processor

    @property
    def sample_rate(self) -> Optional[int]:
        return self.processor.sample_rate

    def update_config(self, config: Optional[Dict[str, Any]]) -> None:
        self.config = dict(config or {})
        self.processor.update_config(self.config)

    @staticmethod
    def _check_interrupt(index: Optional[int] = None, total: Optional[int] = None) -> None:
        try:
            import comfy.model_management as model_management

            if getattr(model_management, "interrupt_processing", False) is True:
                location = f" at subtitle {index + 1}/{total}" if index is not None else ""
                raise InterruptedError(f"audio.cpp SRT generation interrupted{location}")
        except ImportError:
            return

    @staticmethod
    def _adjustment(index: int, subtitle: Any, audio: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
        natural = audio.shape[-1] / sample_rate
        target = float(subtitle.duration)
        ratio = target / natural if natural > 0 else 1.0
        return {
            "index": index,
            "segment_index": index,
            "sequence": subtitle.sequence,
            "natural_duration": natural,
            "target_start": subtitle.start_time,
            "target_end": subtitle.end_time,
            "target_duration": target,
            "start_time": subtitle.start_time,
            "end_time": subtitle.end_time,
            "stretch_factor": ratio,
            "needs_stretching": abs(ratio - 1.0) > 0.05,
            "stretch_type": "compress" if ratio < 1 else "expand" if ratio > 1 else "none",
            "adjustment": natural - target,
            "adjusted_start": subtitle.start_time,
            "adjusted_end": subtitle.end_time,
            "adjusted_duration": natural,
        }

    def process_srt_content(
        self,
        srt_content: str,
        voice_mapping: Optional[Dict[str, Any]],
        seed: int,
        timing_mode: str,
        timing_params: Optional[Dict[str, Any]],
        enable_audio_cache: bool = True,
    ) -> Tuple[Dict[str, Any], str, str, str]:
        self._check_interrupt()
        subtitles = self.SRTParser().parse_srt_content(srt_content, allow_overlaps=True)
        if not subtitles:
            raise ValueError("audio.cpp SRT input contains no subtitles")

        has_overlaps = SRTOverlapHandler.detect_overlaps(subtitles)
        active_mode, switched = SRTOverlapHandler.handle_smart_natural_fallback(
            timing_mode, has_overlaps, "audio.cpp SRT"
        )
        self.processor.reset_sample_rate()
        audio_segments: List[Optional[torch.Tensor]] = []
        for index, subtitle in enumerate(subtitles):
            self._check_interrupt(index, len(subtitles))
            text = str(subtitle.text or "").strip()
            if not text:
                audio_segments.append(None)
                continue
            records = self.processor.process_text(
                text=text,
                voice_mapping=voice_mapping or {},
                seed=int(seed) + index,
                enable_chunking=False,
                enable_audio_cache=enable_audio_cache,
                apply_edit_postprocessing=True,
                show_text_logging=True,
                reset_sample_rate=False,
            )
            if not records:
                raise RuntimeError(f"audio.cpp produced no audio for subtitle {index + 1}")
            audio = self.processor.combine_audio_segments(
                records, method="auto", silence_ms=0, original_text=text
            )
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            elif audio.dim() == 3 and audio.shape[0] == 1:
                audio = audio.squeeze(0)
            audio_segments.append(audio.detach().to(device="cpu", dtype=torch.float32))

        sample_rate = self.processor.sample_rate
        if sample_rate is None:
            raise ValueError("audio.cpp could not determine a sample rate from the SRT content")
        completed_segments: List[torch.Tensor] = []
        for subtitle, audio in zip(subtitles, audio_segments):
            if audio is None:
                audio = torch.zeros(1, int(float(subtitle.duration) * sample_rate), dtype=torch.float32)
            completed_segments.append(audio)

        adjustments = [
            self._adjustment(index, subtitle, completed_segments[index], sample_rate)
            for index, subtitle in enumerate(subtitles)
        ]
        self._check_interrupt()
        final_audio, replacement, stretch_method = self._assemble(
            completed_segments, subtitles, active_mode, dict(timing_params or {}), sample_rate
        )
        if replacement is not None:
            adjustments = replacement

        reporter = SRTReportGenerator()
        report = reporter.generate_timing_report(
            subtitles,
            adjustments,
            active_mode,
            has_overlaps,
            switched,
            timing_mode if switched else None,
            stretch_method,
        )
        adjusted_srt = reporter.generate_adjusted_srt_string(subtitles, adjustments, active_mode)
        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0).unsqueeze(0)
        elif final_audio.dim() == 2:
            final_audio = final_audio.unsqueeze(0)
        duration = final_audio.shape[-1] / sample_rate
        mode_info = f"{active_mode} (switched from {timing_mode})" if switched else active_mode
        info = (
            f"Generated {duration:.1f}s audio.cpp SRT audio from {len(subtitles)} subtitles "
            f"using {mode_info} mode at {sample_rate} Hz"
        )
        return {"waveform": final_audio, "sample_rate": sample_rate}, info, report, adjusted_srt

    @staticmethod
    def _assemble(
        audio_segments: List[torch.Tensor],
        subtitles: List[Any],
        mode: str,
        params: Dict[str, Any],
        sample_rate: int,
    ):
        fade = params.get("fade_for_StretchToFit", 0.01)
        if mode == "stretch_to_fit":
            from engines.chatterbox.audio_timing import TimedAudioAssembler

            assembler = TimedAudioAssembler(sample_rate)
            audio, method = assembler.assemble_timed_audio(
                audio_segments,
                [(item.start_time, item.end_time) for item in subtitles],
                fade_duration=fade,
            )
            return audio, None, method

        assembler = AudioAssemblyEngine(sample_rate)
        if mode == "pad_with_silence":
            audio = assembler.assemble_with_overlaps(audio_segments, subtitles, torch.device("cpu"))
            return audio, None, None

        timing = TimingEngine(sample_rate)
        if mode == "concatenate":
            replacements = timing.calculate_concatenation_adjustments(audio_segments, subtitles)
            audio = assembler.assemble_concatenation(audio_segments, fade)
            return audio, replacements, None

        replacements, processed = timing.calculate_smart_timing_adjustments(
            audio_segments,
            subtitles,
            params.get("timing_tolerance", 2.0),
            params.get("max_stretch_ratio", 1.0),
            params.get("min_stretch_ratio", 0.5),
            torch.device("cpu"),
        )
        audio = assembler.assemble_smart_natural(
            audio_segments, processed, replacements, subtitles, torch.device("cpu")
        )
        return audio, replacements, None


AudioCppSubtitleProcessor = AudioCppSRTProcessor
AudioCPPSRTProcessor = AudioCppSRTProcessor
