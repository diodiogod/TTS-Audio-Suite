"""SRT timing/orchestration for TADA using the normal TADA text processor."""

import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import comfy.model_management as model_management
import torch

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.adapters.tada_adapter import TadaEngineAdapter


class TadaSRTProcessor:
    """Generate each subtitle through :class:`TadaProcessor`, then assemble timing."""

    SAMPLE_RATE = 24000

    def __init__(self, node_instance, config: Dict[str, Any]):
        self.node_instance = node_instance
        self.config = config.copy() if config else {}
        self.adapter = TadaEngineAdapter(self.config)
        self._tts_processor = None
        self.srt_available = False
        self.SRTParser = None
        self._load_srt_modules()

    def _load_srt_modules(self) -> None:
        try:
            from utils.system.import_manager import import_manager

            success, modules, message = import_manager.import_srt_modules()
            if not success:
                print(f"⚠️ TADA SRT: SRT module not available: {message}")
                return
            self.SRTParser = modules.get("SRTParser")
            self.srt_available = self.SRTParser is not None
        except Exception as exc:
            print(f"⚠️ TADA SRT: Failed to load SRT modules: {exc}")

    @property
    def processor(self):
        if self._tts_processor is None:
            processor_path = os.path.join(current_dir, "tada_processor.py")
            spec = importlib.util.spec_from_file_location("tada_processor_module", processor_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._tts_processor = module.TadaProcessor(self.adapter, self.config)
        return self._tts_processor

    def update_config(self, new_config: Dict[str, Any]) -> None:
        self.config = new_config.copy() if new_config else {}
        self.adapter.update_config(self.config)
        if self._tts_processor is not None:
            self._tts_processor.update_config(self.config)

    @staticmethod
    def _check_interrupt(index: Optional[int] = None, total: Optional[int] = None) -> None:
        if model_management.interrupt_processing:
            location = f" at subtitle {index + 1}/{total}" if index is not None and total else ""
            raise InterruptedError(f"TADA SRT generation interrupted{location}")

    def process_srt_content(
        self,
        srt_content: str,
        voice_mapping: Dict[str, Any],
        seed: int,
        timing_mode: str,
        timing_params: Dict[str, Any],
        enable_audio_cache: bool = True,
    ) -> Tuple[Dict[str, Any], str, str, str]:
        if not self.srt_available:
            raise ImportError("TADA SRT support is unavailable because the shared SRT parser did not load")

        self._check_interrupt()
        subtitles = self.SRTParser().parse_srt_content(srt_content, allow_overlaps=True)

        from utils.timing.overlap_detection import SRTOverlapHandler

        has_overlaps = SRTOverlapHandler.detect_overlaps(subtitles)
        current_mode, mode_switched = SRTOverlapHandler.handle_smart_natural_fallback(
            timing_mode, has_overlaps, "TADA SRT"
        )
        print(f"🚀 TADA SRT: Processing {len(subtitles)} subtitles in {current_mode} mode")

        audio_segments, adjustments = self._generate_subtitles(
            subtitles, voice_mapping, seed, enable_audio_cache
        )
        self._check_interrupt()
        final_audio, final_adjustments, stretch_method = self._assemble(
            audio_segments, subtitles, current_mode, timing_params
        )
        if final_adjustments is not None:
            adjustments = final_adjustments

        from utils.timing.reporting import SRTReportGenerator

        reporter = SRTReportGenerator()
        timing_report = reporter.generate_timing_report(
            subtitles,
            adjustments,
            current_mode,
            has_overlaps,
            mode_switched,
            timing_mode if mode_switched else None,
            stretch_method,
        )
        adjusted_srt = reporter.generate_adjusted_srt_string(subtitles, adjustments, current_mode)

        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0).unsqueeze(0)
        elif final_audio.dim() == 2:
            final_audio = final_audio.unsqueeze(0)

        duration = final_audio.shape[-1] / self.SAMPLE_RATE
        mode_info = current_mode
        if mode_switched:
            mode_info = f"{current_mode} (switched from {timing_mode} due to overlaps)"
        info = (
            f"Generated {duration:.1f}s TADA SRT-timed audio from {len(subtitles)} subtitles "
            f"using {mode_info} mode"
        )
        return {"waveform": final_audio.cpu().float(), "sample_rate": self.SAMPLE_RATE}, info, timing_report, adjusted_srt

    def _generate_subtitles(
        self,
        subtitles: List,
        voice_mapping: Dict[str, Any],
        seed: int,
        enable_audio_cache: bool,
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        audio_segments: List[torch.Tensor] = []
        adjustments: List[Dict[str, Any]] = []
        try:
            from comfy.utils import ProgressBar

            progress = ProgressBar(len(subtitles))
        except Exception:
            progress = None

        for index, subtitle in enumerate(subtitles):
            self._check_interrupt(index, len(subtitles))
            text = (subtitle.text or "").strip()
            target_duration = subtitle.end_time - subtitle.start_time

            if text:
                print(f"📖 TADA SRT Subtitle {index + 1}/{len(subtitles)}")
                records = self.processor.process_text(
                    text=text,
                    voice_mapping=voice_mapping,
                    seed=seed + index,
                    enable_chunking=False,
                    enable_audio_cache=enable_audio_cache,
                    apply_edit_postprocessing=True,
                )
                if len(records) > 1:
                    audio, _ = self.processor.combine_audio_segments(
                        records, method="auto", silence_ms=0, original_text=text, return_info=True
                    )
                elif records:
                    audio = records[0]["waveform"]
                else:
                    audio = torch.zeros(1, 0)
            else:
                audio = torch.zeros(1, max(0, int(target_duration * self.SAMPLE_RATE)))

            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            elif audio.dim() == 3:
                audio = audio.squeeze(0)
            audio = audio.detach().cpu().float()
            audio_segments.append(audio)

            natural_duration = audio.shape[-1] / self.SAMPLE_RATE
            stretch_factor = target_duration / natural_duration if natural_duration > 0 else 1.0
            adjustments.append(
                {
                    "index": index,
                    "segment_index": index,
                    "sequence": subtitle.sequence,
                    "natural_duration": natural_duration,
                    "target_start": subtitle.start_time,
                    "target_end": subtitle.end_time,
                    "target_duration": target_duration,
                    "start_time": subtitle.start_time,
                    "end_time": subtitle.end_time,
                    "stretch_factor": stretch_factor,
                    "needs_stretching": abs(stretch_factor - 1.0) > 0.05,
                    "stretch_type": (
                        "compress" if stretch_factor < 1.0 else "expand" if stretch_factor > 1.0 else "none"
                    ),
                    "adjustment": natural_duration - target_duration,
                    "adjusted_start": subtitle.start_time,
                    "adjusted_end": subtitle.end_time,
                    "adjusted_duration": natural_duration,
                }
            )
            print(
                f"✅ TADA Subtitle {index + 1}/{len(subtitles)}: "
                f"{natural_duration:.2f}s (target {target_duration:.2f}s)"
            )
            if progress is not None:
                progress.update(1)

        return audio_segments, adjustments

    def _assemble(self, audio_segments, subtitles, timing_mode, timing_params):
        self._check_interrupt()
        if timing_mode == "stretch_to_fit":
            from engines.chatterbox.audio_timing import TimedAudioAssembler

            assembler = TimedAudioAssembler(self.SAMPLE_RATE)
            target_timings = [(subtitle.start_time, subtitle.end_time) for subtitle in subtitles]
            final_audio, stretch_method = assembler.assemble_timed_audio(
                audio_segments,
                target_timings,
                fade_duration=timing_params.get("fade_for_StretchToFit", 0.01),
            )
            return final_audio, None, stretch_method

        from utils.timing.assembly import AudioAssemblyEngine

        assembler = AudioAssemblyEngine(self.SAMPLE_RATE)
        if timing_mode == "pad_with_silence":
            return assembler.assemble_with_overlaps(audio_segments, subtitles, torch.device("cpu")), None, None

        from utils.timing.engine import TimingEngine

        timing_engine = TimingEngine(self.SAMPLE_RATE)
        if timing_mode == "concatenate":
            adjustments = timing_engine.calculate_concatenation_adjustments(audio_segments, subtitles)
            audio = assembler.assemble_concatenation(
                audio_segments, timing_params.get("fade_for_StretchToFit", 0.01)
            )
            return audio, adjustments, None

        adjustments, processed = timing_engine.calculate_smart_timing_adjustments(
            audio_segments,
            subtitles,
            timing_params.get("timing_tolerance", 2.0),
            timing_params.get("max_stretch_ratio", 1.0),
            timing_params.get("min_stretch_ratio", 0.5),
            torch.device("cpu"),
        )
        audio = assembler.assemble_smart_natural(
            audio_segments, processed, adjustments, subtitles, torch.device("cpu")
        )
        return audio, adjustments, None
