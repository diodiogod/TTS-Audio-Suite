"""
ZONOS3-Mini SRT processor.

Handles full SRT subtitle processing for ZONOS3-Mini.
"""

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

from engines.adapters.zonos3_mini_adapter import Zonos3MiniEngineAdapter


class Zonos3MiniSRTProcessor:
    """Complete SRT processor for ZONOS3-Mini."""

    SAMPLE_RATE = 44100

    def __init__(self, node_instance, config: Dict[str, Any]):
        self.node_instance = node_instance
        self.config = config.copy() if config else {}
        self.adapter = Zonos3MiniEngineAdapter(self.config)
        self._tts_processor = None
        self.srt_available = False
        self.SRTParser = None
        self.SRTSubtitle = None
        self.SRTParseError = None
        self._load_srt_modules()

    def _load_srt_modules(self):
        try:
            from utils.system.import_manager import import_manager

            success, modules, msg = import_manager.import_srt_modules()
            if not success:
                print(f"⚠️ ZONOS3-Mini SRT: SRT module not available: {msg}")
                return

            self.SRTParser = modules.get("SRTParser")
            self.SRTSubtitle = modules.get("SRTSubtitle")
            self.SRTParseError = modules.get("SRTParseError")
            if self.SRTParser is None:
                print("⚠️ ZONOS3-Mini SRT: SRT parser not available")
                return
            self.srt_available = True
        except Exception as e:
            print(f"⚠️ ZONOS3-Mini SRT: Failed to load SRT modules: {e}")

    @property
    def processor(self):
        if self._tts_processor is None:
            processor_path = os.path.join(current_dir, "zonos3_mini_processor.py")
            spec = importlib.util.spec_from_file_location("zonos3_mini_processor_module", processor_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._tts_processor = module.Zonos3MiniProcessor(self.adapter, self.config)
        return self._tts_processor

    def update_config(self, new_config: Dict[str, Any]):
        self.config = new_config.copy() if new_config else {}
        self.adapter.update_config(self.config)
        if self._tts_processor is not None:
            self._tts_processor.update_config(self.config)

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
            raise ImportError("SRT support not available - missing required SRT parser")

        self._check_interrupt()
        srt_parser = self.SRTParser()
        subtitles = srt_parser.parse_srt_content(srt_content, allow_overlaps=True)

        from utils.timing.overlap_detection import SRTOverlapHandler

        has_overlaps = SRTOverlapHandler.detect_overlaps(subtitles)
        current_timing_mode, mode_switched = SRTOverlapHandler.handle_smart_natural_fallback(
            timing_mode, has_overlaps, "ZONOS3-Mini SRT"
        )

        print(f"🚀 ZONOS3-Mini SRT: Processing {len(subtitles)} subtitles in {current_timing_mode} mode")
        audio_segments, adjustments = self._process_all_subtitles(
            subtitles,
            voice_mapping,
            seed,
            enable_audio_cache=enable_audio_cache,
        )
        self._check_interrupt()

        final_audio, final_adjustments, stretch_method = self._assemble_final_audio(
            audio_segments,
            subtitles,
            current_timing_mode,
            timing_params,
            adjustments,
        )
        if final_adjustments is not None:
            adjustments = final_adjustments

        timing_report = self._generate_timing_report(
            subtitles,
            adjustments,
            current_timing_mode,
            has_overlaps,
            mode_switched,
            timing_mode if mode_switched else None,
            stretch_method,
        )
        adjusted_srt_string = self._generate_adjusted_srt_string(subtitles, adjustments, current_timing_mode)

        total_duration = final_audio.shape[-1] / self.SAMPLE_RATE
        mode_info = current_timing_mode
        if mode_switched:
            mode_info = f"{current_timing_mode} (switched from {timing_mode} due to overlaps)"
        info = (
            f"Generated {total_duration:.1f}s ZONOS3-Mini SRT-timed audio from {len(subtitles)} subtitles "
            f"using {mode_info} mode"
        )

        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0).unsqueeze(0)
        elif final_audio.dim() == 2:
            final_audio = final_audio.unsqueeze(0)

        audio_output = {"waveform": final_audio, "sample_rate": self.SAMPLE_RATE}
        return audio_output, info, timing_report, adjusted_srt_string

    def _check_interrupt(
        self,
        subtitle_index: Optional[int] = None,
        total_subtitles: Optional[int] = None,
    ):
        if model_management.interrupt_processing:
            if subtitle_index is not None and total_subtitles is not None:
                raise InterruptedError(
                    f"ZONOS3-Mini SRT generation interrupted at subtitle {subtitle_index + 1}/{total_subtitles}"
                )
            raise InterruptedError("ZONOS3-Mini SRT generation interrupted by user")

    def _process_all_subtitles(
        self,
        subtitles: List,
        voice_mapping: Dict[str, Any],
        seed: int,
        enable_audio_cache: bool = True,
    ) -> Tuple[List[torch.Tensor], List[Dict]]:
        audio_segments: List[torch.Tensor] = []
        adjustments: List[Dict[str, Any]] = []
        try:
            for i, sub in enumerate(subtitles):
                self._check_interrupt(i, len(subtitles))

                target_start = sub.start_time
                target_end = sub.end_time
                target_duration = target_end - target_start
                text = sub.text.strip()
                if not text:
                    audio = torch.zeros(1, int(target_duration * self.SAMPLE_RATE))
                else:
                    print(f"📖 ZONOS3-Mini SRT Subtitle {i + 1}/{len(subtitles)}")
                    segment_records = self.processor.process_text(
                        text=text,
                        voice_mapping=voice_mapping,
                        seed=seed + i,
                        enable_chunking=False,
                        enable_audio_cache=enable_audio_cache,
                        apply_edit_postprocessing=False,
                    )
                    if len(segment_records) > 1:
                        audio, _ = self.processor.combine_audio_segments(
                            segments=segment_records,
                            method="auto",
                            silence_ms=0,
                            original_text=text,
                            return_info=True,
                        )
                    elif segment_records:
                        audio = segment_records[0]["waveform"]
                    else:
                        audio = torch.zeros(1, 0)

                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                    elif audio.dim() == 3:
                        audio = audio.squeeze(0)
                    audio = audio.cpu()

                audio_segments.append(audio)
                natural_duration = audio.shape[-1] / self.SAMPLE_RATE
                stretch_factor = target_duration / natural_duration if natural_duration > 0 else 1.0
                adjustments.append(
                    {
                        "index": i,
                        "segment_index": i,
                        "sequence": sub.sequence,
                        "natural_duration": natural_duration,
                        "target_start": target_start,
                        "target_end": target_end,
                        "target_duration": target_duration,
                        "start_time": target_start,
                        "end_time": target_end,
                        "stretch_factor": stretch_factor,
                        "needs_stretching": abs(stretch_factor - 1.0) > 0.05,
                        "stretch_type": "compress" if stretch_factor < 1.0 else "expand" if stretch_factor > 1.0 else "none",
                        "adjustment": natural_duration - target_duration,
                        "adjusted_start": target_start,
                        "adjusted_end": target_end,
                        "adjusted_duration": natural_duration,
                    }
                )
                if text:
                    print(
                        f"✅ ZONOS3-Mini Subtitle {i + 1}/{len(subtitles)}: "
                        f"{natural_duration:.3f}s (expected {target_duration:.3f}s)"
                    )

            return audio_segments, adjustments
        finally:
            self.processor.update_config(self.config)

    def _assemble_final_audio(self, audio_segments, subtitles, timing_mode, timing_params, adjustments):
        if timing_mode == "stretch_to_fit":
            from engines.chatterbox.audio_timing import TimedAudioAssembler

            assembler = TimedAudioAssembler(self.SAMPLE_RATE)
            target_timings = [(sub.start_time, sub.end_time) for sub in subtitles]
            fade_duration = timing_params.get("fade_for_StretchToFit", 0.01)
            final_audio, stretch_method_used = assembler.assemble_timed_audio(
                audio_segments,
                target_timings,
                fade_duration=fade_duration,
            )
            return final_audio, None, stretch_method_used

        if timing_mode == "pad_with_silence":
            from utils.timing.assembly import AudioAssemblyEngine

            assembler = AudioAssemblyEngine(self.SAMPLE_RATE)
            final_audio = assembler.assemble_with_overlaps(audio_segments, subtitles, torch.device("cpu"))
            return final_audio, None, None

        if timing_mode == "concatenate":
            from utils.timing.assembly import AudioAssemblyEngine
            from utils.timing.engine import TimingEngine

            timing_engine = TimingEngine(self.SAMPLE_RATE)
            assembler = AudioAssemblyEngine(self.SAMPLE_RATE)
            new_adjustments = timing_engine.calculate_concatenation_adjustments(audio_segments, subtitles)
            fade_duration = timing_params.get("fade_for_StretchToFit", 0.01)
            final_audio = assembler.assemble_concatenation(audio_segments, fade_duration)
            return final_audio, new_adjustments, None

        from utils.timing.assembly import AudioAssemblyEngine
        from utils.timing.engine import TimingEngine

        timing_engine = TimingEngine(self.SAMPLE_RATE)
        assembler = AudioAssemblyEngine(self.SAMPLE_RATE)
        smart_adjustments, processed_segments = timing_engine.calculate_smart_timing_adjustments(
            audio_segments,
            subtitles,
            timing_params.get("timing_tolerance", 2.0),
            timing_params.get("max_stretch_ratio", 1.0),
            timing_params.get("min_stretch_ratio", 0.5),
            torch.device("cpu"),
        )
        final_audio = assembler.assemble_smart_natural(
            audio_segments,
            processed_segments,
            smart_adjustments,
            subtitles,
            torch.device("cpu"),
        )
        return final_audio, smart_adjustments, None

    def _generate_timing_report(
        self,
        subtitles,
        adjustments,
        current_timing_mode,
        has_overlaps=False,
        mode_switched=False,
        original_mode=None,
        stretch_method=None,
    ) -> str:
        from utils.timing.reporting import SRTReportGenerator

        reporter = SRTReportGenerator()
        return reporter.generate_timing_report(
            subtitles,
            adjustments,
            current_timing_mode,
            has_overlaps,
            mode_switched,
            original_mode,
            stretch_method,
        )

    def _generate_adjusted_srt_string(self, subtitles, adjustments, timing_mode: str) -> str:
        from utils.timing.reporting import SRTReportGenerator

        reporter = SRTReportGenerator()
        return reporter.generate_adjusted_srt_string(subtitles, adjustments, timing_mode)
