"""VoxCPM SRT orchestration using the shared text processor."""

import importlib.util
import os
import sys
from typing import Any, Dict, Optional, Tuple

import comfy.model_management as model_management
import torch

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.adapters.voxcpm_adapter import VoxCPMEngineAdapter
from utils.system.import_manager import import_manager
from utils.timing.assembly import AudioAssemblyEngine
from utils.timing.engine import TimingEngine
from utils.timing.overlap_detection import SRTOverlapHandler
from utils.timing.reporting import SRTReportGenerator


class VoxCPMSRTProcessor:
    """Generate VoxCPM speech per subtitle, then apply suite timing modes."""

    def __init__(self, node_instance, config: Dict[str, Any]):
        self.node_instance = node_instance
        self.config = dict(config or {})
        self.adapter = VoxCPMEngineAdapter(self.config)
        self._processor = None

        success, modules, message = import_manager.import_srt_modules()
        if not success:
            raise ImportError(f"VoxCPM SRT unavailable: {message}")
        self.SRTParser = modules["SRTParser"]

    @property
    def processor(self):
        if self._processor is None:
            processor_path = os.path.join(current_dir, "voxcpm_processor.py")
            spec = importlib.util.spec_from_file_location(
                "voxcpm_processor_module", processor_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load VoxCPM processor from {processor_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self._processor = module.VoxCPMProcessor(self.adapter, self.config)
        return self._processor

    @property
    def sample_rate(self) -> int:
        return self.processor.sample_rate

    @property
    def SAMPLE_RATE(self) -> int:
        return self.sample_rate

    def get_sample_rate(self) -> int:
        return self.sample_rate

    def update_config(self, new_config: Dict[str, Any]):
        self.config = dict(new_config or {})
        if self._processor is not None:
            self._processor.update_config(self.config)
        else:
            self.adapter.update_config(self.config)

    @staticmethod
    def _check_interrupt(
        subtitle_index: Optional[int] = None,
        total_subtitles: Optional[int] = None,
    ):
        if model_management.interrupt_processing:
            where = ""
            if subtitle_index is not None and total_subtitles is not None:
                where = f" before subtitle {subtitle_index + 1}/{total_subtitles}"
            raise InterruptedError(f"VoxCPM SRT generation interrupted{where}")

    @staticmethod
    def _normalize_audio(audio: torch.Tensor) -> torch.Tensor:
        audio = audio.detach().float().cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        if audio.dim() != 2:
            raise ValueError(
                f"VoxCPM SRT received invalid audio shape {tuple(audio.shape)}"
            )
        return audio

    @staticmethod
    def _build_adjustment(index, subtitle, natural_duration: float) -> Dict[str, Any]:
        target_duration = subtitle.duration
        ratio = target_duration / natural_duration if natural_duration > 0 else 1.0
        return {
            "index": index,
            "segment_index": index,
            "sequence": subtitle.sequence,
            "natural_duration": natural_duration,
            "target_start": subtitle.start_time,
            "target_end": subtitle.end_time,
            "target_duration": target_duration,
            "start_time": subtitle.start_time,
            "end_time": subtitle.end_time,
            "stretch_factor": ratio,
            "needs_stretching": abs(ratio - 1.0) > 0.05,
            "stretch_type": (
                "compress" if ratio < 1.0 else "expand" if ratio > 1.0 else "none"
            ),
            "adjustment": natural_duration - target_duration,
            "adjusted_start": subtitle.start_time,
            "adjusted_end": subtitle.end_time,
            "adjusted_duration": natural_duration,
        }

    def process_srt_content(
        self,
        srt_content: str,
        voice_mapping: Dict[str, Any],
        seed: int,
        timing_mode: str,
        timing_params: Dict[str, Any],
        enable_audio_cache: bool = True,
    ) -> Tuple[Dict[str, Any], str, str, str]:
        self._check_interrupt()
        subtitles = self.SRTParser().parse_srt_content(
            srt_content, allow_overlaps=True
        )

        if not subtitles:
            sample_rate = self.sample_rate
            audio = torch.zeros(1, 1, 0, dtype=torch.float32)
            return (
                {"waveform": audio, "sample_rate": sample_rate},
                "Generated 0.0s VoxCPM SRT audio from 0 subtitles",
                "No subtitles to process.",
                srt_content or "",
            )

        has_overlaps = SRTOverlapHandler.detect_overlaps(subtitles)
        active_mode, mode_switched = (
            SRTOverlapHandler.handle_smart_natural_fallback(
                timing_mode, has_overlaps, "VoxCPM SRT"
            )
        )

        print(
            f"🚀 VoxCPM SRT: Processing {len(subtitles)} subtitles "
            f"in {active_mode} mode"
        )
        audio_segments = []
        adjustments = []
        active_sample_rate = None

        for subtitle_index, subtitle in enumerate(subtitles):
            self._check_interrupt(subtitle_index, len(subtitles))
            subtitle_text = (subtitle.text or "").strip()

            if subtitle_text:
                records = self.processor.process_text(
                    text=subtitle_text,
                    voice_mapping=voice_mapping,
                    seed=0 if seed == 0 else seed + subtitle_index,
                    enable_chunking=False,
                    enable_audio_cache=enable_audio_cache,
                    apply_edit_postprocessing=True,
                )
                subtitle_audio, _ = self.processor.combine_audio_segments(
                    records,
                    method="auto",
                    silence_ms=0,
                    original_text=subtitle_text,
                    return_info=True,
                )
                subtitle_rate = (
                    int(records[0]["sample_rate"]) if records else self.sample_rate
                )
            else:
                subtitle_rate = self.sample_rate
                subtitle_audio = torch.zeros(
                    1,
                    max(0, int(round(subtitle.duration * subtitle_rate))),
                    dtype=torch.float32,
                )

            self._check_interrupt(subtitle_index, len(subtitles))
            subtitle_audio = self._normalize_audio(subtitle_audio)

            if active_sample_rate is None:
                active_sample_rate = subtitle_rate
            elif subtitle_rate != active_sample_rate:
                raise ValueError(
                    "VoxCPM model sample rate changed during one SRT job: "
                    f"{active_sample_rate} -> {subtitle_rate}"
                )

            audio_segments.append(subtitle_audio)
            natural_duration = subtitle_audio.shape[-1] / active_sample_rate
            adjustments.append(
                self._build_adjustment(
                    subtitle_index, subtitle, natural_duration
                )
            )

        self._check_interrupt()
        final_audio, replacement_adjustments, stretch_method = self._assemble(
            audio_segments,
            subtitles,
            active_mode,
            timing_params or {},
            active_sample_rate,
        )
        if replacement_adjustments is not None:
            adjustments = replacement_adjustments

        reporter = SRTReportGenerator()
        timing_report = reporter.generate_timing_report(
            subtitles,
            adjustments,
            active_mode,
            has_overlaps,
            mode_switched,
            timing_mode if mode_switched else None,
            stretch_method,
        )
        adjusted_srt = reporter.generate_adjusted_srt_string(
            subtitles, adjustments, active_mode
        )

        final_audio = final_audio.detach().float().cpu()
        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0).unsqueeze(0)
        elif final_audio.dim() == 2:
            final_audio = final_audio.unsqueeze(0)
        elif final_audio.dim() != 3:
            raise ValueError(
                f"VoxCPM SRT assembler returned invalid shape {tuple(final_audio.shape)}"
            )

        duration = final_audio.shape[-1] / active_sample_rate
        mode_description = active_mode
        if mode_switched:
            mode_description += f" (switched from {timing_mode} due to overlaps)"
        info = (
            f"Generated {duration:.1f}s VoxCPM SRT audio from "
            f"{len(subtitles)} subtitles using {mode_description} mode"
        )
        return (
            {"waveform": final_audio, "sample_rate": active_sample_rate},
            info,
            timing_report,
            adjusted_srt,
        )

    @staticmethod
    def _assemble(
        audio_segments,
        subtitles,
        timing_mode: str,
        timing_params: Dict[str, Any],
        sample_rate: int,
    ):
        if timing_mode == "stretch_to_fit":
            from engines.chatterbox.audio_timing import TimedAudioAssembler

            assembler = TimedAudioAssembler(sample_rate)
            audio, method = assembler.assemble_timed_audio(
                audio_segments,
                [
                    (subtitle.start_time, subtitle.end_time)
                    for subtitle in subtitles
                ],
                fade_duration=timing_params.get("fade_for_StretchToFit", 0.01),
            )
            return audio, None, method

        assembler = AudioAssemblyEngine(sample_rate)
        cpu_device = torch.device("cpu")
        if timing_mode == "pad_with_silence":
            audio = assembler.assemble_with_overlaps(
                audio_segments, subtitles, cpu_device
            )
            return audio, None, None

        timing_engine = TimingEngine(sample_rate)
        if timing_mode == "concatenate":
            replacements = timing_engine.calculate_concatenation_adjustments(
                audio_segments, subtitles
            )
            audio = assembler.assemble_concatenation(
                audio_segments,
                timing_params.get("fade_for_StretchToFit", 0.01),
            )
            return audio, replacements, None

        replacements, processed_segments = (
            timing_engine.calculate_smart_timing_adjustments(
                audio_segments,
                subtitles,
                timing_params.get("timing_tolerance", 2.0),
                timing_params.get("max_stretch_ratio", 1.0),
                timing_params.get("min_stretch_ratio", 0.5),
                cpu_device,
            )
        )
        audio = assembler.assemble_smart_natural(
            audio_segments,
            processed_segments,
            replacements,
            subtitles,
            cpu_device,
        )
        return audio, replacements, None
