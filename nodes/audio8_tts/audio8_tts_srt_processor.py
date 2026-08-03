"""SRT orchestration for Audio8 TTS."""

import importlib.util
import os

import comfy.model_management as model_management
import torch

from engines.adapters.audio8_tts_adapter import Audio8TTSEngineAdapter
from utils.system.import_manager import import_manager
from utils.timing.assembly import AudioAssemblyEngine
from utils.timing.engine import TimingEngine
from utils.timing.overlap_detection import SRTOverlapHandler
from utils.timing.reporting import SRTReportGenerator


class Audio8TTSSRTProcessor:
    """Generate Audio8 speech per subtitle and apply shared timing modes."""

    SAMPLE_RATE = 44100

    def __init__(self, node_instance, config):
        self.node_instance = node_instance
        self.config = dict(config or {})
        self.adapter = Audio8TTSEngineAdapter(self.config)
        self._processor = None

        success, modules, message = import_manager.import_srt_modules()
        if not success:
            raise ImportError(f"Audio8 TTS SRT unavailable: {message}")
        self.SRTParser = modules["SRTParser"]

    @property
    def processor(self):
        if self._processor is None:
            processor_path = os.path.join(
                os.path.dirname(__file__), "audio8_tts_processor.py"
            )
            spec = importlib.util.spec_from_file_location(
                "audio8_tts_processor_module", processor_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._processor = module.Audio8TTSProcessor(self.adapter, self.config)
        return self._processor

    def update_config(self, config):
        self.config = dict(config or {})
        self.adapter.update_config(self.config)
        if self._processor is not None:
            self._processor.update_config(self.config)

    @staticmethod
    def _check_interrupt(index=None, total=None):
        if not model_management.interrupt_processing:
            return
        location = f" at subtitle {index + 1}/{total}" if index is not None else ""
        raise InterruptedError(f"Audio8 TTS SRT generation interrupted{location}")

    def process_srt_content(
        self,
        srt_content,
        voice_mapping,
        seed,
        timing_mode,
        timing_params,
        enable_audio_cache=True,
    ):
        self._check_interrupt()
        subtitles = self.SRTParser().parse_srt_content(srt_content, allow_overlaps=True)
        has_overlaps = SRTOverlapHandler.detect_overlaps(subtitles)
        active_mode, switched = SRTOverlapHandler.handle_smart_natural_fallback(
            timing_mode, has_overlaps, "Audio8 TTS SRT"
        )

        audio_segments = []
        adjustments = []
        for index, subtitle in enumerate(subtitles):
            self._check_interrupt(index, len(subtitles))
            text = (subtitle.text or "").strip()
            if text:
                records = self.processor.process_text(
                    text=text,
                    voice_mapping=voice_mapping,
                    seed=seed + index,
                    enable_chunking=False,
                    enable_audio_cache=enable_audio_cache,
                    apply_edit_postprocessing=True,
                )
                audio, _ = self.processor.combine_audio_segments(
                    records,
                    method="auto",
                    silence_ms=0,
                    original_text=text,
                    return_info=True,
                )
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
            else:
                audio = torch.zeros(1, int(subtitle.duration * self.SAMPLE_RATE))

            audio = audio.detach().cpu().float()
            audio_segments.append(audio)
            natural_duration = audio.shape[-1] / self.SAMPLE_RATE
            target_duration = subtitle.duration
            stretch_factor = (
                target_duration / natural_duration if natural_duration else 1.0
            )
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
                    "needs_stretching": (abs(stretch_factor - 1.0) > 0.05),
                    "stretch_type": (
                        "compress"
                        if stretch_factor < 1.0
                        else "expand"
                        if stretch_factor > 1.0
                        else "none"
                    ),
                    "adjustment": natural_duration - target_duration,
                    "adjusted_start": subtitle.start_time,
                    "adjusted_end": subtitle.end_time,
                    "adjusted_duration": natural_duration,
                }
            )

        self._check_interrupt()
        final_audio, replacement, stretch_method = self._assemble(
            audio_segments, subtitles, active_mode, timing_params or {}
        )
        self._check_interrupt()
        if replacement is not None:
            adjustments = replacement

        reporter = SRTReportGenerator()
        timing_report = reporter.generate_timing_report(
            subtitles,
            adjustments,
            active_mode,
            has_overlaps,
            switched,
            timing_mode if switched else None,
            stretch_method,
        )
        adjusted_srt = reporter.generate_adjusted_srt_string(
            subtitles, adjustments, active_mode
        )

        final_audio = final_audio.detach().cpu().float()
        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0).unsqueeze(0)
        elif final_audio.dim() == 2:
            final_audio = final_audio.unsqueeze(0)
        duration = final_audio.shape[-1] / self.SAMPLE_RATE
        generation_info = (
            f"Generated {duration:.1f}s Audio8 TTS SRT audio from "
            f"{len(subtitles)} subtitles using {active_mode} mode"
        )
        return (
            {"waveform": final_audio, "sample_rate": self.SAMPLE_RATE},
            generation_info,
            timing_report,
            adjusted_srt,
        )

    def _assemble(self, audio_segments, subtitles, mode, params):
        if mode == "stretch_to_fit":
            from engines.chatterbox.audio_timing import TimedAudioAssembler

            assembler = TimedAudioAssembler(self.SAMPLE_RATE)
            audio, method = assembler.assemble_timed_audio(
                audio_segments,
                [(subtitle.start_time, subtitle.end_time) for subtitle in subtitles],
                fade_duration=params.get("fade_for_StretchToFit", 0.01),
            )
            return audio, None, method

        assembler = AudioAssemblyEngine(self.SAMPLE_RATE)
        if mode == "pad_with_silence":
            audio = assembler.assemble_with_overlaps(
                audio_segments, subtitles, torch.device("cpu")
            )
            return audio, None, None

        timing = TimingEngine(self.SAMPLE_RATE)
        if mode == "concatenate":
            replacements = timing.calculate_concatenation_adjustments(
                audio_segments, subtitles
            )
            audio = assembler.assemble_concatenation(
                audio_segments,
                params.get("fade_for_StretchToFit", 0.01),
            )
            return audio, replacements, None

        if mode != "smart_natural":
            raise ValueError(f"Unsupported Audio8 SRT timing mode: {mode}")
        replacements, processed = timing.calculate_smart_timing_adjustments(
            audio_segments,
            subtitles,
            params.get("timing_tolerance", 2.0),
            params.get("max_stretch_ratio", 1.0),
            params.get("min_stretch_ratio", 0.5),
            torch.device("cpu"),
        )
        audio = assembler.assemble_smart_natural(
            audio_segments,
            processed,
            replacements,
            subtitles,
            torch.device("cpu"),
        )
        return audio, replacements, None
