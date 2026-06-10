"""Higgs Audio v3 SRT processor."""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict, List, Tuple

import torch

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.text.character_parser import CharacterParser
from utils.voice.discovery import get_available_characters, get_character_mapping, voice_discovery

processor_path = os.path.join(current_dir, "higgs_audio_v3_processor.py")
processor_spec = importlib.util.spec_from_file_location("higgs_audio_v3_processor_module", processor_path)
processor_module = importlib.util.module_from_spec(processor_spec)
processor_spec.loader.exec_module(processor_module)
HiggsAudioV3Processor = processor_module.HiggsAudioV3Processor


class HiggsAudioV3SRTProcessor:
    """Complete SRT processor for Higgs Audio v3."""

    SAMPLE_RATE = 24000

    def __init__(self, node_instance, engine_config: Dict[str, Any]):
        self.node_instance = node_instance
        self.engine_config = engine_config.copy()
        self.processor = HiggsAudioV3Processor(node_instance, engine_config)
        self._load_srt_modules()
        self._setup_character_parser()

    def _load_srt_modules(self):
        try:
            from utils.system.import_manager import import_manager

            success, srt_modules, msg = import_manager.import_srt_modules()
            if success:
                self.SRTParser = srt_modules.get("SRTParser")
                self.TimedAudioAssembler = srt_modules.get("TimedAudioAssembler")
                self.srt_available = True
            else:
                print(f"⚠️ SRT module not available: {msg}")
                self.srt_available = False
        except Exception as e:
            print(f"⚠️ SRT module not available: {e}")
            self.srt_available = False

    def _setup_character_parser(self):
        self.character_parser = CharacterParser()
        aliases = voice_discovery.get_character_aliases()
        for alias, target in aliases.items():
            self.character_parser.add_character_fallback(alias, target)
        for char, lang in voice_discovery.get_character_language_defaults().items():
            self.character_parser.set_character_language_default(char, lang)

    def update_config(self, new_config: Dict[str, Any]):
        self.engine_config.update(new_config)
        self.processor.update_config(new_config)

    def _check_interrupt(self):
        if hasattr(self.node_instance, "check_interrupt"):
            self.node_instance.check_interrupt()

    def process_srt_content(
        self,
        srt_content: str,
        voice_mapping: Dict[str, Any],
        seed: int,
        timing_mode: str,
        timing_params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str, str, str]:
        if not self.srt_available:
            raise ImportError("SRT modules not available - missing required SRT parser")

        self._check_interrupt()
        subtitles = self.SRTParser().parse_srt_content(srt_content, allow_overlaps=True)

        from utils.timing.overlap_detection import SRTOverlapHandler

        has_overlaps = SRTOverlapHandler.detect_overlaps(subtitles)
        current_timing_mode, mode_switched = SRTOverlapHandler.handle_smart_natural_fallback(
            timing_mode, has_overlaps, "Higgs Audio v3 SRT"
        )

        print(f"🎬 Higgs Audio v3 SRT: Processing {len(subtitles)} subtitle(s)")
        audio_segments, adjustments = self._process_all_subtitles(subtitles, voice_mapping, seed)
        self._check_interrupt()

        final_audio, final_adjustments, stretch_method = self._assemble_final_audio(
            audio_segments,
            subtitles,
            current_timing_mode,
            timing_params,
        )
        if final_adjustments is not None:
            adjustments = final_adjustments

        timing_report = self._generate_timing_report(
            subtitles,
            adjustments,
            current_timing_mode,
            has_original_overlaps=has_overlaps,
            mode_switched=mode_switched,
            original_mode=timing_mode if mode_switched else None,
            stretch_method=stretch_method,
        )
        adjusted_srt = self._generate_adjusted_srt_string(subtitles, adjustments, current_timing_mode)

        total_duration = final_audio.shape[-1] / self.SAMPLE_RATE if final_audio.numel() else 0.0
        mode_info = current_timing_mode
        if mode_switched:
            mode_info = f"{current_timing_mode} (switched from {timing_mode} due to overlaps)"
        info = f"Generated {total_duration:.1f}s Higgs Audio v3 SRT-timed audio from {len(subtitles)} subtitles using {mode_info} mode"

        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0).unsqueeze(0)
        elif final_audio.dim() == 2:
            final_audio = final_audio.unsqueeze(0)

        return {"waveform": final_audio, "sample_rate": self.SAMPLE_RATE}, info, timing_report, adjusted_srt

    def _process_all_subtitles(self, subtitles: List, voice_mapping: Dict[str, Any], seed: int):
        audio_segments = []
        adjustments = []

        character_aliases = voice_discovery.get_character_aliases()
        available_chars = get_available_characters()
        all_available = {"narrator"}
        if available_chars:
            all_available.update(available_chars)
        for alias, target in character_aliases.items():
            all_available.add(alias.lower())
            all_available.add(target.lower())
        for sub in subtitles:
            for tag in self.character_parser.CHARACTER_TAG_PATTERN.findall(sub.text):
                if not tag.startswith("pause:"):
                    all_available.add(tag.split("|")[0].strip().lower())

        self.character_parser.set_available_characters(list(all_available))
        self.character_parser.reset_session_cache()

        unique_characters = set()
        for sub in subtitles:
            for char, _text in self.character_parser.split_by_character(sub.text, include_language=False):
                if char:
                    unique_characters.add(char)

        character_mapping = {}
        if unique_characters:
            character_mapping = get_character_mapping(list(unique_characters), engine_type="higgs_audio_v3")

        simple_voice_mapping = {}
        for char_name, voice_data in character_mapping.items():
            if isinstance(voice_data, tuple):
                char_audio, char_text = voice_data
                if char_audio:
                    simple_voice_mapping[char_name] = {"audio_path": char_audio, "reference_text": char_text or ""}
            elif isinstance(voice_data, dict):
                simple_voice_mapping[char_name] = voice_data

        if voice_mapping and voice_mapping.get("narrator"):
            simple_voice_mapping["narrator"] = voice_mapping["narrator"]

        for i, sub in enumerate(subtitles):
            self._check_interrupt()
            text = sub.text.strip()
            target_start = sub.start_time
            target_end = sub.end_time
            target_duration = target_end - target_start

            if not text:
                audio = torch.zeros(1, int(target_duration * self.SAMPLE_RATE))
            else:
                print(f"📖 Higgs Audio v3 SRT subtitle {i + 1}/{len(subtitles)}: '{text[:50]}...'")
                segment_dicts = self.processor.process_text(
                    text=text,
                    voice_mapping=simple_voice_mapping,
                    seed=seed + i if seed else 0,
                    enable_chunking=False,
                    max_chars_per_chunk=400,
                )
                audio = self.processor.combine_audio_segments(
                    segments=segment_dicts,
                    method="auto",
                    silence_ms=100,
                    text_length=len(text),
                    return_info=False,
                )

            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            elif audio.dim() == 3:
                audio = audio.squeeze(0)
            audio_segments.append(audio)

            natural_duration = audio.shape[-1] / self.SAMPLE_RATE if audio.numel() else 0.0
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
        return audio_segments, adjustments

    def _assemble_final_audio(
        self,
        audio_segments: List[torch.Tensor],
        subtitles: List,
        timing_mode: str,
        timing_params: Dict[str, Any],
    ):
        if timing_mode == "stretch_to_fit":
            assembler = self.TimedAudioAssembler(self.SAMPLE_RATE)
            target_timings = [(sub.start_time, sub.end_time) for sub in subtitles]
            fade_duration = timing_params.get("fade_for_StretchToFit", 0.01)
            final_audio, stretch_method_used = assembler.assemble_timed_audio(
                audio_segments, target_timings, fade_duration=fade_duration
            )
            return final_audio, None, stretch_method_used

        if timing_mode == "pad_with_silence":
            from utils.timing.assembly import AudioAssemblyEngine

            assembler = AudioAssemblyEngine(self.SAMPLE_RATE)
            return assembler.assemble_with_overlaps(audio_segments, subtitles, torch.device("cpu")), None, None

        if timing_mode == "concatenate":
            from utils.timing.assembly import AudioAssemblyEngine
            from utils.timing.engine import TimingEngine

            timing_engine = TimingEngine(self.SAMPLE_RATE)
            assembler = AudioAssemblyEngine(self.SAMPLE_RATE)
            adjustments = timing_engine.calculate_concatenation_adjustments(audio_segments, subtitles)
            fade_duration = timing_params.get("fade_for_StretchToFit", 0.01)
            return assembler.assemble_concatenation(audio_segments, fade_duration), adjustments, None

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
            audio_segments, processed_segments, smart_adjustments, subtitles, torch.device("cpu")
        )
        return final_audio, smart_adjustments, None

    def _generate_timing_report(self, subtitles, adjustments, timing_mode, **kwargs) -> str:
        from utils.timing.reporting import SRTReportGenerator

        return SRTReportGenerator().generate_timing_report(subtitles, adjustments, timing_mode, **kwargs)

    def _generate_adjusted_srt_string(self, subtitles, adjustments, timing_mode) -> str:
        from utils.timing.reporting import SRTReportGenerator

        return SRTReportGenerator().generate_adjusted_srt_string(subtitles, adjustments, timing_mode)
