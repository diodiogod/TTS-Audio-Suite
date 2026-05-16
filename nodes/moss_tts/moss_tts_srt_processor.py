"""
MOSS-TTS SRT processor.

Implements SRT timing workflow for the official MOSS-TTS models.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.text.character_parser import CharacterParser
from utils.voice.discovery import get_available_characters, voice_discovery
from utils.audio.edit_post_processor import process_segments as apply_edit_post_processing

import importlib.util

moss_tts_processor_path = os.path.join(current_dir, "moss_tts_processor.py")
moss_tts_spec = importlib.util.spec_from_file_location("moss_tts_processor_module", moss_tts_processor_path)
moss_tts_module = importlib.util.module_from_spec(moss_tts_spec)
moss_tts_spec.loader.exec_module(moss_tts_module)
MossTTSProcessor = moss_tts_module.MossTTSProcessor


class MossTTSSRTProcessor:
    """Complete SRT processor for MOSS-TTS."""

    SAMPLE_RATE = 24000

    def __init__(self, node_instance, engine_config: Dict[str, Any]):
        self.node_instance = node_instance
        self.engine_config = engine_config.copy()
        self.processor = MossTTSProcessor(node_instance, engine_config)
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
        available_chars = get_available_characters()
        if available_chars:
            self.character_parser.set_available_characters(list(available_chars))

        for alias, target in voice_discovery.get_character_aliases().items():
            self.character_parser.add_character_fallback(alias, target)

        for char, lang in voice_discovery.get_character_language_defaults().items():
            self.character_parser.set_character_language_default(char, lang)

    def update_config(self, new_config: Dict[str, Any]):
        self.engine_config.update(new_config)
        if self.processor:
            self.processor.update_config(new_config)

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
        parser = self.SRTParser()
        subtitles = parser.parse_srt_content(srt_content, allow_overlaps=True)

        from utils.timing.overlap_detection import SRTOverlapHandler

        has_overlaps = SRTOverlapHandler.detect_overlaps(subtitles)
        current_timing_mode, mode_switched = SRTOverlapHandler.handle_smart_natural_fallback(
            timing_mode,
            has_overlaps,
            "MOSS-TTS SRT",
        )

        print(f"🎬 MOSS-TTS SRT: Processing {len(subtitles)} subtitle(s)")

        audio_segments, adjustments = self._process_all_subtitles(subtitles, voice_mapping, seed)
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
        info = f"Generated {total_duration:.1f}s MOSS-TTS SRT-timed audio from {len(subtitles)} subtitles using {mode_info} mode"

        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0).unsqueeze(0)
        elif final_audio.dim() == 2:
            final_audio = final_audio.unsqueeze(0)

        return {"waveform": final_audio, "sample_rate": self.SAMPLE_RATE}, info, timing_report, adjusted_srt

    def _check_interrupt(self):
        if hasattr(self.node_instance, "check_interrupt"):
            self.node_instance.check_interrupt()

    def _process_all_subtitles(self, subtitles: List[Any], voice_mapping: Dict[str, Any], seed: int):
        audio_segments = []
        adjustments = []
        all_segments_for_editing = []
        global_native_character_map = None

        if self.engine_config.get("multi_speaker_mode") == "Native Multi-Speaker Dialogue":
            global_native_character_map = self.processor.build_native_character_map_from_texts(
                [subtitle.text for subtitle in subtitles if getattr(subtitle, "text", "").strip()]
            )
            if global_native_character_map:
                print(f"🎭 MOSS-TTSD global SRT character mapping: {global_native_character_map}")

        for idx, sub in enumerate(subtitles):
            self._check_interrupt()
            text = sub.text.strip()
            target_start = sub.start_time
            target_end = sub.end_time
            target_duration = target_end - target_start

            if not text:
                silence = torch.zeros(1, int(target_duration * self.SAMPLE_RATE))
                audio_segments.append(silence)
                adjustments.append(self._make_adjustment(idx, sub, target_duration, target_duration))
                continue

            print(f"📖 MOSS-TTS SRT subtitle {idx + 1}/{len(subtitles)}: '{text[:50]}...'")

            subtitle_voice_mapping = dict(voice_mapping or {})
            narrator_ref = subtitle_voice_mapping.get("narrator")
            if narrator_ref:
                narrator_audio = narrator_ref.get("audio")
                if narrator_audio is None:
                    narrator_audio = narrator_ref.get("waveform")
                subtitle_voice_mapping.update(
                    self.processor.build_voice_mapping(
                        text,
                        narrator_audio=narrator_audio,
                        reference_text=narrator_ref.get("reference_text", ""),
                        narrator_audio_path=narrator_ref.get("audio_path"),
                    )
                )
            else:
                subtitle_voice_mapping.update(self.processor.build_voice_mapping(text))

            if global_native_character_map:
                subtitle_voice_mapping[self.processor.NATIVE_MAPPING_META_KEY] = global_native_character_map

            segment_dicts = self.processor.process_text(
                text=text,
                voice_mapping=subtitle_voice_mapping,
                seed=seed + idx,
                enable_chunking=False,
                max_chars_per_chunk=400,
                apply_edit_postprocessing=False,
            )

            subtitle_has_edit_tags = any(seg.get("edit_tags") for seg in segment_dicts)

            if not segment_dicts:
                audio = torch.zeros(1, int(target_duration * self.SAMPLE_RATE))
                audio_segments.append(audio)
                adjustments.append(self._make_adjustment(idx, sub, target_duration, target_duration))
                continue
            elif subtitle_has_edit_tags:
                for seg_idx, seg in enumerate(segment_dicts):
                    w = seg["waveform"]
                    if w.dim() == 1:
                        w = w.unsqueeze(0)
                    elif w.dim() == 3:
                        w = w.squeeze(0)
                    all_segments_for_editing.append({
                        **seg,
                        "waveform": w.cpu(),
                        "subtitle_index": idx,
                        "segment_index": seg_idx,
                    })
                audio_segments.append(None)
                adjustments.append(self._make_adjustment(idx, sub, 0.0, target_duration))
                print(f"✅ MOSS-TTS SRT subtitle {idx + 1}: queued for batch edit processing")
                continue
            elif len(segment_dicts) > 1:
                audio = self.processor.combine_audio_segments(
                    segments=segment_dicts,
                    method="auto",
                    silence_ms=100,
                    text_length=len(text),
                    return_info=False,
                )
            else:
                audio = segment_dicts[0]["waveform"]

            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            elif audio.dim() == 3:
                audio = audio.squeeze(0)
            audio_segments.append(audio)

            natural_duration = audio.shape[-1] / self.SAMPLE_RATE
            adjustments.append(self._make_adjustment(idx, sub, natural_duration, target_duration))
            print(f"✅ MOSS-TTS SRT subtitle {idx + 1}: {natural_duration:.2f}s (target {target_duration:.2f}s)")

        if all_segments_for_editing:
            print(
                f"\n🎨 MOSS-TTS SRT: Applying edit post-processing to "
                f"{len(all_segments_for_editing)} segment(s) from all subtitles..."
            )
            processed = apply_edit_post_processing(all_segments_for_editing, self.processor.config)

            by_subtitle: Dict[int, List[Any]] = {}
            for seg in processed:
                sub_idx = seg["subtitle_index"]
                by_subtitle.setdefault(sub_idx, []).append(seg)

            for sub_idx, segs in by_subtitle.items():
                segs_sorted = sorted(segs, key=lambda s: s["segment_index"])
                for seg in segs_sorted:
                    w = seg["waveform"]
                    if w.dim() == 3:
                        w = w.squeeze(0)
                    elif w.dim() == 1:
                        w = w.unsqueeze(0)
                    seg["waveform"] = w

                if len(segs_sorted) > 1:
                    audio = self.processor.combine_audio_segments(
                        segments=segs_sorted,
                        method="auto",
                        silence_ms=100,
                        text_length=len(subtitles[sub_idx].text),
                        return_info=False,
                    )
                else:
                    audio = segs_sorted[0]["waveform"]

                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                elif audio.dim() == 3:
                    audio = audio.squeeze(0)
                audio_segments[sub_idx] = audio

                natural_duration = audio.shape[-1] / self.SAMPLE_RATE
                target_duration = adjustments[sub_idx]["target_duration"]
                adjustments[sub_idx] = self._make_adjustment(sub_idx, subtitles[sub_idx], natural_duration, target_duration)
                print(f"✅ MOSS-TTS SRT subtitle {sub_idx + 1}: {natural_duration:.2f}s after edit processing")

        return audio_segments, adjustments

    @staticmethod
    def _make_adjustment(index: int, sub, natural_duration: float, target_duration: float) -> Dict[str, Any]:
        stretch_factor = target_duration / natural_duration if natural_duration > 0 else 1.0
        return {
            "index": index,
            "segment_index": index,
            "sequence": sub.sequence,
            "natural_duration": natural_duration,
            "target_start": sub.start_time,
            "target_end": sub.end_time,
            "target_duration": target_duration,
            "start_time": sub.start_time,
            "end_time": sub.end_time,
            "stretch_factor": stretch_factor,
            "needs_stretching": abs(stretch_factor - 1.0) > 0.05,
            "stretch_type": "compress" if stretch_factor < 1.0 else "expand" if stretch_factor > 1.0 else "none",
            "adjustment": natural_duration - target_duration,
            "adjusted_start": sub.start_time,
            "adjusted_end": sub.end_time,
            "adjusted_duration": natural_duration,
        }

    def _assemble_final_audio(
        self,
        audio_segments: List[torch.Tensor],
        subtitles: List[Any],
        timing_mode: str,
        timing_params: Dict[str, Any],
        adjustments: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, Any]]], Optional[str]]:
        if timing_mode == "stretch_to_fit":
            assembler = self.TimedAudioAssembler(self.SAMPLE_RATE)
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
            return assembler.assemble_with_overlaps(audio_segments, subtitles, torch.device("cpu")), None, None

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

    @staticmethod
    def _generate_timing_report(
        subtitles: List[Any],
        adjustments: List[Dict[str, Any]],
        timing_mode: str,
        has_original_overlaps: bool = False,
        mode_switched: bool = False,
        original_mode: Optional[str] = None,
        stretch_method: Optional[str] = None,
    ) -> str:
        from utils.timing.reporting import SRTReportGenerator

        return SRTReportGenerator().generate_timing_report(
            subtitles,
            adjustments,
            timing_mode,
            has_original_overlaps,
            mode_switched,
            original_mode,
            stretch_method,
        )

    @staticmethod
    def _generate_adjusted_srt_string(
        subtitles: List[Any],
        adjustments: List[Dict[str, Any]],
        timing_mode: str,
    ) -> str:
        from utils.timing.reporting import SRTReportGenerator

        return SRTReportGenerator().generate_adjusted_srt_string(subtitles, adjustments, timing_mode)

    def cleanup(self):
        if self.processor:
            self.processor.cleanup()
            self.processor = None
