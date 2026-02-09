"""
Echo-TTS SRT Processor - Handles complete SRT subtitle processing for Echo-TTS engine.
Called by unified SRT node when using Echo-TTS engine.
"""

import os
import sys
import importlib.util
from typing import Dict, Any, List, Tuple, Optional

import torch
import comfy.model_management as model_management

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.adapters.echo_tts_adapter import EchoTTSEngineAdapter
from utils.voice.discovery import get_available_characters, get_character_mapping
from utils.text.character_parser import character_parser


class EchoTTSSRTProcessor:
    """
    Complete SRT processor for Echo-TTS engine.
    Handles SRT parsing, character switching, timing modes, and reporting.
    """

    SAMPLE_RATE = 44100

    def __init__(self, node_instance, config: Dict[str, Any]):
        self.node_instance = node_instance
        self.config = config.copy() if config else {}
        self.processor = EchoTTSEngineAdapter(self.config)
        self.srt_available = False
        self.SRTParser = None
        self.SRTSubtitle = None
        self.SRTParseError = None
        self._load_srt_modules()

    def _load_srt_modules(self):
        """Load SRT modules using the import manager."""
        try:
            from utils.system.import_manager import import_manager
            success, modules, msg = import_manager.import_srt_modules()
            if not success:
                print(f"⚠️ Echo-TTS SRT: SRT module not available: {msg}")
                return

            self.SRTParser = modules.get("SRTParser")
            self.SRTSubtitle = modules.get("SRTSubtitle")
            self.SRTParseError = modules.get("SRTParseError")
            if self.SRTParser is None:
                print("⚠️ Echo-TTS SRT: SRT parser not available")
                return

            self.srt_available = True
        except Exception as e:
            print(f"⚠️ Echo-TTS SRT: Failed to load SRT modules: {e}")

    def update_config(self, new_config: Dict[str, Any]):
        """Update processor configuration with new parameters."""
        self.config = new_config.copy() if new_config else {}
        if self.processor:
            self.processor.update_config(self.config)

    def process_srt_content(
        self,
        srt_content: str,
        voice_mapping: Dict[str, Any],
        seed: int,
        timing_mode: str,
        timing_params: Dict[str, Any],
        enable_audio_cache: bool = True,
    ) -> Tuple[Dict[str, Any], str, str, str]:
        """
        Process complete SRT content and generate timed audio with Echo-TTS.

        Args:
            srt_content: Complete SRT subtitle content
            voice_mapping: Mapping of character names to voice references
            seed: Random seed for generation
            timing_mode: SRT timing mode (stretch_to_fit, pad_with_silence, etc.)
            timing_params: Additional timing parameters
            enable_audio_cache: Enable audio caching for generated subtitle segments

        Returns:
            Tuple of (audio_output, generation_info, timing_report, adjusted_srt)
        """
        if not self.srt_available:
            raise ImportError("SRT support not available - missing required SRT parser")

        # Check for interrupt before starting expensive processing
        self._check_interrupt()

        # Parse SRT content with overlap support
        srt_parser = self.SRTParser()
        subtitles = srt_parser.parse_srt_content(srt_content, allow_overlaps=True)

        # Check for overlaps and handle smart_natural fallback
        from utils.timing.overlap_detection import SRTOverlapHandler
        has_overlaps = SRTOverlapHandler.detect_overlaps(subtitles)
        current_timing_mode, mode_switched = SRTOverlapHandler.handle_smart_natural_fallback(
            timing_mode, has_overlaps, "Echo-TTS SRT"
        )

        # Set up character parser with available characters
        available_chars = get_available_characters()
        character_parser.set_available_characters(list(available_chars))
        character_parser.reset_session_cache()

        # Process subtitles and generate audio segments
        print(f"🚀 Echo-TTS SRT: Processing {len(subtitles)} subtitles in {current_timing_mode} mode")
        audio_segments, adjustments = self._process_all_subtitles(
            subtitles, voice_mapping, seed, enable_audio_cache=enable_audio_cache
        )

        # Check for interrupt before timing assembly/reporting
        self._check_interrupt()

        # Assemble final audio based on timing mode
        final_audio, final_adjustments, stretch_method = self._assemble_final_audio(
            audio_segments, subtitles, current_timing_mode, timing_params, adjustments
        )

        if final_adjustments is not None:
            adjustments = final_adjustments

        # Generate timing report and adjusted SRT
        timing_report = self._generate_timing_report(
            subtitles, adjustments, current_timing_mode, has_overlaps, mode_switched, timing_mode if mode_switched else None, stretch_method
        )
        adjusted_srt_string = self._generate_adjusted_srt_string(subtitles, adjustments, current_timing_mode)

        total_duration = final_audio.shape[-1] / self.SAMPLE_RATE
        mode_info = f"{current_timing_mode}"
        if mode_switched:
            mode_info = f"{current_timing_mode} (switched from {timing_mode} due to overlaps)"

        info = (f"Generated {total_duration:.1f}s Echo-TTS SRT-timed audio from {len(subtitles)} subtitles "
                f"using {mode_info} mode")

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
        character: Optional[str] = None,
    ):
        """Check if generation should be interrupted by user."""
        if model_management.interrupt_processing:
            if subtitle_index is not None and total_subtitles is not None:
                if character:
                    raise InterruptedError(
                        f"Echo-TTS SRT generation interrupted at subtitle "
                        f"{subtitle_index + 1}/{total_subtitles}, character '{character}'"
                    )
                raise InterruptedError(
                    f"Echo-TTS SRT generation interrupted at subtitle "
                    f"{subtitle_index + 1}/{total_subtitles}"
                )
            raise InterruptedError("Echo-TTS SRT generation interrupted by user")

    def _process_all_subtitles(
        self,
        subtitles: List,
        voice_mapping: Dict[str, Any],
        seed: int,
        enable_audio_cache: bool = True,
    ) -> Tuple[List[torch.Tensor], List[Dict]]:
        """
        Generate audio for each subtitle and compute timing adjustments.
        """
        audio_segments: List[torch.Tensor] = []
        adjustments: List[Dict[str, Any]] = []

        # Build character mapping from available voices
        unique_characters = set()
        for sub in subtitles:
            segments = character_parser.split_by_character(sub.text, include_language=False)
            for char, _ in segments:
                if char:
                    unique_characters.add(char)

        character_mapping = {}
        if unique_characters:
            character_mapping = get_character_mapping(list(unique_characters), engine_type="audio_only")

        from utils.text.pause_processor import PauseTagProcessor
        from utils.text.segment_parameters import apply_segment_parameters
        from utils.text.step_audio_editx_special_tags import get_edit_tags_for_segment
        from utils.audio.edit_post_processor import process_segments as apply_edit_post_processing
        import re

        def _strip_s1_tag(text_value: str) -> str:
            return re.sub(r'\[s1\]\s*', '', text_value, flags=re.IGNORECASE)

        subtitle_lengths = [max(1, len((sub.text or "").strip())) for sub in subtitles]
        self.processor.start_job(total_blocks=len(subtitles), block_texts=subtitle_lengths)
        try:
            for i, sub in enumerate(subtitles):
                self.processor.set_current_block(i)
                self._check_interrupt(i, len(subtitles))

                text = sub.text.strip()
                if not text:
                    target_duration = sub.duration
                    silence = torch.zeros(1, int(target_duration * self.SAMPLE_RATE))
                    audio_segments.append(silence)
                    adjustments.append({
                        'index': i,
                        'segment_index': i,
                        'sequence': sub.sequence,
                        'natural_duration': target_duration,
                        'target_start': sub.start_time,
                        'target_end': sub.end_time,
                        'target_duration': target_duration,
                        'start_time': sub.start_time,
                        'end_time': sub.end_time,
                        'stretch_factor': 1.0,
                        'needs_stretching': False,
                        'stretch_type': 'none',
                        'adjustment': 0.0,
                        'adjusted_start': sub.start_time,
                        'adjusted_end': sub.end_time,
                        'adjusted_duration': target_duration
                    })
                    self.processor.complete_block()
                    continue

                print(f"📖 Echo-TTS Subtitle {i+1}/{len(subtitles)}: Processing '{text[:50]}...'")

                narrator_info = (voice_mapping or {}).get('narrator', {})
                narrator_audio = narrator_info.get('audio')
                narrator_reference = narrator_info.get('reference_text', '')
                if narrator_audio is None:
                    narrator_audio = narrator_info.get('audio_path')

                # Split by character tags and generate per-segment audio (with parameters + pause tags + edit tags)
                segment_audio_list: List[torch.Tensor] = []
                segment_records: List[Dict[str, Any]] = []
                segments = character_parser.parse_text_segments(_strip_s1_tag(text))
                if not segments:
                    segments = character_parser.parse_text_segments("narrator " + _strip_s1_tag(text))

                for seg in segments:
                    current_character = seg.character or "narrator"
                    self._check_interrupt(i, len(subtitles), current_character)

                    segment_text = (seg.text or "").strip()
                    if not segment_text:
                        continue

                    speaker_audio = narrator_audio
                    reference_text = narrator_reference

                    if seg.character == "narrator":
                        # Always prefer the provided narrator voice input for narrator segments
                        if narrator_audio is not None:
                            speaker_audio = narrator_audio
                            reference_text = narrator_reference or reference_text
                    elif seg.character and seg.character in character_mapping:
                        char_audio, char_text = character_mapping[seg.character]
                        if char_audio:
                            speaker_audio = char_audio
                            reference_text = char_text or reference_text
                            print(f"📖 Echo-TTS: Using character voice '{seg.character}'")
                        else:
                            print(f"⚠️ Echo-TTS: No voice file found for '{seg.character}', using narrator voice")

                    segment_params = seg.parameters if seg.parameters else {}
                    current_seed = seed + i
                    current_config = self.config
                    if segment_params:
                        current_config = apply_segment_parameters(self.config, segment_params, "echo_tts")
                        if 'seed' in current_config:
                            current_seed = int(current_config.get('seed', current_seed))
                        print(f"📊 Echo-TTS SRT segment: Character '{seg.character}' with parameters {segment_params}")

                    self.processor.update_config(current_config)

                    clean_text, edit_tags = get_edit_tags_for_segment(segment_text)
                    clean_text, pause_segments = PauseTagProcessor.preprocess_text_with_pause_tags(
                        clean_text,
                        enable_pause_tags=True
                    )
                    if pause_segments and any(
                        current_config.get(k) is not None for k in (
                            "speaker_kv_scale", "speaker_kv_max_layers", "speaker_kv_min_t"
                        )
                    ):
                        raise ValueError(
                            "Echo-TTS SRT: Pause tags are not compatible with force_speaker_kv settings. "
                            "Disable force_speaker_kv (speaker_kv_*) or remove pause tags."
                        )

                    pause_mode = pause_segments is not None
                    seed_offset = 0

                    def _tts_generate_func(text_content: str) -> torch.Tensor:
                        nonlocal seed_offset
                        self._check_interrupt(i, len(subtitles), current_character)
                        segment_seed = current_seed + seed_offset
                        seed_offset += 1
                        # Ensure full config (including speaker_kv_*) is applied for each text segment
                        self.processor.update_config(current_config)
                        audio = self.processor.process_text(
                            text=text_content,
                            speaker_audio=speaker_audio,
                            reference_text=reference_text or "",
                            seed=segment_seed,
                            enable_chunking=False,
                            enable_audio_cache=enable_audio_cache,
                            return_info=False
                        )
                        if isinstance(audio, tuple):
                            audio = audio[0]
                        if isinstance(audio, torch.Tensor):
                            audio_tensor = audio
                        else:
                            audio_tensor = torch.tensor(audio, dtype=torch.float32)
                        if audio_tensor.dim() > 1:
                            audio_tensor = audio_tensor.squeeze()
                        return audio_tensor

                    if pause_segments:
                        segment_audio = PauseTagProcessor.generate_audio_with_pauses(
                            pause_segments,
                            _tts_generate_func,
                            sample_rate=self.SAMPLE_RATE
                        )
                        if segment_audio.dim() > 1:
                            segment_audio = segment_audio.squeeze()
                    else:
                        segment_audio = _tts_generate_func(clean_text)

                    segment_audio_list.append(segment_audio)
                    segment_records.append({
                        "waveform": segment_audio,
                        "sample_rate": self.SAMPLE_RATE,
                        "text": clean_text,
                        "edit_tags": edit_tags
                    })

                if segment_records and any(seg["edit_tags"] for seg in segment_records):
                    segment_records = apply_edit_post_processing(segment_records, engine_config=self.config)
                    segment_audio_list = []
                    for seg in segment_records:
                        wf = seg["waveform"]
                        # Normalize 3D [B,C,S] -> 2D [C,S] for timing engine compatibility
                        if hasattr(wf, "dim") and wf.dim() == 3:
                            wf = wf.squeeze(0)
                        seg["waveform"] = wf
                        segment_audio_list.append(wf)

                if segment_audio_list:
                    audio = torch.cat(segment_audio_list, dim=-1)
                else:
                    audio = torch.zeros(1, 0)

                audio_segments.append(audio)

                natural_duration = audio.shape[-1] / self.SAMPLE_RATE
                target_start = sub.start_time
                target_end = sub.end_time
                target_duration = target_end - target_start
                stretch_factor = target_duration / natural_duration if natural_duration > 0 else 1.0

                adjustments.append({
                    'index': i,
                    'segment_index': i,
                    'sequence': sub.sequence,
                    'natural_duration': natural_duration,
                    'target_start': target_start,
                    'target_end': target_end,
                    'target_duration': target_duration,
                    'start_time': target_start,
                    'end_time': target_end,
                    'stretch_factor': stretch_factor,
                    'needs_stretching': abs(stretch_factor - 1.0) > 0.05,
                    'stretch_type': 'compress' if stretch_factor < 1.0 else 'expand' if stretch_factor > 1.0 else 'none',
                    'adjustment': natural_duration - target_duration,
                    'adjusted_start': target_start,
                    'adjusted_end': target_end,
                    'adjusted_duration': natural_duration
                })

                self.processor.complete_block()
                print(f"✅ Echo-TTS Subtitle {i+1}/{len(subtitles)}: {natural_duration:.2f}s (expected {target_duration:.2f}s)")
        finally:
            self.processor.end_job()

        return audio_segments, adjustments

    def _assemble_final_audio(
        self,
        audio_segments: List[torch.Tensor],
        subtitles: List,
        timing_mode: str,
        timing_params: Dict[str, Any],
        adjustments: List[Dict]
    ) -> Tuple[torch.Tensor, Optional[List[Dict]], Optional[str]]:
        """Assemble final audio based on timing mode using modern timing system."""
        if timing_mode == "stretch_to_fit":
            from engines.chatterbox.audio_timing import TimedAudioAssembler
            assembler = TimedAudioAssembler(self.SAMPLE_RATE)
            target_timings = [(sub.start_time, sub.end_time) for sub in subtitles]
            fade_duration = timing_params.get('fade_for_StretchToFit', 0.01)
            final_audio, stretch_method_used = assembler.assemble_timed_audio(
                audio_segments, target_timings, fade_duration=fade_duration
            )
            return final_audio, None, stretch_method_used

        if timing_mode == "pad_with_silence":
            from utils.timing.assembly import AudioAssemblyEngine
            assembler = AudioAssemblyEngine(self.SAMPLE_RATE)
            final_audio = assembler.assemble_with_overlaps(audio_segments, subtitles, torch.device('cpu'))
            return final_audio, None, None

        if timing_mode == "concatenate":
            from utils.timing.engine import TimingEngine
            from utils.timing.assembly import AudioAssemblyEngine
            timing_engine = TimingEngine(self.SAMPLE_RATE)
            assembler = AudioAssemblyEngine(self.SAMPLE_RATE)
            new_adjustments = timing_engine.calculate_concatenation_adjustments(audio_segments, subtitles)
            fade_duration = timing_params.get('fade_for_StretchToFit', 0.01)
            final_audio = assembler.assemble_concatenation(audio_segments, fade_duration)
            return final_audio, new_adjustments, None

        # smart_natural
        from utils.timing.engine import TimingEngine
        from utils.timing.assembly import AudioAssemblyEngine
        timing_engine = TimingEngine(self.SAMPLE_RATE)
        assembler = AudioAssemblyEngine(self.SAMPLE_RATE)

        tolerance = timing_params.get('timing_tolerance', 2.0)
        max_stretch_ratio = timing_params.get('max_stretch_ratio', 1.0)
        min_stretch_ratio = timing_params.get('min_stretch_ratio', 0.5)

        smart_adjustments, processed_segments = timing_engine.calculate_smart_timing_adjustments(
            audio_segments, subtitles, tolerance, max_stretch_ratio, min_stretch_ratio, torch.device('cpu')
        )

        final_audio = assembler.assemble_smart_natural(
            audio_segments, processed_segments, smart_adjustments, subtitles, torch.device('cpu')
        )
        return final_audio, smart_adjustments, None

    def _generate_timing_report(
        self,
        subtitles: List,
        adjustments: List[Dict],
        timing_mode: str,
        has_original_overlaps: bool = False,
        mode_switched: bool = False,
        original_mode: str = None,
        stretch_method: str = None
    ) -> str:
        from utils.timing.reporting import SRTReportGenerator
        reporter = SRTReportGenerator()
        return reporter.generate_timing_report(
            subtitles, adjustments, timing_mode,
            has_original_overlaps, mode_switched, original_mode, stretch_method
        )

    def _generate_adjusted_srt_string(
        self,
        subtitles: List,
        adjustments: List[Dict],
        timing_mode: str
    ) -> str:
        from utils.timing.reporting import SRTReportGenerator
        reporter = SRTReportGenerator()
        return reporter.generate_adjusted_srt_string(subtitles, adjustments, timing_mode)
