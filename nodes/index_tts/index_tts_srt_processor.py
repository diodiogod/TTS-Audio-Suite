"""
IndexTTS-2 SRT Processor - Handles complete SRT subtitle processing for IndexTTS-2 engine
Called by unified SRT nodes when using IndexTTS-2 engine

This implements the full SRT workflow including timing, assembly, and reports
using the existing modular utilities (same as VibeVoice pattern).

Key Features:
- Character switching with emotion references using existing character parser
- Full SRT timing support (stretch_to_fit, pad_with_silence, smart_natural, concatenate)
- Pause tag support with emotion control
- Duration control and emotion disentanglement
- Smart language grouping using language mapper
"""

import torch
import tempfile
import torchaudio
from typing import Dict, Any, Optional, List, Tuple
import os
import sys
import comfy.model_management as model_management

# Add project root to path
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.system.import_manager import import_manager
from utils.text.character_parser import parse_character_text, character_parser
from utils.text.pause_processor import PauseTagProcessor
from utils.text.segment_parameters import apply_segment_parameters
from utils.audio.processing import AudioProcessingUtils
from utils.voice.discovery import get_available_characters, get_character_mapping
from engines.processors.index_tts_processor import IndexTTSProcessor


class IndexTTSSRTProcessor:
    """
    Complete SRT processor for IndexTTS-2 engine.
    Handles full SRT workflow including timing, assembly, and reports with emotion control.
    Uses the existing IndexTTSProcessor for actual generation.
    """

    def __init__(self, node_instance, engine_config: Dict[str, Any]):
        """
        Initialize IndexTTS-2 SRT processor.

        Args:
            node_instance: Parent node instance
            engine_config: Engine configuration from IndexTTS-2 Engine node
        """
        self.node = node_instance
        self.config = engine_config
        self.sample_rate = 22050

        # Load SRT modules
        self.srt_available = False
        self.srt_modules = {}
        self._load_srt_modules()

        # Initialize the working IndexTTS-2 processor
        self.tts_processor = IndexTTSProcessor(engine_config)

    def _load_srt_modules(self):
        """Load SRT modules using the import manager."""
        success, modules, source = import_manager.import_srt_modules()
        self.srt_available = success
        self.srt_modules = modules

        if success:
            # Extract frequently used classes for easier access
            self.SRTParser = modules.get("SRTParser")
            self.SRTSubtitle = modules.get("SRTSubtitle")
            self.SRTParseError = modules.get("SRTParseError")
            self.AudioTimingUtils = modules.get("AudioTimingUtils")
            self.TimedAudioAssembler = modules.get("TimedAudioAssembler")
            self.calculate_timing_adjustments = modules.get("calculate_timing_adjustments")
            self.AudioTimingError = modules.get("AudioTimingError")
            self.FFmpegTimeStretcher = modules.get("FFmpegTimeStretcher")
            self.PhaseVocoderTimeStretcher = modules.get("PhaseVocoderTimeStretcher")

    def update_config(self, new_config: Dict[str, Any]):
        """Update processor configuration with new parameters."""

        self.config.update(new_config)
        # Also update the IndexTTS processor's config so emotion_audio gets passed through
        if hasattr(self.tts_processor, 'config'):
            self.tts_processor.config.update(new_config)
            # Updated processor configuration with new parameters

    def process_srt_content(self,
                           srt_content: str,
                           voice_mapping: Dict[str, Any],
                           seed: int,
                           timing_mode: str,
                           timing_params: Dict[str, Any]) -> Tuple[torch.Tensor, str, str, str]:
        """
        Process complete SRT content and generate timed audio with IndexTTS-2.
        This is the main entry point that handles the full SRT workflow.

        Args:
            srt_content: Complete SRT subtitle content
            voice_mapping: Mapping of character names to voice references
            seed: Random seed for generation
            timing_mode: SRT timing mode (stretch_to_fit, pad_with_silence, etc.)
            timing_params: Additional timing parameters

        Returns:
            Tuple of (final_audio, generation_info, timing_report, adjusted_srt)
        """
        if not self.srt_available:
            raise ImportError("SRT support not available - missing required modules")

        # Parse SRT content with overlap support
        srt_parser = self.SRTParser()
        subtitles = srt_parser.parse_srt_content(srt_content, allow_overlaps=True)

        # Check for overlaps and handle smart_natural mode fallback
        from utils.timing.overlap_detection import SRTOverlapHandler
        has_overlaps = SRTOverlapHandler.detect_overlaps(subtitles)
        current_timing_mode, mode_switched = SRTOverlapHandler.handle_smart_natural_fallback(
            timing_mode, has_overlaps, "IndexTTS-2 SRT"
        )

        # Set up character parser with available characters
        available_chars = get_available_characters()
        character_parser.set_available_characters(list(available_chars))
        character_parser.reset_session_cache()
        character_parser.set_engine_aware_default_language("IndexTTS-2", "index_tts")

        # Process subtitles and generate audio segments using existing processor
        print(f"🚀 IndexTTS-2 SRT: Processing {len(subtitles)} subtitles with emotion control")

        audio_segments, natural_durations, any_segment_cached = self._process_all_subtitles(
            subtitles, voice_mapping, seed
        )

        # Calculate timing adjustments
        target_timings = [(sub.start_time, sub.end_time) for sub in subtitles]
        adjustments = self.calculate_timing_adjustments(natural_durations, target_timings)

        # Add sequence numbers to adjustments
        for i, (adj, subtitle) in enumerate(zip(adjustments, subtitles)):
            adj['sequence'] = subtitle.sequence

        # Assemble final audio based on timing mode using existing utils
        final_audio, final_adjustments = self._assemble_final_audio(
            audio_segments, subtitles, current_timing_mode, timing_params, adjustments
        )

        # Use final adjustments if returned (for smart_natural mode)
        if final_adjustments is not None:
            adjustments = final_adjustments

        # Generate reports using existing utils
        timing_report = self._generate_timing_report(
            subtitles, adjustments, current_timing_mode, has_overlaps, mode_switched, timing_mode if mode_switched else None
        )
        adjusted_srt_string = self._generate_adjusted_srt_string(subtitles, adjustments, current_timing_mode)

        # Generate info
        total_duration = self.AudioTimingUtils.get_audio_duration(final_audio, self.sample_rate)
        cache_status = "cached" if any_segment_cached else "generated"
        mode_info = f"{current_timing_mode}"
        if mode_switched:
            mode_info = f"{current_timing_mode} (switched from {timing_mode} due to overlaps)"

        info = (f"Generated {total_duration:.1f}s IndexTTS-2 SRT-timed audio from {len(subtitles)} subtitles "
               f"using {mode_info} mode ({cache_status} segments, IndexTTS-2)")

        # Format final audio for ComfyUI (ensure proper 3D format: [batch, channels, samples])
        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif final_audio.dim() == 2:
            final_audio = final_audio.unsqueeze(0)  # Add batch dimension

        # Create proper ComfyUI audio format
        audio_output = {"waveform": final_audio, "sample_rate": self.sample_rate}

        return audio_output, info, timing_report, adjusted_srt_string

    def _process_all_subtitles(self,
                              subtitles: List,
                              voice_mapping: Dict[str, Any],
                              seed: int) -> Tuple[List[torch.Tensor], List[float], bool]:
        """
        Process all subtitles and generate audio segments using existing IndexTTS-2 processor.

        Args:
            subtitles: List of SRT subtitle objects
            voice_mapping: Voice mapping with emotion references
            seed: Random seed

        Returns:
            Tuple of (audio_segments, natural_durations, any_segment_cached)
        """
        audio_segments = [None] * len(subtitles)
        natural_durations = [0.0] * len(subtitles)
        any_segment_cached = False

        # Build speaker audio from voice mapping (narrator fallback)
        speaker_audio = voice_mapping.get("narrator")
        reference_text = ""

        # Process each subtitle using the existing working processor
        for i, subtitle in enumerate(subtitles):
            # Check for interruption before processing each subtitle
            if model_management.interrupt_processing:
                raise InterruptedError(f"IndexTTS-2 SRT subtitle {i+1}/{len(subtitles)} interrupted by user")
            if not subtitle.text.strip():
                # Empty subtitle - create silence
                natural_duration = subtitle.duration
                wav = self.AudioTimingUtils.create_silence(
                    duration_seconds=natural_duration,
                    sample_rate=self.sample_rate,
                    channels=1,
                    device=torch.device('cpu')
                )
                print(f"🤫 IndexTTS-2 SRT Subtitle {i+1} (Seq {subtitle.sequence}): Empty text, generating {natural_duration:.2f}s silence.")
                audio_segments[i] = wav
                natural_durations[i] = natural_duration
                continue

            print(f"🎭 IndexTTS-2 SRT Subtitle {i+1}/{len(subtitles)} (Seq {subtitle.sequence}): Processing '{subtitle.text[:50]}...'")

            # Use existing processor with all character switching and emotion support
            wav = self.tts_processor.process_text(
                text=subtitle.text,
                speaker_audio=speaker_audio,
                reference_text=reference_text,
                seed=seed + i,  # Vary seed per subtitle
                enable_chunking=False,  # Disable chunking for SRT segments
                max_chars_per_chunk=400,
                silence_between_chunks_ms=100
            )

            # Ensure correct tensor format
            if wav.dim() == 3:
                wav = wav.squeeze(0)
            elif wav.dim() == 1:
                wav = wav.unsqueeze(0)

            # Calculate duration and store
            natural_duration = self.AudioTimingUtils.get_audio_duration(wav, self.sample_rate)
            audio_segments[i] = wav
            natural_durations[i] = natural_duration

        return audio_segments, natural_durations, any_segment_cached


    def _assemble_final_audio(self,
                             audio_segments: List[torch.Tensor],
                             subtitles: List,
                             timing_mode: str,
                             timing_params: Dict[str, Any],
                             adjustments: List[Dict]) -> Tuple[torch.Tensor, Optional[List[Dict]]]:
        """
        Assemble final audio based on timing mode using existing utils.

        Args:
            audio_segments: List of audio segments
            subtitles: List of subtitle objects
            timing_mode: Timing mode
            timing_params: Timing parameters
            adjustments: Timing adjustments

        Returns:
            Tuple of (final_audio_tensor, updated_adjustments_or_None)
        """
        if timing_mode == "stretch_to_fit":
            # Use time stretching to match exact timing
            assembler = self.TimedAudioAssembler(self.sample_rate)
            target_timings = [(sub.start_time, sub.end_time) for sub in subtitles]
            fade_duration = timing_params.get('fade_for_StretchToFit', 0.01)
            final_audio = assembler.assemble_timed_audio(
                audio_segments, target_timings, fade_duration=fade_duration
            )
            return final_audio, None  # No updated adjustments

        elif timing_mode == "pad_with_silence":
            # Add silence to match timing without stretching
            from utils.timing.assembly import AudioAssemblyEngine
            assembler = AudioAssemblyEngine(self.sample_rate)
            final_audio = assembler.assemble_with_overlaps(audio_segments, subtitles, torch.device('cpu'))
            return final_audio, None  # No updated adjustments

        elif timing_mode == "concatenate":
            # Concatenate audio naturally and recalculate SRT timings
            from utils.timing.engine import TimingEngine
            from utils.timing.assembly import AudioAssemblyEngine

            timing_engine = TimingEngine(self.sample_rate)
            assembler = AudioAssemblyEngine(self.sample_rate)

            # Calculate new timings for concatenation
            new_adjustments = timing_engine.calculate_concatenation_adjustments(audio_segments, subtitles)

            # Assemble audio with optional crossfading
            fade_duration = timing_params.get('fade_for_StretchToFit', 0.01)
            final_audio = assembler.assemble_concatenation(audio_segments, fade_duration)
            return final_audio, new_adjustments  # Return updated adjustments

        else:  # smart_natural
            # Smart balanced timing: use natural audio but add minimal adjustments within tolerance
            from utils.timing.engine import TimingEngine
            from utils.timing.assembly import AudioAssemblyEngine

            timing_engine = TimingEngine(self.sample_rate)
            assembler = AudioAssemblyEngine(self.sample_rate)

            tolerance = timing_params.get('timing_tolerance', 2.0)
            max_stretch_ratio = timing_params.get('max_stretch_ratio', 1.0)
            min_stretch_ratio = timing_params.get('min_stretch_ratio', 0.5)

            # Calculate smart adjustments - these have the detailed fields needed for reporting
            smart_adjustments, processed_segments = timing_engine.calculate_smart_timing_adjustments(
                audio_segments, subtitles, tolerance, max_stretch_ratio, min_stretch_ratio, torch.device('cpu')
            )

            # Assemble with smart natural timing
            final_audio = assembler.assemble_smart_natural(
                audio_segments, processed_segments, smart_adjustments, subtitles, torch.device('cpu')
            )
            return final_audio, smart_adjustments  # Return the detailed smart adjustments

    def _generate_timing_report(self, subtitles: List, adjustments: List[Dict], timing_mode: str,
                               has_original_overlaps: bool = False, mode_switched: bool = False,
                               original_mode: str = None) -> str:
        """Generate detailed timing report using existing utils."""
        from utils.timing.reporting import SRTReportGenerator
        reporter = SRTReportGenerator()
        return reporter.generate_timing_report(subtitles, adjustments, timing_mode,
                                             has_original_overlaps, mode_switched, original_mode)

    def _generate_adjusted_srt_string(self, subtitles: List, adjustments: List[Dict], timing_mode: str) -> str:
        """Generate adjusted SRT string from final timings using existing utils."""
        from utils.timing.reporting import SRTReportGenerator
        reporter = SRTReportGenerator()
        return reporter.generate_adjusted_srt_string(subtitles, adjustments, timing_mode)

    def cleanup(self):
        """Clean up resources"""
        if self.tts_processor:
            self.tts_processor.cleanup()