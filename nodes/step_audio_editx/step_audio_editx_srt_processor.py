"""
Step Audio EditX SRT Processor - Handles complete SRT subtitle processing
Called by unified SRT nodes when using Step Audio EditX engine

Delegates to StepAudioEditXProcessor for actual TTS generation,
following the IndexTTS-2 pattern for SRT orchestration.
"""

import torch
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
from utils.text.character_parser import character_parser
from utils.voice.discovery import get_available_characters, voice_discovery

# Import StepAudioEditXProcessor using file-based import (avoids package issues)
import importlib.util
step_audio_processor_path = os.path.join(current_dir, "step_audio_editx_processor.py")
step_audio_spec = importlib.util.spec_from_file_location("step_audio_editx_processor_module", step_audio_processor_path)
step_audio_module = importlib.util.module_from_spec(step_audio_spec)
step_audio_spec.loader.exec_module(step_audio_module)
StepAudioEditXProcessor = step_audio_module.StepAudioEditXProcessor


class StepAudioEditXSRTProcessor:
    """
    Complete SRT processor for Step Audio EditX engine.
    Handles full SRT workflow including timing, assembly, and reports.
    Delegates to StepAudioEditXProcessor for actual TTS generation.
    """

    def __init__(self, node_instance, engine_config: Dict[str, Any]):
        """
        Initialize Step Audio EditX SRT processor.

        Args:
            node_instance: Parent node instance
            engine_config: Engine configuration from Step Audio EditX Engine node
        """
        self.node = node_instance
        self.config = engine_config
        self.sample_rate = 24000  # Step Audio EditX native sample rate

        # Load SRT modules
        self.srt_available = False
        self.srt_modules = {}
        self._load_srt_modules()

        # Initialize the TTS processor for actual generation
        self.tts_processor = StepAudioEditXProcessor(node_instance, engine_config)

    def _load_srt_modules(self):
        """Load SRT modules using the import manager."""
        success, modules, source = import_manager.import_srt_modules()
        self.srt_available = success
        self.srt_modules = modules

        if success:
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
        if hasattr(self.tts_processor, 'update_config'):
            self.tts_processor.update_config(new_config)

    def process_srt_content(self,
                           srt_content: str,
                           voice_mapping: Dict[str, Any],
                           seed: int,
                           timing_mode: str,
                           timing_params: Dict[str, Any]) -> Tuple[torch.Tensor, str, str, str]:
        """
        Process complete SRT content and generate timed audio.

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
            timing_mode, has_overlaps, "Step Audio EditX SRT"
        )

        # Set up character parser with available characters
        available_chars = get_available_characters()
        character_aliases = voice_discovery.get_character_aliases()

        all_available = set()
        if available_chars:
            all_available.update(available_chars)
        for alias, target in character_aliases.items():
            all_available.add(alias.lower())
            all_available.add(target.lower())
        all_available.add("narrator")

        character_parser.set_available_characters(list(all_available))
        character_parser.reset_session_cache()

        # Set language defaults
        char_lang_defaults = voice_discovery.get_character_language_defaults()
        for char, lang in char_lang_defaults.items():
            character_parser.set_character_language_default(char, lang)

        # Build complete voice mapping for all characters in subtitles
        voice_mapping = self._build_voice_mapping(subtitles, voice_mapping)

        print(f"🚀 Step Audio EditX SRT: Processing {len(subtitles)} subtitles")

        # Process subtitles and generate audio segments
        audio_segments, natural_durations, any_segment_cached = self._process_all_subtitles(
            subtitles, voice_mapping, seed
        )

        # Calculate timing adjustments
        target_timings = [(sub.start_time, sub.end_time) for sub in subtitles]
        adjustments = self.calculate_timing_adjustments(natural_durations, target_timings)

        # Add sequence numbers to adjustments
        for i, (adj, subtitle) in enumerate(zip(adjustments, subtitles)):
            adj['sequence'] = subtitle.sequence

        # Assemble final audio based on timing mode
        final_audio, final_adjustments = self._assemble_final_audio(
            audio_segments, subtitles, current_timing_mode, timing_params, adjustments
        )

        if final_adjustments is not None:
            adjustments = final_adjustments

        # Generate reports
        timing_report = self._generate_timing_report(
            subtitles, adjustments, current_timing_mode, has_overlaps, mode_switched,
            timing_mode if mode_switched else None
        )
        adjusted_srt_string = self._generate_adjusted_srt_string(subtitles, adjustments, current_timing_mode)

        # Generate info
        total_duration = self.AudioTimingUtils.get_audio_duration(final_audio, self.sample_rate)
        cache_status = "cached" if any_segment_cached else "generated"
        mode_info = f"{current_timing_mode}"
        if mode_switched:
            mode_info = f"{current_timing_mode} (switched from {timing_mode} due to overlaps)"

        info = (f"Generated {total_duration:.1f}s Step Audio EditX SRT-timed audio from {len(subtitles)} subtitles "
               f"using {mode_info} mode ({cache_status} segments)")

        # Format final audio for ComfyUI (ensure proper 3D format: [batch, channels, samples])
        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0).unsqueeze(0)
        elif final_audio.dim() == 2:
            final_audio = final_audio.unsqueeze(0)

        audio_output = {"waveform": final_audio, "sample_rate": self.sample_rate}

        return audio_output, info, timing_report, adjusted_srt_string

    def _build_voice_mapping(self, subtitles: List, narrator_voice: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build complete voice mapping for all characters in subtitles.

        Args:
            subtitles: List of SRT subtitle objects
            narrator_voice: Voice reference for narrator (from SRT node input)

        Returns:
            Complete voice mapping with all characters
        """
        from utils.voice.discovery import get_character_mapping
        import re

        # Extract all character names from subtitle text
        all_characters = set()
        for subtitle in subtitles:
            # Find character tags like [Alice] or [Bob|seed:42]
            tags = re.findall(r'\[([^\]]+)\]', subtitle.text)
            for tag in tags:
                if not tag.startswith('pause:'):
                    char_name = tag.split('|')[0].strip().lower()
                    all_characters.add(char_name)

        # Always include narrator
        all_characters.add("narrator")

        # Also add resolved aliases (character parser resolves alice -> female_01)
        character_aliases = voice_discovery.get_character_aliases()
        resolved_characters = set()
        for char in all_characters:
            resolved_characters.add(char)
            # If this is an alias, also add the resolved name
            if char in character_aliases:
                resolved_characters.add(character_aliases[char].lower())
            # If this is a target, also check if any alias points to it
            for alias, target in character_aliases.items():
                if target.lower() == char:
                    resolved_characters.add(alias.lower())

        # Get character voice mapping from discovery system
        character_mapping = get_character_mapping(list(resolved_characters), engine_type="step_audio_editx")

        # Build voice mapping with fallback to narrator
        voice_mapping = {}
        narrator_ref = narrator_voice.get("narrator", {})

        for character in resolved_characters:
            if character == "narrator":
                voice_mapping[character] = narrator_ref
            else:
                # Try character-specific voice first
                audio_path, ref_text = character_mapping.get(character, (None, None))
                if audio_path and ref_text:
                    voice_mapping[character] = {
                        'prompt_audio_path': audio_path,
                        'prompt_text': ref_text
                    }
                    print(f"🎭 SRT: Using character voice for '{character}'")
                else:
                    # Fallback to narrator voice
                    voice_mapping[character] = narrator_ref
                    print(f"🔄 SRT: Using narrator voice fallback for '{character}'")

        return voice_mapping

    def _process_all_subtitles(self,
                              subtitles: List,
                              voice_mapping: Dict[str, Any],
                              seed: int) -> Tuple[List[torch.Tensor], List[float], bool]:
        """
        Process all subtitles and generate audio segments.

        Args:
            subtitles: List of SRT subtitle objects
            voice_mapping: Voice mapping
            seed: Random seed

        Returns:
            Tuple of (audio_segments, natural_durations, any_segment_cached)
        """
        audio_segments = [None] * len(subtitles)
        natural_durations = [0.0] * len(subtitles)
        any_segment_cached = False

        # Start job tracking at SRT level with all subtitle text lengths
        subtitle_texts = [len(sub.text) for sub in subtitles]
        self.tts_processor.adapter.start_job(total_blocks=len(subtitles), block_texts=subtitle_texts)

        for i, subtitle in enumerate(subtitles):
            # Check for interruption
            if model_management.interrupt_processing:
                raise InterruptedError(f"Step Audio EditX SRT subtitle {i+1}/{len(subtitles)} interrupted by user")

            if not subtitle.text.strip():
                # Empty subtitle - create silence
                natural_duration = subtitle.duration
                wav = self.AudioTimingUtils.create_silence(
                    duration_seconds=natural_duration,
                    sample_rate=self.sample_rate,
                    channels=1,
                    device=torch.device('cpu')
                )
                print(f"🤫 Subtitle {i+1} (Seq {subtitle.sequence}): Empty text, generating {natural_duration:.2f}s silence.")
                audio_segments[i] = wav
                natural_durations[i] = natural_duration
                continue

            print(f"🎭 Subtitle {i+1}/{len(subtitles)} (Seq {subtitle.sequence}): Processing '{subtitle.text[:50]}...'")

            # Set current block for progress tracking
            self.tts_processor.adapter.set_current_block(i)

            # Use existing processor - it handles character switching internally
            # Skip internal job tracking since we manage it at SRT level
            segments = self.tts_processor.process_text(
                text=subtitle.text,
                voice_mapping=voice_mapping,
                seed=seed + i,  # Vary seed per subtitle
                enable_chunking=False,  # Disable chunking for SRT segments
                max_chars_per_chunk=400
            )

            # Mark block as complete for progress tracking
            self.tts_processor.adapter.complete_block()

            # Combine segments if multiple (character switching within subtitle)
            if len(segments) == 1:
                wav = segments[0]['waveform']
            else:
                combined = self.tts_processor.combine_audio_segments(segments, method="auto", silence_ms=50)
                wav = combined

            # Ensure correct tensor format
            if wav.dim() == 3:
                wav = wav.squeeze(0)
            elif wav.dim() == 1:
                wav = wav.unsqueeze(0)

            natural_duration = self.AudioTimingUtils.get_audio_duration(wav, self.sample_rate)
            audio_segments[i] = wav
            natural_durations[i] = natural_duration

        # End job tracking
        self.tts_processor.adapter.end_job()

        return audio_segments, natural_durations, any_segment_cached

    def _assemble_final_audio(self,
                             audio_segments: List[torch.Tensor],
                             subtitles: List,
                             timing_mode: str,
                             timing_params: Dict[str, Any],
                             adjustments: List[Dict]) -> Tuple[torch.Tensor, Optional[List[Dict]]]:
        """
        Assemble final audio based on timing mode.

        Returns:
            Tuple of (final_audio_tensor, updated_adjustments_or_None)
        """
        if timing_mode == "stretch_to_fit":
            assembler = self.TimedAudioAssembler(self.sample_rate)
            target_timings = [(sub.start_time, sub.end_time) for sub in subtitles]
            fade_duration = timing_params.get('fade_for_StretchToFit', 0.01)
            final_audio = assembler.assemble_timed_audio(
                audio_segments, target_timings, fade_duration=fade_duration
            )
            return final_audio, None

        elif timing_mode == "pad_with_silence":
            from utils.timing.assembly import AudioAssemblyEngine
            assembler = AudioAssemblyEngine(self.sample_rate)
            final_audio = assembler.assemble_with_overlaps(audio_segments, subtitles, torch.device('cpu'))
            return final_audio, None

        elif timing_mode == "concatenate":
            from utils.timing.engine import TimingEngine
            from utils.timing.assembly import AudioAssemblyEngine

            timing_engine = TimingEngine(self.sample_rate)
            assembler = AudioAssemblyEngine(self.sample_rate)

            new_adjustments = timing_engine.calculate_concatenation_adjustments(audio_segments, subtitles)
            fade_duration = timing_params.get('fade_for_StretchToFit', 0.01)
            final_audio = assembler.assemble_concatenation(audio_segments, fade_duration)
            return final_audio, new_adjustments

        else:  # smart_natural
            from utils.timing.engine import TimingEngine
            from utils.timing.assembly import AudioAssemblyEngine

            timing_engine = TimingEngine(self.sample_rate)
            assembler = AudioAssemblyEngine(self.sample_rate)

            tolerance = timing_params.get('timing_tolerance', 2.0)
            max_stretch_ratio = timing_params.get('max_stretch_ratio', 1.0)
            min_stretch_ratio = timing_params.get('min_stretch_ratio', 0.5)

            smart_adjustments, processed_segments = timing_engine.calculate_smart_timing_adjustments(
                audio_segments, subtitles, tolerance, max_stretch_ratio, min_stretch_ratio, torch.device('cpu')
            )

            final_audio = assembler.assemble_smart_natural(
                audio_segments, processed_segments, smart_adjustments, subtitles, torch.device('cpu')
            )
            return final_audio, smart_adjustments

    def _generate_timing_report(self, subtitles: List, adjustments: List[Dict], timing_mode: str,
                               has_original_overlaps: bool = False, mode_switched: bool = False,
                               original_mode: str = None) -> str:
        """Generate detailed timing report."""
        from utils.timing.reporting import SRTReportGenerator
        reporter = SRTReportGenerator()
        return reporter.generate_timing_report(subtitles, adjustments, timing_mode,
                                             has_original_overlaps, mode_switched, original_mode)

    def _generate_adjusted_srt_string(self, subtitles: List, adjustments: List[Dict], timing_mode: str) -> str:
        """Generate adjusted SRT string from final timings."""
        from utils.timing.reporting import SRTReportGenerator
        reporter = SRTReportGenerator()
        return reporter.generate_adjusted_srt_string(subtitles, adjustments, timing_mode)

    def cleanup(self):
        """Clean up resources"""
        if self.tts_processor:
            self.tts_processor.cleanup()
