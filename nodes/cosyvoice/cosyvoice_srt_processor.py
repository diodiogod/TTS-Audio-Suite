"""
CosyVoice3 SRT Processor - Handles complete SRT subtitle processing for CosyVoice3 engine
Called by unified SRT nodes when using CosyVoice3 engine

This implements the full SRT workflow including timing, assembly, and reports.
"""

import torch
from typing import Dict, Any, Optional, List, Tuple
import os
import sys

# Add project root to path
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.text.character_parser import CharacterParser
from utils.text.segment_parameters import apply_segment_parameters
from utils.audio.processing import AudioProcessingUtils
from utils.voice.discovery import get_available_characters, get_character_mapping, voice_discovery
from engines.processors.cosyvoice_processor import CosyVoiceProcessor


class CosyVoiceSRTProcessor:
    """
    Complete SRT processor for CosyVoice3 engine.
    Handles full SRT workflow including timing, assembly, and reports.
    Uses the existing CosyVoiceProcessor for actual generation.
    """

    SAMPLE_RATE = 24000  # CosyVoice3 native sample rate

    def __init__(self, node_instance, engine_config: Dict[str, Any]):
        """
        Initialize CosyVoice3 SRT processor.
        
        Args:
            node_instance: Parent node instance
            engine_config: Engine configuration from CosyVoice3 Engine node
        """
        self.node_instance = node_instance
        self.engine_config = engine_config
        
        # Initialize processor
        self.processor = CosyVoiceProcessor(engine_config)
        
        # Load SRT modules
        self._load_srt_modules()
        
        # Setup character parser
        self._setup_character_parser()

    def _load_srt_modules(self):
        """Load SRT modules using the import manager."""
        try:
            from utils.system.import_manager import import_manager
            success, srt_modules, msg = import_manager.import_srt_modules()
            if success:
                # Extract SRT classes from modules dict
                self.SRTParser = srt_modules.get('SRTParser')
                self.SRTSubtitle = srt_modules.get('SRTSubtitle')
                self.SRTParseError = srt_modules.get('SRTParseError')
                self.AudioTimingUtils = srt_modules.get('AudioTimingUtils')
                self.TimedAudioAssembler = srt_modules.get('TimedAudioAssembler')
                self.calculate_timing_adjustments = srt_modules.get('calculate_timing_adjustments')
                self.AudioTimingError = srt_modules.get('AudioTimingError')
                self.srt_available = True
            else:
                print(f"⚠️ SRT module not available: {msg}")
                self.srt_available = False
        except Exception as e:
            print(f"⚠️ SRT module not available: {e}")
            self.srt_available = False

    def _setup_character_parser(self):
        """Set up character parser with available characters and aliases."""
        self.character_parser = CharacterParser()
        
        # Get available characters
        available_chars = get_available_characters()
        if available_chars:
            self.character_parser.set_available_characters(list(available_chars))
        
        # Get character aliases
        character_aliases = voice_discovery.get_character_aliases()
        for alias, target in character_aliases.items():
            self.character_parser.add_character_fallback(alias, target)
        
        # Get character language defaults
        char_lang_defaults = voice_discovery.get_character_language_defaults()
        for char, lang in char_lang_defaults.items():
            self.character_parser.set_character_language_default(char, lang)

    def update_config(self, new_config: Dict[str, Any]):
        """Update processor configuration with new parameters."""
        self.engine_config.update(new_config)
        # Update processor's dynamic config (speed, reference_text, instruct_text)
        # Model reloading is handled by unified model loader based on cache key
        if self.processor:
            self.processor.update_config(new_config)

    def process_srt_content(
        self,
        srt_content: str,
        voice_mapping: Dict[str, Any],
        seed: int,
        timing_mode: str,
        timing_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str, str, str]:
        """
        Process complete SRT content and generate timed audio with CosyVoice3.
        This is the main entry point that handles the full SRT workflow.

        Args:
            srt_content: Complete SRT subtitle content
            voice_mapping: Voice mapping with character references
            seed: Random seed for reproducibility
            timing_mode: Timing mode (stretch_to_fit, pad_with_silence, concatenate, smart_natural)
            timing_params: Timing parameters

        Returns:
            Tuple of (audio_output, generation_info, timing_report, adjusted_srt)
        """
        if not self.srt_available:
            raise ImportError("SRT modules not available - missing required SRT parser")

        # Check for interrupt before starting
        self._check_interrupt()

        # Parse SRT content
        srt_parser = self.SRTParser()
        subtitles = srt_parser.parse_srt_content(srt_content, allow_overlaps=True)
        total_subtitles = len(subtitles)
        
        print(f"🎬 CosyVoice3 SRT: Processing {total_subtitles} subtitle(s)")

        # Generate audio for all subtitles
        audio_segments, adjustments = self._process_all_subtitles(
            subtitles,
            voice_mapping,
            seed
        )

        # Check for interrupt
        self._check_interrupt()

        # Assemble final audio based on timing mode
        final_audio, final_adjustments, stretch_method = self._assemble_final_audio(
            audio_segments,
            subtitles,
            timing_mode,
            timing_params,
            adjustments
        )

        # Update adjustments if assembly returned new ones
        if final_adjustments is not None:
            adjustments = final_adjustments

        # Generate timing report
        timing_report = self._generate_timing_report(
            subtitles,
            adjustments,
            timing_mode,
            has_original_overlaps=False,
            mode_switched=False,
            original_mode=None,
            stretch_method=stretch_method
        )

        # Generate adjusted SRT
        adjusted_srt = self._generate_adjusted_srt_string(
            subtitles,
            adjustments,
            timing_mode
        )

        # Generate info
        total_duration = final_audio.shape[-1] / self.SAMPLE_RATE
        mode_info = f"{timing_mode}"

        info = (f"Generated {total_duration:.1f}s CosyVoice3 SRT-timed audio from {len(subtitles)} subtitles "
               f"using {mode_info} mode")

        # Format final audio for ComfyUI (ensure proper 3D format: [batch, channels, samples])
        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0).unsqueeze(0)
        elif final_audio.dim() == 2:
            final_audio = final_audio.unsqueeze(0)

        audio_output = {"waveform": final_audio, "sample_rate": self.SAMPLE_RATE}

        return audio_output, info, timing_report, adjusted_srt

    def _check_interrupt(self):
        """Check if generation should be interrupted."""
        if hasattr(self.node_instance, 'check_interrupt'):
            self.node_instance.check_interrupt()

    def _process_all_subtitles(
        self,
        subtitles: List,
        voice_mapping: Dict[str, Any],
        seed: int
    ) -> Tuple[List[torch.Tensor], List[Dict]]:
        """
        Process all subtitles and generate audio segments using CosyVoice3 processor.
        
        Args:
            subtitles: List of SRT subtitle objects
            voice_mapping: Voice mapping with character references
            seed: Random seed
            
        Returns:
            Tuple of (audio_segments, adjustments)
        """
        audio_segments = []
        adjustments = []

        # Get character mapping for all unique characters in subtitles
        unique_characters = set()
        for sub in subtitles:
            # Parse character from subtitle text
            segments = self.character_parser.split_by_character(sub.text, include_language=False)
            for char, _ in segments:
                if char:
                    unique_characters.add(char)

        print(f"🔍 Unique characters found in SRT: {sorted(unique_characters)}")

        character_mapping = {}
        if unique_characters:
            character_mapping = get_character_mapping(list(unique_characters), engine_type="cosyvoice")
            print(f"🎭 Character mapping keys: {list(character_mapping.keys())}")

        for i, sub in enumerate(subtitles):
            # Check for interrupt
            self._check_interrupt()

            # Get subtitle text
            text = sub.text.strip()
            if not text:
                # Empty subtitle - create silence
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
                continue

            print(f"📖 Subtitle {i+1}/{len(subtitles)}: Processing '{text[:50]}...'")

            # Build character mapping for processor
            # Priority: connected narrator > character folders
            processor_character_mapping = {}

            # Start with character folder mappings
            for char_name, (char_audio, char_text) in character_mapping.items():
                processor_character_mapping[char_name] = (char_audio, char_text)

            # Override narrator if connected narrator is available
            if voice_mapping and voice_mapping.get('narrator'):
                narrator_voice = voice_mapping['narrator']
                narrator_audio = narrator_voice.get('audio_path')
                narrator_text = narrator_voice.get('reference_text', '')
                processor_character_mapping['narrator'] = (narrator_audio, narrator_text)

            # Use processor to handle character switching and multi-line segments
            audio, _ = self.processor.process_text(
                text=text,
                speaker_audio=voice_mapping.get('narrator') if voice_mapping else None,
                seed=seed + i,
                enable_chunking=False,  # SRT handles timing
                enable_pause_tags=True,
                return_info=False,
                character_mapping=processor_character_mapping
            )

            audio_segments.append(audio)

            # Calculate timing adjustment with all required fields
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
                'adjusted_start': target_start,  # Will be updated by timing assembly
                'adjusted_end': target_end,      # Will be updated by timing assembly
                'adjusted_duration': natural_duration
            })

            print(f"✅ Subtitle {i+1}/{len(subtitles)}: {natural_duration:.2f}s (expected {target_duration:.2f}s)")

        return audio_segments, adjustments

    def _assemble_final_audio(
        self,
        audio_segments: List[torch.Tensor],
        subtitles: List,
        timing_mode: str,
        timing_params: Dict[str, Any],
        adjustments: List[Dict]
    ) -> Tuple[torch.Tensor, Optional[List[Dict]], Optional[str]]:
        """
        Assemble final audio based on timing mode using modern timing system.

        Args:
            audio_segments: List of audio segments
            subtitles: List of subtitle objects
            timing_mode: Timing mode (stretch_to_fit, pad_with_silence, concatenate, smart_natural)
            timing_params: Timing parameters
            adjustments: List of timing adjustments

        Returns:
            Tuple of (final_audio, updated_adjustments_or_None, stretch_method_or_None)
        """
        if timing_mode == "stretch_to_fit":
            assembler = self.TimedAudioAssembler(self.SAMPLE_RATE)
            target_timings = [(sub.start_time, sub.end_time) for sub in subtitles]
            fade_duration = timing_params.get('fade_for_StretchToFit', 0.01)
            final_audio, stretch_method_used = assembler.assemble_timed_audio(
                audio_segments, target_timings, fade_duration=fade_duration
            )
            return final_audio, None, stretch_method_used

        elif timing_mode == "pad_with_silence":
            from utils.timing.assembly import AudioAssemblyEngine
            assembler = AudioAssemblyEngine(self.SAMPLE_RATE)
            final_audio = assembler.assemble_with_overlaps(audio_segments, subtitles, torch.device('cpu'))
            return final_audio, None, None

        elif timing_mode == "concatenate":
            from utils.timing.engine import TimingEngine
            from utils.timing.assembly import AudioAssemblyEngine

            timing_engine = TimingEngine(self.SAMPLE_RATE)
            assembler = AudioAssemblyEngine(self.SAMPLE_RATE)

            new_adjustments = timing_engine.calculate_concatenation_adjustments(audio_segments, subtitles)
            fade_duration = timing_params.get('fade_for_StretchToFit', 0.01)
            final_audio = assembler.assemble_concatenation(audio_segments, fade_duration)
            return final_audio, new_adjustments, None

        else:  # smart_natural
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
        """Generate detailed timing report using modern reporting system."""
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
        """Generate adjusted SRT string from final timings using modern reporting system."""
        from utils.timing.reporting import SRTReportGenerator
        reporter = SRTReportGenerator()
        return reporter.generate_adjusted_srt_string(subtitles, adjustments, timing_mode)

    def cleanup(self):
        """Clean up resources"""
        if self.processor:
            self.processor.cleanup()
            self.processor = None
