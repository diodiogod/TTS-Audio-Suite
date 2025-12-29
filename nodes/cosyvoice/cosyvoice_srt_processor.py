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
            srt_module = import_manager.import_module('srt')
            self.srt = srt_module
        except Exception as e:
            print(f"⚠️ SRT module not available: {e}")
            self.srt = None

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
            self.character_parser.set_character_fallback(alias, target)
        
        # Get character language defaults
        char_lang_defaults = voice_discovery.get_character_language_defaults()
        for char, lang in char_lang_defaults.items():
            self.character_parser.set_character_language_default(char, lang)

    def update_config(self, new_config: Dict[str, Any]):
        """Update processor configuration with new parameters."""
        self.engine_config.update(new_config)
        # Reinitialize processor with new config
        if self.processor:
            self.processor.cleanup()
        self.processor = CosyVoiceProcessor(self.engine_config)

    def process_srt_content(
        self,
        srt_content: str,
        voice_mapping: Dict[str, Any],
        seed: int,
        timing_mode: str,
        timing_params: Dict[str, Any]
    ) -> Tuple[torch.Tensor, str, str]:
        """
        Process complete SRT content and generate timed audio with CosyVoice3.
        This is the main entry point that handles the full SRT workflow.
        
        Args:
            srt_content: Complete SRT subtitle content
            voice_mapping: Voice mapping with character references
            seed: Random seed for reproducibility
            timing_mode: Timing mode (strict, natural, stretch)
            timing_params: Timing parameters
            
        Returns:
            Tuple of (final_audio, adjusted_srt, timing_report)
        """
        if self.srt is None:
            raise ImportError("SRT module not available. Please install with: pip install srt")

        # Check for interrupt before starting
        self._check_interrupt()

        # Parse SRT content
        subtitles = list(self.srt.parse(srt_content))
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
        final_audio = self._assemble_final_audio(
            audio_segments,
            subtitles,
            timing_mode,
            timing_params,
            adjustments
        )

        # Generate timing report
        timing_report = self._generate_timing_report(
            subtitles,
            adjustments,
            timing_mode
        )

        # Generate adjusted SRT
        adjusted_srt = self._generate_adjusted_srt_string(
            subtitles,
            adjustments,
            timing_mode
        )

        return final_audio, adjusted_srt, timing_report

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
            segments = self.character_parser.split_by_character(sub.content, include_language=False)
            for char, _ in segments:
                if char:
                    unique_characters.add(char)

        character_mapping = {}
        if unique_characters:
            character_mapping = get_character_mapping(list(unique_characters), engine_type="cosyvoice")

        for i, sub in enumerate(subtitles):
            # Check for interrupt
            self._check_interrupt()

            # Get subtitle text and character
            text = sub.content.strip()
            if not text:
                # Empty subtitle - create silence
                expected_duration = (sub.end - sub.start).total_seconds()
                silence = torch.zeros(1, int(expected_duration * self.SAMPLE_RATE))
                audio_segments.append(silence)
                adjustments.append({
                    'index': i,
                    'original_start': sub.start,
                    'original_end': sub.end,
                    'actual_duration': expected_duration,
                    'expected_duration': expected_duration,
                    'adjustment': 0.0
                })
                continue

            # Parse character from text
            segments = self.character_parser.split_by_character(text, include_language=False)
            character = segments[0][0] if segments else None
            clean_text = segments[0][1] if segments else text

            # Get speaker audio for this character
            speaker_audio = None
            reference_text = self.engine_config.get('reference_text', '')
            
            if character and character in character_mapping:
                char_audio, char_text = character_mapping[character]
                if char_audio:
                    speaker_audio = char_audio
                    if char_text:
                        reference_text = char_text
                    print(f"📖 Subtitle {i+1}: Using voice '{character}'")
            elif voice_mapping:
                # Use default voice mapping
                default_voice = voice_mapping.get('narrator') or voice_mapping.get('default')
                if default_voice:
                    speaker_audio = default_voice.get('audio_path')
                    reference_text = default_voice.get('reference_text', reference_text)

            # Generate audio
            audio, _ = self.processor.process_text(
                text=clean_text,
                speaker_audio={'audio_path': speaker_audio} if speaker_audio else None,
                seed=seed + i,  # Vary seed per subtitle
                enable_chunking=False,  # SRT handles its own chunking
                enable_pause_tags=True,
                return_info=False
            )

            audio_segments.append(audio)

            # Calculate timing adjustment
            expected_duration = (sub.end - sub.start).total_seconds()
            actual_duration = audio.shape[-1] / self.SAMPLE_RATE
            adjustment = actual_duration - expected_duration

            adjustments.append({
                'index': i,
                'original_start': sub.start,
                'original_end': sub.end,
                'actual_duration': actual_duration,
                'expected_duration': expected_duration,
                'adjustment': adjustment
            })

            print(f"✅ Subtitle {i+1}/{len(subtitles)}: {actual_duration:.2f}s (expected {expected_duration:.2f}s)")

        return audio_segments, adjustments

    def _assemble_final_audio(
        self,
        audio_segments: List[torch.Tensor],
        subtitles: List,
        timing_mode: str,
        timing_params: Dict[str, Any],
        adjustments: List[Dict]
    ) -> torch.Tensor:
        """
        Assemble final audio based on timing mode using existing utils.
        
        Args:
            audio_segments: List of audio segments
            subtitles: List of subtitle objects
            timing_mode: Timing mode
            timing_params: Timing parameters
            adjustments: List of timing adjustments
            
        Returns:
            Final assembled audio tensor
        """
        from utils.audio.srt_timing import SRTTimingAssembler

        assembler = SRTTimingAssembler(sample_rate=self.SAMPLE_RATE)
        
        # Prepare segment data for assembler
        segment_data = []
        for i, (audio, sub, adj) in enumerate(zip(audio_segments, subtitles, adjustments)):
            segment_data.append({
                'audio': audio,
                'start': sub.start,
                'end': sub.end,
                'index': i,
                'adjustment': adj
            })

        # Assemble based on timing mode
        if timing_mode == 'strict':
            final_audio = assembler.assemble_strict(segment_data)
        elif timing_mode == 'natural':
            final_audio = assembler.assemble_natural(segment_data, timing_params)
        elif timing_mode == 'stretch':
            stretch_method = timing_params.get('stretch_method', 'resample')
            final_audio = assembler.assemble_stretch(segment_data, stretch_method)
        else:
            # Default to natural
            final_audio = assembler.assemble_natural(segment_data, timing_params)

        return final_audio

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
        """Generate detailed timing report using existing utils."""
        from utils.audio.srt_timing import generate_timing_report
        
        return generate_timing_report(
            subtitles=subtitles,
            adjustments=adjustments,
            timing_mode=timing_mode,
            sample_rate=self.SAMPLE_RATE,
            has_original_overlaps=has_original_overlaps,
            mode_switched=mode_switched,
            original_mode=original_mode,
            stretch_method=stretch_method
        )

    def _generate_adjusted_srt_string(
        self,
        subtitles: List,
        adjustments: List[Dict],
        timing_mode: str
    ) -> str:
        """Generate adjusted SRT string from final timings using existing utils."""
        from utils.audio.srt_timing import generate_adjusted_srt
        
        return generate_adjusted_srt(
            subtitles=subtitles,
            adjustments=adjustments,
            timing_mode=timing_mode
        )

    def cleanup(self):
        """Clean up resources"""
        if self.processor:
            self.processor.cleanup()
            self.processor = None
