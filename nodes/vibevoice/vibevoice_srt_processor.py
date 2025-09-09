"""
VibeVoice SRT Processor - Handles complete SRT subtitle processing for VibeVoice engine
Called by unified SRT nodes when using VibeVoice engine

This implements the full SRT workflow including timing, assembly, and reports
using the existing modular utilities (same as ChatterBox and F5-TTS).

Key Features:
- Native Multi-Speaker Mode: One segment per subtitle using Speaker 1:, Speaker 2: format
- Custom Character Switching Mode: Multiple segments per subtitle (one per character)
- Full SRT timing support (stretch_to_fit, pad_with_silence, smart_natural, concatenate)
- Pause tag support in both modes
- Character switching with voice discovery
- Smart language grouping and model switching
"""

import torch
import hashlib
from typing import Dict, Any, Optional, List, Tuple
import os
import sys

# Add project root to path
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.system.import_manager import import_manager
from utils.text.character_parser import parse_character_text, character_parser
from utils.text.pause_processor import PauseTagProcessor
from utils.audio.processing import AudioProcessingUtils
from utils.voice.discovery import get_available_characters, get_character_mapping
from engines.adapters.vibevoice_adapter import VibeVoiceEngineAdapter


class VibeVoiceSRTProcessor:
    """
    Complete SRT processor for VibeVoice engine.
    Handles full SRT workflow including timing, assembly, and reports.
    """
    
    def __init__(self, node_instance, engine_config: Dict[str, Any]):
        """
        Initialize VibeVoice SRT processor.
        
        Args:
            node_instance: Parent node instance
            engine_config: Engine configuration from VibeVoice Engine node
        """
        self.node = node_instance
        self.config = engine_config
        self.adapter = VibeVoiceEngineAdapter(node_instance)
        self.pause_processor = PauseTagProcessor()
        
        # Load SRT modules
        self.srt_available = False
        self.srt_modules = {}
        self._load_srt_modules()
        
        # Load model with enhanced parameters
        model_name = engine_config.get('model', 'vibevoice-1.5B')
        device = engine_config.get('device', 'auto')
        attention_mode = engine_config.get('attention_mode', 'auto')
        quantize_llm_4bit = engine_config.get('quantize_llm_4bit', False)
        
        # Load model with new parameters (Credits: based on wildminder/ComfyUI-VibeVoice enhancements)
        self.adapter.load_base_model(model_name, device, attention_mode, quantize_llm_4bit)
    
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
    
    def process_srt_content(self,
                           srt_content: str,
                           voice_mapping: Dict[str, Any],
                           seed: int,
                           timing_mode: str,
                           timing_params: Dict[str, Any]) -> Tuple[torch.Tensor, str, str, str]:
        """
        Process complete SRT content and generate timed audio.
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
            timing_mode, has_overlaps, "VibeVoice SRT"
        )
        
        # Set up character parser with available characters
        available_chars = get_available_characters()
        character_parser.set_available_characters(list(available_chars))
        character_parser.reset_session_cache()
        character_parser.set_engine_aware_default_language(self.config.get('model', 'vibevoice-1.5B'), "vibevoice")
        
        # Process subtitles and generate audio segments  
        current_mode = self.config.get('multi_speaker_mode', 'Custom Character Switching')
        print(f"🚀 VibeVoice SRT: Processing {len(subtitles)} subtitles in {current_mode} mode")
        
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
        total_duration = self.AudioTimingUtils.get_audio_duration(final_audio, 24000)
        cache_status = "cached" if any_segment_cached else "generated"
        mode_info = f"{current_timing_mode}"
        if mode_switched:
            mode_info = f"{current_timing_mode} (switched from {timing_mode} due to overlaps)"
        
        info = (f"Generated {total_duration:.1f}s VibeVoice SRT-timed audio from {len(subtitles)} subtitles "
               f"using {mode_info} mode ({cache_status} segments, {self.config.get('model', 'vibevoice-1.5B')})")
        
        # Format final audio for ComfyUI (ensure proper 3D format: [batch, channels, samples])
        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif final_audio.dim() == 2:
            final_audio = final_audio.unsqueeze(0)  # Add batch dimension
        
        # Create proper ComfyUI audio format
        audio_output = {"waveform": final_audio, "sample_rate": 24000}
        
        return audio_output, info, timing_report, adjusted_srt_string
    
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
        
        # Build complete voice references including character-specific voices
        complete_voice_refs = voice_mapping.copy()
        try:
            available_chars = get_available_characters()
            char_mapping = get_character_mapping(list(available_chars), "audio_only")
            for char in available_chars:
                char_audio_path, _ = char_mapping.get(char, (None, None))
                if char_audio_path:
                    complete_voice_refs[char] = {"audio_path": char_audio_path}
        except ImportError:
            pass
        
        # Process each subtitle
        for i, subtitle in enumerate(subtitles):
            if not subtitle.text.strip():
                # Empty subtitle - create silence
                natural_duration = subtitle.duration
                wav = self.AudioTimingUtils.create_silence(
                    duration_seconds=natural_duration,
                    sample_rate=24000,
                    channels=1,
                    device=torch.device('cpu')
                )
                print(f"🤫 VibeVoice SRT Subtitle {i+1} (Seq {subtitle.sequence}): Empty text, generating {natural_duration:.2f}s silence.")
                audio_segments[i] = wav
                natural_durations[i] = natural_duration
                continue
            
            print(f"🎭 VibeVoice SRT Subtitle {i+1}/{len(subtitles)} (Seq {subtitle.sequence}): Processing '{subtitle.text[:50]}...'")
            
            # Parse character segments
            character_segments = parse_character_text(subtitle.text, None)
            
            # Determine processing mode
            multi_speaker_mode = self.config.get('multi_speaker_mode', 'Custom Character Switching')
            
            if multi_speaker_mode == "Native Multi-Speaker":
                # Check if we can use native mode (max 4 characters) and no pause tags
                unique_chars = list(set([char for char, _ in character_segments]))
                full_text = " ".join([text for _, text in character_segments])
                has_pause_tags = PauseTagProcessor.has_pause_tags(full_text)
                
                if len(unique_chars) <= 4 and not has_pause_tags:
                    print(f"🎙️ Using VibeVoice native multi-speaker mode for {len(unique_chars)} speakers")
                    # Generate single multi-speaker segment
                    audio = self.adapter._generate_native_multispeaker(
                        character_segments, complete_voice_refs, self.config
                    )
                    wav = audio['waveform']
                    if wav.dim() == 3:
                        wav = wav.squeeze(0)
                else:
                    print(f"🔄 VibeVoice: Falling back to custom mode (too many speakers or pause tags)")
                    # Fall back to custom character switching
                    wav = self._process_custom_character_switching_subtitle(
                        character_segments, complete_voice_refs, seed
                    )
            else:
                # Custom character switching mode
                wav = self._process_custom_character_switching_subtitle(
                    character_segments, complete_voice_refs, seed
                )
            
            # Calculate duration and store
            natural_duration = self.AudioTimingUtils.get_audio_duration(wav, 24000)
            audio_segments[i] = wav
            natural_durations[i] = natural_duration
        
        return audio_segments, natural_durations, any_segment_cached
    
    def _process_custom_character_switching_subtitle(self,
                                                   segments: List[Tuple[str, str]],
                                                   voice_mapping: Dict[str, Any],
                                                   seed: int) -> torch.Tensor:
        """
        Process subtitle using custom character switching mode.
        Generate separate audio for each character and combine.
        
        Args:
            segments: Character segments
            voice_mapping: Voice mapping
            seed: Random seed
            
        Returns:
            Combined audio tensor for this subtitle
        """
        params = self.config.copy()
        params['seed'] = seed
        params['multi_speaker_mode'] = self.config.get('multi_speaker_mode', 'Custom Character Switching')
        
        audio_parts = []
        
        for character, text in segments:
            voice_ref = voice_mapping.get(character)
            
            # Generate audio for this character using pause tag support
            audio_tensor = self.adapter.generate_vibevoice_with_pause_tags(
                text, voice_ref, params, True, character
            )
            
            # Ensure proper tensor format
            if audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(0)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            audio_parts.append(audio_tensor)
        
        # Combine all character parts
        if len(audio_parts) == 1:
            return audio_parts[0]
        else:
            return torch.cat(audio_parts, dim=-1)
    
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
        sample_rate = 24000
        
        if timing_mode == "stretch_to_fit":
            # Use time stretching to match exact timing
            assembler = self.TimedAudioAssembler(sample_rate)
            target_timings = [(sub.start_time, sub.end_time) for sub in subtitles]
            fade_duration = timing_params.get('fade_for_StretchToFit', 0.01)
            final_audio = assembler.assemble_timed_audio(
                audio_segments, target_timings, fade_duration=fade_duration
            )
            return final_audio, None  # No updated adjustments
        
        elif timing_mode == "pad_with_silence":
            # Add silence to match timing without stretching
            from utils.timing.assembly import AudioAssemblyEngine
            assembler = AudioAssemblyEngine(sample_rate)
            final_audio = assembler.assemble_with_overlaps(audio_segments, subtitles, torch.device('cpu'))
            return final_audio, None  # No updated adjustments
        
        elif timing_mode == "concatenate":
            # Concatenate audio naturally and recalculate SRT timings
            from utils.timing.engine import TimingEngine
            from utils.timing.assembly import AudioAssemblyEngine
            
            timing_engine = TimingEngine(sample_rate)
            assembler = AudioAssemblyEngine(sample_rate)
            
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
            
            timing_engine = TimingEngine(sample_rate)
            assembler = AudioAssemblyEngine(sample_rate)
            
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
        if self.adapter:
            self.adapter.cleanup()