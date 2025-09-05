"""
ChatterBox Official 23-Lang SRT Processor - Handles SRT processing for ChatterBox Official 23-Lang
Internal processor used by UnifiedTTSSRTNode - not a ComfyUI node itself

Based on the clean modular approach used by Higgs Audio SRT processor.
Uses existing timing utilities for proper SRT functionality.
"""

import torch
import os
from typing import Dict, Any, Optional, List, Tuple

# Add project root to path for imports
import sys
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class ChatterboxOfficial23LangSRTProcessor:
    """
    ChatterBox Official 23-Lang SRT Processor - Internal SRT processing engine for ChatterBox Official 23-Lang
    Handles SRT parsing, character switching, timing modes, and audio assembly
    """
    
    def __init__(self, tts_node, engine_config: Dict[str, Any]):
        """
        Initialize the SRT processor
        
        Args:
            tts_node: ChatterboxOfficial23LangTTSNode instance
            engine_config: Engine configuration dictionary
        """
        self.tts_node = tts_node
        self.config = engine_config.copy()
        self.sample_rate = 24000  # ChatterBox Official 23-Lang uses 24000 Hz (S3GEN_SR)
    
    def process_srt_content(self, srt_content: str, voice_mapping: Dict[str, Any],
                           seed: int, timing_mode: str, timing_params: Dict[str, Any]) -> Tuple[torch.Tensor, str, str, str]:
        """
        Process SRT content with ChatterBox Official 23-Lang TTS engine
        
        Args:
            srt_content: SRT subtitle content
            voice_mapping: Voice mapping for characters  
            seed: Random seed for generation
            timing_mode: How to align audio with SRT timings
            timing_params: Additional timing parameters (fade, stretch ratios, etc.)
            
        Returns:
            Tuple of (audio_output, generation_info, timing_report, adjusted_srt)
        """
        try:
            # Import required utilities
            from utils.timing.parser import SRTParser
            from utils.text.character_parser import parse_character_text
            from utils.voice.discovery import get_character_mapping
            from utils.text.pause_processor import PauseTagProcessor
            from utils.timing.engine import TimingEngine  
            from utils.timing.assembly import AudioAssemblyEngine
            from utils.timing.reporting import SRTReportGenerator
            
            print(f"📺 ChatterBox Official 23-Lang SRT: Processing SRT with multilingual support")
            
            # Parse SRT content
            srt_parser = SRTParser()
            srt_segments = srt_parser.parse_srt_content(srt_content, allow_overlaps=True)
            print(f"📺 ChatterBox Official 23-Lang SRT: Found {len(srt_segments)} SRT segments")
            
            # Check for overlaps and handle smart_natural mode fallback using modular utility
            from utils.timing.overlap_detection import SRTOverlapHandler
            has_overlaps = SRTOverlapHandler.detect_overlaps(srt_segments)
            current_timing_mode, mode_switched = SRTOverlapHandler.handle_smart_natural_fallback(
                timing_mode, has_overlaps, "ChatterBox Official 23-Lang SRT"
            )
            
            # Analyze all text for character discovery
            all_text = " ".join([seg.text for seg in srt_segments])
            character_segments = parse_character_text(all_text)
            all_characters = set(char for char, _ in character_segments)
            all_characters.add("narrator")  # Always include narrator for fallback mapping
            
            if len(all_characters) > 1 or (len(all_characters) == 1 and "narrator" not in all_characters):
                print(f"🎭 ChatterBox Official 23-Lang SRT: Detected character switching - {', '.join(sorted(all_characters))}")
            
            # Note: Extract generation parameters fresh from current config each time
            # This ensures that if config was updated via update_config(), we use the new values
            
            # Generate audio for each SRT segment
            audio_segments = []
            timing_segments = []
            total_duration = 0.0
            
            # Build voice references for character switching 
            # ChatterBox only uses audio files/paths, not reference text like Higgs Audio
            voice_refs = {}
            
            # Get narrator voice from voice_mapping and use existing handle_reference_audio method
            narrator_voice_input = voice_mapping.get("narrator", "")
            if narrator_voice_input:
                # Use the TTS node's handle_reference_audio method to convert ComfyUI tensors to file paths
                if isinstance(narrator_voice_input, str):
                    # Already a file path
                    voice_refs['narrator'] = self.tts_node.handle_reference_audio(None, narrator_voice_input)
                else:
                    # ComfyUI audio tensor - convert using base class method
                    voice_refs['narrator'] = self.tts_node.handle_reference_audio(narrator_voice_input, "")
                
                if voice_refs['narrator']:
                    print(f"📖 SRT: Using narrator voice reference: {voice_refs['narrator']}")
                else:
                    voice_refs['narrator'] = None
                    print(f"⚠️ SRT: Failed to process narrator voice reference")
            
            # Build character-specific voices using ChatterBox character mapping
            character_mapping = get_character_mapping(list(all_characters), engine_type="chatterbox")
            
            for character in all_characters:
                if character.lower() == "narrator":
                    continue
                
                audio_path, char_ref_text = character_mapping.get(character, (None, None))
                if audio_path and os.path.exists(audio_path):
                    # ChatterBox only needs the audio file path, not reference text
                    voice_refs[character] = audio_path
                    print(f"🎭 {character}: Loaded voice file: {audio_path}")
                else:
                    # Use narrator voice as fallback
                    voice_refs[character] = voice_refs.get('narrator')
                    if voice_refs.get('narrator'):
                        print(f"🎭 {character}: Using narrator voice fallback")
                    else:
                        print(f"⚠️ {character}: No voice available, using basic TTS")
            
            # Process each SRT segment
            for i, segment in enumerate(srt_segments):
                segment_text = segment.text
                segment_start = segment.start_time
                segment_end = segment.end_time
                expected_duration = segment_end - segment_start
                
                # Use character switching with pause tag support
                def srt_tts_generate_func(text_content: str) -> torch.Tensor:
                    """TTS generation function for SRT segment with character switching"""
                    if '[' in text_content and ']' in text_content:
                        # Handle character switching within this SRT segment
                        # Use character parser with language information
                        from utils.text.character_parser import character_parser
                        
                        # Set up character parser for this engine
                        character_parser.set_available_characters(list(all_characters))
                        character_parser.set_engine_aware_default_language(generation_params["language"], "chatterbox")
                        
                        # Parse character segments with language support
                        char_segments = character_parser.split_by_character_with_language(text_content)
                        
                        # Generate audio for each character segment
                        segment_audios = []
                        for char_name, char_text, char_language in char_segments:
                            if not char_text.strip():
                                continue
                                
                            # Get voice reference for this character
                            char_voice_ref = voice_refs.get(char_name, voice_refs.get('narrator'))
                            
                            # Get voice reference file path for ChatterBox
                            char_voice_path = char_voice_ref if isinstance(char_voice_ref, str) else ""
                            
                            
                            # Generate audio using the TTS node with current config parameters
                            char_audio, _ = self.tts_node.generate_speech(
                                text=char_text,
                                language=char_language,  # Use language from character parser
                                device=self.config.get("device", "auto"),
                                exaggeration=self.config.get("exaggeration", 0.5),
                                temperature=self.config.get("temperature", 0.8),
                                cfg_weight=self.config.get("cfg_weight", 0.5),
                                repetition_penalty=self.config.get("repetition_penalty", 1.2),
                                min_p=self.config.get("min_p", 0.05),
                                top_p=self.config.get("top_p", 1.0),
                                seed=seed,
                                reference_audio=None,
                                audio_prompt_path=char_voice_path,
                                enable_audio_cache=True
                            )
                            
                            
                            # Extract waveform from ComfyUI format
                            if isinstance(char_audio, dict) and "waveform" in char_audio:
                                char_waveform = char_audio["waveform"]
                            else:
                                char_waveform = char_audio
                            
                            # Ensure proper tensor format
                            if char_waveform.dim() == 3:
                                char_waveform = char_waveform.squeeze(0).squeeze(0)
                            elif char_waveform.dim() == 2:
                                char_waveform = char_waveform.squeeze(0)
                            
                            segment_audios.append(char_waveform)
                        
                        # Concatenate all character segments
                        if segment_audios:
                            return torch.cat(segment_audios, dim=0)
                        else:
                            # Empty segment - generate silence
                            return torch.zeros(int(expected_duration * self.sample_rate))
                    else:
                        # No character switching - use default narrator
                        narrator_voice_ref = voice_refs.get('narrator')
                        narrator_voice_path = narrator_voice_ref if isinstance(narrator_voice_ref, str) else ""
                        
                        
                        # Generate audio using the TTS node with current config parameters
                        narrator_audio, _ = self.tts_node.generate_speech(
                            text=text_content,
                            language=self.config.get("language", "English"),
                            device=self.config.get("device", "auto"),
                            exaggeration=self.config.get("exaggeration", 0.5),
                            temperature=self.config.get("temperature", 0.8),
                            cfg_weight=self.config.get("cfg_weight", 0.5),
                            repetition_penalty=self.config.get("repetition_penalty", 1.2),
                            min_p=self.config.get("min_p", 0.05),
                            top_p=self.config.get("top_p", 1.0),
                            seed=seed,
                            reference_audio=None,
                            audio_prompt_path=narrator_voice_path,
                            enable_audio_cache=True
                        )
                        
                        
                        # Extract waveform from ComfyUI format
                        if isinstance(narrator_audio, dict) and "waveform" in narrator_audio:
                            narrator_waveform = narrator_audio["waveform"]
                        else:
                            narrator_waveform = narrator_audio
                        
                        # Ensure proper tensor format
                        if narrator_waveform.dim() == 3:
                            narrator_waveform = narrator_waveform.squeeze(0).squeeze(0)
                        elif narrator_waveform.dim() == 2:
                            narrator_waveform = narrator_waveform.squeeze(0)
                        
                        return narrator_waveform
                
                # Preprocess segment text for pause tags first
                processed_text, pause_segments = PauseTagProcessor.preprocess_text_with_pause_tags(
                    segment_text, enable_pause_tags=True
                )
                
                # Generate audio for this SRT segment with pause tag support
                if pause_segments is None:
                    # No pause tags - generate audio directly
                    segment_audio = srt_tts_generate_func(segment_text)
                else:
                    # Has pause tags - use pause processor
                    segment_audio = PauseTagProcessor.generate_audio_with_pauses(
                        pause_segments, srt_tts_generate_func, self.sample_rate
                    )
                
                # Calculate actual duration
                actual_duration = len(segment_audio) / self.sample_rate
                audio_segments.append(segment_audio)
                timing_segments.append({
                    'expected': expected_duration,
                    'actual': actual_duration,
                    'start': segment_start,
                    'end': segment_end,
                    'sequence': segment.sequence
                })
                
                print(f"📺 ChatterBox Official 23-Lang SRT Segment {i+1}/{len(srt_segments)} (Seq {segment.sequence}): "
                      f"Generated {actual_duration:.2f}s audio (expected {expected_duration:.2f}s)")
                
                total_duration += actual_duration
            
            # Use existing timing and assembly utilities
            timing_engine = TimingEngine(sample_rate=self.sample_rate)
            assembly_engine = AudioAssemblyEngine(sample_rate=self.sample_rate)
            
            # Handle timing mode routing properly
            if current_timing_mode == "smart_natural":
                # Calculate smart timing adjustments
                adjustments, processed_segments = timing_engine.calculate_smart_timing_adjustments(
                    audio_segments, 
                    srt_segments,
                    timing_params.get("timing_tolerance", 2.0),
                    timing_params.get("max_stretch_ratio", 1.0),
                    timing_params.get("min_stretch_ratio", 0.5),
                    torch.device('cpu')
                )
                
                # Use unified assembly method with proper routing
                final_audio = assembly_engine.assemble_by_timing_mode(
                    audio_segments, srt_segments, current_timing_mode, torch.device('cpu'),
                    adjustments=adjustments, processed_segments=processed_segments
                )
            elif current_timing_mode == "concatenate":
                # Use existing modular timing engine for concatenate mode
                adjustments = timing_engine.calculate_concatenation_adjustments(audio_segments, srt_segments)
                final_audio = assembly_engine.assemble_by_timing_mode(
                    audio_segments, srt_segments, current_timing_mode, torch.device('cpu'),
                    fade_duration=timing_params.get("fade_for_StretchToFit", 0.01)
                )
            else:
                # For other modes (pad_with_silence, stretch_to_fit) - use unified assembly
                final_audio = assembly_engine.assemble_by_timing_mode(
                    audio_segments, srt_segments, current_timing_mode, torch.device('cpu'),
                    fade_duration=timing_params.get("fade_for_StretchToFit", 0.01)
                )
                
                # Use existing overlap timing calculation for pad_with_silence
                if current_timing_mode == "pad_with_silence":
                    _, adjustments = timing_engine.calculate_overlap_timing(audio_segments, srt_segments)
                else:
                    # For stretch_to_fit - create minimal adjustments (stretch logic handled by assembly)
                    adjustments = []
                    for i, (segment, subtitle) in enumerate(zip(audio_segments, srt_segments)):
                        natural_duration = len(segment) / self.sample_rate
                        target_duration = subtitle.end_time - subtitle.start_time
                        stretch_factor = target_duration / natural_duration if natural_duration > 0 else 1.0
                        
                        adjustments.append({
                            'segment_index': i,
                            'sequence': subtitle.sequence,
                            'start_time': subtitle.start_time,
                            'end_time': subtitle.end_time,
                            'natural_audio_duration': natural_duration,
                            'original_srt_start': subtitle.start_time,
                            'original_srt_end': subtitle.end_time,
                            'original_srt_duration': target_duration,
                            'original_text': subtitle.text,
                            'final_srt_start': subtitle.start_time,
                            'final_srt_end': subtitle.end_time,
                            'needs_stretching': True,
                            'stretch_factor_applied': stretch_factor,
                            'stretch_factor': stretch_factor,
                            'stretch_type': 'time_stretch' if abs(stretch_factor - 1.0) > 0.01 else 'none',
                            'final_segment_duration': target_duration,
                            'actions': [f"Audio stretched from {natural_duration:.2f}s to {target_duration:.2f}s (factor: {stretch_factor:.2f}x)"]
                        })
            
            # Map adjustment keys for report generator compatibility
            mapped_adjustments = []
            for adj in adjustments:
                mapped_adj = adj.copy()
                mapped_adj['start_time'] = adj.get('final_srt_start', adj.get('original_srt_start', 0))
                mapped_adj['end_time'] = adj.get('final_srt_end', adj.get('original_srt_end', 0))  
                mapped_adj['natural_duration'] = adj.get('natural_audio_duration', 0)
                mapped_adjustments.append(mapped_adj)
            
            # Generate reports
            report_generator = SRTReportGenerator()
            timing_report = report_generator.generate_timing_report(
                srt_segments, mapped_adjustments, current_timing_mode, has_overlaps, mode_switched
            )
            adjusted_srt = report_generator.generate_adjusted_srt_string(
                srt_segments, mapped_adjustments, current_timing_mode
            )
            
            # Generate info
            final_duration = len(final_audio) / self.sample_rate
            mode_info = f"{current_timing_mode}"
            if mode_switched:
                mode_info = f"{current_timing_mode} (switched from {timing_mode} due to overlaps)"
            
            info = (f"Generated {final_duration:.1f}s ChatterBox Official 23-Lang SRT-timed audio from {len(srt_segments)} subtitles "
                   f"using {mode_info} mode ({self.config.get('language', 'English')})")
            
            # Format final audio for ComfyUI (ensure proper 3D format: [batch, channels, samples])
            if final_audio.dim() == 1:
                final_audio = final_audio.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            elif final_audio.dim() == 2:
                final_audio = final_audio.unsqueeze(0)  # Add batch dimension
            
            # Create proper ComfyUI audio format
            audio_output = {"waveform": final_audio, "sample_rate": self.sample_rate}
            
            return audio_output, info, timing_report, adjusted_srt
            
        except Exception as e:
            print(f"❌ ChatterBox Official 23-Lang SRT processing failed: {e}")
            import traceback
            traceback.print_exc()
            raise