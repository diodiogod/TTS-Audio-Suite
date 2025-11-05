"""
ChatterBox Official 23-Lang SRT Processor - Handles SRT processing for ChatterBox Official 23-Lang
Internal processor used by UnifiedTTSSRTNode - not a ComfyUI node itself

Based on the clean modular approach used by Higgs Audio SRT processor.
Uses existing timing utilities for proper SRT functionality.
"""

import torch
import os
from typing import Dict, Any, Optional, List, Tuple
import comfy.model_management as model_management

# Add project root to path for imports
import sys
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.text.segment_parameters import apply_segment_parameters


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
                           seed: int, timing_mode: str, timing_params: Dict[str, Any],
                           tts_params: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, str, str, str]:
        """
        Process SRT content with ChatterBox Official 23-Lang TTS engine
        
        Args:
            srt_content: SRT subtitle content
            voice_mapping: Voice mapping for characters  
            seed: Random seed for generation
            timing_mode: How to align audio with SRT timings
            timing_params: Additional timing parameters (fade, stretch ratios, etc.)
            tts_params: Current TTS parameters from UI (exaggeration, temperature, etc.)
            
        Returns:
            Tuple of (audio_output, generation_info, timing_report, adjusted_srt)
        """
        # Use actual runtime TTS parameters instead of config defaults
        if tts_params is None:
            tts_params = {}
            
        # Get current parameters with proper fallbacks
        current_exaggeration = tts_params.get('exaggeration', self.config.get("exaggeration", 0.5))
        current_temperature = tts_params.get('temperature', self.config.get("temperature", 0.8))
        current_cfg_weight = tts_params.get('cfg_weight', self.config.get("cfg_weight", 0.5))
        current_repetition_penalty = tts_params.get('repetition_penalty', self.config.get("repetition_penalty", 1.2))
        current_min_p = tts_params.get('min_p', self.config.get("min_p", 0.05))
        current_top_p = tts_params.get('top_p', self.config.get("top_p", 1.0))
        current_language = tts_params.get('language', self.config.get("language", "English"))
        current_device = tts_params.get('device', self.config.get("device", "auto"))
        current_model_version = tts_params.get('model_version', self.config.get("model_version", "v2"))
        
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
            
            # Discover available characters (like regular ChatterBox - line 680)
            from utils.voice.discovery import get_available_characters, get_character_mapping
            from utils.text.character_parser import character_parser as cp
            available_chars = get_available_characters()
            cp.set_available_characters(list(available_chars))

            # CRITICAL FIX: Reset character parser session to prevent contamination (line 684)
            cp.reset_session_cache()
            cp.set_engine_aware_default_language(self.config.get("language", "English"), "chatterbox")

            # Note: Extract generation parameters fresh from current config each time
            # This ensures that if config was updated via update_config(), we use the new values

            # Generate audio for each SRT segment
            audio_segments = []
            timing_segments = []
            total_duration = 0.0
            all_subtitle_segments = []
            subtitle_language_groups = {}

            # Build voice references for characters (same as regular ChatterBox - line 1076-1085)
            voice_refs = {'narrator': None}
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

            try:
                available_chars = get_available_characters()
                char_mapping = get_character_mapping(list(available_chars), "chatterbox")
                for char in available_chars:
                    char_audio_path, _ = char_mapping.get(char, (voice_refs['narrator'], None))
                    voice_refs[char] = char_audio_path
            except Exception:
                pass

            # FIRST PASS: Analyze all subtitles and categorize them (lines 691-740 from regular ChatterBox)
            for i, subtitle in enumerate(srt_segments):
                if not subtitle.text.strip():
                    # Empty subtitle - will be handled separately
                    all_subtitle_segments.append((i, subtitle, 'empty', None, None))
                    continue

                # Parse character segments with parameters (handles both language and character tags)
                segment_objects = cp.parse_text_segments(subtitle.text)

                # Convert to 4-tuple format with parameters
                # Use original_character (before alias resolution) to preserve the actual character from the tag
                character_segments_with_lang = [(seg.original_character or seg.character, seg.text, seg.language, seg.parameters if seg.parameters else {}) for seg in segment_objects]

                # Check if we have character switching, language switching, or parameter changes
                characters = list(set(char for char, _, _, _ in character_segments_with_lang))
                languages = list(set(lang for _, _, lang, _ in character_segments_with_lang))
                has_parameter_changes = len(set(str(params) for _, _, _, params in character_segments_with_lang)) > 1

                has_multiple_characters_in_subtitle = len(characters) > 1 or (len(characters) == 1 and characters[0] != "narrator")
                has_multiple_languages_in_subtitle = len(languages) > 1

                if has_multiple_characters_in_subtitle or has_multiple_languages_in_subtitle or has_parameter_changes:
                    # Complex subtitle - group by dominant language or mark as multilingual
                    primary_lang = languages[0] if languages else 'en'
                    if has_parameter_changes:
                        # Parameter changes require segment-by-segment processing
                        subtitle_type = 'parameter_switching'
                    elif has_multiple_languages_in_subtitle:
                        subtitle_type = 'multilingual'
                    else:
                        subtitle_type = 'multicharacter'
                    all_subtitle_segments.append((i, subtitle, subtitle_type, primary_lang, character_segments_with_lang))

                    # Add to language groups for smart processing
                    if primary_lang not in subtitle_language_groups:
                        subtitle_language_groups[primary_lang] = []
                    subtitle_language_groups[primary_lang].append((i, subtitle, subtitle_type, character_segments_with_lang))
                else:
                    # Simple subtitle - group by language
                    single_char, single_text, single_lang, single_params = character_segments_with_lang[0]
                    all_subtitle_segments.append((i, subtitle, 'simple', single_lang, character_segments_with_lang))

                    if single_lang not in subtitle_language_groups:
                        subtitle_language_groups[single_lang] = []
                    subtitle_language_groups[single_lang].append((i, subtitle, 'simple', character_segments_with_lang))

            # SECOND PASS: Process each language group with proper routing
            for lang_code in sorted(subtitle_language_groups.keys()):
                lang_subtitles = subtitle_language_groups[lang_code]

                print(f"📋 Processing {len(lang_subtitles)} SRT subtitle(s) in '{lang_code}' language group...")

                # Process each subtitle in this language group
                for i, subtitle, subtitle_type, character_segments_with_lang in lang_subtitles:
                    # Check for interruption before processing each segment
                    if model_management.interrupt_processing:
                        raise InterruptedError(f"ChatterBox 23-Lang SRT segment {i+1}/{len(srt_segments)} interrupted by user")
                    segment_text = subtitle.text
                    segment_start = subtitle.start_time
                    segment_end = subtitle.end_time
                    expected_duration = segment_end - segment_start

                    if subtitle_type == 'empty':
                        # Skip empty subtitles
                        continue

                    print(f"📺 Generating SRT segment {i+1}/{len(srt_segments)} (Seq {subtitle.sequence}) in {lang_code}...")

                    if subtitle_type == 'parameter_switching':
                        # Process each segment individually with its own parameters (lines 1110-1156)
                        print(f"🔀 ChatterBox Official 23-Lang SRT Segment {i+1} (Seq {subtitle.sequence}): Per-segment parameter switching")

                        segment_audio_parts = []
                        for seg_idx, (char, text, seg_lang, seg_params) in enumerate(character_segments_with_lang):
                            # Get character-specific voice reference
                            char_voice = voice_refs.get(char, voice_refs.get("narrator", None))

                            # Apply segment parameters
                            seg_exag = current_exaggeration
                            seg_temp = current_temperature
                            seg_cfg = current_cfg_weight
                            seg_seed_val = seed

                            if seg_params:
                                segment_config = apply_segment_parameters(
                                    {'exaggeration': current_exaggeration, 'temperature': current_temperature, 'cfg_weight': current_cfg_weight, 'seed': seed},
                                    seg_params,
                                    "chatterbox_official_23lang"
                                )
                                seg_exag = segment_config.get('exaggeration', current_exaggeration)
                                seg_temp = segment_config.get('temperature', current_temperature)
                                seg_cfg = segment_config.get('cfg_weight', current_cfg_weight)
                                seg_seed_val = segment_config.get('seed', seed)
                                print(f"  📊 Segment {seg_idx+1}: Character '{char}' with parameters {seg_params}")

                            if not text.strip():
                                continue

                            # Generate audio for this segment with its parameters and character voice
                            segment_wav, _ = self.tts_node.generate_speech(
                                text=text,
                                language=seg_lang,
                                device=current_device,
                                model_version=current_model_version,
                                exaggeration=seg_exag,
                                temperature=seg_temp,
                                cfg_weight=seg_cfg,
                                repetition_penalty=current_repetition_penalty,
                                min_p=current_min_p,
                                top_p=current_top_p,
                                seed=seg_seed_val,
                                reference_audio=None,
                                audio_prompt_path=char_voice if isinstance(char_voice, str) else "",
                                enable_audio_cache=True,
                                character=char
                            )

                            # Extract waveform from ComfyUI format
                            if isinstance(segment_wav, dict) and "waveform" in segment_wav:
                                segment_wav = segment_wav["waveform"]

                            # Ensure proper tensor format
                            if segment_wav.dim() == 3:
                                segment_wav = segment_wav.squeeze(0).squeeze(0)
                            elif segment_wav.dim() == 2:
                                segment_wav = segment_wav.squeeze(0)

                            segment_audio_parts.append(segment_wav)

                        # Concatenate all segment audio
                        if segment_audio_parts:
                            segment_audio = torch.cat(segment_audio_parts, dim=-1)
                        else:
                            segment_audio = torch.zeros(int(expected_duration * self.sample_rate))

                    elif subtitle_type == 'multilingual' or subtitle_type == 'multicharacter':
                        # Use modular multilingual engine for character/language switching (lines 1158-1227)
                        characters = list(set(char for char, _, _, _ in character_segments_with_lang))
                        languages = list(set(lang for _, _, lang, _ in character_segments_with_lang))

                        if len(languages) > 1:
                            print(f"🌍 ChatterBox Official 23-Lang SRT Segment {i+1} (Seq {subtitle.sequence}): Language switching - {', '.join(languages)}")
                        if len(characters) > 1 or (len(characters) == 1 and characters[0] != "narrator"):
                            print(f"🎭 ChatterBox Official 23-Lang SRT Segment {i+1} (Seq {subtitle.sequence}): Character switching - {', '.join(characters)}")

                        # Lazy load modular components
                        if not hasattr(self, 'multilingual_engine') or self.multilingual_engine is None:
                            from utils.voice.multilingual_engine import MultilingualEngine
                            from engines.adapters.chatterbox_adapter import ChatterBoxEngineAdapter
                            self.multilingual_engine = MultilingualEngine("chatterbox")
                            self.chatterbox_adapter = ChatterBoxEngineAdapter(self)

                        # Extract first segment's parameters if any (will be applied to entire subtitle)
                        first_params = None
                        for _, _, _, seg_params in character_segments_with_lang:
                            if seg_params:
                                first_params = seg_params
                                break

                        seg_exag = current_exaggeration
                        seg_temp = current_temperature
                        seg_cfg = current_cfg_weight
                        seg_seed_val = seed

                        if first_params:
                            segment_config = apply_segment_parameters(
                                {'exaggeration': current_exaggeration, 'temperature': current_temperature, 'cfg_weight': current_cfg_weight, 'seed': seed},
                                first_params,
                                "chatterbox_official_23lang"
                            )
                            seg_exag = segment_config.get('exaggeration', current_exaggeration)
                            seg_temp = segment_config.get('temperature', current_temperature)
                            seg_cfg = segment_config.get('cfg_weight', current_cfg_weight)
                            seg_seed_val = segment_config.get('seed', seed)
                            print(f"  📊 SRT: Applying segment parameters: {first_params}")

                        # Use modular multilingual engine
                        result = self.multilingual_engine.process_multilingual_text(
                            text=subtitle.text,
                            engine_adapter=self.chatterbox_adapter,
                            model=lang_code,
                            device=current_device,
                            main_audio_reference=voice_refs.get('narrator'),
                            main_text_reference="",
                            temperature=seg_temp,
                            exaggeration=seg_exag,
                            cfg_weight=seg_cfg,
                            seed=seg_seed_val,
                            enable_audio_cache=True,
                            crash_protection_template=""
                        )

                        segment_audio = result.audio

                        # Ensure proper tensor format
                        if segment_audio.dim() == 3:
                            segment_audio = segment_audio.squeeze(0).squeeze(0)
                        elif segment_audio.dim() == 2:
                            segment_audio = segment_audio.squeeze(0)

                    else:  # subtitle_type == 'simple'
                        # Single character mode - model already loaded for this language group (lines 1229-1260)
                        single_char, single_text, single_lang, single_params = character_segments_with_lang[0]

                        seg_exag = current_exaggeration
                        seg_temp = current_temperature
                        seg_cfg = current_cfg_weight
                        seg_seed_val = seed

                        if single_params:
                            segment_config = apply_segment_parameters(
                                {'exaggeration': current_exaggeration, 'temperature': current_temperature, 'cfg_weight': current_cfg_weight, 'seed': seed},
                                single_params,
                                "chatterbox_official_23lang"
                            )
                            seg_exag = segment_config.get('exaggeration', current_exaggeration)
                            seg_temp = segment_config.get('temperature', current_temperature)
                            seg_cfg = segment_config.get('cfg_weight', current_cfg_weight)
                            seg_seed_val = segment_config.get('seed', seed)
                            print(f"  📊 SRT: Applying segment parameters: {single_params}")

                        # Get narrator voice for simple case (no character switching)
                        narrator_voice = voice_refs.get('narrator')
                        narrator_voice_path = narrator_voice if isinstance(narrator_voice, str) else ""

                        # Direct generation for simple subtitles
                        segment_audio, _ = self.tts_node.generate_speech(
                            text=single_text,
                            language=single_lang,
                            device=current_device,
                            model_version=current_model_version,
                            exaggeration=seg_exag,
                            temperature=seg_temp,
                            cfg_weight=seg_cfg,
                            repetition_penalty=current_repetition_penalty,
                            min_p=current_min_p,
                            top_p=current_top_p,
                            seed=seg_seed_val,
                            reference_audio=None,
                            audio_prompt_path=narrator_voice_path,
                            enable_audio_cache=True,
                            character="narrator"
                        )

                        # Extract waveform from ComfyUI format
                        if isinstance(segment_audio, dict) and "waveform" in segment_audio:
                            segment_audio = segment_audio["waveform"]

                        # Ensure proper tensor format
                        if segment_audio.dim() == 3:
                            segment_audio = segment_audio.squeeze(0).squeeze(0)
                        elif segment_audio.dim() == 2:
                            segment_audio = segment_audio.squeeze(0)

                    # Calculate actual duration
                    actual_duration = len(segment_audio) / self.sample_rate
                    audio_segments.append(segment_audio)
                    timing_segments.append({
                        'expected': expected_duration,
                        'actual': actual_duration,
                        'start': segment_start,
                        'end': segment_end,
                        'sequence': subtitle.sequence
                    })

                    print(f"📺 ChatterBox Official 23-Lang SRT Segment {i+1}/{len(srt_segments)} (Seq {subtitle.sequence}): "
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