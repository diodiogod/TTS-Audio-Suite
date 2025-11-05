"""
ChatterBox Official 23-Lang SRT Processor - Handles SRT processing for ChatterBox Official 23-Lang
Internal processor used by UnifiedTTSSRTNode - not a ComfyUI node itself

Uses direct adapter pattern (like VibeVoice/HiggsAudio) for clean architecture:
- Load model once at initialization
- Call adapter.generate_segment_audio() for each SRT segment
- Support character switching via [CharacterName] tags with voice discovery
- Support language switching via [language:] tags passed as language_id parameter
- Support parameter switching via [seed:X], [exaggeration:Y] tags per segment
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
from utils.voice.discovery import get_available_characters, get_character_mapping
from engines.adapters.chatterbox_adapter import ChatterBoxEngineAdapter


class ChatterboxOfficial23LangSRTProcessor:
    """
    ChatterBox Official 23-Lang SRT Processor - Internal SRT processing engine for ChatterBox Official 23-Lang

    Uses direct adapter pattern:
    - Single model load at initialization via adapter
    - Per-segment generation with language/character/parameter switching
    - Voice discovery for character switching
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

        # Initialize adapter once - this loads the model
        self.adapter = ChatterBoxEngineAdapter(tts_node)
        print(f"⚙️ ChatterBox Official 23-Lang SRT: Adapter initialized, model will load on first use")

    def process_srt_content(self, srt_content: str, voice_mapping: Dict[str, Any],
                           seed: int, timing_mode: str, timing_params: Dict[str, Any],
                           tts_params: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, str, str, str]:
        """
        Process SRT content with ChatterBox Official 23-Lang TTS engine.

        Uses direct adapter pattern:
        - Parse SRT segments
        - For each segment: extract character/language/parameter tags
        - Call adapter.generate_segment_audio() with proper parameters
        - Handle timing and assembly using existing utilities

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
            from utils.text.character_parser import character_parser as cp
            from utils.timing.engine import TimingEngine
            from utils.timing.assembly import AudioAssemblyEngine
            from utils.timing.reporting import SRTReportGenerator

            print(f"📺 ChatterBox Official 23-Lang SRT: Processing SRT with character/language/parameter switching")

            # Parse SRT content
            srt_parser = SRTParser()
            srt_segments = srt_parser.parse_srt_content(srt_content, allow_overlaps=True)
            print(f"📺 ChatterBox Official 23-Lang SRT: Found {len(srt_segments)} SRT segments")

            # Check for overlaps and handle smart_natural mode fallback
            from utils.timing.overlap_detection import SRTOverlapHandler
            has_overlaps = SRTOverlapHandler.detect_overlaps(srt_segments)
            current_timing_mode, mode_switched = SRTOverlapHandler.handle_smart_natural_fallback(
                timing_mode, has_overlaps, "ChatterBox Official 23-Lang SRT"
            )

            # Set up character parser with available characters
            available_chars = get_available_characters()
            cp.set_available_characters(list(available_chars))
            cp.reset_session_cache()
            cp.set_engine_aware_default_language(self.config.get("language", "English"), "chatterbox")

            # Build voice references for characters
            voice_refs = {'narrator': None}
            narrator_voice_input = voice_mapping.get("narrator", "")
            if narrator_voice_input:
                if isinstance(narrator_voice_input, str):
                    voice_refs['narrator'] = self.tts_node.handle_reference_audio(None, narrator_voice_input)
                else:
                    voice_refs['narrator'] = self.tts_node.handle_reference_audio(narrator_voice_input, "")

                if voice_refs['narrator']:
                    print(f"📖 SRT: Using narrator voice reference: {voice_refs['narrator']}")

            try:
                char_mapping = get_character_mapping(list(available_chars), "chatterbox")
                for char in available_chars:
                    char_audio_path, _ = char_mapping.get(char, (voice_refs['narrator'], None))
                    voice_refs[char] = char_audio_path
            except Exception:
                pass

            # Generate audio for each SRT segment using adapter directly
            audio_segments = []
            timing_segments = []

            for i, subtitle in enumerate(srt_segments):
                # Check for interruption
                if model_management.interrupt_processing:
                    raise InterruptedError(f"ChatterBox 23-Lang SRT segment {i+1}/{len(srt_segments)} interrupted by user")

                if not subtitle.text.strip():
                    continue

                segment_start = subtitle.start_time
                segment_end = subtitle.end_time
                expected_duration = segment_end - segment_start

                print(f"📺 Generating SRT segment {i+1}/{len(srt_segments)} (Seq {subtitle.sequence})...")

                # Parse character/language/parameter tags in this segment
                segment_objects = cp.parse_text_segments(subtitle.text)
                character_segments_with_lang = [
                    (seg.original_character or seg.character, seg.text, seg.language,
                     seg.parameters if seg.parameters else {})
                    for seg in segment_objects
                ]

                # Check what types of switching are needed
                characters = list(set(char for char, _, _, _ in character_segments_with_lang))
                languages = list(set(lang for _, _, lang, _ in character_segments_with_lang))
                has_parameter_changes = len(set(str(params) for _, _, _, params in character_segments_with_lang)) > 1

                has_char_switching = len(characters) > 1 or (len(characters) == 1 and characters[0] != "narrator")
                has_lang_switching = len(languages) > 1

                if has_char_switching or has_lang_switching or has_parameter_changes:
                    # Complex segment - process each part individually
                    print(f"🔀 Segment {i+1}: Multiple parts (characters: {characters}, languages: {languages})")
                    segment_audio_parts = []

                    for seg_idx, (char, text, seg_lang, seg_params) in enumerate(character_segments_with_lang):
                        if not text.strip():
                            continue

                        # Get character voice reference
                        char_voice = voice_refs.get(char, voice_refs.get("narrator", None))

                        # Apply segment-specific parameters
                        seg_exag = current_exaggeration
                        seg_temp = current_temperature
                        seg_cfg = current_cfg_weight
                        seg_seed_val = seed

                        if seg_params:
                            segment_config = apply_segment_parameters(
                                {
                                    'exaggeration': current_exaggeration,
                                    'temperature': current_temperature,
                                    'cfg_weight': current_cfg_weight,
                                    'seed': seed
                                },
                                seg_params,
                                "chatterbox_official_23lang"
                            )
                            seg_exag = segment_config.get('exaggeration', current_exaggeration)
                            seg_temp = segment_config.get('temperature', current_temperature)
                            seg_cfg = segment_config.get('cfg_weight', current_cfg_weight)
                            seg_seed_val = segment_config.get('seed', seed)
                            print(f"  📊 Part {seg_idx+1}: {char} [{seg_lang}] params={seg_params}")

                        # Generate audio for this part via adapter
                        part_audio = self.adapter.generate_segment_audio(
                            text=text,
                            char_audio=char_voice if isinstance(char_voice, str) else None,
                            character=char,
                            current_language=seg_lang,
                            exaggeration=seg_exag,
                            temperature=seg_temp,
                            cfg_weight=seg_cfg,
                            seed=seg_seed_val,
                            device=current_device,
                            enable_audio_cache=True
                        )

                        # Ensure proper tensor format
                        if isinstance(part_audio, dict) and "waveform" in part_audio:
                            part_audio = part_audio["waveform"]

                        if part_audio.dim() == 3:
                            part_audio = part_audio.squeeze(0).squeeze(0)
                        elif part_audio.dim() == 2:
                            part_audio = part_audio.squeeze(0)

                        segment_audio_parts.append(part_audio)

                    # Concatenate all parts
                    if segment_audio_parts:
                        segment_audio = torch.cat(segment_audio_parts, dim=-1)
                    else:
                        segment_audio = torch.zeros(int(expected_duration * self.sample_rate))

                else:
                    # Simple segment - single call
                    single_char, single_text, single_lang, single_params = character_segments_with_lang[0]

                    seg_exag = current_exaggeration
                    seg_temp = current_temperature
                    seg_cfg = current_cfg_weight
                    seg_seed_val = seed

                    if single_params:
                        segment_config = apply_segment_parameters(
                            {
                                'exaggeration': current_exaggeration,
                                'temperature': current_temperature,
                                'cfg_weight': current_cfg_weight,
                                'seed': seed
                            },
                            single_params,
                            "chatterbox_official_23lang"
                        )
                        seg_exag = segment_config.get('exaggeration', current_exaggeration)
                        seg_temp = segment_config.get('temperature', current_temperature)
                        seg_cfg = segment_config.get('cfg_weight', current_cfg_weight)
                        seg_seed_val = segment_config.get('seed', seed)
                        print(f"  📊 Parameters: {single_params}")

                    char_voice = voice_refs.get(single_char, voice_refs.get("narrator", None))

                    # Generate via adapter
                    segment_audio = self.adapter.generate_segment_audio(
                        text=single_text,
                        char_audio=char_voice if isinstance(char_voice, str) else None,
                        character=single_char,
                        current_language=single_lang,
                        exaggeration=seg_exag,
                        temperature=seg_temp,
                        cfg_weight=seg_cfg,
                        seed=seg_seed_val,
                        device=current_device,
                        enable_audio_cache=True
                    )

                    # Ensure proper tensor format
                    if isinstance(segment_audio, dict) and "waveform" in segment_audio:
                        segment_audio = segment_audio["waveform"]

                    if segment_audio.dim() == 3:
                        segment_audio = segment_audio.squeeze(0).squeeze(0)
                    elif segment_audio.dim() == 2:
                        segment_audio = segment_audio.squeeze(0)

                # Calculate actual duration and add to segments
                actual_duration = len(segment_audio) / self.sample_rate
                audio_segments.append(segment_audio)
                timing_segments.append({
                    'expected': expected_duration,
                    'actual': actual_duration,
                    'start': segment_start,
                    'end': segment_end,
                    'sequence': subtitle.sequence
                })

                print(f"  ✓ Generated {actual_duration:.2f}s (expected {expected_duration:.2f}s)")

            # Use existing timing and assembly utilities
            timing_engine = TimingEngine(sample_rate=self.sample_rate)
            assembly_engine = AudioAssemblyEngine(sample_rate=self.sample_rate)

            # Handle timing mode
            if current_timing_mode == "smart_natural":
                adjustments, processed_segments = timing_engine.calculate_smart_timing_adjustments(
                    audio_segments,
                    srt_segments,
                    timing_params.get("timing_tolerance", 2.0),
                    timing_params.get("max_stretch_ratio", 1.0),
                    timing_params.get("min_stretch_ratio", 0.5),
                    torch.device('cpu')
                )
                final_audio = assembly_engine.assemble_by_timing_mode(
                    audio_segments, srt_segments, current_timing_mode, torch.device('cpu'),
                    adjustments=adjustments, processed_segments=processed_segments
                )
            elif current_timing_mode == "concatenate":
                adjustments = timing_engine.calculate_concatenation_adjustments(audio_segments, srt_segments)
                final_audio = assembly_engine.assemble_by_timing_mode(
                    audio_segments, srt_segments, current_timing_mode, torch.device('cpu'),
                    fade_duration=timing_params.get("fade_for_StretchToFit", 0.01)
                )
            else:
                final_audio = assembly_engine.assemble_by_timing_mode(
                    audio_segments, srt_segments, current_timing_mode, torch.device('cpu'),
                    fade_duration=timing_params.get("fade_for_StretchToFit", 0.01)
                )

                if current_timing_mode == "pad_with_silence":
                    _, adjustments = timing_engine.calculate_overlap_timing(audio_segments, srt_segments)
                else:
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
                   f"using {mode_info} mode")

            # Format final audio for ComfyUI (ensure proper 3D format: [batch, channels, samples])
            if final_audio.dim() == 1:
                final_audio = final_audio.unsqueeze(0).unsqueeze(0)
            elif final_audio.dim() == 2:
                final_audio = final_audio.unsqueeze(0)

            # Create proper ComfyUI audio format
            audio_output = {"waveform": final_audio, "sample_rate": self.sample_rate}

            return audio_output, info, timing_report, adjusted_srt

        except Exception as e:
            print(f"❌ ChatterBox Official 23-Lang SRT processing failed: {e}")
            import traceback
            traceback.print_exc()
            raise