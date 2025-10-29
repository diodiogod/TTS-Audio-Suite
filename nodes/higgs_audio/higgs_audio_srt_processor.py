"""
Higgs Audio SRT Processor - Handles SRT processing for Higgs Audio 2 TTS engine
Internal processor used by UnifiedTTSSRTNode - not a ComfyUI node itself
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


class HiggsAudioSRTProcessor:
    """
    Higgs Audio SRT Processor - Internal SRT processing engine for Higgs Audio 2
    Handles SRT parsing, character switching, timing modes, and audio assembly
    """
    
    def __init__(self, engine_wrapper):
        """
        Initialize the SRT processor

        Args:
            engine_wrapper: HiggsAudioWrapper instance with adapter and config
        """
        self.engine_wrapper = engine_wrapper
        self.sample_rate = 24000

    def generate_srt_speech(self, srt_content: str, multi_speaker_mode: str, 
                           audio_tensor: Optional[Dict] = None, reference_text: str = "",
                           seed: int = 1, timing_mode: str = "smart_natural",
                           fade_for_StretchToFit: float = 0.01, max_stretch_ratio: float = 1.0,
                           min_stretch_ratio: float = 0.5, timing_tolerance: float = 2.0,
                           enable_audio_cache: bool = True, **params):
        """
        Process SRT content with Higgs Audio 2 TTS engine
        
        Args:
            srt_content: SRT subtitle content
            multi_speaker_mode: Multi-speaker processing mode
            audio_tensor: Reference audio tensor
            reference_text: Reference text for voice cloning
            seed: Random seed for generation
            timing_mode: How to align audio with SRT timings
            fade_for_StretchToFit: Crossfade duration for stretch_to_fit mode
            max_stretch_ratio: Maximum stretch ratio for smart_natural mode
            min_stretch_ratio: Minimum stretch ratio for smart_natural mode
            timing_tolerance: Timing tolerance for smart_natural mode
            enable_audio_cache: Enable audio caching
            **params: Additional parameters
            
        Returns:
            Tuple of (formatted_audio, generation_info, timing_report, adjusted_srt)
        """
        try:
            # Import required utilities
            from utils.timing.parser import SRTParser
            from utils.text.character_parser import parse_character_text, character_parser
            from utils.voice.discovery import get_character_mapping, get_available_characters
            from utils.text.pause_processor import PauseTagProcessor
            from utils.timing.engine import TimingEngine
            from utils.timing.assembly import AudioAssemblyEngine
            from utils.timing.reporting import SRTReportGenerator

            print(f"📺 Higgs Audio SRT: Processing SRT with multi-speaker support")

            # Set up character parser with available characters BEFORE processing
            available_chars = get_available_characters()
            character_parser.set_available_characters(list(available_chars))

            # Parse SRT content
            srt_parser = SRTParser()
            srt_segments = srt_parser.parse_srt_content(srt_content, allow_overlaps=True)
            print(f"📺 Higgs Audio SRT: Found {len(srt_segments)} SRT segments")

            # Check for overlaps and handle smart_natural mode fallback using modular utility
            from utils.timing.overlap_detection import SRTOverlapHandler
            has_overlaps = SRTOverlapHandler.detect_overlaps(srt_segments)
            current_timing_mode, mode_switched = SRTOverlapHandler.handle_smart_natural_fallback(
                timing_mode, has_overlaps, "Higgs Audio SRT"
            )

            print(f"🎭 Higgs Audio SRT: Using mode '{multi_speaker_mode}'")

            # Analyze all text for character discovery
            all_text = " ".join([seg.text for seg in srt_segments])
            character_segments = parse_character_text(all_text)
            all_characters = set(char for char, _ in character_segments)
            all_characters.add("narrator")  # Always include narrator for fallback mapping
            
            if len(all_characters) > 1 or (len(all_characters) == 1 and "narrator" not in all_characters):
                print(f"🎭 Higgs Audio SRT: Detected character switching - {', '.join(sorted(all_characters))}")
            
            # Extract generation parameters from engine config to pass to all segment calls
            generation_params = {
                "temperature": self.engine_wrapper.config.get("temperature", 0.8),
                "top_p": self.engine_wrapper.config.get("top_p", 0.6),
                "top_k": self.engine_wrapper.config.get("top_k", 80),
                "max_new_tokens": self.engine_wrapper.config.get("max_new_tokens", 2048),
                "system_prompt": self.engine_wrapper.config.get("system_prompt", "Generate audio following instruction.")
            }
            
            # Generate audio for each SRT segment
            audio_segments = []
            timing_segments = []
            total_duration = 0.0

            # Check for interruption before starting segment generation
            if model_management.interrupt_processing:
                raise InterruptedError("Higgs Audio SRT: Pre-generation setup interrupted by user")

            # Build voice references for character switching mode
            voice_refs = {}
            if multi_speaker_mode == "Custom Character Switching":
                character_mapping = get_character_mapping(list(all_characters), engine_type="f5tts")
                
                # Build voice references with proper fallback hierarchy for narrator:
                # 1. opt_narrator (already in audio_tensor/reference_text)
                # 2. narrator_voice dropdown (handled by unified node)  
                # 3. Mapped narrator (from character alias map - e.g., David Attenborough)
                # 4. None (if user removed narrator from mapping)
                
                narrator_voice_dict = None
                narrator_ref_text = ""
                
                if audio_tensor is not None:
                    # Priority 1 & 2: Connected narrator (from Character Voices or dropdown)
                    narrator_voice_dict = {"waveform": audio_tensor["waveform"], "sample_rate": audio_tensor["sample_rate"]}
                    narrator_ref_text = reference_text or ""
                    ref_display = f"'{narrator_ref_text[:50]}...'" if narrator_ref_text else "No reference text"
                    print(f"📖 SRT: Using connected narrator voice (Character Voices node or dropdown) | Ref: {ref_display}")
                else:
                    # Priority 3: Check mapped narrator (e.g., David Attenborough from alias map)
                    mapped_narrator = character_mapping.get("narrator", (None, None))
                    if mapped_narrator[0] and os.path.exists(mapped_narrator[0]):
                        import torchaudio
                        waveform, sample_rate = torchaudio.load(mapped_narrator[0])
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                        narrator_voice_dict = {"waveform": waveform, "sample_rate": sample_rate}
                        narrator_ref_text = mapped_narrator[1] or ""
                        ref_display = f"'{narrator_ref_text[:50]}...'" if narrator_ref_text else "No reference text"
                        print(f"📖 SRT: Using mapped narrator voice: {mapped_narrator[0]} | Ref: {ref_display}")
                    else:
                        # Priority 4: No narrator available
                        print(f"⚠️ SRT: No narrator voice available - characters will use basic TTS when no character voice found")
                
                voice_refs = {'narrator': narrator_voice_dict}
                ref_texts = {'narrator': narrator_ref_text}
                
                # Add character-specific voices
                for character in all_characters:
                    if character.lower() == "narrator":
                        continue
                    
                    audio_path, char_ref_text = character_mapping.get(character, (None, None))
                    if audio_path and os.path.exists(audio_path):
                        import torchaudio
                        waveform, sample_rate = torchaudio.load(audio_path)
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                        voice_refs[character] = {"waveform": waveform, "sample_rate": sample_rate}
                        
                        # Use character-specific reference text from mapping
                        if char_ref_text and char_ref_text.strip():
                            ref_texts[character] = char_ref_text.strip()
                            print(f"🎭 {character}: Loaded voice + ref text: '{char_ref_text[:30]}...'")
                        else:
                            ref_texts[character] = ""  # Empty ref text for character's own voice
                            print(f"🎭 {character}: Loaded voice, no ref text")
                    else:
                        voice_refs[character] = narrator_voice_dict
                        ref_texts[character] = reference_text or ""
                        if narrator_voice_dict:
                            print(f"🎭 {character}: Using narrator voice + ref text")
                        else:
                            print(f"⚠️ {character}: No voice available, using basic TTS")
            
            for i, segment in enumerate(srt_segments):
                # Check for interruption before processing each segment
                if model_management.interrupt_processing:
                    raise InterruptedError(f"Higgs Audio SRT segment {i+1}/{len(srt_segments)} interrupted by user")

                segment_text = segment.text
                segment_start = segment.start_time
                segment_end = segment.end_time
                expected_duration = segment_end - segment_start

                if multi_speaker_mode == "Custom Character Switching":
                    # Use character switching with pause tag support
                    def srt_tts_generate_func(text_content: str) -> torch.Tensor:
                        """TTS generation function for SRT segment with character switching"""
                        if '[' in text_content and ']' in text_content:
                            # Handle character switching within this SRT segment
                            # Use character parser with language information
                            from utils.text.character_parser import character_parser
                            char_segments_with_lang = character_parser.split_by_character_with_language(text_content)
                            segment_audio_parts = []
                            
                            # Get full segment info with explicit language flag
                            char_segments_full = character_parser.parse_text_segments(text_content)
                            
                            for segment in char_segments_full:
                                character = segment.character
                                # Check for interruption during character segment processing
                                if model_management.interrupt_processing:
                                    raise InterruptedError(f"Higgs Audio SRT character segment ({character}) interrupted by user")
                                segment_text = segment.text
                                language = segment.language
                                explicit_language = segment.explicit_language
                                segment_params = segment.parameters if segment.parameters else {}

                                char_audio_dict = voice_refs.get(character, voice_refs.get("narrator"))
                                char_ref_text = ref_texts.get(character, reference_text or "")

                                # Add language hint for Higgs Audio 2 ONLY if language was explicit in tag
                                higgs_text = segment_text
                                if explicit_language and language:
                                    display_name = character_parser.get_language_display_name(language)
                                    higgs_text = f"[{display_name}] {segment_text}"

                                # Log what's actually being processed
                                print(f"📺 Processing SRT segment {i+1}/{len(srt_segments)} ({character}): '{higgs_text[:50]}...' ({expected_duration:.2f}s)")

                                # Apply per-segment parameters
                                current_params = dict(generation_params)
                                current_seed = seed
                                if segment_params:
                                    from utils.text.segment_parameters import apply_segment_parameters
                                    current_params = apply_segment_parameters(current_params, segment_params, "higgs_audio")
                                    # If segment has a seed parameter, use that instead of the default
                                    if 'seed' in current_params:
                                        current_seed = current_params.pop('seed')
                                    print(f"📊 SRT segment {i+1}: Character '{character}' with parameters {segment_params}")

                                segment_result = self.engine_wrapper.generate_srt_audio(
                                    srt_content="",  # Individual segment processing
                                    text=higgs_text,  # Use text with language hints
                                    char_audio=char_audio_dict,
                                    char_text=char_ref_text,
                                    character=character,
                                    seed=current_seed,
                                    enable_audio_cache=enable_audio_cache,
                                    **current_params  # Pass through all generation parameters including segment-specific ones
                                )
                                
                                # Convert to tensor format
                                if isinstance(segment_result, dict) and "waveform" in segment_result:
                                    segment_result = segment_result["waveform"]
                                if segment_result.dim() == 3:
                                    segment_result = segment_result.squeeze(0)
                                elif segment_result.dim() == 1:
                                    segment_result = segment_result.unsqueeze(0)
                                
                                segment_audio_parts.append(segment_result)
                            
                            # Combine character segments
                            if segment_audio_parts:
                                return torch.cat(segment_audio_parts, dim=-1)
                            else:
                                return torch.zeros(1, 0)
                        else:
                            # Simple text segment - use narrator voice
                            narrator_audio = voice_refs.get("narrator")
                            
                            segment_result = self.engine_wrapper.generate_srt_audio(
                                srt_content="",  # Individual segment processing
                                text=text_content,
                                char_audio=narrator_audio,
                                char_text=ref_texts.get("narrator", reference_text or ""),
                                character="narrator",
                                seed=seed,
                                enable_audio_cache=enable_audio_cache,
                                **generation_params  # Pass through all generation parameters
                            )
                            
                            # Convert to tensor format
                            if isinstance(segment_result, dict) and "waveform" in segment_result:
                                segment_result = segment_result["waveform"]
                            if segment_result.dim() == 3:
                                segment_result = segment_result.squeeze(0)
                            elif segment_result.dim() == 1:
                                segment_result = segment_result.unsqueeze(0)
                            
                            return segment_result
                    
                    # Process with pause tag handling
                    pause_processor = PauseTagProcessor()
                    segments, clean_text = pause_processor.parse_pause_tags(segment_text)
                    
                    if segments:
                        segment_audio_tensor = pause_processor.generate_audio_with_pauses(
                            segments=segments,
                            tts_generate_func=srt_tts_generate_func,
                            sample_rate=self.sample_rate
                        )
                    else:
                        segment_audio_tensor = srt_tts_generate_func(segment_text)
                
                else:
                    # Native multi-speaker modes - process entire segment as single unit
                    reference_audio_dict = None
                    if audio_tensor is not None:
                        reference_audio_dict = {
                            "waveform": audio_tensor["waveform"],
                            "sample_rate": audio_tensor["sample_rate"]
                        }
                    
                    # Get second narrator for native modes
                    opt_second_narrator = self.engine_wrapper.config.get("opt_second_narrator")
                    
                    segment_audio = self.engine_wrapper.generate_srt_audio(
                        srt_content="",  # Individual segment processing
                        text=segment_text,
                        char_audio=reference_audio_dict,
                        char_text="",  # Higgs Audio doesn't need reference text
                        character="SPEAKER0",
                        seed=seed,
                        enable_audio_cache=enable_audio_cache,
                        multi_speaker_mode=multi_speaker_mode,
                        second_narrator_audio=opt_second_narrator,
                        second_narrator_text="",
                        **generation_params  # Pass through all generation parameters
                    )
                    
                    # Convert to tensor format
                    if isinstance(segment_audio, dict) and "waveform" in segment_audio:
                        segment_audio_tensor = segment_audio["waveform"]
                    else:
                        segment_audio_tensor = segment_audio
                    
                    if segment_audio_tensor.dim() == 3:
                        segment_audio_tensor = segment_audio_tensor.squeeze(0)
                    elif segment_audio_tensor.dim() == 1:
                        segment_audio_tensor = segment_audio_tensor.unsqueeze(0)
                
                # Convert to audio dict format
                segment_audio_dict = {
                    "waveform": segment_audio_tensor.cpu(),
                    "sample_rate": self.sample_rate
                }
                
                audio_segments.append(segment_audio_dict)
                segment_duration = segment_audio_dict["waveform"].size(-1) / self.sample_rate
                total_duration += segment_duration
                
                # Store timing info for advanced timing modes
                timing_segments.append({
                    "index": i,
                    "expected_start": segment_start,
                    "expected_end": segment_end,
                    "expected_duration": expected_duration,
                    "actual_duration": segment_duration,
                    "text": segment_text
                })

            # Check for interruption before timing assembly
            if model_management.interrupt_processing:
                raise InterruptedError("Higgs Audio SRT: Preparing audio assembly interrupted by user")

            # Apply timing modes using unified timing utilities
            timing_engine = TimingEngine(self.sample_rate)
            assembler = AudioAssemblyEngine(self.sample_rate)
            audio_tensors = [seg["waveform"] for seg in audio_segments]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if current_timing_mode == "stretch_to_fit":
                # Use the same modular approach as F5-TTS and ChatterBox
                from utils.system.import_manager import import_manager
                success, modules, source = import_manager.import_srt_modules()
                if not success:
                    raise ImportError("SRT modules not available for timing calculations")
                
                calculate_timing_adjustments = modules.get("calculate_timing_adjustments")
                if not calculate_timing_adjustments:
                    raise ImportError("calculate_timing_adjustments function not found")
                
                # Calculate natural durations and target timings
                natural_durations = [audio_tensors[i].size(-1) / self.sample_rate for i in range(len(audio_tensors))]
                target_timings = [(seg.start_time, seg.end_time) for seg in srt_segments]
                
                # Use modular timing adjustment calculation (same as F5-TTS)
                adjustments = calculate_timing_adjustments(natural_durations, target_timings)
                
                # Add sequence numbers and original text to adjustments
                for i, (adj, seg) in enumerate(zip(adjustments, srt_segments)):
                    adj['sequence'] = seg.sequence
                    adj['original_text'] = seg.text
                
                # Assemble final audio
                final_audio_tensor = assembler.assemble_stretch_to_fit(audio_tensors, target_timings, fade_for_StretchToFit)
            elif current_timing_mode == "pad_with_silence":
                final_audio_tensor = assembler.assemble_with_overlaps(audio_tensors, srt_segments, device)
                # Generate basic adjustments for reporting
                adjustments = []
                for i, seg in enumerate(srt_segments):
                    adjustments.append({
                        "sequence": seg.sequence,
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "original_text": seg.text,
                        "needs_stretching": False,
                        "natural_duration": audio_tensors[i].size(-1) / self.sample_rate if i < len(audio_tensors) else 0.0
                    })
            elif current_timing_mode == "concatenate":
                adjustments = timing_engine.calculate_concatenation_adjustments(audio_tensors, srt_segments)
                final_audio_tensor = assembler.assemble_concatenation(audio_tensors, fade_for_StretchToFit)
            else:  # smart_natural
                adjustments, processed_segments = timing_engine.calculate_smart_timing_adjustments(
                    audio_tensors, srt_segments, timing_tolerance, max_stretch_ratio, min_stretch_ratio, device
                )
                final_audio_tensor = assembler.assemble_smart_natural(
                    audio_tensors, processed_segments, adjustments, srt_segments, device
                )
            
            # Ensure correct tensor dimensions for ComfyUI: [channels, samples]
            if final_audio_tensor.dim() == 1:
                final_audio_tensor = final_audio_tensor.unsqueeze(0)  # Add channel dimension
            elif final_audio_tensor.dim() == 3:
                # Handle [batch, channels, samples] -> [channels, samples]
                if final_audio_tensor.size(0) == 1:  # batch size of 1
                    final_audio_tensor = final_audio_tensor.squeeze(0)  # Remove batch dimension
                else:
                    final_audio_tensor = final_audio_tensor[0]  # Take first item from batch
            
            # Format using audio processing utilities
            from utils.audio.processing import AudioProcessingUtils
            formatted_audio = AudioProcessingUtils.format_for_comfyui(final_audio_tensor.cpu(), self.sample_rate)
            
            # Generate proper timing report using unified utilities  
            reporter = SRTReportGenerator()
            timing_report = reporter.generate_timing_report(srt_segments, adjustments, current_timing_mode, has_overlaps, mode_switched, timing_mode if mode_switched else None)
            
            # Generate adjusted SRT string
            adjusted_srt = reporter.generate_adjusted_srt_string(srt_segments, adjustments, current_timing_mode)
            
            # Generate summary
            character_summary = f" with {len(all_characters)} characters ({', '.join(sorted(all_characters))})" if len(all_characters) > 1 else ""
            mode_info = f" using {multi_speaker_mode}"
            generation_info = f"Higgs Audio SRT: Generated {len(srt_segments)} segments{character_summary}{mode_info}"
            
            return (formatted_audio, generation_info, timing_report, adjusted_srt)
            
        except Exception as e:
            print(f"❌ Higgs Audio SRT processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty audio and error info
            from utils.audio.processing import AudioProcessingUtils
            empty_audio = AudioProcessingUtils.create_silence(1.0, self.sample_rate)
            empty_comfy = AudioProcessingUtils.format_for_comfyui(empty_audio, self.sample_rate)
            
            return (empty_comfy, f"Higgs Audio SRT error: {e}", "Error: No timing report available", "Error: No adjusted SRT available")