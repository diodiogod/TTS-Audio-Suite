"""
Higgs Audio SRT Processor - Handles SRT processing for Higgs Audio 2 TTS engine
Internal processor used by UnifiedTTSSRTNode - not a ComfyUI node itself
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
            from utils.text.character_parser import parse_character_text
            from utils.voice.discovery import get_character_mapping
            from utils.text.pause_processor import PauseTagProcessor
            from utils.timing.engine import TimingEngine  
            from utils.timing.assembly import AudioAssemblyEngine
            from utils.timing.reporting import SRTReportGenerator
            
            print(f"📺 Higgs Audio SRT: Processing SRT with multi-speaker support")
            
            # Parse SRT content
            srt_parser = SRTParser()
            srt_segments = srt_parser.parse_srt_content(srt_content, allow_overlaps=True)
            print(f"📺 Higgs Audio SRT: Found {len(srt_segments)} SRT segments")
            
            print(f"🎭 Higgs Audio SRT: Using mode '{multi_speaker_mode}'")
            
            # Analyze all text for character discovery
            all_text = " ".join([seg.text for seg in srt_segments])
            character_segments = parse_character_text(all_text)
            all_characters = set(char for char, _ in character_segments)
            
            if len(all_characters) > 1 or (len(all_characters) == 1 and "narrator" not in all_characters):
                print(f"🎭 Higgs Audio SRT: Detected character switching - {', '.join(sorted(all_characters))}")
            
            # Generate audio for each SRT segment
            audio_segments = []
            timing_segments = []
            total_duration = 0.0
            
            # Build voice references for character switching mode
            voice_refs = {}
            if multi_speaker_mode == "Custom Character Switching":
                character_mapping = get_character_mapping(list(all_characters), engine_type="higgs_audio")
                
                # Start with narrator using connected voice
                narrator_voice_dict = None
                if audio_tensor is not None:
                    narrator_voice_dict = {"waveform": audio_tensor["waveform"], "sample_rate": audio_tensor["sample_rate"]}
                
                voice_refs = {'narrator': narrator_voice_dict}
                
                # Add character-specific voices
                for character in all_characters:
                    if character.lower() == "narrator":
                        continue
                    
                    audio_path, _ = character_mapping.get(character, (None, None))
                    if audio_path and os.path.exists(audio_path):
                        import torchaudio
                        waveform, sample_rate = torchaudio.load(audio_path)
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                        voice_refs[character] = {"waveform": waveform, "sample_rate": sample_rate}
                    else:
                        voice_refs[character] = narrator_voice_dict
            
            for i, segment in enumerate(srt_segments):
                segment_text = segment.text
                segment_start = segment.start_time
                segment_end = segment.end_time
                expected_duration = segment_end - segment_start
                
                print(f"📺 Processing SRT segment {i+1}/{len(srt_segments)}: '{segment_text[:50]}...' ({expected_duration:.2f}s)")
                
                if multi_speaker_mode == "Custom Character Switching":
                    # Use character switching with pause tag support
                    def srt_tts_generate_func(text_content: str) -> torch.Tensor:
                        """TTS generation function for SRT segment with character switching"""
                        if '[' in text_content and ']' in text_content:
                            # Handle character switching within this SRT segment
                            char_segments = parse_character_text(text_content)
                            segment_audio_parts = []
                            
                            for character, segment_text in char_segments:
                                char_audio_dict = voice_refs.get(character, voice_refs.get("narrator"))
                                char_ref_text = reference_text or ""
                                
                                segment_result = self.engine_wrapper.generate_srt_audio(
                                    srt_content="",  # Individual segment processing
                                    text=segment_text,
                                    char_audio=char_audio_dict,
                                    char_text=char_ref_text,
                                    character=character,
                                    seed=seed + i,  # Vary seed per segment
                                    enable_audio_cache=enable_audio_cache
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
                                char_text=reference_text or "",
                                character="narrator",
                                seed=seed + i,  # Vary seed per segment
                                enable_audio_cache=enable_audio_cache
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
                        seed=seed + i,  # Vary seed per segment
                        enable_audio_cache=enable_audio_cache,
                        multi_speaker_mode=multi_speaker_mode,
                        second_narrator_audio=opt_second_narrator,
                        second_narrator_text=""
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
            
            # Apply timing modes using unified timing utilities
            timing_engine = TimingEngine(self.sample_rate)
            assembler = AudioAssemblyEngine(self.sample_rate)
            audio_tensors = [seg["waveform"] for seg in audio_segments]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if timing_mode == "stretch_to_fit":
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
            elif timing_mode == "pad_with_silence":
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
            elif timing_mode == "concatenate":
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
            timing_report = reporter.generate_timing_report(srt_segments, adjustments, timing_mode)
            
            # Generate adjusted SRT string
            adjusted_srt = reporter.generate_adjusted_srt_string(srt_segments, adjustments, timing_mode)
            
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