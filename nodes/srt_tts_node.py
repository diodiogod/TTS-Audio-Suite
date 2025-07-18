"""
ChatterBox SRT TTS Node - Migrated to use new foundation
SRT Subtitle-aware Text-to-Speech node using ChatterboxTTS with enhanced timing
"""

import torch
import numpy as np
import tempfile
import os
import hashlib
import gc
from typing import Dict, Any, Optional, List, Tuple

# Use direct file imports that work when loaded via importlib
import os
import sys
import importlib.util

# Add parent directory to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Load base_node module directly
base_node_path = os.path.join(current_dir, "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseTTSNode = base_module.BaseTTSNode

from core.import_manager import import_manager
from core.audio_processing import AudioProcessingUtils
from core.voice_discovery import get_available_voices, load_voice_reference, get_available_characters, get_character_mapping
from core.character_parser import parse_character_text, character_parser
import comfy.model_management as model_management

# Global audio cache - SAME AS ORIGINAL
GLOBAL_AUDIO_CACHE = {}


class ChatterboxSRTTTSNode(BaseTTSNode):
    """
    SRT Subtitle-aware Text-to-Speech node using ChatterboxTTS
    Generates timed audio that matches SRT subtitle timing
    """
    
    def __init__(self):
        super().__init__()
        self.srt_available = False
        self.srt_modules = {}
        self._load_srt_modules()
    
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
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "srt_content": ("STRING", {
                    "multiline": True,
                    "default": """1
00:00:01,000 --> 00:00:04,000
Hello! This is the first subtitle. I'll make it long on purpose.

2
00:00:04,500 --> 00:00:09,500
This is the second subtitle with precise timing.

3
00:00:10,000 --> 00:00:14,000
The audio will match these exact timings.""",
                    "tooltip": "The SRT subtitle content. Each entry defines a text segment and its precise start and end times."
                }),
                "device": (["auto", "cuda", "cpu"], {"default": "auto", "tooltip": "The device to run the TTS model on (auto, cuda, or cpu)."}),
                "exaggeration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.25,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Controls the expressiveness and emphasis of the generated speech. Higher values increase exaggeration."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.05,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Controls the randomness and creativity of the generated speech. Higher values lead to more varied outputs."
                }),
                "cfg_weight": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Classifier-Free Guidance weight. Influences how strongly the model adheres to the input text."
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "tooltip": "Seed for reproducible speech generation. Set to 0 for random."}),
                "timing_mode": (["stretch_to_fit", "pad_with_silence", "smart_natural"], {
                    "default": "smart_natural",
                    "tooltip": "Determines how audio segments are aligned with SRT timings:\n🔹 stretch_to_fit: Stretches/compresses audio to exactly match SRT segment durations.\n🔹 pad_with_silence: Places natural audio at SRT start times, padding gaps with silence. May result in overlaps.\n🔹 smart_natural: Intelligently adjusts timings within 'timing_tolerance', prioritizing natural audio and shifting subsequent segments. Applies stretch/shrink within limits if needed."
                }),
            },
            "optional": {
                "reference_audio": ("AUDIO", {"tooltip": "Optional reference audio input from another ComfyUI node for voice cloning or style transfer. This is an alternative to 'audio_prompt_path'."}),
                "audio_prompt_path": ("STRING", {"default": "", "tooltip": "Path to an audio file on disk to use as a prompt for voice cloning or style transfer. This is an alternative to 'reference_audio'."}),
                "enable_audio_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, generated audio segments will be cached in memory to speed up subsequent runs with identical parameters."
                }),
                "fade_for_StretchToFit": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Duration (in seconds) for crossfading between audio segments in 'stretch_to_fit' mode."
                }),
                "max_stretch_ratio": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Maximum factor to slow down audio in 'smart_natural' mode. (e.g., 2.0x means audio can be twice as long). Recommend leaving at 1.0 for natural speech preservation and silence addition."
                }),
                "min_stretch_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Minimum factor to speed up audio in 'smart_natural' mode. (e.g., 0.5x means audio can be half as long). min=faster speech"
                }),
                "timing_tolerance": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Maximum allowed deviation (in seconds) for timing adjustments in 'smart_natural' mode. Higher values allow more flexibility."
                }),
                "crash_protection_template": ("STRING", {
                    "default": "hmm ,, {seg} hmm ,,",
                    "tooltip": "Custom padding template for short text segments to prevent ChatterBox crashes. ChatterBox has a bug where text shorter than ~21 characters causes CUDA tensor errors in sequential generation. Use {seg} as placeholder for the original text. Examples: '...ummmmm {seg}' (default hesitation), '{seg}... yes... {seg}' (repetition), 'Well, {seg}' (natural prefix), or empty string to disable padding. This only affects ChatterBox nodes, not F5-TTS nodes."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "generation_info", "timing_report", "Adjusted_SRT")
    FUNCTION = "generate_srt_speech"
    CATEGORY = "ChatterBox Voice"

    def _pad_short_text_for_chatterbox(self, text: str, padding_template: str = "...ummmmm {seg}", min_length: int = 21) -> str:
        """
        Add custom padding to short text to prevent ChatterBox crashes.
        
        ChatterBox has a bug where short text segments cause CUDA tensor indexing errors
        in sequential generation scenarios. Adding meaningful tokens with custom templates
        prevents these crashes while allowing user customization.
        
        Args:
            text: Input text to check and pad if needed
            padding_template: Custom template with {seg} placeholder for original text
            min_length: Minimum text length threshold (default: 21 characters)
            
        Returns:
            Original text or text with custom padding template if too short
        """
        stripped_text = text.strip()
        if len(stripped_text) < min_length:
            # If template is empty, disable padding
            if not padding_template.strip():
                return text
            # Replace {seg} placeholder with original text
            return padding_template.replace("{seg}", stripped_text)
        return text

    def _safe_generate_tts_audio(self, text, audio_prompt, exaggeration, temperature, cfg_weight):
        """
        Wrapper around generate_tts_audio - simplified to just call the base method.
        CUDA crash recovery was removed as it didn't work reliably.
        """
        try:
            return self.generate_tts_audio(text, audio_prompt, exaggeration, temperature, cfg_weight)
        except Exception as e:
            error_msg = str(e)
            is_cuda_crash = ("srcIndex < srcSelectDimSize" in error_msg or 
                           "CUDA" in error_msg or 
                           "device-side assert" in error_msg or
                           "an illegal memory access" in error_msg)
            
            if is_cuda_crash:
                print(f"🚨 ChatterBox CUDA crash detected: '{text[:50]}...'")
                print(f"🛡️ This is a known ChatterBox bug with certain text patterns.")
                raise RuntimeError(f"ChatterBox CUDA crash occurred. Text: '{text[:50]}...' - Try using padding template or longer text, or restart ComfyUI.")
            else:
                raise

    def _generate_segment_cache_key(self, subtitle_text: str, exaggeration: float, temperature: float, 
                                   cfg_weight: float, seed: int, audio_prompt_component: str, 
                                   model_source: str, device: str) -> str:
        """Generate cache key for a single audio segment based on generation parameters."""
        cache_data = {
            'text': subtitle_text,
            'exaggeration': exaggeration,
            'temperature': temperature,
            'cfg_weight': cfg_weight,
            'seed': seed,
            'audio_prompt_component': audio_prompt_component,
            'model_source': model_source,
            'device': device
        }
        cache_string = str(sorted(cache_data.items()))
        cache_key = hashlib.md5(cache_string.encode()).hexdigest()
        return cache_key

    def _get_cached_segment_audio(self, segment_cache_key: str) -> Optional[Tuple[torch.Tensor, float]]:
        """Retrieve cached audio for a single segment if available from global cache - ORIGINAL BEHAVIOR"""
        return GLOBAL_AUDIO_CACHE.get(segment_cache_key)

    def _cache_segment_audio(self, segment_cache_key: str, audio_tensor: torch.Tensor, natural_duration: float):
        """Cache generated audio for a single segment in global cache - ORIGINAL BEHAVIOR"""
        GLOBAL_AUDIO_CACHE[segment_cache_key] = (audio_tensor.clone(), natural_duration)
    
    def _detect_overlaps(self, subtitles: List) -> bool:
        """Detect if subtitles have overlapping time ranges."""
        for i in range(len(subtitles) - 1):
            current = subtitles[i]
            next_sub = subtitles[i + 1]
            if current.end_time > next_sub.start_time:
                return True
        return False

    def generate_srt_speech(self, srt_content, device, exaggeration, temperature, cfg_weight, seed,
                            timing_mode, reference_audio=None, audio_prompt_path="",
                            max_stretch_ratio=2.0, min_stretch_ratio=0.5, fade_for_StretchToFit=0.01, 
                            enable_audio_cache=True, timing_tolerance=2.0, 
                            crash_protection_template="hmm ,, {seg} hmm ,,"):
        
        def _process():
            # Check if SRT support is available
            if not self.srt_available:
                raise ImportError("SRT support not available - missing required modules")
            
            # Load TTS model
            self.load_tts_model(device)
            
            # Set seed for reproducibility
            self.set_seed(seed)
            
            # Determine audio prompt component for cache key generation (stable identifier)
            # This must be done BEFORE handle_reference_audio to avoid using temporary file paths
            stable_audio_prompt_component = ""
            # print(f"🔍 Stable Cache DEBUG: reference_audio is None: {reference_audio is None}")
            # print(f"🔍 Stable Cache DEBUG: audio_prompt_path: {repr(audio_prompt_path)}")
            if reference_audio is not None:
                waveform_hash = hashlib.md5(reference_audio["waveform"].cpu().numpy().tobytes()).hexdigest()
                stable_audio_prompt_component = f"ref_audio_{waveform_hash}_{reference_audio['sample_rate']}"
                # print(f"🔍 Stable Cache DEBUG: Using reference_audio hash: {stable_audio_prompt_component}")
            elif audio_prompt_path:
                stable_audio_prompt_component = audio_prompt_path
                # print(f"🔍 Stable Cache DEBUG: Using audio_prompt_path: {stable_audio_prompt_component}")
            else:
                # print(f"🔍 Stable Cache DEBUG: No reference audio or path provided")
                pass
            
            # Handle reference audio (this may create temporary files, but we don't use them in cache key)
            audio_prompt = self.handle_reference_audio(reference_audio, audio_prompt_path)
            
            # Parse SRT content with overlap support
            srt_parser = self.SRTParser()
            subtitles = srt_parser.parse_srt_content(srt_content, allow_overlaps=True)
            
            # Check if subtitles have overlaps and handle smart_natural mode
            has_overlaps = self._detect_overlaps(subtitles)
            current_timing_mode = timing_mode
            mode_switched = False
            if has_overlaps and current_timing_mode == "smart_natural":
                print("⚠️ ChatterBox SRT: Overlapping subtitles detected, switching from smart_natural to pad_with_silence mode")
                current_timing_mode = "pad_with_silence"
                mode_switched = True
            
            # Generate audio segments
            audio_segments = []
            natural_durations = []
            any_segment_cached = False
            
            for i, subtitle in enumerate(subtitles):
                # Check for interruption
                self.check_interruption(f"SRT generation segment {i+1}/{len(subtitles)} (Seq {subtitle.sequence})")
                
                if not subtitle.text.strip():
                    # Handle empty text by creating silence
                    natural_duration = subtitle.duration
                    wav = self.AudioTimingUtils.create_silence(
                        duration_seconds=natural_duration,
                        sample_rate=self.tts_model.sr,
                        channels=1,
                        device=self.device
                    )
                    print(f"🤫 Segment {i+1} (Seq {subtitle.sequence}): Empty text, generating {natural_duration:.2f}s silence.")
                else:
                    # ENHANCED: Parse character tags from subtitle text (always parse to respect line breaks)
                    character_segments = parse_character_text(subtitle.text)
                    
                    if len(character_segments) > 1 or (len(character_segments) == 1 and character_segments[0][0] != "narrator"):
                        # Character switching within this subtitle
                        # print(f"🎭 ChatterBox SRT Segment {i+1} (Seq {subtitle.sequence}): Character switching detected")
                        
                        # Set up character parser with available characters
                        available_chars = get_available_characters()
                        character_parser.set_available_characters(list(available_chars))
                        
                        # Get character mapping for ChatterBox (audio-only)
                        characters = [char for char, _ in character_segments]
                        character_mapping = get_character_mapping(characters, engine_type="chatterbox")
                        
                        # Generate audio for each character segment within this subtitle
                        segment_audio_parts = []
                        for char, segment_text in character_segments:
                            # Get character voice or fallback to main
                            char_audio, _ = character_mapping.get(char, (None, None))
                            # print(f"🔍 Character Mapping DEBUG: char='{char}', char_audio={repr(char_audio)}")
                            # print(f"🔍 Character Mapping DEBUG: audio_prompt={repr(audio_prompt)}")
                            # print(f"🔍 Character Mapping DEBUG: stable_audio_prompt_component={repr(stable_audio_prompt_component)}")
                            
                            # Store the original char_audio for cache key generation
                            original_char_audio = char_audio
                            
                            if not char_audio:
                                char_audio = audio_prompt  # For actual TTS generation
                                # Character not found, will use main voice (no message needed, handled in generation)
                            # else:
                            #     print(f"🎭 Using character voice for '{char}' in subtitle {subtitle.sequence}")
                            
                            # Generate cache key for this character segment
                            # Use stable component for cache key when falling back to main voice
                            audio_component = original_char_audio or stable_audio_prompt_component
                            # print(f"🔍 Cache DEBUG: Character '{char}' using audio_component: {repr(audio_component)}")
                            char_segment_cache_key = self._generate_segment_cache_key(
                                f"{char}:{segment_text}", exaggeration, temperature, cfg_weight, seed,
                                audio_component, self.model_manager.get_model_source("tts"), device
                            )
                            # print(f"🔍 Cache DEBUG: Generated cache key: {char_segment_cache_key[:50]}...")
                            
                            # Try to get cached audio
                            cached_data = self._get_cached_segment_audio(char_segment_cache_key) if enable_audio_cache else None
                            
                            if cached_data:
                                char_wav, _ = cached_data
                                any_segment_cached = True
                                print(f"💾 Using cached audio for character '{char}'")
                            else:
                                # Show generation message with character info
                                if char == "narrator":
                                    print(f"📺 Generating SRT segment {i+1}/{len(subtitles)} (Seq {subtitle.sequence})...")
                                else:
                                    print(f"🎭 Generating SRT segment {i+1}/{len(subtitles)} (Seq {subtitle.sequence}) using '{char}'")
                                # BUGFIX: Pad short text with custom template to prevent ChatterBox sequential generation crashes
                                processed_segment_text = self._pad_short_text_for_chatterbox(segment_text, crash_protection_template)
                                
                                # DEBUG: Show actual text being sent to ChatterBox when padding might occur
                                if len(segment_text.strip()) < 21:
                                    print(f"🔍 DEBUG: Original text: '{segment_text}' → Processed: '{processed_segment_text}' (len: {len(processed_segment_text)})")
                                
                                # Generate new audio for this character segment with CUDA recovery
                                char_wav = self._safe_generate_tts_audio(
                                    processed_segment_text, char_audio, exaggeration, temperature, cfg_weight
                                )
                                
                                if enable_audio_cache:
                                    char_duration = self.AudioTimingUtils.get_audio_duration(char_wav, self.tts_model.sr)
                                    self._cache_segment_audio(char_segment_cache_key, char_wav, char_duration)
                            
                            segment_audio_parts.append(char_wav)
                        
                        # Concatenate all character segments for this subtitle
                        wav = torch.cat(segment_audio_parts, dim=1)
                        natural_duration = self.AudioTimingUtils.get_audio_duration(wav, self.tts_model.sr)
                        
                    else:
                        # Single character/narrator mode (original behavior)
                        # Generate cache key for this segment
                        model_source = self.model_manager.get_model_source("tts")
                        segment_cache_key = self._generate_segment_cache_key(
                            subtitle.text, exaggeration, temperature, cfg_weight, seed,
                            stable_audio_prompt_component, model_source, device
                        )
                        
                        # Try to get cached audio
                        cached_data = self._get_cached_segment_audio(segment_cache_key) if enable_audio_cache else None
                        
                        if cached_data:
                            wav, natural_duration = cached_data
                            any_segment_cached = True
                        else:
                            # Generate new audio
                            print(f"📺 Generating SRT segment {i+1}/{len(subtitles)} (Seq {subtitle.sequence})...")
                            
                            # BUGFIX: Pad short text with custom template to prevent ChatterBox sequential generation crashes
                            processed_subtitle_text = self._pad_short_text_for_chatterbox(subtitle.text, crash_protection_template)
                            
                            # DEBUG: Show actual text being sent to ChatterBox when padding might occur
                            if len(subtitle.text.strip()) < 21:
                                print(f"🔍 DEBUG: Original text: '{subtitle.text}' → Processed: '{processed_subtitle_text}' (len: {len(processed_subtitle_text)})")
                            
                            # Generate new audio with CUDA recovery
                            wav = self._safe_generate_tts_audio(
                                processed_subtitle_text, audio_prompt, exaggeration, temperature, cfg_weight
                            )
                            natural_duration = self.AudioTimingUtils.get_audio_duration(wav, self.tts_model.sr)
                            
                            if enable_audio_cache:
                                self._cache_segment_audio(segment_cache_key, wav, natural_duration)
                
                audio_segments.append(wav)
                natural_durations.append(natural_duration)
            
            # Calculate timing adjustments
            target_timings = [(sub.start_time, sub.end_time) for sub in subtitles]
            adjustments = self.calculate_timing_adjustments(natural_durations, target_timings)
            
            # Add sequence numbers to adjustments
            for i, (adj, subtitle) in enumerate(zip(adjustments, subtitles)):
                adj['sequence'] = subtitle.sequence
            
            # Assemble final audio based on timing mode - ORIGINAL LOGIC
            if current_timing_mode == "stretch_to_fit":
                # Use time stretching to match exact timing - ORIGINAL IMPLEMENTATION
                assembler = self.TimedAudioAssembler(self.tts_model.sr)
                final_audio = assembler.assemble_timed_audio(
                    audio_segments, target_timings, fade_duration=fade_for_StretchToFit
                )
            elif current_timing_mode == "pad_with_silence":
                # Add silence to match timing without stretching - ORIGINAL IMPLEMENTATION
                final_audio = self._assemble_audio_with_overlaps(audio_segments, subtitles, self.tts_model.sr)
            else:  # smart_natural
                # Smart balanced timing: use natural audio but add minimal adjustments within tolerance - ORIGINAL IMPLEMENTATION
                final_audio, smart_adjustments = self._assemble_with_smart_timing(
                    audio_segments, subtitles, self.tts_model.sr, timing_tolerance,
                    max_stretch_ratio, min_stretch_ratio
                )
                adjustments = smart_adjustments
            
            # Generate reports
            timing_report = self._generate_timing_report(subtitles, adjustments, current_timing_mode, has_overlaps, mode_switched, timing_mode if mode_switched else None)
            adjusted_srt_string = self._generate_adjusted_srt_string(subtitles, adjustments, current_timing_mode)
            
            # Generate info with cache status and stretching method - ORIGINAL LOGIC FROM LINES 1141-1168
            total_duration = self.AudioTimingUtils.get_audio_duration(final_audio, self.tts_model.sr)
            cache_status = "cached" if any_segment_cached else "generated"
            model_source = self.model_manager.get_model_source("tts")
            stretch_info = ""
            
            # Get stretching method info - ORIGINAL LOGIC
            if current_timing_mode == "stretch_to_fit":
                current_stretcher = assembler.time_stretcher
            elif current_timing_mode == "smart_natural":
                # Use the stored stretcher type for smart_natural mode
                if hasattr(self, '_smart_natural_stretcher'):
                    if self._smart_natural_stretcher == "ffmpeg":
                        stretch_info = ", Stretching method: FFmpeg"
                    else:
                        stretch_info = ", Stretching method: Phase Vocoder"
                else:
                    stretch_info = ", Stretching method: Unknown"
            
            # For stretch_to_fit mode, examine the actual stretcher - ORIGINAL LOGIC
            if current_timing_mode == "stretch_to_fit" and 'current_stretcher' in locals():
                if isinstance(current_stretcher, self.FFmpegTimeStretcher):
                    stretch_info = ", Stretching method: FFmpeg"
                elif isinstance(current_stretcher, self.PhaseVocoderTimeStretcher):
                    stretch_info = ", Stretching method: Phase Vocoder"
                else:
                    stretch_info = f", Stretching method: {current_stretcher.__class__.__name__}"
            
            mode_info = f"{current_timing_mode}"
            if mode_switched:
                mode_info = f"{current_timing_mode} (switched from {timing_mode} due to overlaps)"
            
            info = (f"Generated {total_duration:.1f}s SRT-timed audio from {len(subtitles)} subtitles "
                   f"using {mode_info} mode ({cache_status} segments, {model_source} models{stretch_info})")
            
            # Format final audio for ComfyUI
            if final_audio.dim() == 1:
                final_audio = final_audio.unsqueeze(0)  # Add channel dimension
            
            return (
                self.format_audio_output(final_audio, self.tts_model.sr),
                info,
                timing_report,
                adjusted_srt_string
            )
        
        return self.process_with_error_handling(_process)
    
    def _assemble_audio_with_overlaps(self, audio_segments: List[torch.Tensor],
                                     subtitles: List, sample_rate: int) -> torch.Tensor:
        """Assemble audio by placing segments at their SRT start times, allowing overlaps."""
        # Delegate to audio assembly engine with EXACT original logic
        from chatterbox_srt.audio_assembly import AudioAssemblyEngine
        assembler = AudioAssemblyEngine(sample_rate)
        return assembler.assemble_with_overlaps(audio_segments, subtitles, self.device)
    
    def _assemble_with_smart_timing(self, audio_segments: List[torch.Tensor],
                                   subtitles: List, sample_rate: int, tolerance: float,
                                   max_stretch_ratio: float, min_stretch_ratio: float) -> Tuple[torch.Tensor, List[Dict]]:
        """Smart timing assembly with intelligent adjustments - ORIGINAL SMART NATURAL LOGIC"""
        # Initialize stretcher for smart_natural mode - ORIGINAL LOGIC FROM LINES 1524-1535
        try:
            # Try FFmpeg first
            print("Smart natural mode: Trying FFmpeg stretcher...")
            time_stretcher = self.FFmpegTimeStretcher()
            self._smart_natural_stretcher = "ffmpeg"
            print("Smart natural mode: Using FFmpeg stretcher")
        except self.AudioTimingError as e:
            # Fall back to Phase Vocoder
            print(f"Smart natural mode: FFmpeg initialization failed ({str(e)}), falling back to Phase Vocoder")
            time_stretcher = self.PhaseVocoderTimeStretcher()
            self._smart_natural_stretcher = "phase_vocoder"
            print("Smart natural mode: Using Phase Vocoder stretcher")
        
        # Delegate to timing engine for complex calculations
        from chatterbox_srt.timing_engine import TimingEngine
        from chatterbox_srt.audio_assembly import AudioAssemblyEngine
        
        timing_engine = TimingEngine(sample_rate)
        assembler = AudioAssemblyEngine(sample_rate)
        
        # Calculate smart adjustments and process segments
        adjustments, processed_segments = timing_engine.calculate_smart_timing_adjustments(
            audio_segments, subtitles, tolerance, max_stretch_ratio, min_stretch_ratio, self.device
        )
        
        # Assemble the final audio
        final_audio = assembler.assemble_smart_natural(audio_segments, processed_segments, adjustments, subtitles, self.device)
        
        return final_audio, adjustments
    
    def _generate_timing_report(self, subtitles: List, adjustments: List[Dict], timing_mode: str, has_original_overlaps: bool = False, mode_switched: bool = False, original_mode: str = None) -> str:
        """Generate detailed timing report."""
        # Delegate to reporting module
        from chatterbox_srt.reporting import SRTReportGenerator
        reporter = SRTReportGenerator()
        return reporter.generate_timing_report(subtitles, adjustments, timing_mode, has_original_overlaps, mode_switched, original_mode)
    
    def _generate_adjusted_srt_string(self, subtitles: List, adjustments: List[Dict], timing_mode: str) -> str:
        """Generate adjusted SRT string from final timings."""
        # Delegate to reporting module
        from chatterbox_srt.reporting import SRTReportGenerator
        reporter = SRTReportGenerator()
        return reporter.generate_adjusted_srt_string(subtitles, adjustments, timing_mode)