"""
F5-TTS SRT Node - SRT Subtitle-aware Text-to-Speech using F5-TTS
Enhanced F5-TTS node with SRT timing support and voice cloning capabilities
"""

import torch
import numpy as np
import tempfile
import os
import hashlib
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

# Load f5tts_base_node module directly
f5tts_base_node_path = os.path.join(current_dir, "f5tts_base_node.py")
f5tts_base_spec = importlib.util.spec_from_file_location("f5tts_base_node_module", f5tts_base_node_path)
f5tts_base_module = importlib.util.module_from_spec(f5tts_base_spec)
sys.modules["f5tts_base_node_module"] = f5tts_base_module
f5tts_base_spec.loader.exec_module(f5tts_base_module)

# Import the base class
BaseF5TTSNode = f5tts_base_module.BaseF5TTSNode

from core.import_manager import import_manager
from core.audio_processing import AudioProcessingUtils
import comfy.model_management as model_management

# Global audio cache - SAME AS ORIGINAL
GLOBAL_AUDIO_CACHE = {}


class F5TTSSRTNode(BaseF5TTSNode):
    """
    SRT Subtitle-aware Text-to-Speech node using F5-TTS
    Generates timed audio that matches SRT subtitle timing with F5-TTS voice cloning
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
    def NAME(cls):
        return "📺 F5-TTS SRT Voice Generation"
    
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        import os
        from pathlib import Path
        
        # Get available reference audio files with companion .txt files from models/voices/
        models_dir = folder_paths.models_dir
        voices_dir = os.path.join(models_dir, "voices")
        reference_files = ["none"]  # Add "none" option for manual input
        
        if os.path.exists(voices_dir):
            voice_files = folder_paths.filter_files_content_types(
                os.listdir(voices_dir), ["audio", "video"]
            )
            for file in voice_files:
                # Check if companion .txt file exists
                full_file_path = os.path.join(voices_dir, file)
                txt_file = cls._get_companion_txt_file(full_file_path)
                if os.path.isfile(txt_file):
                    reference_files.append(file)
        
        reference_files = sorted(reference_files)
        
        return {
            "required": {
                "srt_content": ("STRING", {
                    "multiline": True,
                    "default": """1
00:00:01,000 --> 00:00:04,000
Hello! This is the first subtitle with F5-TTS voice cloning.

2
00:00:04,500 --> 00:00:09,500
This is the second subtitle with precise timing.

3
00:00:10,000 --> 00:00:14,000
The audio will match these exact timings using F5-TTS.""",
                    "tooltip": "The SRT subtitle content. Each entry defines a text segment and its precise start and end times."
                }),
                "reference_audio_file": (reference_files, {
                    "default": "none",
                    "tooltip": "Reference voice from models/voices/ folder (with companion .txt file). Select 'none' to use direct inputs below."
                }),
                "opt_reference_text": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Direct reference text input (required when using opt_reference_audio or when reference_audio_file is 'none')."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run F5-TTS model on. 'auto' selects best available (GPU if available, otherwise CPU)."
                }),
                "model": (["F5TTS_Base", "F5TTS_v1_Base", "E2TTS_Base"], {
                    "default": "F5TTS_Base",
                    "tooltip": "F5-TTS model variant to use. F5TTS_Base is the standard model, F5TTS_v1_Base is improved version, E2TTS_Base is enhanced variant."
                }),
                "seed": ("INT", {
                    "default": 1, "min": 0, "max": 2**32 - 1,
                    "tooltip": "Seed for reproducible F5-TTS generation. Same seed with same inputs will produce identical results. Set to 0 for random generation."
                }),
                "timing_mode": (["stretch_to_fit", "pad_with_silence", "smart_natural"], {
                    "default": "smart_natural",
                    "tooltip": "Determines how audio segments are aligned with SRT timings:\n🔹 stretch_to_fit: Stretches/compresses audio to exactly match SRT segment durations.\n🔹 pad_with_silence: Places natural audio at SRT start times, padding gaps with silence. May result in overlaps.\n🔹 smart_natural: Intelligently adjusts timings within 'timing_tolerance', prioritizing natural audio and shifting subsequent segments. Applies stretch/shrink within limits if needed."
                }),
            },
            "optional": {
                "opt_reference_audio": ("AUDIO", {
                    "tooltip": "Direct reference audio input (used when reference_audio_file is 'none')"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "Controls randomness in F5-TTS generation. Higher values = more creative/varied speech, lower values = more consistent/predictable speech."
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1,
                    "tooltip": "F5-TTS native speech speed control. 1.0 = normal speed, 0.5 = half speed (slower), 2.0 = double speed (faster). This affects natural speech generation, separate from SRT timing modes."
                }),
                "target_rms": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Target audio volume level (Root Mean Square). Controls output loudness normalization. Higher values = louder audio output."
                }),
                "cross_fade_duration": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Duration in seconds for smooth audio transitions between F5-TTS segments. Prevents audio clicks/pops by blending segment boundaries."
                }),
                "nfe_step": ("INT", {
                    "default": 32, "min": 1, "max": 71,
                    "tooltip": "Neural Function Evaluation steps for F5-TTS inference. Higher values = better quality but slower generation. 32 is a good balance. Values above 71 may cause ODE solver issues."
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Classifier-Free Guidance strength. Controls how strictly F5-TTS follows the reference text. Higher values = more adherence to reference, lower values = more creative freedom."
                }),
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
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "generation_info", "timing_report", "Adjusted_SRT")
    FUNCTION = "generate_srt_speech"
    CATEGORY = "F5-TTS Voice"

    @staticmethod
    def _get_companion_txt_file(audio_file_path):
        """Get the path to companion .txt file for an audio file"""
        from pathlib import Path
        p = Path(audio_file_path)
        return os.path.join(os.path.dirname(audio_file_path), p.stem + ".txt")
    
    def _load_reference_from_file(self, reference_audio_file):
        """Load reference audio and text from models/voices/ folder"""
        import folder_paths
        
        if reference_audio_file == "none":
            return None, None
        
        # Get full audio file path from models/voices/
        models_dir = folder_paths.models_dir
        voices_dir = os.path.join(models_dir, "voices")
        audio_path = os.path.join(voices_dir, reference_audio_file)
        
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {audio_path}")
        
        # Get companion text file
        txt_file = self._get_companion_txt_file(audio_path)
        
        if not os.path.isfile(txt_file):
            raise FileNotFoundError(f"Companion text file not found: {txt_file}")
        
        # Read reference text
        with open(txt_file, 'r', encoding='utf-8') as file:
            ref_text = file.read().strip()
        
        return audio_path, ref_text
    
    def _handle_reference_with_priority_chain(self, inputs):
        """Handle reference audio and text with improved priority chain"""
        reference_audio_file = inputs.get("reference_audio_file", "none")
        opt_reference_audio = inputs.get("opt_reference_audio")
        opt_reference_text = inputs.get("opt_reference_text", "").strip()
        
        # PRIORITY 1: Check reference_audio_file first
        if reference_audio_file != "none":
            try:
                audio_path, auto_ref_text = self._load_reference_from_file(reference_audio_file)
                if audio_path and auto_ref_text:
                    print(f"✅ F5-TTS SRT: Using reference file '{reference_audio_file}' with auto-detected text")
                    return audio_path, auto_ref_text
            except Exception as e:
                print(f"⚠️ F5-TTS SRT: Failed to load reference file '{reference_audio_file}': {e}")
                print("🔄 F5-TTS SRT: Falling back to manual inputs...")
        
        # PRIORITY 2: Use opt_reference_audio + opt_reference_text (both required)
        if opt_reference_audio is not None:
            # Handle the audio input to get file path
            audio_prompt = self.handle_reference_audio(opt_reference_audio, "")
            
            if audio_prompt:
                # Check if opt_reference_text is provided
                if opt_reference_text and opt_reference_text.strip():
                    print(f"📝 F5-TTS SRT: Using direct reference audio + text inputs")
                    return audio_prompt, opt_reference_text.strip()
                
                # Error - audio provided but no text
                raise ValueError(
                    "F5-TTS SRT requires reference text. Please connect text to opt_reference_text input."
                )
        
        # FINAL: No reference inputs provided at all
        raise ValueError(
            "F5-TTS SRT requires reference audio and text. Please provide either:\n"
            "1. Select a reference_audio_file with companion .txt file, OR\n"
            "2. Connect opt_reference_audio input and provide opt_reference_text"
        )

    def _generate_segment_cache_key(self, subtitle_text: str, model_name: str, device: str,
                                   audio_prompt_component: str, ref_text: str, temperature: float,
                                   speed: float, target_rms: float, cross_fade_duration: float,
                                   nfe_step: int, cfg_strength: float, seed: int) -> str:
        """Generate cache key for a single F5-TTS audio segment based on generation parameters."""
        cache_data = {
            'text': subtitle_text,
            'model_name': model_name,
            'device': device,
            'audio_prompt_component': audio_prompt_component,
            'ref_text': ref_text,
            'temperature': temperature,
            'speed': speed,
            'target_rms': target_rms,
            'cross_fade_duration': cross_fade_duration,
            'nfe_step': nfe_step,
            'cfg_strength': cfg_strength,
            'seed': seed,
            'engine': 'f5tts'
        }
        cache_string = str(sorted(cache_data.items()))
        cache_key = hashlib.md5(cache_string.encode()).hexdigest()
        return cache_key

    def _get_cached_segment_audio(self, segment_cache_key: str) -> Optional[Tuple[torch.Tensor, float]]:
        """Retrieve cached audio for a single segment if available from global cache"""
        return GLOBAL_AUDIO_CACHE.get(segment_cache_key)

    def _cache_segment_audio(self, segment_cache_key: str, audio_tensor: torch.Tensor, natural_duration: float):
        """Cache generated audio for a single segment in global cache"""
        GLOBAL_AUDIO_CACHE[segment_cache_key] = (audio_tensor.clone(), natural_duration)

    def generate_srt_speech(self, srt_content, reference_audio_file, opt_reference_text, device, model, seed,
                           timing_mode, opt_reference_audio=None, temperature=0.8, speed=1.0, target_rms=0.1,
                           cross_fade_duration=0.15, nfe_step=32, cfg_strength=2.0, enable_audio_cache=True,
                           fade_for_StretchToFit=0.01, max_stretch_ratio=1.0, min_stretch_ratio=0.5,
                           timing_tolerance=2.0):
        
        def _process():
            # Check if SRT support is available
            if not self.srt_available:
                raise ImportError("SRT support not available - missing required modules")
            
            # Check if F5-TTS is available
            if not self.f5tts_available:
                raise ImportError("F5-TTS support not available - missing required modules")
            
            # Load F5-TTS model
            self.load_f5tts_model(model, device)
            
            # Set seed for reproducibility
            self.set_seed(seed)
            
            # Prepare inputs for reference handling
            inputs = {
                "reference_audio_file": reference_audio_file,
                "opt_reference_audio": opt_reference_audio,
                "opt_reference_text": opt_reference_text
            }
            
            # Handle reference audio and text with priority chain
            audio_prompt, validated_ref_text = self._handle_reference_with_priority_chain(inputs)
            
            # Parse SRT content
            srt_parser = self.SRTParser()
            subtitles = srt_parser.parse_srt_content(srt_content)
            
            # Determine audio prompt component for cache key generation
            audio_prompt_component = ""
            if opt_reference_audio is not None:
                waveform_hash = hashlib.md5(opt_reference_audio["waveform"].cpu().numpy().tobytes()).hexdigest()
                audio_prompt_component = f"ref_audio_{waveform_hash}_{opt_reference_audio['sample_rate']}"
            elif reference_audio_file != "none":
                audio_prompt_component = f"ref_file_{reference_audio_file}"
            elif audio_prompt:
                audio_prompt_component = audio_prompt
            
            # Generate audio segments
            audio_segments = []
            natural_durations = []
            any_segment_cached = False
            
            for i, subtitle in enumerate(subtitles):
                # Check for interruption
                self.check_interruption(f"F5-TTS SRT generation segment {i+1}/{len(subtitles)} (Seq {subtitle.sequence})")
                
                if not subtitle.text.strip():
                    # Handle empty text by creating silence
                    natural_duration = subtitle.duration
                    wav = self.AudioTimingUtils.create_silence(
                        duration_seconds=natural_duration,
                        sample_rate=self.f5tts_sample_rate,
                        channels=1,
                        device=self.device
                    )
                    print(f"🤫 F5-TTS SRT Segment {i+1} (Seq {subtitle.sequence}): Empty text, generating {natural_duration:.2f}s silence.")
                else:
                    # Generate cache key for this segment
                    segment_cache_key = self._generate_segment_cache_key(
                        subtitle.text, model, device, audio_prompt_component, validated_ref_text,
                        temperature, speed, target_rms, cross_fade_duration, nfe_step, cfg_strength, seed
                    )
                    
                    # Try to get cached audio
                    cached_data = self._get_cached_segment_audio(segment_cache_key) if enable_audio_cache else None
                    
                    if cached_data:
                        wav, natural_duration = cached_data
                        any_segment_cached = True
                        print(f"💾 F5-TTS SRT Segment {i+1} (Seq {subtitle.sequence}): Using cached audio")
                    else:
                        # Generate new audio
                        print(f"🎤 Generating F5-TTS SRT segment {i+1}/{len(subtitles)} (Seq {subtitle.sequence})...")
                        
                        # Validate and clamp nfe_step to prevent ODE solver issues
                        safe_nfe_step = max(1, min(nfe_step, 71))
                        if safe_nfe_step != nfe_step:
                            print(f"⚠️ F5-TTS: Clamped nfe_step from {nfe_step} to {safe_nfe_step} to prevent ODE solver issues")
                        
                        wav = self.generate_f5tts_audio(
                            text=subtitle.text,
                            ref_audio_path=audio_prompt,
                            ref_text=validated_ref_text,
                            temperature=temperature,
                            speed=speed,
                            target_rms=target_rms,
                            cross_fade_duration=cross_fade_duration,
                            nfe_step=safe_nfe_step,
                            cfg_strength=cfg_strength
                        )
                        natural_duration = self.AudioTimingUtils.get_audio_duration(wav, self.f5tts_sample_rate)
                        
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
            
            # Assemble final audio based on timing mode
            if timing_mode == "stretch_to_fit":
                # Use time stretching to match exact timing
                assembler = self.TimedAudioAssembler(self.f5tts_sample_rate)
                final_audio = assembler.assemble_timed_audio(
                    audio_segments, target_timings, fade_duration=fade_for_StretchToFit
                )
            elif timing_mode == "pad_with_silence":
                # Add silence to match timing without stretching
                final_audio = self._assemble_audio_with_overlaps(audio_segments, subtitles, self.f5tts_sample_rate)
            else:  # smart_natural
                # Smart balanced timing: use natural audio but add minimal adjustments within tolerance
                final_audio, smart_adjustments = self._assemble_with_smart_timing(
                    audio_segments, subtitles, self.f5tts_sample_rate, timing_tolerance,
                    max_stretch_ratio, min_stretch_ratio
                )
                adjustments = smart_adjustments
            
            # Generate reports
            timing_report = self._generate_timing_report(subtitles, adjustments, timing_mode)
            adjusted_srt_string = self._generate_adjusted_srt_string(subtitles, adjustments, timing_mode)
            
            # Generate info with cache status and stretching method
            total_duration = self.AudioTimingUtils.get_audio_duration(final_audio, self.f5tts_sample_rate)
            cache_status = "cached" if any_segment_cached else "generated"
            stretch_info = ""
            
            # Get stretching method info
            if timing_mode == "stretch_to_fit":
                current_stretcher = assembler.time_stretcher
            elif timing_mode == "smart_natural":
                # Use the stored stretcher type for smart_natural mode
                if hasattr(self, '_smart_natural_stretcher'):
                    if self._smart_natural_stretcher == "ffmpeg":
                        stretch_info = ", Stretching method: FFmpeg"
                    else:
                        stretch_info = ", Stretching method: Phase Vocoder"
                else:
                    stretch_info = ", Stretching method: Unknown"
            
            # For stretch_to_fit mode, examine the actual stretcher
            if timing_mode == "stretch_to_fit" and 'current_stretcher' in locals():
                if isinstance(current_stretcher, self.FFmpegTimeStretcher):
                    stretch_info = ", Stretching method: FFmpeg"
                elif isinstance(current_stretcher, self.PhaseVocoderTimeStretcher):
                    stretch_info = ", Stretching method: Phase Vocoder"
                else:
                    stretch_info = f", Stretching method: {current_stretcher.__class__.__name__}"
            
            info = (f"Generated {total_duration:.1f}s F5-TTS SRT-timed audio from {len(subtitles)} subtitles "
                   f"using {timing_mode} mode ({cache_status} segments, F5-TTS {model}{stretch_info})")
            
            # Format final audio for ComfyUI
            if final_audio.dim() == 1:
                final_audio = final_audio.unsqueeze(0)  # Add channel dimension
            
            return (
                self.format_f5tts_audio_output(final_audio),
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
        """Smart timing assembly with intelligent adjustments"""
        # Initialize stretcher for smart_natural mode
        try:
            # Try FFmpeg first
            print("F5-TTS SRT Smart natural mode: Trying FFmpeg stretcher...")
            time_stretcher = self.FFmpegTimeStretcher()
            self._smart_natural_stretcher = "ffmpeg"
            print("F5-TTS SRT Smart natural mode: Using FFmpeg stretcher")
        except self.AudioTimingError as e:
            # Fall back to Phase Vocoder
            print(f"F5-TTS SRT Smart natural mode: FFmpeg initialization failed ({str(e)}), falling back to Phase Vocoder")
            time_stretcher = self.PhaseVocoderTimeStretcher()
            self._smart_natural_stretcher = "phase_vocoder"
            print("F5-TTS SRT Smart natural mode: Using Phase Vocoder stretcher")
        
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
    
    def _generate_timing_report(self, subtitles: List, adjustments: List[Dict], timing_mode: str) -> str:
        """Generate detailed timing report."""
        # Delegate to reporting module
        from chatterbox_srt.reporting import SRTReportGenerator
        reporter = SRTReportGenerator()
        return reporter.generate_timing_report(subtitles, adjustments, timing_mode)
    
    def _generate_adjusted_srt_string(self, subtitles: List, adjustments: List[Dict], timing_mode: str) -> str:
        """Generate adjusted SRT string from final timings."""
        # Delegate to reporting module
        from chatterbox_srt.reporting import SRTReportGenerator
        reporter = SRTReportGenerator()
        return reporter.generate_adjusted_srt_string(subtitles, adjustments, timing_mode)