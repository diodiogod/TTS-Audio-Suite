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

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load f5tts_base_node module directly
f5tts_base_node_path = os.path.join(nodes_dir, "base", "f5tts_base_node.py")
f5tts_base_spec = importlib.util.spec_from_file_location("f5tts_base_node_module", f5tts_base_node_path)
f5tts_base_module = importlib.util.module_from_spec(f5tts_base_spec)
sys.modules["f5tts_base_node_module"] = f5tts_base_module
f5tts_base_spec.loader.exec_module(f5tts_base_module)

# Import the base class
BaseF5TTSNode = f5tts_base_module.BaseF5TTSNode

from utils.system.import_manager import import_manager
from utils.audio.processing import AudioProcessingUtils
from utils.voice.discovery import get_available_voices, load_voice_reference, get_available_characters, get_character_mapping
from utils.text.character_parser import parse_character_text, character_parser
# Lazy imports for modular components (loaded when needed to avoid torch import issues during node registration)
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
        self.multilingual_engine = None  # Lazy loaded
    
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
        # Get available reference audio files from both models/voices/ and voices_examples/
        reference_files = get_available_voices()
        
        return {
            "required": {
                "srt_content": ("STRING", {
                    "multiline": True,
                    "default": """1
00:00:01,000 --> 00:00:04,000
Hello! This is F5-TTS SRT with character switching.

2
00:00:04,500 --> 00:00:09,500
[Alice] Hi there! I'm Alice speaking with precise timing.

3
00:00:10,000 --> 00:00:14,000
[Bob] And I'm Bob! The audio matches these exact SRT timings.""",
                    "tooltip": "The SRT subtitle content. Each entry defines a text segment and its precise start and end times."
                }),
                "reference_audio_file": (reference_files, {
                    "default": "none",
                    "tooltip": "Reference voice from models/voices/ or voices_examples/ folders (with companion .txt/.reference.txt file). Select 'none' to use direct inputs below."
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
                "model": (BaseF5TTSNode.get_available_models_for_dropdown(), {
                    "default": "F5TTS_v1_Base",
                    "tooltip": "F5-TTS model variant to use. F5TTS_Base is the standard model, F5TTS_v1_Base is improved version, E2TTS_Base is enhanced variant."
                }),
                "seed": ("INT", {
                    "default": 1, "min": 0, "max": 2**32 - 1,
                    "tooltip": "Seed for reproducible F5-TTS generation. Same seed with same inputs will produce identical results. Set to 0 for random generation."
                }),
                "timing_mode": (["stretch_to_fit", "pad_with_silence", "smart_natural", "concatenate"], {
                    "default": "smart_natural",
                    "tooltip": "Determines how audio segments are aligned with SRT timings:\n🔹 stretch_to_fit: Stretches/compresses audio to exactly match SRT segment durations.\n🔹 pad_with_silence: Places natural audio at SRT start times, padding gaps with silence. May result in overlaps.\n🔹 smart_natural: Intelligently adjusts timings within 'timing_tolerance', prioritizing natural audio and shifting subsequent segments. Applies stretch/shrink within limits if needed.\n🔹 concatenate: Ignores original SRT timings, concatenates audio naturally and generates new SRT with actual timings."
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
                    "tooltip": "Speech generation control. Lower values (1.0-1.5) = more natural, conversational delivery. Higher values (3.0-5.0) = crisper, more articulated speech with stronger emphasis. Default 2.0 balances naturalness and clarity."
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
    CATEGORY = "TTS Audio Suite/👄 F5-TTS"

    @staticmethod
    def _get_companion_txt_file(audio_file_path):
        """Get the path to companion .txt file for an audio file (legacy method)"""
        from pathlib import Path
        p = Path(audio_file_path)
        return os.path.join(os.path.dirname(audio_file_path), p.stem + ".txt")
    
    def _load_reference_from_file(self, reference_audio_file):
        """Load reference audio and text from voice discovery system"""
        if reference_audio_file == "none":
            return None, None
        
        # Use the new voice discovery system
        audio_path, ref_text = load_voice_reference(reference_audio_file)
        
        if not audio_path or not ref_text:
            raise FileNotFoundError(f"Reference voice '{reference_audio_file}' not found or has no companion text file")
        
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {audio_path}")
        
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
                                   nfe_step: int, cfg_strength: float, seed: int, auto_phonemization: bool = True) -> str:
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
            'auto_phonemization': auto_phonemization,
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
                           timing_tolerance=2.0, auto_phonemization=True):
        
        def _process():
            # Check if SRT support is available
            if not self.srt_available:
                raise ImportError("SRT support not available - missing required modules")
            
            # Check if F5-TTS is available
            if not self.f5tts_available:
                raise ImportError("F5-TTS support not available - missing required modules")
            
            # Set seed for reproducibility (do this before model loading)
            self.set_seed(seed)
            
            # Prepare inputs for reference handling
            inputs = {
                "reference_audio_file": reference_audio_file,
                "opt_reference_audio": opt_reference_audio,
                "opt_reference_text": opt_reference_text
            }
            
            # Handle reference audio and text with priority chain
            audio_prompt, validated_ref_text = self._handle_reference_with_priority_chain(inputs)
            
            # Parse SRT content with overlap support
            srt_parser = self.SRTParser()
            subtitles = srt_parser.parse_srt_content(srt_content, allow_overlaps=True)
            
            # Check for overlaps and handle smart_natural mode fallback using modular utility
            from utils.timing.overlap_detection import SRTOverlapHandler
            has_overlaps = SRTOverlapHandler.detect_overlaps(subtitles)
            current_timing_mode, mode_switched = SRTOverlapHandler.handle_smart_natural_fallback(
                timing_mode, has_overlaps, "F5-TTS SRT"
            )
            
            # Determine audio prompt component for cache key generation (stable identifier)
            audio_prompt_component = ""
            if opt_reference_audio is not None:
                waveform_hash = hashlib.md5(opt_reference_audio["waveform"].cpu().numpy().tobytes()).hexdigest()
                audio_prompt_component = f"ref_audio_{waveform_hash}_{opt_reference_audio['sample_rate']}"
            elif reference_audio_file != "none":
                audio_prompt_component = f"ref_file_{reference_audio_file}"
            elif audio_prompt:
                # Handle potential temporary file paths by creating stable identifier
                if audio_prompt.startswith('/tmp/') or 'tmp' in str(audio_prompt):
                    try:
                        if os.path.exists(audio_prompt):
                            with open(audio_prompt, 'rb') as f:
                                content = f.read()
                                content_hash = hashlib.md5(content).hexdigest()
                                audio_prompt_component = f"temp_audio_{content_hash}"
                        else:
                            audio_prompt_component = str(audio_prompt)
                    except Exception:
                        audio_prompt_component = str(audio_prompt)
                else:
                    audio_prompt_component = audio_prompt
            
            # Set up character parser with available characters BEFORE processing subtitles
            available_chars = get_available_characters()
            character_parser.set_available_characters(list(available_chars))
            
            # Set engine-aware default language to prevent unnecessary model switching
            character_parser.set_engine_aware_default_language(model, "f5tts")
            
            # SMART OPTIMIZATION: Group individual character segments by language to minimize model switching
            segment_language_groups = {}
            all_character_segments = []
            
            # First pass: extract ALL character segments from all subtitles and group by language
            for i, subtitle in enumerate(subtitles):
                if not subtitle.text.strip():
                    # Empty subtitle - will be handled separately
                    all_character_segments.append((i, None, subtitle, 'empty', None, None, None, None))
                    continue

                # Parse character segments with language and parameter awareness
                character_segment_objects = character_parser.parse_text_segments(subtitle.text)
                character_segments_with_lang = [(seg.character, seg.text, seg.language, seg.parameters if seg.parameters else {}) for seg in character_segment_objects]

                # Process each individual character segment separately
                for seg_idx, (character, text, language, segment_params) in enumerate(character_segments_with_lang):
                    segment_info = (i, seg_idx, subtitle, 'character_segment', character, text, language, segment_params)
                    all_character_segments.append(segment_info)

                    # Group this individual segment by its language
                    if language not in segment_language_groups:
                        segment_language_groups[language] = []
                    segment_language_groups[language].append(segment_info)
            
            # Generate audio segments using smart segment-level language grouping
            print(f"🚀 SRT: Processing languages: {sorted(segment_language_groups.keys())}")
            audio_segments = [None] * len(subtitles)  # Pre-allocate in correct order  
            natural_durations = [0.0] * len(subtitles)
            
            # Track generated segment audio by (subtitle_idx, segment_idx)
            generated_segment_audio = {}
            any_segment_cached = False
            
            # Process each language group (now containing individual character segments)
            for lang_code in sorted(segment_language_groups.keys()):
                lang_segments = segment_language_groups[lang_code]
                
                # Skip empty segments
                if lang_code == 'empty':
                    for subtitle_idx, seg_idx, subtitle, segment_type, character, text, language, segment_params in lang_segments:
                        # Handle empty text by creating silence
                        natural_duration = subtitle.duration
                        wav = self.AudioTimingUtils.create_silence(
                            duration_seconds=natural_duration,
                            sample_rate=self.f5tts_sample_rate,
                            channels=1,
                            device=self.device
                        )
                        print(f"🤫 F5-TTS Segment {subtitle_idx+1} (Seq {subtitle.sequence}): Empty text, generating {natural_duration:.2f}s silence.")
                        generated_segment_audio[(subtitle_idx, seg_idx)] = (wav, natural_duration)
                    continue
                
                print(f"📋 Processing {len(lang_segments)} F5-TTS character segment(s) in '{lang_code}' language group...")
                
                # Check if all segments in this language group are cached before loading model
                if enable_audio_cache:
                    all_segments_cached = True
                    cached_segments_info = []
                    
                    # Get required model for this language group for consistent cache key generation
                    from utils.models.language_mapper import get_model_for_language
                    required_model = get_model_for_language("f5tts", lang_code, model)
                    
                    for subtitle_idx, seg_idx, subtitle, segment_type, character, text, language, segment_params in lang_segments:
                        # PRIORITY FIX: For "narrator" character, always use main reference when provided
                        # This ensures Character Voices node and dropdown selection work correctly
                        if character == "narrator" and audio_prompt and validated_ref_text:
                            # Use main reference for narrator (from Character Voices or dropdown)
                            char_audio = audio_prompt
                            char_text = validated_ref_text
                        else:
                            # Get character-specific voice for named characters
                            from utils.voice.discovery import get_character_mapping
                            character_mapping = get_character_mapping([character], engine_type="f5tts")
                            char_audio, char_text = character_mapping.get(character, (None, None))
                            if not char_audio or not char_text:
                                char_audio = audio_prompt
                                char_text = validated_ref_text

                        # Validate and clamp nfe_step
                        safe_nfe_step = max(1, min(nfe_step, 71))

                        # Apply per-segment parameters for cache key generation
                        current_config = {
                            'temperature': temperature,
                            'speed': speed,
                            'target_rms': target_rms,
                            'cross_fade_duration': cross_fade_duration,
                            'cfg_strength': cfg_strength,
                            'seed': seed
                        }
                        if segment_params:
                            from utils.text.segment_parameters import apply_segment_parameters
                            current_config = apply_segment_parameters(current_config, segment_params, "f5tts")

                        # Generate cache key using the required model for this language
                        # This ensures consistent cache keys regardless of current model state
                        cache_key = self._generate_segment_cache_key(
                            f"{character}:{text}", required_model, self.device, audio_prompt_component,
                            char_text, current_config['temperature'], current_config['speed'],
                            current_config['target_rms'], current_config['cross_fade_duration'],
                            safe_nfe_step, current_config['cfg_strength'], current_config['seed'], auto_phonemization
                        )
                        cached_data = self._get_cached_segment_audio(cache_key)

                        if cached_data:
                            cached_segments_info.append((subtitle_idx, seg_idx, cached_data, character, text, language))
                        else:
                            all_segments_cached = False
                            break
                    
                    # If all segments are cached, skip model loading and use cached data
                    if all_segments_cached:
                        print(f"💾 All {len(lang_segments)} segment(s) in '{lang_code}' language group are cached - skipping model load")
                        for subtitle_idx, seg_idx, cached_data, character, text, language in cached_segments_info:
                            wav, natural_duration = cached_data
                            print(f"💾 Using cached audio for {character} ({language}) segment {subtitle_idx+1}.{seg_idx+1}: '{text[:30]}...'")
                            generated_segment_audio[(subtitle_idx, seg_idx)] = (wav, natural_duration)
                            any_segment_cached = True
                        continue
                
                # Load model for this language group (only if we have non-cached segments)
                # Smart model selection:
                # - If the language is from a character's language mapping (not 'en'), use language-appropriate model
                # - Otherwise, use the user's explicitly chosen model
                from utils.models.language_mapper import get_model_for_language
                if lang_code != 'en':
                    # This is likely a character with language mapping (e.g., Alice -> German)
                    # Use the appropriate language model
                    required_model = get_model_for_language("f5tts", lang_code, model)
                else:
                    # Default/unmapped character - use user's chosen model
                    required_model = model
                
                current_model = getattr(self, 'current_model_name', None)
                if current_model != required_model:
                    print(f"🎯 SRT: Loading {required_model} model for {len(lang_segments)} segment(s)")
                    self.load_f5tts_model(required_model, device)
                else:
                    print(f"✅ SRT: Using {required_model} model for {len(lang_segments)} segment(s) (already loaded)")
                
                # Process each character segment in this language group
                for subtitle_idx, seg_idx, subtitle, segment_type, character, text, language, segment_params in lang_segments:
                    # Check for interruption
                    self.check_interruption(f"F5-TTS character segment {subtitle_idx+1}.{seg_idx+1} (Seq {subtitle.sequence})")
                    
                    # Show generation message
                    if character == "narrator":
                        if language != 'en':
                            print(f"🎤 Generating F5-TTS segment {subtitle_idx+1}.{seg_idx+1} (Seq {subtitle.sequence}) in {language}...")
                        else:
                            print(f"🎤 Generating F5-TTS segment {subtitle_idx+1}.{seg_idx+1} (Seq {subtitle.sequence})...")
                    else:
                        if language != 'en':
                            print(f"🎭 Generating F5-TTS segment {subtitle_idx+1}.{seg_idx+1} using '{character}' in {language}")
                        else:
                            print(f"🎭 Generating F5-TTS segment {subtitle_idx+1}.{seg_idx+1} using '{character}'")
                    
                    # PRIORITY FIX: For "narrator" character, always use main reference when provided
                    # This ensures Character Voices node and dropdown selection work correctly
                    if character == "narrator" and audio_prompt and validated_ref_text:
                        # Use main reference for narrator (from Character Voices or dropdown)
                        char_audio = audio_prompt
                        char_text = validated_ref_text
                        print(f"🎯 F5-TTS: Using main reference for narrator (from Character Voices/dropdown)")
                    else:
                        # Get character-specific voice for named characters
                        from utils.voice.discovery import get_character_mapping
                        character_mapping = get_character_mapping([character], engine_type="f5tts")
                        char_audio, char_text = character_mapping.get(character, (None, None))
                        
                        if not char_audio or not char_text:
                            # Fallback to main reference if character-specific voice not found
                            char_audio = audio_prompt
                            char_text = validated_ref_text
                            print(f"🔄 F5-TTS: Using main reference fallback for '{character}'")
                        else:
                            print(f"🎭 F5-TTS: Using character-specific voice for '{character}'")
                    
                    # Validate and clamp nfe_step to prevent ODE solver issues
                    safe_nfe_step = max(1, min(nfe_step, 71))
                    if safe_nfe_step != nfe_step:
                        print(f"⚠️ F5-TTS: Clamped nfe_step from {nfe_step} to {safe_nfe_step} to prevent ODE solver issues")

                    # Apply per-segment parameters
                    current_config = {
                        'temperature': temperature,
                        'speed': speed,
                        'target_rms': target_rms,
                        'cross_fade_duration': cross_fade_duration,
                        'cfg_strength': cfg_strength,
                        'seed': seed
                    }
                    if segment_params:
                        from utils.text.segment_parameters import apply_segment_parameters
                        current_config = apply_segment_parameters(current_config, segment_params, "f5tts")
                        print(f"📊 Segment {subtitle_idx+1}.{seg_idx+1}: Character '{character}' with parameters {segment_params}")

                    # Check cache first before generating (using consistent required model name)
                    if enable_audio_cache:
                        cache_key = self._generate_segment_cache_key(
                            f"{character}:{text}", required_model, self.device, audio_prompt_component,
                            char_text, current_config['temperature'], current_config['speed'],
                            current_config['target_rms'], current_config['cross_fade_duration'],
                            safe_nfe_step, current_config['cfg_strength'], current_config['seed']
                        )
                        cached_data = self._get_cached_segment_audio(cache_key)
                        if cached_data:
                            wav = cached_data[0]
                            natural_duration = cached_data[1]
                            print(f"💾 Using cached audio for {character} ({language}) segment {subtitle_idx+1}.{seg_idx+1}: '{text[:30]}...'")
                            generated_segment_audio[(subtitle_idx, seg_idx)] = (wav, natural_duration)
                            any_segment_cached = True
                            continue

                    # Generate audio for this character segment
                    wav = self.generate_f5tts_with_pause_tags(
                        text=text,
                        ref_audio_path=char_audio,
                        ref_text=char_text,
                        enable_pause_tags=True,
                        character=character,
                        temperature=current_config['temperature'],
                        speed=current_config['speed'],
                        target_rms=current_config['target_rms'],
                        cross_fade_duration=current_config['cross_fade_duration'],
                        nfe_step=safe_nfe_step,
                        cfg_strength=current_config['cfg_strength'],
                        seed=current_config['seed'],
                        enable_cache=enable_audio_cache,
                        cache_fn=None,  # Use standard caching
                        auto_phonemization=auto_phonemization
                    )

                    # Calculate duration and store
                    natural_duration = self.AudioTimingUtils.get_audio_duration(wav, self.f5tts_sample_rate)
                    generated_segment_audio[(subtitle_idx, seg_idx)] = (wav, natural_duration)

                    # Cache the generated segment for future use (using consistent required model name)
                    if enable_audio_cache:
                        cache_key = self._generate_segment_cache_key(
                            f"{character}:{text}", required_model, self.device, audio_prompt_component,
                            char_text, current_config['temperature'], current_config['speed'],
                            current_config['target_rms'], current_config['cross_fade_duration'],
                            safe_nfe_step, current_config['cfg_strength'], current_config['seed']
                        )
                        self._cache_segment_audio(cache_key, wav, natural_duration)
            
            # Reassemble character segments back into complete subtitles
            print(f"🔗 Assembling {len(generated_segment_audio)} character segments back into {len(subtitles)} subtitles...")
            for subtitle_idx in range(len(subtitles)):
                subtitle_segments = []
                
                # Collect all segments for this subtitle in order
                seg_idx = 0
                while (subtitle_idx, seg_idx) in generated_segment_audio:
                    wav, duration = generated_segment_audio[(subtitle_idx, seg_idx)]
                    subtitle_segments.append(wav)
                    seg_idx += 1
                
                if subtitle_segments:
                    # Concatenate all character segments for this subtitle
                    if len(subtitle_segments) == 1:
                        audio_segments[subtitle_idx] = subtitle_segments[0]
                    else:
                        # Concatenate multiple character segments with crossfade using TimedAudioAssembler
                        assembler = self.TimedAudioAssembler(self.f5tts_sample_rate)
                        # Create sequential timings for the segments
                        segment_timings = []
                        current_time = 0.0
                        for seg_audio in subtitle_segments:
                            seg_duration = self.AudioTimingUtils.get_audio_duration(seg_audio, self.f5tts_sample_rate)
                            segment_timings.append((current_time, current_time + seg_duration))
                            current_time += seg_duration
                        
                        audio_segments[subtitle_idx] = assembler.assemble_timed_audio(
                            subtitle_segments, segment_timings, fade_duration=cross_fade_duration
                        )
                    
                    # Calculate total duration for this subtitle
                    natural_durations[subtitle_idx] = self.AudioTimingUtils.get_audio_duration(
                        audio_segments[subtitle_idx], self.f5tts_sample_rate
                    )
            
            # Calculate timing adjustments
            target_timings = [(sub.start_time, sub.end_time) for sub in subtitles]
            adjustments = self.calculate_timing_adjustments(natural_durations, target_timings)
            
            # Add sequence numbers to adjustments
            for i, (adj, subtitle) in enumerate(zip(adjustments, subtitles)):
                adj['sequence'] = subtitle.sequence
            
            # Assemble final audio based on timing mode
            if current_timing_mode == "stretch_to_fit":
                # Use time stretching to match exact timing
                assembler = self.TimedAudioAssembler(self.f5tts_sample_rate)
                final_audio = assembler.assemble_timed_audio(
                    audio_segments, target_timings, fade_duration=fade_for_StretchToFit
                )
            elif current_timing_mode == "pad_with_silence":
                # Add silence to match timing without stretching
                final_audio = self._assemble_audio_with_overlaps(audio_segments, subtitles, self.f5tts_sample_rate)
            elif current_timing_mode == "concatenate":
                # Concatenate audio naturally and recalculate SRT timings using modular approach
                from utils.timing.engine import TimingEngine
                from utils.timing.assembly import AudioAssemblyEngine
                
                timing_engine = TimingEngine(self.f5tts_sample_rate)
                assembler = AudioAssemblyEngine(self.f5tts_sample_rate)
                
                # Calculate new timings for concatenation
                adjustments = timing_engine.calculate_concatenation_adjustments(audio_segments, subtitles)
                
                # Assemble audio with optional crossfading
                final_audio = assembler.assemble_concatenation(audio_segments, fade_for_StretchToFit)
            else:  # smart_natural
                # Smart balanced timing: use natural audio but add minimal adjustments within tolerance
                from utils.timing.engine import TimingEngine
                from utils.timing.assembly import AudioAssemblyEngine
                
                timing_engine = TimingEngine(self.f5tts_sample_rate)
                assembler = AudioAssemblyEngine(self.f5tts_sample_rate)
                
                # Calculate smart natural adjustments within tolerance
                adjustments, processed_segments = timing_engine.calculate_smart_timing_adjustments(
                    audio_segments, subtitles, timing_tolerance, max_stretch_ratio, min_stretch_ratio, self.device
                )
                
                # Assemble with smart natural timing
                final_audio = assembler.assemble_smart_natural(
                    audio_segments, processed_segments, adjustments, subtitles, self.device
                )
            
            # Log timing information if mode was switched due to overlaps
            if mode_switched:
                print(f"⚠️ F5-TTS SRT: Mode switched from smart_natural to {current_timing_mode} due to overlapping subtitles")
            
            # Generate reports
            timing_report = self._generate_timing_report(subtitles, adjustments, current_timing_mode, has_overlaps, mode_switched, timing_mode if mode_switched else None)
            adjusted_srt_string = self._generate_adjusted_srt_string(subtitles, adjustments, current_timing_mode)
            
            # Generate info with cache status
            total_duration = self.AudioTimingUtils.get_audio_duration(final_audio, self.f5tts_sample_rate)
            cache_status = "cached" if any_segment_cached else "generated"
            
            mode_info = f"{current_timing_mode}"
            if mode_switched:
                mode_info = f"{current_timing_mode} (switched from {timing_mode} due to overlaps)"
            
            info = (f"Generated {total_duration:.1f}s F5-TTS SRT-timed audio from {len(subtitles)} subtitles "
                   f"using {mode_info} mode ({cache_status} segments, F5-TTS {model})")
            
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
        from utils.timing.assembly import AudioAssemblyEngine
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
        from utils.timing.engine import TimingEngine
        from utils.timing.assembly import AudioAssemblyEngine
        
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
        from utils.timing.reporting import SRTReportGenerator
        reporter = SRTReportGenerator()
        return reporter.generate_timing_report(subtitles, adjustments, timing_mode, has_original_overlaps, mode_switched, original_mode)
    
    def _generate_adjusted_srt_string(self, subtitles: List, adjustments: List[Dict], timing_mode: str) -> str:
        """Generate adjusted SRT string from final timings."""
        # Delegate to reporting module
        from utils.timing.reporting import SRTReportGenerator
        reporter = SRTReportGenerator()
        return reporter.generate_adjusted_srt_string(subtitles, adjustments, timing_mode)
    
    def _process_single_segment_for_streaming(self, original_idx, character, segment_text, 
                                               language, voice_path, inputs):
        """
        Process a single SRT segment for the streaming processor.
        
        This method enables F5-TTS SRT to work with the universal streaming system,
        preserving all existing SRT-specific functionality including timing,
        reference audio handling, and caching.
        
        Args:
            original_idx: Original segment index for result ordering
            character: Character name for this segment
            segment_text: Text to generate
            language: Language code for this segment
            voice_path: Path to voice reference file
            inputs: Processing parameters dict including SRT-specific params
            
        Returns:
            Generated audio tensor
        """
        try:
            # Extract F5-TTS parameters from inputs
            model = inputs.get('model', 'F5TTS_Base')
            device = inputs.get('device', 'auto')
            temperature = inputs.get('temperature', 0.8)
            speed = inputs.get('speed', 1.0)
            target_rms = inputs.get('target_rms', 0.1)
            cross_fade_duration = inputs.get('cross_fade_duration', 0.15)
            nfe_step = inputs.get('nfe_step', 32)
            cfg_strength = inputs.get('cfg_strength', 2.0)
            seed = inputs.get('seed', 1)
            enable_cache = inputs.get('enable_audio_cache', True)
            
            # Get the F5-TTS model for this language using language mapper
            from utils.models.language_mapper import get_model_for_language
            required_model = get_model_for_language('f5tts', language, model)
            
            # Ensure correct model is loaded (streaming coordinator should pre-load)
            current_model = getattr(self, 'current_model_name', None)
            if current_model != required_model:
                self.load_f5tts_model(required_model, device)
            
            # Handle reference audio and text for this character
            ref_audio_path = None
            ref_text = None
            
            if voice_path and voice_path != 'none' and os.path.exists(voice_path):
                # Load reference from character voice file
                ref_audio_path = voice_path
                
                # Check for companion text file
                text_file = voice_path.replace('.wav', '.txt').replace('.mp3', '.txt')
                if os.path.exists(text_file):
                    with open(text_file, 'r', encoding='utf-8') as f:
                        ref_text = f.read().strip()
                
                if not ref_text:
                    # Try .reference.txt pattern
                    ref_text_file = voice_path.replace('.wav', '.reference.txt').replace('.mp3', '.reference.txt')
                    if os.path.exists(ref_text_file):
                        with open(ref_text_file, 'r', encoding='utf-8') as f:
                            ref_text = f.read().strip()
                
                if not ref_text:
                    ref_text = "This is a reference voice for synthesis."
            else:
                # Use main reference from inputs
                ref_audio_path = inputs.get('audio_prompt_path', '')
                ref_text = inputs.get('ref_text', "This is a reference voice for synthesis.")
            
            # Generate stable audio component for caching
            if character != "narrator":
                stable_audio_component = f"srt_streaming_char_{character}_{language}_{original_idx}"
            else:
                stable_audio_component = f"srt_streaming_{language}_{original_idx}"
            
            # Validate and clamp nfe_step (F5-TTS specific range)
            safe_nfe_step = max(1, min(nfe_step, 71))
            
            # Check cache if enabled
            if enable_cache:
                cache_key = self._generate_segment_cache_key(
                    f"{character}:{segment_text}", required_model, device,
                    stable_audio_component, ref_text, temperature, speed,
                    target_rms, cross_fade_duration, safe_nfe_step, cfg_strength, seed,
                    auto_phonemization
                )
                cached_audio = self._get_cached_segment_audio(cache_key)
                if cached_audio:
                    # Set phonemization setting for consistency (even when using cache)
                    import os
                    os.environ['F5TTS_AUTO_PHONEMIZATION'] = str(auto_phonemization).lower()
                    print(f"💾 F5-TTS SRT streaming: Cache hit for {character} segment {original_idx}")
                    return cached_audio[0]
            
            # Generate audio for this segment
            print(f"🔄 F5-TTS SRT streaming: Generating {character} segment {original_idx} in {language}")
            
            # Generate with F5-TTS (using base class method)
            wav = self.generate_f5tts_with_pause_tags(
                segment_text, ref_audio_path, ref_text, temperature, speed,
                target_rms, cross_fade_duration, safe_nfe_step, cfg_strength, seed,
                auto_phonemization=auto_phonemization
            )
            
            # Cache the result if enabled and generation succeeded
            if enable_cache and wav is not None:
                natural_duration = wav.shape[-1] / self.f5tts_sample_rate
                GLOBAL_AUDIO_CACHE[cache_key] = (wav, natural_duration)
            
            return wav
            
        except Exception as e:
            print(f"❌ F5-TTS SRT streaming segment {original_idx} failed: {e}")
            # Return silence as fallback (1 second at F5-TTS sample rate)
            return torch.zeros((1, self.f5tts_sample_rate))

