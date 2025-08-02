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

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load base_node module directly
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseTTSNode = base_module.BaseTTSNode

from utils.system.import_manager import import_manager
from utils.audio.processing import AudioProcessingUtils
from utils.voice.discovery import get_available_voices, load_voice_reference, get_available_characters, get_character_mapping
from utils.text.character_parser import parse_character_text, character_parser
from utils.text.pause_processor import PauseTagProcessor
# Lazy imports for modular components (loaded when needed to avoid torch import issues during node registration)
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
    def INPUT_TYPES(cls):
        # Import language models for dropdown
        try:
            from engines.chatterbox.language_models import get_available_languages
            available_languages = get_available_languages()
        except ImportError:
            available_languages = ["English"]
        
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
                "language": (available_languages, {
                    "default": "English",
                    "tooltip": "Language model to use for text-to-speech generation. Local models are preferred over remote downloads."
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
                "timing_mode": (["stretch_to_fit", "pad_with_silence", "smart_natural", "concatenate"], {
                    "default": "smart_natural",
                    "tooltip": "Determines how audio segments are aligned with SRT timings:\n🔹 stretch_to_fit: Stretches/compresses audio to exactly match SRT segment durations.\n🔹 pad_with_silence: Places natural audio at SRT start times, padding gaps with silence. May result in overlaps.\n🔹 smart_natural: Intelligently adjusts timings within 'timing_tolerance', prioritizing natural audio and shifting subsequent segments. Applies stretch/shrink within limits if needed.\n🔹 concatenate: Ignores original SRT timings, concatenates audio naturally and generates new SRT with actual timings."
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

    def _generate_tts_with_pause_tags(self, text: str, audio_prompt, exaggeration: float, 
                                    temperature: float, cfg_weight: float, language: str = "English",
                                    enable_pause_tags: bool = True, character: str = "narrator", 
                                    seed: int = 0, enable_cache: bool = True,
                                    crash_protection_template: str = "hmm ,, {seg} hmm ,,", 
                                    stable_audio_component: str = None) -> torch.Tensor:
        """
        Generate ChatterBox TTS audio with pause tag support for SRT node.
        
        Args:
            text: Input text potentially with pause tags
            audio_prompt: Audio prompt for TTS generation
            exaggeration: ChatterBox exaggeration parameter
            temperature: ChatterBox temperature parameter
            cfg_weight: ChatterBox CFG weight parameter
            enable_pause_tags: Whether to process pause tags
            character: Character name for cache key
            seed: Seed for reproducibility and cache key
            enable_cache: Whether to use caching
            
        Returns:
            Generated audio tensor with pauses
        """
        # Preprocess text for pause tags
        processed_text, pause_segments = PauseTagProcessor.preprocess_text_with_pause_tags(
            text, enable_pause_tags
        )
        
        if pause_segments is None:
            # No pause tags, use regular generation with caching
            if enable_cache:
                # Use stable audio component if provided, otherwise fallback to temp path
                audio_component = stable_audio_component if stable_audio_component else (getattr(audio_prompt, 'name', str(audio_prompt)) if audio_prompt else "")
                cache_key = self._generate_segment_cache_key(
                    f"{character}:{processed_text}", exaggeration, temperature, cfg_weight, seed,
                    audio_component, self.model_manager.get_model_source("tts"), self.device, language
                )
                
                # Try cache first
                cached_data = self._get_cached_segment_audio(cache_key)
                if cached_data:
                    return cached_data[0]
                
                # Generate and cache
                audio = self._safe_generate_tts_audio(processed_text, audio_prompt, exaggeration, temperature, cfg_weight)
                duration = self.AudioTimingUtils.get_audio_duration(audio, self.tts_model.sr)
                self._cache_segment_audio(cache_key, audio, duration)
                return audio
            else:
                return self._safe_generate_tts_audio(processed_text, audio_prompt, exaggeration, temperature, cfg_weight)
        
        # Generate audio with pause tags, caching individual text segments
        def tts_generate_func(text_content: str) -> torch.Tensor:
            """TTS generation function for pause tag processor with caching"""
            if enable_cache:
                # Use stable audio component if provided, otherwise fallback to temp path
                audio_component = stable_audio_component if stable_audio_component else (getattr(audio_prompt, 'name', str(audio_prompt)) if audio_prompt else "")
                
                # Apply crash protection to individual text segment FIRST
                protected_text = self._pad_short_text_for_chatterbox(text_content, crash_protection_template)
                if len(text_content.strip()) < 21:
                    print(f"🔍 DEBUG: Pause segment original: '{text_content}' → Protected: '{protected_text}' (len: {len(protected_text)})")
                
                # Use protected text for BOTH lookup and caching to ensure consistency
                cache_key = self._generate_segment_cache_key(
                    f"{character}:{protected_text}", exaggeration, temperature, cfg_weight, seed,
                    audio_component, self.model_manager.get_model_source("tts"), self.device, language
                )
                
                # Try cache first  
                cached_data = self._get_cached_segment_audio(cache_key)
                if cached_data:
                    print(f"💾 CACHE HIT for text: '{text_content[:30]}...'")
                    return cached_data[0]
                
                # Generate and cache
                audio = self._safe_generate_tts_audio(protected_text, audio_prompt, exaggeration, temperature, cfg_weight)
                duration = self.AudioTimingUtils.get_audio_duration(audio, self.tts_model.sr)
                self._cache_segment_audio(cache_key, audio, duration)
                return audio
            else:
                # Apply crash protection to individual text segment
                protected_text = self._pad_short_text_for_chatterbox(text_content, crash_protection_template)
                if len(text_content.strip()) < 21:
                    print(f"🔍 DEBUG: Pause segment original: '{text_content}' → Protected: '{protected_text}' (len: {len(protected_text)})")
                
                return self._safe_generate_tts_audio(protected_text, audio_prompt, exaggeration, temperature, cfg_weight)
        
        return PauseTagProcessor.generate_audio_with_pauses(
            pause_segments, tts_generate_func, self.tts_model.sr if hasattr(self, 'tts_model') and self.tts_model else 22050
        )

    def _generate_segment_cache_key(self, subtitle_text: str, exaggeration: float, temperature: float, 
                                   cfg_weight: float, seed: int, audio_prompt_component: str, 
                                   model_source: str, device: str, language: str = "English") -> str:
        """Generate cache key for a single audio segment based on generation parameters."""
        cache_data = {
            'text': subtitle_text,
            'exaggeration': exaggeration,
            'temperature': temperature,
            'cfg_weight': cfg_weight,
            'seed': seed,
            'audio_prompt_component': audio_prompt_component,
            'model_source': model_source,
            'device': device,
            'language': language,
            'engine': 'chatterbox_srt'
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

    def generate_srt_speech(self, srt_content, language, device, exaggeration, temperature, cfg_weight, seed,
                            timing_mode, reference_audio=None, audio_prompt_path="",
                            enable_audio_cache=True, fade_for_StretchToFit=0.01, 
                            max_stretch_ratio=2.0, min_stretch_ratio=0.5, timing_tolerance=2.0,
                            crash_protection_template="hmm ,, {seg} hmm ,,"):
        
        def _process():
            # Check if SRT support is available
            if not self.srt_available:
                raise ImportError("SRT support not available - missing required modules")
            
            # Set seed for reproducibility (do this before model loading)
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
            
            # Set up character parser with available characters BEFORE processing subtitles
            available_chars = get_available_characters()
            character_parser.set_available_characters(list(available_chars))
            
            # SMART OPTIMIZATION: Group subtitles by language to minimize model switching
            subtitle_language_groups = {}
            all_subtitle_segments = []
            
            # First pass: analyze all subtitles and group by language
            for i, subtitle in enumerate(subtitles):
                if not subtitle.text.strip():
                    # Empty subtitle - will be handled separately
                    all_subtitle_segments.append((i, subtitle, 'empty', None, None))
                    continue
                
                # Parse character segments with language awareness
                character_segments_with_lang = character_parser.split_by_character_with_language(subtitle.text)
                
                # Check if we have character switching or language switching
                characters = list(set(char for char, _, _ in character_segments_with_lang))
                languages = list(set(lang for _, _, lang in character_segments_with_lang))
                has_multiple_characters_in_subtitle = len(characters) > 1 or (len(characters) == 1 and characters[0] != "narrator")
                has_multiple_languages_in_subtitle = len(languages) > 1
                
                if has_multiple_characters_in_subtitle or has_multiple_languages_in_subtitle:
                    # Complex subtitle - group by dominant language or mark as multilingual
                    primary_lang = languages[0] if languages else 'en'
                    subtitle_type = 'multilingual' if has_multiple_languages_in_subtitle else 'multicharacter'
                    all_subtitle_segments.append((i, subtitle, subtitle_type, primary_lang, character_segments_with_lang))
                    
                    # Add to language groups for smart processing
                    if primary_lang not in subtitle_language_groups:
                        subtitle_language_groups[primary_lang] = []
                    subtitle_language_groups[primary_lang].append((i, subtitle, subtitle_type, character_segments_with_lang))
                else:
                    # Simple subtitle - group by language
                    single_char, single_text, single_lang = character_segments_with_lang[0]
                    all_subtitle_segments.append((i, subtitle, 'simple', single_lang, character_segments_with_lang))
                    
                    if single_lang not in subtitle_language_groups:
                        subtitle_language_groups[single_lang] = []
                    subtitle_language_groups[single_lang].append((i, subtitle, 'simple', character_segments_with_lang))
            
            # SMART INITIALIZATION: Load the first language model we'll actually need
            first_language_code = sorted(subtitle_language_groups.keys())[0] if subtitle_language_groups else 'en'
            from utils.models.language_mapper import get_model_for_language
            required_language = get_model_for_language("chatterbox", first_language_code, language)
            print(f"🚀 SRT: Smart initialization - loading {required_language} model for first language '{first_language_code}'")
            self.load_tts_model(device, required_language)
            self.current_language = required_language
            
            # Generate audio segments using smart language grouping
            audio_segments = [None] * len(subtitles)  # Pre-allocate in correct order
            natural_durations = [0.0] * len(subtitles)
            any_segment_cached = False
            
            # Process each language group
            for lang_code in sorted(subtitle_language_groups.keys()):
                lang_subtitles = subtitle_language_groups[lang_code]
                
                print(f"📋 Processing {len(lang_subtitles)} SRT subtitle(s) in '{lang_code}' language group...")
                
                # Check if we need to switch models for this language group
                required_language = get_model_for_language("chatterbox", lang_code, language)
                if self.current_language != required_language:
                    print(f"🎯 SRT: Switching to {required_language} model for {len(lang_subtitles)} subtitle(s) in '{lang_code}'")
                    self.load_tts_model(device, required_language)
                    self.current_language = required_language
                else:
                    print(f"✅ SRT: Using {required_language} model for {len(lang_subtitles)} subtitle(s) in '{lang_code}' (already loaded)")
                
                # Process each subtitle in this language group
                for i, subtitle, subtitle_type, character_segments_with_lang in lang_subtitles:
                    print(f"📺 Generating SRT segment {i+1}/{len(subtitles)} (Seq {subtitle.sequence}) in {lang_code}...")
                    
                    # Check for interruption
                    self.check_interruption(f"SRT generation segment {i+1}/{len(subtitles)} (Seq {subtitle.sequence})")
                    
                    if subtitle_type == 'multilingual' or subtitle_type == 'multicharacter':
                        # Use modular multilingual engine for character/language switching
                        characters = list(set(char for char, _, _ in character_segments_with_lang))
                        languages = list(set(lang for _, _, lang in character_segments_with_lang))
                        
                        if len(languages) > 1:
                            print(f"🌍 ChatterBox SRT Segment {i+1} (Seq {subtitle.sequence}): Language switching - {', '.join(languages)}")
                        if len(characters) > 1 or (len(characters) == 1 and characters[0] != "narrator"):
                            print(f"🎭 ChatterBox SRT Segment {i+1} (Seq {subtitle.sequence}): Character switching - {', '.join(characters)}")
                        
                        print(f"🔧 Note: Multilingual engine may load additional models for character/language switching within this segment")
                        
                        # Lazy load modular components
                        if self.multilingual_engine is None:
                            from utils.voice.multilingual_engine import MultilingualEngine
                            from engines.adapters.chatterbox_adapter import ChatterBoxEngineAdapter
                            self.multilingual_engine = MultilingualEngine("chatterbox")
                            self.chatterbox_adapter = ChatterBoxEngineAdapter(self)
                        
                        # Use modular multilingual engine
                        result = self.multilingual_engine.process_multilingual_text(
                            text=subtitle.text,
                            engine_adapter=self.chatterbox_adapter,
                            model=language,
                            device=device,
                            main_audio_reference=audio_prompt,
                            main_text_reference="",  # ChatterBox doesn't use text reference
                            stable_audio_component=stable_audio_prompt_component,
                            temperature=temperature,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            seed=seed,
                            enable_audio_cache=enable_audio_cache,
                            crash_protection_template=crash_protection_template
                        )
                        
                        wav = result.audio
                        natural_duration = self.AudioTimingUtils.get_audio_duration(wav, self.tts_model.sr)
                        
                    else:  # subtitle_type == 'simple'
                        # Single character mode - model already loaded for this language group
                        single_char, single_text, single_lang = character_segments_with_lang[0]
                        
                        # BUGFIX: Pad short text with custom template to prevent ChatterBox sequential generation crashes
                        processed_subtitle_text = self._pad_short_text_for_chatterbox(single_text, crash_protection_template)
                        
                        # DEBUG: Show actual text being sent to ChatterBox when padding might occur
                        if len(single_text.strip()) < 21:
                            print(f"🔍 DEBUG: Original text: '{single_text}' → Processed: '{processed_subtitle_text}' (len: {len(processed_subtitle_text)})")
                        
                        # Generate new audio with pause tag support (includes internal caching)
                        wav = self._generate_tts_with_pause_tags(
                            processed_subtitle_text, audio_prompt, exaggeration, temperature, cfg_weight, self.current_language,
                            True, character="narrator", seed=seed, enable_cache=enable_audio_cache,
                            crash_protection_template=crash_protection_template,
                            stable_audio_component=stable_audio_prompt_component
                        )
                        natural_duration = self.AudioTimingUtils.get_audio_duration(wav, self.tts_model.sr)
                    
                    # Store results in correct position
                    audio_segments[i] = wav
                    natural_durations[i] = natural_duration
            
            # Handle empty subtitles separately
            for i, subtitle, subtitle_type, _, _ in all_subtitle_segments:
                if subtitle_type == 'empty':
                    # Handle empty text by creating silence
                    natural_duration = subtitle.duration
                    wav = self.AudioTimingUtils.create_silence(
                        duration_seconds=natural_duration,
                        sample_rate=self.tts_model.sr,
                        channels=1,
                        device=self.device
                    )
                    print(f"🤫 Segment {i+1} (Seq {subtitle.sequence}): Empty text, generating {natural_duration:.2f}s silence.")
                    audio_segments[i] = wav
                    natural_durations[i] = natural_duration
            
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
            elif current_timing_mode == "concatenate":
                # Concatenate audio naturally and recalculate SRT timings using modular approach
                from utils.timing.engine import TimingEngine
                from utils.timing.assembly import AudioAssemblyEngine
                
                timing_engine = TimingEngine(self.tts_model.sr)
                assembler = AudioAssemblyEngine(self.tts_model.sr)
                
                # Calculate new timings for concatenation
                adjustments = timing_engine.calculate_concatenation_adjustments(audio_segments, subtitles)
                
                # Assemble audio with optional crossfading
                final_audio = assembler.assemble_concatenation(audio_segments, fade_for_StretchToFit)
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
        from utils.timing.assembly import AudioAssemblyEngine
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