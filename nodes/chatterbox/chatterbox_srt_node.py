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
from utils.text.segment_parameters import apply_segment_parameters
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
                "batch_size": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Parallel processing: 0=traditional mode (sequential), 1+=streaming parallel workers. Higher values = faster generation but more memory usage."
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
    
    def _get_stable_audio_component(self, voice_path, reference_audio=None):
        """Generate stable audio component identifier for cache consistency."""
        # Use robust import system (fix for issue #12)
        from utils.robust_import import robust_from_import
        attrs = robust_from_import('utils.audio.audio_hash', ['generate_stable_audio_component'])
        generate_stable_audio_component = attrs['generate_stable_audio_component']
        return generate_stable_audio_component(reference_audio, voice_path)

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
                # CRITICAL FIX: Use unified cache system to match streaming method
                from utils.audio.cache import create_cache_function
                
                # Use stable audio component if provided, otherwise use audio prompt path
                audio_component = stable_audio_component if stable_audio_component else str(audio_prompt or "main_reference")
                
                # Create cache function using same parameters as streaming method
                cache_fn = create_cache_function(
                    engine_type="chatterbox",
                    character=character,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    seed=seed,
                    audio_component=audio_component,
                    model_source=self.model_manager.get_model_source("tts") or "unknown",
                    device=self.device,
                    language=language
                )
                
                # Try cache first
                cached_audio = cache_fn(processed_text)
                if cached_audio is not None:
                    return cached_audio
                
                # Generate and cache
                audio = self._safe_generate_tts_audio(processed_text, audio_prompt, exaggeration, temperature, cfg_weight)
                
                # Cache the result using the unified cache system
                cache_fn(processed_text, audio_result=audio)
                return audio
            else:
                return self._safe_generate_tts_audio(processed_text, audio_prompt, exaggeration, temperature, cfg_weight)
        
        # Generate audio with pause tags, caching individual text segments
        def tts_generate_func(text_content: str) -> torch.Tensor:
            """TTS generation function for pause tag processor with caching"""
            if enable_cache:
                # CRITICAL FIX: Use unified cache system for pause segments too
                from utils.audio.cache import create_cache_function
                
                # Apply crash protection to individual text segment FIRST
                protected_text = self._pad_short_text_for_chatterbox(text_content, crash_protection_template)
                if len(text_content.strip()) < 21:
                    print(f"🔍 DEBUG: Pause segment original: '{text_content}' → Protected: '{protected_text}' (len: {len(protected_text)})")
                
                # Use stable audio component if provided, otherwise use audio prompt path
                audio_component = stable_audio_component if stable_audio_component else str(audio_prompt or "main_reference")
                
                # Create cache function using same parameters as streaming method
                cache_fn = create_cache_function(
                    engine_type="chatterbox",
                    character=character,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    seed=seed,
                    audio_component=audio_component,
                    model_source=self.model_manager.get_model_source("tts") or "unknown",
                    device=self.device,
                    language=language
                )
                
                # Try cache first with protected text
                cached_audio = cache_fn(protected_text)
                if cached_audio is not None:
                    print(f"💾 CACHE HIT for pause segment: '{text_content[:30]}...'")
                    return cached_audio
                
                # Generate and cache
                audio = self._safe_generate_tts_audio(protected_text, audio_prompt, exaggeration, temperature, cfg_weight)
                
                # Cache the result using the unified cache system
                cache_fn(protected_text, audio_result=audio)
                return audio
            else:
                # Apply crash protection to individual text segment
                protected_text = self._pad_short_text_for_chatterbox(text_content, crash_protection_template)
                if len(text_content.strip()) < 21:
                    print(f"🔍 DEBUG: Pause segment original: '{text_content}' → Protected: '{protected_text}' (len: {len(protected_text)})")
                
                return self._safe_generate_tts_audio(protected_text, audio_prompt, exaggeration, temperature, cfg_weight)
        
        return PauseTagProcessor.generate_audio_with_pauses(
            pause_segments, tts_generate_func, self.tts_model.sr if hasattr(self, 'tts_model') and self.tts_model else 24000
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
    
    
    def _generate_with_preloaded_model(self, model, text: str, voice_path: str, language: str, 
                                     character: str, exaggeration: float, temperature: float, 
                                     cfg_weight: float, seed: int, enable_cache: bool, 
                                     crash_protection_template: str, stable_audio_component: str) -> torch.Tensor:
        """
        Generate audio using a specific pre-loaded model without affecting shared node state.
        Thread-safe method for streaming workers.
        
        Args:
            model: Pre-loaded ChatterBox TTS model instance
            text: Text to generate
            voice_path: Voice reference path
            language: Language code
            character: Character name
            exaggeration: ChatterBox exaggeration parameter  
            temperature: ChatterBox temperature parameter
            cfg_weight: ChatterBox CFG weight parameter
            seed: Seed for reproducibility
            enable_cache: Whether to use caching
            crash_protection_template: Crash protection template
            stable_audio_component: Stable audio component identifier
            
        Returns:
            Generated audio tensor
        """
        # Set seed for reproducibility
        self.set_seed(seed)
        
        # Generate audio using the pre-loaded model directly with caching
        print(f"🔄 Generating {language} audio with preloaded model (ID: {id(model)})")
        
        # Apply crash protection to text if needed
        if len(text.strip()) < 21:
            protected_text = self._pad_short_text_for_chatterbox(text, crash_protection_template)
            print(f"🛡️ Applied crash protection: '{text}' → '{protected_text}'")
            text = protected_text
        
        # Handle caching for streaming generation - use same unified cache system as traditional
        if enable_cache:
            from utils.audio.cache import create_cache_function
            
            # Use stable audio component for consistent caching
            audio_component = stable_audio_component if stable_audio_component else str(voice_path or "main_reference")
            
            # Create cache function using same parameters as traditional method for consistency
            cache_fn = create_cache_function(
                engine_type="chatterbox",
                character=character,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
                seed=seed,
                audio_component=audio_component,
                model_source="streaming_preloaded",  # Distinguish streaming cache from traditional
                device=self.device if hasattr(self, 'device') else "auto",
                language=language
            )
            
            # Try cache first
            cached_audio = cache_fn(text)
            if cached_audio is not None:
                print(f"💾 CACHE HIT for streaming segment: '{text[:30]}...'")
                return cached_audio
        
        # Generate audio using the pre-loaded model
        try:
            with torch.no_grad():
                audio = model.generate(
                    text=text,
                    audio_prompt_path=voice_path if voice_path != "none" else None,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight
                )
                
                # Ensure tensor is completely detached
                if hasattr(audio, 'detach'):
                    audio = audio.detach()
                
                print(f"✅ Generated {language} audio using preloaded model (shape: {audio.shape})")
                
                # Cache the result if caching is enabled
                if enable_cache:
                    cache_fn(text, audio_result=audio)
                
                return audio
        except Exception as e:
            print(f"❌ Failed to generate {language} audio: {e}")
            return torch.zeros(1, 1000)

    def _process_single_segment_for_streaming(self, original_idx, character, segment_text, language, voice_path, inputs):
        """Process a single segment for the streaming processor using pre-loaded models."""
        # This method is called by the streaming worker
        try:
            # Get the stateless wrapper for thread safety
            if hasattr(self, '_streaming_model_manager'):
                stateless_model = self._streaming_model_manager.get_stateless_model_for_language(language)
                if stateless_model:
                    # Check if text has pause tags
                    from utils.text.pause_processor import PauseTagProcessor
                    if PauseTagProcessor.has_pause_tags(segment_text):
                        # Process pause tags manually for stateless wrapper
                        pause_segments, clean_text = PauseTagProcessor.parse_pause_tags(segment_text)
                        audio_segments = []
                        
                        for segment_type, content in pause_segments:
                            if segment_type == 'text' and content.strip():
                                # Apply crash protection to short segments
                                text_for_generation = content
                                if len(content.strip()) < 21:
                                    text_for_generation = self._pad_short_text_for_chatterbox(
                                        content, 
                                        inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,")
                                    )
                                
                                # Generate audio for text segment with individual caching
                                enable_cache = inputs.get("enable_audio_cache", True)
                                cached_audio = None
                                
                                # Try cache first for this individual segment
                                if enable_cache:
                                    from utils.audio.cache import create_cache_function
                                    
                                    # Debug cache parameters
                                    cache_params = {
                                        'character': character,
                                        'exaggeration': inputs.get("exaggeration", 0.5),
                                        'temperature': inputs.get("temperature", 0.8),
                                        'cfg_weight': inputs.get("cfg_weight", 0.5),
                                        'seed': inputs.get("seed", 42),
                                        'audio_component': self._get_stable_audio_component(voice_path, inputs.get("reference_audio")),
                                        'model_source': "streaming_stateless",
                                        'device': "auto",
                                        'language': language
                                    }
                                    cache_fn = create_cache_function("chatterbox", **cache_params)
                                    cached_audio = cache_fn(text_for_generation)
                                
                                if cached_audio is not None:
                                    print(f"💾 CACHE HIT for segment: '{text_for_generation[:20]}...'")
                                    audio = cached_audio
                                else:
                                    # Generate using stateless wrapper
                                    audio = stateless_model.generate_stateless(
                                        text=text_for_generation,
                                        audio_prompt_path=voice_path if voice_path != "none" else None,
                                        exaggeration=inputs.get("exaggeration", 0.5),
                                        temperature=inputs.get("temperature", 0.8),
                                        cfg_weight=inputs.get("cfg_weight", 0.5),
                                        seed=inputs.get("seed", 42)
                                    )
                                    
                                    # Cache the result
                                    if enable_cache:
                                        cache_fn(text_for_generation, audio_result=audio)
                                        print(f"💾 CACHED segment: '{text_for_generation[:20]}...'")
                                
                                audio_segments.append(audio)
                            elif segment_type == 'pause':
                                # Generate silence for pause segment
                                silence_samples = int(content * 24000)  # content is duration in seconds
                                silence = torch.zeros(1, silence_samples)
                                audio_segments.append(silence)
                        
                        # Concatenate all segments
                        if audio_segments:
                            segment_audio = torch.cat(audio_segments, dim=-1)
                        else:
                            segment_audio = torch.zeros(1, 1000)
                    else:
                        # No pause tags, use stateless generation directly
                        # Apply crash protection to short segments
                        text_for_generation = segment_text
                        if len(segment_text.strip()) < 21:
                            text_for_generation = self._pad_short_text_for_chatterbox(
                                segment_text, 
                                inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,")
                            )
                        
                        # No pause tags, use stateless generation with caching
                        enable_cache = inputs.get("enable_audio_cache", True)
                        cached_audio = None
                        
                        # Try cache first
                        if enable_cache:
                            from utils.audio.cache import create_cache_function
                            cache_fn = create_cache_function(
                                "chatterbox",
                                character=character,
                                exaggeration=inputs.get("exaggeration", 0.5),
                                temperature=inputs.get("temperature", 0.8),
                                cfg_weight=inputs.get("cfg_weight", 0.5),
                                seed=inputs.get("seed", 42),
                                audio_component=self._get_stable_audio_component(voice_path, inputs.get("reference_audio")),
                                model_source="streaming_stateless",
                                device="auto",
                                language=language
                            )
                            cached_audio = cache_fn(text_for_generation)
                        
                        if cached_audio is not None:
                            print(f"💾 CACHE HIT for segment: '{text_for_generation[:20]}...'")
                            segment_audio = cached_audio
                        else:
                            # Generate using stateless wrapper
                            segment_audio = stateless_model.generate_stateless(
                                text=text_for_generation,
                                audio_prompt_path=voice_path if voice_path != "none" else None,
                                exaggeration=inputs.get("exaggeration", 0.5),
                                temperature=inputs.get("temperature", 0.8),
                                cfg_weight=inputs.get("cfg_weight", 0.5),
                                seed=inputs.get("seed", 42)
                            )
                            
                            # Cache the result
                            if enable_cache:
                                cache_fn(text_for_generation, audio_result=segment_audio)
                                print(f"💾 CACHED segment: '{text_for_generation[:20]}...'")
                else:
                    # Fallback: load model and use pause-aware generation
                    self.load_tts_model(inputs.get("device", "auto"), language)
                    # Generate stable audio component for cache consistency
                    stable_audio_component = self._get_stable_audio_component(voice_path, inputs.get("reference_audio"))
                    
                    segment_audio = self._generate_tts_with_pause_tags(
                        segment_text, voice_path, inputs.get("exaggeration", 0.5),
                        inputs.get("temperature", 0.8), inputs.get("cfg_weight", 0.5), language,
                        True, character=character, seed=inputs.get("seed", 42),
                        enable_cache=inputs.get("enable_audio_cache", True),
                        crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                        stable_audio_component=stable_audio_component
                    )
            else:
                # No streaming model manager, fallback to direct model loading
                self.load_tts_model(inputs.get("device", "auto"), language)
                # Generate stable audio component for cache consistency
                stable_audio_component = self._get_stable_audio_component(voice_path, inputs.get("reference_audio"))
                
                segment_audio = self._generate_tts_with_pause_tags(
                    segment_text, voice_path, inputs.get("exaggeration", 0.5),
                    inputs.get("temperature", 0.8), inputs.get("cfg_weight", 0.5), language,
                    True, character=character, seed=inputs.get("seed", 42),
                    enable_cache=inputs.get("enable_audio_cache", True),
                    crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                    stable_audio_component=stable_audio_component
                )
            
            return segment_audio
            
        except Exception as e:
            print(f"❌ Failed to process streaming segment: {e}")
            return torch.zeros(1, 1000)

    def generate_srt_speech(self, srt_content, language, device, exaggeration, temperature, cfg_weight, seed,
                            timing_mode, reference_audio=None, audio_prompt_path="",
                            enable_audio_cache=True, fade_for_StretchToFit=0.01, 
                            max_stretch_ratio=2.0, min_stretch_ratio=0.5, timing_tolerance=2.0,
                            crash_protection_template="hmm ,, {seg} hmm ,,", batch_size=0):
        
        def _process():
            # Check if SRT support is available
            if not self.srt_available:
                raise ImportError("SRT support not available - missing required modules")
            
            # Set seed for reproducibility (do this before model loading)
            self.set_seed(seed)
            
            # Determine audio prompt component for cache key generation (stable identifier)
            # This must be done BEFORE handle_reference_audio to avoid using temporary file paths
            # Use robust import system (fix for issue #12)
            from utils.robust_import import robust_from_import
            attrs = robust_from_import('utils.audio.audio_hash', ['generate_stable_audio_component'])
            generate_stable_audio_component = attrs['generate_stable_audio_component']
            stable_audio_prompt_component = generate_stable_audio_component(reference_audio, audio_prompt_path)
            
            # Handle reference audio (this may create temporary files, but we don't use them in cache key)
            audio_prompt = self.handle_reference_audio(reference_audio, audio_prompt_path)
            
            # Parse SRT content with overlap support
            srt_parser = self.SRTParser()
            subtitles = srt_parser.parse_srt_content(srt_content, allow_overlaps=True)
            
            # Check if subtitles have overlaps and handle smart_natural mode using modular utility
            from utils.timing.overlap_detection import SRTOverlapHandler
            has_overlaps = SRTOverlapHandler.detect_overlaps(subtitles)
            current_timing_mode, mode_switched = SRTOverlapHandler.handle_smart_natural_fallback(
                timing_mode, has_overlaps, "ChatterBox SRT"
            )
            
            # Set up character parser with available characters BEFORE processing subtitles
            available_chars = get_available_characters()
            character_parser.set_available_characters(list(available_chars))
            
            # CRITICAL FIX: Reset character parser session to prevent language contamination from previous executions
            character_parser.reset_session_cache()
            character_parser.set_engine_aware_default_language(language, "chatterbox")
            
            # SMART OPTIMIZATION: Group subtitles by language to minimize model switching
            subtitle_language_groups = {}
            all_subtitle_segments = []
            
            # First pass: analyze all subtitles and group by language
            for i, subtitle in enumerate(subtitles):
                if not subtitle.text.strip():
                    # Empty subtitle - will be handled separately
                    all_subtitle_segments.append((i, subtitle, 'empty', None, None))
                    continue
                
                # Parse character segments with parameters (handles both language and character tags)
                segment_objects = character_parser.parse_text_segments(subtitle.text)

                # Convert to 4-tuple format with parameters
                # Use original_character (before alias resolution) to preserve the actual character from the tag
                # This ensures Alice stays as Alice for voice lookup, not converted to narrator
                character_segments_with_lang = [(seg.original_character or seg.character, seg.text, seg.language, seg.parameters if seg.parameters else {}) for seg in segment_objects]

                # Check if we have character switching, language switching, or parameter changes
                characters = list(set(char for char, _, _, _ in character_segments_with_lang))
                languages = list(set(lang for _, _, lang, _ in character_segments_with_lang))
                # Check if parameters differ across segments (indicates parameter switching)
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
                    print(f"🔍 SRT DEBUG: Subtitle {i+1} '{subtitle.text[:30]}...' detected as language '{single_lang}' (expected: '{language}')")
                    all_subtitle_segments.append((i, subtitle, 'simple', single_lang, character_segments_with_lang))

                    if single_lang not in subtitle_language_groups:
                        subtitle_language_groups[single_lang] = []
                    subtitle_language_groups[single_lang].append((i, subtitle, 'simple', character_segments_with_lang))
            
            # Route to streaming or traditional processing using universal system
            if batch_size > 0:
                # Use universal streaming system
                from utils.streaming import StreamingCoordinator, StreamingConfig
                from engines.adapters.chatterbox_streaming_adapter import ChatterBoxStreamingAdapter
                
                # Convert SRT data to universal segments with original character info
                srt_segments_data = []
                for lang_code, lang_subtitles in subtitle_language_groups.items():
                    for i, subtitle, subtitle_type, character_segments_with_lang in lang_subtitles:
                        if subtitle_type == 'multilingual' or subtitle_type == 'multicharacter':
                            # Handle complex subtitles with character switching
                            # Need to get original character info before alias resolution
                            detailed_segments = character_parser.parse_text_segments(subtitle.text)

                            # Build segment data with original character info and parameters
                            segment_data = []
                            for seg_idx, (char, text, seg_lang) in enumerate(character_segments_with_lang):
                                # Get original character and parameters from detailed segments if available
                                if seg_idx < len(detailed_segments):
                                    segment_obj = detailed_segments[seg_idx]
                                    original_char = segment_obj.original_character or char
                                    segment_params = segment_obj.parameters if segment_obj.parameters else {}
                                else:
                                    original_char = char
                                    segment_params = {}
                                segment_data.append((char, text, seg_lang, original_char, segment_params))

                            srt_segments_data.append((i, subtitle, segment_data))
                        else:
                            # Simple subtitle - single narrator
                            srt_segments_data.append((i, subtitle, [('narrator', subtitle.text, lang_code, 'narrator', {})]))
                
                # Build voice references for characters
                # CRITICAL FIX: Use audio_prompt (processed result) instead of audio_prompt_path (input parameter)
                voice_refs = {'narrator': audio_prompt or None}
                try:
                    # Use already imported functions (imported at module level)
                    available_chars = get_available_characters()
                    char_mapping = get_character_mapping(list(available_chars), "chatterbox")
                    for char in available_chars:
                        char_audio_path, _ = char_mapping.get(char, (audio_prompt or None, None))
                        voice_refs[char] = char_audio_path
                except ImportError:
                    pass
                
                # Convert to universal segments
                segments = StreamingCoordinator.convert_node_data_to_segments(
                    node_type='srt',
                    data=srt_segments_data,
                    voice_refs=voice_refs
                )
                
                # Create streaming configuration
                config = StreamingConfig(
                    batch_size=batch_size,
                    enable_model_preloading=True,
                    fallback_to_traditional=True,
                    streaming_threshold=1,
                    engine_config={'device': device}
                )
                
                # Create adapter and process
                adapter = ChatterBoxStreamingAdapter(self)
                
                # Pre-load models using streaming model manager for thread safety
                print(f"🚀 SRT STREAMING: Pre-loading models for {len(subtitle_language_groups)} languages")
                from engines.chatterbox.streaming_model_manager import StreamingModelManager
                
                # Initialize streaming model manager
                self._streaming_model_manager = StreamingModelManager(language)
                self._streaming_model_manager.preload_models(
                    language_codes=list(subtitle_language_groups.keys()),
                    model_manager=self.model_manager,
                    device=device
                )
                
                # Process with universal coordinator
                results, metrics, success = StreamingCoordinator.process(
                    segments=segments,
                    adapter=adapter,
                    config=config,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    seed=seed,
                    enable_audio_cache=enable_audio_cache,
                    crash_protection_template=crash_protection_template
                )
                
                # Convert results to SRT format
                audio_segments, natural_durations, any_segment_cached = StreamingCoordinator.convert_results_to_node_format(
                    node_type='srt',
                    results=results,
                    original_data=subtitles,
                    sample_rate=24000,
                    enable_audio_cache=enable_audio_cache,
                    segments=segments  # Pass segments for proper subtitle ordering
                )
                
                print(f"✅ SRT streaming complete: {metrics.get_summary()['completed_segments']} segments processed")
                
                # Keep streaming models for post-processing (only needs .sr sample rate)
                if hasattr(self, '_streaming_model_manager') and self._streaming_model_manager.preloaded_models:
                    # Set tts_model to any preloaded model for sample rate access
                    self.tts_model = next(iter(self._streaming_model_manager.preloaded_models.values()))
            else:
                # Traditional sequential processing (existing logic)
                # SMART INITIALIZATION: Load the first language model we'll actually need
                # Use first language group (alphabetical order) since that's the processing order
                first_language_code = sorted(subtitle_language_groups.keys())[0] if subtitle_language_groups else 'en'
                from utils.models.language_mapper import get_model_for_language
                required_language = get_model_for_language("chatterbox", first_language_code, language)
                print(f"🚀 SRT: Smart initialization - loading {required_language} model for first language group '{first_language_code}'")
                self.load_tts_model(device, required_language)
                self.current_language = required_language
                self.current_model_name = required_language  # For multilingual engine compatibility
                audio_segments, natural_durations, any_segment_cached = self._process_traditional_srt_logic(
                    subtitles, subtitle_language_groups, language, device, exaggeration, temperature,
                    cfg_weight, seed, reference_audio, audio_prompt_path, enable_audio_cache,
                    crash_protection_template, stable_audio_prompt_component, all_subtitle_segments, audio_prompt
                )
            
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
            model_source = f"chatterbox_{language.lower()}"
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
        
        # Filter out failed segments (None audio) and their corresponding subtitles
        filtered_audio = []
        filtered_subtitles = []
        failed_indices = []
        
        for i, (audio, subtitle) in enumerate(zip(audio_segments, subtitles)):
            if audio is not None:
                filtered_audio.append(audio)
                filtered_subtitles.append(subtitle)
            else:
                failed_indices.append(i)
                print(f"⚠️ SMART TIMING: Skipping failed segment {i} (subtitle: '{subtitle.text[:50]}...')")
        
        if failed_indices:
            print(f"⚠️ SMART TIMING: {len(failed_indices)} segments failed and will be skipped")
            
        # Calculate smart adjustments and process segments (only successful ones)
        filtered_adjustments, processed_segments = timing_engine.calculate_smart_timing_adjustments(
            filtered_audio, filtered_subtitles, tolerance, max_stretch_ratio, min_stretch_ratio, self.device
        )
        
        # Map filtered adjustments back to original subtitle indices
        adjustments = []
        filtered_idx = 0
        for i in range(len(subtitles)):
            if i in failed_indices:
                # Create dummy adjustment for failed segment
                adjustments.append({
                    'segment_index': i,
                    'sequence': subtitles[i].sequence,
                    'status': 'failed',
                    'natural_duration': 0.0,
                    'target_duration': subtitles[i].end_time - subtitles[i].start_time,
                    'stretch_ratio': 1.0,
                    'method': 'none'
                })
            else:
                # Use adjustment from successful segment
                if filtered_idx < len(filtered_adjustments):
                    adjustments.append(filtered_adjustments[filtered_idx])
                filtered_idx += 1
        
        # Assemble the final audio (use filtered data)
        final_audio = assembler.assemble_smart_natural(filtered_audio, processed_segments, filtered_adjustments, filtered_subtitles, self.device)
        
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
    
    def _process_traditional_srt_logic(self, subtitles, subtitle_language_groups, language, device, exaggeration,
                                     temperature, cfg_weight, seed, reference_audio, audio_prompt_path,
                                     enable_audio_cache, crash_protection_template, stable_audio_prompt_component,
                                     all_subtitle_segments, audio_prompt):
        """Traditional sequential SRT processing logic - preserves ALL original functionality."""
        from utils.models.language_mapper import get_model_for_language

        # Initialize result arrays
        audio_segments = [None] * len(subtitles)
        natural_durations = [0.0] * len(subtitles)
        any_segment_cached = False

        # Build voice references for characters (same as streaming path)
        voice_refs = {'narrator': audio_prompt or None}
        try:
            available_chars = get_available_characters()
            char_mapping = get_character_mapping(list(available_chars), "chatterbox")
            for char in available_chars:
                char_audio_path, _ = char_mapping.get(char, (audio_prompt or None, None))
                voice_refs[char] = char_audio_path
        except ImportError:
            pass
        
        # Process each language group with ALL original logging and logic
        for lang_code in sorted(subtitle_language_groups.keys()):
            lang_subtitles = subtitle_language_groups[lang_code]
            
            print(f"📋 Processing {len(lang_subtitles)} SRT subtitle(s) in '{lang_code}' language group...")
            
            # Check if we need to switch models for this language group
            required_language = get_model_for_language("chatterbox", lang_code, language)
            if self.current_language != required_language:
                print(f"🎯 SRT: Switching to {required_language} model for {len(lang_subtitles)} subtitle(s) in '{lang_code}'")
                self.load_tts_model(device, required_language)
                self.current_language = required_language
                self.current_model_name = required_language  # For multilingual engine compatibility
            else:
                print(f"✅ SRT: Using {required_language} model for {len(lang_subtitles)} subtitle(s) in '{lang_code}' (already loaded)")
            
            # Process each subtitle in this language group
            for i, subtitle, subtitle_type, character_segments_with_lang in lang_subtitles:
                print(f"📺 Generating SRT segment {i+1}/{len(subtitles)} (Seq {subtitle.sequence}) in {lang_code}...")

                # Check for interruption
                self.check_interruption(f"SRT generation segment {i+1}/{len(subtitles)} (Seq {subtitle.sequence})")

                if subtitle_type == 'parameter_switching':
                    # Process each segment individually with its own parameters
                    print(f"🔀 ChatterBox SRT Segment {i+1} (Seq {subtitle.sequence}): Per-segment parameter switching")

                    segment_audio_parts = []
                    for seg_idx, (char, text, lang, seg_params) in enumerate(character_segments_with_lang):
                        # Get character-specific voice reference
                        char_voice = voice_refs.get(char, voice_refs.get("narrator", None))

                        # Apply segment parameters
                        current_temp = temperature
                        current_exag = exaggeration
                        current_cfg = cfg_weight
                        current_seed = seed

                        if seg_params:
                            segment_config = apply_segment_parameters(
                                {'temperature': temperature, 'exaggeration': exaggeration, 'cfg_weight': cfg_weight, 'seed': seed},
                                seg_params,
                                "chatterbox"
                            )
                            current_temp = segment_config.get('temperature', temperature)
                            current_exag = segment_config.get('exaggeration', exaggeration)
                            current_cfg = segment_config.get('cfg_weight', cfg_weight)
                            current_seed = segment_config.get('seed', seed)
                            print(f"  📊 Segment {seg_idx+1}: Applying parameters {seg_params}")

                        # Pad short text with crash protection
                        processed_text = self._pad_short_text_for_chatterbox(text, crash_protection_template)

                        # Generate audio for this segment with its parameters and character voice
                        segment_wav = self._generate_tts_with_pause_tags(
                            processed_text, char_voice, current_exag, current_temp, current_cfg, lang,
                            True, character=char, seed=current_seed, enable_cache=enable_audio_cache,
                            crash_protection_template=crash_protection_template,
                            stable_audio_component=stable_audio_prompt_component
                        )
                        segment_audio_parts.append(segment_wav)

                    # Concatenate all segment audio
                    if segment_audio_parts:
                        import torch
                        wav = torch.cat(segment_audio_parts, dim=-1)
                    else:
                        wav = torch.zeros(1, 1000)

                    natural_duration = self.AudioTimingUtils.get_audio_duration(wav, self.tts_model.sr)

                elif subtitle_type == 'multilingual' or subtitle_type == 'multicharacter':
                    # Use modular multilingual engine for character/language switching
                    characters = list(set(char for char, _, _, _ in character_segments_with_lang))
                    languages = list(set(lang for _, _, lang, _ in character_segments_with_lang))
                    
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

                    # Apply per-segment parameters if any segment has them
                    current_temp = temperature
                    current_exag = exaggeration
                    current_cfg = cfg_weight
                    current_seed = seed

                    # Check if any segment has parameters and apply them
                    segment_has_params = any(seg_params for _, _, _, seg_params in character_segments_with_lang)
                    if segment_has_params:
                        # Extract first segment with parameters (will be applied to entire subtitle for now)
                        for _, _, _, seg_params in character_segments_with_lang:
                            if seg_params:
                                segment_config = apply_segment_parameters(
                                    {'temperature': temperature, 'exaggeration': exaggeration, 'cfg_weight': cfg_weight, 'seed': seed},
                                    seg_params,
                                    "chatterbox"
                                )
                                current_temp = segment_config.get('temperature', temperature)
                                current_exag = segment_config.get('exaggeration', exaggeration)
                                current_cfg = segment_config.get('cfg_weight', cfg_weight)
                                current_seed = segment_config.get('seed', seed)
                                print(f"  📊 SRT: Applying segment parameters: {seg_params}")
                                break  # Use first segment's parameters for the entire subtitle

                    # Use modular multilingual engine
                    result = self.multilingual_engine.process_multilingual_text(
                        text=subtitle.text,
                        engine_adapter=self.chatterbox_adapter,
                        model=language,
                        device=device,
                        main_audio_reference=audio_prompt,
                        main_text_reference="",  # ChatterBox doesn't use text reference
                        stable_audio_component=stable_audio_prompt_component,
                        temperature=current_temp,
                        exaggeration=current_exag,
                        cfg_weight=current_cfg,
                        seed=current_seed,
                        enable_audio_cache=enable_audio_cache,
                        crash_protection_template=crash_protection_template
                    )
                    
                    # CRITICAL FIX: Restore the language model for this language group 
                    # The multilingual engine may have switched to other models during processing
                    if self.current_language != required_language:
                        print(f"🔄 Restoring {required_language} model after multilingual processing")
                        self.load_tts_model(device, required_language)
                        self.current_language = required_language
                        self.current_model_name = required_language
                    
                    wav = result.audio
                    natural_duration = self.AudioTimingUtils.get_audio_duration(wav, self.tts_model.sr)
                    
                else:  # subtitle_type == 'simple'
                    # Single character mode - model already loaded for this language group
                    single_char, single_text, single_lang, single_params = character_segments_with_lang[0]

                    # Apply per-segment parameters if available
                    current_temp = temperature
                    current_exag = exaggeration
                    current_cfg = cfg_weight
                    current_seed = seed

                    if single_params:
                        segment_config = apply_segment_parameters(
                            {'temperature': temperature, 'exaggeration': exaggeration, 'cfg_weight': cfg_weight, 'seed': seed},
                            single_params,
                            "chatterbox"
                        )
                        current_temp = segment_config.get('temperature', temperature)
                        current_exag = segment_config.get('exaggeration', exaggeration)
                        current_cfg = segment_config.get('cfg_weight', cfg_weight)
                        current_seed = segment_config.get('seed', seed)
                        print(f"  📊 SRT: Applying segment parameters: {single_params}")

                    # BUGFIX: Pad short text with custom template to prevent ChatterBox sequential generation crashes
                    processed_subtitle_text = self._pad_short_text_for_chatterbox(single_text, crash_protection_template)

                    # DEBUG: Show actual text being sent to ChatterBox when padding might occur
                    if len(single_text.strip()) < 21:
                        print(f"🔍 DEBUG: Original text: '{single_text}' → Processed: '{processed_subtitle_text}' (len: {len(processed_subtitle_text)})")

                    # Show what model is actually being used for generation
                    model_path = getattr(self.tts_model, 'model_dir', 'unknown') if hasattr(self, 'tts_model') else 'no_model'
                    print(f"🔧 TRADITIONAL SRT: Generating subtitle {i+1} using '{self.current_language}' model")
                    print(f"📁 MODEL PATH: {model_path}")

                    # DEBUG: Show actual model state like multilingual engine does
                    actual_current_model = getattr(self, 'current_model_name', 'unknown')
                    print(f"🔧 ACTUAL MODEL: Traditional SRT subtitle {i+1} using '{actual_current_model}' model")

                    # Generate new audio with pause tag support (includes internal caching)
                    wav = self._generate_tts_with_pause_tags(
                        processed_subtitle_text, audio_prompt, current_exag, current_temp, current_cfg, self.current_language,
                        True, character="narrator", seed=current_seed, enable_cache=enable_audio_cache,
                        crash_protection_template=crash_protection_template,
                        stable_audio_component=stable_audio_prompt_component
                    )
                    natural_duration = self.AudioTimingUtils.get_audio_duration(wav, self.tts_model.sr)
                
                # Store results in correct position
                audio_segments[i] = wav
                natural_durations[i] = natural_duration
                
                # Track if any segments were cached (approximate - would need deeper integration)
                if enable_audio_cache:
                    any_segment_cached = True
        
        return audio_segments, natural_durations, any_segment_cached
    
    
    def _preload_language_models(self, language_codes, device):
        """Pre-load all required language models for streaming to prevent worker conflicts."""
        from engines.chatterbox.streaming_model_manager import StreamingModelManager
        
        # Create streaming model manager if not exists
        if not hasattr(self, '_streaming_model_manager'):
            self._streaming_model_manager = StreamingModelManager()
        
        # Pre-load models using the streaming model manager
        self._streaming_model_manager.preload_models(
            language_codes=list(language_codes),
            model_manager=self,  # Pass self as model_manager (has load_tts_model method)
            device=device
        )
        
        print(f"🚀 SRT Pre-loading complete: {len(self._streaming_model_manager.preloaded_models)} models ready")