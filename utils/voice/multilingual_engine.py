"""
Multilingual Engine - Central orchestrator for multilingual TTS generation
Handles language switching, character management, and cache optimization for any TTS engine
"""

import torch
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass

from utils.text.character_parser import character_parser
from utils.voice.discovery import get_available_characters, get_character_mapping
from utils.text.pause_processor import PauseTagProcessor


@dataclass
class AudioSegmentResult:
    """Result of audio generation for a single segment"""
    audio: torch.Tensor
    duration: float
    character: str
    text: str
    language: str
    original_index: int


@dataclass
class MultilingualResult:
    """Complete result of multilingual processing"""
    audio: torch.Tensor
    total_duration: float
    segments: List[AudioSegmentResult]
    languages_used: List[str]
    characters_used: List[str]
    info_message: str


class MultilingualEngine:
    """
    Central orchestrator for multilingual TTS generation.
    
    Handles:
    - Language-aware character parsing
    - Smart model loading with cache optimization
    - Language grouping for efficient processing
    - Proper segment ordering and audio assembly
    """
    
    def __init__(self, engine_type: str):
        """
        Initialize multilingual engine.
        
        Args:
            engine_type: "f5tts" or "chatterbox"
        """
        self.engine_type = engine_type
        self.sample_rate = 24000 if engine_type == "f5tts" else 44100
        self.loaded_models = set()  # Track loaded models across calls
        
    def process_multilingual_text(self, text: str, engine_adapter, **params) -> MultilingualResult:
        """
        Main entry point for multilingual processing.
        
        Args:
            text: Input text with character/language tags
            engine_adapter: Engine-specific adapter (F5TTSEngineAdapter or ChatterBoxEngineAdapter)
            **params: Engine-specific parameters
            
        Returns:
            MultilingualResult with generated audio and metadata
        """
        # 1. Parse character segments with languages from original text
        character_segments_with_lang = character_parser.split_by_character_with_language(text)
        
        # Get detailed segments to access original character information
        detailed_segments = character_parser.parse_text_segments(text)
        
        # 2. Analyze segments
        characters = list(set(char for char, _, _ in character_segments_with_lang))
        languages = list(set(lang for _, _, lang in character_segments_with_lang))
        has_multiple_characters = len(characters) > 1 or (len(characters) == 1 and characters[0] != "narrator")
        has_multiple_languages = len(languages) > 1
        
        # Print analysis
        if has_multiple_languages:
            print(f"🌍 {self.engine_type.title()}: Language switching mode - found languages: {', '.join(languages)}")
        if has_multiple_characters:
            print(f"🎭 {self.engine_type.title()}: Character switching mode - found characters: {', '.join(characters)}")
        
        # 3. Group segments by language to optimize model loading
        language_groups = self._group_segments_by_language_with_original(detailed_segments)
        
        # 4. Get character mapping for all characters
        character_mapping = get_character_mapping(characters, engine_type=self.engine_type)
        
        # 5. Check cache optimization opportunities
        cache_info = self._analyze_cache_coverage(language_groups, character_mapping, engine_adapter, **params)
        
        # 6. Process each language group with smart model loading  
        all_audio_segments = []
        base_model_loaded = False
        
        # Initialize loaded_models tracking with currently loaded model (if any)
        current_model = getattr(engine_adapter.node, 'current_model_name', None)
        if current_model and current_model not in self.loaded_models:
            self.loaded_models.add(current_model)
            print(f"💾 Multilingual engine: Detected already loaded model '{current_model}'")
        
        # Use persistent loaded_models tracking across calls
        
        for lang_code, lang_segments in language_groups.items():
            # Get required model for this language
            required_model = engine_adapter.get_model_for_language(lang_code, params.get("model", "default"))
            
            # Check if all segments in this language group are cached
            if cache_info.get(lang_code, {}).get("all_cached", False):
                print(f"💾 Skipping model load for language '{lang_code}' - all {len(lang_segments)} segments cached")
            else:
                # Load base model if this is the first time we need to generate anything
                if not base_model_loaded:
                    # CRITICAL FIX: Load the required model for this language, not the default model
                    # This ensures the correct tokenizer is loaded
                    print(f"🔄 Loading base {self.engine_type.title()} model for generation ({required_model})")
                    engine_adapter.load_base_model(required_model, params.get("device", "auto"))
                    base_model_loaded = True
                
                # Only load language model if we haven't loaded this specific model yet
                if required_model not in self.loaded_models:
                    print(f"🌍 Loading {self.engine_type.title()} model '{required_model}' for language '{lang_code}' ({len(lang_segments)} segments)")
                    try:
                        engine_adapter.load_language_model(required_model, params.get("device", "auto"))
                        self.loaded_models.add(required_model)  # Track that this model is now loaded
                        # Update node's current language state
                        engine_adapter.node.current_language = required_model
                        engine_adapter.node.current_model_name = required_model
                        print(f"🔄 Updated current_language to '{required_model}'")
                    except Exception as e:
                        print(f"⚠️ Failed to load model '{required_model}' for language '{lang_code}': {e}")
                        print(f"🔄 Falling back to English model")
                        engine_adapter.load_base_model("English", params.get("device", "auto"))
                else:
                    print(f"💾 Model '{required_model}' already loaded - reusing for language '{lang_code}' ({len(lang_segments)} segments)")
                    # CRITICAL FIX: Force model manager to switch to the correct cached model instance
                    # The model is cached but self.tts_model needs to point to the right instance
                    engine_adapter.node.load_tts_model(params.get("device", "auto"), required_model)
                    engine_adapter.node.current_language = required_model
                    engine_adapter.node.current_model_name = required_model
                    print(f"🔄 Updated current_language to '{required_model}' (cached)")
            
            # Process each segment in this language group
            for segment_data in lang_segments:
                original_idx, character, segment_text, segment_lang, original_character = segment_data
                segment_display_idx = original_idx + 1  # For display (1-based)
                
                # Get character voice or fallback to main
                if self.engine_type == "f5tts":
                    char_audio, char_text = character_mapping.get(character, (None, None))
                    if not char_audio or not char_text:
                        char_audio = params.get("main_audio_reference")
                        char_text = params.get("main_text_reference") 
                else:  # chatterbox
                    # CRITICAL FIX: Check if original character (before alias resolution) was "narrator"
                    # to detect language-only tags like [de:] that should use main voice
                    main_ref = params.get("main_audio_reference")
                    
                    # If the original tag was language-only (defaulted to "narrator") and we have main ref,
                    # prioritize main reference over any character alias mapping
                    should_use_main_ref = (original_character == "narrator" and main_ref)
                    
                    if should_use_main_ref:
                        # Language-only tag like [de:] - use main narrator voice (user's selected voice)
                        print(f"✅ Using main narrator voice (user-selected) for language-only tag in {segment_lang}")
                        char_audio = main_ref
                    else:
                        # Explicit character tag like [de:Alice] - use character voice
                        char_audio_tuple = character_mapping.get(character, (None, None))
                        if char_audio_tuple[0]:
                            char_audio = char_audio_tuple[0]  # Only get the audio path
                        else:
                            char_audio = main_ref
                
                # Show generation message with character and language info
                # Check if we're using main voice for narrator (language-only tags)
                is_using_main_voice = (self.engine_type == "chatterbox" and 
                                     original_character == "narrator" and 
                                     params.get("main_audio_reference"))
                
                if is_using_main_voice:
                    # Language-only tag using main voice
                    if segment_lang != 'en':
                        print(f"🎤 Generating {self.engine_type.title()} segment {segment_display_idx} using main voice in {segment_lang}...")
                    else:
                        print(f"🎤 Generating {self.engine_type.title()} segment {segment_display_idx} using main voice...")
                elif character == "narrator":
                    if segment_lang != 'en':
                        print(f"🎤 Generating {self.engine_type.title()} segment {segment_display_idx} in {segment_lang}...")
                    else:
                        print(f"🎤 Generating {self.engine_type.title()} segment {segment_display_idx}...")
                else:
                    if segment_lang != 'en':
                        print(f"🎭 Generating {self.engine_type.title()} segment {segment_display_idx} using '{character}' in {segment_lang}")
                    else:
                        print(f"🎭 Generating {self.engine_type.title()} segment {segment_display_idx} using '{character}'")
                
                # Show what model is actually being used for generation (for verification)
                current_model = getattr(engine_adapter.node, 'current_language', 'unknown')
                print(f"🔧 ACTUAL MODEL: Generating segment {segment_display_idx} using '{current_model}' model")
                
                # Show the final text that will go to the TTS model
                print(f"🔤 Final text to {self.engine_type.upper()} via multilingual engine ({character}): '{segment_text}'")
                
                # CRITICAL FIX: For language-only tags, use "narrator" as character for cache consistency
                cache_character = character
                if self.engine_type == "chatterbox" and original_character == "narrator":
                    cache_character = "narrator"
                
                # CRITICAL FIX: Update current_language in params to match loaded model
                # The multilingual engine loads models but the adapter needs the updated language
                updated_params = params.copy()
                updated_params['current_language'] = getattr(engine_adapter.node, 'current_language', segment_lang)
                
                # CRITICAL FIX: Handle pause tags within character segments
                # This ensures pause changes don't invalidate cache for text content
                from utils.text.pause_processor import PauseTagProcessor
                if PauseTagProcessor.has_pause_tags(segment_text):
                    # Process pause tags for this character segment
                    if self.engine_type == "f5tts":
                        segment_audio = engine_adapter.generate_segment_audio(
                            text=segment_text,  # Let adapter handle pause tags internally
                            char_audio=char_audio,
                            char_text=char_text,
                            character=cache_character,
                            enable_pause_tags=True,  # Enable pause processing in adapter
                            **updated_params
                        )
                    else:  # chatterbox
                        segment_audio = engine_adapter.generate_segment_audio(
                            text=segment_text,  # Let adapter handle pause tags internally
                            char_audio=char_audio,
                            character=cache_character,
                            enable_pause_tags=True,  # Enable pause processing in adapter
                            **updated_params
                        )
                else:
                    # No pause tags, standard generation
                    if self.engine_type == "f5tts":
                        segment_audio = engine_adapter.generate_segment_audio(
                            text=segment_text,
                            char_audio=char_audio,
                            char_text=char_text,
                            character=cache_character,
                            **updated_params
                        )
                    else:  # chatterbox
                        segment_audio = engine_adapter.generate_segment_audio(
                            text=segment_text,
                            char_audio=char_audio,
                            character=cache_character,
                            **updated_params
                        )
                
                # Calculate duration
                duration = self._get_audio_duration(segment_audio)
                
                # Store result with original index for proper ordering
                all_audio_segments.append(AudioSegmentResult(
                    audio=segment_audio,
                    duration=duration,
                    character=character,
                    text=segment_text,
                    language=segment_lang,
                    original_index=original_idx
                ))
        
        # 7. Reorder segments back to original order and combine
        all_audio_segments.sort(key=lambda x: x.original_index)
        ordered_audio = [seg.audio for seg in all_audio_segments]
        combined_audio = torch.cat(ordered_audio, dim=1) if ordered_audio else torch.zeros(1, 0)
        
        # 8. Calculate total duration and create info message
        total_duration = sum(seg.duration for seg in all_audio_segments)
        info_message = self._generate_info_message(
            total_duration, len(all_audio_segments), characters, languages, 
            has_multiple_characters, has_multiple_languages
        )
        
        return MultilingualResult(
            audio=combined_audio,
            total_duration=total_duration,
            segments=all_audio_segments,
            languages_used=languages,
            characters_used=characters,
            info_message=info_message
        )
    
    def _group_segments_by_language(self, character_segments_with_lang: List[Tuple[str, str, str]]) -> Dict[str, List[Tuple[int, str, str, str]]]:
        """Group character segments by language for efficient processing."""
        language_groups = {}
        for original_idx, (character, segment_text, segment_lang) in enumerate(character_segments_with_lang):
            if segment_lang not in language_groups:
                language_groups[segment_lang] = []
            language_groups[segment_lang].append((original_idx, character, segment_text, segment_lang))
        return language_groups
    
    def _group_segments_by_language_with_original(self, segments: List) -> Dict[str, List[Tuple[int, str, str, str, str]]]:
        """Group segments by language for efficient processing, preserving original character info."""
        language_groups = {}
        for original_idx, segment in enumerate(segments):
            segment_lang = segment.language
            if segment_lang not in language_groups:
                language_groups[segment_lang] = []
            language_groups[segment_lang].append((
                original_idx, 
                segment.character, 
                segment.text, 
                segment_lang, 
                segment.original_character or segment.character
            ))
        return language_groups
    
    def _analyze_cache_coverage(self, language_groups: Dict, character_mapping: Dict, 
                               engine_adapter, **params) -> Dict[str, Dict[str, Any]]:
        """Analyze cache coverage for each language group to optimize model loading."""
        cache_info = {}
        
        # Only check cache if caching is enabled
        enable_cache = params.get("enable_audio_cache", True)
        if not enable_cache:
            for lang_code, lang_segments in language_groups.items():
                cache_info[lang_code] = {
                    "all_cached": False,
                    "segments_count": len(lang_segments)
                }
            return cache_info
        
        for lang_code, lang_segments in language_groups.items():
            cached_segments = 0
            total_segments = len(lang_segments)
            
            for segment_data in lang_segments:
                original_idx, character, segment_text, segment_lang = segment_data[:4]  # Take only first 4 elements
                # Check if this specific segment is cached
                if self._is_segment_cached(character, segment_text, segment_lang, character_mapping, **params):
                    cached_segments += 1
            
            all_cached = cached_segments == total_segments
            cache_info[lang_code] = {
                "all_cached": all_cached,
                "cached_segments": cached_segments,
                "segments_count": total_segments
            }
            
            if all_cached:
                print(f"💾 Cache optimization: All {total_segments} segments in '{lang_code}' are cached")
        
        return cache_info
    
    def _is_segment_cached(self, character: str, text: str, language: str, character_mapping: Dict, **params) -> bool:
        """Check if a specific segment is cached."""
        try:
            # Import cache function
            from utils.audio.cache import create_cache_function
            
            # Get character voice information
            if self.engine_type == "f5tts":
                char_audio, char_text = character_mapping.get(character, (None, None))
                if not char_audio or not char_text:
                    char_audio = params.get("main_audio_reference")
                    char_text = params.get("main_text_reference", "")
                # CRITICAL FIX: Use main narrator's stable component instead of generic fallback
                main_stable_component = params.get("stable_audio_component", "main_reference")
                audio_component = char_audio or main_stable_component
                ref_text_component = char_text or ""
            else:  # chatterbox
                char_audio_tuple = character_mapping.get(character, (None, None))
                char_audio = char_audio_tuple[0] if char_audio_tuple[0] else params.get("main_audio_reference")
                # CRITICAL FIX: Use main narrator's stable component instead of generic fallback
                main_stable_component = params.get("stable_audio_component", "main_reference")
                audio_component = char_audio or main_stable_component
                ref_text_component = ""
            
            # CRITICAL FIX: For language-only tags using main narrator voice, cache as "narrator"
            # This prevents cache pollution when narrator voice changes
            cache_character = character
            if audio_component == main_stable_component:
                # This character is using main narrator voice (language-only tag)
                cache_character = "narrator"
            
            # Create cache function to check if segment exists
            cache_fn = create_cache_function(
                text_content=text,
                audio_component=str(audio_component),
                ref_text_component=ref_text_component,
                character=cache_character,
                language=language,
                model_name=params.get("model", "default"),
                temperature=params.get("temperature", 0.8),
                speed=params.get("speed", 1.0) if self.engine_type == "f5tts" else None,
                nfe_step=params.get("nfe_step", 32) if self.engine_type == "f5tts" else None,
                cfg_strength=params.get("cfg_strength", 2.0) if self.engine_type == "f5tts" else None,
                exaggeration=params.get("exaggeration", 0.5) if self.engine_type == "chatterbox" else None,
                cfg_weight=params.get("cfg_weight", 0.5) if self.engine_type == "chatterbox" else None
            )
            
            # Check cache - if it returns data, segment is cached
            cached_data = cache_fn(text, audio_result=None)
            return cached_data is not None
            
        except Exception as e:
            # If cache checking fails, assume not cached
            return False
    
    def _get_audio_duration(self, audio_tensor: torch.Tensor) -> float:
        """Calculate audio duration in seconds."""
        if audio_tensor.dim() == 1:
            num_samples = audio_tensor.shape[0]
        elif audio_tensor.dim() == 2:
            num_samples = audio_tensor.shape[1]  # Assume shape (channels, samples)
        else:
            num_samples = audio_tensor.numel()
        
        return num_samples / self.sample_rate
    
    def _generate_info_message(self, total_duration: float, num_segments: int, 
                             characters: List[str], languages: List[str],
                             has_multiple_characters: bool, has_multiple_languages: bool) -> str:
        """Generate descriptive info message about the generation."""
        character_info = f"characters: {', '.join(characters)}" if has_multiple_characters else "narrator"
        language_info = f" across {len(languages)} languages ({', '.join(languages)})" if has_multiple_languages else ""
        
        return f"Generated {total_duration:.1f}s audio from {num_segments} segments using {character_info}{language_info} ({self.engine_type.title()} models)"
    
    def is_multilingual_or_multicharacter(self, text: str) -> bool:
        """Quick check if text needs multilingual processing."""
        character_segments_with_lang = character_parser.split_by_character_with_language(text)
        characters = list(set(char for char, _, _ in character_segments_with_lang))
        languages = list(set(lang for _, _, lang in character_segments_with_lang))
        
        has_multiple_characters = len(characters) > 1 or (len(characters) == 1 and characters[0] != "narrator")
        has_multiple_languages = len(languages) > 1
        
        return has_multiple_characters or has_multiple_languages