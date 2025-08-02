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
        language_groups = self._group_segments_by_language(character_segments_with_lang)
        
        # 4. Get character mapping for all characters
        character_mapping = get_character_mapping(characters, engine_type=self.engine_type)
        
        # 5. Check cache optimization opportunities
        cache_info = self._analyze_cache_coverage(language_groups, character_mapping, engine_adapter, **params)
        
        # 6. Process each language group with smart model loading
        all_audio_segments = []
        base_model_loaded = False
        
        for lang_code, lang_segments in language_groups.items():
            # Get required model for this language
            required_model = engine_adapter.get_model_for_language(lang_code, params.get("model", "default"))
            
            # Check if all segments in this language group are cached
            if cache_info.get(lang_code, {}).get("all_cached", False):
                print(f"💾 Skipping model load for language '{lang_code}' - all {len(lang_segments)} segments cached")
            else:
                # Load base model if this is the first time we need to generate anything
                if not base_model_loaded:
                    print(f"🔄 Loading base {self.engine_type.title()} model for generation")
                    engine_adapter.load_base_model(params.get("model", "default"), params.get("device", "auto"))
                    base_model_loaded = True
                
                print(f"🌍 Loading {self.engine_type.title()} model '{required_model}' for language '{lang_code}' ({len(lang_segments)} segments)")
                try:
                    engine_adapter.load_language_model(required_model, params.get("device", "auto"))
                except Exception as e:
                    print(f"⚠️ Failed to load model '{required_model}' for language '{lang_code}': {e}")
                    print(f"🔄 Falling back to default model")
                    engine_adapter.load_base_model(params.get("model", "default"), params.get("device", "auto"))
            
            # Process each segment in this language group
            for original_idx, character, segment_text, segment_lang in lang_segments:
                segment_display_idx = original_idx + 1  # For display (1-based)
                
                # Get character voice or fallback to main
                if self.engine_type == "f5tts":
                    char_audio, char_text = character_mapping.get(character, (None, None))
                    if not char_audio or not char_text:
                        char_audio = params.get("main_audio_reference")
                        char_text = params.get("main_text_reference") 
                else:  # chatterbox
                    char_audio_tuple = character_mapping.get(character, (None, None))
                    if char_audio_tuple[0]:
                        char_audio = char_audio_tuple[0]  # Only get the audio path
                    else:
                        char_audio = params.get("main_audio_reference")
                
                # Show generation message with character and language info
                if character == "narrator":
                    if segment_lang != 'en':
                        print(f"🎤 Generating {self.engine_type.title()} segment {segment_display_idx} in {segment_lang}...")
                    else:
                        print(f"🎤 Generating {self.engine_type.title()} segment {segment_display_idx}...")
                else:
                    if segment_lang != 'en':
                        print(f"🎭 Generating {self.engine_type.title()} segment {segment_display_idx} using '{character}' in {segment_lang}")
                    else:
                        print(f"🎭 Generating {self.engine_type.title()} segment {segment_display_idx} using '{character}'")
                
                # Generate audio using engine adapter
                if self.engine_type == "f5tts":
                    segment_audio = engine_adapter.generate_segment_audio(
                        text=segment_text,
                        char_audio=char_audio,
                        char_text=char_text,
                        character=character,
                        **params
                    )
                else:  # chatterbox
                    segment_audio = engine_adapter.generate_segment_audio(
                        text=segment_text,
                        char_audio=char_audio,
                        character=character,
                        **params
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
    
    def _analyze_cache_coverage(self, language_groups: Dict, character_mapping: Dict, 
                               engine_adapter, **params) -> Dict[str, Dict[str, Any]]:
        """Analyze cache coverage for each language group."""
        cache_info = {}
        
        for lang_code, lang_segments in language_groups.items():
            # For now, assume no caching analysis (can be enhanced later)
            cache_info[lang_code] = {
                "all_cached": False,
                "segments_count": len(lang_segments)
            }
        
        return cache_info
    
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