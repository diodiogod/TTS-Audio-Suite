"""
Base Character Parser - Core character tag parsing functionality

Extracted from the original 930-line character_parser.py to maintain modularity.
Contains the core CharacterParser class and CharacterSegment dataclass.
"""

import re
import os
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path

from .types import CharacterSegment
from .language_resolver import LanguageResolver
from .segment_processor import SegmentProcessor
from .validation import ValidationMixin


class CharacterParser(ValidationMixin):
    """
    Universal character parsing system for both F5TTS and ChatterBox TTS.
    
    Features:
    - Parse [CharacterName] and [language:CharacterName] tags from text
    - Split text into character-specific segments with language awareness
    - Robust fallback system for missing characters
    - Support for both single text and SRT subtitle processing
    - Compatible with voice folder structure
    - Language-aware character switching
    """
    
    # Regex pattern for character tags: [CharacterName] or [language:CharacterName]
    # Excludes: pause tags, standalone Italian tags [it]/[italian] (but allows [it:]/[italian:])
    CHARACTER_TAG_PATTERN = re.compile(r'\[(?!(?:pause|wait|stop):)(?!(?:it|IT|italian|Italian)\])([^\]]+)\]')
    
    def __init__(self, default_character: str = "narrator", default_language: Optional[str] = None):
        """
        Initialize character parser.
        
        Args:
            default_character: Default character name for untagged text
            default_language: Default language for characters without explicit language
        """
        self.default_character = default_character
        self.default_language = default_language or "en"
        self.available_characters = set()
        self.character_fallbacks = {}
        self.character_language_defaults = {}
        
        # Initialize helper components
        self.language_resolver = LanguageResolver(self.default_language)
        self.segment_processor = SegmentProcessor(self.CHARACTER_TAG_PATTERN, self.default_character)
        
        # Cache for character language resolution to prevent duplicate logging
        self._character_language_cache = {}
        self._logged_characters = set()
        self._logged_character_warnings = set()
    
    def set_available_characters(self, characters: List[str]):
        """
        Set list of available character voices.
        
        Args:
            characters: List of character names that have voice references
        """
        self.available_characters = set(char.lower() for char in characters)
    
    def add_character_fallback(self, character: str, fallback: str):
        """
        Add a fallback mapping for a character.
        
        Args:
            character: Character name that might not exist
            fallback: Character name to use as fallback
        """
        self.character_fallbacks[character.lower()] = fallback.lower()
    
    def set_character_language_default(self, character: str, language: str):
        """
        Set default language for a character.
        
        Args:
            character: Character name
            language: Default language code (e.g., 'en', 'de', 'fr')
        """
        self.character_language_defaults[character.lower()] = language.lower()
    
    def parse_language_character_tag(self, tag_content: str) -> Tuple[Optional[str], str]:
        """
        Parse character tag content to extract language and character.
        
        Args:
            tag_content: Content inside character brackets (e.g., "Alice" or "de:Alice")
            
        Returns:
            Tuple of (language, character_name) where language can be None
        """
        return self.language_resolver.parse_language_character_tag(tag_content)
    
    def resolve_character_language(self, character: str, explicit_language: Optional[str] = None) -> str:
        """
        Resolve the language to use for a character with caching to prevent log spam.
        
        Priority:
        1. Explicitly provided language (from [lang:character] tag)
        2. Character's default language (from alias system)
        3. Global default language
        
        Args:
            character: Character name
            explicit_language: Language explicitly specified in tag
            
        Returns:
            Language code to use (normalized, no local: prefix)
        """
        return self.language_resolver.resolve_character_language(
            character, explicit_language, self.character_language_defaults,
            self._character_language_cache, self._logged_characters
        )
    
    def reset_session_cache(self):
        """Reset caches for a new parsing session to allow fresh logging."""
        self._character_language_cache.clear()
        self._logged_characters.clear()
        self._logged_character_warnings.clear()
        self.language_resolver.reset_cache()
    
    def normalize_character_name(self, character_name: str, skip_narrator_alias: bool = False) -> str:
        """
        Normalize character name and apply alias resolution and fallback if needed.
        
        Args:
            character_name: Raw character name from tag
            skip_narrator_alias: If True, skip alias resolution for "narrator" character
            
        Returns:
            Normalized character name or fallback
        """
        # Clean and normalize
        normalized = character_name.strip().lower()
        
        # Remove common punctuation from character names
        normalized = re.sub(r'[：:,，]', '', normalized)
        
        # First, try to resolve through alias system
        # Skip alias resolution for "narrator" when it comes from language-only tags
        # This preserves user's narrator voice priority (opt_narrator > dropdown > character map)
        if not (skip_narrator_alias and normalized == "narrator"):
            try:
                from utils.voice.discovery import voice_discovery
                resolved = voice_discovery.resolve_character_alias(normalized)
                if resolved != normalized:
                    normalized = resolved
            except Exception as e:
                # If alias resolution fails, continue with original name
                pass
        
        # Check if character is available
        if normalized in self.available_characters:
            return normalized
        
        # Check fallback mapping
        if normalized in self.character_fallbacks:
            fallback = self.character_fallbacks[normalized]
            print(f"🔄 Character Parser: '{character_name}' → '{fallback}' (fallback)")
            return fallback
        
        # Default fallback - only log once per character per session
        if character_name not in self._logged_character_warnings:
            print(f"⚠️ Character Parser: Character '{character_name}' not found, using '{self.default_character}'")
            self._logged_character_warnings.add(character_name)
        return self.default_character
    
    def parse_text_segments(self, text: str) -> List[CharacterSegment]:
        """
        Parse text into character-specific segments with proper line-by-line processing.
        
        Args:
            text: Input text with [Character] tags
            
        Returns:
            List of CharacterSegment objects
        """
        return self.segment_processor.parse_text_segments(
            text, self.language_resolver, self
        )
    
    def parse_character_mapping(self, text: str) -> Dict[str, List[str]]:
        """
        Parse text and return character-to-text mapping.
        
        Args:
            text: Input text with [Character] tags
            
        Returns:
            Dictionary mapping character names to list of text segments
        """
        segments = self.parse_text_segments(text)
        character_mapping = {}
        
        for segment in segments:
            if segment.character not in character_mapping:
                character_mapping[segment.character] = []
            character_mapping[segment.character].append(segment.text)
        
        return character_mapping
    
    def get_character_list(self, text: str) -> List[str]:
        """
        Get list of unique characters used in text.
        
        Args:
            text: Input text with [Character] tags
            
        Returns:
            List of unique character names used
        """
        segments = self.parse_text_segments(text)
        return list(set(segment.character for segment in segments))
    
    def remove_character_tags(self, text: str) -> str:
        """
        Remove all character tags from text, leaving only the speech content.
        
        Args:
            text: Input text with [Character] tags
            
        Returns:
            Text with character tags removed
        """
        return self.CHARACTER_TAG_PATTERN.sub('', text).strip()
    
    def split_by_character(self, text: str, include_language: bool = False) -> Union[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
        """
        Split text by character, returning (character, text, language) tuples.
        This is the main method used by TTS nodes.
        
        Args:
            text: Input text with [Character] tags
            include_language: If True, returns (character, text, language) tuples
            
        Returns:
            List of (character_name, text_content, language) tuples if include_language=True
            List of (character_name, text_content) tuples if include_language=False (backward compatibility)
        """
        segments = self.parse_text_segments(text)
        
        if include_language:
            result = [(segment.character, segment.text, segment.language) for segment in segments]
        else:
            # Backward compatibility: return old tuple format
            result = [(segment.character, segment.text) for segment in segments]
        
        return result
    
    def split_by_character_with_language(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Split text by character, returning (character, text, language) tuples.
        Convenience method that always includes language information.
        
        Args:
            text: Input text with [Character] or [language:Character] tags
            
        Returns:
            List of (character_name, text_content, language_code) tuples
        """
        segments = self.parse_text_segments(text)
        return [(segment.character, segment.text, segment.language) for segment in segments]
    
    def split_by_character_with_language_and_explicit_flag(self, text: str) -> List[Tuple[str, str, str, bool]]:
        """
        Split text by character, returning (character, text, language, explicit_language) tuples.
        Extended method that includes explicit language flag for Italian prefix handling.
        
        Args:
            text: Input text with [Character] or [language:Character] tags
            
        Returns:
            List of (character_name, text_content, language_code, explicit_language) tuples
        """
        segments = self.parse_text_segments(text)
        
        # Apply Italian prefix directly to segments before returning
        processed_segments = []
        for segment in segments:
            processed_text = self.apply_italian_prefix_if_needed(
                segment.text, segment.character, segment.language, segment.explicit_language
            )
            processed_segments.append((segment.character, processed_text, segment.language, segment.explicit_language))
        
        return processed_segments
    
    def split_by_character_with_emotions(self, text: str) -> List[Tuple[str, str, Optional[str], Optional[str]]]:
        """
        Split text by character with emotion support for advanced TTS engines.
        
        NEW METHOD - Engines must explicitly call this to get emotion data.
        Backward compatibility: existing engines continue using split_by_character() or
        split_by_character_with_language() and will never see emotion data.
        
        Args:
            text: Input text with [Character], [language:Character], or [Character:emotion] tags
            
        Returns:
            List of (character_name, text_content, language_code, emotion_reference) tuples
        """
        segments = self.parse_text_segments(text)
        return [(segment.character, segment.text, segment.language, segment.emotion) for segment in segments]

    def get_emotion_voice_path(self, emotion_reference: str) -> Optional[str]:
        """
        Resolve emotion reference to voice file path using same system as character resolution.

        Args:
            emotion_reference: Emotion reference name (e.g., "Bob", "angry_alice")

        Returns:
            File path to emotion voice audio, or None if not found
        """
        if not emotion_reference:
            return None

        try:
            # Resolve emotion reference using same alias system as characters
            dummy_segments = self.split_by_character_with_emotions(f"[{emotion_reference}] dummy")
            if not dummy_segments:
                return None

            resolved_character = dummy_segments[0][0]  # Get resolved character name

            # Use voice discovery to get the audio path for resolved character
            from utils.voice.discovery import get_character_mapping
            char_mapping = get_character_mapping([resolved_character], "audio_only")
            audio_path, _ = char_mapping.get(resolved_character, (None, None))

            return audio_path if audio_path and os.path.exists(audio_path) else None

        except Exception as e:
            print(f"⚠️ Failed to resolve emotion reference '{emotion_reference}': {e}")
            return None
    
    def set_engine_aware_default_language(self, model_or_language: str, engine_type: str):
        """
        Set default language based on engine model or language code.
        Delegates to the language resolver.
        
        Args:
            model_or_language: Either a model name (F5-TTS) or language code (ChatterBox)
            engine_type: Engine type ("f5tts" or "chatterbox")
        """
        self.language_resolver.set_engine_aware_default_language(model_or_language, engine_type)
        # Also update our own default_language for consistency
        self.default_language = self.language_resolver.default_language
    
    def apply_italian_prefix_if_needed(self, text: str, character: str, language: str, explicit_language: bool) -> str:
        """
        Apply [it] prefix to text if Italian language is detected.
        
        The Italian ChatterBox model requires [it] prefix for Italian text but only when:
        - User uses [it:Alice] or [italian:Bob] syntax (explicit_language=True with language='it')
        - User uses [it:] or [italian:] language switching (explicit_language=True with language='it')
        - Character is mapped as Italian in alias system (language='it' from character defaults)
        
        NOT when just using Italian model globally (it's bilingual Italian/English).
        
        Args:
            text: Text content to potentially prefix (should be clean text without character tags)
            character: Character name
            language: Resolved language code
            explicit_language: Whether language was explicitly specified in tag
            
        Returns:
            Text with [it] prefix added if Italian requirements are met
        """
        # Only apply Italian prefix if language is Italian
        if language != 'it':
            return text
        
        # Apply prefix if:
        # 1. Language was explicitly set via [it:Alice] or [italian:] syntax
        # 2. Character has Italian as default language from alias system (not explicit but character-mapped)
        should_apply_prefix = (
            explicit_language or  # User explicitly used [it:] or [italian:] syntax
            (not explicit_language and language == 'it')  # Character mapped to Italian via alias system
        )
        
        if should_apply_prefix:
            # Avoid double-prefixing - check if [it] is already at the start
            if not text.strip().startswith('[it]'):
                return f"[it] {text.strip()}"
        
        return text


# Global instance for use across nodes
character_parser = CharacterParser()


def parse_character_text(text: str, available_characters: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Convenience function to parse character text.

    Args:
        text: Input text with [Character] tags
        available_characters: Optional list of available character voices

    Returns:
        List of (character_name, text_content) tuples
    """
    if available_characters:
        character_parser.set_available_characters(available_characters)
    else:
        # Clear any previous available_characters to allow auto-discovery
        character_parser.available_characters = set()
        # Auto-discover characters if not provided
        try:
            from utils.voice.discovery import get_available_characters
            auto_chars = get_available_characters()
            if auto_chars:
                character_parser.set_available_characters(list(auto_chars))
        except Exception as e:
            print(f"⚠️ Character Parser: Auto-discovery failed: {e}")

    # Parse segments and filter out parameter-only tags (which have empty text)
    segments = character_parser.parse_text_segments(text)
    result = [(seg.character, seg.text) for seg in segments if seg.text.strip()]

    return result