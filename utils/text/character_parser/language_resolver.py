"""
Language Resolver - Language resolution and mapping functionality

Extracted from character parser to handle language detection, resolution, and caching.
Supports the flexible tag syntax for emotion parsing.
"""

import re
from typing import Optional, Tuple, Dict, Set
from utils.models.language_mapper import resolve_language_alias, LANGUAGE_ALIASES


class LanguageResolver:
    """
    Handles language resolution and mapping for character tags.
    
    Supports flexible tag parsing with emotion extensions:
    - [de:] → language="de", character="narrator"
    - [de:Alice] → language="de", character="Alice"  
    - [Alice] → language=None, character="Alice"
    - [Alice:angry_bob] → language=None, character="Alice", emotion="angry_bob"
    - [de:Alice:angry_bob] → language="de", character="Alice", emotion="angry_bob"
    """
    
    # Regex to parse language:character format (supports flexible language names)
    LANGUAGE_CHARACTER_PATTERN = re.compile(r'^([a-zA-Z0-9\-_À-ÿ\s]+):(.*)$')
    
    def __init__(self, default_language: str = "en"):
        """
        Initialize language resolver.
        
        Args:
            default_language: Default language code
        """
        self.default_language = default_language
        self._known_languages = self._build_known_languages()
        
    def _build_known_languages(self) -> Set[str]:
        """Build set of known language codes from language aliases."""
        known = set()
        
        # Add all language aliases (keys) and canonical codes (values)
        known.update(LANGUAGE_ALIASES.keys())
        known.update(LANGUAGE_ALIASES.values())
        
        return {lang.lower() for lang in known}
    
    def is_known_language_code(self, text: str) -> bool:
        """
        Check if text is a known language code.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is a recognized language code
        """
        return text.lower().strip() in self._known_languages
    
    def parse_language_character_tag(self, tag_content: str) -> Tuple[Optional[str], str]:
        """
        Parse character tag content to extract language and character.
        
        Supports flexible syntax:
        - "Alice" → (None, "Alice")
        - "de:Alice" → ("de", "Alice") 
        - "de:" → ("de", "narrator")
        - "Alice:angry_bob" → (None, "Alice:angry_bob") if Alice not a language
        
        Args:
            tag_content: Content inside character brackets
            
        Returns:
            Tuple of (language, character_name) where language can be None
        """
        # Check if it's in language:character format
        match = self.LANGUAGE_CHARACTER_PATTERN.match(tag_content.strip())
        if match:
            potential_language = match.group(1).strip()
            character_part = match.group(2).strip()
            
            # Check if the first part is actually a language code
            if self.is_known_language_code(potential_language):
                # It's a language:character format
                language = resolve_language_alias(potential_language)
                character = character_part if character_part else "narrator"
                return language, character
            else:
                # First part is not a language, treat entire string as character
                # This handles cases like "Alice:angry_bob" where Alice is not a language
                return None, tag_content.strip()
        else:
            # Just a character name, no colons
            return None, tag_content.strip()
    
    def parse_flexible_tag(self, tag_content: str) -> Dict[str, Optional[str]]:
        """
        Parse flexible character tag with emotion support for IndexTTS-2.

        Handles all combinations:
        - [Alice] → character="Alice"
        - [de] → language="de", character="narrator"
        - [de:Alice] → language="de", character="Alice"
        - [Alice:angry_bob] → character="Alice", emotion="angry_bob"
        - [de:Alice:angry_bob] → language="de", character="Alice", emotion="angry_bob"

        NOTE: This does NOT handle parameters. Use parse_tag_with_parameters for pipe-separated params.

        Args:
            tag_content: Content inside brackets

        Returns:
            Dictionary with keys: language, character, emotion
        """
        parts = [part.strip() for part in tag_content.split(':')]
        result = {'language': None, 'character': None, 'emotion': None}

        if len(parts) == 1:
            # [Alice] or [de]
            if self.is_known_language_code(parts[0]):
                result['language'] = resolve_language_alias(parts[0])
                result['character'] = "narrator"
            else:
                result['character'] = parts[0] or "narrator"

        elif len(parts) == 2:
            # [de:Alice] or [Alice:angry_bob]
            if self.is_known_language_code(parts[0]):
                result['language'] = resolve_language_alias(parts[0])
                result['character'] = parts[1] or "narrator"
            else:
                result['character'] = parts[0] or "narrator"
                result['emotion'] = parts[1] if parts[1] else None

        elif len(parts) == 3:
            # [de:Alice:angry_bob]
            result['language'] = resolve_language_alias(parts[0]) if parts[0] else None
            result['character'] = parts[1] or "narrator"
            result['emotion'] = parts[2] if parts[2] else None

        else:
            # More than 3 parts - treat as character name with colons
            result['character'] = tag_content or "narrator"

        return result

    def parse_tag_with_parameters(self, tag_content: str) -> Dict[str, any]:
        """
        Parse character tag with pipe-separated parameters.

        Handles all combinations:
        - [Alice] → character="Alice", parameters={}
        - [Alice|seed:42] → character="Alice", parameters={"seed": 42}
        - [de:Alice|seed:42|temp:0.5] → character="Alice", language="de", parameters={"seed": 42, "temperature": 0.5}
        - [seed:42|de:Alice] → Order-independent parameter extraction

        Args:
            tag_content: Content inside brackets with potential pipe-separated parameters

        Returns:
            Dictionary with keys: language, character, emotion, parameters
        """
        # Import here to avoid circular imports
        from utils.text.segment_parameters import SegmentParameterParser, ParameterValidator

        # First split by pipes to separate character/language info from parameters
        tag_segments, parameters = SegmentParameterParser.parse_tag_segments(tag_content)

        # Now parse the character/language segment using flexible tag parser
        reconstructed_tag = SegmentParameterParser.reconstruct_character_tag(tag_segments)

        # Parse the reconstructed tag (without parameters) using the regular flexible tag parser
        result = self.parse_flexible_tag(reconstructed_tag) if reconstructed_tag else {
            'language': None,
            'character': 'narrator',
            'emotion': None
        }

        # Add parameters to result
        result['parameters'] = parameters

        return result
    
    def resolve_character_language(self, character: str, explicit_language: Optional[str], 
                                 character_language_defaults: Dict[str, str],
                                 character_language_cache: Dict[str, str],
                                 logged_characters: Set[str]) -> str:
        """
        Resolve the language to use for a character with caching.
        
        Priority:
        1. Explicitly provided language (from [lang:character] tag)
        2. Character's default language (from alias system)
        3. Global default language
        
        Args:
            character: Character name
            explicit_language: Language explicitly specified in tag
            character_language_defaults: Character language mapping
            character_language_cache: Cache for resolved languages
            logged_characters: Set of characters already logged
            
        Returns:
            Language code to use (normalized, no local: prefix)
        """
        if explicit_language:
            # Normalize explicit language immediately
            if explicit_language.startswith("local:"):
                explicit_language = explicit_language[6:]
            return explicit_language
        
        character_lower = character.lower()
        
        # Check cache first to avoid repeated lookups and logging
        cache_key = character_lower
        if cache_key in character_language_cache:
            return character_language_cache[cache_key]
        
        resolved_language = None
        
        # Priority 1: Character language defaults (internal cache)
        if character_lower in character_language_defaults:
            resolved_language = character_language_defaults[character_lower]
            # Normalize alias language
            if resolved_language and resolved_language.startswith("local:"):
                resolved_language = resolved_language[6:]

        # Priority 2: Check voice discovery system for language defaults
        if not resolved_language:
            try:
                from utils.voice.discovery import voice_discovery
                alias_language = voice_discovery.get_character_default_language(character_lower)
                if alias_language:
                    resolved_language = alias_language
                    # Normalize alias language from voice discovery
                    if resolved_language and resolved_language.startswith("local:"):
                        resolved_language = resolved_language[6:]
            except Exception:
                pass  # Silently handle voice discovery errors

        # Log once per character only if language was explicitly resolved (not default)
        if resolved_language and resolved_language != self.default_language:
            if character_lower not in logged_characters:
                print(f"🎭 Character '{character}' auto-switching to 🚨 alias default language '{resolved_language}'")
                logged_characters.add(character_lower)
        
        # Priority 3: Fall back to global default
        if not resolved_language:
            resolved_language = self.default_language
        
        # Normalize local: prefix for consistency
        if resolved_language and resolved_language.startswith("local:"):
            resolved_language = resolved_language[6:]
        
        # Cache the result
        character_language_cache[cache_key] = resolved_language
        return resolved_language
    
    def reset_cache(self):
        """Reset internal caches (called by main parser)."""
        # Currently no internal caches, but method provided for consistency
        pass
    
    def get_character_language_summary(self, character_language_cache: Dict[str, str]) -> str:
        """
        Get a consolidated summary of character language mappings for logging.
        
        Args:
            character_language_cache: Cache of character→language mappings
            
        Returns:
            Formatted summary string of character→language mappings
        """
        if not character_language_cache:
            return ""
        
        # Group characters by language
        lang_groups = {}
        for char, lang in character_language_cache.items():
            if lang not in lang_groups:
                lang_groups[lang] = []
            lang_groups[lang].append(char)
        
        # Format as: Alice(de), Bob(fr→en fallback), Others(en)
        summary_parts = []
        for lang, chars in sorted(lang_groups.items()):
            if len(chars) == 1:
                summary_parts.append(f"{chars[0]}({lang})")
            else:
                summary_parts.append(f"{', '.join(chars)}({lang})")
        
        return ', '.join(summary_parts)
    
    def set_engine_aware_default_language(self, model_or_language: str, engine_type: str):
        """
        Set default language based on engine model or language code.
        
        Args:
            model_or_language: Either a model name (F5-TTS) or language code (ChatterBox)
            engine_type: Engine type ("f5tts" or "chatterbox")
        """
        # ChatterBox passes language codes directly, F5-TTS passes model names
        if engine_type == "chatterbox":
            # ChatterBox gives us language names (with possible local: prefix)
            # Need to normalize to language codes
            normalized_language = self._normalize_chatterbox_language(model_or_language)
            self.default_language = normalized_language
        else:
            # F5-TTS gives us model names, need to infer language
            inferred_language = self._infer_language_from_engine_model(model_or_language, engine_type)
            if inferred_language:
                self.default_language = inferred_language
    
    def _infer_language_from_engine_model(self, model_name: str, engine_type: str) -> Optional[str]:
        """
        Infer language code from engine model name.
        
        Args:
            model_name: Model name (e.g., "F5-PT-BR", "F5TTS_Base")
            engine_type: Engine type ("f5tts" or "chatterbox")
            
        Returns:
            Inferred language code or None if can't infer
        """
        try:
            # Use the existing language mapper system
            from utils.models.language_mapper import get_language_mapper
            
            mapper = get_language_mapper(engine_type)
            mappings = mapper.get_all_mappings().get(engine_type, {})
            
            # Normalize model name for reverse lookup (remove local: prefix)
            normalized_model = model_name
            if normalized_model.startswith("local:"):
                normalized_model = normalized_model[6:]
            
            # Reverse lookup: find language code that maps to this model
            for lang_code, mapped_model in mappings.items():
                if mapped_model == normalized_model:
                    return lang_code
            
            # Fallback for base models that aren't in specific language mappings
            if engine_type == "f5tts" and any(x in normalized_model.lower() for x in ['f5tts_base', 'f5tts_v1_base', 'e2tts_base']):
                return 'en'
            elif engine_type == "chatterbox" and 'english' in normalized_model.lower():
                return 'en'
                
        except ImportError:
            # Fallback if language mapper not available
            pass
        
        return None  # Can't infer language
    
    def _normalize_chatterbox_language(self, language_input: str) -> str:
        """
        Normalize ChatterBox language input to canonical language code.
        
        Args:
            language_input: ChatterBox language (e.g., "German", "local:German", "de")
            
        Returns:
            Normalized language code (local: prefix removed for consistency)
        """
        # Remove local: prefix for consistency - model loading will still use local models
        if language_input.startswith("local:"):
            language_input = language_input[6:]
        
        # Map ChatterBox model names to language codes
        chatterbox_language_map = {
            # Legacy ChatterBox models
            "english": "en",
            "german": "de", 
            "norwegian": "no",
            "french": "fr",
            "italian": "it", 
            "russian": "ru",
            "korean": "ko",
            "japanese": "ja",
            "armenian": "hy",
            "georgian": "ka",
            
            # ChatterBox Official 23-Lang (single model name resolves to default language)
            "chatterbox official 23-lang": None,  # Will use actual language selection from UI
        }
        
        # Check if it's a ChatterBox model name
        normalized = language_input.lower()
        if normalized in chatterbox_language_map:
            mapped = chatterbox_language_map[normalized]
            if mapped is not None:
                return mapped
        
        # For ChatterBox Official 23-Lang or unknown languages, resolve using language aliases
        # This handles language names like "Arabic", "Turkish", "Chinese", etc.
        resolved = resolve_language_alias(language_input)
        return resolved