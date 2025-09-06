"""
Character Parser for ChatterBox Voice - Universal Text Processing
Handles character tag parsing for both F5TTS and ChatterBox TTS nodes
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
from utils.models.language_mapper import resolve_language_alias, LANGUAGE_ALIASES


@dataclass
class CharacterSegment:
    """Represents a single text segment with character assignment and language"""
    character: str
    text: str
    start_pos: int
    end_pos: int
    language: Optional[str] = None
    original_character: Optional[str] = None  # Original character before alias resolution
    explicit_language: bool = False  # True if language was explicitly specified in tag (e.g., [German:Bob])
    
    def __str__(self) -> str:
        lang_info = f", lang='{self.language}'" if self.language else ""
        orig_info = f", orig='{self.original_character}'" if self.original_character and self.original_character != self.character else ""
        return f"CharacterSegment(character='{self.character}'{orig_info}{lang_info}, text='{self.text[:50]}...', pos={self.start_pos}-{self.end_pos})"


class CharacterParser:
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
    
    # Regex to parse language:character format (supports flexible language names)
    LANGUAGE_CHARACTER_PATTERN = re.compile(r'^([a-zA-Z0-9\-_À-ÿ\s]+):(.*)$')
    
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
        
        # Cache for character language resolution to prevent duplicate logging
        self._character_language_cache = {}
        self._logged_characters = set()
        self._logged_character_warnings = set()
        
        # Note: Using centralized language alias system from utils.models.language_mapper
    
    def resolve_language_alias(self, language_input: str) -> str:
        """
        Resolve language alias to canonical language code using centralized system.
        
        Args:
            language_input: User input language (e.g., "German", "brasil", "pt-BR")
            
        Returns:
            Canonical language code (e.g., "de", "pt-br")
        """
        return resolve_language_alias(language_input)
    
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
        # Check if it's in language:character format
        match = self.LANGUAGE_CHARACTER_PATTERN.match(tag_content.strip())
        if match:
            raw_language = match.group(1)
            character = match.group(2).strip()
            # Resolve language alias to canonical form
            language = self.resolve_language_alias(raw_language)
            # If character is empty (e.g., [fr:]), default to narrator
            if not character:
                character = self.default_character
            return language, character
        else:
            # Just a character name, no explicit language
            return None, tag_content.strip()
    
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
        if explicit_language:
            # Normalize explicit language immediately
            if explicit_language.startswith("local:"):
                explicit_language = explicit_language[6:]
            return explicit_language
        
        character_lower = character.lower()
        
        # Check cache first to avoid repeated lookups and logging
        cache_key = character_lower
        if cache_key in self._character_language_cache:
            return self._character_language_cache[cache_key]
        
        resolved_language = None
        
        # Priority 1: Character language defaults (internal cache)
        if character_lower in self.character_language_defaults:
            resolved_language = self.character_language_defaults[character_lower]
            # Normalize alias language
            if resolved_language and resolved_language.startswith("local:"):
                resolved_language = resolved_language[6:]
            # Only log once per character
            if character_lower not in self._logged_characters:
                print(f"🎭 Character '{character}' auto-switching to 🚨 alias default language '{resolved_language}'")
                self._logged_characters.add(character_lower)
        
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
                    # Only log once per character
                    if character_lower not in self._logged_characters:
                        print(f"🎭 Character '{character}' auto-switching to 🚨 alias default language '{resolved_language}'")
                        self._logged_characters.add(character_lower)
                # Remove spam: don't log "has no language default" for every character
            except Exception:
                pass  # Silently handle voice discovery errors
        
        # Priority 3: Fall back to global default
        if not resolved_language:
            resolved_language = self.default_language
        
        # Normalize local: prefix for consistency
        if resolved_language and resolved_language.startswith("local:"):
            resolved_language = resolved_language[6:]
        
        # Cache the result
        self._character_language_cache[cache_key] = resolved_language
        return resolved_language
    
    def reset_session_cache(self):
        """Reset caches for a new parsing session to allow fresh logging."""
        self._character_language_cache.clear()
        self._logged_characters.clear()
        self._logged_character_warnings.clear()
    
    def get_character_language_summary(self) -> str:
        """
        Get a consolidated summary of character language mappings for logging.
        
        Returns:
            Formatted summary string of character→language mappings
        """
        if not self._character_language_cache:
            return ""
        
        # Group characters by language
        lang_groups = {}
        for char, lang in self._character_language_cache.items():
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
            # print(f"🔧 Character Parser: Default language set to '{normalized_language}' for ChatterBox (from '{model_or_language}')")
        else:
            # F5-TTS gives us model names, need to infer language
            inferred_language = self._infer_language_from_engine_model(model_or_language, engine_type)
            if inferred_language:
                self.default_language = inferred_language
                # print(f"🔧 Character Parser: Default language set to '{inferred_language}' based on F5-TTS model '{model_or_language}'")
    
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
            # Use the existing language mapper system - much cleaner and flexible!
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
            # Fallback if language mapper not available - shouldn't happen but just in case
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
        resolved = self.resolve_language_alias(language_input)
        return resolved
    
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
                    # print(f"🗂️ Character Parser: '{character_name}' → '{resolved}' (alias)")
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
        segments = []
        
        # Split text by lines and process each line completely independently
        lines = text.split('\n')
        global_pos = 0
        
        for line in lines:
            line_start_pos = global_pos
            original_line = line
            line = line.strip()
            
            if not line:
                global_pos += len(original_line) + 1  # Account for empty line + newline
                continue
            
            # Each line is processed independently - no character state carries over
            line_segments = self._parse_single_line(line, line_start_pos)
            segments.extend(line_segments)
            
            global_pos += len(original_line) + 1  # +1 for newline
        
        # If no segments were created, treat entire text as default character
        if not segments and text.strip():
            segments.append(CharacterSegment(
                character=self.default_character,
                text=text.strip(),
                start_pos=0,
                end_pos=len(text),
                language=self.resolve_character_language(self.default_character),
                original_character=self.default_character,
                explicit_language=False
            ))
        
        return segments
    
    def _parse_single_line(self, line: str, line_start_pos: int) -> List[CharacterSegment]:
        """
        Parse a single line for character tags, treating it completely independently.
        
        Args:
            line: Single line of text (no newlines)
            line_start_pos: Starting position of this line in the original text
            
        Returns:
            List of CharacterSegment objects for this line only
        """
        segments = []
        current_pos = 0
        current_character = self.default_character
        current_language = self.default_language
        
        # IMPORTANT: Each line starts fresh with narrator as default
        # If the line doesn't start with a character tag, everything is narrator
        
        # Check for manual "Speaker N:" format before processing character tags
        speaker_match = self._parse_speaker_format_line(line)
        if speaker_match:
            speaker_name, speaker_text = speaker_match
            if speaker_text.strip():
                segments.append(CharacterSegment(
                    character=speaker_name,  # Use "speaker 1", "speaker 2", etc.
                    text=speaker_text.strip(),
                    start_pos=line_start_pos,
                    end_pos=line_start_pos + len(line),
                    language=self.resolve_character_language(speaker_name),
                    original_character=speaker_name,
                    explicit_language=False
                ))
            return segments
        
        # Quick check: if line doesn't contain any character tags, it's all narrator
        if not self.CHARACTER_TAG_PATTERN.search(line):
            if line.strip():
                segments.append(CharacterSegment(
                    character=self.default_character,
                    text=line.strip(),
                    start_pos=line_start_pos,
                    end_pos=line_start_pos + len(line),
                    language=self.resolve_character_language(self.default_character),
                    original_character=self.default_character,
                    explicit_language=False
                ))
            return segments
        
        # Find all character tags in this line
        for match in self.CHARACTER_TAG_PATTERN.finditer(line):
            # Add text before this tag (if any) with current character (narrator)
            before_tag = line[current_pos:match.start()].strip()
            if before_tag:
                segments.append(CharacterSegment(
                    character=current_character,
                    text=before_tag,
                    start_pos=line_start_pos + current_pos,
                    end_pos=line_start_pos + match.start(),
                    language=current_language,
                    original_character=current_character,  # Before this tag, it's already resolved
                    explicit_language=False  # Text before tags doesn't have explicit language
                ))
            
            # Parse language and character from the tag
            raw_tag_content = match.group(1)
            explicit_language, raw_character = self.parse_language_character_tag(raw_tag_content)
            
            # Update current character for text after this tag
            # IMPORTANT: Resolve language using original alias name before character normalization
            current_language = self.resolve_character_language(raw_character, explicit_language)
            original_character = raw_character  # Store original before normalization
            current_explicit_language = explicit_language is not None  # Track if language was explicit
            
            # Detect language-only tags: if the original tag had empty character part and raw_character
            # was defaulted to narrator, skip alias resolution to preserve narrator voice priority
            is_language_only_tag = (raw_character == self.default_character and 
                                   ':' in raw_tag_content and 
                                   raw_tag_content.split(':', 1)[1].strip() == '')
            
            current_character = self.normalize_character_name(raw_character, skip_narrator_alias=is_language_only_tag)
            current_pos = match.end()
        
        # Add remaining text after last tag (or entire line if no tags)
        remaining_text = line[current_pos:].strip()
        if remaining_text:
            segments.append(CharacterSegment(
                character=current_character,
                text=remaining_text,
                start_pos=line_start_pos + current_pos,
                end_pos=line_start_pos + len(line),
                language=current_language,
                original_character=original_character,
                explicit_language=current_explicit_language if 'current_explicit_language' in locals() else False
            ))
        elif not segments and line.strip():
            # Line with only tags and no text after - still need a segment for the line
            # This handles edge cases like a line that is just "[character]"
            segments.append(CharacterSegment(
                character=current_character,
                text="",
                start_pos=line_start_pos,
                end_pos=line_start_pos + len(line),
                language=current_language,
                original_character=original_character if 'original_character' in locals() else current_character,
                explicit_language=current_explicit_language if 'current_explicit_language' in locals() else False
            ))
        
        return segments
    
    def _parse_speaker_format_line(self, line: str) -> Optional[Tuple[str, str]]:
        """
        Parse a line for manual "Speaker N:" format.
        
        Args:
            line: Single line to check
            
        Returns:
            Tuple of (speaker_name, text) if found, None otherwise
        """
        import re
        # Match "Speaker N: text" (case insensitive)
        match = re.match(r'^(speaker\s*\d+)\s*:\s*(.*)$', line.strip(), re.IGNORECASE)
        if match:
            speaker_name = match.group(1).lower().strip()  # Normalize to "speaker 1", "speaker 2", etc.
            speaker_text = match.group(2)
            # Normalize speaker name format
            speaker_name = re.sub(r'\s+', ' ', speaker_name)  # "speaker  1" -> "speaker 1"
            return speaker_name, speaker_text
        return None
    
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
    
    def convert_to_language_hints_for_higgs(self, text: str) -> str:
        """
        Convert character tags with language prefixes to language hints for Higgs Audio 2.
        
        Converts [En:Alice] Hello there -> [English] Hello there
        This preserves language context for the smart Higgs model while removing character switching.
        
        Args:
            text: Input text with [language:character] tags
            
        Returns:
            Text with language hints for Higgs Audio 2
        """
        def replace_tag(match):
            tag_content = match.group(1)
            
            # Parse language:character format
            lang_match = self.LANGUAGE_CHARACTER_PATTERN.match(tag_content.strip())
            if lang_match:
                raw_language = lang_match.group(1).strip()
                # Resolve to canonical form first, then get display name
                canonical_lang = self.resolve_language_alias(raw_language)
                
                # Convert canonical codes to display names using existing alias system
                # Find a human-readable alias that maps to this canonical code
                display_name = None
                for alias, canonical in LANGUAGE_ALIASES.items():
                    if (canonical == canonical_lang and 
                        alias.isalpha() and 
                        len(alias) > 2 and  # Skip short codes like 'en', 'de'
                        alias.islower()):   # Skip uppercase variations
                        display_name = alias.title()
                        break
                
                # Fallback to canonical code if no readable alias found
                if not display_name:
                    display_name = canonical_lang.upper()
                
                return f'[{display_name}]'
            else:
                # Not a language:character tag, remove entirely
                return ''
        
        # Apply conversion
        converted_text = self.CHARACTER_TAG_PATTERN.sub(replace_tag, text)
        
        # Clean up any double spaces
        converted_text = ' '.join(converted_text.split())
        
        return converted_text
    
    def get_language_display_name(self, language_code: str) -> str:
        """
        Get readable display name for a language code using existing alias system.
        
        Args:
            language_code: Language code (e.g., 'en', 'de', 'pt-br')
            
        Returns:
            Readable language name (e.g., 'English', 'German', 'Portuguese')
        """
        # Resolve to canonical form first
        canonical_lang = self.resolve_language_alias(language_code)
        
        # Preferred display names for better readability
        preferred_names = {
            'pt-br': 'Portuguese',
            'pt-pt': 'Portuguese', 
            'zh-cn': 'Chinese',
            'zh-tw': 'Chinese'
        }
        
        if canonical_lang in preferred_names:
            return preferred_names[canonical_lang]
        
        # Find a human-readable alias that maps to this canonical code
        # Prefer explicit common language names first
        preferred_aliases = ['english', 'german', 'french', 'spanish', 'italian', 'norwegian', 'chinese', 'japanese', 'russian', 'portuguese', 'dutch', 'korean']
        
        for preferred in preferred_aliases:
            if preferred in LANGUAGE_ALIASES and LANGUAGE_ALIASES[preferred] == canonical_lang:
                return preferred.title()
        
        # Fallback: find any readable alias (avoid short codes and abbreviations)
        best_alias = None
        for alias, canonical in LANGUAGE_ALIASES.items():
            if (canonical == canonical_lang and 
                alias.isalpha() and 
                len(alias) >= 4 and  # Skip short codes like 'en', 'de', 'usa', 'eng'
                not alias.isupper()):  # Skip uppercase abbreviations
                if not best_alias or len(alias) > len(best_alias):  # Prefer longer names
                    best_alias = alias
        
        if best_alias:
            return best_alias.title()
        
        # Fallback to canonical code if no readable alias found
        return canonical_lang.upper()
    
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
        # print(f"🔍 Character Parser DEBUG: Input text: {repr(text)}")
        segments = self.parse_text_segments(text)
        
        if include_language:
            result = [(segment.character, segment.text, segment.language) for segment in segments]
        else:
            # Backward compatibility: return old tuple format
            result = [(segment.character, segment.text) for segment in segments]
        
        # print(f"🔍 Character Parser DEBUG: Parsed segments: {result}")
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
    
    def validate_character_tags(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate character tags in text and return any issues.
        
        Args:
            text: Input text with [Character] tags
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Find all character tags
        tags = self.CHARACTER_TAG_PATTERN.findall(text)
        
        # Check for empty tags
        empty_tags = [tag for tag in tags if not tag.strip()]
        if empty_tags:
            issues.append(f"Found {len(empty_tags)} empty character tag(s)")
        
        # Check for unmatched brackets
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        if open_brackets != close_brackets:
            issues.append(f"Unmatched brackets: {open_brackets} '[' vs {close_brackets} ']'")
        
        # Check for characters not in available list (if set)
        if self.available_characters:
            unknown_chars = []
            for tag in tags:
                normalized = self.normalize_character_name(tag)
                if normalized == self.default_character and tag.strip().lower() != self.default_character:
                    unknown_chars.append(tag)
            
            if unknown_chars:
                issues.append(f"Unknown characters (will use fallback): {', '.join(unknown_chars)}")
        
        return len(issues) == 0, issues
    
    def get_statistics(self, text: str) -> Dict[str, any]:
        """
        Get statistics about character usage in text.
        
        Args:
            text: Input text with [Character] tags
            
        Returns:
            Dictionary with statistics
        """
        segments = self.parse_text_segments(text)
        
        character_counts = {}
        character_lengths = {}
        
        for segment in segments:
            char = segment.character
            character_counts[char] = character_counts.get(char, 0) + 1
            character_lengths[char] = character_lengths.get(char, 0) + len(segment.text)
        
        total_chars = sum(character_counts.values())
        total_length = sum(character_lengths.values())
        
        return {
            "total_segments": len(segments),
            "unique_characters": len(character_counts),
            "character_counts": character_counts,
            "character_lengths": character_lengths,
            "total_character_switches": total_chars - 1,
            "total_text_length": total_length,
            "average_segment_length": total_length / len(segments) if segments else 0
        }


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
    
    return character_parser.split_by_character(text)


def validate_character_text(text: str) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate character text.
    
    Args:
        text: Input text with [Character] tags
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    return character_parser.validate_character_tags(text)