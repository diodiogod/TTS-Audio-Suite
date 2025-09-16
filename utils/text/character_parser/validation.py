"""
Validation and Statistics - Character parser validation functionality

Extracted from character parser for better modularity.
Handles tag validation, statistics, and error reporting.
"""

import re
from typing import Dict, List, Tuple, Any, Pattern


class ValidationMixin:
    """
    Mixin class providing validation and statistics functionality for character parsers.
    
    Extracted to separate concerns and maintain modularity.
    """
    
    def validate_character_tags(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate character tags in text and return any issues.
        
        Args:
            text: Input text with [Character] tags
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Use the character tag pattern from the main parser
        character_tag_pattern = getattr(self, 'CHARACTER_TAG_PATTERN', None)
        if not character_tag_pattern:
            # Fallback pattern if not available
            character_tag_pattern = re.compile(r'\[(?!(?:pause|wait|stop):)(?!(?:it|IT|italian|Italian)\])([^\]]+)\]')
        
        # Find all character tags
        tags = character_tag_pattern.findall(text)
        
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
        available_characters = getattr(self, 'available_characters', set())
        if available_characters:
            unknown_chars = []
            for tag in tags:
                normalized = self.normalize_character_name(tag)
                default_character = getattr(self, 'default_character', 'narrator')
                if normalized == default_character and tag.strip().lower() != default_character:
                    unknown_chars.append(tag)
            
            if unknown_chars:
                issues.append(f"Unknown characters (will use fallback): {', '.join(unknown_chars)}")
        
        return len(issues) == 0, issues
    
    def get_statistics(self, text: str) -> Dict[str, Any]:
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
        emotion_usage = {}
        
        for segment in segments:
            char = segment.character
            character_counts[char] = character_counts.get(char, 0) + 1
            character_lengths[char] = character_lengths.get(char, 0) + len(segment.text)
            
            # Track emotion usage if present
            if hasattr(segment, 'emotion') and segment.emotion:
                if char not in emotion_usage:
                    emotion_usage[char] = set()
                emotion_usage[char].add(segment.emotion)
        
        total_chars = sum(character_counts.values())
        total_length = sum(character_lengths.values())
        
        stats = {
            "total_segments": len(segments),
            "unique_characters": len(character_counts),
            "character_counts": character_counts,
            "character_lengths": character_lengths,
            "total_character_switches": total_chars - 1,
            "total_text_length": total_length,
            "average_segment_length": total_length / len(segments) if segments else 0
        }
        
        # Add emotion statistics if any emotions were found
        if emotion_usage:
            stats["emotion_usage"] = {char: list(emotions) for char, emotions in emotion_usage.items()}
            stats["characters_with_emotions"] = len(emotion_usage)
            stats["total_emotion_references"] = sum(len(emotions) for emotions in emotion_usage.values())
        
        return stats
    
    def get_character_language_summary(self) -> str:
        """
        Get a consolidated summary of character language mappings for logging.
        
        Returns:
            Formatted summary string of character→language mappings
        """
        character_language_cache = getattr(self, '_character_language_cache', {})
        
        if hasattr(self, 'language_resolver'):
            return self.language_resolver.get_character_language_summary(character_language_cache)
        
        # Fallback implementation if language resolver not available
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
        # Use the character tag pattern
        character_tag_pattern = getattr(self, 'CHARACTER_TAG_PATTERN', None)
        if not character_tag_pattern:
            character_tag_pattern = re.compile(r'\[(?!(?:pause|wait|stop):)(?!(?:it|IT|italian|Italian)\])([^\]]+)\]')
        
        # Language-character pattern
        language_character_pattern = re.compile(r'^([a-zA-Z0-9\-_À-ÿ\s]+):(.*)$')
        
        def replace_tag(match):
            tag_content = match.group(1)
            
            # Parse language:character format
            lang_match = language_character_pattern.match(tag_content.strip())
            if lang_match:
                raw_language = lang_match.group(1).strip()
                
                # Get language resolver if available
                language_resolver = getattr(self, 'language_resolver', None)
                if language_resolver:
                    # Resolve to canonical form first, then get display name
                    from utils.models.language_mapper import resolve_language_alias
                    canonical_lang = resolve_language_alias(raw_language)
                    display_name = language_resolver.get_language_display_name(canonical_lang)
                else:
                    # Fallback - just use the raw language
                    display_name = raw_language.title()
                
                return f'[{display_name}]'
            else:
                # Not a language:character tag, remove entirely
                return ''
        
        # Apply conversion
        converted_text = character_tag_pattern.sub(replace_tag, text)
        
        # Clean up any double spaces
        converted_text = ' '.join(converted_text.split())
        
        return converted_text


def validate_character_text(text: str) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate character text.
    
    Args:
        text: Input text with [Character] tags
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    # Create a minimal validator instance
    class BasicValidator(ValidationMixin):
        def normalize_character_name(self, name):
            return name.lower().strip()
        
        def parse_text_segments(self, text):
            return []  # Minimal implementation for validation only
    
    validator = BasicValidator()
    return validator.validate_character_tags(text)