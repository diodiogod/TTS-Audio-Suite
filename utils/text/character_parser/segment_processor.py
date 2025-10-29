"""
Segment Processor - Text segmentation and parsing functionality

Handles the core text processing logic for character tag parsing.
Extracted from character parser for better modularity.
"""

import re
from typing import List, Optional, Tuple, Pattern
from .base_parser import CharacterSegment


class SegmentProcessor:
    """
    Handles text segmentation and character tag processing.
    
    Responsible for:
    - Line-by-line text processing
    - Character tag detection and parsing
    - Text segment creation with proper positioning
    - Speaker format detection ("Speaker 1:", etc.)
    """
    
    def __init__(self, character_tag_pattern: Pattern, default_character: str):
        """
        Initialize segment processor.
        
        Args:
            character_tag_pattern: Regex pattern for detecting character tags
            default_character: Default character for untagged text
        """
        self.CHARACTER_TAG_PATTERN = character_tag_pattern
        self.default_character = default_character
    
    def parse_text_segments(self, text: str, language_resolver, character_parser) -> List[CharacterSegment]:
        """
        Parse text into character-specific segments with proper line-by-line processing.
        
        Args:
            text: Input text with [Character] tags
            language_resolver: LanguageResolver instance for language handling
            character_parser: Main CharacterParser instance for character normalization
            
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
            line_segments = self._parse_single_line(
                line, line_start_pos, language_resolver, character_parser
            )
            segments.extend(line_segments)
            
            global_pos += len(original_line) + 1  # +1 for newline
        
        # If no segments were created, treat entire text as default character
        if not segments and text.strip():
            segments.append(CharacterSegment(
                character=self.default_character,
                text=text.strip(),
                start_pos=0,
                end_pos=len(text),
                language=language_resolver.resolve_character_language(
                    self.default_character, None,
                    character_parser.character_language_defaults,
                    character_parser._character_language_cache,
                    character_parser._logged_characters
                ),
                original_character=self.default_character,
                explicit_language=False,
                parameters={}  # No parameters for untagged text
            ))
        
        return segments
    
    def _parse_single_line(self, line: str, line_start_pos: int, language_resolver, character_parser) -> List[CharacterSegment]:
        """
        Parse a single line for character tags, treating it completely independently.

        Args:
            line: Single line of text (no newlines)
            line_start_pos: Starting position of this line in the original text
            language_resolver: LanguageResolver instance
            character_parser: Main CharacterParser instance

        Returns:
            List of CharacterSegment objects for this line only
        """
        segments = []
        current_pos = 0
        current_character = self.default_character
        current_language = language_resolver.default_language
        current_emotion = None
        current_parameters = {}  # Track per-segment parameters
        
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
                    language=language_resolver.resolve_character_language(
                        speaker_name, None,
                        character_parser.character_language_defaults,
                        character_parser._character_language_cache,
                        character_parser._logged_characters
                    ),
                    original_character=speaker_name,
                    explicit_language=False,
                    parameters={}  # No parameters for speaker format
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
                    language=language_resolver.resolve_character_language(
                        self.default_character, None,
                        character_parser.character_language_defaults,
                        character_parser._character_language_cache,
                        character_parser._logged_characters
                    ),
                    original_character=self.default_character,
                    explicit_language=False,
                    parameters={}  # No parameters for untagged text
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
                    explicit_language=False,  # Text before tags doesn't have explicit language
                    emotion=current_emotion,
                    parameters=current_parameters  # Use current parameters state
                ))
            
            # Parse language, character, emotion, and parameters from the tag
            raw_tag_content = match.group(1)

            # Use flexible tag parser with parameter support to handle all syntax
            tag_info = language_resolver.parse_tag_with_parameters(raw_tag_content)

            # Extract individual components
            explicit_language = tag_info.get('language')
            raw_character = tag_info.get('character') or self.default_character
            emotion_reference = tag_info.get('emotion')
            segment_parameters = tag_info.get('parameters', {})
            
            # Update current character for text after this tag
            # IMPORTANT: Resolve language using original alias name before character normalization
            current_language = language_resolver.resolve_character_language(
                raw_character, explicit_language,
                character_parser.character_language_defaults,
                character_parser._character_language_cache,
                character_parser._logged_characters
            )
            original_character = raw_character  # Store original before normalization
            current_explicit_language = explicit_language is not None  # Track if language was explicit
            
            # Detect language-only tags: if the original tag had empty character part and raw_character
            # was defaulted to narrator, skip alias resolution to preserve narrator voice priority
            is_language_only_tag = (raw_character == self.default_character and
                                   ':' in raw_tag_content and
                                   raw_tag_content.split(':', 1)[1].strip() == '')

            # Also detect parameter-only tags: if raw_character is narrator and we have parameters
            # but no explicit language specification, it's a parameter-only tag like [seed:4|exag:1.0]
            is_parameter_only_tag = (raw_character == self.default_character and
                                    segment_parameters and
                                    explicit_language is None)

            # Skip narrator alias resolution for language-only or parameter-only tags
            skip_alias = is_language_only_tag or is_parameter_only_tag

            current_character = character_parser.normalize_character_name(
                raw_character, skip_narrator_alias=skip_alias
            )
            
            # Update emotion state if provided in tag
            if emotion_reference is not None:
                current_emotion = emotion_reference

            # Update parameter state if provided in tag
            if segment_parameters:
                current_parameters = segment_parameters.copy()

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
                explicit_language=current_explicit_language if 'current_explicit_language' in locals() else False,
                emotion=current_emotion,
                parameters=current_parameters  # Include current parameters
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
                explicit_language=current_explicit_language if 'current_explicit_language' in locals() else False,
                emotion=current_emotion,
                parameters=current_parameters  # Include current parameters
            ))
        
        return segments
    
    def _parse_speaker_format_line(self, line: str) -> Optional[Tuple[str, str]]:
        """
        Parse a line for manual speaker formats:
        - "Speaker N: text"
        - "[N] text" (concise format)

        Args:
            line: Single line to check

        Returns:
            Tuple of (speaker_name, text) if found, None otherwise
        """
        import re

        # First try "Speaker N: text" format (case insensitive)
        speaker_match = re.match(r'^(speaker\s*\d+)\s*:\s*(.*)$', line.strip(), re.IGNORECASE)
        if speaker_match:
            speaker_name = speaker_match.group(1).lower().strip()  # Normalize to "speaker 1", "speaker 2", etc.
            speaker_text = speaker_match.group(2)
            # Normalize speaker name format
            speaker_name = re.sub(r'\s+', ' ', speaker_name)  # "speaker  1" -> "speaker 1"
            return speaker_name, speaker_text

        # Then try "[N] text" concise format
        bracket_match = re.match(r'^\[(\d+)\]\s*(.*)$', line.strip())
        if bracket_match:
            speaker_num = bracket_match.group(1)
            speaker_text = bracket_match.group(2)
            if speaker_text.strip():  # Only if there's actual text after the number
                speaker_name = f"speaker {speaker_num}"  # Convert [1] to "speaker 1"
                return speaker_name, speaker_text

        return None