"""
Character Parser Module - Modular text processing for TTS engines

This module provides unified character tag parsing with emotion support for all TTS engines.
Backward compatible with existing CharacterParser usage.
"""

# Import all classes and functions from the modular structure
from .types import CharacterSegment
from .base_parser import CharacterParser
from .validation import validate_character_text

# Global instances for backward compatibility
from .base_parser import character_parser

# Convenience functions for backward compatibility
from .base_parser import parse_character_text

__all__ = [
    'CharacterParser',
    'CharacterSegment', 
    'character_parser',
    'parse_character_text',
    'validate_character_text'
]