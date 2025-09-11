"""
Character Parser for ChatterBox Voice - Universal Text Processing
Backward Compatibility Module

The original 930-line character parser has been modularized for better maintainability.
This module maintains backward compatibility by delegating to the new modular structure.

New modular structure:
- utils/text/character_parser/
  ├── __init__.py                    # Main exports
  ├── base_parser.py                 # Core CharacterParser class (~300 lines)
  ├── language_resolver.py           # Language resolution logic (~200 lines)
  ├── segment_processor.py           # Text segmentation logic (~200 lines)
  └── validation.py                  # Validation and statistics (~150 lines)

Each module is now ~150-300 lines, following the 500-600 line policy.
"""

# Import everything from the modular structure for backward compatibility
from .character_parser import *

# Ensure global instances are available at module level
from .character_parser import character_parser

# Additional convenience exports that may be expected
__all__ = [
    'CharacterParser',
    'CharacterSegment',
    'character_parser', 
    'parse_character_text',
    'validate_character_text'
]