"""
Character Parser Types - Shared data structures

Contains dataclasses and type definitions used across the character parser modules.
Separated to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Optional


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
    emotion: Optional[str] = None  # Emotion reference for advanced TTS engines (IndexTTS-2)
    
    def __str__(self) -> str:
        language_part = f" ({self.language})" if self.language else ""
        emotion_part = f" [emotion: {self.emotion}]" if self.emotion else ""
        return f"[{self.character}]{language_part}{emotion_part}: {self.text[:30]}{'...' if len(self.text) > 30 else ''}"