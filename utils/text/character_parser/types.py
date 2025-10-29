"""
Character Parser Types - Shared data structures

Contains dataclasses and type definitions used across the character parser modules.
Separated to avoid circular imports.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class CharacterSegment:
    """Represents a single text segment with character assignment, language, and parameters"""
    character: str
    text: str
    start_pos: int
    end_pos: int
    language: Optional[str] = None
    original_character: Optional[str] = None  # Original character before alias resolution
    explicit_language: bool = False  # True if language was explicitly specified in tag (e.g., [German:Bob])
    emotion: Optional[str] = None  # Emotion reference for advanced TTS engines (IndexTTS-2)
    parameters: Dict[str, Any] = field(default_factory=dict)  # Per-segment parameters like seed, temperature

    def __str__(self) -> str:
        language_part = f" ({self.language})" if self.language else ""
        emotion_part = f" [emotion: {self.emotion}]" if self.emotion else ""
        params_part = f" {self.parameters}" if self.parameters else ""
        return f"[{self.character}]{language_part}{emotion_part}{params_part}: {self.text[:30]}{'...' if len(self.text) > 30 else ''}"