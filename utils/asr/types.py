"""
ASR data types for unified transcription pipeline.
Keep engine-specific output normalization outside adapters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ASRWord:
    start: float
    end: float
    text: str


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: List[ASRWord] = field(default_factory=list)


@dataclass
class ASRResult:
    text: str
    language: Optional[str] = None
    segments: List[ASRSegment] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None


@dataclass
class ASRRequest:
    audio: Dict[str, Any]
    language: Optional[str] = None
    task: str = "transcribe"
    timestamps: str = "none"  # "none" or "word"
    chunk_size: int = 30
    overlap: int = 2
    max_new_tokens: int = 256
    precision: str = "auto"
    attn_implementation: str = "auto"
    use_forced_aligner: bool = False
    forced_aligner_model: Optional[str] = None
