"""
ASR data types for unified transcription pipeline.
Keep engine-specific output normalization outside adapters.
"""

from dataclasses import dataclass, field
import json
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


def asr_word_to_dict(word: ASRWord) -> Dict[str, Any]:
    return {
        "start": float(word.start),
        "end": float(word.end),
        "text": str(word.text),
    }


def asr_segment_to_dict(segment: ASRSegment) -> Dict[str, Any]:
    return {
        "start": float(segment.start),
        "end": float(segment.end),
        "text": str(segment.text),
        "speaker": segment.speaker,
        "words": [asr_word_to_dict(word) for word in (segment.words or [])],
    }


def asr_result_to_dict(result: ASRResult) -> Dict[str, Any]:
    return {
        "text": str(result.text or ""),
        "language": result.language,
        "segments": [asr_segment_to_dict(segment) for segment in (result.segments or [])],
        "raw": result.raw,
    }


def asr_result_to_json(result: ASRResult, *, indent: int = 2) -> str:
    return json.dumps(asr_result_to_dict(result), ensure_ascii=False, indent=indent)
