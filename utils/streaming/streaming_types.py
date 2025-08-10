"""
Universal streaming data structures for TTS Audio Suite

Defines common data types used across all streaming implementations,
eliminating the need for format-specific conversions.
"""

import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class StreamingMode(Enum):
    """Streaming processing modes."""
    TRADITIONAL = "traditional"  # Sequential processing
    STREAMING = "streaming"      # Parallel worker processing


@dataclass
class StreamingSegment:
    """
    Universal segment structure for streaming processing.
    
    All node types (TTS, SRT, VC) convert their data to this format,
    eliminating the need for format-specific routers and conversions.
    """
    index: int                    # Original segment index for result ordering
    text: str                     # Text content to process
    character: str                # Character/voice name for this segment
    language: str                 # Language code (e.g., 'en', 'de', 'fr')
    voice_path: str              # Path to voice reference file
    metadata: Dict[str, Any]      # Node-specific data (SRT timings, etc.)
    
    def __post_init__(self):
        """Validate segment data after initialization."""
        if not isinstance(self.index, int) or self.index < 0:
            raise ValueError(f"Invalid segment index: {self.index}")
        if not self.text.strip():
            raise ValueError("Segment text cannot be empty")
        if not self.character:
            self.character = "narrator"  # Default character
        if not self.language:
            raise ValueError("Language code is required")


@dataclass 
class StreamingResult:
    """Result from streaming segment processing."""
    index: int                    # Original segment index
    audio: torch.Tensor          # Generated audio tensor
    duration: float              # Audio duration in seconds
    character: str               # Character that was processed
    language: str                # Language that was processed
    processing_time: float       # Time taken to process this segment
    worker_id: int               # ID of worker that processed this segment
    success: bool = True         # Whether processing succeeded
    error_msg: str = ""          # Error message if processing failed
    metadata: Dict[str, Any] = None  # Additional result metadata
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StreamingConfig:
    """Configuration for streaming processing."""
    batch_size: int = 4          # Number of parallel workers
    enable_model_preloading: bool = True  # Pre-load models for efficiency
    fallback_to_traditional: bool = True  # Fall back if streaming fails
    streaming_threshold: int = 1  # Minimum segments to enable streaming
    max_workers: int = 12        # Maximum number of workers
    worker_timeout: float = 300.0  # Worker timeout in seconds
    
    # Engine-specific configurations
    engine_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate and initialize configuration."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1") 
        if self.batch_size > self.max_workers:
            self.batch_size = self.max_workers
        if self.engine_config is None:
            self.engine_config = {}


@dataclass
class LanguageGroup:
    """Group of segments by language for optimized processing."""
    language: str
    segments: List[StreamingSegment]
    model_name: str = None       # Engine-specific model name for this language
    
    def __post_init__(self):
        """Validate language group."""
        if not self.segments:
            raise ValueError("Language group cannot be empty")
        
        # Validate all segments have same language
        for segment in self.segments:
            if segment.language != self.language:
                raise ValueError(f"Segment language {segment.language} doesn't match group language {self.language}")


@dataclass
class CharacterGroup:
    """Group of segments by character within a language."""
    character: str
    language: str
    segments: List[StreamingSegment]
    voice_path: str = None
    
    def __post_init__(self):
        """Validate character group."""
        if not self.segments:
            raise ValueError("Character group cannot be empty")
            
        # Validate all segments have same character and language
        for segment in self.segments:
            if segment.character != self.character:
                raise ValueError(f"Segment character {segment.character} doesn't match group character {self.character}")
            if segment.language != self.language:
                raise ValueError(f"Segment language {segment.language} doesn't match group language {self.language}")
        
        # Use voice path from first segment if not provided
        if self.voice_path is None and self.segments:
            self.voice_path = self.segments[0].voice_path


class StreamingMetrics:
    """Metrics for streaming performance monitoring."""
    
    def __init__(self):
        self.total_segments = 0
        self.completed_segments = 0
        self.failed_segments = 0
        self.total_processing_time = 0.0
        self.total_audio_duration = 0.0
        self.worker_stats = {}  # worker_id -> stats
        self.language_stats = {}  # language -> stats
        
    def add_result(self, result: StreamingResult):
        """Add a streaming result to metrics."""
        self.completed_segments += 1
        self.total_processing_time += result.processing_time
        self.total_audio_duration += result.duration
        
        if not result.success:
            self.failed_segments += 1
            
        # Update worker stats
        if result.worker_id not in self.worker_stats:
            self.worker_stats[result.worker_id] = {
                'segments': 0, 'processing_time': 0.0, 'audio_duration': 0.0
            }
        self.worker_stats[result.worker_id]['segments'] += 1
        self.worker_stats[result.worker_id]['processing_time'] += result.processing_time
        self.worker_stats[result.worker_id]['audio_duration'] += result.duration
        
        # Update language stats
        if result.language not in self.language_stats:
            self.language_stats[result.language] = {
                'segments': 0, 'processing_time': 0.0, 'audio_duration': 0.0
            }
        self.language_stats[result.language]['segments'] += 1
        self.language_stats[result.language]['processing_time'] += result.processing_time
        self.language_stats[result.language]['audio_duration'] += result.duration
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of streaming metrics."""
        return {
            'total_segments': self.total_segments,
            'completed_segments': self.completed_segments, 
            'failed_segments': self.failed_segments,
            'success_rate': self.completed_segments / max(self.total_segments, 1),
            'total_processing_time': self.total_processing_time,
            'total_audio_duration': self.total_audio_duration,
            'throughput': self.completed_segments / max(self.total_processing_time, 0.001),
            'active_workers': len(self.worker_stats),
            'languages_processed': list(self.language_stats.keys())
        }