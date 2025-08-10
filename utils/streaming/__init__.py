"""
Universal Streaming System for TTS Audio Suite

Provides engine-agnostic streaming architecture that eliminates the need for
format-specific routers and bridges between different node types.
"""

from .streaming_types import StreamingSegment, StreamingResult, StreamingConfig
from .streaming_interface import StreamingEngineAdapter
from .streaming_coordinator import StreamingCoordinator
from .work_queue_processor import UniversalWorkQueueProcessor

__all__ = [
    'StreamingSegment',
    'StreamingResult', 
    'StreamingConfig',
    'StreamingEngineAdapter',
    'StreamingCoordinator',
    'UniversalWorkQueueProcessor'
]