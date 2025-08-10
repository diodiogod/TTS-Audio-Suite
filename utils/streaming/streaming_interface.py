"""
Abstract streaming interface for TTS engines

Defines the contract that all streaming-capable engines must implement.
This enables engine-agnostic streaming without hardcoded dependencies.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from .streaming_types import StreamingSegment, StreamingResult, LanguageGroup, CharacterGroup


class StreamingEngineAdapter(ABC):
    """
    Abstract base class for engine streaming adapters.
    
    Each TTS engine (ChatterBox, F5-TTS, etc.) implements this interface
    to become compatible with the universal streaming system.
    """
    
    def __init__(self, node_instance):
        """
        Initialize adapter with reference to the engine node.
        
        Args:
            node_instance: The TTS node instance (e.g., ChatterBoxTTSNode, F5TTSNode)
        """
        self.node = node_instance
        self.engine_name = self._get_engine_name()
        self.supported_languages = self._get_supported_languages()
    
    @abstractmethod
    def _get_engine_name(self) -> str:
        """Return the name of this engine (e.g., 'chatterbox', 'f5tts')."""
        pass
    
    @abstractmethod
    def _get_supported_languages(self) -> List[str]:
        """Return list of language codes supported by this engine."""
        pass
    
    @abstractmethod
    def process_segment(self, segment: StreamingSegment, **kwargs) -> StreamingResult:
        """
        Process a single streaming segment.
        
        This is the core method called by streaming workers.
        Must handle model loading, language switching, and audio generation.
        
        Args:
            segment: StreamingSegment to process
            **kwargs: Additional processing parameters
            
        Returns:
            StreamingResult with generated audio and metadata
        """
        pass
    
    @abstractmethod
    def load_model_for_language(self, language: str, device: str = "auto") -> bool:
        """
        Load or switch to the model for specified language.
        
        Args:
            language: Language code (e.g., 'en', 'de', 'fr')
            device: Device to load model on
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_for_language(self, language: str) -> str:
        """
        Get the model name for specified language.
        
        Args:
            language: Language code
            
        Returns:
            Model name for this language
        """
        pass
    
    def preload_models(self, languages: List[str], device: str = "auto") -> Dict[str, bool]:
        """
        Pre-load models for multiple languages for streaming efficiency.
        
        Default implementation loads models sequentially.
        Engines can override for optimized batch loading.
        
        Args:
            languages: List of language codes to preload
            device: Device to load models on
            
        Returns:
            Dict mapping language -> success status
        """
        results = {}
        for language in languages:
            try:
                success = self.load_model_for_language(language, device)
                results[language] = success
                if success:
                    print(f"✅ Pre-loaded {self.engine_name} model for {language}")
                else:
                    print(f"❌ Failed to pre-load {self.engine_name} model for {language}")
            except Exception as e:
                print(f"❌ Error pre-loading {self.engine_name} model for {language}: {e}")
                results[language] = False
        return results
    
    def group_segments_by_language(self, segments: List[StreamingSegment]) -> Dict[str, LanguageGroup]:
        """
        Group segments by language for optimized processing.
        
        Args:
            segments: List of segments to group
            
        Returns:
            Dict mapping language code -> LanguageGroup
        """
        groups = {}
        for segment in segments:
            if segment.language not in groups:
                groups[segment.language] = LanguageGroup(
                    language=segment.language,
                    segments=[segment],  # Initialize with first segment to avoid empty validation error
                    model_name=self.get_model_for_language(segment.language)
                )
            else:
                groups[segment.language].segments.append(segment)
        return groups
    
    def group_segments_by_character(self, segments: List[StreamingSegment]) -> Dict[str, Dict[str, CharacterGroup]]:
        """
        Group segments by language and then by character.
        
        Args:
            segments: List of segments to group
            
        Returns:
            Dict mapping language -> Dict mapping character -> CharacterGroup
        """
        language_groups = self.group_segments_by_language(segments)
        result = {}
        
        for language, lang_group in language_groups.items():
            result[language] = {}
            
            # Group by character within this language
            for segment in lang_group.segments:
                char = segment.character
                if char not in result[language]:
                    result[language][char] = CharacterGroup(
                        character=char,
                        language=language,
                        segments=[segment],  # Initialize with first segment to avoid empty validation error
                        voice_path=segment.voice_path
                    )
                else:
                    result[language][char].segments.append(segment)
                
        return result
    
    def validate_segment(self, segment: StreamingSegment) -> bool:
        """
        Validate that this adapter can process the given segment.
        
        Args:
            segment: Segment to validate
            
        Returns:
            True if segment can be processed, False otherwise
        """
        # Check language support
        if segment.language not in self.supported_languages:
            print(f"❌ {self.engine_name} doesn't support language: {segment.language}")
            return False
            
        # Basic validation
        if not segment.text.strip():
            print(f"❌ Empty text in segment {segment.index}")
            return False
            
        return True
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about this engine adapter.
        
        Returns:
            Dict with engine information
        """
        return {
            'name': self.engine_name,
            'supported_languages': self.supported_languages,
            'node_type': type(self.node).__name__,
            'adapter_version': '1.0'
        }


class StreamingCapableMixin:
    """
    Mixin class for nodes that want to support streaming.
    
    Provides common streaming functionality that can be added to any node.
    """
    
    def __init__(self):
        self._streaming_adapter = None
        self._streaming_enabled = False
    
    def get_streaming_adapter(self) -> Optional[StreamingEngineAdapter]:
        """Get the streaming adapter for this node."""
        return self._streaming_adapter
    
    def set_streaming_adapter(self, adapter: StreamingEngineAdapter):
        """Set the streaming adapter for this node."""
        self._streaming_adapter = adapter
        self._streaming_enabled = True
    
    def is_streaming_capable(self) -> bool:
        """Check if this node supports streaming."""
        return self._streaming_adapter is not None
    
    def should_use_streaming(self, batch_size: int, segment_count: int, min_segments: int = 1) -> bool:
        """
        Determine if streaming should be used based on configuration.
        
        Args:
            batch_size: Requested batch size (0 = disable streaming)
            segment_count: Number of segments to process  
            min_segments: Minimum segments required for streaming
            
        Returns:
            True if streaming should be used, False for traditional processing
        """
        if not self.is_streaming_capable():
            return False
            
        if batch_size <= 0:
            return False
            
        if segment_count < min_segments:
            return False
            
        return True