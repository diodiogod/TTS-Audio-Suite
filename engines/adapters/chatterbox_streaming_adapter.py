"""
ChatterBox Streaming Adapter

Bridges the existing ChatterBox TTS implementation to the universal streaming system.
Preserves all existing ChatterBox functionality while enabling universal streaming.
"""

import torch
import os
from typing import Dict, Any, List, Optional
import sys

# Add project root to path for imports
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.streaming.streaming_interface import StreamingEngineAdapter
from utils.streaming.streaming_types import StreamingSegment, StreamingResult


class ChatterBoxStreamingAdapter(StreamingEngineAdapter):
    """
    Streaming adapter for ChatterBox TTS engine.
    
    Bridges existing ChatterBox functionality to the universal streaming system,
    preserving all existing features including language switching, character voices,
    pause tags, crash protection, and audio caching.
    """
    
    def __init__(self, node_instance):
        """
        Initialize ChatterBox streaming adapter.
        
        Args:
            node_instance: ChatterBoxTTSNode or ChatterBoxSRTNode instance
        """
        super().__init__(node_instance)
        
        # Cache for pre-loaded models (for streaming efficiency)
        self._preloaded_models = {}
        self._current_model_language = None
    
    def _get_engine_name(self) -> str:
        """Return the name of this engine."""
        return "chatterbox"
    
    def _get_supported_languages(self) -> List[str]:
        """Return list of language codes supported by ChatterBox."""
        try:
            from engines.chatterbox.language_models import get_available_languages
            from utils.models.language_mapper import chatterbox_language_mapper
            
            # Get available ChatterBox language models
            available_models = get_available_languages()
            
            # Get supported language codes from mapper
            supported_codes = chatterbox_language_mapper.get_supported_languages()
            
            # Add 'local:' prefixed models as supported languages
            for model in available_models:
                if model.startswith('local:'):
                    # Local models use their language code after 'local:'
                    supported_codes.append(model)
            
            return supported_codes
            
        except ImportError:
            # Fallback to default languages
            return ['en', 'de', 'no', 'nb', 'nn']
    
    def process_segment(self, segment: StreamingSegment, **kwargs) -> StreamingResult:
        """
        Process a single streaming segment using ChatterBox.
        
        This method bridges to the existing ChatterBox processing logic,
        preserving all functionality including pause tags, crash protection,
        chunking, and audio caching.
        
        Args:
            segment: StreamingSegment to process
            **kwargs: Processing parameters (exaggeration, temperature, etc.)
            
        Returns:
            StreamingResult with generated audio
        """
        start_time = self._get_current_time()
        
        try:
            # Ensure we have the correct model loaded for this language
            if self._current_model_language != segment.language:
                self.load_model_for_language(segment.language, kwargs.get('device', 'auto'))
            
            # Check if node has the streaming processing method
            if hasattr(self.node, '_process_single_segment_for_streaming'):
                # Use existing streaming method (preserves all ChatterBox features)
                audio = self.node._process_single_segment_for_streaming(
                    original_idx=segment.index,
                    character=segment.character,
                    segment_text=segment.text,
                    language=segment.language,
                    voice_path=segment.voice_path,
                    inputs=kwargs
                )
            else:
                # Fallback to standard generation method
                audio = self._generate_with_chatterbox(
                    text=segment.text,
                    voice_path=segment.voice_path,
                    character=segment.character,
                    language=segment.language,
                    **kwargs
                )
            
            # Calculate audio duration
            sample_rate = kwargs.get('sample_rate', 22050)
            duration = float(audio.shape[-1]) / sample_rate if audio is not None else 0.0
            
            # Create successful result
            return StreamingResult(
                index=segment.index,
                audio=audio,
                duration=duration,
                character=segment.character,
                language=segment.language,
                processing_time=self._get_current_time() - start_time,
                worker_id=0,  # Will be set by worker
                success=True,
                metadata=segment.metadata
            )
            
        except Exception as e:
            # Return error result
            return StreamingResult(
                index=segment.index,
                audio=torch.zeros(1, 1000),
                duration=0.0,
                character=segment.character,
                language=segment.language,
                processing_time=self._get_current_time() - start_time,
                worker_id=0,
                success=False,
                error_msg=str(e),
                metadata=segment.metadata
            )
    
    def load_model_for_language(self, language: str, device: str = "auto") -> bool:
        """
        Load or switch to the ChatterBox model for specified language.
        
        Args:
            language: Language code or model name (e.g., 'en', 'de', 'local:CustomModel')
            device: Device to load model on
            
        Returns:
            True if model loaded successfully
        """
        try:
            # Get the model name for this language
            model_name = self.get_model_for_language(language)
            
            # Check if we already have this model loaded
            if self._current_model_language == language and hasattr(self.node, 'tts_model'):
                return True
            
            # Load the model using node's existing method
            if hasattr(self.node, 'load_tts_model'):
                self.node.load_tts_model(device, model_name)
                self._current_model_language = language
                return True
            else:
                print(f"❌ Node doesn't have load_tts_model method")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load ChatterBox model for {language}: {e}")
            return False
    
    def get_model_for_language(self, language: str) -> str:
        """
        Get the ChatterBox model name for specified language.
        
        Args:
            language: Language code (e.g., 'en', 'de', 'no')
            
        Returns:
            Model name for this language
        """
        try:
            from utils.models.language_mapper import get_model_for_language
            
            # Handle local models
            if language.startswith('local:'):
                return language  # Return as-is for local models
            
            # Get model name from mapper
            default_model = getattr(self.node, 'default_language', 'English')
            return get_model_for_language('chatterbox', language, default_model)
            
        except ImportError:
            # Fallback mapping
            language_map = {
                'en': 'English',
                'de': 'German',
                'no': 'Norwegian',
                'nb': 'Norwegian',
                'nn': 'Norwegian'
            }
            return language_map.get(language, 'English')
    
    def preload_models(self, languages: List[str], device: str = "auto") -> Dict[str, bool]:
        """
        Pre-load ChatterBox models for multiple languages.
        
        Uses the node's existing _preload_language_models method if available,
        or falls back to sequential loading.
        
        Args:
            languages: List of language codes to preload
            device: Device to load models on
            
        Returns:
            Dict mapping language -> success status
        """
        results = {}
        
        # Check if node has streaming model manager (for efficient pre-loading)
        if hasattr(self.node, '_preload_language_models'):
            try:
                # Use existing pre-loading method
                self.node._preload_language_models(languages, device)
                for lang in languages:
                    results[lang] = True
                print(f"✅ Pre-loaded {len(languages)} ChatterBox models using streaming manager")
                return results
            except Exception as e:
                print(f"⚠️ Failed to use streaming model manager: {e}, falling back to sequential loading")
        
        # Fallback to sequential loading
        return super().preload_models(languages, device)
    
    def _generate_with_chatterbox(self, text: str, voice_path: str, character: str, 
                                   language: str, **kwargs) -> torch.Tensor:
        """
        Generate audio using ChatterBox TTS model.
        
        This is a fallback method when the node doesn't have 
        _process_single_segment_for_streaming.
        
        Args:
            text: Text to generate
            voice_path: Path to voice reference file
            character: Character name
            language: Language code
            **kwargs: Additional parameters
            
        Returns:
            Generated audio tensor
        """
        # Extract parameters with defaults
        exaggeration = kwargs.get('exaggeration', 0.5)
        temperature = kwargs.get('temperature', 0.8)
        cfg_weight = kwargs.get('cfg_weight', 0.5)
        seed = kwargs.get('seed', 42)
        enable_cache = kwargs.get('enable_audio_cache', True)
        crash_protection = kwargs.get('crash_protection_template', 'hmm ,, {seg} hmm ,,')
        
        # Check if node has generation method
        if hasattr(self.node, '_generate_tts_with_pause_tags'):
            # Use pause tag-aware generation
            return self.node._generate_tts_with_pause_tags(
                text, voice_path, exaggeration, temperature, cfg_weight,
                language, True, character=character, seed=seed,
                enable_cache=enable_cache,
                crash_protection_template=crash_protection,
                stable_audio_component=""
            )
        elif hasattr(self.node, 'tts_model') and self.node.tts_model:
            # Direct model generation
            return self.node.tts_model.generate(
                text=text,
                audio_prompt_path=voice_path if voice_path != 'none' else None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                seed=seed
            )
        else:
            raise RuntimeError("ChatterBox node doesn't have a generation method available")
    
    def _get_current_time(self) -> float:
        """Get current time for performance measurement."""
        import time
        return time.time()
    
    def validate_segment(self, segment: StreamingSegment) -> bool:
        """
        Validate that ChatterBox can process this segment.
        
        Args:
            segment: Segment to validate
            
        Returns:
            True if segment can be processed
        """
        # Check language support
        if segment.language not in self.supported_languages:
            # Check if it's a local model
            if not segment.language.startswith('local:'):
                print(f"❌ ChatterBox doesn't support language: {segment.language}")
                return False
        
        # Basic validation
        if not segment.text.strip():
            # ChatterBox can handle empty text with crash protection
            # So we allow it through
            pass
        
        return True