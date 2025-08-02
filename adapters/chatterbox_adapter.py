"""
ChatterBox Engine Adapter - Engine-specific adapter for ChatterBox TTS
Provides standardized interface for ChatterBox operations in multilingual engine
"""

import torch
from typing import Dict, Any, Optional, List
from ..core.language_model_mapper import get_model_for_language


class ChatterBoxEngineAdapter:
    """Engine-specific adapter for ChatterBox TTS."""
    
    def __init__(self, node_instance):
        """
        Initialize ChatterBox adapter.
        
        Args:
            node_instance: ChatterboxTTSNode or SRTTTSNode instance
        """
        self.node = node_instance
        self.engine_type = "chatterbox"
    
    def get_model_for_language(self, lang_code: str, default_model: str) -> str:
        """
        Get ChatterBox model name for specified language.
        
        Args:
            lang_code: Language code (e.g., 'en', 'de', 'no')
            default_model: Default model name (language)
            
        Returns:
            ChatterBox model name (language) for the specified language code
        """
        return get_model_for_language(self.engine_type, lang_code, default_model)
    
    def load_base_model(self, language: str, device: str):
        """
        Load base ChatterBox model.
        
        Args:
            language: Language model to load (e.g., "English", "German")
            device: Device to load model on
        """
        self.node.load_tts_model(device, language)
    
    def load_language_model(self, language: str, device: str):
        """
        Load language-specific ChatterBox model.
        
        Args:
            language: Language model to load (e.g., "German", "Norwegian")
            device: Device to load model on
        """
        self.node.load_tts_model(device, language)
    
    def generate_segment_audio(self, text: str, char_audio: str, 
                             character: str = "narrator", **params) -> torch.Tensor:
        """
        Generate ChatterBox audio for a text segment.
        
        Args:
            text: Text to generate audio for
            char_audio: Reference audio file path
            character: Character name for caching
            **params: Additional ChatterBox parameters
            
        Returns:
            Generated audio tensor
        """
        # Extract ChatterBox specific parameters
        exaggeration = params.get("exaggeration", 1.0)
        temperature = params.get("temperature", 0.8)
        cfg_weight = params.get("cfg_weight", 1.0)
        seed = params.get("seed", 0)
        enable_cache = params.get("enable_audio_cache", True)
        
        # Create cache function if caching is enabled
        cache_fn = None
        if enable_cache and hasattr(self.node, '_generate_segment_cache_key'):
            def create_cache_fn():
                def cache_fn_impl(text_content: str, audio_result=None):
                    # Get audio component for cache key
                    audio_component = params.get("stable_audio_component", "main_reference")
                    if character != "narrator":
                        audio_component = f"char_file_{character}"
                    
                    # Get current language/model for cache key
                    current_language = params.get("current_language", params.get("language", "English"))
                    model_source = params.get("model_source") or self.node.model_manager.get_model_source("tts")
                    
                    cache_key = self.node._generate_segment_cache_key(
                        f"{character}:{text_content}", exaggeration, temperature,
                        cfg_weight, seed, audio_component, model_source,
                        params.get("device", "auto"), current_language, character
                    )
                    
                    if audio_result is None:
                        # Get from cache
                        cached_data = self.node._get_cached_segment_audio(cache_key)
                        if cached_data:
                            print(f"💾 Using cached audio for character '{character}' ({current_language}) text: '{text_content[:30]}...'")
                            return cached_data[0]
                        return None
                    else:
                        # Store in cache
                        duration = self._get_audio_duration(audio_result)
                        self.node._cache_segment_audio(cache_key, audio_result, duration)
                
                return cache_fn_impl
            
            cache_fn = create_cache_fn()
        
        # Generate audio using ChatterBox with pause tag support
        return self.node._generate_tts_with_pause_tags(
            text=text,
            audio_prompt=char_audio,
            enable_pause_tags=True,
            character=character,
            seed=seed,
            enable_cache=enable_cache,
            cache_fn=cache_fn,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight
        )
    
    def combine_audio_chunks(self, audio_segments: List[torch.Tensor], **params) -> torch.Tensor:
        """
        Combine ChatterBox audio segments.
        
        Args:
            audio_segments: List of audio tensors to combine
            **params: Combination parameters
            
        Returns:
            Combined audio tensor
        """
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        # ChatterBox uses simple concatenation
        from ..core.audio_processing import AudioProcessingUtils
        return AudioProcessingUtils.concatenate_audio_segments(audio_segments, "simple")
    
    def _get_audio_duration(self, audio_tensor: torch.Tensor) -> float:
        """Calculate audio duration in seconds."""
        # ChatterBox uses 44.1kHz sample rate
        if audio_tensor.dim() == 1:
            num_samples = audio_tensor.shape[0]
        elif audio_tensor.dim() == 2:
            num_samples = audio_tensor.shape[1]
        else:
            num_samples = audio_tensor.numel()
        
        return num_samples / 44100  # ChatterBox sample rate