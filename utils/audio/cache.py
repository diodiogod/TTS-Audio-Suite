"""
Audio Cache Module - Unified caching system for TTS engines
Provides centralized cache management with engine-specific cache key generation
"""

import hashlib
import torch
from typing import Dict, Any, Optional, Tuple, Callable
from abc import ABC, abstractmethod


# Global audio cache shared across all engines
GLOBAL_AUDIO_CACHE = {}


class CacheKeyGenerator(ABC):
    """Abstract base class for engine-specific cache key generation."""
    
    @abstractmethod
    def generate_cache_key(self, **params) -> str:
        """Generate cache key from engine-specific parameters."""
        pass


class F5TTSCacheKeyGenerator(CacheKeyGenerator):
    """Cache key generator for F5-TTS engine."""
    
    def generate_cache_key(self, **params) -> str:
        """Generate F5-TTS cache key from parameters."""
        cache_data = {
            'text': params.get('text', ''),
            'model_name': params.get('model_name', ''),
            'device': params.get('device', ''),
            'audio_component': params.get('audio_component', ''),
            'ref_text': params.get('ref_text', ''),
            'temperature': params.get('temperature', 0.8),
            'speed': params.get('speed', 1.0),
            'target_rms': params.get('target_rms', 0.1),
            'cross_fade_duration': params.get('cross_fade_duration', 0.15),
            'nfe_step': params.get('nfe_step', 32),
            'cfg_strength': params.get('cfg_strength', 2.0),
            'seed': params.get('seed', 0),
            'character': params.get('character', 'narrator'),
            'engine': 'f5tts'
        }
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()


class ChatterBoxCacheKeyGenerator(CacheKeyGenerator):
    """Cache key generator for ChatterBox engine."""
    
    def generate_cache_key(self, **params) -> str:
        """Generate ChatterBox cache key from parameters."""
        cache_data = {
            'text': params.get('text', ''),
            'exaggeration': params.get('exaggeration', 1.0),
            'temperature': params.get('temperature', 0.8),
            'cfg_weight': params.get('cfg_weight', 1.0),
            'seed': params.get('seed', 0),
            'audio_component': params.get('audio_component', ''),
            'model_source': params.get('model_source', ''),
            'device': params.get('device', ''),
            'language': params.get('language', 'English'),
            'character': params.get('character', 'narrator'),
            'engine': 'chatterbox'
        }
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()


class AudioCache:
    """Unified audio cache manager for all TTS engines."""
    
    def __init__(self):
        self.cache_key_generators = {
            'f5tts': F5TTSCacheKeyGenerator(),
            'chatterbox': ChatterBoxCacheKeyGenerator()
        }
    
    def register_cache_key_generator(self, engine_type: str, generator: CacheKeyGenerator):
        """Register a cache key generator for a specific engine."""
        self.cache_key_generators[engine_type] = generator
    
    def generate_cache_key(self, engine_type: str, **params) -> str:
        """Generate cache key for specified engine type."""
        if engine_type not in self.cache_key_generators:
            raise ValueError(f"Unknown engine type: {engine_type}")
        
        generator = self.cache_key_generators[engine_type]
        return generator.generate_cache_key(**params)
    
    def get_cached_audio(self, cache_key: str) -> Optional[Tuple[torch.Tensor, float]]:
        """Retrieve cached audio by cache key."""
        return GLOBAL_AUDIO_CACHE.get(cache_key)
    
    def cache_audio(self, cache_key: str, audio_tensor: torch.Tensor, duration: float):
        """Cache audio tensor with duration."""
        GLOBAL_AUDIO_CACHE[cache_key] = (audio_tensor.clone(), duration)
    
    def create_cache_function(self, engine_type: str, **static_params) -> Callable:
        """
        Create a cache function for use with TTS generation.
        
        Args:
            engine_type: "f5tts" or "chatterbox"
            **static_params: Parameters that don't change between calls
            
        Returns:
            Cache function that can be called with (text, audio_result=None)
        """
        def cache_fn(text_content: str, audio_result=None):
            # Combine static params with dynamic text
            cache_params = static_params.copy()
            cache_params['text'] = f"{cache_params.get('character', 'narrator')}:{text_content}"
            
            # Generate cache key
            cache_key = self.generate_cache_key(engine_type, **cache_params)
            
            if audio_result is None:
                # Get from cache
                cached_data = self.get_cached_audio(cache_key)
                if cached_data:
                    character = cache_params.get('character', 'narrator')
                    language = cache_params.get('language', cache_params.get('model_name', ''))
                    if language and language != 'English':
                        print(f"💾 Using cached audio for '{character}' ({language}): '{text_content[:30]}...'")
                    else:
                        print(f"💾 Using cached audio for '{character}': '{text_content[:30]}...'")
                    return cached_data[0]
                return None
            else:
                # Store in cache
                duration = self._calculate_duration(audio_result, engine_type)
                self.cache_audio(cache_key, audio_result, duration)
        
        return cache_fn
    
    def _calculate_duration(self, audio_tensor: torch.Tensor, engine_type: str) -> float:
        """Calculate audio duration based on engine type."""
        if audio_tensor.dim() == 1:
            num_samples = audio_tensor.shape[0]
        elif audio_tensor.dim() == 2:
            num_samples = audio_tensor.shape[1]
        else:
            num_samples = audio_tensor.numel()
        
        # Use engine-specific sample rates
        sample_rate = 24000 if engine_type == 'f5tts' else 44100
        return num_samples / sample_rate
    
    def clear_cache(self):
        """Clear all cached audio."""
        global GLOBAL_AUDIO_CACHE
        GLOBAL_AUDIO_CACHE.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_items = len(GLOBAL_AUDIO_CACHE)
        total_memory = sum(
            audio.numel() * audio.element_size() + 8  # 8 bytes for duration float
            for audio, _ in GLOBAL_AUDIO_CACHE.values()
        )
        
        return {
            'total_items': total_items,
            'total_memory_bytes': total_memory,
            'total_memory_mb': total_memory / (1024 * 1024)
        }


# Global cache instance
audio_cache = AudioCache()


def get_audio_cache() -> AudioCache:
    """Get the global audio cache instance."""
    return audio_cache


def create_cache_function(engine_type: str, **static_params) -> Callable:
    """
    Convenience function to create a cache function.
    
    Args:
        engine_type: "f5tts" or "chatterbox"
        **static_params: Parameters that don't change between calls
        
    Returns:
        Cache function for use with TTS generation
    """
    return audio_cache.create_cache_function(engine_type, **static_params)