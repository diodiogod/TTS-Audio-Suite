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
            'engine': 'chatterbox',
            # ChatterBox Official 23-Lang specific parameters
            'repetition_penalty': params.get('repetition_penalty', 1.0),
            'min_p': params.get('min_p', 0.0),
            'top_p': params.get('top_p', 1.0)
        }
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()


class ChatterBoxOfficial23LangCacheKeyGenerator(CacheKeyGenerator):
    """Cache key generator for ChatterBox Official 23-Lang engine."""

    def generate_cache_key(self, **params) -> str:
        """Generate ChatterBox Official 23-Lang cache key from parameters."""
        cache_data = {
            'text': params.get('text', ''),
            'exaggeration': params.get('exaggeration', 0.5),
            'temperature': params.get('temperature', 0.8),
            'cfg_weight': params.get('cfg_weight', 0.5),
            'seed': params.get('seed', 0),
            'audio_component': params.get('audio_component', ''),
            'model_source': params.get('model_source', ''),
            'model_version': params.get('model_version', 'v1'),  # v1 or v2 - ensures cache invalidation between versions
            'device': params.get('device', ''),
            'language': params.get('language', 'English'),
            'character': params.get('character', 'narrator'),
            'engine': 'chatterbox_official_23lang',
            # ChatterBox Official 23-Lang specific parameters
            'repetition_penalty': params.get('repetition_penalty', 1.0),
            'min_p': params.get('min_p', 0.0),
            'top_p': params.get('top_p', 1.0)
        }
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()


class HiggsAudioCacheKeyGenerator(CacheKeyGenerator):
    """Cache key generator for Higgs Audio engine."""
    
    def generate_cache_key(self, **params) -> str:
        """Generate Higgs Audio cache key from parameters."""
        cache_data = {
            'text': params.get('text', ''),
            'model_path': params.get('model_path', ''),
            'tokenizer_path': params.get('tokenizer_path', ''),
            'device': params.get('device', ''),
            'reference_text': params.get('reference_text', ''),
            'system_prompt': params.get('system_prompt', ''),
            'temperature': params.get('temperature', 1.0),
            'top_p': params.get('top_p', 0.95),
            'top_k': params.get('top_k', 50),
            'max_new_tokens': params.get('max_new_tokens', 2048),
            'force_audio_gen': params.get('force_audio_gen', False),
            'ras_win_len': params.get('ras_win_len', 7),
            'ras_max_num_repeat': params.get('ras_max_num_repeat', 2),
            'seed': params.get('seed', 0),
            'character': params.get('character', 'narrator'),
            'max_chars_per_chunk': params.get('max_chars_per_chunk', 400),
            # Note: silence_between_chunks_ms excluded - it's post-processing, doesn't affect generation
            'engine': 'higgs_audio'
        }
        
        # Add reference audio hash if present
        if 'reference_audio' in params and params['reference_audio'] is not None:
            audio_dict = params['reference_audio']
            if isinstance(audio_dict, dict) and 'waveform' in audio_dict:
                # Create hash of audio waveform
                import hashlib
                waveform = audio_dict['waveform']
                if hasattr(waveform, 'cpu'):  # torch tensor
                    audio_hash = hashlib.md5(waveform.cpu().numpy().tobytes()).hexdigest()[:8]
                    cache_data['audio_hash'] = audio_hash
        
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()


class VibeVoiceCacheKeyGenerator(CacheKeyGenerator):
    """Cache key generator for VibeVoice engine."""
    
    def generate_cache_key(self, **params) -> str:
        """Generate VibeVoice cache key from parameters."""
        # Fix floating point precision issues by rounding to 3 decimal places
        cfg_scale = params.get('cfg_scale', 1.3)
        temperature = params.get('temperature', 0.95) 
        top_p = params.get('top_p', 0.95)
        
        if isinstance(cfg_scale, (int, float)):
            cfg_scale = round(float(cfg_scale), 3)
        if isinstance(temperature, (int, float)):
            temperature = round(float(temperature), 3)
        if isinstance(top_p, (int, float)):
            top_p = round(float(top_p), 3)
        
        cache_data = {
            'text': params.get('text', ''),
            'cfg_scale': cfg_scale,
            'temperature': temperature,
            'top_p': top_p,
            'use_sampling': params.get('use_sampling', False),
            'seed': params.get('seed', 42),
            'model_source': params.get('model_source', 'vibevoice-1.5B'),
            'device': params.get('device', 'auto'),
            'max_new_tokens': params.get('max_new_tokens'),
            'multi_speaker_mode': params.get('multi_speaker_mode', 'Custom Character Switching'),
            'audio_component': params.get('audio_component', ''),
            'character': params.get('character', 'narrator'),
            'inference_steps': params.get('inference_steps', 20),  # Include diffusion inference steps in cache key
            'attention_mode': params.get('attention_mode', 'eager'),  # Include attention mode in cache key
            'quantize': params.get('quantize', False),  # Include quantization in cache key
            'engine': 'vibevoice'
        }
        
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()


class IndexTTSCacheKeyGenerator(CacheKeyGenerator):
    """Cache key generator for IndexTTS-2 engine."""
    
    def generate_cache_key(self, **params) -> str:
        """Generate IndexTTS-2 cache key from parameters."""
        # Round floating point values to avoid precision issues
        temperature = params.get('temperature', 0.8)
        top_p = params.get('top_p', 0.8)
        emotion_alpha = params.get('emotion_alpha', 1.0)
        
        if isinstance(temperature, (int, float)):
            temperature = round(float(temperature), 3)
        if isinstance(top_p, (int, float)):
            top_p = round(float(top_p), 3)
        if isinstance(emotion_alpha, (int, float)):
            emotion_alpha = round(float(emotion_alpha), 3)
        
        cache_data = {
            'text': params.get('text', ''),
            'speaker_audio': params.get('speaker_audio', ''),
            'emotion_audio': params.get('emotion_audio', ''),
            'emotion_alpha': emotion_alpha,
            'emotion_vector': params.get('emotion_vector'),  # List or None
            'use_emotion_text': params.get('use_emotion_text', False),
            'emotion_text': params.get('emotion_text', ''),
            'use_random': params.get('use_random', False),
            'seed': params.get('seed', 0),  # Seed for reproducible generation
            'temperature': temperature,
            'top_p': top_p,
            'top_k': params.get('top_k', 30),
            'length_penalty': params.get('length_penalty', 0.0),
            'repetition_penalty': params.get('repetition_penalty', 10.0),
            'max_mel_tokens': params.get('max_mel_tokens', 1500),
            'max_text_tokens_per_segment': params.get('max_text_tokens_per_segment', 120),
            'interval_silence': params.get('interval_silence', 200),
            'model_name': params.get('model_name', 'IndexTTS-2'),
            'device': params.get('device', 'auto'),
            'character': params.get('character', 'narrator'),
            'engine': 'index_tts'
        }
        
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()


class AudioCache:
    """Unified audio cache manager for all TTS engines."""
    
    def __init__(self):
        self.cache_key_generators = {
            'f5tts': F5TTSCacheKeyGenerator(),
            'chatterbox': ChatterBoxCacheKeyGenerator(),
            'chatterbox_official_23lang': ChatterBoxOfficial23LangCacheKeyGenerator(),  # Uses specialized generator with advanced params
            'higgs_audio': HiggsAudioCacheKeyGenerator(),
            'vibevoice': VibeVoiceCacheKeyGenerator(),
            'index_tts': IndexTTSCacheKeyGenerator()
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