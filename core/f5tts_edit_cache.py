"""
F5-TTS Edit Cache Management
Cache system for F5-TTS generation to improve iteration speed
Based on SRT node cache infrastructure
"""

import hashlib
import torch
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict

# Global cache instance
GLOBAL_F5TTS_EDIT_CACHE = {}

class F5TTSEditCache:
    """
    Cache management for F5-TTS edit operations
    Implements LRU cache with configurable size limits
    """
    
    def __init__(self, cache_size_limit: int = 100):
        """
        Initialize cache with size limit
        
        Args:
            cache_size_limit: Maximum number of cached entries
        """
        self.cache_size_limit = cache_size_limit
        self.cache = OrderedDict()
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    def _generate_cache_key(self, audio_hash: str, original_text: str, target_text: str,
                          edit_regions: List[Tuple[float, float]], fix_durations: Optional[List[float]],
                          temperature: float, speed: float, target_rms: float, nfe_step: int,
                          cfg_strength: float, sway_sampling_coef: float, ode_method: str,
                          model_name: str) -> str:
        """
        Generate cache key for F5-TTS edit based on generation parameters
        
        Args:
            audio_hash: Hash of input audio content
            original_text: Original text for editing
            target_text: Target text for editing
            edit_regions: List of edit regions as (start, end) tuples
            fix_durations: Optional fixed durations for each region
            temperature: F5-TTS temperature parameter
            speed: F5-TTS speed parameter
            target_rms: Target RMS level
            nfe_step: Neural Function Evaluation steps
            cfg_strength: Classifier-Free Guidance strength
            sway_sampling_coef: Sway sampling coefficient
            ode_method: ODE solver method
            model_name: F5-TTS model name
            
        Returns:
            MD5 hash string as cache key
        """
        cache_data = {
            'audio_hash': audio_hash,
            'original_text': original_text,
            'target_text': target_text,
            'edit_regions': edit_regions,
            'fix_durations': fix_durations,
            'temperature': temperature,
            'speed': speed,
            'target_rms': target_rms,
            'nfe_step': nfe_step,
            'cfg_strength': cfg_strength,
            'sway_sampling_coef': sway_sampling_coef,
            'ode_method': ode_method,
            'model_name': model_name
        }
        
        # Convert to string and sort for consistent hashing
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _generate_audio_hash(self, audio_tensor: torch.Tensor) -> str:
        """
        Generate hash for audio content
        
        Args:
            audio_tensor: Input audio tensor
            
        Returns:
            MD5 hash of audio content
        """
        # Convert to CPU and numpy for consistent hashing
        audio_cpu = audio_tensor.cpu().numpy()
        audio_bytes = audio_cpu.tobytes()
        return hashlib.md5(audio_bytes).hexdigest()
    
    def get(self, audio_tensor: torch.Tensor, original_text: str, target_text: str,
            edit_regions: List[Tuple[float, float]], fix_durations: Optional[List[float]],
            temperature: float, speed: float, target_rms: float, nfe_step: int,
            cfg_strength: float, sway_sampling_coef: float, ode_method: str,
            model_name: str) -> Optional[torch.Tensor]:
        """
        Get cached audio if available
        
        Returns:
            Cached audio tensor or None if not found
        """
        audio_hash = self._generate_audio_hash(audio_tensor)
        cache_key = self._generate_cache_key(
            audio_hash, original_text, target_text, edit_regions, fix_durations,
            temperature, speed, target_rms, nfe_step, cfg_strength, 
            sway_sampling_coef, ode_method, model_name
        )
        
        if cache_key in self.cache:
            # Move to end (most recently used)
            cached_audio = self.cache.pop(cache_key)
            self.cache[cache_key] = cached_audio
            self.cache_stats["hits"] += 1
            return cached_audio
        else:
            self.cache_stats["misses"] += 1
            return None
    
    def put(self, audio_tensor: torch.Tensor, generated_audio: torch.Tensor,
            original_text: str, target_text: str, edit_regions: List[Tuple[float, float]],
            fix_durations: Optional[List[float]], temperature: float, speed: float,
            target_rms: float, nfe_step: int, cfg_strength: float, 
            sway_sampling_coef: float, ode_method: str, model_name: str) -> None:
        """
        Cache generated audio
        
        Args:
            audio_tensor: Input audio tensor
            generated_audio: Generated audio to cache
            ...: All F5-TTS parameters used for generation
        """
        audio_hash = self._generate_audio_hash(audio_tensor)
        cache_key = self._generate_cache_key(
            audio_hash, original_text, target_text, edit_regions, fix_durations,
            temperature, speed, target_rms, nfe_step, cfg_strength,
            sway_sampling_coef, ode_method, model_name
        )
        
        # Add to cache
        self.cache[cache_key] = generated_audio.clone()
        
        # Enforce size limit using LRU eviction
        while len(self.cache) > self.cache_size_limit:
            # Remove oldest (least recently used) item
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            self.cache_stats["evictions"] += 1
    
    def clear(self) -> None:
        """Clear all cached entries"""
        self.cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "cache_limit": self.cache_size_limit,
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "evictions": self.cache_stats["evictions"],
            "hit_rate_percent": round(hit_rate, 1),
            "total_requests": total_requests
        }
    
    def resize(self, new_size_limit: int) -> None:
        """
        Resize cache limit and evict entries if needed
        
        Args:
            new_size_limit: New cache size limit
        """
        self.cache_size_limit = new_size_limit
        
        # Evict entries if current size exceeds new limit
        while len(self.cache) > self.cache_size_limit:
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            self.cache_stats["evictions"] += 1


def get_global_cache() -> F5TTSEditCache:
    """
    Get or create global cache instance
    
    Returns:
        Global F5TTSEditCache instance
    """
    if "default" not in GLOBAL_F5TTS_EDIT_CACHE:
        GLOBAL_F5TTS_EDIT_CACHE["default"] = F5TTSEditCache()
    return GLOBAL_F5TTS_EDIT_CACHE["default"]


def clear_global_cache() -> None:
    """Clear global cache"""
    if "default" in GLOBAL_F5TTS_EDIT_CACHE:
        GLOBAL_F5TTS_EDIT_CACHE["default"].clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics"""
    cache = get_global_cache()
    return cache.get_stats()