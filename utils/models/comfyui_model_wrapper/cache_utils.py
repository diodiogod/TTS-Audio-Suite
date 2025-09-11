"""
Cache utilities for ComfyUI model wrapper system
"""

import time

# Global cache invalidation flag to force recreation of all engine instances
# When models are unloaded, this timestamp is updated to invalidate all node caches
_global_cache_invalidation_flag = 0.0


def is_engine_cache_valid(cache_timestamp: float) -> bool:
    """
    Check if an engine cache is still valid based on global invalidation flag.
    
    Args:
        cache_timestamp: When the cache entry was created
        
    Returns:
        True if cache is still valid, False if it should be invalidated
    """
    return cache_timestamp > _global_cache_invalidation_flag


def invalidate_all_caches():
    """Set global cache invalidation flag to force engine recreation"""
    global _global_cache_invalidation_flag
    _global_cache_invalidation_flag = time.time()
    print(f"🗑️ Set global cache invalidation flag to force engine recreation")