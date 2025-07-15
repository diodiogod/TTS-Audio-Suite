# F5-TTS Edit Node Improvements Plan

## Context & Problem Analysis

### Current Issues Identified
- **Stuttering/clipping at boundaries** when editing small segments (e.g., "55" to "7 year")
- **Fixed 50ms crossfade** hardcoded in audio_compositing.py, not optimal for all cases
- **No caching system** unlike SRT node which has sophisticated caching for fast iteration
- **Issue exists in BOTH old and new nodes** - not compositing-specific
- **Main node becoming cluttered** with too many parameters

### Research Findings
- **SRT Node has excellent caching:**
  - Global audio cache with MD5 cache keys
  - `enable_audio_cache` boolean parameter  
  - Cache based on generation parameters (text, temperature, etc.)
  - Cache status reporting ("cached" vs "generated")
- **Current crossfade is basic:**
  - Linear crossfade only, hardcoded 50ms
  - No adaptive behavior for different segment sizes
  - No alternative curve types

## Implementation Plan

### Phase 1: Create F5-TTS Edit Options Node

#### New Node Specifications
- **File:** `nodes/f5tts_edit_options_node.py`
- **Class:** `F5TTSEditOptionsNode`
- **Title:** `🔧 F5-TTS Edit Options` (following Audio Analyzer pattern)
- **Category:** `F5-TTS Voice`
- **Export Name:** `ChatterBoxF5TTSEditOptions`

#### Node Parameters
```python
INPUT_TYPES = {
    "required": {},
    "optional": {
        "crossfade_duration_ms": ("INT", {
            "default": 50, "min": 0, "max": 500, "step": 10,
            "tooltip": "Crossfade duration in milliseconds for smooth transitions between segments"
        }),
        "crossfade_curve": (["linear", "cosine", "exponential"], {
            "default": "linear",
            "tooltip": "Crossfade curve type: linear (constant), cosine (smooth), exponential (sharp)"
        }),
        "adaptive_crossfade": ("BOOLEAN", {
            "default": False,
            "tooltip": "Automatically adjust crossfade duration based on segment size"
        }),
        "enable_cache": ("BOOLEAN", {
            "default": True,
            "tooltip": "Cache F5-TTS generation to speed up subsequent runs with identical parameters"
        }),
        "cache_size_limit": ("INT", {
            "default": 100, "min": 10, "max": 1000,
            "tooltip": "Maximum number of cached audio segments to store in memory"
        })
    }
}

RETURN_TYPES = ("F5TTS_EDIT_OPTIONS",)
RETURN_NAMES = ("edit_options",)
```

#### Main Node Updates
- **Add optional input:** `edit_options`: `("F5TTS_EDIT_OPTIONS", {"tooltip": "Optional advanced editing options"})`
- **Remove NO advanced parameters** from main node interface
- **Use defaults when options node not connected**

### Phase 2: Implement Caching System

#### Cache Infrastructure (Copy from SRT Node)
```python
# Global cache in f5tts_edit_node.py
GLOBAL_F5TTS_EDIT_CACHE = {}

def _generate_edit_cache_key(self, audio_hash: str, original_text: str, target_text: str, 
                           edit_regions: List[Tuple[float, float]], fix_durations: Optional[List[float]],
                           temperature: float, speed: float, target_rms: float, nfe_step: int,
                           cfg_strength: float, sway_sampling_coef: float, ode_method: str,
                           model_name: str) -> str:
    """Generate cache key for F5-TTS edit based on generation parameters."""
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
    cache_string = str(sorted(cache_data.items()))
    return hashlib.md5(cache_string.encode()).hexdigest()
```

#### Smart Cache Invalidation
- **Cache F5-TTS generation only** (before compositing)
- **Crossfade/compositing changes = no regeneration needed**
- **Cache based on audio content hash + F5-TTS parameters**
- **LRU eviction when cache size limit reached**

### Phase 3: Advanced Crossfade Implementation

#### Update Audio Compositing
```python
# core/audio_compositing.py updates
def composite_edited_audio(original_audio: torch.Tensor, generated_audio: torch.Tensor, 
                          edit_regions: List[Tuple[float, float]], sample_rate: int, 
                          crossfade_duration_ms: int = 50, 
                          crossfade_curve: str = "linear",
                          adaptive_crossfade: bool = False) -> torch.Tensor:
```

#### Crossfade Curve Types
```python
def _apply_crossfade_curve(fade_length: int, curve_type: str, device) -> torch.Tensor:
    """Generate crossfade weights based on curve type."""
    if curve_type == "linear":
        return torch.linspace(0.0, 1.0, fade_length, device=device)
    elif curve_type == "cosine":
        t = torch.linspace(0, np.pi/2, fade_length, device=device)
        return torch.sin(t)
    elif curve_type == "exponential":
        t = torch.linspace(0, 1, fade_length, device=device)
        return t ** 2
```

#### Adaptive Crossfade Logic
```python
def _calculate_adaptive_crossfade(segment_duration: float, base_crossfade_ms: int) -> int:
    """Calculate adaptive crossfade duration based on segment size."""
    if segment_duration < 0.5:  # Very short segments
        return min(int(segment_duration * 1000 * 0.3), base_crossfade_ms * 2)
    elif segment_duration < 1.0:  # Short segments  
        return min(int(segment_duration * 1000 * 0.2), base_crossfade_ms * 1.5)
    else:  # Normal segments
        return base_crossfade_ms
```

### Phase 4: File Structure & Organization

#### New Files to Create
```
nodes/f5tts_edit_options_node.py     # Options node implementation
core/f5tts_edit_cache.py             # Cache management utilities
```

#### Files to Modify
```
nodes/f5tts_edit_node.py             # Add options input, cache integration
core/audio_compositing.py            # Enhanced crossfade functionality
core/f5tts_edit_engine.py            # Cache integration
nodes/__init__.py                    # Register new options node
```

### Phase 5: Testing Strategy

#### Test Cases for Smooth Transitions
1. **Problematic segments:** "55" to "7 year", short word replacements
2. **Different crossfade settings:** 0ms, 50ms, 100ms, 200ms
3. **Different curve types:** linear vs cosine vs exponential
4. **Adaptive mode:** on/off comparison
5. **Cache performance:** generation time with/without cache

#### Success Criteria
- ✅ **No stuttering/clipping** at segment boundaries
- ✅ **Fast iteration** when adjusting crossfade settings (cached generation)
- ✅ **Clean main node interface** (advanced options separate)
- ✅ **Cache hit rate >80%** for repeated operations
- ✅ **Smooth transitions** for segments <1 second

### Phase 6: Implementation Order

1. **First:** Create options node with basic parameters
2. **Second:** Implement cache infrastructure in edit engine
3. **Third:** Enhanced crossfade curves in audio compositing
4. **Fourth:** Adaptive crossfade logic
5. **Fifth:** Integration testing and tuning
6. **Sixth:** Update main node to use options input

### Expected Benefits

- ✅ **Eliminates stuttering** with configurable crossfade
- ✅ **Fast experimentation** with caching system
- ✅ **Clean UX** following Audio Analyzer pattern
- ✅ **Professional results** with advanced crossfade curves
- ✅ **Scales to complex edits** with adaptive behavior

## Technical Notes

### Cache Key Components
- Audio content hash (MD5 of waveform)
- All F5-TTS generation parameters
- Model name and version
- Edit regions and durations

### Memory Management
- LRU cache eviction
- Configurable cache size limit
- Memory usage monitoring
- Cache statistics in edit_info

### Backward Compatibility
- Main node works without options node (uses defaults)
- Existing workflows continue to function
- Old node remains available for comparison

## Implementation Status
- [ ] Phase 1: Options node creation
- [ ] Phase 2: Cache system implementation  
- [ ] Phase 3: Advanced crossfade features
- [ ] Phase 4: Integration and testing
- [ ] Phase 5: Documentation and cleanup

---
*Created for F5-TTS Edit Node improvement project*
*Target: Eliminate boundary stuttering and improve iteration speed*