# TTS Audio Suite Cache Issues - Executive Summary

## The Problem

Generation cache is **hard to fix and often misses** due to **architectural fragmentation** and **inconsistent parameter handling**.

## Quick Facts

- **5 separate cache systems** operating independently
- **8 major root causes** identified with code locations
- **22 specific cache key construction issues** across engines
- **At least 5 real-world cache miss scenarios** documented

## Root Causes (Ranked by Impact)

### 1. Multiple Cache Dictionaries (CRITICAL)
**Files:**
- `/home/linux/ComfyUI_TTS_Audio_Suite/utils/audio/cache.py:13` (Unified cache)
- `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/chatterbox/chatterbox_srt_node.py:46` (Legacy ChatterBox)
- `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/f5tts/f5tts_srt_node.py:43` (Legacy F5-TTS)

**Problem:** Using unified "🎤 TTS Text" node then switching to "ChatterBox TTS SRT" legacy node = cache miss because they use different `GLOBAL_AUDIO_CACHE` dicts.

### 2. No Float Rounding (HIGH PRIORITY FIX)
**Affected files:**
- `/home/linux/ComfyUI_TTS_Audio_Suite/utils/audio/cache.py:50-73` (ChatterBox)
- `/home/linux/ComfyUI_TTS_Audio_Suite/utils/audio/cache.py:25-47` (F5-TTS)
- All legacy SRT nodes

**Problem:** Temperature=0.8 vs 0.80000001 = different cache keys despite being same parameter

**Impact:** Moving UI slider = cache miss

### 3. Device Parameter Inconsistency (HIGH)
**Problem:** Cache key stores "auto" but engine instance cache resolves to "cuda"
- Engine cache resolved: `resolve_torch_device("auto")` → "cuda"
- Generation cache literal: 'device': "auto"
- Result: Cache misses when switching explicit device selections

### 4. Text Pre-Processing Side-Effects (HIGH)
**File:** `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/chatterbox/chatterbox_tts_node.py:449`

**Problem:**
```
1. Crash protection padding applied AFTER cache key generation
2. Pause tag preprocessing applied separately
3. Cache key is for "hello", audio generated from "hmm ,, hello hmm ,,"
4. If crash_protection_template changed, wrong audio returned
```

### 5. Crash Protection Not In Cache Key (MEDIUM)
**File:** `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/chatterbox/chatterbox_tts_node.py`

**Problem:** Changing `crash_protection_template` doesn't invalidate cache
- Same text → same cache key
- But protection template changes actual text used for generation
- Result: Returns audio for different protected text

### 6. Audio Component Hashing Fragile (MEDIUM)
**File:** `/home/linux/ComfyUI_TTS_Audio_Suite/utils/audio/audio_hash.py:24`

**Problem:** Same voice file, different loading methods = different hashes
- From dropdown: hashes actual file → "file_audio_abc123"
- From Character Voices (temp): hashes temp file → "ref_audio_def456"
- From direct audio: hashes waveform → "ref_audio_ghi789"
- All are same audio, different hashes = cache misses

### 7. Parameter Duplication in F5-TTS (LOW)
**File:** `/home/linux/ComfyUI_TTS_Audio_Suite/utils/audio/cache.py:25-47`

**Problem:** F5-TTS cache key includes post-processing parameters
- `speed` doesn't affect generation (F5 handles this)
- `target_rms` is normalization (post-processing)
- `cross_fade_duration` is for combining chunks (post-processing)
- Changing these shouldn't invalidate cache, but does

### 8. No Cache Invalidation Strategy (MEDIUM)
**Problem:**
- No TTL/LRU eviction
- Memory grows unbounded
- No detection of "stale" entries after model updates
- Manual `audio_cache.clear_cache()` only clears unified cache, not legacy caches

## Design Issues Summary

### Issue #1: Architectural Fragmentation
```
Unified Cache (utils/audio/cache.py)
├─ ChatterBoxCacheKeyGenerator
├─ F5TTSCacheKeyGenerator
├─ VibeVoiceCacheKeyGenerator
└─ IndexTTSCacheKeyGenerator
   
Legacy ChatterBox Cache (nodes/chatterbox/chatterbox_srt_node.py)
   └─ Independent _generate_segment_cache_key()

Legacy F5-TTS Cache (nodes/f5tts/f5tts_srt_node.py)
   └─ Same as ChatterBox SRT

Voice Conversion Cache (nodes/unified/voice_changer_node.py)
   └─ GLOBAL_RVC_ITERATION_CACHE (separate dict)

Engine Instance Cache (tts_text_node.py)
   └─ self._cached_engine_instances (confused with generation cache)

Voice Discovery Cache (utils/voice/cache_manager.py)
   └─ Persistent disk cache (unrelated to generation)
```

**Result:** Can't centrally manage cache; users confused about what's cached

### Issue #2: Parameter Normalization Missing
No consistent float rounding across engines:
- VibeVoice: `round(x, 3)` for 3 params only
- IndexTTS: `round(x, 3)` for 3 params, others unrounded
- ChatterBox: No rounding
- F5-TTS: No rounding

### Issue #3: Scope Separation Missing
Cache key conflates different parameter types:
- **Per-segment** (text, seed, audio) - should invalidate cache
- **Workflow** (device, model, language) - should NOT invalidate individual segments
- **Processing** (chunking, silence) - should NOT invalidate cache

Currently all in same namespace, causing false invalidations.

## Real-World Examples

### Example 1: Temperature Slider (Most Common)
```
1. Set temperature=0.8, generate audio
2. Adjust slider slightly (0.800001)
3. Expect cache hit, get cache miss
```

### Example 2: Node Switching
```
1. Use "🎤 TTS Text" node → cache stored in utils/audio/cache.py
2. Switch to "ChatterBox TTS SRT" legacy node → looks in nodes/chatterbox/
3. Same text but different cache dict = miss
```

### Example 3: Crash Protection
```
1. Generate "hello" with crash_protection="hmm ,, {seg} hmm ,,"
   Audio generated from: "hmm ,, hello hmm ,,"
2. Disable crash protection and regenerate
3. Cache returns old audio with padding, user expects bare audio
```

### Example 4: Device Parameter
```
1. Use device="auto" on GPU machine
2. Set device="cuda" explicitly
3. Engine cache reuses (good) but generation cache misses (bad)
```

## Parameters That SHOULD/SHOULDN'T Invalidate Cache

### ChatterBox Should Invalidate
- text, character, exaggeration, temperature, cfg_weight, seed, audio_component, language
- **MISSING:** crash_protection_template (changes actual text!)

### ChatterBox Should NOT Invalidate
- device (doesn't affect generation output)
- chunk_combination_method, silence_between_chunks_ms (post-processing)
- batch_size (streaming mode setting)

### F5-TTS Should Invalidate
- text, ref_text, model_name, seed, temperature, cfg_strength, nfe_step, audio_component, character

### F5-TTS Should NOT Invalidate
- speed (F5 handles this, doesn't affect segment generation)
- target_rms (post-processing normalization)
- cross_fade_duration (post-processing for combining segments)
- device (doesn't affect generation output)

## Fix Priority

### Must Fix (Breaks functionality)
1. Add float rounding (trivial, high impact)
2. Consolidate cache dicts (medium effort, high impact)
3. Add crash_protection_template to cache key (trivial, high impact)

### Should Fix (Improves reliability)
4. Remove device from generation cache key (trivial)
5. Remove post-processing params from F5-TTS key (trivial)
6. Fix audio_component hashing consistency (medium effort)

### Nice to Have (Long-term improvements)
7. Implement cache size limits with LRU eviction
8. Add cache statistics/debugging UI
9. Implement TTL-based invalidation after model updates
10. Option for persistent (disk-based) cache

## Current Cache Key Examples

### ChatterBox (Unified, Line 50-73)
```python
cache_data = {
    'text': "narrator:protected_hello",     # Includes prefix + crash protection
    'exaggeration': 0.5,                    # NO ROUNDING
    'temperature': 0.8,                     # NO ROUNDING
    'cfg_weight': 0.5,                      # NO ROUNDING
    'seed': 42,
    'audio_component': 'file_audio_abc_24k',
    'model_source': 'chatterbox_english',
    'device': 'auto',                       # NOT RESOLVED
    'language': 'English',
    'character': 'narrator',
    'repetition_penalty': 1.0,              # NO ROUNDING
    'min_p': 0.0,                           # NO ROUNDING
    'top_p': 1.0,                           # NO ROUNDING
    'engine': 'chatterbox'
}
```

### F5-TTS (Unified, Line 25-47)
```python
cache_data = {
    'text': "narrator:hello",
    'model_name': 'F5TTS_Base',
    'device': 'auto',                       # NOT RESOLVED
    'audio_component': 'file_audio_def_24k',
    'ref_text': 'reference text here',      # Correct: different ref = different voice
    'temperature': 0.8,                     # NO ROUNDING
    'speed': 1.0,                           # WRONG: post-processing
    'target_rms': 0.1,                      # WRONG: post-processing
    'cross_fade_duration': 0.15,            # WRONG: post-processing
    'nfe_step': 32,
    'cfg_strength': 2.0,                    # NO ROUNDING
    'seed': 0,
    'character': 'narrator',
    'engine': 'f5tts'
}
```

### ChatterBox SRT Legacy (Line 361-379)
Uses separate cache dict, different engine name ('chatterbox_srt' vs 'chatterbox'),
different parameters → cache misses when switching nodes!

## File References

All specific issues traced to source:

**Unified Cache Core:**
- `/home/linux/ComfyUI_TTS_Audio_Suite/utils/audio/cache.py` (339 lines)
  - ChatterBoxCacheKeyGenerator (50-73)
  - F5TTSCacheKeyGenerator (25-47)
  - VibeVoiceCacheKeyGenerator (144-181)
  - IndexTTSCacheKeyGenerator (184-232)
  - AudioCache class (235-336)

**Audio Hashing:**
- `/home/linux/ComfyUI_TTS_Audio_Suite/utils/audio/audio_hash.py` (142 lines)
  - generate_stable_audio_component (24-126)

**Node Implementation:**
- `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/chatterbox/chatterbox_tts_node.py`
  - _pad_short_text_for_chatterbox (136-177) - text transformation
  - _generate_tts_with_pause_tags (408-546) - text preprocessing
  - Crash protection applied line 449 AFTER cache key generation

- `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/chatterbox/chatterbox_srt_node.py`
  - _generate_segment_cache_key (361-379) - separate cache logic

- `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/f5tts/f5tts_srt_node.py`
  - GLOBAL_AUDIO_CACHE (43) - duplicate dict

- `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/unified/tts_text_node.py`
  - _create_proper_engine_node_instance (136-441) - engine caching
  - GLOBAL_AUDIO_CACHE (44) - local storage

## Conclusion

The generation cache system has **good intentions but fragmented execution**. The unified cache exists but is underutilized; legacy implementations use separate dicts; engine caching is conflated with generation caching.

**Key insight:** The system needs **consolidation and normalization**, not complete redesign. All the pieces are there, they just need to work together consistently.

See CACHE_ANALYSIS.md for detailed technical breakdown with line numbers and code examples.
