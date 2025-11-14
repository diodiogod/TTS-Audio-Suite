# TTS Audio Suite Generation/Audio Caching System - Comprehensive Analysis

## Executive Summary

The TTS Audio Suite has **MULTIPLE FRAGMENTED CACHING SYSTEMS** operating independently with inconsistent cache key construction. The generation cache is hard to fix and often misses because:

1. **Multiple independent cache implementations** (at least 3 separate systems)
2. **Inconsistent cache key generation** across engines
3. **Missing parameter normalization** causing floating-point precision issues
4. **Undocumented parameter handling** with hidden transformations
5. **Multiple cache levels** (local, unified, instance-level) not coordinated
6. **Text transformation side-effects** affecting cache keys

---

## 1. CURRENT CACHING SYSTEMS

### System 1: Unified Audio Cache (Primary but underused)
**Location:** `/home/linux/ComfyUI_TTS_Audio_Suite/utils/audio/cache.py`

- **Global Storage:** `GLOBAL_AUDIO_CACHE = {}` (line 13)
- **Class:** `AudioCache` (line 235)
- **Access:** `get_audio_cache()` function (line 343)
- **Key Generator:** Abstract `CacheKeyGenerator` with engine-specific implementations
  - `F5TTSCacheKeyGenerator` (line 25)
  - `ChatterBoxCacheKeyGenerator` (line 50)
  - `ChatterBoxOfficial23LangCacheKeyGenerator` (line 76)
  - `HiggsAudioCacheKeyGenerator` (line 103)
  - `VibeVoiceCacheKeyGenerator` (line 144)
  - `IndexTTSCacheKeyGenerator` (line 184)

**Issue:** This unified system exists but is ONLY used in ChatterBox TTS node through `create_cache_function()`. Most nodes use local caches instead.

### System 2: Node-Local Audio Cache (ChatterBox)
**Location:** `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/chatterbox/chatterbox_srt_node.py`

- **Global Storage:** `GLOBAL_AUDIO_CACHE = {}` (line 46)
- **Separate from unified cache** - different dict, different module
- **Methods:**
  - `_generate_segment_cache_key()` (line 361)
  - `_get_cached_segment_audio()` (line 381)
  - `_cache_segment_audio()` (line 385)

**Issue:** This is a DUPLICATE system that doesn't use the unified cache. Creates cache misses when switching between unified node and legacy node.

### System 3: Voice Conversion Iteration Cache
**Location:** `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/unified/voice_changer_node.py`

- **Global Storage:** `GLOBAL_RVC_ITERATION_CACHE = {}` (line 48)
- **Purpose:** Cache iterative refinement passes for RVC voice conversion
- **Methods:**
  - `_generate_rvc_cache_key()` (line 581)
  - `_get_cached_rvc_iterations()` (line 620)
  - `_cache_rvc_result()` (line 628)

**Issue:** RVC-specific, limited to 5 iterations max (line 634), not integrated with audio cache.

### System 4: Engine Instance Cache (Multiple locations)
**Location:** `tts_text_node.py` (line 134), `voice_changer_node.py` (line 90)

- **Storage:** Instance variable `self._cached_engine_instances = {}`
- **Purpose:** Cache engine node instances to prevent model reloading
- **Cache Key:** Hash of `engine_type` + stable parameters (line 182)
- **Timestamp-based invalidation:** Uses global `_global_cache_invalidation_flag`

**Issue:** This is ENGINE caching, not GENERATION caching. Confuses the user about what's being cached.

### System 5: Voice Discovery Cache (Persistent)
**Location:** `/home/linux/ComfyUI_TTS_Audio_Suite/utils/voice/cache_manager.py`

- **Storage:** Disk-based JSON in `.cache/voice_discovery.json`
- **Purpose:** Cache voice character discovery to speed up initialization
- **Background refresh:** Uses threading to update cache after ComfyUI loads

**Issue:** This is voice DISCOVERY cache, not generation audio cache.

---

## 2. CACHE KEY CONSTRUCTION - DETAILED ANALYSIS

### ChatterBox Unified Cache Key
**File:** `/home/linux/ComfyUI_TTS_Audio_Suite/utils/audio/cache.py:50-73`

```python
cache_data = {
    'text': params.get('text', ''),                    # FULL TEXT WITH CHARACTER PREFIX
    'exaggeration': params.get('exaggeration', 1.0),
    'temperature': params.get('temperature', 0.8),
    'cfg_weight': params.get('cfg_weight', 1.0),
    'seed': params.get('seed', 0),
    'audio_component': params.get('audio_component', ''),  # HASHED AUDIO CONTENT
    'model_source': params.get('model_source', ''),        # LANGUAGE + SOURCE
    'device': params.get('device', ''),                     # NOT RESOLVED (can vary)
    'language': params.get('language', 'English'),
    'character': params.get('character', 'narrator'),
    'engine': 'chatterbox',
    'repetition_penalty': params.get('repetition_penalty', 1.0),
    'min_p': params.get('min_p', 0.0),
    'top_p': params.get('top_p', 1.0)
}
cache_string = str(sorted(cache_data.items()))
cache_key = hashlib.md5(cache_string.encode()).hexdigest()
```

**Problems:**
1. **No floating-point rounding** - 0.8 vs 0.80000001 = different keys
2. **`audio_component` is hashed waveform** - but text still includes character prefix (duplication)
3. **Device resolution inconsistency** - stores "auto" literally, causing misses when it resolves to "cuda"
4. **Character prefixed to text** - line 282 of cache.py: `cache_params['text'] = f"{cache_params.get('character', 'narrator')}:{text_content}"`

### ChatterBox Legacy Cache Key
**File:** `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/chatterbox/chatterbox_srt_node.py:361-379`

```python
cache_data = {
    'text': subtitle_text,
    'exaggeration': exaggeration,
    'temperature': temperature,
    'cfg_weight': cfg_weight,
    'seed': seed,
    'audio_prompt_component': audio_prompt_component,
    'model_source': model_source,
    'device': device,
    'language': language,
    'engine': 'chatterbox_srt'
}
cache_string = str(sorted(cache_data.items()))
cache_key = hashlib.md5(cache_string.encode()).hexdigest()
```

**Problems:**
1. Uses `chatterbox_srt` as engine (different from `chatterbox`)
2. No character prefix in text (different from unified)
3. Stores different parameters (no repetition_penalty, min_p, top_p)
4. **Result: Cache misses when using SRT node vs TTS node**

### F5-TTS Unified Cache Key
**File:** `/home/linux/ComfyUI_TTS_Audio_Suite/utils/audio/cache.py:25-47`

```python
cache_data = {
    'text': params.get('text', ''),
    'model_name': params.get('model_name', ''),        # INCLUDES MODEL NAME
    'device': params.get('device', ''),
    'audio_component': params.get('audio_component', ''),
    'ref_text': params.get('ref_text', ''),            # REFERENCE TEXT (!)
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
```

**Critical Issues:**
1. **Includes `ref_text` in cache key** - Same text with different reference audio = different key (CORRECT)
2. **No floating-point rounding** on ANY of the 9 float parameters
3. **Device not resolved** - "auto" won't match "cuda"
4. **`cross_fade_duration`** is POST-PROCESSING, shouldn't be in key (doesn't affect generation)
5. **`audio_component`** should be enough to distinguish references, shouldn't also need ref_text

### VibeVoice Cache Key
**File:** `/home/linux/ComfyUI_TTS_Audio_Suite/utils/audio/cache.py:144-181`

```python
# FLOATING POINT ROUNDING PRESENT (but inconsistent)
cfg_scale = round(float(cfg_scale), 3)
temperature = round(float(temperature), 3)
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
    'inference_steps': params.get('inference_steps', 20),
    'attention_mode': params.get('attention_mode', 'eager'),
    'quantize': params.get('quantize', False),
    'engine': 'vibevoice'
}
```

**Good:** Has rounding for floats
**Bad:** Only rounds 3 params, misses others. Device still "auto". Includes post-processing params like attention_mode (affects inference, not generation output).

### IndexTTS Cache Key
**File:** `/home/linux/ComfyUI_TTS_Audio_Suite/utils/audio/cache.py:184-232`

```python
# SELECTIVE ROUNDING (inconsistent)
temperature = round(float(temperature), 3)
top_p = round(float(top_p), 3)
emotion_alpha = round(float(emotion_alpha), 3)

cache_data = {
    'text': params.get('text', ''),
    'speaker_audio': params.get('speaker_audio', ''),      # AUDIO PATH STRING
    'emotion_audio': params.get('emotion_audio', ''),      # AUDIO PATH STRING
    'emotion_alpha': emotion_alpha,
    'emotion_vector': params.get('emotion_vector'),        # LIST (unhashable!)
    'use_emotion_text': params.get('use_emotion_text', False),
    'emotion_text': params.get('emotion_text', ''),
    'use_random': params.get('use_random', False),
    'seed': params.get('seed', 0),
    'do_sample': params.get('do_sample', True),
    'temperature': temperature,
    'top_p': top_p,
    'top_k': params.get('top_k', 30),
    'length_penalty': params.get('length_penalty', 0.0),   # NOT ROUNDED!
    'num_beams': params.get('num_beams', 3),
    'repetition_penalty': params.get('repetition_penalty', 10.0),  # NOT ROUNDED!
    'max_mel_tokens': params.get('max_mel_tokens', 1500),
    'max_text_tokens_per_segment': params.get('max_text_tokens_per_segment', 120),
    'interval_silence': params.get('interval_silence', 200),
    'model_name': params.get('model_name', 'IndexTTS-2'),
    'device': params.get('device', 'auto'),
    'character': params.get('character', 'narrator'),
    'use_torch_compile': params.get('use_torch_compile', False),
    'use_accel': params.get('use_accel', False),
    'stream_return': params.get('stream_return', False),
    'more_segment_before': params.get('more_segment_before', 0),
    'engine': 'index_tts'
}
```

**Critical Issues:**
1. **Stores audio paths as strings** - temp paths will vary, causing cache misses
2. **`emotion_vector` is a list** - unhashable, str() conversion is fragile
3. **Selective float rounding** - only 3 params rounded, others like `length_penalty`, `repetition_penalty` are not
4. **Mixed use of AUDIO and PATH** - speaker_audio and emotion_audio are path strings, not hashes

---

## 3. CACHE MISS ROOT CAUSES

### Root Cause #1: Duplicate Cache Systems
Two separate `GLOBAL_AUDIO_CACHE` dicts exist:
- `/home/linux/ComfyUI_TTS_Audio_Suite/utils/audio/cache.py:13`
- `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/chatterbox/chatterbox_srt_node.py:46`
- `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/f5tts/f5tts_srt_node.py:43`

**Result:** Using unified TTS Text node and then SRT node = cache miss (different dicts)

### Root Cause #2: Floating-Point Precision
Example from ChatterBox:
```python
# Temperature=0.8 in unified cache
temperature = params.get('temperature', 0.8)  # Key generated with 0.8

# Temperature from ComfyUI slider
temperature = 0.80000001  # Different key!
```

**Affected engines:** ChatterBox, F5-TTS (all 9 float params), all others without rounding

### Root Cause #3: Device Resolution Inconsistency
```python
# Unified cache stores device literally
'device': params.get('device', ''),  # Stores "auto"

# But actual resolution varies
tts_text_node.py line 159:
from utils.device import resolve_torch_device
resolved_device = resolve_torch_device(config.get('device', 'auto'))  # Returns "cuda"
```

**Result:** Engine cache key includes resolved device, generation cache key includes "auto". Mismatch!

### Root Cause #4: Text Transformation Side-Effects
**In `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/chatterbox/chatterbox_tts_node.py`:**

Line 282 of unified cache adds character prefix:
```python
cache_params['text'] = f"{cache_params.get('character', 'narrator')}:{text_content}"
```

But line 449 applies crash protection AFTER cache key generation:
```python
protected_text = self._pad_short_text_for_chatterbox(processed_text, crash_protection_template)
# Cache key was generated with unpadded text
# But generation uses padded text
```

**Result:** 
- Cache key generated: `narrator:hello`
- Cache lookup tries: `narrator:hello`
- Cache miss because audio was generated with: `narrator:hmm ,, hello hmm ,,`

### Root Cause #5: Audio Component Hash Inconsistency
```python
# chatterbox_tts_node.py line 603-605
stable_audio_component = self._generate_stable_audio_component(
    inputs.get("reference_audio"), inputs.get("audio_prompt_path", "")
)

# Generates hash from WAVEFORM CONTENT
# But if same file accessed via:
# 1. Dropdown (direct path)
# 2. Character Voices node (copied to temp)
# = Different file path = Different hash = CACHE MISS
```

### Root Cause #6: Parameter Duplication in Cache Key
**In F5-TTS cache key:**
```python
'audio_component': params.get('audio_component', ''),  # Hash of waveform
'ref_text': params.get('ref_text', ''),                # Reference text
```

If same text with different reference audio:
- audio_component differs ✓ (good - prevents collision)
- But if reference audio is different FILE with same CONTENT:
  - audio_component might be same ✓
  - ref_text is same ✓
  - But actual voice characteristics differ ✗

### Root Cause #7: Untracked Parameter Transformations
```python
# chatterbox_tts_node.py line 434-436
processed_text, pause_segments = PauseTagProcessor.preprocess_text_with_pause_tags(
    inputs["text"], True
)
# Cache key uses processed_text, but what if PauseTagProcessor changes the text?
# [pause:500ms] removed? Transformed? Cache key would differ!
```

### Root Cause #8: Engine Instance Cache vs Generation Cache Confusion
```python
# tts_text_node.py line 182
cache_key = f"{engine_type}_{hashlib.md5(str(sorted(stable_params.items())).encode()).hexdigest()[:8]}"
# This is ENGINE instance cache

# But users think they're caching generation results
# They're actually just reusing the same engine model
# Generation parameters affect the OUTPUT, not the CACHE KEY
```

---

## 4. PARAMETER ANALYSIS: What Should/Shouldn't Invalidate Cache

### ChatterBox Parameters

| Parameter | Should Invalidate? | Current Behavior | Issue |
|-----------|-------------------|------------------|-------|
| `text` | YES | YES | Text transformation side-effects |
| `character` | YES | YES | Included in cache key |
| `exaggeration` | YES | YES | No rounding |
| `temperature` | YES | YES | No rounding - floating point issues |
| `cfg_weight` | YES | YES | No rounding |
| `seed` | YES | YES | ✓ Correct |
| `audio_component` | YES | YES | ✓ Hashed correctly |
| `language` | YES | YES | ✓ Correct |
| `device` | NO* | YES | *Should NOT - output doesn't differ |
| `reference_audio` | YES (indirect) | YES | Via audio_component ✓ |
| `crash_protection_template` | YES | NO* | *MISSING - changes text! |
| `enable_chunking` | MAYBE | NO | Pre-processing, doesn't affect segment cache |
| `chunk_combination_method` | NO | NO | Post-processing ✓ |
| `silence_between_chunks_ms` | NO | NO | Post-processing ✓ |
| `batch_size` | NO | NO | Streaming mode, doesn't affect generation ✓ |

### F5-TTS Parameters

| Parameter | Should Invalidate? | Current Behavior | Issue |
|-----------|-------------------|------------------|-------|
| `text` | YES | YES | ✓ |
| `ref_text` | YES | YES | ✓ Correct - different reference = different voice |
| `model_name` | YES | YES | ✓ Different model |
| `seed` | YES | YES | ✓ |
| `temperature` | YES | YES | No rounding |
| `speed` | NO* | YES | *WRONG - doesn't affect generation, F5 handles this |
| `target_rms` | NO* | YES | *WRONG - post-processing normalization |
| `cfg_strength` | YES | YES | Affects generation quality |
| `nfe_step` | YES | YES | Affects generation quality |
| `cross_fade_duration` | NO | YES | *WRONG - post-processing |
| `audio_component` | YES | YES | ✓ Reference audio identifier |
| `character` | YES | YES | ✓ |
| `device` | NO | YES | *WRONG - doesn't affect generation |

---

## 5. DESIGN FLAWS IDENTIFIED

### Flaw #1: Architectural Fragmentation
**Problem:** 5 independent cache systems with no coordination
- Unified cache underused
- Legacy SRT nodes use separate cache
- Voice conversion has separate cache
- Engine instance cache conflates with generation cache
- Voice discovery cache is orthogonal

**Impact:** Cache misses when switching between node types, impossible to manage/clear all caches

### Flaw #2: Floating-Point Precision Not Normalized
**Problem:** 
```python
# Different rounding across engines
VibeVoice: round(x, 3)
IndexTTS: round(x, 3) but only 3 params
ChatterBox: No rounding
F5-TTS: No rounding
```

**Impact:** Workflow parameter change from UI slider (0.8) vs code (0.80000001) = cache miss

### Flaw #3: Device Parameter Included in Cache Key
**Problem:**
```python
# Unified cache stores device
'device': params.get('device', ''),

# But device="auto" + resolved device="cuda" = different keys
# Yet output is identical!
```

**Impact:** Unnecessary cache misses

### Flaw #4: Text Pre-Processing Side-Effects Not Tracked
**Problems:**
1. Crash protection padding happens AFTER cache key generation
2. Pause tag preprocessing happens AFTER cache key generation
3. Character prefix added during cache key generation
4. Text chunking happens separately

**Impact:** Cache key is for unprocessed text, but generation uses processed text

### Flaw #5: Audio Component Hashing Fragile
**Problem:**
```python
# audio_component = hash of waveform bytes
# If reference audio loaded different ways:
# 1. From dropdown -> actual file path
# 2. From Character Voices -> temp file
# 3. From direct audio input -> memory waveform
# = Same audio, potentially different hashes
```

**Impact:** Cache misses for same voice when referenced differently

### Flaw #6: Mixed Scopes in Cache Keys
**Problem:**
```python
# Per-segment cache parameters (text, audio, seed, temperature)
# vs
# Workflow-level parameters (device, model, language)
# = Same cache key namespace treats both equally

# Result: Changing device invalidates ALL segment caches
# But device doesn't affect generation!
```

### Flaw #7: No Cache Invalidation Strategy
**Problem:**
- Global caches cleared how? Only manual `audio_cache.clear_cache()`
- No TTL/LRU eviction
- Memory can grow unbounded
- No detection of "stale" cache entries after model updates

**Impact:** Users can't trust cache after model updates; memory leaks

### Flaw #8: Engine Instance Cache != Generation Cache
**Problem:**
```python
# Users expect:
# "Cache generation results for fast re-run"

# What actually happens:
# "Cache engine instance to avoid reloading model"

# Different concepts, but code conflates them!
```

**Impact:** Users confused why re-running with different parameters still reuses cache

---

## 6. SPECIFIC EXAMPLES OF CACHE MISSES

### Example 1: Temperature Slider Changes
```
User workflow:
1. Set temperature=0.8, generate (CACHE HIT expected next time)
2. Adjust slider to temperature=0.80000001
3. Run again, expect CACHE HIT
4. Result: CACHE MISS (different float representation)
```

### Example 2: Using Unified TTS vs Legacy SRT Node
```
User workflow:
1. Use "🎤 TTS Text" (unified) node with ChatterBox engine
   -> Audio stored in utils/audio/cache.py:GLOBAL_AUDIO_CACHE
2. Switch to "ChatterBox TTS SRT" legacy node
   -> Looks in nodes/chatterbox/chatterbox_srt_node.py:GLOBAL_AUDIO_CACHE
3. Result: CACHE MISS (different dict)
```

### Example 3: Crash Protection Changes Cache
```
User workflow:
1. Generate with default crash_protection_template="hmm ,, {seg} hmm ,,"
   Cache key: hash("narrator:hello")
   Audio generated from: "narrator:hmm ,, hello hmm ,,"
   
2. Change crash_protection_template="" (disable padding)
3. Try to regenerate same text "hello"
4. Result: CACHE HIT returns audio for "hmm ,, hello hmm ,,"
   But user expects audio for "hello"
   = WRONG AUDIO RETURNED
```

### Example 4: Device Parameter Mismatch
```
User workflow:
1. Set device="auto" on first run
   Engine instance cache key: "f5tts_abc123"
   Generation cache key includes 'device': 'auto'
   
2. Device resolves to "cuda" internally
   
3. Switch device="cuda" explicitly
   Engine instance cache key: "f5tts_abc123" (same!)
   Generation cache key now includes 'device': 'cuda' (different!)
   
4. Result: Engine reused (good) but generation cache misses (bad)
```

### Example 5: Reference Audio Path Variation
```
User workflow:
1. Use Character Voices node
   -> Reference audio hashed from temp file
   -> audio_component = "ref_audio_abc123_24000"
   -> Cache key includes this hash
   
2. Switch to dropdown selector (same voice file)
   -> Reference audio hashed from permanent file
   -> audio_component = "file_audio_def456_24000"
   -> DIFFERENT HASH despite same voice content!
   
3. Result: CACHE MISS
```

---

## 7. CURRENT IMPLEMENTATION PATTERNS

### Pattern 1: ChatterBox TTS Node (Good Model)
File: `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/chatterbox/chatterbox_tts_node.py:454-479`

```python
# Generates stable audio component ONCE
stable_audio_component = self._generate_stable_audio_component(...)

# Creates cache function with ALL parameters
cache_fn = create_cache_function(
    "chatterbox",
    character=character,
    exaggeration=exaggeration,
    # ... all params
    audio_component=audio_component,
)

# Uses cache function for lookup/store
cached_audio = cache_fn(protected_text)
if cached_audio:
    return cached_audio

audio = self.generate_tts_audio(protected_text, ...)
cache_fn(protected_text, audio_result=audio)
```

**Strengths:**
- Uses unified cache system
- Generates audio component once
- Uses cache_fn pattern

**Weaknesses:**
- Applies crash protection AFTER defining cache params
- Cache key includes protected text, but params generated with unprotected text
- No floating-point rounding

### Pattern 2: F5-TTS Node (Different Approach)
File: `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/f5tts/f5tts_node.py`

**Observation:** Doesn't implement caching at generation level! Instead:
- Relies on unified cache through base node
- Or relies on F5-TTS model-level caching internally

### Pattern 3: Voice Conversion (Iteration Caching)
File: `/home/linux/ComfyUI_TTS_Audio_Suite/nodes/unified/voice_changer_node.py:154-175`

```python
cache_key = self._generate_rvc_cache_key(processed_source_audio, rvc_model, config)
cached_iterations = self._get_cached_rvc_iterations(cache_key, refinement_passes)

if refinement_passes in cached_iterations:
    return cached_iterations[refinement_passes]  # Full cache hit

# Start from highest cached iteration
for i in range(refinement_passes, 0, -1):
    if i in cached_iterations:
        current_audio = cached_iterations[i][0]
        start_iteration = i
        break

# Continue from cached iteration
for iteration in range(start_iteration, refinement_passes):
    converted_audio = rvc_engine.convert_voice(...)
    self._cache_rvc_result(cache_key, iteration_num, ...)
```

**Strengths:**
- Smart iteration-level caching
- Can resume from partial progress
- Limited to 5 iterations (memory bound)

**Weaknesses:**
- Separate system from generation cache
- Not integrated with audio cache

---

## 8. ROOT CAUSE RANKING

| Rank | Root Cause | Impact | Difficulty to Fix |
|------|-----------|--------|-------------------|
| 1 | Multiple cache dicts (fragmentation) | Very High | Medium |
| 2 | No floating-point rounding | High | Low |
| 3 | Device parameter in cache key | High | Low |
| 4 | Text pre-processing side-effects | High | Medium |
| 5 | Crash protection not in cache key | Medium | Low |
| 6 | Audio component hashing fragile | Medium | Medium |
| 7 | Parameter duplication in F5-TTS key | Low | Low |
| 8 | No cache invalidation strategy | Medium | High |

---

## 9. DESIGN ISSUES SUMMARY

### Consolidated Cache Architecture Needed
Current: 5 independent systems
Desired: 1 unified system with compartments

### Parameter Normalization Missing
- No rounding of floats to consistent precision
- Device parameter should be resolved before cache key
- Text transformations should be applied before cache key generation

### Scope Separation Missing
- Per-segment parameters (text, seed, audio)
- Workflow parameters (device, model, language)
- Processing parameters (chunking, silence)

Should use:
- Segment-level cache key = text + audio + generation params
- Workflow-level identifier = device + model + language (separate tracking)

### Cache Key Construction Inconsistent
- F5-TTS includes too many parameters (speed, target_rms, cross_fade_duration)
- ChatterBox missing crash_protection in key
- IndexTTS stores audio paths instead of hashes
- No engine has consistent float rounding

---

## 10. RECOMMENDATIONS

### Short Term (Easy Fixes)
1. **Add float rounding** to all engines (3 decimal places)
2. **Remove device from generation cache key** (doesn't affect output)
3. **Remove cross_fade/target_rms/speed from F5-TTS cache key**
4. **Add crash_protection_template to ChatterBox cache key**
5. **Fix audio_component hashing** to be consistent regardless of load method

### Medium Term (Moderate Effort)
1. **Consolidate cache dicts** into single GLOBAL_AUDIO_CACHE
2. **Separate cache scopes:**
   - Segment cache key = hash(text + audio_component + generation_params)
   - Workflow tracking = (engine_type, model_name, device, language)
3. **Move text transformation BEFORE cache key generation**
4. **Implement cache size limits** with LRU eviction

### Long Term (Architectural)
1. **Unified cache system** with clear API
2. **Configuration-driven cache parameters** per engine (explicit list of what invalidates cache)
3. **Cache statistics and debugging** UI in ComfyUI
4. **TTL-based invalidation** after model updates
5. **Persistent cache option** for heavy workflows (disk-based)

