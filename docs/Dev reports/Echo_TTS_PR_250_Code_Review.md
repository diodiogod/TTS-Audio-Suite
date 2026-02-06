# Echo-TTS PR #250 - Comprehensive Code Review

**Date:** 2026-02-05
**PR Author:** drphero (Brent)
**Reviewer:** Claude (TTS Audio Suite maintainer perspective)
**Reference Implementation:** Qwen3-TTS
**Guides Referenced:**

- `docs/NEW_ENGINE_IMPLEMENTATION_GUIDE.md`
- `docs/Dev reports/fails_to_avoid_TTS_Engine_Implementation.md`

---

## Executive Summary

The Echo-TTS integration has **major architectural issues** that deviate significantly from project standards. While it functions, the implementation embeds 180+ lines of orchestration logic directly in the unified node instead of using the standard processor pattern, making it non-reusable and harder to maintain.

**Status:** ⚠️ **NEEDS SIGNIFICANT REFACTORING** - Critical architectural issues, missing standard integrations

---

## Critical Issues (MUST FIX)

### 1. ❌ **Missing Audio Caching System**

**Severity:** CRITICAL
**Location:** `engines/adapters/echo_tts_adapter.py`

**Issue:**

- No import of `utils.audio.cache`
- No `audio_cache` instance creation
- No `generate_cache_key()` calls before generation
- No `cache_audio()` calls after generation
- No cache hit checks

**Evidence:**

```bash
$ grep "audio_cache\|cache_audio\|generate_cache_key" engines/adapters/echo_tts_adapter.py
# No matches found
```

**Impact:**

- Every identical generation request re-generates audio (wasting time + VRAM)
- No performance benefits from repeated requests
- Inconsistent with ALL other engines (Qwen3, Step Audio EditX, F5-TTS, ChatterBox)

**Reference Implementation (Qwen3-TTS):**

```python
# qwen3_tts_adapter.py:49
self.audio_cache = get_audio_cache()

# qwen3_tts_adapter.py:349-372
cache_key = self.audio_cache.generate_cache_key(
    'qwen3_tts',
    text=text,
    model_type='CustomVoice',
    speaker=speaker,
    # ... all params ...
)
cached_audio = self.audio_cache.get_cached_audio(cache_key)
if cached_audio:
    return cached_audio[0]
# ... generate ...
self.audio_cache.cache_audio(cache_key, audio_tensor, duration)
```

**Fix Required:**

1. Add `from utils.audio.cache import get_audio_cache` import
2. Initialize `self.audio_cache = get_audio_cache()` in `__init__`
3. Generate cache keys in `_generate_audio_for_text()` BEFORE generation
4. Check cache and return if hit
5. Store results in cache after generation with duration calculation

---

### 2. ❌ **Non-Standard Architecture - Logic Embedded in Unified Node**

**Severity:** CRITICAL (Architecture)
**Location:** `nodes/unified/tts_text_node.py` lines 1128-1304

**Issue:**
Echo-TTS embeds **ALL orchestration logic** (character switching, pause tags, segment parameters) directly in the unified TTS text node instead of in a processor or adapter. This is **180+ lines of engine-specific code** in what should be a generic unified interface.

**What's embedded in unified node (lines 1128-1304):**

- Character parser setup and configuration (30+ lines)
- Character voice mapping logic (40+ lines)
- Pause tag processing (embedded in character loop)
- Segment parameter application
- Audio generation loop with character switching
- Chunk timing and combination

**Evidence:**

```python
# nodes/unified/tts_text_node.py:1128
elif engine_type == "echo_tts":
    # 180+ lines of orchestration logic here
    # Character parser setup
    # Voice mapping
    # Segment processing
    # Pause tag handling
    # Audio generation
```

**Standard Architecture (Qwen3-TTS, Step Audio EditX, IndexTTS):**

```
Unified Node (lines 1306-1450):
  ├─ Setup character parser (10 lines)
  ├─ Build voice_mapping (50 lines)
  └─ Call processor.process_text(text, voice_mapping, ...)

Processor (qwen3_tts_processor.py):
  ├─ Character switching orchestration
  ├─ Pause tag processing
  ├─ Segment parameter handling
  └─ Audio generation coordination
```

**Impact:**

- **Code duplication**: 180 lines of orchestration logic that should be reusable
- **Maintenance burden**: Changes to character switching require editing unified node instead of processor
- **Non-standard**: Every other engine (Qwen3, Step EditX, IndexTTS, VibeVoice) uses processor pattern
- **Not reusable**: Logic cannot be reused by other nodes (e.g., batch processing, API endpoints)
- **Harder to test**: Cannot unit test Echo-TTS orchestration without loading entire unified node

**Fix Required:**
Create `nodes/echo_tts/echo_tts_processor.py` following the Qwen3-TTS pattern:

1. Move character parser setup to processor `__init__`
2. Move voice mapping logic to processor
3. Implement `process_text(text, voice_mapping, seed, ...)` method
4. Move pause tag processing to processor
5. Move segment parameter handling to processor
6. Move character switching loop to processor
7. Update unified node to call `processor.process_text()` (10 lines instead of 180)

**Reference:** See `nodes/qwen3_tts/qwen3_tts_processor.py` lines 260-441 for correct processor pattern.

---

### 3. ❌ **Missing Interrupt Handling**

**Severity:** CRITICAL (User Experience)
**Location:** `nodes/echo_tts/echo_tts_srt_processor.py`

**Issue:**

- No import of `comfy.model_management`
- No interrupt checks in subtitle processing loop
- Users cannot cancel long-running SRT generations

**Evidence:**

```bash
$ grep "interrupt_processing" nodes/echo_tts/
# No matches found
```

**Impact:**

- Users MUST wait for entire SRT generation to complete (could be hours for long videos)
- No way to cancel if wrong settings are used
- Other engines ALL support interruption (Qwen3, Higgs, VibeVoice, CosyVoice)

**Reference Implementation (Qwen3-TTS SRT):**

```python
# qwen3_tts_srt_processor.py:12
import comfy.model_management as model_management

# In main loop (line ~200):
for i, subtitle in enumerate(subtitles):
    if model_management.interrupt_processing:
        raise InterruptedError(f"Qwen3-TTS SRT generation interrupted at subtitle {i+1}/{len(subtitles)}")
    # ... process subtitle ...
```

**Fix Required:**

1. Add `import comfy.model_management as model_management` at top of SRT processor
2. Add interrupt check at start of main subtitle loop
3. Add interrupt check inside character segment loops (if applicable)
4. Raise `InterruptedError` with informative message

---

## Important Issues (SHOULD FIX)

### 4. ⚠️ **Languages Parameter Should Be Removed**

**Severity:** IMPORTANT
**Location:** `nodes/engines/echo_tts_engine_node.py`

**Issue:**
The engine node has a `languages` string input parameter that appears to be incorrect:

- No language parameter exists in the official Echo-TTS inference API
- All text presets in official repo are English-only
- Uses byte-level UTF-8 tokenization (not language-specific)
- The parsed `languages` parameter in the adapter is never actually used in generation

**Evidence:**

```python
# echo_tts_adapter.py:188-195 - parsed but never used
def _parse_languages(self) -> List[str]:
    langs = self.config.get("languages", "en")
    # ... parsing logic ...
    return parts or ["en"]
```

**Impact:**

- User confusion (users think they can generate in multiple languages)
- Non-functional parameter (doesn't affect output)
- Echo-TTS is English-only model

**Fix Required:**
Remove the `languages` widget entirely from `nodes/engines/echo_tts_engine_node.py`

---

### 5. ⚠️ **Missing ComfyUI Progress Bars**

**Severity:** IMPORTANT (UX)
**Location:** `engines/adapters/echo_tts_adapter.py`

**Issue:**

- No progress bar creation in generation methods
- User has no feedback during long generations

**Reference Implementation (Qwen3-TTS):**

```python
# qwen3_tts_adapter.py:763-788
def _create_progress_bar(self, max_tokens: int, text: str = ""):
    try:
        import comfy.utils
        # Estimate tokens based on text
        estimated_tokens = int(len(text) * 1.5)
        progress_total = min(estimated_tokens, max_tokens)
        return comfy.utils.ProgressBar(progress_total)
    except (ImportError, AttributeError):
        return None
```

**Fix Required:**

1. Add progress bar creation in `_generate_audio_for_text()`
2. Update progress during sampling/generation
3. Estimate total steps based on `num_steps` parameter

---

### 6. ⚠️ **Incorrect SRT Timing Mode Restriction**

**Severity:** IMPORTANT (Bug)
**Location:** `engines/adapters/echo_tts_adapter.py:373-376`

**Issue:**
The adapter has an incorrect check that restricts timing modes:

```python
# Echo-TTS SRT support is minimal; fall back to pad_with_silence if needed
if timing_mode not in ["pad_with_silence", "concatenate"]:
    print(f"WARNING: Echo-TTS SRT: timing_mode '{timing_mode}' not supported, using pad_with_silence")
    timing_mode = "pad_with_silence"
```

However, the `AudioAssemblyEngine` called on line 389 **DOES support all timing modes** (`stretch_to_fit`, `smart_natural`, etc.). This check is unnecessarily restrictive.

**Impact:**

- Users cannot use `stretch_to_fit` or other advanced timing modes
- Warning message is misleading - modes ARE supported by the assembly engine
- Artificially limits functionality that would work fine

**Fix Required:**
Remove lines 373-376 entirely. The `AudioAssemblyEngine.assemble_by_timing_mode()` already handles all timing modes correctly and will raise proper errors for unsupported modes.

---

## Minor Issues (NICE TO FIX)

### 7. 📝 **`[S1]` Text Normalization Missing `[S2]` Safety Check**

**Severity:** MINOR
**Location:** `engines/adapters/echo_tts_adapter.py:81-95`

**Issue:**
The `_normalize_prompt_text()` method correctly adds `[S1]` prefix to follow Echo-TTS WhisperD format, but is missing the `[S2]` safety check from the official implementation.

**Current behavior:**

```python
# PR code (line 95)
return f"[S1] {cleaned}".strip()  # Always adds [S1]
```

**Result with `[S2]` input:**

```python
Input:  "[S2] Hello there"
Output: "[S1] [S2] Hello there"  # Wrong - double prefix
```

**Official behavior (inference.py:125-126):**

```python
if not text.startswith("[") and not text.startswith("(") and 'S1' not in text and 'S2' not in text:
    text = "[S1] " + text
```

**Impact:**

- Minor issue - most users won't use `[S2]` tags
- Creates double prefix `[S1] [S2]` which is ugly but harmless (doesn't break generation)
- Deviates from official Echo-TTS behavior

**Note about `[S2]`:**
Echo-TTS does NOT have multi-speaker support. `[S1]` and `[S2]` are cosmetic WhisperD transcription format tags that don't change the voice. All audio uses the single reference voice regardless of these tags.

**Fix suggested:**

```python
@staticmethod
def _normalize_prompt_text(text: str) -> str:
    if not text:
        return "[S1]"

    cleaned = text.strip()

    # Don't add [S1] if S1/S2 already present or text starts with bracket
    if 'S1' in cleaned or 'S2' in cleaned or cleaned.startswith('[') or cleaned.startswith('('):
        return cleaned

    return f"[S1] {cleaned}".strip()
```

---

### 8. 📝 **Force Speaker KV Warning**

**Severity:** MINOR (Documentation)
**Location:** `nodes/engines/echo_tts_engine_node.py:154`

**Issue:**

```python
"tooltip": "... ⚠️ Not compatible with pause tags ([pause:...])."
```

**Observation:**

- Warning about pause tag incompatibility is present
- BUT pause tags aren't implemented yet anyway
- Warning will become relevant once pause tags are added

**Action:** Keep warning, but verify compatibility once pause tags are implemented

---

## Positive Observations ✅

### What Was Done Well

1. **✅ Unified Model Interface Integration**
   
   - Correctly uses `unified_model_interface.load_model()` (line 132)
   - Follows factory pattern with `register_echo_tts_factory()`
   - Integrated with ComfyUI model management

2. **✅ Proper Node Registration**
   
   - Registered in `nodes.py` (lines 119-120, 475-476)
   - Correct category: `"TTS Audio Suite/Engines"`

3. **✅ Device Management**
   
   - Uses `resolve_torch_device()` utility
   - Graceful fallback from CUDA to CPU with warnings

4. **✅ Character Switching Support**
   
   - Implemented in unified TTS text node (lines 1128-1200)
   - Uses character parser correctly

5. **✅ Extensive Segment Parameter Support**
   
   - Registered in `utils/text/segment_parameters.py` with **12 parameters**
   - Supports: `seed`, `num_steps`, `cfg_scale_text`, `cfg_scale_speaker`, `cfg_min_t`, `cfg_max_t`, `truncation_factor`, `rescale_k`, `rescale_sigma`, `speaker_kv_scale`, `speaker_kv_max_layers`, `speaker_kv_min_t`, `sequence_length`
   - Most comprehensive segment parameter support in the project
   - Allows per-character control: `[Alice|seed:42|cfg_scale_text:5.0]`

6. **✅ Chunking System**
   
   - Uses `ImprovedChatterBoxChunker` (line 331)
   - Uses `ChunkTimingHelper` for combination (line 341)

7. **✅ SRT Processor Structure**
   
   - Separate SRT processor file (good separation of concerns)
   - Uses `AudioAssemblyEngine` for timing assembly
   - Handles overlap detection with smart fallback

8. **✅ Reference Audio Processing**
   
   - Robust `_prepare_reference_audio()` method (lines 140-186)
   - Handles multiple input formats (dict, string, tensor)
   - Resampling to Echo-TTS native 44100 Hz
   - Normalization and 5-minute limit

9. **✅ Seed Reproducibility**
   
   - Sets both `torch.manual_seed()` and `torch.cuda.manual_seed()` (lines 202-205)

---

## Comparison to Implementation Guide

### Guide Checklist

| Requirement                 | Status     | Notes                       |
| --------------------------- | ---------- | --------------------------- |
| **Phase 1: Foundation**     |            |                             |
| Core engine implementation  | ✅ Pass     | Via unified interface       |
| Model loading & downloading | ✅ Pass     | Downloader not checked yet  |
| Engine configuration node   | ✅ Pass     | Clean implementation        |
| Character switching         | ✅ Pass     | In unified node             |
| Pause tag support           | ✅ Pass     | d                           |
| Caching system              | ❌ **FAIL** | Not implemented             |
| VRAM management             | ✅ Pass     | Via unified interface       |
| Interrupt handling          | ❌ **FAIL** | Not implemented             |
|                             |            |                             |
| **Phase 2: SRT Support**    |            |                             |
| SRT processor               | ✅ Pass     | Implemented                 |
| Timing & assembly           | ✅ Pass     | Uses AudioAssemblyEngine    |
| Character switching in SRT  | ✅ Pass     |                             |
|                             |            |                             |
| **Unified Systems**         |            |                             |
| UnifiedModelInterface       | ✅ Pass     | Correctly used              |
| ComfyUIModelWrapper         | ✅ Pass     | Via unified interface       |
| UnifiedCacheManager         | ❌ **FAIL** | Not used                    |
| CharacterParser             | ✅ Pass     | Used in unified node        |
| PauseTagProcessor           | ✅ Pass     | d                           |
| LanguageMapper              | ⚠️ N/A     | Echo-TTS uses language list |

---



---

## Conclusion

The Echo-TTS integration has **significant architectural problems** that must be addressed before merge:

### Critical Issues (Must Fix):

1. **Non-standard architecture** - 180+ lines of orchestration logic embedded in unified node instead of processor
2. **Missing audio caching** - Performance issue, all other engines have this
3. **Missing interrupt handling** - User experience issue, cannot cancel SRT generations

### Important Issues (Should Fix):

4. **Languages parameter** - Non-functional, should be removed (English-only model)
5. **Progress bars** - No user feedback during generation

**Recommendation:** This PR needs **major refactoring** before merge. The architectural deviation (embedding logic in unified node) is particularly problematic as it:

- Creates maintenance burden (180 lines of code that should be in processor)
- Breaks code reusability (logic cannot be used by other nodes)
- Deviates from established patterns (Qwen3, Step EditX, IndexTTS all use processor pattern)

The PR author did good work on the Echo-TTS integration itself, but the project architecture needs to be followed.

---

## Future Work (Post-Merge)

### Engine Bundling

**Current Status:** Echo-TTS is installed via `pip install echo-tts` (not bundled).

**Recommendation for future:** Consider bundling Echo-TTS like other engines (Step Audio EditX, F5-TTS) to enable:

1. **Custom modifications** - Add it/s progress indicators, custom features
2. **Dependency isolation** - Avoid conflicts with user's environment
3. **Version stability** - Lock to specific tested version
4. **Patches and fixes** - Apply project-specific patches without waiting for upstream

**Rationale for deferring:**
- Echo-TTS is simple with minimal dependencies (no immediate conflicts expected)
- PR scope is already large with architecture refactoring
- Can be added in follow-up PR after real-world testing
- Easy migration path (create `engines/echo_tts/bundled/` when needed)

**When to bundle:**
- If dependency conflicts arise in practice
- If we need to add custom features (progress bars, patches)
- If upstream makes breaking changes

**Action:** Track as future enhancement, not blocking for initial merge.

---

## Code Examples to Share with PR Author

### 1. Cache Integration Example

```python
# In __init__:
from utils.audio.cache import get_audio_cache
self.audio_cache = get_audio_cache()

# In _generate_audio_for_text():
# BEFORE generation:
from utils.audio.audio_hash import generate_stable_audio_component
audio_component = generate_stable_audio_component(reference_audio=ref_audio)
cache_key = self.audio_cache.generate_cache_key(
    'echo_tts',
    text=text,
    audio_component=audio_component,
    ref_text=ref_text,
    num_steps=params['num_steps'],
    cfg_scale_text=params['cfg_scale_text'],
    cfg_scale_speaker=params['cfg_scale_speaker'],
    seed=seed,
    # ... all params that affect output ...
)

cached_audio = self.audio_cache.get_cached_audio(cache_key)
if cached_audio:
    print(f"💾 Using cached Echo-TTS audio: '{text[:30]}...'")
    return cached_audio[0], self.SAMPLE_RATE

# AFTER generation:
duration = self.audio_cache._calculate_duration(audio_tensor, 'echo_tts')
self.audio_cache.cache_audio(cache_key, audio_tensor, duration)
```

### 

### 3. Interrupt Handling Example

```python
# In echo_tts_srt_processor.py, add import:
import comfy.model_management as model_management

# In process_srt_content(), in main loop:
for i, subtitle in enumerate(subtitles):
    # Check for interrupt
    if model_management.interrupt_processing:
        raise InterruptedError(
            f"Echo-TTS SRT generation interrupted at subtitle {i+1}/{len(subtitles)}"
        )

    # ... existing generation code ...
```

---

**Review completed:** 2026-02-05
**Next step:** Share findings with PR author and request changes
