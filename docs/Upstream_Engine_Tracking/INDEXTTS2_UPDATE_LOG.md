# IndexTTS-2 Updates Log

This document tracks updates applied to our bundled IndexTTS-2 code from the upstream repository.

## Update Process
1. **Check for upstream changes:** `cd IgnoredForGitHubDocs/For_reference/index_tts && git pull`
2. **Review changes:** `git diff OLD_COMMIT..NEW_COMMIT -- indextts/infer_v2.py`
3. **Apply relevant changes** to `engines/index_tts/indextts/infer_v2.py`
4. **Test functionality** with our IndexTTS-2 nodes
5. **Document changes** in this log

---

## 2025-09-18: Major Update - Cache & Emotion Improvements

**Reference commit range:** `8336824..64cb31a` (September 11 → September 18, 2025)
**Upstream commits reviewed:** 5 commits affecting `infer_v2.py`

### 🔧 Changes Applied:

#### 1. **Emotion Vector Normalization Function** (commit: 8aa8064)
- **Added:** `normalize_emo_vec()` helper function
- **Purpose:** Prevents voice identity loss from overly strong emotion settings
- **Implementation:**
  - Applies emotion bias factors to reduce problematic emotions
  - Caps total emotion sum to 0.8 maximum
  - Protects against extreme emotion vector values
- **User benefit:** Better emotion control without losing speaker characteristics

#### 2. **Audio Length Limiting** (commit: 0828dcb)
- **Added:** `_load_and_cut_audio()` helper function
- **Purpose:** Prevents memory/VRAM issues from overly long reference audio
- **Implementation:**
  - Automatically truncates audio to 15 seconds maximum
  - Supports different sample rates (16kHz, 22kHz)
  - Provides verbose logging of truncation
- **User benefit:** More stable generation with long reference audio files

#### 3. **Persistent Cache Buildup Fix** (commit: 64cb31a)
- **Added:** Proper cache clearing with `torch.cuda.empty_cache()`
- **Purpose:** Solves memory accumulation issues during multiple generations
- **Implementation:**
  - Clears old cache variables before loading new audio
  - Calls `torch.cuda.empty_cache()` to free GPU memory
  - Applied to both speaker and emotion cache invalidation
- **User benefit:** Better memory management during batch processing

#### 4. **BigVGAN CUDA Kernel Import Fix** (commits: ee23371, e409c4a)
- **Fixed:** Corrected import path for BigVGAN custom CUDA kernel
- **Changed:** `indextts.BigVGAN.alias_free_activation.cuda` → `indextts.s2mel.modules.bigvgan.alias_free_activation.cuda`
- **Purpose:** Ensures proper CUDA acceleration when available
- **User benefit:** Better performance with CUDA-enabled setups

### 📋 Implementation Status:

| Change | Applied to Bundled Code | Tested | Notes |
|--------|------------------------|--------|-------|
| `normalize_emo_vec()` | ✅ **Applied** | ⏳ Pending | Core emotion control improvement |
| `_load_and_cut_audio()` | ✅ **Applied** | ⏳ Pending | Memory stability fix |
| Cache clearing fixes | ✅ **Applied** | ⏳ Pending | Critical for batch processing |
| BigVGAN import fix | ✅ **Applied** | ⏳ Pending | Performance improvement |

### 🎯 User-Facing Improvements:
- **Better emotion control:** Emotion vectors now auto-normalize to prevent voice identity loss
- **Memory stability:** Long reference audio automatically truncated to prevent crashes
- **Batch processing:** Improved cache management for multiple generations
- **Performance:** Fixed CUDA kernel loading for faster inference

### 🧪 Testing Required:
- [ ] Emotion vector controls in radar chart widget
- [ ] Long reference audio handling (>15 seconds)
- [ ] Multiple sequential generations (cache clearing)
- [ ] CUDA performance with FP16 enabled

---

---

## 2025-11-06: Performance & Stability Improvements

**Reference commit range:** `64cb31a..1d5d079` (September 18 → November 6, 2025)
**Upstream commits reviewed:** 6 commits affecting `infer_v2.py`

### 🔧 New Changes Identified:

#### 1. **Empty Generator → IndexError Fix** (commit: 750d9d9)
- **Issue:** Non-streaming `infer()` would crash with IndexError on empty generator
- **Fix:** Wrapped generator list conversion in try/except, returns `None` on error
- **Changes to `infer_v2.py`:**
  - Lines 354-364: Added exception handling
  - Lines 680, 690: Changed `return` to `yield` for proper generator flow
- **Priority:** 🔴 **Critical** - Prevents runtime crashes

#### 2. **Streaming Implementation** (commit: b0c6ab8)
- **Added:** `stream_return` and `more_segment_before` parameters for low-latency streaming
- **Purpose:** Enable first-chunk audio output before full generation completes
- **Changes:**
  - `stream_return` (bool): Return generator instead of final result
  - `more_segment_before` (int): Apply extra segmentation for faster first output (0-80 recommended)
  - Modified `indextts/utils/front.py` for tokenization changes
- **Priority:** 🟡 **Important** - Enables streaming use cases

#### 3. **UNK Token Warnings** (commit: 34be9bf)
- **Added:** Detection and warning for unknown tokens in input text
- **Behavior:**
  - Detects tokens the BPE model cannot encode
  - Prints warnings with token IDs and suggested fixes
  - Helps users debug encoding issues
- **Changes to `infer_v2.py`:** Lines 450-459, added token validation logic
- **Priority:** 🟢 **Nice-to-have** - Improves user experience with error diagnosis

#### 4. **S2Mel Stage Optimization** (commit: 31e7e85)
- **Focus:** Optimized mel-spectrogram generation pipeline
- **Changes:**
  - `indextts/s2mel/modules/commons.py`: 15 lines added/modified
  - `indextts/s2mel/modules/diffusion_transformer.py`: 2 lines modified
  - `indextts/s2mel/modules/flow_matching.py`: 15 lines added (new optimization code)
  - `indextts/infer_v2.py`: 28 lines added/modified
- **Expected Impact:** Faster mel-spectrogram generation (stage 2 of 3)
- **Priority:** 🟡 **Important** - Performance improvement

#### 5. **GPT2 Inference Acceleration** (commit: c1ef414)
- **Focus:** Major performance optimization for GPT2 stage (stage 1)
- **Changes:**
  - **New module:** `indextts/accel/` directory (609 lines added):
    - `accel_engine.py`: Core acceleration logic (609 lines)
    - `attention.py`: Optimized attention mechanism (154 lines)
    - `gpt2_accel.py`: GPT2-specific optimizations (181 lines)
    - `kv_manager.py`: Key-value cache management (209 lines)
  - Modified `indextts/gpt/model_v2.py`: 63 lines changed (+47 new)
  - Modified `indextts/infer_v2.py`: 17 lines changed
- **Expected Impact:** Significantly faster GPT2 inference (first stage bottleneck)
- **Priority:** 🔴 **Critical** - Major performance gain, but new code complexity

#### 6. **Merge Commit** (commit: 5d67f62 → 1d5d079)
- Merge of acceleration branch into main
- Contains all above changes integrated

### 📊 Summary

| Change | Files Modified | Lines Changed | Priority | Status |
|--------|---------------|---------------|----------|--------|
| Empty generator fix | 1 | +12/-9 | 🔴 Critical | ⏳ Pending |
| Streaming support | 2 | +72/-18 | 🟡 Important | ⏳ Pending |
| UNK token warnings | 1 | +7 | 🟢 Nice-to-have | ⏳ Pending |
| S2Mel optimization | 4 | +54/-6 | 🟡 Important | ⏳ Pending |
| GPT2 acceleration | 6 | +1228/-14 | 🔴 Critical | ⏳ Pending |

### 🚨 Integration Concerns

1. **New acceleration module complexity** - The `indextts/accel/` package adds significant complexity with KV-cache management and specialized attention mechanisms. Needs careful testing.

2. **Potential API changes** - Streaming parameters (`stream_return`, `more_segment_before`) may require adapter updates if our wrapper doesn't expose these.

3. **GPU memory implications** - KV-cache optimization might have different memory profiles; needs profiling on different GPU setups.

4. **File structure expansion** - New `accel/` subdirectory adds to maintenance burden.

### 🧪 Testing Required

- [ ] Empty generator fix - test non-streaming inference with various inputs
- [ ] Streaming latency - measure first-chunk output time with `stream_return=True`
- [ ] GPT2 acceleration - benchmark inference speed (expect 1.5-3x speedup)
- [ ] S2Mel optimization - verify audio quality unchanged
- [ ] UNK token handling - test with multilingual/special character inputs
- [ ] Memory profiling - GPU memory usage with new KV-cache manager
- [ ] ComfyUI integration - ensure all changes work through our node interface

### ✅ Applied Changes Summary (2025-11-06)

**All upstream commits successfully integrated:**

1. ✅ **Empty generator fix (750d9d9)** - Refactored `infer()` to use generator pattern:
   - Created wrapper `infer()` method that delegates to `infer_generator()`
   - Refactored monolithic method into `infer_generator()` with yield statements
   - Changed final `return` statements to `yield` for generator compatibility
   - Handles both streaming (`stream_return=True`) and non-streaming modes

2. ✅ **Streaming implementation (b0c6ab8)** - Added streaming parameters:
   - Added `stream_return` parameter to `infer()` and `infer_generator()`
   - Added `more_segment_before` parameter for streaming segmentation
   - Integrated with generator-based inference for low-latency streaming

3. ✅ **UNK token warnings (34be9bf)** - Added token validation:
   - Added token validation in `infer_generator()`
   - Warns users about unknown tokens with suggestions

4. ✅ **S2Mel optimization (31e7e85)** - Applied all module updates:
   - `commons.py`: Updated `fused_add_tanh_sigmoid_multiply()` with torch.split
   - `diffusion_transformer.py`: Fixed `sequence_mask()` call with max_length parameter
   - `flow_matching.py`: Added `enable_torch_compile()` method to CFM class
   - `infer_v2.py`: Added `use_torch_compile` parameter and setup

5. ✅ **GPT2 acceleration (c1ef414)** - Applied complete acceleration module:
   - Created `/accel/` directory with 5 files (init, accel_engine, attention, gpt2_accel, kv_manager)
   - Modified `model_v2.py`: Added `use_accel` parameter to UnifiedVoice
   - Modified `model_v2.py`: Added accel engine initialization in `post_init_gpt2_config()`
   - Modified `model_v2.py`: Added accel-aware inference in `inference_speech()`

### 🔧 Custom Modifications Preserved

- QwenEmotion lazy loading with path fallbacks (lines 81-112 in infer_v2.py)
- W2V-BERT model loading from TTS folder with unified downloader (lines 145-166)
- All custom logging and error handling in infer_v2.py and throughout
- Custom model initialization sequence
- Custom emotion vector normalization and processing

---

---

## 2026-07-11: Upstream Audit Before Release

**Upstream repository checked:** `index-tts/index-tts` (`main`)
**Upstream head observed:** `b5bd657` (2026-07-08)
**Check performed:** 2026-07-11

### Relevant upstream changes reviewed

| Commit | Upstream change | Bundled status | Release action |
|---|---|---|---|
| `843972e` | Coerce QwenEmotion JSON emotion scores to `float` and reject non-numeric values clearly | **Applied** in bundled `clamp_score()` | Run a text-emotion generation with numeric-string classifier output |
| `b154a1b` | WebUI text/vector preset save/load management | **Already covered differently** by the ComfyUI-native preset manager and `emotion_presets.json` integration | No direct port needed |
| `b5bd657` | Expose `--accel` and `--torch-compile`, add optional acceleration extras and WebUI settings | **Applied selectively**; suite now forwards `use_accel` into the bundled GPT path, while retaining ComfyUI-owned dependency handling | Validate acceleration fallback on compatible and non-accelerated setups |
| `7264ce2` | Improve IndexTTS-2 model resource checks and HF cache handling | **Suite-owned downloader differs** and needs a separate comparison if download failures are reported | No blind copy into bundled code |

### Findings

- No upstream change was found that invalidates the current eight-emotion vector order,
  Qwen text-emotion syntax, audio-reference blending, or the suite's inline tag format.
- The upstream QwenEmotion string-score fix is directly relevant to the suite's text-emotion
  path and should be applied before a release.
- The upstream acceleration work is not a drop-in replacement because this repository
  bundles and adapts IndexTTS-2. The suite already contains the acceleration modules, but
  `utils/models/unified_model_interface.py` should be checked so `use_accel` reaches the
  bundled `IndexTTS2` constructor.
- Upstream WebUI presets are not copied verbatim: the suite's ComfyUI editor has a richer
  vector/radar, inline-tag, sidebar, and filesystem preset implementation.

### Release follow-up checklist

- [x] Apply the upstream `clamp_score()` numeric coercion.
- [x] Pass `use_accel` through the unified IndexTTS-2 factory; validate fallback behavior.
- [ ] Run a Qwen text-emotion generation using numeric-string classifier output.
- [ ] Verify acceleration flags on a compatible CUDA installation and on a setup without
      optional acceleration dependencies.
- [ ] Recheck bundled model-resource validation against the current upstream `check` logic.

---

## Next Update Check: 2025-11-20

**Monitoring:** Watch for commits to `indextts/infer_v2.py` in https://github.com/index-tts/index-tts

**Update command:**
```bash
cd /tmp/index_tts_upstream
git pull origin main
git log --oneline 1d5d079..HEAD -- indextts/infer_v2.py
```

**Priority changes to watch for:**
- Additional performance optimizations
- Bug fixes in new acceleration code
- Breaking API changes
- Model loading improvements
