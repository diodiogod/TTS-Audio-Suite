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

## Next Update Check: 2025-10-18

**Monitoring:** Watch for commits to `indextts/infer_v2.py` in https://github.com/index-tts/index-tts

**Update command:**
```bash
cd IgnoredForGitHubDocs/For_reference/index_tts
git pull origin main
git log --oneline LAST_COMMIT..HEAD -- indextts/infer_v2.py
```

**Priority changes to watch for:**
- Performance optimizations
- Memory management improvements
- Bug fixes in emotion processing
- New model loading features