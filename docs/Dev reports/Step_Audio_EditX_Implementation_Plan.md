# Step Audio EditX Engine Implementation Plan

## Executive Summary

**Implementation Approach**: Bundle Step Audio EditX code (like IndexTTS pattern) with TTS Suite integration

**Key Decisions:**
- ✅ **Bundle code** in `step_audio_editx_impl/` (NOT pip install)
- ✅ **Skip Whisper** dependency (only needed for external audio transcription)
- ✅ **Unpinned transformers** (patch if needed via TTS Suite system)
- ✅ **Progress bars** via ComfyUI (UX improvement)
- ✅ **Manual transcripts** for Audio Editor (Whisper integration later)
- ✅ **Phased implementation**: Clone → Edit → SRT

**What Step Audio EditX Is:**
- 3B parameter QWen-based LLM for semantic generation
- Custom dual-codebook tokenizer (VQ02 + VQ06)
- CosyVoice vocoder for final audio generation (NOT full CosyVoice)
- Unique emotion/style/speed editing capabilities
- Worth implementing alongside future CosyVoice engine

---

## Analysis of Both Implementations

### Original Step Audio EditX (stepfun-ai)
- **Core Features**:
  - Zero-shot voice cloning (3-30s reference audio)
  - Audio editing: emotion, style, speed, paralinguistics
  - Supports Mandarin, English, Sichuanese, Cantonese, Japanese, Korean
  - Iterative editing (n_edit_iter parameter)
  - Denoise and VAD
  - Polyphone pronunciation control
  - 3B parameter model with dual-codebook tokenizer

- **Architecture**:
  - `tokenizer.py` (StepAudioTokenizer) - Dual-codebook tokenizer
  - `tts.py` (StepAudioTTS) - Main TTS engine with clone() and edit() methods
  - `model_loader.py` - Model loading from HuggingFace/ModelScope/local
  - `config/edit_config.py` - Edit type configurations
  - Uses bundled `funasr_detach/` for FunASR models (paraformer tokenizer)
  - CosyVoice vocoder for audio generation (24kHz output)

### Saganaki22's ComfyUI Implementation
- **Native Features**: All of the above
- **Custom Additions (NON-NATIVE)**:
  1. **Longform chunking system** (`core/longform_chunker.py`):
     - Splits long text at sentence boundaries
     - Estimates tokens (1 token ≈ 4 chars)
     - Creates overlapping chunks
     - This is a ComfyUI-specific optimization NOT in original

  2. **Audio stitching** (`core/audio_stitcher.py`):
     - Concatenates audio chunks from longform generation
     - Crossfade support
     - NOT part of original implementation

  3. **Progress bar integration**:
     - ComfyUI progress tracking during clone generation
     - NOT in original (original has no progress callbacks)

  4. **Model wrapper architecture**:
     - `StepAudioModelWrapper` with to_cpu()/to_cuda() methods
     - CUDA graph cleanup logic
     - More sophisticated VRAM management than original

**CRITICAL**: Speed control (faster/slower) IS native - it's in the original edit_config.py, so this is NOT a custom addition.

---

## Recommended Implementation Approach

### Use Original as Base (NOT Saganaki22)

Saganaki22's implementation is excellent for ComfyUI but adds non-native features. Since you want faithful implementation, we start fresh.

### Phase 1: Core Engine Implementation

#### Files to Create:

1. **`engines/step_audio_editx/step_audio_editx.py`**
   - Main engine wrapper
   - Wraps original `StepAudioTTS` class
   - Implements `.to()` method for ComfyUI VRAM management
   - Device checking before generation
   - Sample rate: 24000 Hz (CosyVoice native)

2. **`engines/step_audio_editx/step_audio_downloader.py`**
   - Uses `utils/downloads/unified_downloader.py`
   - Downloads Step-Audio-EditX and Step-Audio-Tokenizer
   - Organized structure: `models/TTS/step_audio_editx/`

3. **`engines/step_audio_editx/__init__.py`**
   - Engine initialization
   - Imports

4. **`engines/step_audio_editx/original_impl/`**
   - Copy entire original Step Audio EditX implementation here
   - Includes: `tts.py`, `tokenizer.py`, `model_loader.py`, `config/`, `funasr_detach/`, etc.
   - Keep it isolated so it doesn't conflict with existing dependencies

#### Key Implementation Notes:

**Audio Format**:
- Original outputs: `torch.Tensor` shape `[samples]` or `[1, samples]`
- Sample rate: 24000 Hz
- Return format: TTS Audio Suite expects `[1, samples]`

**Generation Methods**:
```python
def clone(self, prompt_wav_path, prompt_text, target_text, temperature=0.7, do_sample=True, max_new_tokens=4096):
    # Call original implementation
    pass

def edit(self, input_audio_path, audio_text, edit_type, edit_info=None, text=None):
    # Edit mode - temperature/do_sample/max_new_tokens are HARDCODED in original
    # Parameters: edit_type (emotion/style/speed/paralinguistic/denoising)
    # Returns audio_tensor, sample_rate
    pass
```

**VRAM Management** (CRITICAL):
- Implement `.to(device)` method
- Move `self.llm` (3B parameter model) between devices
- Move `self.cosy_model` (vocoder) between devices
- Check device before generation and reload if needed
- Follow patterns from IndexTTS implementation

---

### Phase 2: Node Implementation

#### Files to Create:

1. **`nodes/engines/step_audio_editx_engine_node.py`**
   - Configuration node for Step Audio EditX
   - Parameters:
     - `model_path` - path to Step-Audio-EditX
     - `device` - cuda/cpu
     - `torch_dtype` - bfloat16/float16/float32
     - `quantization` - none/int4/int8
     - `attention_mechanism` - sdpa/eager/flash_attn
   - Returns: ENGINE config

2. **`nodes/step_audio_editx/step_audio_editx_processor.py`**
   - Main TTS processor (internal, used by Unified TTS Text node)
   - Handles:
     - Character switching with `[CharacterName]` tags
     - Pause tags with `[pause:1.5s]` syntax
     - Caching via `utils/audio/cache.py`
     - Voice discovery via `utils/voice/discovery.py`
     - **Longform chunking via `utils/audio/chunk_combiner.py`** (TTS Suite unified)
   - Sequential generation with interrupt handling

3. **`nodes/step_audio_editx_special/step_audio_editx_audio_editor_node.py`** (Special Feature)
   - Dedicated "🎨 Step Audio EditX - Audio Editor" node
   - This is NOT voice conversion - it's specialized audio manipulation
   - **CRITICAL**: Requires `audio_text` parameter (transcript of input audio)
     - User must provide transcript manually (no Whisper transcription)
     - Future: When Whisper engine is added, can auto-feed transcripts
   - Parameters:
     - `input_audio` - audio to edit (0.5-30s limit)
     - `audio_text` - REQUIRED transcript of input audio
     - `edit_type` - emotion/style/speed/paralinguistic/denoising
     - `emotion` - happy/sad/angry/excited/etc (14 options)
     - `style` - whisper/serious/child/older/etc (32 options)
     - `speed` - faster/slower/more faster/more slower
     - `paralinguistic` - [Laughter]/[Breathing]/[Sigh]/etc (10 options)
     - `denoising` - denoise/vad
     - `n_edit_iterations` - 1-5 (strength through iteration)
     - `paralinguistic_text` - where to insert effect
   - Returns edited audio

---

### Phase 3: Unified Systems Integration

#### Integration Points:

1. **Unified Model Loading**:
   ```python
   from utils.models.unified_model_interface import UnifiedModelInterface

   # Register Step Audio EditX
   unified_model_interface.register_model(
       engine_type="step_audio_editx",
       model=step_audio_engine,
       model_info=ModelInfo(...)
   )
   ```

2. **Character System**:
   - Use `utils/text/character_parser.py` for `[CharacterName]` tags
   - Use `utils/voice/discovery.py` for voice file loading
   - Support both .wav and .txt voice references

3. **Pause Tags**:
   - Use `utils/text/pause_processor.py` for `[pause:xx]` syntax
   - Integrate with audio generation pipeline

4. **Caching**:
   ```python
   from utils.audio/cache import UnifiedCacheManager
   from utils.audio.audio_hash import create_content_hash

   cache_key = create_content_hash(
       text=text,
       voice_path=voice_path,
       engine="step_audio_editx",
       parameters={...}
   )
   ```

5. **Interrupt Handling** (CRITICAL):
   ```python
   import comfy.model_management as model_management

   # Before each generation
   if model_management.interrupt_processing:
       raise InterruptedError("Step Audio EditX generation interrupted")
   ```

---

### Phase 4: Dependencies

#### Add to `requirements.txt`:
```txt
# Step Audio EditX dependencies
librosa>=0.10.2
soundfile>=0.12.1  # Critical for ComfyUI audio preview
onnxruntime>=1.17.0
transformers>=4.30.0  # Unpinned - use TTS Suite transformer patches if needed
sentencepiece>=0.1.99
omegaconf>=2.3.0
hyperpyyaml>=1.2.2
# NOTE: openai-whisper NOT required for TTS (only for external audio transcription)
# Future: When Whisper transcription engine is added, it can feed Audio Editor node
```

**Dependency Strategy:**
- ✅ **Bundle code** (like IndexTTS) - not pip installable
- ✅ **Skip Whisper** - only needed for external audio → edit workflow
- ✅ **Unpinned transformers** - patch via TTS Suite system if 4.54+ breaks
- ✅ **Windows-friendly** - no complex model downloads for basic TTS

---

### Phase 5: SRT Implementation (Later)

**Strategy**: Sequential processing with language grouping
- Group SRT segments by language (Mandarin/English/Japanese/Korean/etc)
- Process each language group sequentially
- No parallel processing (model doesn't support it well)
- Support character switching within segments
- Use timing modes from `utils/timing/`

**File**: `nodes/step_audio_editx/step_audio_editx_srt_processor.py`

---

## Key Differences from Saganaki22 Implementation

| Feature | Original | Saganaki22 | Our Implementation |
|---------|----------|------------|-------------------|
| Longform chunking | ❌ No | ✅ Yes (custom) | ✅ Yes (TTS Suite unified) |
| Progress bars | ❌ No | ✅ Yes (clone only) | ✅ Yes (ComfyUI standard) |
| Whisper transcription | ⚠️ Optional (ASR) | ⚠️ Optional (ASR) | ❌ Skip (future: Whisper engine) |
| Audio stitching | ❌ No | ✅ Yes (custom) | ❌ No (not native) |
| Iterative editing | ✅ Yes | ✅ Yes | ✅ Yes (native) |
| Speed control | ✅ Yes | ✅ Yes | ✅ Yes (native) |
| Character switching | ❌ No | ❌ No | ✅ Yes (TTS Suite feature) |
| Pause tags | ❌ No | ❌ No | ✅ Yes (TTS Suite feature) |
| Caching | ❌ No | ❌ No | ✅ Yes (TTS Suite feature) |
| Interrupt handling | ❌ No | ❌ No | ✅ Yes (TTS Suite feature) |

---

## Critical Implementation Requirements

### 1. VRAM Management (from guide lines 96-264)
- Implement `.to(device)` method
- Move LLM (self.llm) and vocoder (self.cosy_model) between devices
- Check device before generation
- Reload from CPU to CUDA if offloaded
- Search for existing wrapper before creating new ones

### 2. Clear VRAM Integration
- Register with ComfyUI model management
- Support "Clear VRAM" button functionality
- Prevent device mismatch errors

### 3. Interrupt Handling (CRITICAL)
- Check `model_management.interrupt_processing` before each generation
- In SRT mode: check before each segment
- Allow users to cancel long operations

### 4. Audio Limitations
- **Clone mode**: No strict length limit (depends on max_new_tokens)
- **Edit mode**: 0.5-30 seconds HARD LIMIT
- Validate audio duration before processing

### 5. Language Support
- Mandarin (primary)
- English
- Sichuanese dialect
- Cantonese dialect
- Japanese
- Korean
- Add `[Sichuanese]`, `[Cantonese]`, `[Japanese]`, `[Korean]` tag support

### 6. Edit Types and Options
All options are from original `config/edit_config.py`:

**Emotions** (14):
- happy, sad, angry, excited, calm, fearful, surprised, disgusted
- confusion, empathy, embarrass, depressed, coldness, admiration

**Styles** (32):
- whisper, serious, child, older, girl, pure, sister, sweet
- exaggerated, ethereal, generous, recite, act_coy, warm, shy
- comfort, authority, chat, radio, soulful, gentle, story, vivid
- program, news, advertising, roar, murmur, shout, deeply, loudly
- arrogant, friendly

**Speed** (4):
- faster, slower, more faster, more slower

**Paralinguistics** (10):
- [Breathing], [Laughter], [Surprise-oh], [Confirmation-en]
- [Uhm], [Surprise-ah], [Surprise-wa], [Sigh]
- [Question-ei], [Dissatisfaction-hnn]

**Denoising** (2):
- denoise, vad

---

## File Structure

```
engines/step_audio_editx/
├── __init__.py
├── step_audio_editx.py (main wrapper - follows IndexTTS pattern)
├── step_audio_editx_downloader.py (unified downloader integration)
└── step_audio_editx_impl/ (bundled original codebase - like IndexTTS)
    ├── tts.py (StepAudioTTS - main engine)
    ├── tokenizer.py (StepAudioTokenizer - dual-codebook)
    ├── model_loader.py (ModelSource, model_loader)
    ├── config/
    │   ├── edit_config.py (edit types/options)
    │   └── prompts.py
    ├── funasr_detach/ (bundled FunASR - custom fork)
    └── stepvocoder/cosyvoice2/ (CosyVoice vocoder)

nodes/engines/
└── step_audio_editx_engine_node.py (config node)

nodes/step_audio_editx/
├── step_audio_editx_processor.py (TTS processor - internal)
└── step_audio_editx_srt_processor.py (SRT processor - internal, later)

nodes/step_audio_editx_special/
└── step_audio_editx_audio_editor_node.py (🎨 Audio Editor - specialized manipulation)

```

---

## Testing Checklist

### Phase 1 Testing (Basic TTS):
- [ ] Model downloads correctly
- [ ] Basic clone generation works
- [ ] Audio format is correct (24kHz, mono)
- [ ] Character switching works
- [ ] Pause tags work
- [ ] Caching works
- [ ] VRAM management works (Clear VRAM button)
- [ ] Device reload works after offload
- [ ] Interrupt handling works

### Phase 2 Testing (Audio Editing):
- [ ] Emotion editing works (all 14 emotions)
- [ ] Style editing works (all 32 styles)
- [ ] Speed editing works (4 speed levels)
- [ ] Paralinguistic editing works (all 10 effects)
- [ ] Denoising works (denoise + vad)
- [ ] Iterative editing works (n_edit_iter 1-5)
- [ ] Audio length validation works (0.5-30s)
- [ ] Edit node integrates with voice changer workflows

### Phase 3 Testing (SRT Mode):
- [ ] SRT parsing works
- [ ] Language grouping works
- [ ] Character switching in SRT works
- [ ] Timing assembly works
- [ ] Multiple languages in same SRT works
- [ ] Interrupt handling in SRT works

---

## Questions for User

Before we proceed with implementation, I need clarification on:

1. **Longform chunking**: Saganaki22 added sentence-based text chunking for long content. This is NOT in the original. Do you want this feature, or should we stick to the original behavior (which may truncate long text at max_new_tokens)?

2. **Progress bars**: Saganaki22 added ComfyUI progress tracking during clone generation. This is NOT in the original. Should we add this for better UX, or skip it?

3. **Separate edit node**: Should we create a dedicated "Step Audio EditX - Edit" node for audio editing (emotion/style/speed), or integrate everything through the unified voice changer node?

4. **Model location**: Should models go in `models/TTS/step_audio_editx/` or `models/step_audio_editx/` (Saganaki22 used `models/Step-Audio-EditX/`)?

5. **Implementation priority**: What should we implement first?
   - Option A: Clone mode only (basic TTS) → test → then edit mode → then SRT
   - Option B: Clone + Edit modes together → test → then SRT
   - Option C: All features at once

---

## Implementation Strategy (User Approved)

**Phase 1**: Clone mode with TTS Suite integration
- Full TTS Suite integration (character switching, pause tags, caching)
- Use `utils/audio/chunk_combiner.py` for longform text
- **Progress bars** via `comfy.utils.ProgressBar` during generation
- VRAM management with `.to()` method (follow IndexTTS pattern)
- Interrupt handling via `model_management.interrupt_processing`
- Bundle code in `step_audio_editx_impl/` (like IndexTTS)
- Test thoroughly before proceeding

**Phase 2**: Audio Editor (specialized node)
- Dedicated "🎨 Step Audio EditX - Audio Editor" node
- NOT integrated with voice changer (different purpose)
- **Requires manual transcript input** (audio_text parameter)
  - Skip Whisper transcription for now
  - Future: Auto-feed from Whisper transcription engine
- All editing capabilities (emotion/style/speed/paralinguistic/denoising)
- Iterative refinement support (n_edit_iter 1-5)
- Audio length validation (0.5-30s)
- Test all editing modes

**Phase 3**: SRT support
- Sequential processing with language grouping
- Character switching within segments
- Timing assembly
- Interrupt handling for cancellation

This approach:
- Allows incremental testing
- Validates VRAM management early
- Ensures unified systems work correctly
- Follows TTS Suite best practices
- Lets you test and approve each phase

