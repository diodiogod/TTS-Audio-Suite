# New TTS Engine Implementation Guide

*Comprehensive guide for LLMs to implement new TTS engines in TTS Audio Suite*

---

Don't ever make TODOs or PLACEHOLDERS during implementation.

## Pre-Implementation Analysis

### 1. Research the Original Implementation

**📁 Reference Storage:**

- Clone the original implementation to: `ComfyUI_TTS_Audio_Suite/IgnoredForGitHubDocs/For_reference/[ENGINE_NAME]/`
- Study the original codebase thoroughly
- Document all features, parameters, and capabilities

**🔍 Key Areas to Analyze:**

- **Audio Format**: Sample rate, bit depth, channels, tensor format
- **Model Architecture**: Input/output requirements, tokenization, generation process
- **Parameters**: All generation parameters, their ranges, default values
- **Dependencies**: Required packages, versions, potential conflicts
- **Unique Features**: Special capabilities not found in other engines
- **Language Support**: Monolingual vs multilingual, language codes/tags
- **Voice Control**: How voices are defined, selected, and applied

### 2. Dependency Analysis

*⚠️ Problematic Dependencies Handling:**

- If dependencies conflict with existing packages → Add to `scripts/install.py`
- If dependencies require specific versions → Document in engine requirements

**Example problematic patterns:**

- Downgrades numpy/librosa/transformers
- Conflicts with other TTS engines

---

## Project Architecture Understanding

### Core Architecture Pattern

```
⚙️ Engine Node (UI Layer real node on ComfyUI)
    ↓
🔌 Engine Adapter (Optional? - Parameter Translation)
    ↓
🏭 Engine Processor (Engine-Specific Logic)
```

### File Structure Template

```
engines/[ENGINE_NAME]/
├── __init__.py                    # Engine initialization
├── [engine_name].py              # Core engine implementation
├── [engine_name]_downloader.py   # Model auto-download (optional)
├── stateless_wrapper.py          # Thread-safe wrapper (if needed)
└── models/                       # Model-specific code (if needed)

engines/adapters/
└── [engine_name]_adapter.py      # Unified interface adapter

nodes/engines/
└── [engine_name]_engine_node.py  # UI configuration node

nodes/[engine_name]/               # Engine-specific processors
├── [engine_name]_processor.py    # Main TTS processor
├── [engine_name]_srt_processor.py # SRT processor
└── [engine_name]_vc_processor.py  # Voice conversion (if applicable)

nodes/[engine_name]_special/       # Special features (if any)
└── [engine_name]_special_node.py  # Special functionality nodes
```

---

## Implementation Steps

### Phase 1: Basic UI Engine Implementation

#### Step 1: Create Core Engine Implementation

**File:** `engines/[ENGINE_NAME]/[engine_name].py`

**Key Implementation Notes:**

- **Audio Format**: Always return `torch.Tensor` in shape `[1, samples]` or `[batch, samples]`
- **Sample Rate**: Must match engine's native sample rate, handle conversion in adapter if needed
- **Device Management**: Support "auto", "cuda", "cpu" device selection
- **Clear VRAM Integration**: CRITICAL - Implement `.to()` method and device checking for ComfyUI model management (see Step 1b below)
- **Error Handling**: Graceful fallbacks, informative error messages

#### Step 1b: Implement Clear VRAM Integration (CRITICAL)

**⚠️ REQUIRED FOR ALL NEW ENGINES**

ComfyUI's "Clear VRAM" button offloads models to CPU. Your engine MUST support this or it will cause device mismatch errors.

**Implementation Requirements:**

**1. Add `.to()` Method to Engine Class**

This method moves ALL model components between devices (CPU ↔ CUDA).

```python
def to(self, device):
    """
    Move all model components to the specified device.

    Critical for ComfyUI model management - ensures all components move together
    when models are detached to CPU and later reloaded to CUDA.
    """
    self.device = device

    # Move the underlying model/engine if loaded
    if self._model is not None:
        # OPTION 1: Simple engines with single model
        if hasattr(self._model, 'to'):
            self._model = self._model.to(device)

        # OPTION 2: Complex engines with multiple components
        # Example: ChatterBox has model.s3gen, model.t3, model.voice_encoder
        for component_name in ['s3gen', 't3', 'voice_encoder', 'tokenizer']:
            if hasattr(self._model, component_name):
                component = getattr(self._model, component_name)
                if hasattr(component, 'to'):
                    setattr(self._model, component_name, component.to(device))

        # OPTION 3: Deeply nested engines (like Index-TTS)
        # Call .to() on engine itself for recursive moving, with manual fallback
        if hasattr(self._model, 'to'):
            self._model = self._model.to(device)
        else:
            # Manually move known nested components
            for attr_name in ['semantic_model', 'gpt_model', 'vocoder']:
                if hasattr(self._model, attr_name):
                    component = getattr(self._model, attr_name)
                    if hasattr(component, 'to'):
                        setattr(self._model, attr_name, component.to(device))

        # Update device attribute on model if it exists
        if hasattr(self._model, 'device'):
            self._model.device = torch.device(device) if isinstance(device, str) else device

    return self
```

**2. Add Device Checking Before Generation**

Check if model was offloaded to CPU and reload to CUDA before generation.

```python
def generate(self, text, reference_audio, **kwargs):
    """Main generation method"""

    # CRITICAL: Check and reload model if offloaded to CPU
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    if self._model is not None:
        # Check current device of model
        # Strategy 1: Check first parameter device (most reliable)
        if hasattr(self._model, 'parameters'):
            try:
                first_param = next(self._model.parameters())
                current_device = str(first_param.device)

                if current_device != target_device:
                    print(f"🔄 Reloading {self.__class__.__name__} from {current_device} to {target_device}")

                    # Try to find wrapper and call wrapper.model_load()
                    wrapper_found = False
                    try:
                        from utils.models.unified_model_interface import unified_model_interface

                        if hasattr(unified_model_interface, 'model_manager'):
                            for cache_key, wrapper in unified_model_interface.model_manager._model_cache.items():
                                model = wrapper.model if hasattr(wrapper, 'model') else None
                                if model is self:
                                    wrapper.model_load(target_device)
                                    print(f"✅ Reloaded via wrapper - ComfyUI tracking in sync")
                                    wrapper_found = True
                                    break
                                # Check through SimpleModelWrapper if present
                                elif hasattr(model, 'model') and model.model is self:
                                    wrapper.model_load(target_device)
                                    print(f"✅ Reloaded via wrapper (unwrapped)")
                                    wrapper_found = True
                                    break

                        if not wrapper_found:
                            # Fallback: direct .to() and check if already registered
                            self.to(target_device)

                            # Check ComfyUI's current_loaded_models for existing wrapper
                            try:
                                import comfy.model_management as model_management

                                already_registered = False
                                if hasattr(model_management, 'current_loaded_models'):
                                    for wrapper in model_management.current_loaded_models:
                                        if hasattr(wrapper, 'model_info') and wrapper.model_info.engine == "[engine_name]":
                                            wrapper.model_load(target_device)
                                            print(f"✅ Reloaded via existing ComfyUI wrapper")
                                            already_registered = True
                                            break

                                if not already_registered:
                                    # Re-register with ComfyUI
                                    from utils.models.comfyui_model_wrapper.base_wrapper import ComfyUIModelWrapper, ModelInfo, SimpleModelWrapper

                                    model_size = ComfyUIModelWrapper.calculate_model_memory(self)
                                    wrapped_model = SimpleModelWrapper(self)

                                    model_info = ModelInfo(
                                        model=wrapped_model,
                                        model_type="tts",
                                        engine="[engine_name]",
                                        device=target_device,
                                        memory_size=model_size,
                                        load_device=target_device
                                    )
                                    new_wrapper = ComfyUIModelWrapper(wrapped_model, model_info)
                                    model_management.current_loaded_models.append(new_wrapper)
                                    print(f"✅ Re-registered with ComfyUI model management")
                            except Exception as e:
                                print(f"⚠️ Re-registration failed: {e}")

                    except Exception as e:
                        print(f"⚠️ Wrapper reload failed ({e}), using direct .to()")
                        self.to(target_device)

            except StopIteration:
                pass

        # Strategy 2: For models with nested components
        elif hasattr(self._model, 'semantic_model') and hasattr(self._model.semantic_model, 'parameters'):
            try:
                first_param = next(self._model.semantic_model.parameters())
                current_device = str(first_param.device)
                # ... same reload logic as above
            except StopIteration:
                pass

    # Continue with normal generation
    # ... your generation code here
```

**3. Important Device Checking Notes:**

- **Always check against INTENDED device** (`"cuda"` if available), NOT `self.device` which gets updated to "cpu" after offload
- **Search for existing wrapper first** before creating new ones to avoid duplicates
- **Use wrapper.model_load()** when possible to keep ComfyUI tracking in sync
- **Fallback to direct .to()** if wrapper not found, then re-register
- **Check both unified interface cache AND ComfyUI's current_loaded_models** for wrappers

**4. Key Implementation Notes:**

- **Always check against INTENDED device** (`"cuda"` if available), NOT `self.device`
- **Search for existing wrapper first** to avoid duplicates
- **Use wrapper.model_load()** when possible to keep ComfyUI tracking in sync
- **Check both unified interface cache AND current_loaded_models**

#### Step 2: Create Model Downloader

**File:** `engines/[ENGINE_NAME]/[engine_name]_downloader.py`

**Follow Unified Download Pattern:**

```python
from utils.downloads.unified_downloader import UnifiedDownloader
```

#### Step 3: Create Engine Adapter

**File:** `engines/adapters/[engine_name]_adapter.py`

#### Step 4: Create Engine Configuration Node

**File:** `nodes/engines/[engine_name]_engine_node.py`

### Phase 2: Unified Systems Integration

#### Step 5: Integrate with Unified Model Loading

**Use ComfyUI Model Management:**

```python
from utils.models.unified_model_interface import UnifiedModelInterface
from utils.models.comfyui_model_wrapper import ComfyUIModelWrapper
```

#### Step 6: Implement Caching System

**Cache Integration:**

```python
from utils.audio.cache import UnifiedCacheManager
from utils.audio.audio_hash import create_content_hash
```

#### Step 7: Character Switching Integration

**Use Unified Character System:**

```python
from utils.text.character_parser import CharacterParser
from utils.voice.discovery import get_character_mapping
```

#### Step 8: Language Switching Integration

**Use Unified Language System:**

```python
from utils.models.language_mapper import LanguageMapper

class [EngineClass]Processor:
    def __init__(self):
        self.language_mapper = LanguageMapper()

    def generate_with_language(self, text, language, **params):
        # Map language code
        engine_language = self.language_mapper.map_language(
            language_code=language,
            engine_type="[engine_name]"
        )

        # Apply language-specific model loading if needed
        if self.supports_multiple_languages:
            self.load_language_model(engine_language)

        # Generate with language parameter
        return self.engine.generate(text, language=engine_language, **params)
```

#### Step 9: Pause Tag Integration

**Use Unified Pause System, search for the unified code**

#### Step 9.5: Implement Interrupt Handling (CRITICAL)

**⚠️ REQUIRED FOR ALL NEW ENGINES - Users Must Be Able to Cancel Operations**

ComfyUI provides an interrupt mechanism that allows users to stop long-running operations (like SRT generation with many segments). Your processors MUST check for interruption signals.

**Import and Check Pattern:**

```python
import comfy.model_management as model_management

# In your main processing loops:
def generate_srt_speech(self, srt_content: str, ...):
    # ... setup code ...

    for i, segment in enumerate(srt_segments):
        # Check for interruption before processing each segment
        if model_management.interrupt_processing:
            raise InterruptedError(f"Engine SRT segment {i+1}/{len(srt_segments)} interrupted by user")

        # Your generation code...
```

**Where to Add Interrupt Checks:**

1. **SRT Processors** - Add checks in main subtitle/segment loops:
   - Before processing each subtitle/segment
   - Before processing character segments within a subtitle
   - Before timing assembly (after all audio generation)

2. **Text Processors** - Add checks in main generation loops:
   - Before character setup/loading
   - Before processing each character segment
   - Before processing chunks (if chunking is enabled)

3. **Special Feature Processors** - Add checks in long-running operations:
   - Before each major processing step
   - At natural breakpoints (before/after audio generation)

**Example SRT Processor Structure:**

```python
class EngineSRTProcessor:
    def generate_srt_speech(self, srt_content, ...):
        # Pre-generation check
        if model_management.interrupt_processing:
            raise InterruptedError("Setup interrupted")

        # Main segment loop
        for i, segment in enumerate(srt_segments):
            if model_management.interrupt_processing:
                raise InterruptedError(f"Segment {i+1}/{len(srt_segments)} interrupted")

            # Character switching support
            for char_segment in char_segments:
                if model_management.interrupt_processing:
                    raise InterruptedError(f"Character {char} interrupted")
                # ... generate audio ...

        # Pre-assembly check
        if model_management.interrupt_processing:
            raise InterruptedError("Assembly interrupted")

        # Timing assembly...
```

**Real Examples:**

- See `nodes/higgs_audio/higgs_audio_srt_processor.py` - SRT interrupts
- See `nodes/vibevoice/vibevoice_processor.py` - TTS processor interrupts
- See `nodes/chatterbox_official_23lang/chatterbox_official_23lang_srt_processor.py` - Complex interrupts

#### Step 10: Create Main TTS Processor

**File:** `nodes/[engine_name]/[engine_name]_processor.py`

#### Step 11: Test TTS Text Implementation

BUT rememeber to register all nodes so they load and work on comfyui.

Also to test, requirements and dependencies need to be added.

**Ask suer to Test Checklist:**

- [ ] Basic text generation works
- [ ] Character switching works with `[CharacterName] text`
- [ ] Language switching works (if applicable)
- [ ] Pause tags work with `[pause:1.5s]`
- [ ] Caching works (same input = cached output)
- [ ] Model auto-download works
- [ ] VRAM management works (model unloads)
- [ ] Different parameter combinations work
- [ ] **Interrupt handling works** - User can stop SRT generation and it stops within ~1 segment

### Phase 4: SRT Implementation

#### Step 12: Analyze SRT Strategies

**Study Existing SRT Implementations:**

- **ChatterBox**: Sequential processing with language grouping
- **F5-TTS**: Language grouping with chunking
- **Higgs Audio**: Character-based processing
- VibeVoice?

**Choose Strategy:**

1. **Sequential**: Process each subtitle line individually
2. **Language Grouping**: Group by language, then process
3. **Character Grouping**: Group by character, then batch process
4. **Hybrid**: Combine multiple strategies

#### Step 13: Create SRT Processor

**File:** `nodes/[engine_name]/[engine_name]_srt_processor.py`

### Phase 5: Special Features Implementation

#### Step 14: Identify Special Features

**Common Special Features:**

- **Speech Editing** (F5-TTS): Edit specific words in audio
- **Voice Conversion** (ChatterBox): Convert voice characteristics
- **Multi-Speaker** (Some engines): Multiple speakers in one generation
- **Style Control**: Emotion, speaking rate, emphasis

#### Step 15: Implement Special Features

Create Dedicated Nodes

---

## Unified Systems Integration

### Character Voice System

**Files to Study:**

- `utils/voice/discovery.py` - Voice file discovery
- `utils/text/character_parser.py` - Character tag parsing
- `nodes/shared/character_voices_node.py` - Character voice management

### Language System

**Files to Study:**

- `utils/models/language_mapper.py` - Language code mapping
- Engine-specific language model files

### Pause Tag System

**Files to Study:**

- `utils/text/pause_processor.py` - Pause tag parsing and generation

**Integration Pattern:**

### Model Management

**Files to Study:**

- `utils.models/unified_model_interface.py` - Unified model loading
- `utils/models/comfyui_model_wrapper.py` - ComfyUI integration

---

## Documentation Updates

### README.md Updates

#### 1. Features Section

Add engine to the features list with its unique capabilities.

#### 2. What's New Section

Add changelog entry for the new engine.

#### 3. Model Download Section

Add download instructions and model requirements.

#### 4. Supported Engines Table

Update the engines comparison table.

### Example README Addition:

```markdown
## What's New in v4.X.X

### 🚀 New [Engine Name] TTS Engine
- High-quality text-to-speech with [unique feature]
- Support for [languages/voices/special capabilities]
- Integrated with unified interface (TTS Text, SRT, Voice Changer)
- Auto-download models with one click

## Features

### 🎤 [Engine Name] TTS Engine
- **[Unique Feature 1]**: Description of what makes this engine special
- **[Unique Feature 2]**: Another special capability
- **Multi-language support**: List of supported languages (if applicable)
- **Voice cloning**: Description of voice capabilities (if applicable)
```

---

## Implementation Phase Strategy

### Phase 1: Foundation (Implement First)

1. Core engine implementation
2. Basic text generation (With character switching, 
   - Language switching (if applicable)
   - Pause tag support
   - Caching system
   - VRAM management)
3. Model loading and downloading
4. Engine configuration node
5. Integration with TTS Text node

**Stop here and test with user before proceeding**

### Phase 2: SRT Support

1. SRT processor implementation
2. Timing and assembly integration
3. Character switching in SRT
4. Performance optimization

### Phase 3: Special Features

1. Engine-specific unique features
2. Special nodes for unique capabilities
3. Advanced parameter controls

### Phase 4: Documentation and Polish

1. README updates
2. Example workflows
3. Performance testing
4. Error handling improvements

---

## Critical Integration Points

### Must Use These Unified Systems

#### ✅ Required Integrations

- **UnifiedModelInterface** - Model loading
- **ComfyUIModelWrapper** - VRAM management
- **UnifiedCacheManager** - Audio caching
- **CharacterParser** - Character tag parsing
- **PauseTagProcessor** - Pause tag handling
- **LanguageMapper** - Language code mapping
- **UnifiedDownloader** - Model downloads
- **model_management.interrupt_processing** - Interrupt signal checking (CRITICAL for cancellation)

#### ❌ Never Duplicate These

- Character parsing logic
- Pause tag parsing logic
- Language mapping logic
- Cache key generation
- Model management logic
- Audio format conversion utilities

---