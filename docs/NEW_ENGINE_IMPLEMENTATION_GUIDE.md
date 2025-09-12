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
- **Error Handling**: Graceful fallbacks, informative error messages

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

#### ❌ Never Duplicate These

- Character parsing logic
- Pause tag parsing logic
- Language mapping logic
- Cache key generation
- Model management logic
- Audio format conversion utilities

---