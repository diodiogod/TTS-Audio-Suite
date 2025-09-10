# New TTS Engine Implementation Guide

*Comprehensive guide for LLMs to implement new TTS engines in TTS Audio Suite*

## Table of Contents

1. [Pre-Implementation Analysis](#pre-implementation-analysis)
2. [Project Architecture Understanding](#project-architecture-understanding)
3. [Implementation Steps](#implementation-steps)
4. [Unified Systems Integration](#unified-systems-integration)
5. [Testing and Validation](#testing-and-validation)
6. [Documentation Updates](#documentation-updates)
7. [Common Patterns and Templates](#common-patterns-and-templates)

---

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

**📋 Requirements Assessment:**
```bash
# Check for problematic dependencies
pip check
```

**⚠️ Problematic Dependencies Handling:**
- If dependencies conflict with existing packages → Add to `scripts/install.py`
- If dependencies require specific versions → Document in engine requirements
- If dependencies are large/optional → Make them optional imports with fallbacks

**Example problematic patterns:**
- Downgrades torch/transformers
- Requires specific CUDA versions
- Conflicts with other TTS engines

---

## Project Architecture Understanding

### Core Architecture Pattern

```
🎛️ Engine Configuration Node (UI Layer)
    ↓
🔄 Unified Interface Nodes (TTS Text, TTS SRT, Voice Changer)
    ↓
🔌 Engine Adapter (Optional - Parameter Translation)
    ↓
🏭 Engine Processor (Engine-Specific Logic)
    ↓
⚙️ Engine Implementation (Core TTS Engine)
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

### Phase 1: Basic Engine Implementation

#### Step 1: Create Core Engine Implementation

**File:** `engines/[ENGINE_NAME]/[engine_name].py`

**Required Methods:**
```python
class [EngineClass]:
    def __init__(self, device="auto", **kwargs):
        """Initialize engine with device and parameters"""
        
    def load_model(self, model_path_or_id, **kwargs):
        """Load model using unified model loading system"""
        
    def generate(self, text, **generation_params):
        """Core generation method - returns torch.Tensor"""
        
    def prepare_conditionals(self, audio_prompt_path, **kwargs):
        """Prepare voice conditioning (if applicable)"""
        
    @property
    def sample_rate(self):
        """Return the engine's sample rate"""
        
    def cleanup(self):
        """Clean up resources for VRAM management"""
```

**Key Implementation Notes:**
- **Audio Format**: Always return `torch.Tensor` in shape `[1, samples]` or `[batch, samples]`
- **Sample Rate**: Must match engine's native sample rate, handle conversion in adapter if needed
- **Device Management**: Support "auto", "cuda", "cpu" device selection
- **Error Handling**: Graceful fallbacks, informative error messages

#### Step 2: Create Model Downloader (if needed)

**File:** `engines/[ENGINE_NAME]/[engine_name]_downloader.py`

**Follow Unified Download Pattern:**
```python
from utils.downloads.unified_downloader import UnifiedDownloader

class [EngineClass]Downloader:
    def __init__(self):
        self.downloader = UnifiedDownloader()
        
    def download_model(self, model_id, **kwargs):
        """Download model using unified system"""
        return self.downloader.download_model(
            model_id=model_id,
            engine_name="[engine_name]",
            **kwargs
        )
```

#### Step 3: Create Engine Adapter

**File:** `engines/adapters/[engine_name]_adapter.py`

**Template:**
```python
from .base_adapter import BaseAdapter  # If exists
from utils.audio.processing import convert_sample_rate, normalize_audio

class [EngineClass]Adapter:
    def __init__(self, engine):
        self.engine = engine
        
    def generate_segment_audio(self, text, char_audio, char_text, character, **params):
        """Unified interface for TTS generation"""
        # Parameter mapping and validation
        mapped_params = self._map_parameters(**params)
        
        # Voice conditioning setup
        if char_audio:
            self.engine.prepare_conditionals(char_audio)
            
        # Generation
        audio = self.engine.generate(text, **mapped_params)
        
        # Audio format normalization
        return self._normalize_output(audio)
        
    def _map_parameters(self, **params):
        """Map unified parameters to engine-specific parameters"""
        
    def _normalize_output(self, audio):
        """Normalize output to unified format"""
        # Ensure correct sample rate, format, etc.
```

#### Step 4: Create Engine Configuration Node

**File:** `nodes/engines/[engine_name]_engine_node.py`

**Template:**
```python
from nodes.base.base_node import BaseNode

class [EngineClass]EngineNode(BaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["auto-download-model-1", "model-2"], {"default": "auto-download-model-1"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                # Engine-specific parameters only
                "engine_param_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "engine_param_2": (["option1", "option2"], {"default": "option1"}),
            }
        }
    
    RETURN_TYPES = ("[ENGINE_NAME]_ENGINE",)
    FUNCTION = "create_engine"
    CATEGORY = "TTS Audio Suite/Engines"
    
    def create_engine(self, **params):
        """Create and return configured engine instance"""
```

### Phase 2: Unified Systems Integration

#### Step 5: Integrate with Unified Model Loading

**Use ComfyUI Model Management:**
```python
from utils.models.unified_model_interface import UnifiedModelInterface
from utils.models.comfyui_model_wrapper import ComfyUIModelWrapper

class [EngineClass]:
    def load_model(self, model_path_or_id):
        # Use unified interface
        model_info = UnifiedModelInterface.load_model(
            model_id=model_path_or_id,
            engine_type="[engine_name]",
            device=self.device
        )
        
        # Wrap for ComfyUI integration
        self.model = ComfyUIModelWrapper(
            model=model_info.model,
            model_type="[engine_name]"
        )
```

#### Step 6: Implement Caching System

**Cache Integration:**
```python
from utils.audio.cache import UnifiedCacheManager
from utils.audio.audio_hash import create_content_hash

class [EngineClass]Processor:
    def __init__(self):
        self.cache_manager = UnifiedCacheManager(engine_name="[engine_name]")
        
    def generate_with_cache(self, text, **params):
        # Create cache key including ALL parameters that affect output
        cache_key = create_content_hash({
            "text": text,
            "engine": "[engine_name]",
            "model": self.model_name,
            **params  # All generation parameters
        })
        
        # Check cache
        cached_audio = self.cache_manager.get_cached_audio(cache_key)
        if cached_audio is not None:
            return cached_audio
            
        # Generate and cache
        audio = self.engine.generate(text, **params)
        self.cache_manager.cache_audio(cache_key, audio)
        return audio
```

#### Step 7: Character Switching Integration

**Use Unified Character System:**
```python
from utils.text.character_parser import CharacterParser
from utils.voice.discovery import get_character_mapping

class [EngineClass]Processor:
    def process_with_characters(self, text, character_voices, **params):
        # Parse character tags
        parser = CharacterParser()
        segments = parser.parse_character_segments(text)
        
        # Get character voice mapping
        voice_mapping = get_character_mapping(character_voices)
        
        # Process each segment
        audio_segments = []
        for segment in segments:
            character = segment.character
            segment_text = segment.text
            
            # Get voice for character
            voice_path = voice_mapping.get(character)
            if voice_path:
                self.engine.prepare_conditionals(voice_path)
                
            # Generate audio
            audio = self.generate_with_cache(segment_text, **params)
            audio_segments.append(audio)
            
        # Combine segments
        return torch.cat(audio_segments, dim=-1)
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

**Use Unified Pause System:**
```python
from utils.text.pause_processor import PauseTagProcessor

class [EngineClass]Processor:
    def generate_with_pauses(self, text, **params):
        processor = PauseTagProcessor(sample_rate=self.engine.sample_rate)
        
        return processor.generate_audio_with_pauses(
            text=text,
            tts_generate_func=lambda t: self.engine.generate(t, **params),
            sample_rate=self.engine.sample_rate
        )
```

### Phase 3: TTS Text Implementation

#### Step 10: Create Main TTS Processor

**File:** `nodes/[engine_name]/[engine_name]_processor.py`

**Template:**
```python
from nodes.base.base_node import BaseNode
from utils.text.character_parser import CharacterParser
from utils.text.pause_processor import PauseTagProcessor
from utils.models.unified_model_interface import UnifiedModelInterface

class [EngineClass]TTSProcessor(BaseNode):
    def __init__(self):
        super().__init__()
        self.engine = None
        self.character_parser = CharacterParser()
        self.pause_processor = PauseTagProcessor()
        
    def generate_speech(self, text, engine_config, character_voices, **params):
        """Main speech generation method"""
        # Initialize engine if needed
        if not self.engine:
            self.engine = self._initialize_engine(engine_config)
            
        # Process with unified systems
        if "[" in text and "]" in text:
            # Character switching detected
            return self._process_with_characters(text, character_voices, **params)
        else:
            # Simple generation
            return self._generate_simple(text, character_voices, **params)
            
    def _initialize_engine(self, engine_config):
        """Initialize engine using unified interface"""
        
    def _process_with_characters(self, text, character_voices, **params):
        """Process text with character switching"""
        
    def _generate_simple(self, text, character_voices, **params):
        """Generate simple text without character switching"""
```

#### Step 11: Test TTS Text Implementation

**Testing Checklist:**
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

**Choose Strategy:**
1. **Sequential**: Process each subtitle line individually
2. **Language Grouping**: Group by language, then process
3. **Character Grouping**: Group by character, then batch process
4. **Hybrid**: Combine multiple strategies

#### Step 13: Create SRT Processor

**File:** `nodes/[engine_name]/[engine_name]_srt_processor.py`

**Template:**
```python
from utils.timing.engine import SRTTimingEngine
from utils.timing.assembly import AudioAssemblyEngine

class [EngineClass]SRTProcessor(BaseNode):
    def __init__(self):
        super().__init__()
        self.timing_engine = SRTTimingEngine()
        self.assembly_engine = AudioAssemblyEngine()
        
    def process_srt(self, srt_content, engine_config, **params):
        """Main SRT processing method"""
        # Parse SRT content
        segments = self.timing_engine.parse_srt(srt_content)
        
        # Choose processing strategy
        if self.supports_batch_processing:
            return self._process_batch(segments, **params)
        else:
            return self._process_sequential(segments, **params)
            
    def _process_batch(self, segments, **params):
        """Batch processing strategy"""
        
    def _process_sequential(self, segments, **params):
        """Sequential processing strategy"""
```

### Phase 5: Special Features Implementation

#### Step 14: Identify Special Features

**Common Special Features:**
- **Speech Editing** (F5-TTS): Edit specific words in audio
- **Voice Conversion** (ChatterBox): Convert voice characteristics
- **Multi-Speaker** (Some engines): Multiple speakers in one generation
- **Style Control**: Emotion, speaking rate, emphasis

#### Step 15: Implement Special Features

**Create Dedicated Nodes:**
```python
# Example: Speech editing feature
class [EngineClass]EditNode(BaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_audio": ("AUDIO",),
                "edit_text": ("STRING", {"multiline": True}),
                "replacement_text": ("STRING", {"multiline": True}),
                "engine_config": ("[ENGINE_NAME]_ENGINE",),
            }
        }
```

---

## Unified Systems Integration

### Character Voice System

**Files to Study:**
- `utils/voice/discovery.py` - Voice file discovery
- `utils/text/character_parser.py` - Character tag parsing
- `nodes/shared/character_voices_node.py` - Character voice management

**Integration Pattern:**
```python
from utils.voice.discovery import get_character_mapping

# In your processor
character_mapping = get_character_mapping(character_voices_input)
voice_path = character_mapping.get(character_name)
```

### Language System

**Files to Study:**
- `utils/models/language_mapper.py` - Language code mapping
- Engine-specific language model files

**Integration Pattern:**
```python
from utils.models.language_mapper import LanguageMapper

mapper = LanguageMapper()
engine_language = mapper.map_language(user_language, "[engine_name]")
```

### Pause Tag System

**Files to Study:**
- `utils/text/pause_processor.py` - Pause tag parsing and generation

**Integration Pattern:**
```python
from utils.text.pause_processor import PauseTagProcessor

processor = PauseTagProcessor(sample_rate=self.sample_rate)
audio = processor.generate_audio_with_pauses(
    text=text,
    tts_generate_func=your_tts_function,
    sample_rate=self.sample_rate
)
```

### Model Management

**Files to Study:**
- `utils.models/unified_model_interface.py` - Unified model loading
- `utils/models/comfyui_model_wrapper.py` - ComfyUI integration

**Integration Pattern:**
```python
from utils.models.unified_model_interface import UnifiedModelInterface

model_info = UnifiedModelInterface.load_model(
    model_id=model_name,
    engine_type="[engine_name]",
    device=device
)
```

---

## Testing and Validation

### Testing Checklist

#### Basic Functionality
- [ ] Model loads correctly
- [ ] Model auto-downloads when needed
- [ ] Basic text generation works
- [ ] Audio output has correct format and sample rate
- [ ] Parameters affect output correctly
- [ ] Engine integrates with unified nodes

#### Character System
- [ ] Character tags are parsed correctly: `[Alice] Hello`
- [ ] Voice switching works between characters
- [ ] Character voice mapping works
- [ ] Mixed character text works: `[Alice] Hi [Bob] Hello`

#### Language System (if applicable)
- [ ] Language detection works
- [ ] Language switching works
- [ ] Multi-language text works

#### Pause System
- [ ] Pause tags work: `[pause:1.5s]`
- [ ] Different pause durations work
- [ ] Pauses are correctly timed in output audio

#### SRT Processing
- [ ] SRT files parse correctly
- [ ] Timing alignment is accurate
- [ ] Character switching in SRT works
- [ ] Audio segments assemble correctly

#### Caching System
- [ ] Identical inputs return cached results
- [ ] Parameter changes invalidate cache
- [ ] Cache keys are unique and stable

#### VRAM Management
- [ ] Model loads on correct device
- [ ] "Clear VRAM" button works
- [ ] Model memory is released properly

### Validation Tests

**Create Test Files:**
```
tests/[engine_name]/
├── test_basic_generation.py
├── test_character_switching.py
├── test_language_switching.py
├── test_pause_tags.py
├── test_srt_processing.py
└── test_caching.py
```

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

## Common Patterns and Templates

### Engine Implementation Template

```python
# engines/[engine_name]/[engine_name].py
import torch
from utils.models.comfyui_model_wrapper import ComfyUIModelWrapper

class [EngineClass]:
    def __init__(self, device="auto", model_name="default"):
        self.device = self._resolve_device(device)
        self.model = None
        self.model_name = model_name
        self.sample_rate = 22050  # Engine's native sample rate
        
    def load_model(self, model_path_or_id):
        """Load model using unified interface"""
        from utils.models.unified_model_interface import UnifiedModelInterface
        
        model_info = UnifiedModelInterface.load_model(
            model_id=model_path_or_id,
            engine_type="[engine_name]",
            device=self.device
        )
        
        self.model = ComfyUIModelWrapper(
            model=model_info.model,
            model_type="[engine_name]"
        )
        
    def prepare_conditionals(self, audio_prompt_path, **kwargs):
        """Prepare voice conditioning"""
        # Load and process reference audio
        # Set up voice conditioning for generation
        
    def generate(self, text, **params):
        """Core generation method"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        # Preprocess text
        # Run inference
        # Postprocess audio
        # Return torch.Tensor in shape [1, samples]
        
    def cleanup(self):
        """Clean up resources"""
        if self.model:
            self.model.cleanup()
            self.model = None
```

### Adapter Template

```python
# engines/adapters/[engine_name]_adapter.py
from utils.audio.processing import convert_sample_rate, normalize_audio

class [EngineClass]Adapter:
    def __init__(self, engine):
        self.engine = engine
        
    def generate_segment_audio(self, text, char_audio, char_text, character, **params):
        """Unified interface for segment generation"""
        try:
            # Setup voice conditioning
            if char_audio:
                self.engine.prepare_conditionals(char_audio)
                
            # Map parameters
            engine_params = self._map_parameters(**params)
            
            # Generate audio
            audio = self.engine.generate(text, **engine_params)
            
            # Normalize output format
            return self._normalize_output(audio)
            
        except Exception as e:
            print(f"❌ Error in {self.__class__.__name__}: {e}")
            raise
            
    def _map_parameters(self, **params):
        """Map unified parameters to engine-specific ones"""
        return {
            # Map standard parameters like temperature, seed, etc.
            # to engine-specific parameter names
        }
        
    def _normalize_output(self, audio):
        """Ensure output matches expected format"""
        # Convert to target sample rate if needed
        # Ensure correct tensor shape and type
        return audio
```

### Processor Template

```python
# nodes/[engine_name]/[engine_name]_processor.py
from nodes.base.base_node import BaseNode
from utils.text.character_parser import CharacterParser
from utils.audio.cache import UnifiedCacheManager

class [EngineClass]Processor(BaseNode):
    def __init__(self):
        super().__init__()
        self.engine = None
        self.adapter = None
        self.character_parser = CharacterParser()
        self.cache_manager = UnifiedCacheManager(engine_name="[engine_name]")
        
    def generate_speech(self, text, engine_config, character_voices, **params):
        """Main generation method called by unified nodes"""
        try:
            # Initialize if needed
            self._ensure_initialized(engine_config)
            
            # Route to appropriate processing method
            if self.character_parser.has_character_tags(text):
                return self._process_with_characters(text, character_voices, **params)
            else:
                return self._process_simple(text, character_voices, **params)
                
        except Exception as e:
            return self.process_with_error_handling(
                lambda: self._generate_fallback(),
                error_context=f"[EngineClass] generation failed: {e}"
            )
            
    def _ensure_initialized(self, engine_config):
        """Ensure engine and adapter are initialized"""
        if self.engine is None:
            # Initialize engine from config
            # Initialize adapter
            
    def _process_with_characters(self, text, character_voices, **params):
        """Process text with character switching"""
        
    def _process_simple(self, text, character_voices, **params):
        """Process simple text without character switching"""
```

---

## Implementation Phase Strategy

### Phase 1: Foundation (Implement First)
1. Core engine implementation
2. Basic text generation
3. Model loading and downloading
4. Engine configuration node
5. Integration with TTS Text node

**Stop here and test with user before proceeding**

### Phase 2: Core Features
1. Character switching integration
2. Language switching (if applicable)
3. Pause tag support
4. Caching system
5. VRAM management

### Phase 3: SRT Support
1. SRT processor implementation
2. Timing and assembly integration
3. Character switching in SRT
4. Performance optimization

### Phase 4: Special Features
1. Engine-specific unique features
2. Special nodes for unique capabilities
3. Advanced parameter controls

### Phase 5: Documentation and Polish
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

### Integration Verification

Before submitting implementation, verify:
- [ ] All unified systems are used correctly
- [ ] No duplicate logic exists
- [ ] Engine follows established patterns
- [ ] All features work with existing engines
- [ ] VRAM management works
- [ ] Caching works correctly
- [ ] Error handling is robust

---

## Notes for Future Updates

This guide should be updated when:
- New unified systems are added
- Architecture patterns change
- New requirements emerge
- Common issues are discovered

Always keep this guide in sync with the current codebase architecture.