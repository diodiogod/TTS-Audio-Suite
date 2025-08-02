# 🌍 Multilanguage System Modularization Plan

## Overview

The current multilanguage implementation has grown to 700-800 lines per node. This plan outlines modularizing the system into reusable components to reduce node complexity to 400-500 lines and enable easy adoption by SRT nodes.

---

## 🎯 Goals

1. **Reduce node file sizes** from 700-800 lines to 400-500 lines
2. **Eliminate code duplication** between F5-TTS and ChatterBox nodes
3. **Enable easy SRT integration** for both engines
4. **Improve maintainability** with centralized logic
5. **Preserve performance** with existing cache optimizations

---

## 🏗️ Proposed Architecture

### Core Modules

#### 1. `core/multilingual_engine.py`
**Central orchestrator for multilingual TTS generation**

```python
class MultilingualEngine:
    def __init__(self, engine_type: str):  # "f5tts" or "chatterbox"
        self.engine_type = engine_type
        self.language_mapper = LanguageModelMapper(engine_type)
        self.cache_manager = CacheManager(engine_type)
        self.character_manager = CharacterVoiceManager()
        
    def process_multilingual_text(self, text: str, **params) -> AudioResult:
        """Main entry point for multilingual processing"""
        # 1. Parse character segments with languages
        # 2. Check cache optimization opportunities  
        # 3. Group by language and process
        # 4. Reorder and combine audio
        # 5. Return results with timing info
```

#### 2. `core/language_model_mapper.py`
**Maps language codes to engine-specific models**

```python
class LanguageModelMapper:
    def __init__(self, engine_type: str):
        self.engine_type = engine_type
        self.mappings = self._load_mappings()
    
    def get_model_for_language(self, lang_code: str, default_model: str) -> str:
        """Map language code to engine-specific model name"""
        
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        
    @staticmethod
    def _load_mappings() -> Dict[str, Dict[str, str]]:
        """Load language mappings from config"""
        return {
            "f5tts": {
                "en": "F5TTS_Base", "de": "F5-DE", "es": "F5-ES", 
                "fr": "F5-FR", "jp": "F5-JP", "it": "F5-IT", 
                "th": "F5-TH", "pt": "F5-PT-BR"
            },
            "chatterbox": {
                "en": "English", "de": "German", "no": "Norwegian",
                "nb": "Norwegian", "nn": "Norwegian"
            }
        }
```

#### 3. `core/cache_manager.py`
**Centralized cache optimization and pre-checking**

```python
class CacheManager:
    def __init__(self, engine_type: str):
        self.engine_type = engine_type
        
    def check_language_group_cached(self, language_groups: Dict, **params) -> Dict[str, bool]:
        """Check which language groups are fully cached"""
        
    def check_pause_segments_cached(self, pause_segments: List, **params) -> bool:
        """Check if pause tag segments are cached"""
        
    def generate_cache_key(self, segment_info: SegmentInfo) -> str:
        """Generate cache key for any segment"""
        
    def should_skip_model_loading(self, language_groups: Dict, **params) -> Tuple[bool, str]:
        """Determine if model loading can be skipped entirely"""
```

#### 4. `core/character_voice_manager.py`
**Centralized character voice resolution and mapping**

```python
class CharacterVoiceManager:
    def build_voice_references(self, characters: List[str], engine_type: str, 
                              main_reference) -> Dict[str, Any]:
        """Build character voice references for any engine"""
        
    def get_character_audio_component(self, character: str, stable_component: str) -> str:
        """Get cache-stable audio component for character"""
        
    def resolve_character_voices(self, character_mapping: Dict, 
                                main_reference, engine_type: str) -> Dict:
        """Resolve character voices with fallbacks"""
```

#### 5. `core/multilingual_processor.py` 
**Language grouping, processing, and audio ordering**

```python
class MultilingualProcessor:
    def __init__(self, engine_type: str):
        self.engine_type = engine_type
        
    def group_segments_by_language(self, segments_with_lang: List) -> Dict:
        """Group segments by language with original order tracking"""
        
    def process_language_groups(self, language_groups: Dict, 
                               generation_func: Callable, **params) -> List:
        """Process each language group and maintain order"""
        
    def reorder_audio_segments(self, audio_segments_with_order: List) -> List:
        """Sort audio segments back to original order"""
        
    def generate_language_info(self, languages: List, characters: List) -> str:
        """Generate descriptive info about processed languages/characters"""
```

---

## 🔧 Node Refactoring

### Simplified Node Structure

#### `nodes/f5tts_node.py` (Target: ~400 lines)
```python
class F5TTSNode(BaseF5TTSNode):
    def __init__(self):
        super().__init__()
        self.multilingual_engine = MultilingualEngine("f5tts")
    
    def generate_speech(self, **inputs):
        def _process():
            # Basic validation and setup (50 lines)
            validated_inputs = self.validate_inputs(**inputs)
            
            # Check if multilingual/multicharacter
            if self._is_multilingual_or_multicharacter(validated_inputs["text"]):
                # Use multilingual engine (10 lines)
                return self.multilingual_engine.process_multilingual_text(
                    text=validated_inputs["text"],
                    engine_adapter=F5TTSEngineAdapter(self),
                    **validated_inputs
                )
            else:
                # Single character/language mode (100 lines)
                return self._process_single_mode(validated_inputs)
        
        return self.process_with_error_handling(_process)
    
    def _process_single_mode(self, inputs):
        """Handle single character/language generation with cache optimization"""
        # Existing single mode logic, but with cache manager
        
    def _is_multilingual_or_multicharacter(self, text: str) -> bool:
        """Quick check if text needs multilingual processing"""
```

#### `adapters/f5tts_adapter.py`
**Engine-specific adapter for F5-TTS**
```python
class F5TTSEngineAdapter:
    def __init__(self, node_instance):
        self.node = node_instance
        
    def load_base_model(self, model_name: str, device: str):
        """Load base F5-TTS model"""
        self.node.load_f5tts_model(model_name, device)
        
    def load_language_model(self, model_name: str, device: str):
        """Load language-specific F5-TTS model"""
        self.node.load_f5tts_model(model_name, device)
        
    def generate_segment_audio(self, text: str, char_audio: str, char_text: str, **params):
        """Generate F5-TTS audio for a segment"""
        return self.node.generate_f5tts_with_pause_tags(
            text=text, ref_audio_path=char_audio, ref_text=char_text, **params
        )
        
    def combine_audio_chunks(self, audio_segments: List, **params):
        """Combine F5-TTS audio chunks"""
        return self.node.combine_f5tts_audio_chunks(audio_segments, **params)
```

#### `adapters/chatterbox_adapter.py`
**Engine-specific adapter for ChatterBox**
```python
class ChatterBoxEngineAdapter:
    def __init__(self, node_instance):
        self.node = node_instance
        
    def load_base_model(self, language: str, device: str):
        """Load base ChatterBox model"""
        self.node.load_tts_model(device, language)
        
    def load_language_model(self, language: str, device: str):
        """Load language-specific ChatterBox model"""
        self.node.load_tts_model(device, language)
        
    def generate_segment_audio(self, text: str, char_audio: str, **params):
        """Generate ChatterBox audio for a segment"""
        return self.node._generate_tts_with_pause_tags(
            text=text, audio_prompt=char_audio, **params
        )
```

---

## 🎬 SRT Integration Benefits

### Easy SRT Node Enhancement

#### `nodes/f5tts_srt_node.py`
```python
class F5TTSSRTNode(BaseF5TTSNode):
    def __init__(self):
        super().__init__()
        self.multilingual_engine = MultilingualEngine("f5tts")
    
    def generate_srt_speech(self, **inputs):
        # Parse SRT content (existing logic - 50 lines)
        subtitles = self._parse_srt_content(inputs["srt_content"])
        
        # Process each subtitle with multilingual support
        for subtitle in subtitles:
            if self._is_multilingual_or_multicharacter(subtitle.text):
                # Use multilingual engine for this subtitle (5 lines)
                audio = self.multilingual_engine.process_multilingual_text(
                    text=subtitle.text,
                    engine_adapter=F5TTSEngineAdapter(self),
                    **inputs
                )
            else:
                # Single mode (existing logic)
                audio = self._process_subtitle_single_mode(subtitle, inputs)
```

---

## 📋 Migration Plan

### Phase 1: Core Module Development
1. Create `MultilingualEngine` class
2. Extract `LanguageModelMapper` from existing nodes
3. Create `CacheManager` with existing cache logic
4. Develop `CharacterVoiceManager`
5. Build `MultilingualProcessor`

### Phase 2: Adapter Development  
1. Create `F5TTSEngineAdapter`
2. Create `ChatterBoxEngineAdapter`
3. Define adapter interface

### Phase 3: Node Refactoring
1. Refactor `f5tts_node.py` to use multilingual engine
2. Refactor `tts_node.py` to use multilingual engine
3. Validate functionality and performance

### Phase 4: SRT Integration
1. Add multilingual support to `f5tts_srt_node.py`
2. Add multilingual support to `srt_tts_node.py`
3. Testing and optimization

### Phase 5: Cleanup
1. Remove duplicated code
2. Update documentation
3. Performance benchmarking

---

## 📊 Expected Benefits

### Code Reduction
- **F5-TTS Node**: 700+ lines → ~400 lines (43% reduction)
- **ChatterBox Node**: 800+ lines → ~420 lines (48% reduction)
- **SRT Nodes**: Add multilingual support with ~50 additional lines

### Maintainability
- ✅ Single source of truth for language mappings
- ✅ Centralized cache optimization logic
- ✅ Consistent behavior across all nodes
- ✅ Easier testing with isolated modules
- ✅ Future language additions require only config changes

### Performance
- ✅ Preserve existing cache optimizations
- ✅ Potential for further optimizations in centralized modules
- ✅ Consistent cache key generation across engines

---

## 🧪 Testing Strategy

### Unit Tests
- Test each core module independently
- Test adapters with mock engines
- Test cache optimization logic

### Integration Tests  
- Test complete multilingual workflows
- Compare performance with current implementation
- Test SRT node integration

### Regression Tests
- Ensure existing functionality is preserved
- Validate cache behavior consistency
- Performance benchmarks

---

## 🚀 Future Enhancements

### Configuration-Driven Language Support
```yaml
# config/language_models.yaml
engines:
  f5tts:
    models:
      en: ["F5TTS_Base", "F5TTS_v1_Base", "E2TTS_Base"]
      de: ["F5-DE"]
      es: ["F5-ES"]
    fallback: "F5TTS_Base"
  
  chatterbox:
    models:
      en: ["English"]
      de: ["German"] 
      no: ["Norwegian"]
    fallback: "English"
```

### Plugin Architecture
- Enable third-party engines through adapters
- Hot-swappable language model configurations
- Dynamic language support detection

### Advanced Optimizations
- Cross-engine cache sharing
- Intelligent model pre-loading
- Language-aware text chunking optimization

---

*This modularization will make the codebase more maintainable, reduce duplication, and enable rapid feature development while preserving all existing functionality and performance optimizations.*