# IndexTTS-2 Implementation Plan

## Overview
IndexTTS-2 is a breakthrough emotionally expressive and duration-controlled autoregressive zero-shot TTS engine featuring emotion disentanglement and precise voice cloning.

## Unique Features
- **Duration Control**: First autoregressive TTS model with precise duration control
- **Emotion Disentanglement**: Separates speaker identity from emotional expression
- **Multi-Modal Emotion Control**: Audio references, text descriptions, manual vectors
- **Zero-Shot Voice Cloning**: Single reference audio for voice cloning
- **Advanced Character Switching**: `[character:emotion_ref]` syntax

## Technical Specifications
- **Sample Rate**: 22,050 Hz
- **Audio Format**: `torch.Tensor` shape `[1, samples]`
- **Dependencies**: `torch==2.8.*`, `transformers==4.52.1`, IndexTTS-2 source code
- **Model Components**: GPT + Semantic codec + S2Mel + BigVGAN + CAMPPlus + QwenEmotion

## Implementation Status

### ✅ Completed
1. **Core Engine** (`engines/index_tts/index_tts.py`)
   - IndexTTSEngine class with unified model interface integration
   - Proper UnifiedModelInterface integration for loading/unloading
   - Factory registration in unified_model_interface.py
   - All official parameters exposed (emotion vectors, text emotions, generation params)

2. **Downloader** (`engines/index_tts/index_tts_downloader.py`)
   - Uses folder_paths.models_dir for correct TTS/IndexTTS/ structure
   - unified_downloader integration 
   - Downloads both HuggingFace model files and GitHub source code
   - Model verification system

3. **Cache Integration** (`utils/audio/cache.py`)
   - IndexTTSCacheKeyGenerator added to unified cache system
   - Handles all IndexTTS-2 parameters for cache key generation
   - Integrated with global audio_cache

4. **Engine Adapter** (`engines/adapters/index_tts_adapter.py` - IN PROGRESS)
   - Character switching support with emotion references
   - Cache integration and parameter mapping
   - Uses existing CharacterParser for tag parsing

### 🚧 In Progress  
5. **Character Tag Extension**
   - Need to handle flexible syntax: `[de:]`, `[de:Alice]`, `[Alice]`, `[Alice:angry_bob]`, `[de:Alice:angry_bob]`
   - Extend CharacterParser or create IndexTTS-specific parsing
   - Avoid ambiguity between language codes and character names

### 📋 Pending
6. **Engine Configuration Node** (`nodes/engines/index_tts_engine_node.py`)
   - UI node with all IndexTTS-2 parameters
   - Model path selection with auto-download
   - Emotion controls (audio references, vectors, text)
   - Generation parameters (temperature, top_p, top_k, etc.)

7. **TTS Text Integration** (`nodes/index_tts/`)
   - `index_tts_processor.py` - Main TTS processor for unified TTS Text node
   - `index_tts_srt_processor.py` - SRT subtitle processing with timing
   - Integration with character switching system

8. **Testing & Validation**
   - Basic functionality testing
   - Character switching validation  
   - Emotion control verification
   - Performance optimization

## Character Switching Syntax Design

### Proposed Flexible Syntax Resolution:
```
[de:]           → language="de", character="narrator", emotion=None
[de:Alice]      → language="de", character="Alice", emotion=None  
[Alice]         → language=None, character="Alice", emotion=None
[Alice:angry_bob] → language=None, character="Alice", emotion="angry_bob"
[de:Alice:angry_bob] → language="de", character="Alice", emotion="angry_bob"
```

### Resolution Strategy:
1. Split by `:` and count parts
2. 1 part: Check if it's a known language code, else treat as character
3. 2 parts: Check if first part is language code, else treat as character:emotion
4. 3 parts: Always language:character:emotion

### Implementation Files:
- Extend `utils/text/character_parser.py` with emotion support
- Or create `engines/index_tts/emotion_parser.py` for IndexTTS-specific parsing

## Model Management Integration

### Unified Systems Used:
- **UnifiedModelInterface**: Standardized loading/unloading
- **unified_downloader**: HuggingFace + Git downloads  
- **AudioCache**: Generation result caching
- **folder_paths**: Correct model directory structure
- **CharacterParser**: Tag parsing and language resolution

### Memory Management:
- ComfyUI model wrapper integration via UnifiedModelInterface
- Automatic VRAM management and "Clear VRAM" button support
- Model factory registration for proper lifecycle management

## Next Steps:
1. Resolve character tag ambiguity with flexible parsing
2. Create IndexTTS engine configuration node
3. Implement TTS Text processor integration
4. Test full character switching + emotion control workflow