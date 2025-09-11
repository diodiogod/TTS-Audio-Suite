# VibeVoice Integration Plan - Final Version

## Overview

Integrate Microsoft VibeVoice (1.5B and 7B models) into TTS Audio Suite following the unified architecture, with full support for all existing features while exposing VibeVoice's unique multi-speaker capabilities.

## Implementation Structure

### 1. Engine Implementation (`engines/vibevoice/`)

#### `vibevoice.py` (Main Engine - ~500 lines)

```python
class VibeVoiceEngine:
    def __init__(self)
    def initialize_engine(model_name, device)
    def generate_speech(text, voice_samples, cfg_scale, seed, ...)
    def generate_multi_speaker(segments, voice_samples, ...)
    def cleanup()  # Memory management
    def unload_models()  # Explicit unload (called by ComfyUI's unload button)
```

- Use `unified_model_interface` for model loading
- Integrate with `utils.audio.cache` for caching
- Support both models (1.5B and 7B)
- Handle memory management via ComfyUI's unload system

#### `vibevoice_downloader.py` (~200 lines)

```python
VIBEVOICE_MODELS = {
    "vibevoice-1.5B": {
        "repo": "microsoft/VibeVoice-1.5B",
        "files": [...],  # 5.4GB total
    },
    "vibevoice-7B": {
        "repo": "microsoft/VibeVoice-Large",
        "files": [...],  # 9.3GB total
    }
}
```

- Use `unified_downloader` for all downloads
- Download to `models/TTS/vibevoice/` structure
- No HuggingFace cache duplication

### 2. Engine Configuration Node (`nodes/engines/vibevoice_engine_node.py`) (~150 lines)

```python
class VibeVoiceEngineNode:
    INPUT_TYPES:
        Required:
            - model: ["VibeVoice-1.5B", "VibeVoice-7B"]
            - device: ["auto", "cuda", "cpu"]
            - multi_speaker_mode: ["Custom Character Switching", "Native Multi-Speaker"]
            - cfg_scale: (1.0-2.0, default 1.3)
            - use_sampling: BOOLEAN
            - temperature: (0.1-2.0, default 0.95)
            - top_p: (0.1-1.0, default 0.95)
            - chunk_minutes: INT (0=disabled, UI shows minutes, backend converts to chars)
        Optional:
            - speaker2_voice: AUDIO
            - speaker3_voice: AUDIO
            - speaker4_voice: AUDIO
```

- Returns TTS_ENGINE configuration dict
- NO seed parameter (that's in unified nodes)
- NO load/unload toggles (handled by ComfyUI system)
- Uses same naming as Higgs: "Custom Character Switching" not "segment_per_character"
- Follows ChatterBox/Higgs patterns exactly

### 3. Engine Adapter (`engines/adapters/vibevoice_adapter.py`) (~400 lines)

```python
class VibeVoiceEngineAdapter:
    def __init__(self, node_instance)
    def get_model_for_language(lang_code, default_model)
    def load_base_model(model_name, device)
    def generate_segment(text, voice_ref, params)
    def process_character_segments(segments, voice_mapping)
    def convert_character_to_speaker_format(text)  # [Alice] → Speaker 0
    def handle_pause_tags(text)  # Use pause_processor
    def handle_language_tags(text)  # Parse [de:Alice] syntax
```

- Bridge unified nodes to VibeVoice engine
- Two modes matching Higgs naming:
  - `Custom Character Switching`: Generate per character (like ChatterBox/Higgs)
  - `Native Multi-Speaker`: Single generation with multiple speakers
- Character mapping: [Alice], [Bob] → Speaker 0, Speaker 1
- Full integration with existing systems

### 4. Internal Processors (`nodes/vibevoice/`)

#### `vibevoice_processor.py` (~300 lines)

- Internal TTS processor (called by unified TTS node)
- Handles actual generation orchestration
- Manages chunking (time-based UI, character-based backend)
- Memory management coordination

#### `vibevoice_srt_processor.py` (~400 lines)

- Internal SRT processor (called by unified SRT node)
- Language grouping optimization
- Multi-speaker dialogue handling
- Timing synchronization

### 5. Model Factory Registration (~50 lines in existing files)

In `utils/models/unified_model_interface.py`:

```python
# Register VibeVoice model factories
interface.register_model_factory(
    "vibevoice", "tts", 
    lambda **kwargs: load_vibevoice_model(**kwargs)
)
```

### 6. Dependencies Update

`requirements.txt`:

```
# VibeVoice dependencies
transformers>=4.44.0
git+https://github.com/microsoft/VibeVoice.git
```

## Key Integration Points

### Memory Management

- Memory unloading handled by ComfyUI's "Unload Models" button
- Engine implements `unload_models()` method
- Uses ComfyUI's model management system
- Auto-cleanup on node destruction

### Unified Systems Integration

- **Cache**: Use `utils.audio.cache` with content-based hashing
- **Characters**: Full `utils.text.character_parser` support
- **Pause Tags**: `utils.text.pause_processor` integration
- **Downloads**: `utils.downloads.unified_downloader` only
- **Model Loading**: `utils.models.unified_model_interface`
- **Chunking**: `utils.text.chunking` with time-based UI

### Two Generation Modes (Named Like Higgs)

1. **Custom Character Switching** (default):
   
   - Preserves all TTS Audio Suite features
   - Generate audio per character, then combine
   - Supports pause tags, language switching per character
   - More flexible but slower

2. **Native Multi-Speaker**:
   
   - Uses VibeVoice's native multi-speaker
   - Single generation pass
   - More efficient but less flexible
   - Auto-converts [Character] → Speaker N format

### Time-Based Chunking

- UI shows minutes: "Chunk every [5] minutes"
- Backend converts: 5 min → ~3750 chars (150 wpm * 5 chars/word * 5)
- More intuitive for long-form content
- Optional (0 = no chunking)

### Seed Parameter Location

- Seed is in unified nodes (TTS Text, TTS SRT) NOT in engine node
- Follows existing pattern where engine provides configuration, unified nodes handle generation parameters

## File Size Limits

- Each file stays under 500-600 lines
- Modular design with clear separation
- Reuses all existing utilities
- No code duplication

## Testing Plan

1. Test both 1.5B and 7B models
2. Validate character switching in both modes
3. Test pause tag processing
4. Verify memory management via ComfyUI unload
5. Test long-form generation with chunking
6. Validate multi-speaker native mode
7. Test cache functionality
8. Verify SRT synchronization

## Benefits

- Full integration with all TTS Audio Suite features
- Choice between flexibility (Custom mode) and efficiency (Native mode)
- Proper memory management via ComfyUI system
- No code duplication
- Follows established patterns exactly (Higgs Audio style)
- Exposes unique 90-minute and 4-speaker capabilities

## Implementation Notes

- Created: 2025-08-29
- Author: Claude (TTS Audio Suite Assistant)
- Status: Ready for implementation