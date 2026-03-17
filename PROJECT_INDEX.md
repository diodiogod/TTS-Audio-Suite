# TTS Audio Suite - Project Index

*Comprehensive file index for the universal multi-engine TTS extension for ComfyUI*

## Architecture Overview

**Unified modular architecture** - all engines share the same layered pattern:

1. **Engine Node** (`nodes/engines/<engine>_engine_node.py`) - ComfyUI configuration UI
2. **Unified Nodes** (`nodes/unified/`) - thin delegation layer, routes to engine processors
3. **Processor** (`nodes/<engine>/<engine>_processor.py`) - orchestration, chunking, character/pause/tag handling
4. **SRT Processor** (`nodes/<engine>/<engine>_srt_processor.py`) - SRT-specific orchestration
5. **Adapter** (`engines/adapters/<engine>_adapter.py`) - bridges processor to engine implementation
6. **Engine Implementation** (`engines/<engine>/`) - actual model inference

**Key architectural rules:**
- Chunking happens in the **processor**, not the adapter (`generate_single()` on adapter = raw single call)
- YAML (`docs/Dev reports/tts_audio_suite_engines.yaml`) is source of truth for engine doc tables → run `python3 scripts/generate_engine_tables.py --readme` to regenerate
- Auxiliary YAML (`docs/Dev reports/tts_audio_suite_aux_models.yaml`) is source of truth for helper/post-process model docs → run `python3 scripts/generate_aux_model_docs.py`
- All models download to `ComfyUI/models/TTS/<model-name>/`
- Engine registry: `utils/models/engine_registry.py`

## Engines

12 engines follow the pattern above:

| Engine | Adapter | Processor | SRT Processor | Engine Node |
|--------|---------|-----------|---------------|-------------|
| ChatterBox | `chatterbox_adapter.py` | `nodes/chatterbox/chatterbox_tts_node.py` | `chatterbox_srt_node.py` | `chatterbox_engine_node.py` |
| ChatterBox 23-Lang | `chatterbox_streaming_adapter.py` | `nodes/chatterbox_official_23lang/` | same folder | `chatterbox_engine_node.py` |
| F5-TTS | `f5tts_adapter.py` | `nodes/f5tts/f5tts_node.py` | `f5tts_srt_node.py` | `f5tts_engine_node.py` |
| Higgs Audio 2 | `higgs_audio_adapter.py` | — | `nodes/higgs_audio/higgs_audio_srt_processor.py` | `higgs_audio_engine_node.py` |
| VibeVoice | `vibevoice_adapter.py` | `nodes/vibevoice/vibevoice_processor.py` | — | `vibevoice_engine_node.py` |
| IndexTTS-2 | — | `engines/index_tts/` | — | `index_tts_engine_node.py` |
| Step Audio EditX | `step_audio_editx_adapter.py` | `nodes/step_audio_editx/step_audio_editx_processor.py` | `step_audio_editx_srt_processor.py` | `step_audio_editx_engine_node.py` |
| CosyVoice3 | `cosyvoice_adapter.py` | `engines/processors/cosyvoice_processor.py` | `nodes/cosyvoice/cosyvoice_srt_processor.py` | `cosyvoice_engine_node.py` |
| Qwen3-TTS | `qwen3_tts_adapter.py` | `nodes/qwen3_tts/qwen3_tts_processor.py` | same folder | `qwen3_tts_engine_node.py` |
| Granite ASR | `asr_granite_adapter.py` | — | — | `granite_asr_engine_node.py` |
| Echo-TTS | `echo_tts_adapter.py` | `nodes/echo_tts/echo_tts_processor.py` | `echo_tts_srt_processor.py` | `echo_tts_engine_node.py` |
| RVC | — | `engines/rvc/` | — | `rvc_engine_node.py` |

**Engine implementations live in:**
- `engines/chatterbox/`, `engines/chatterbox_official_23lang/`, `engines/f5tts/`, `engines/higgs_audio/`, `engines/vibevoice_engine/`, `engines/step_audio_editx/`, `engines/cosyvoice/`, `engines/qwen3_tts/`, `engines/qwen3_asr/`, `engines/granite_asr/`, `engines/rvc/`

## Documentation Files

**README.md** - Main project docs, installation, features overview
**CLAUDE.md** - Dev guidelines for Claude Code
**CHANGELOG.md** - Full version history

### User Docs (`docs/`)
- `CHARACTER_SWITCHING_GUIDE.md` - [CharacterName] tag system
- `PARAMETER_SWITCHING_GUIDE.md` - Per-segment parameter override syntax
- `INLINE_EDIT_TAGS_USER_GUIDE.md` - Step Audio EditX inline tags
- `COSYVOICE3_TAGS_GUIDE.md` - CosyVoice3 native paralinguistic tags
- `CHATTERBOX_V2_SPECIAL_TOKENS.md` - ChatterBox v2 emotion tokens
- `IndexTTS2_Emotion_Control_Guide.md` - IndexTTS-2 emotion vectors
- `VOCAL_REMOVAL_GUIDE.md` - Vocal separation guide
- `qwen3_tts_optimizations.md` - Qwen3-TTS torch.compile setup
- `MODEL_DOWNLOAD_SOURCES.md` - All HF repo links (auto-generated)
- `MODEL_LAYOUTS.md` - Folder structures (auto-generated)
- `AUX_MODEL_SOURCES.md`, `AUX_MODEL_LAYOUTS.md` - Auxiliary helper-model registries (auto-generated)
- `ENGINE_COMPARISON.md`, `LANGUAGE_SUPPORT.md`, `FEATURE_COMPARISON.md` - Auto-generated tables

### Dev Docs (`docs/Dev reports/`)
- `tts_audio_suite_engines.yaml` - **Source of truth** for all engine metadata
- `tts_audio_suite_aux_models.yaml` - **Source of truth** for helper/post-process model metadata
- `BUMP_SCRIPT_INSTRUCTIONS.md` - Version bump process
- `SRT_IMPLEMENTATION.md` - SRT timing technical details

## Core Files

**`__init__.py`** - ComfyUI entry point
**`nodes.py`** - Central node discovery and registration
**`install.py`** - Dependency installer with smart skip guards

## Node Implementations

### Unified Interface
- `nodes/unified/tts_text_node.py` - Universal TTS text node
- `nodes/unified/tts_srt_node.py` - Universal SRT node
- `nodes/unified/voice_changer_node.py` - Universal voice conversion
- `nodes/unified/asr_transcribe_node.py` - Universal ASR node

### Shared / Special Nodes
- `nodes/shared/character_voices_node.py` - Character voice management (NARRATOR_VOICE output)
- `nodes/text/phoneme_text_normalizer_node.py` - Multilingual text preprocessing
- `nodes/text/asr_punctuation_truecase_node.py` - Standalone punctuation / truecase cleanup for raw ASR text
- `nodes/subtitles/text_to_srt_builder_node.py` - Build SRT from transcript text plus timing data
- `nodes/subtitles/srt_advanced_options_node.py` - Subtitle readability / segmentation policy options
- `nodes/text/tts_tag_editor_node.py` - 🏷️ Multiline TTS Tag Editor: rich text editor with character/language/parameter dropdowns, preset system, syntax highlighting, undo/redo — pairs with `web/string_multiline_tag_editor.js`
- `nodes/step_audio_editx_special/step_audio_editx_audio_editor_node.py` - 🎨 Audio Editor: post-process ANY engine's audio with Step Audio EditX (14 emotions, 32 styles, paralinguistic effects like `<Laughter>`, speed control) — universal, not just for Step Audio EditX engine
- `nodes/engines/index_tts_emotion_options_node.py` - IndexTTS-2 emotion radar chart
- `nodes/qwen3_tts/qwen3_tts_voice_designer_node.py` - Qwen3 voice-from-text-description

### Audio / Video Nodes
- `nodes/audio/analyzer_node.py` - Audio Wave Analyzer
- `nodes/audio/vocal_removal_node.py` - Vocal/instrumental separation
- `nodes/audio/recorder_node.py` - Microphone recording
- `nodes/audio/merge_audio_node.py` - Audio mixing
- `nodes/video/mouth_movement_analyzer_node.py` - Silent speech timing extractor
- `nodes/models/load_rvc_model_node.py` - RVC model loader

## Utility Systems

### Model Management (`utils/models/`)
- `unified_model_interface.py` - Universal factory pattern for all engines
- `engine_registry.py` - Engine capability definitions
- `manager.py` - Model discovery and caching
- `comfyui_model_wrapper/` - ComfyUI native model management integration
- `extra_paths.py` - extra_model_paths.yaml support

### Audio (`utils/audio/`)
- `processing.py` - Tensor manipulation, normalization, format conversion
- `cache.py` - Unified TTS caching system
- `chunk_combiner.py` - Smart chunk combination (auto/concatenate/crossfade/silence_padding)
- `chunk_timing.py` - Standardized chunk timing info across engines
- `audio_hash.py` - Content-based cache key hashing
- `analysis.py` - Waveform analysis, silence detection, timing extraction
- `edit_post_processor.py` - Batch inline edit tag post-processing

### Text (`utils/text/`)
- `character_parser.py` - [CharacterName] tag parsing
- `chunking.py` - Sentence-boundary-aware text splitting
- `pause_processor.py` - [pause:Ns] tag parsing and silence generation
- `segment_parameters.py` - Per-segment parameter parsing ([seed:42|temp:0.5])
- `step_audio_editx_special_tags.py` - Inline edit tag system
- `cosyvoice_special_tags.py` - CosyVoice3 native tags
- `phonemizer_utils.py` - F5-TTS multilingual phonemization

### Timing / SRT (`utils/timing/`)
- `engine.py` - SRT timing engine
- `assembly.py` - Audio segment assembly (multiple timing modes)
- `parser.py` - SRT parsing and validation
- `reporting.py` - Timing report generation

### Other Utils
- `utils/voice/discovery.py` - Voice file discovery with multi-path fallback
- `utils/downloads/unified_downloader.py` - Centralized HF download system
- `utils/compatibility/transformers_patches.py` - transformers version compatibility patches
- `utils/compatibility/numba_compat.py` - Numba/Librosa Python 3.13+ compatibility
- `utils/ffmpeg_utils.py` - FFmpeg with graceful fallback
- `utils/asr/` - ASR types, adapter registry, pipeline

## Web Interface

### TTS Tag Editor (modular JS)
`web/string_multiline_tag_editor.js` + `widget-*.js`, `editor-state.js`, `tag-utilities.js`, `syntax-highlighter.js`, `font-controls.js`

### Audio Analyzer
`web/audio_analyzer_*.js` (core, ui, visualization, regions, controls, widgets, drawing, events, layout, node_integration)

### Other Web Files
- `web/chatterbox_voice_capture.js` - Microphone recording UI
- `web/index_tts_emotion_radar.js` + `emotion_radar_canvas_widget.js` - IndexTTS-2 radar chart
- `web/qwen3_tts_widgets.js` - Qwen3 conditional instruction field
- `web/asr_srt_preset_widgets.js` - ASR SRT preset locking

## Scripts & Config
- `scripts/bump_version_enhanced.py` - Version bump with changelog (use `patch`/`minor`/`major`)
- `scripts/generate_engine_tables.py` - Regenerate all docs from YAML (`--readme` flag for README too)
- `requirements.txt`, `pyproject.toml` - Dependencies and project metadata
