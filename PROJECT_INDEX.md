# TTS Audio Suite - Project Index

*Comprehensive file index for the universal multi-engine TTS extension for ComfyUI*

## Architecture Overview

This extension features a **unified modular architecture** supporting multiple TTS engines:
- **Unified Node Interface**: Single set of nodes (TTS Text, TTS SRT, Voice Changer) that work with any engine via clean delegation
- **ComfyUI Model Management Integration**: All TTS models now integrate with ComfyUI's native model management system, enabling "Clear VRAM" functionality and automatic memory management
- **Universal Model Loading System**: Standardized model loading interface across all engines (ChatterBox, F5-TTS, Higgs Audio, VibeVoice, RVC, Audio Separation) with fallback support
- **Engine Processors**: Internal processing engines for each TTS system (ChatterBox, F5-TTS, Higgs Audio, VibeVoice) handling engine-specific orchestration
- **Engine Adapters**: Modular adapters for ChatterBox, F5-TTS, Higgs Audio 2, VibeVoice, and RVC voice conversion
- **Centralized Download System**: Unified downloader eliminates HuggingFace cache duplication with direct downloads to organized TTS/ folder structure, with full extra_model_paths.yaml support for shared model directories
- **Thread-Safe Architecture**: Stateless ChatterBox wrapper eliminates shared state corruption (Note: parallel processing is slower than sequential)
- **Universal Streaming Infrastructure**: Unified streaming system with configurable workers (batch_size parameter) - sequential mode (batch_size=0) recommended for optimal performance
- **Multilingual Support**: German and Norwegian models for ChatterBox, plus ChatterBox Official 23-Lang supporting 23 languages including Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Swahili, and Turkish
- **Smart Language Grouping**: SRT processing by language groups to minimize model switching
- **Character Voice Management**: Centralized character voice system with flexible input types
- **Comprehensive Audio Processing**: Interactive waveform analyzer, vocal separation, and audio mixing
- **Professional Voice Conversion**: RVC implementation with UVR5 vocal separation and advanced pitch control

## Documentation Files

**README.md** - Main project documentation with installation guide, features overview, and usage instructions for the TTS Audio Suite

**CLAUDE.md** - Development guidelines and project context for Claude Code when working with this codebase, including critical thinking requirements and response style guidelines

**CHANGELOG.md** - Complete version history with detailed feature additions, bug fixes, and architectural improvements across all versions

### User Documentation

**docs/CHARACTER_SWITCHING_GUIDE.md** - Guide for multi-character TTS using [CharacterName] tags, voice organization, and character voice management system

**docs/CHATTERBOX_V2_SPECIAL_TOKENS.md** - Complete guide for ChatterBox Official 23-Lang v2 special emotion and sound tokens (<giggle>, <whisper>, <sigh>, etc.) with 30+ expressive tokens for enhanced TTS generation

**docs/🌊_Audio_Wave_Analyzer-Complete_User_Guide.md** - Complete user guide for Audio Wave Analyzer with interactive waveform visualization and timing extraction

**docs/VERSION_3.1_RELEASE_GUIDE.md** - Release documentation for character switching system and overlapping subtitles support

### Developer Documentation

**docs/Dev reports/F5TTS_INTEGRATION_SPECIFICATION.md** - Technical specification for F5-TTS integration architecture

**docs/Dev reports/F5TTS_STREAMING_INTEGRATION_PLAN.md** - Universal streaming architecture plan to eliminate spaghetti code and enable F5-TTS streaming

**docs/Dev reports/STREAMING_ARCHITECTURE.md** - Streaming parallel processing architecture documentation with feasibility analysis for multiple engines

**docs/Dev reports/STATELESS_WRAPPER_IMPLEMENTATION_PLAN.md** - Stateless ChatterBox wrapper implementation solving shared model state corruption in parallel processing

**docs/Dev reports/POST_V4.2.3_DEVELOPMENT_REVIEW.md** - Comprehensive analysis of 45 commits implementing universal streaming architecture and performance insights

**docs/Dev reports/PAUSE_TAGS_IMPLEMENTATION_REPORT.md** - Implementation details for pause tag system with timing control syntax

**docs/Dev reports/REFACTORING_PLAN.md** - Architectural refactoring plan and migration strategy documentation

## Core Architecture Files

**__init__.py** - Main ComfyUI extension entry point defining WEB_DIRECTORY and delegating to nodes.py for node registration

**nodes.py** - Central module loader with sophisticated node discovery, conditional imports, and startup diagnostics for all TTS/VC/Audio features

## Unified Engine Architecture

### Engine Implementations

**engines/chatterbox/** - Complete ChatterBox TTS and VC engine implementation
- **tts.py** - ChatterBox TTS engine with multilingual support, perth watermarking, and ComfyUI model management integration
- **vc.py** - ChatterBox Voice Conversion with multilingual model loading
- **language_models.py** - Language model registry for English, German, and Norwegian models
- **audio_timing.py** - Audio timing utilities and time-stretching functionality
- **streaming_model_manager.py** - Pre-loading and management of multiple language models for streaming workers
- **batch_processor.py** - Batch processor for character group processing with true parallel inference
- **character_grouper.py** - Character grouping utilities for batch processing optimization
- **stateless_wrapper.py** - Thread-safe stateless wrapper eliminating shared state corruption in parallel generation
- **models/** - Complete ChatterBox model architecture (S3Gen, T3, tokenizers, voice encoder)

**engines/chatterbox_official_23lang/** - ChatterBox Official 23-Lang multilingual engine implementation
- **tts.py** - Main ChatterBox Official 23-Lang TTS engine with 23-language support
- **vc.py** - Voice conversion engine supporting all 23 languages with refinement passes
- **language_models.py** - Language model configuration and multilingual support definitions
- **stateless_wrapper.py** - Thread-safe wrapper for multilingual parallel processing
- **batch_processor.py**, **adaptive_processor.py**, **gpu_aware_processor.py** - Advanced processing systems
- **models/** - Complete multilingual model architecture (S3Gen, T3, tokenizers, voice encoder) optimized for 23-language generation

**engines/f5tts/** - F5-TTS engine implementation and editing capabilities
- **f5tts.py** - Main F5-TTS wrapper with ComfyUI integration
- **f5tts_edit_engine.py** - Speech editing engine for targeted word replacement
- **audio_compositing.py** - Audio compositing with crossfade and segment processing

**engines/higgs_audio/** - Higgs Audio 2 engine implementation with voice cloning
- **higgs_audio.py** - Main Higgs Audio 2 wrapper with voice cloning and chunking support
- **higgs_audio_downloader.py** - Model auto-download system for generation and tokenizer models
- **boson_multimodal/** - Complete boson_multimodal implementation from HiggsAudio team

**engines/vibevoice_engine/** - VibeVoice engine implementation with multi-speaker and long-form capabilities
- **vibevoice_engine.py** - Main VibeVoice wrapper with 90-minute generation and multi-speaker support
- **vibevoice_downloader.py** - Model auto-download system for Microsoft VibeVoice models (1.5B and 7B)

**engines/rvc/** - RVC (Real-time Voice Conversion) engine implementation
- **__init__.py** - RVC engine initialization and ComfyUI integration
- **hubert_downloader.py** - HuBERT model auto-download from Hugging Face with TTS/ folder organization
- **impl/** - Complete RVC implementation with vocal separation and voice conversion
  - **rvc_audio.py** - Audio processing utilities for RVC pipeline
  - **rvc_downloader.py** - Model downloading system for RVC, UVR, and Karafan models
  - **rvc_utils.py** - Utility functions for device detection and file management
  - **uvr5_cli.py** - UVR5 vocal separation command-line interface
  - **lib/** - Core RVC libraries and model implementations
    - **separators.py** - Audio separation algorithms (MDX, VR, UVR5)
    - **model_utils.py** - Model loading and hash utilities
    - **audio.py** - Audio I/O and format conversion utilities
    - **utils.py** - General utilities and merge functions
    - **karafan/** - Karafan separation model implementation
    - **infer_pack/** - RVC inference and model loading components
    - **uvr5_pack/** - UVR5 vocal separation algorithms

### Engine Adapters

**engines/adapters/chatterbox_adapter.py** - ChatterBox engine adapter providing standardized interface for unified nodes

**engines/adapters/f5tts_adapter.py** - F5-TTS engine adapter with parameter mapping and cache integration

**engines/adapters/higgs_audio_adapter.py** - Higgs Audio 2 engine adapter with voice cloning and parameter validation

**engines/adapters/vibevoice_adapter.py** - VibeVoice engine adapter with multi-speaker format conversion and parameter mapping

**engines/adapters/chatterbox_streaming_adapter.py** - ChatterBox streaming adapter bridging existing implementation to universal streaming system

**engines/adapters/f5tts_streaming_adapter.py** - F5-TTS streaming adapter enabling parallel processing with language model switching

## Node Implementations

### Engine Configuration Nodes

**nodes/engines/chatterbox_engine_node.py** - ChatterBox engine configuration node for language, device, and generation parameters

**nodes/engines/f5tts_engine_node.py** - F5-TTS engine configuration node with model selection and generation settings

**nodes/engines/higgs_audio_engine_node.py** - Higgs Audio 2 engine configuration node with voice cloning parameters and generation settings

**nodes/engines/vibevoice_engine_node.py** - VibeVoice engine configuration node with multi-speaker modes and long-form generation settings

**nodes/engines/rvc_engine_node.py** - RVC voice conversion node with pitch control and quality settings

**nodes/engines/index_tts_engine_node.py** - IndexTTS-2 engine configuration node with character voice selection and generation parameters

**nodes/engines/index_tts_emotion_options_node.py** - 🌈 IndexTTS-2 Emotion Vectors node with interactive radar chart for 8-emotion control

### Unified Interface Nodes

**nodes/unified/tts_text_node.py** - Universal TTS text generation node working with any configured engine

**nodes/unified/tts_srt_node.py** - Universal SRT subtitle processing node - clean delegation layer that routes to engine-specific processors

**nodes/unified/voice_changer_node.py** - Universal voice conversion node with multilingual model support and flexible audio inputs

### Shared Components

**nodes/shared/character_voices_node.py** - Character voice management system providing NARRATOR_VOICE outputs for any TTS node

### Text Processing Nodes

**nodes/text/phoneme_text_normalizer_node.py** - 📝 Phoneme Text Normalizer with multilingual text preprocessing for improved TTS pronunciation. Features 4 processing methods: Pass-through, Unicode Decomposition (ą→a̧), IPA Phonemization (espeak backend), and Character Mapping (ASCII fallback). Supports auto-language detection and cross-platform phonemizer support.

### Engine-Specific Nodes

*Note: These are internal processors/engines used by the Unified nodes, not direct ComfyUI interface nodes. They handle engine-specific orchestration while the Unified nodes provide the user interface.*

**nodes/chatterbox/** - ChatterBox engine implementation nodes (called by Unified nodes)
- **chatterbox_tts_node.py** - ChatterBox TTS engine node with streaming batch processing and character switching
- **chatterbox_srt_node.py** - ChatterBox SRT engine node with streaming parallel processing and timing modes  
- **chatterbox_vc_node.py** - ChatterBox voice conversion engine with iterative refinement

**nodes/chatterbox_official_23lang/** - ChatterBox Official 23-Lang multilingual processors
- **chatterbox_official_23lang_processor.py** - Main TTS processor supporting all 23 languages with character switching
- **chatterbox_official_23lang_srt_processor.py** - SRT subtitle processor with multilingual timing and audio assembly
- **chatterbox_official_23lang_vc_processor.py** - Voice conversion processor supporting all 23 languages with refinement passes

**nodes/f5tts/** - F5-TTS specific nodes
- **f5tts_node.py** - Direct F5-TTS generation node
- **f5tts_srt_node.py** - F5-TTS SRT processing with language grouping
- **f5tts_edit_node.py** - F5-TTS speech editor for word replacement
- **f5tts_edit_options_node.py** - Advanced F5-TTS editing configuration

**nodes/higgs_audio/** - Higgs Audio 2 internal processors
- **higgs_audio_srt_processor.py** - Higgs Audio SRT orchestrator with multi-speaker support, character switching, and timing modes (internal processor used by Unified SRT node)

**nodes/vibevoice/** - VibeVoice internal processors  
- **vibevoice_processor.py** - VibeVoice TTS orchestrator with multi-speaker support and long-form generation handling (internal processor used by Unified TTS node)

### Audio Processing System

**nodes/audio/analyzer_node.py** - Interactive Audio Wave Analyzer with web interface and timing extraction

**nodes/audio/analyzer_options_node.py** - Configuration options for audio analysis (silence detection, peak analysis)

**nodes/audio/recorder_node.py** - Voice recording node with microphone input and silence detection

**nodes/audio/vocal_removal_node.py** - Professional vocal/instrumental separation using UVR5, MDX, VR, and Karafan models

**nodes/audio/merge_audio_node.py** - Advanced audio mixing with multiple algorithms and pitch control

**nodes/audio/rvc_pitch_options_node.py** - Advanced pitch extraction settings for RVC voice conversion

### Video Analysis System

**nodes/video/mouth_movement_analyzer_node.py** - 🗣️ Silent Speech Analyzer for detecting mouth movement in silent video frames to extract precise timing for TTS SRT synchronization

**nodes/video/viseme_options_node.py** - Advanced viseme detection configuration with vowel/consonant detection, word prediction, and temporal analysis settings

### RVC Model Management

**nodes/models/load_rvc_model_node.py** - RVC model loader with FAISS index support and automatic downloading

## Base Classes and Foundation

**nodes/base/base_node.py** - Universal base class for all nodes with device resolution, model management, and cleanup

**nodes/base/f5tts_base_node.py** - F5-TTS specific base class extending universal foundation with F5-TTS requirements

## Utility Systems

### Model Management

**utils/models/manager.py** - Intelligent model discovery and caching with multilingual support for both TTS and VC models, integrated with ComfyUI model management and extra_model_paths.yaml for shared storage support

**utils/models/smart_loader.py** - Universal smart model loader preventing duplicate model loading across all engines and modes

**utils/models/f5tts_manager.py** - F5-TTS specific model management extending base manager functionality, now with ComfyUI integration

**utils/models/language_mapper.py** - Language-to-model mapping system with fallback support

**utils/models/fallback_utils.py** - Model fallback and error recovery utilities

**utils/models/comfyui_model_wrapper/** - Modular ComfyUI model wrapper system with engine-specific handlers (base_wrapper.py, model_manager.py, cache_utils.py, engine_handlers.py) enabling TTS models to integrate with ComfyUI's native model management

**utils/models/unified_model_interface.py** - Universal model loading interface providing standardized factory pattern for all engines (ChatterBox, F5-TTS, Higgs Audio, RVC, Audio Separation) with ComfyUI integration

**utils/models/extra_paths.py** - ComfyUI extra_model_paths.yaml integration system enabling shared model directories across multiple ComfyUI installations with intelligent path resolution and fallback support

### Audio Processing

**utils/audio/processing.py** - Comprehensive audio utilities for tensor manipulation, normalization, and format conversion

**utils/ffmpeg_utils.py** - Centralized FFmpeg dependency handling with graceful fallbacks for audio conversion and timing analysis

**utils/audio/analysis.py** - Advanced audio analysis including waveform processing, silence detection, and timing extraction

**utils/audio/audio_hash.py** - Centralized content-based hashing for consistent cache keys across all processing modes

**utils/audio/cache.py** - Unified caching system for TTS engines with engine-specific cache management

**utils/audio/chunk_combiner.py** - Modular chunk combination utility with smart auto-selection based on text analysis (sentence boundaries, commas, forced splits) and detailed timing information

**utils/audio/chunk_timing.py** - Chunk combination timing information helper providing standardized timing info integration across all TTS engines

### Text Processing

**utils/text/character_parser.py** - Universal character switching system with [CharacterName] tag parsing and language-aware processing

**utils/text/chunking.py** - Enhanced text chunking with sentence boundary detection and character limits

**utils/text/pause_processor.py** - Pause tag parsing supporting [pause:xx] syntax for precise timing control

**utils/text/segment_parameters.py** - Per-segment parameter system enabling inline control of seed, temperature, cfg, speed, and other engine-specific parameters via pipe-separated syntax like [Alice|seed:42|temperature:0.5]. Supports aliases (cfg_weight→cfg, temp→temperature) and case-insensitive input with universal parameter filtering and validation across all TTS engines

**utils/text/phonemizer_utils.py** - F5-TTS multilingual phonemization system with IPA conversion, cross-platform backend support (espeak-phonemizer-windows/phonemizer), smart language detection, and model-specific exceptions for optimal quality

### Voice Management

**utils/voice/discovery.py** - Enhanced voice file discovery with multi-path fallback system (models/voices/, models/TTS/voices/, extra_model_paths.yaml directories, voices_examples/), character mapping, and alias loading with priority support

**utils/voice/multilingual_engine.py** - Central orchestrator for multilingual TTS with language switching optimization

### System Integration

**utils/system/import_manager.py** - Smart dependency resolution managing bundled vs system installations

**utils/system/subprocess.py** - Safe subprocess execution with error handling and process isolation

### Compatibility System

**utils/compatibility/transformers_patches.py** - Centralized transformers compatibility patches (monkey patches) managing version compatibility across different transformers library versions including FlashAttentionKwargs, BaseStreamer, DynamicCache properties, and VibeVoice generation method signatures

**utils/compatibility/numba_compat.py** - Centralized Numba/Librosa compatibility system with fast startup testing and intelligent JIT fallback management for Python 3.13+ compatibility

### Downloads

**utils/downloads/unified_downloader.py** - Centralized download system for all engines eliminating HuggingFace cache duplication with direct HTTP downloads to organized TTS/ structure, featuring full extra_model_paths.yaml support for shared model directories

**utils/downloads/model_downloader.py** - RVC-specific model auto-download system with direct downloads (legacy, integrated with unified system)

### Timing and SRT

**utils/timing/engine.py** - Advanced SRT timing engine with smart audio synchronization

**utils/timing/assembly.py** - Audio segment assembly supporting multiple timing modes

**utils/timing/parser.py** - SRT subtitle parsing with timestamp extraction and validation

**utils/timing/reporting.py** - Timing analysis and report generation

### Streaming System

**utils/streaming/__init__.py** - Universal streaming system initialization

**utils/streaming/streaming_types.py** - Universal data structures (StreamingSegment, StreamingResult) eliminating format conversions

**utils/streaming/streaming_interface.py** - Abstract streaming interface for engine adapters

**utils/streaming/streaming_coordinator.py** - Universal coordinator replacing all format-specific routers

**utils/streaming/work_queue_processor.py** - Engine-agnostic parallel processing system

## Web Interface Components

**web/audio_analyzer_interface.js** - Main ComfyUI integration and node communication for Audio Wave Analyzer

**web/audio_analyzer_core.js** - Core JavaScript functionality for waveform visualization

**web/audio_analyzer_ui.js** - User interface components and control elements

**web/audio_analyzer_visualization.js** - Waveform rendering and canvas-based visualization

**web/audio_analyzer_regions.js** - Timing region management and interaction system

**web/audio_analyzer_controls.js** - Audio playback controls and user interaction handling

**web/audio_analyzer_widgets.js** - Custom ComfyUI widget implementations

**web/audio_analyzer_drawing.js** - Canvas drawing utilities and rendering functions

**web/audio_analyzer_events.js** - Event handling for mouse and keyboard interactions

**web/audio_analyzer_layout.js** - Layout management and UI positioning

**web/audio_analyzer_node_integration.js** - Node integration bridge for ComfyUI communication

**web/chatterbox_voice_capture.js** - Voice recording interface for microphone input

**web/index_tts_emotion_radar.js** - ComfyUI integration for 🌈 IndexTTS-2 Emotion Vectors radar chart

**web/emotion_radar_canvas_widget.js** - Interactive radar chart widget with dynamic color blending and click/drag controls

**web/chatterbox_srt_showcontrol.js** - SRT subtitle display and timing controls

**web/audio_analyzer.css** - Styling for audio analyzer interface components

## Development Tools

**scripts/bump_version_enhanced.py** - Advanced automated version management with changelog generation and git integration

**scripts/version_utils.py** - Version management utilities and helper functions

## Configuration Files

**requirements.txt** - Core dependencies including ChatterboxTTS, audio processing libraries, and optional F5-TTS dependencies

**pyproject.toml** - Project metadata with ComfyUI registry configuration and dependency specifications

## Example Workflows

**example_workflows/📺 Chatterbox SRT.json** - Complete SRT processing workflow with ChatterBox engine

**example_workflows/Chatterbox integration.json** - General ChatterBox TTS and Voice Conversion demonstration

**example_workflows/👄 F5-TTS Speech Editor Workflow.json** - F5-TTS speech editing with Audio Wave Analyzer

**example_workflows/🎤 📺 F5-TTS SRT and Normal Generation.json** - F5-TTS SRT processing and standard generation

## Voice Examples

**voices_examples/** - Character voice reference files with both audio samples and text descriptions
- **male/** - Male voice examples with reference audio and text
- **female/** - Female voice examples with reference audio and text
- **higgs_audio/** - Higgs Audio 2 voice presets (belinda, en_woman, en_man, chadwick, vex, etc.) with config.json
- **#character_alias_map.txt** - Character alias mapping configuration
- Individual character voice files (Clint Eastwood, David Attenborough, Morgan Freeman, Sophie Anderson)

## Project Management

**IgnoredForGitHubDocs/** - Development documentation excluded from main repository
- **FOLDER_STRUCTURE_REFACTORING_PLAN.md** - Architectural refactoring planning
- **CHATTERBOX_MODEL_BUG_REPORT.md** - Bug tracking and resolution documentation

## Architecture Summary

The TTS Audio Suite follows a **unified modular architecture** where:

1. **Engine Nodes** (`nodes/engines/`) provide user configuration interfaces
2. **Unified Nodes** (`nodes/unified/`) serve as clean delegation layers routing to appropriate engine processors  
3. **Engine Processors** (`nodes/chatterbox/`, `nodes/f5tts/`, `nodes/higgs_audio/`) handle engine-specific orchestration and workflow logic
4. **Engine Implementations** (`engines/`) handle the actual TTS/VC processing and model inference
5. **Adapters** (`engines/adapters/`) bridge low-level engines to higher-level processors
6. **Utilities** (`utils/`) provide shared functionality across all components
7. **Web Interface** (`web/`) enables interactive features like audio analysis

This layered design ensures consistent user experience while allowing engine-specific optimizations. The unified nodes act as thin delegation layers, eliminating code duplication and maintaining architectural consistency across all TTS engines.