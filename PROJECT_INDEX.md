# ComfyUI ChatterBox Voice - Project Index

*Concise file index for reference - describes purpose and role of each file in the modular TTS/Voice system*

## Architecture Overview

This extension features a **modular multilingual architecture** that provides:
- **Smart Language Grouping**: SRT nodes process subtitles by language groups (English→German→French) instead of sequential order (1→2→3→4) to minimize model switching
- **Unified Engine Support**: Both F5-TTS and ChatterBox engines share the same multilingual processing core with engine-specific adapters
- **Comprehensive Caching**: Engine-specific cache key generation with global cache management for optimal performance
- **Character & Language Switching**: Advanced parsing supports `[character]` tags and language codes like `[de:Alice]` with automatic fallbacks

## Documentation Files

**README.md** - Main project documentation with installation, features, and usage guide for ChatterBox Voice TTS/VC extension

**CLAUDE.md** - Development guidelines and instructions for Claude Code when working with this codebase

**CHANGELOG.md** - Complete version history with detailed feature additions, fixes, and changes from v1.1.0 to v3.2.2

### User Documentation

**docs/CHARACTER_SWITCHING_GUIDE.md** - Comprehensive guide for multi-character TTS using [CharacterName] tags, voice folder organization, and alias mapping

**docs/🌊_Audio_Wave_Analyzer-Complete_User_Guide.md** - Complete guide for Audio Wave Analyzer with interactive waveform visualization, timing extraction, and F5-TTS integration

**docs/VERSION_3.1_RELEASE_GUIDE.md** - Release documentation for v3.1.0 featuring character switching system and overlapping subtitles support

### Developer Documentation

**docs/Dev reports/F5TTS_INTEGRATION_SPECIFICATION.md** - Comprehensive technical specification and architecture guide for F5-TTS integration with ChatterBox Voice extension

**docs/Dev reports/CLAUDE_VERSION_MANAGEMENT_GUIDE.md** - Detailed instructions for automated version bumping using enhanced scripts with multiline changelog support

**docs/PAUSE_TAGS_IMPLEMENTATION_REPORT.md** - Implementation status report for pause tag system with syntax, usage examples, and known issues

**docs/Dev reports/F5TTS_IMPLEMENTATION_SUMMARY.md** - Implementation status summary for F5-TTS integration core components and features

**docs/Dev reports/SRT_IMPLEMENTATION.md** - Complete technical guide for SRT subtitle timing support with usage examples and API reference

## Core Architecture Files

**__init__.py** - Main ComfyUI extension entry point that dynamically imports from nodes.py, registers all available TTS/VC/SRT/F5-TTS nodes with graceful fallbacks for missing dependencies

**nodes.py** - Central module loader orchestrating the entire extension with sophisticated node discovery, conditional imports, and startup diagnostics for ChatterBox/F5-TTS/SRT/Audio features

**nodes/__init__.py** - Empty package marker for nodes directory

## Modular Multilingual Architecture

**utils/voice/multilingual_engine.py** - Central orchestrator for multilingual TTS generation handling language switching, character management, and smart model loading optimization for any TTS engine

**utils/models/language_mapper.py** - Language-to-model mapping system for F5-TTS and ChatterBox engines with fallback support and unsupported language warnings

**utils/audio/cache.py** - Unified caching system for TTS engines with engine-specific cache key generators, global cache management, and modular cache function factory

**engines/adapters/f5tts_adapter.py** - F5-TTS engine adapter providing standardized interface for F5-TTS operations in the modular multilingual engine with cache integration and parameter handling

**engines/adapters/chatterbox_adapter.py** - ChatterBox engine adapter providing standardized interface for ChatterBox operations in the modular multilingual engine with external caching and parameter mapping

## Core Foundation

**utils/system/import_manager.py** - Smart dependency resolution system managing bundled ChatterBox vs system installations with graceful fallbacks and import status tracking

**utils/models/manager.py** - Intelligent model discovery and caching across bundled models, ComfyUI directories, and HuggingFace auto-download with source prioritization

**nodes/base/base_node.py** - Base class providing common functionality for all ChatterBox Voice nodes including device resolution and temp file cleanup

**utils/text/chunking.py** - Enhanced text chunker with character-based limits, sentence boundary detection, and Orpheus TTS-inspired splitting algorithms

**utils/audio/processing.py** - Audio utility functions for tensor manipulation, duration calculation, normalization, and common audio operations

**utils/text/character_parser.py** - Universal character switching system using [CharacterName] tags with language-aware parsing, regex fix for empty character names like [fr:], and voice folder integration

**utils/text/pause_processor.py** - Pause tag parsing and audio generation supporting [pause:xx] syntax in seconds/milliseconds for precise timing control

**utils/models/f5tts_manager.py** - F5-TTS specific model manager extending base ModelManager with F5-TTS model discovery, loading, and caching functionality

**core/f5tts_edit_engine.py** - Core F5-TTS speech editing engine for targeted word/phrase replacement while maintaining voice characteristics

**core/f5tts_edit_cache.py** - Cache management system for F5-TTS edit operations with LRU caching and configurable size limits for improved iteration speed

**core/audio_compositing.py** - Audio compositing utilities for F5-TTS editing with crossfade curves, adaptive duration, and segment-by-segment processing

**core/chatterbox_subprocess.py** - Subprocess wrapper for ChatterBox TTS generation providing CUDA crash isolation and process safety

**core/__init__.py** - Core package initialization with version info and module exports for ModelManager, ImportManager, and utility classes

**core/f5tts_edit_engine_modularized.py** - Modularized F5-TTS editing engine with clean separation of audio processing and inference logic

**core/audio_compositing_modularized.py** - Modularized audio compositing utilities with original audio preservation and smooth crossfade transitions

**core/Original OLD f5tts_edit_node.py** - Legacy F5-TTS edit node implementation preserved for reference and compatibility

## Node Implementations

**nodes/tts_node.py** - Enhanced ChatterBox TTS node with modular multilingual engine, character switching, pause tags, smart language grouping, and improved text chunking

**nodes/vc_node.py** - ChatterBox Voice Conversion node with iterative refinement system and intelligent caching for progressive quality improvement

**nodes/f5tts_base_node.py** - Base class for F5-TTS nodes extending ChatterBox foundation with 24kHz audio support and F5-TTS specific requirements

**nodes/f5tts_edit_node.py** - F5-TTS Speech Editor node for targeted word/phrase replacement in existing speech while maintaining voice characteristics

**nodes/f5tts_srt_node.py** - F5-TTS SRT node with smart language grouping, modular multilingual engine integration, and optimized subtitle timing for efficient multilingual SRT processing

**nodes/audio_recorder_node.py** - Voice recording node with microphone input, silence detection, and smart audio capture functionality

**nodes/audio_analyzer_options_node.py** - Configuration provider for Audio Analyzer with advanced settings for silence detection, energy analysis, and peak detection

**nodes/f5tts_edit_options_node.py** - Advanced configuration options for F5-TTS Speech Editor including crossfade settings, caching, and post-processing controls

## SRT Subtitle System

**chatterbox_srt/timing_engine.py** - Advanced SRT timing engine handling smart audio synchronization, stretch calculations, and complex timing adjustments

**chatterbox_srt/audio_assembly.py** - Audio segment assembly engine supporting stretch-to-fit, overlap, and time synchronization modes for SRT subtitle integration

**chatterbox_srt/reporting.py** - SRT timing report generation and analysis providing detailed timing statistics and adjustment summaries

**chatterbox_srt/__init__.py** - SRT package initialization and module exports for subtitle processing functionality

**nodes/srt_tts_node.py** - ChatterBox SRT TTS node with smart language grouping, modular multilingual engine integration, and optimized subtitle processing for efficient multilingual SRT generation

**chatterbox/srt_parser.py** - SRT subtitle format parser with timestamp extraction, validation, and comprehensive error handling for subtitle processing

**core/voice_discovery.py** - Advanced voice file discovery system with dual folder support, character mapping, smart text file priority, and performance caching

## F5-TTS Implementation

**nodes/f5tts_node.py** - F5-TTS text-to-speech node with modular multilingual engine integration, character switching, smart language grouping, and comprehensive caching system

**chatterbox/f5tts/f5tts.py** - F5-TTS wrapper class bridging F5-TTS API with ChatterBox interface standards including model configurations and sample rate management

## ChatterBox Engine

**chatterbox/tts.py** - Core ChatterBox TTS engine with text preprocessing, punctuation normalization, and perth watermarking integration

**chatterbox/vc.py** - ChatterBox Voice Conversion engine with S3Gen models and perth watermarking for voice transformation capabilities

**chatterbox/audio_timing.py** - Audio timing utilities and time-stretching functionality for ChatterBox TTS timing synchronization

**chatterbox/__init__.py** - ChatterBox package initialization and core module exports

**chatterbox/f5tts/__init__.py** - F5-TTS integration package initialization with graceful import handling and error reporting

## Audio Analysis System

**nodes/audio_analyzer_node.py** - Interactive waveform visualization node for precise timing extraction with web interface and F5-TTS speech editing integration

**core/audio_analysis.py** - Core audio analysis functionality providing waveform analysis, silence detection, and timing region extraction for speech processing

## Development Tools

**scripts/bump_version_enhanced.py** - Enhanced automated version bumping script with multiline changelog support, git integration, and rollback capabilities

**scripts/bump_version.py** - Basic version bumping script for updating version numbers across project files

**scripts/version_utils.py** - Version management utilities and helper functions for automated version control and file updates

## Configuration Files

**requirements.txt** - Core dependency specifications including ChatterboxTTS, SRT support, audio processing, and optional F5-TTS dependencies with troubleshooting guidance

**pyproject.toml** - Project metadata and ComfyUI registry configuration with dependency specifications and repository information

## Web Interface

**web/audio_analyzer_core.js** - Core JavaScript functionality for interactive waveform visualization and audio analysis interface

**web/audio_analyzer_ui.js** - User interface components and controls for the audio analyzer web interface

**web/audio_analyzer_visualization.js** - Waveform rendering and visualization logic for interactive audio analysis

**web/audio_analyzer_regions.js** - Region management system for timing regions in the audio analyzer interface

**web/chatterbox_voice_capture.js** - Voice recording interface for microphone input and audio capture functionality

**web/chatterbox_srt_showcontrol.js** - SRT subtitle display controls and timing visualization interface

**web/audio_analyzer_controls.js** - Audio playback controls and user interaction handling for the audio analyzer

**web/audio_analyzer_drawing.js** - Canvas drawing utilities and waveform rendering functions for audio visualization

**web/audio_analyzer_events.js** - Event handling system for mouse and keyboard interactions in the audio analyzer

**web/audio_analyzer_interface.js** - Main interface coordination and component integration for the audio analyzer

**web/audio_analyzer_layout.js** - Layout management and UI positioning for audio analyzer components

**web/audio_analyzer_node_integration.js** - ComfyUI node integration and communication bridge for the audio analyzer

**web/audio_analyzer_widgets.js** - Custom widget implementations and UI components for audio analysis controls

## Example Workflows

**example_workflows/📺 Chatterbox SRT.json** - Complete SRT subtitle timing and TTS generation workflow example

**example_workflows/Chatterbox integration.json** - General ChatterBox TTS and Voice Conversion workflow demonstration

**example_workflows/👄 F5-TTS Speech Editor Workflow.json** - Interactive F5-TTS speech editing with Audio Wave Analyzer integration

**example_workflows/🎤 📺 F5-TTS SRT and Normal Generation.json** - F5-TTS integration with SRT subtitle processing workflow

## ChatterBox Model Architecture

**chatterbox/models/t3/** - T3 transformer model implementation with inference backend, llama configurations, and conditional encoding modules

**chatterbox/models/s3gen/** - S3Gen speech generation model with flow matching, transformer layers, and HiFiGAN vocoder components

**chatterbox/models/s3tokenizer/** - S3 tokenization system for speech token processing and semantic representation

**chatterbox/models/voice_encoder/** - Voice encoding models for speaker embedding and voice characteristic extraction

**chatterbox/models/tokenizers/** - Text tokenization utilities for preprocessing and encoding text input for TTS generation