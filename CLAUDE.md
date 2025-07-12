# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI ChatterBox SRT Voice (v2.0.2) is a sophisticated ComfyUI custom node extension providing high-quality Text-to-Speech (TTS) and Voice Conversion capabilities. This is a refactored version of the original ChatterBox Voice with enhanced SRT timing support and F5-TTS integration.

## Core Architecture

- **ComfyUI Custom Node Extension**: Integrates with ComfyUI's node system
- **Modular Design**: Clean separation between core functionality, nodes, and ChatterBox package
- **Graceful Degradation**: Conditional feature loading based on available dependencies
- **Plugin-Based**: Self-contained with bundled models support

## Key Directories

- `core/`: Core functionality (model_manager, import_manager, audio_processing, text_chunking)
- `nodes/`: Node implementations extending `base_node.py`
- `chatterbox/`: ChatterBox TTS/VC implementation and models
- `srt/`: SRT timing engine and audio assembly
- `web/`: JavaScript UI components for voice capture and controls

## Main Entry Points

- `__init__.py`: Primary ComfyUI entry point, registers all nodes
- `nodes.py`: Node loading and management with import fallbacks
- `nodes/base_node.py`: Base class for all node implementations

## Architecture Patterns

### Import Management

- Uses `core/import_manager.py` for complex dependency handling
- Graceful fallbacks for missing optional dependencies
- Conditional feature loading based on available packages

### Model Loading

- Centralized model management via `core/model_manager.py`
- Priority-based loading: bundled → ComfyUI models → HuggingFace
- Caching and memory optimization

### Node Structure

- All nodes extend `nodes/base_node.py`
- Consistent error handling and logging
- Modular functionality with clear separation of concerns

### Text Processing

- Intelligent chunking via `core/text_chunking.py`
- Sentence boundary preservation
- Multiple combination methods (auto, concatenate, silence_padding, crossfade)

## Key Features

### TTS Capabilities

- ChatterBox TTS with unlimited text length
- F5-TTS voice synthesis with reference audio
- Multi-language support (English, German, Spanish, French, Japanese)
- Smart text chunking for long content

### SRT Integration

- SRT timing-synchronized TTS generation
- Smart timing modes with flexible shifting logic
- Segment-level caching for efficiency

### Audio Processing

- Voice capture with silence detection
- Voice conversion between speakers
- Audio analysis and format support

### Audio Analyzer Node

- Interactive waveform visualization with real-time analysis
- Connected audio input support with priority over file paths
- Multiple input scenarios: file-only, connected-only, or both (connected takes priority)
- Real-time audio playback synchronized with waveform visualization
- Automatic temporary file generation for web-accessible connected audio playback

 

## Dependencies

Core dependencies (from pyproject.toml):

- `transformers==4.46.3` (version pinned for compatibility)
- `s3tokenizer>=0.1.7`
- `resemble-perth`
- `librosa`, `scipy`, `omegaconf`, `accelerate`
- `torch`, `torchaudio`, `numpy`, `einops`
- `conformer>=0.3.2`, `phonemizer`, `g2p-en`
- `soundfile`, `resampy`, `webrtcvad`

## Development Notes

- Uses MIT License
- No formal linting/testing commands configured
- Manual testing via ComfyUI interface
- Focus on backward compatibility and graceful degradation
- Extensive error handling and user feedback through console logging
- User don't want any testing file to be created, he will test it on ComfyUI by LLM request

## Successful Approaches & Key Learnings

### Audio Analyzer Connected Audio Implementation (2025-01-11)
**SUCCESS**: Full connected audio input support with playback functionality

#### Problem Solved
Connected audio inputs (from other nodes) worked for analysis but had no audio playback, showing "No audio source available" errors.

#### Solution Architecture
1. **Python Backend** (`nodes/audio_analyzer_node.py`):
   - Detects audio source type with priority: connected audio > file audio
   - For connected audio: saves tensor as `connected_audio_{node_id}.wav` in ComfyUI input directory
   - For file audio: copies file to ComfyUI input directory if needed
   - Adds appropriate metadata to visualization data (`web_audio_filename` vs `file_path`)

2. **JavaScript Frontend** (multiple files):
   - **Priority Detection**: Checks `web_audio_filename` first, then `file_path`
   - **Method Context Fix**: Use `this.core.node.setupAudioPlayback()` not `this.setupAudioPlayback()`
   - **URL Generation**: Tries multiple ComfyUI API endpoints for audio access
   - **Deduplication**: Prevents multiple audio setups for same file

#### Critical Method Context Bug
**ISSUE**: `this.setupAudioPlayback()` in node integration was calling wrong method context
**FIX**: Change to `this.core.node.setupAudioPlayback()` to call actual node prototype method
**SYMPTOM**: Filename "audio.wav" became full path "C:\path\audio.wav" causing 404 errors

#### Working Scenarios
- **Scenario 1** (connected audio only): ✅ Real waveform + connected audio playback
- **Scenario 2** (file path only): ✅ Real waveform + file audio playback  
- **Scenario 3** (both inputs): ✅ Real waveform + connected audio playback (correct priority)

#### Key Files Modified
- `nodes/audio_analyzer_node.py`: Audio source detection and temporary file generation
- `web/audio_analyzer_node_integration.js`: Method context fix and priority logic
- `web/audio_analyzer_interface.js`: Audio setup deduplication and URL handling

#### Debugging Lessons
- Keep debug logs until ALL scenarios work, then clean up
- Method context issues can cause subtle data transformation bugs
- Test all input scenarios systematically (single inputs + combinations)
- Audio playback requires web-accessible URLs, not local file paths

## Failed Approaches (DO NOT REPEAT)

### Audio Analyzer node_id Widget Removal (2025-01-11)
**FAILED**: Auto-discovery approach using directory listing and multiple file attempts
- Tried: Directory listing fetch, timestamp-based ID guessing, multiple fallback patterns
- Problem: Generated hundreds of 404 errors, crashed console, still loaded fake test data
- Root issue: Complex ID matching between JS timestamp and Python execution ID was unreliable
- Lesson: Need simple, direct approach - either use widget or find a guaranteed ID sync method