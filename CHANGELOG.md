# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.2] - 2025-07-14

### Added

- Fix Features section formatting for proper GitHub markdown rendering
- Add placeholder for F5-TTS Audio Analyzer screenshot
- Restructure SRT_IMPLEMENTATION.md documentation
- Add comprehensive table of contents and Quick Start section
- Add enhanced version bumping scripts with multiline support
- Create automated changelog generation with categorization

### Fixed

- Documentation improvements and fixes
- Fix README.md formatting and image placement
- Restore both ChatterBox TTS and Voice Capture images side by side
- Fix code block formatting and usage examples for ComfyUI users
- Polish language and maintain professional tone
- Version management automation system
- Optimize CLAUDE.md for token efficiency

### Changed

- Improve image organization and section clarity
- Improve document organization with Quick Reference tables
## [3.0.1] - 2025-07-14

### Fixed

- Added `sounddevice` to requirements.txt to prevent ModuleNotFoundError when using voice recording functionality
- Removed optional sounddevice installation section from README as it's now included by default

### Changed

- Voice recording dependencies are now installed automatically with the main requirements
- Simplified installation process by removing optional dependency steps

## [3.0.0] - 2025-07-13

### Added

- Implemented F5-TTS nodes for high-quality voice synthesis with reference audio + text cloning.
- Added Audio Wave Analyzer node for interactive waveform visualization and precise timing extraction for F5-TTS workflows. [📖 Complete Guide](docs/🌊_Audio_Wave_Analyzer-Complete_User_Guide.md)
- Added F5TTSEditNode for speech editing workflows.
- Added F5TTSSRTNode for generating TTS from SRT files using F5-TTS.

### New Nodes

- F5TTSNode
- F5TTSSRTNode
- F5TTSEditNode
- AudioAnalyzerNode
- AudioAnalyzerOptionsNode

### Contributors

- Diogod

## [2.0.2] - 2025-06-27

### Fixed

- **Transformers Compatibility**: Fixed compatibility issues with newer versions of the transformers library after ComfyUI updates
  - Resolved `LlamaModel.__init__() got an unexpected keyword argument 'attn_implementation'` error by removing direct parameter passing to LlamaModel constructor
  - Fixed `PretrainedConfig.update() got an unexpected keyword argument 'output_attentions'` error by using direct attribute setting instead of config.update()
  - Fixed `DynamicCache.update() missing 2 required positional arguments` error by simplifying cache handling to work with different transformers versions
- **Cache Management**: Updated cache handling in the T3 inference backend to be compatible with both older and newer transformers API versions
- **Configuration Safety**: Added safer configuration handling to prevent compatibility issues across different transformers versions

### Improved

- **Error Reporting**: Enhanced error messages in model loading to provide better debugging information
- **Version Compatibility**: Made the codebase more resilient to transformers library version changes

## [2.0.1] - 2025-06-17

### Changed

- **Node Renaming for Conflict Resolution**: Renamed nodes to avoid conflicts with the original ChatterBox Voice repository
- Added "(diogod)" suffix to distinguish from original implementation:
  - `ChatterBoxVoiceTTS` → `ChatterBoxVoiceTTSDiogod` (displayed as "🎤 ChatterBox Voice TTS (diogod)")
  - `ChatterBoxVoiceVC` → `ChatterBoxVoiceVCDiogod` (displayed as "🔄 ChatterBox Voice Conversion (diogod)")
  - `ChatterBoxVoiceCapture` → `ChatterBoxVoiceCaptureDiogod` (displayed as "🎙️ ChatterBox Voice Capture (diogod)")
- **Note**: "📺 ChatterBox SRT Voice TTS" remains unchanged as it was unique to this implementation

## [2.0.0] - 2025-06-14

### Changed

- **MAJOR ARCHITECTURAL REFACTORING**: Transformed the project from a monolithic structure to a clean, modular architecture
- Decomposed the massive 1,922-line [`nodes.py`](nodes.py:1) into specialized, focused modules for improved maintainability and LLM-friendly file sizes
- Created structured directory architecture:
  - [`nodes/`](nodes/__init__.py:1) - Individual node implementations ([`tts_node.py`](nodes/tts_node.py:1), [`vc_node.py`](nodes/vc_node.py:1), [`srt_tts_node.py`](nodes/srt_tts_node.py:1), [`audio_recorder_node.py`](nodes/audio_recorder_node.py:1))
  - [`core/`](core/__init__.py:1) - Core functionality modules ([`model_manager.py`](core/model_manager.py:1), [`audio_processing.py`](core/audio_processing.py:1), [`text_chunking.py`](core/text_chunking.py:1), [`import_manager.py`](core/import_manager.py:1))
  - [`srt/`](srt/__init__.py:1) - SRT-specific functionality ([`timing_engine.py`](srt/timing_engine.py:1), [`audio_assembly.py`](srt/audio_assembly.py:1), [`reporting.py`](srt/reporting.py:1))
- Extracted specialized functionality into focused modules:
  - Model management and loading logic → [`core/model_manager.py`](core/model_manager.py:1)
  - Audio processing utilities → [`core/audio_processing.py`](core/audio_processing.py:1)
  - Text chunking algorithms → [`core/text_chunking.py`](core/text_chunking.py:1)
  - Import and dependency management → [`core/import_manager.py`](core/import_manager.py:1)
  - SRT timing calculations → [`srt/timing_engine.py`](srt/timing_engine.py:1)
  - Audio segment assembly → [`srt/audio_assembly.py`](srt/audio_assembly.py:1)
  - Timing report generation → [`srt/reporting.py`](srt/reporting.py:1)
- Integrated audio recorder node functionality into the unified architecture
- Established clean separation of concerns with well-defined interfaces between modules
- Implemented proper inheritance hierarchy with [`BaseNode`](nodes/base_node.py:1) class for shared functionality

### Fixed

- Resolved original functionality issues discovered during the refactoring process
- Fixed module import paths and dependencies across the codebase
- Corrected audio processing pipeline inconsistencies
- Addressed timing calculation edge cases in SRT generation

### Maintained

- **100% backward compatibility** - All existing workflows and integrations continue to work without modification
- Preserved all original API interfaces and node signatures
- Maintained feature parity across all TTS, voice conversion, and SRT generation capabilities
- Kept all existing configuration options and parameters intact

### Improved

- **Maintainability**: Each module now has a single, well-defined responsibility
- **Readability**: Code is organized into logical, easily navigable modules
- **Testability**: Modular structure enables isolated unit testing of individual components
- **Extensibility**: Clean architecture makes it easier to add new features and nodes
- **LLM-friendly**: Smaller, focused files are more manageable for AI-assisted development
- **Development workflow**: Reduced cognitive load when working on specific functionality

### Technical Details

- Maintained centralized node registration through [`__init__.py`](nodes/__init__.py:1)
- Preserved ComfyUI integration patterns and node lifecycle management
- Kept all original error handling and progress reporting mechanisms
- Maintained thread safety and resource management practices

## [1.2.0] - 2025-06-13

### Updated

- Updated `README.md` and `requirements.txt` with proactive advice to upgrade `pip`, `setuptools`, and `wheel` before installing dependencies. This aims to prevent common installation issues with `s3tokenizer` on certain Python environments (e.g., Python 3.10, Stability Matrix setups).

### Added

- Added progress indicators to TTS generation showing current segment/chunk progress (e.g., "🎤 Generating TTS chunk 2/5..." or "📺 Generating SRT segment 3/124...") to help users estimate remaining time and track generation progress.

### Fixed

- Fixed interruption handling in ChatterBox TTS and SRT nodes by using ComfyUI's `comfy.model_management.interrupt_processing` instead of the deprecated `execution.interrupt_processing` attribute. This resolves the "ComfyUI's 'execution.interrupt_processing' attribute not found" warning and enables proper interruption between chunks/segments during generation.
- Fixed interruption behavior to properly signal to ComfyUI that generation was interrupted by raising `InterruptedError` instead of gracefully continuing. This prevents ComfyUI from caching interrupted results and ensures the node will re-run properly on subsequent executions.
- Fixed IndexError crashes in timing report and SRT string generation functions when called with empty lists, adding proper edge case handling for immediate interruption scenarios.

### Improved

- Improved smart natural timing mode to distinguish between significant and insignificant audio truncations. Truncations smaller than 50ms are now shown as "Fine-tuning audio duration" without the alarming 🚧 emoji, while only meaningful truncations (>50ms) that indicate real timing conflicts are highlighted with the warning emoji. This reduces noise in timing reports and helps users focus on actual issues.
- Reduced console verbosity by removing detailed FFmpeg processing messages (filter chains, channel processing details, etc.) during time stretching operations. The timing information is still available in the detailed timing reports, making the console output much cleaner while maintaining full functionality.
- Optimized progress messages for SRT generation to only show "📺 Generating SRT segment..." when actually generating new audio, not when loading from cache. This eliminates console spam when cached segments load instantly and provides more accurate progress indication for actual generation work.

### Fixed

- Fixed sequence numbering preservation in timing reports and Adjusted_SRT output for stretch_to_fit and pad_with_silence modes. All timing modes now correctly preserve the original SRT sequence numbers (e.g., 1, 1, 14) instead of renumbering them sequentially (1, 2, 3), maintaining consistency with smart_natural mode and ensuring more accurate SRT output.

## [1.1.1] - 2025-06-11

### Fixed

- Resolved a tensor device mismatch error (`cuda:0` vs `cpu`) in the "ChatterBox SRT Voice TTS" node. This issue occurred when processing SRT files, particularly those with empty text entries, in "stretch_to_fit" and "pad_with_silence" timing modes. The fix ensures all audio tensors are consistently handled on the target processing device (`self.device`) throughout the audio generation and assembly pipeline.

## [1.1.0] - 2025-06-10

### Added

- Added the ability to handle subtitles with empty strings or silence in the SRT node.