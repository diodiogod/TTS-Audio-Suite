# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.2.9] - 2025-08-01

### Fixed

- Fix seed validation range error - clamp seed values to NumPy valid range (0 to 2^32-1)
## [3.2.8] - 2025-07-27

### Added

- Add graceful fallback when PortAudio is missing
- Add startup diagnostic for missing dependencies

### Fixed

- Fix PortAudio dependency handling for voice recording

### Changed

- Update README with system dependency requirements
## [3.2.7] - 2025-07-23

### Fixed

- Fix SRT node crash protection template not respecting user input
## [3.2.6] - 2025-07-23

### Fixed

- Fix F5-TTS progress bars and variable scope issues
## [3.2.5] - 2025-07-23

### Added

- **Dynamic Model Discovery**: Automatically detect local models in `ComfyUI/models/F5-TTS/` directory
- **Multi-Language Support**: Added support for 9 language variants (German, Spanish, French, Japanese, Italian, Thai, Brazilian Portuguese)
- **Custom Download Logic**: Implemented language-specific model repository structure handling
- **Smart Model Config Detection**: Automatic model config detection based on folder/model name
- **Enhanced Model Support**: F5-DE, F5-ES, F5-FR, F5-JP, F5-IT, F5-TH, F5-PT-BR alongside standard models

### Fixed

- **Config Mismatch Issues**: Resolved configuration problems affecting audio quality
- **Vocabulary File Handling**: Smart handling of vocabulary files for different language models
- **Cross-Platform Compatibility**: Improved international character set support

### Changed

- **Model Loading System**: Normalized model loading across base and language-specific models
- **Error Handling**: Enhanced error handling and console output for better debugging
- **Download Warnings**: Added download size and quality warnings for specific models
- **Model Name Handling**: Improved model name handling and caching mechanisms

**Technical Note**: Resolves GitHub issue #3, significantly improving F5-TTS model detection and language support capabilities.
## [3.2.4] - 2025-07-23

### Added

- Add concatenate timing mode for line-by-line processing without timing constraints
- Add concatenate option to timing_mode dropdown in both ChatterBox SRT and F5-TTS SRT nodes
- Implement TimingEngine.calculate_concatenation_adjustments() for sequential timing calculations
- Add AudioAssemblyEngine.assemble_concatenation() with optional crossfading support
- Enhanced reporting system shows original SRT → new timings with duration changes

### Fixed

- Fastest processing mode with zero audio manipulation for highest quality
- Perfect for long-form content while maintaining line-by-line SRT processing benefits
## [3.2.3] - 2025-07-22

### Added

- Add snail 🐌 and rabbit 🐰 emojis to stretch-to-fit timing reports for compress and expand modes in both ChatterBox and F5-TTS SRT nodes
## [3.2.2] - 2025-07-21

### Added

- Add detailed F5-TTS diagnostic messages to help users troubleshoot installation issues. F5-TTS import errors are now always shown during initialization, making it easier to identify missing dependencies without requiring development mode.
## [3.2.1] - 2025-07-19

### Changed

- Voice Conversion Enhancements: Iterative refinement with intelligent caching system for progressive quality improvement and instant experimentation
## [3.2.0] - 2025-07-19

### Added

- MAJOR NEW FEATURES:
- Automatic processing with no additional UI parameters
- Added full caching support to ChatterBox TTS and F5-TTS nodes
- Implemented stable audio component hashing for consistent cache keys
- This release brings substantial performance improvements and new creative possibilities for speech generation workflows\!

### Fixed

- Version 3.2.0: Pause Tags System and Universal Caching
- ⏸️ Pause Tags System - Universal pause insertion with intelligent syntax
- Smart pause syntax: [pause:1s], [pause:500ms], [pause:2]
- Seamless character integration and parser protection
- Universal support across all TTS nodes (ChatterBox, F5-TTS, SRT)
- 🚀 Universal Audio Caching - Comprehensive caching system for all nodes
- Intelligent cache keys prevent invalidation from temporary file paths
- Individual segment caching with character-aware separation
- Cache hit/miss logging for performance monitoring
- 🔧 Cache Architecture Overhaul
- Fixed cache instability issues across all SRT and TTS nodes
- Resolved cache lookup/store mismatch causing permanent cache misses
- Optimized pause tag processing to cache text segments independently
- Fixed character parser conflicts with pause tag detection
- 🛠️ Code Quality & Performance
- Streamlined codebase with comprehensive pause tag processor

### Changed

- Intelligent caching: pause changes don't invalidate text cache
- Significant speed improvements for iterative workflows
- TECHNICAL IMPROVEMENTS:
- 🎭 Character System Enhancements
- Updated text processing order for proper pause/character integration
- Enhanced character switching compatibility with pause tags
- Improved progress messaging consistency across all nodes
- Enhanced crash protection integration with pause tag system

### Removed

- Removed unnecessary enable_pause_tags UI parameters (automatic now)
## [3.1.4] - 2025-07-18

### Added

- Clean up ChatterBox crash prevention and rename padding parameter
## [3.1.3] - 2025-07-18

### Fixed

- ChatterBox character switching crashes with short text segments by implementing dynamic space padding
- Sequential generation CUDA tensor indexing errors in character switching mode
- Version bump script now prevents downgrade attempts
## [3.1.2] - 2025-07-17

### Added

- Implement user-friendly character alias system with #character_alias_map.txt file
- Add comprehensive alias documentation to CHARACTER_SWITCHING_GUIDE.md with examples
- Update README features to highlight new alias system and improve emoji clarity

### Fixed

- Support flexible alias formats: 'Alias = Character' and 'Alias[TAB]Character' with smart parsing
- Replace old JSON character_alias_map.json with more accessible text format
- Maintain backward compatibility with existing JSON files for seamless migration
## [3.1.1] - 2025-07-17

### Added

- Update character switching documentation to reflect new system

### Fixed

- Fix character discovery system to use filename-based character names instead of folder names
- Folders now used for organization only, improving usability and clarity
## [3.1.0] - 2025-07-17

### Added

#### 🎭 Character Switching System
- **NEW**: Universal `[Character]` tag support across all TTS nodes
- **NEW**: Character alias mapping with JSON configuration files
- **NEW**: Dual voice discovery (models/voices + voices_examples directories)
- **NEW**: Line-by-line character parsing for natural narrator fallback
- **NEW**: Robust fallback system for missing characters
- **ENHANCED**: Voice discovery with flat file and folder structure support
- **ENHANCED**: Character-aware caching system
- **DOCS**: Added comprehensive CHARACTER_SWITCHING_GUIDE.md

#### 🎙️ Overlapping Subtitles Support
- **NEW**: Support for overlapping subtitles in SRT nodes
- **NEW**: Automatic mode switching (smart_natural → pad_with_silence)
- **NEW**: Enhanced audio mixing for conversation patterns
- **ENHANCED**: SRT parser with overlap detection and optional validation
- **ENHANCED**: Audio assembly with overlap-aware timing

### Enhanced

#### 🔧 Technical Improvements
- **ENHANCED**: SRT parser preserves newlines for character switching
- **ENHANCED**: Character parsing with punctuation normalization
- **ENHANCED**: Voice discovery initialization on startup
- **ENHANCED**: Timing reports distinguish original vs generated overlaps
- **ENHANCED**: Mode switching info displayed in generation output

### Fixed

- **FIXED**: Line-by-line processing in SRT mode for proper narrator fallback
- **FIXED**: Character tag removal before TTS generation
- **FIXED**: "Back to me" bug in character parsing
- **FIXED**: ChatterBox SRT caching issue with character system
- **FIXED**: UnboundLocalError in timing mode processing
## [3.0.13] - 2025-07-16

### Added

- Add F5-TTS SRT workflow and fix README workflow links
- Added new F5-TTS SRT and Normal Generation workflow

### Fixed

- Fixed broken SRT workflow link in README (missing emoji prefix)
- All workflow links now point to correct files

### Changed

- Updated workflow section to properly categorize Advanced workflows
## [3.0.12] - 2025-07-16

### Added

- Added F5-TTS availability checking to initialization messages

### Fixed

- Fix F5-TTS model switching and improve initialization messages
- Fixed F5-TTS model cache not reloading when changing model names
- Removed redundant SRT success messages (only show on actual issues)
- Enhanced error handling for missing F5-TTS dependencies

### Changed

- Improved F5-TTS model loading to only check matching local folders
## [3.0.11] - 2025-07-16

### Removed

- Optimize dependencies - remove unused packages to reduce installation time and conflicts
## [3.0.10] - 2025-07-15

### Fixed

- Fix missing diffusers dependency
- Fix record button not showing due to node name mismatch in JavaScript extension
## [3.0.9] - 2025-07-15

### Added

- Add enhanced voice discovery system with dual folder support for F5-TTS nodes
## [3.0.8] - 2025-07-15

### Fixed

- Fix tensor dimension mismatch in audio concatenation for 5+ TTS chunks
## [3.0.7] - 2025-07-15

### Added

- Add comprehensive parameter migration checklist documentation

### Fixed

- Improve F5-TTS Edit parameter organization and fix RMS normalization
- Move target_rms to advanced options as post_rms_normalization for clarity
- Fix RMS normalization to preserve original segments volume

### Removed

- Remove non-functional speed parameter from edit mode
## [3.0.6] - 2025-07-15

### Fixed

- Fix SRT package naming conflict - resolves issue #2
- Rename internal 'srt' package to 'chatterbox_srt' to avoid conflict with PyPI srt library

### Changed

- Update all imports in nodes/srt_tts_node.py and nodes/f5tts_srt_node.py
## [3.0.5] - 2025-07-14

### Fixed

- Fix import detection initialization order - resolves ChatterboxTTS availability detection
## [3.0.4] - 2025-07-14

### Fixed

- Fix ChatterBox import detection to find bundled packages
## [3.0.3] - 2025-07-14

### Fixed

- Fix F5-TTS device mismatch error for MP3 audio editing
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