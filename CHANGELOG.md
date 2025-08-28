# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.5.27] - 2025-08-28

### Added

- Enable safe VRAM management in Memory Safe mode

### Changed

- Add CUDA Graph toggle: High Performance (55+ tok/s) vs Memory Safe (12 tok/s)

### Fixed

- Fix Higgs Audio model unloading crashes with performance toggle
- Fix crashes when using 'Unload Models' with Higgs Audio engine
- Fix multiple generation cycles and cache state issues
- Resolve fundamental CUDA Graph vs ComfyUI memory management conflict
## [4.5.26] - 2025-08-27

### Fixed

- Fix Higgs Audio advanced RAS parameter support
- Fix parameter cache invalidation for proper audio regeneration 
- Fix configuration flow ensuring parameters reach underlying model correctly

### Added

- Add missing RAS (Repetition Avoidance Sampling) controls to Higgs Audio engine

### Changed

- Improve speech quality control with force_audio_gen and repetition settings
## [4.5.25] - 2025-08-27

### Fixed

- Fix Silent Speech Analyzer preview not reflecting post-processing parameters
- Fix cache invalidation causing unnecessary video re-analysis when adjusting merge threshold

### Changed

- Improve Silent Speech Analyzer performance with optimized caching system
## [4.5.24] - 2025-08-27

### Added

- Fix F5-TTS Speech Editor workflow compatibility with v4 unified architecture
## [4.5.23] - 2025-08-27

### Added

- Fix F5-TTS speech editing issues with E2TTS models and vocabulary handling

Resolves compatibility problems when using E2TTS models with the F5-TTS speech editor.
- Speech editing now works correctly with all supported F5-TTS model variants including:

- Fixed vocab size mismatch issues between different model types
- Integrated edit engine with unified model interface  
- Eliminated duplicate model loading in edit operations
- Improved error handling for missing vocabulary files
## [4.5.22] - 2025-08-27

### Added

- Add automatic detection for missing Linux system dependencies with helpful error messages and install instructions
## [4.5.21] - 2025-08-27

### Added

- Add new Unified 📺 TTS SRT workflow showcasing all engines and features, remove outdated individual engine workflows
## [4.5.20] - 2025-08-26

### Added

- Enhanced Higgs Audio 2 language processing for better multilingual support

Unlike ChatterBox/F5-TTS which use separate models per language, Higgs Audio 2's base model supports multiple languages natively. Improved language tag processing:
- Smart language hint conversion: `[En:Alice]` → `[English] Hello there` for better model context
- Explicit language detection: Only adds hints when user specifies language prefix (not character defaults)
- Character switching preservation: Maintains proper voice assignment while converting language tags
- Console logging improvements: Shows actual processed text sent to engine instead of raw SRT content
- Support for all language variations: English, German, Norwegian, French, Spanish, Portuguese, etc.

Result: Higgs Audio 2 receives meaningful language context for optimal multilingual performance while keeping processing clean and efficient.
## [4.5.19] - 2025-08-26

### Added

- Fix numba version compatibility by upgrading to 0.61.2+ for NumPy 2.2+ support, resolving engine loading failures
## [4.5.18] - 2025-08-26

### Changed

- Fix Higgs Audio 2 Engine Character Voices compatibility

• Higgs Audio 2 Engine now accepts 🎭 Character Voices as secondary narrator input
• Native multi-speaker modes properly use reference text from both Character Voices  
• Improved voice cloning quality for SPEAKER1 in native multi-speaker conversations
• Secondary narrator now uses dedicated reference text instead of sharing primary narrator text
## [4.5.17] - 2025-08-26

### Fixed

- Fix F5-TTS narrator segments using wrong language when selecting non-English models
## [4.5.16] - 2025-08-26

### Changed

- Remove excessive debug messages and consolidate redundant logs for cleaner user experience

Cleaned up verbose logging in:
- ChatterBox engine: Consolidated model loading messages, removed cache debug spam
- F5-TTS engine: Silenced path verbosity, fixed Vocos local detection, removed internal logs
- Higgs Audio engine: Commented out boson_multimodal INFO logs, unified completion messages
- Universal nodes: Removed engine creation debug messages
- Smart Loader: Silenced routine cache operations
- Character Voices: Consolidated multiple setup messages into single line
- Model management: Removed technical integration details

Result: Significantly reduced console noise while preserving important error and completion messages
## [4.5.15] - 2025-08-26

### Changed

- Fix PyTorch installation failing with CUDA systems and improve compatibility detection for different GPU configurations
## [4.5.14] - 2025-08-26

### Added

- Fix critical 'NoneType' object is not callable error by implementing lazy watermarker initialization.
- Watermarking now disabled by default with graceful fallback if initialization fails, eliminating model loading failures in Python 3.13 environments
## [4.5.13] - 2025-08-26

### Changed

- Remove unnecessary debug messages from startup logs - cleaned up model factory registration spam, SRT module loading messages, and redundant status messages to provide cleaner startup experience while keeping essential error reporting and progress indicators
## [4.5.12] - 2025-08-26

### Added

- Fix 'NoneType' object is not callable error when loading ChatterBox TTS models in Python 3.13 Windows environments by adding Unicode encoding error handling to import_manager
## [4.5.11] - 2025-08-25

### Added

- Implement ComfyUI memory management integration for TTS models

Add ComfyUI model management support:
- ChatterBox and F5-TTS models can now be unloaded via ComfyUI Manager
- Fix device assignment after memory cycles
- Clean up debug message spam
- Reduce repetitive Higgs Audio warnings

Note: Higgs Audio models cannot be unloaded due to CUDA graph limitations

Relates to #6
## [4.5.10] - 2025-08-25

### Added

- Fix Higgs Audio seed parameter not affecting cache key - changing seeds now properly regenerates audio instead of hitting cache

Implement comprehensive ComfyUI model management integration:
- All TTS models now integrate with ComfyUI's native model management system
- 'Clear VRAM' and 'Unload models' buttons now work with TTS models for automatic memory management
- Create unified model loading interface standardizing all engines (ChatterBox, F5-TTS, Higgs Audio, RVC, Audio Separation)
- Replace static model caches with ComfyUI-managed dynamic loading
- Add ComfyUI-compatible model wrapper enabling proper integration with ComfyUI's memory management
- Standardize factory pattern for model creation across all engines
- Enhance model fallback utilities with generic local-first loading behavior
## [4.5.9] - 2025-08-25

### Fixed

- Fix memory allocation issues when running Higgs Audio after other TTS models by reducing KV cache sizes from [2048, 8192, 16384] to [1024, 2048, 4096].
- This prevents out-of-memory errors on 24GB GPUs while maintaining full functionality for typical TTS usage.
## [4.5.8] - 2025-08-25

### Added

- Complete Python 3.13 compatibility by implementing comprehensive librosa fallback system for ChatterBox engine.
- Replaces all librosa calls with safe fallbacks that try librosa first (for quality) then torchaudio (for Python 3.13 compatibility).
- Fixes numba compilation errors in voice encoder melspec module.
## [4.5.7] - 2025-08-25

### Fixed

- Python 3.13 compatibility: Higgs Audio now works on Python 3.13 by using torchaudio instead of librosa (no quality impact, better performance)
## [4.5.6] - 2025-08-24

### Fixed

- Complete comprehensive HuggingFace cache detection system across all model downloaders - prevents duplicate downloads by checking local files, HuggingFace cache, then downloading to local (never to cache).
- Fix HuBERT model download structure that was preventing downloads from completing successfully.
## [4.5.5] - 2025-08-22

### Fixed

- Fix critical protobuf dependency conflict preventing ChatterBox, F5-TTS, and Higgs Audio engines from loading.
- Move descript-audiotools to --no-deps installation to preserve protobuf 6.x compatibility.
- All 19 nodes now load successfully on Python 3.13.6.
## [4.5.4] - 2025-08-22

### Added

- Fix Windows Unicode encoding error preventing installation on clean systems
Fix UnicodeEncodeError when Windows console cannot display emoji characters
Add graceful fallback to text-based logging for Windows CP1252 encoding
Resolve install script crash that prevented ComfyUI Manager installations
## [4.5.3] - 2025-08-22

### Added

- Add comprehensive installation system with intelligent dependency management
Add Python 3.13 full compatibility support with MediaPipe to OpenSeeFace fallback
Add intelligent install.py script with automatic conflict resolution
Add environment detection and safety warnings for system Python usage
Add NumPy version constraints to prevent Numba compatibility issues
Add automatic RVC dependencies installation support
Update requirements.txt with comprehensive dependency documentation
Update README with detailed installation instructions and Python 3.13 compatibility notes
Add dedicated Higgs Audio 2 model installation section with complete folder structure
Verify ComfyUI Manager integration with automatic install.py execution
Fix all bundled engines compatibility: ChatterBox, F5-TTS, Higgs Audio, RVC
## [4.5.2] - 2025-08-22

### Changed

- Fix Higgs Audio engine compatibility with transformers 4.46+ including attention API changes, cache handling, and generation loop updates
## [4.5.1] - 2025-08-21

### Added

- Fix critical Higgs Audio download and loading issues

- Fix sharded model file downloads (model-00001-of-00003.safetensors, etc.)
- Add missing tokenizer files (tokenizer.json, tokenizer_config.json, special_tokens_map.json) to model directory
- Fix tokenizer loading from organized local structure instead of HuggingFace cache
- Add proper completeness validation for downloaded model directories
- Implement fallback tokenizer loading with LlamaTokenizer for custom configs
- Ensure all Higgs Audio models load from organized TTS/HiggsAudio/ structure without cache duplication
## [4.5.0] - 2025-08-21

### Added

- 🎙️ Higgs Audio 2 Voice Cloning Integration

## 🌟 Major New Features

### 🎙️ Higgs Audio 2 Voice Cloning Engine
- State-of-the-art voice cloning from 30+ second reference audio samples
- Multi-speaker conversation support with seamless character switching
- Real-time voice replication with exceptional audio quality
- Universal integration - works with existing TTS Text and TTS SRT nodes
- Advanced generation controls - fine-tune temperature, top-p, top-k, and token limits
- Multi-language capabilities - English (tested), with potential support for Chinese, Korean, German, and Spanish

### 🧠 Smart Audio Processing System
- Intelligent chunk combination with automatic boundary detection
- Per-junction analysis for optimal audio segment merging
- Universal implementation across all TTS engines (ChatterBox, F5-TTS, Higgs Audio)
- Enhanced audio quality through smart text structure analysis

## 🏗️ Architecture Improvements

### ⚙️ Modular Processor Architecture
- Complete architectural consistency across all TTS engines
- Unified delegation pattern - clean separation between user interface and engine processing
- HiggsAudio SRT and TTS processors following established patterns
- Simplified maintenance with modular, testable components

### 🔄 Enhanced Processing Pipeline
- Modular SRT overlap detection across all engines
- Centralized cache management with improved invalidation
- Smart model loading and initialization optimizations
- Progress tracking with real-time feedback for long operations

## 🎯 User Experience Enhancements

### 📊 Improved Feedback and Documentation
- Real-time progress bars for Higgs Audio generation
- Enhanced tooltips with detailed parameter explanations
- Optimized parameter ranges for better generation control
- Voice reference management with flexible discovery system

### 🚀 Performance Optimizations
- Instant cache regeneration for previously processed content
- Automatic model management with organized TTS/HiggsAudio/ structure
- Memory-efficient processing with smart resource utilization
- Seamless audio combination eliminating artifacts between segments

## 🔧 Technical Improvements

### 🎵 Audio Quality Enhancements
- Advanced chunking algorithms with sentence boundary detection
- Intelligent silence insertion between audio segments
- Cache-optimized processing maintaining quality while improving speed
- Professional-grade audio output across all supported engines

### 🌐 Integration and Compatibility
- Voice cloning compatibility with existing character switching system
- Reference audio support for custom voice creation
- Multi-engine harmony - all engines work consistently with unified nodes
- Backward compatibility maintained for existing workflows
## [4.4.0] - 2025-08-16

### Added

- Add new 🗣️ Silent Speech Analyzer node for video mouth movement analysis
Features experimental viseme detection for vowels (A, E, I, O, U) and consonants (B, F, M, etc.)
Provides 3-level analysis: frame detection → syllable grouping → word prediction
Generates base SRT timing files for manual editing and use with TTS SRT nodes
Includes MediaPipe integration for production-ready mouth movement tracking
Supports visual feedback in preview videos with detection overlays
Word predictions use CMU Pronouncing Dictionary (135K+ words) as phonetic placeholders
Optimized default values for better detection sensitivity and response time
Note: OpenSeeFace provider available but experimental - MediaPipe recommended
Important: Results are experimental approximations requiring manual editing
## [4.3.7] - 2025-08-13

### Added

- Implement TTS/ structure with legacy support for cleaner model folder organization
- Add native safetensors support for Japanese and Korean Hubert models
- Update all engines (Chatterbox, F5-TTS, RVC, UVR) to use organized TTS/ paths

### Fixed

- Fix RVC Hubert model compatibility issues with automatic .pt to .safetensors conversion
- Fix misleading hubert-base-rvc model that failed but claimed to be recommended
- Ensure models download to clean TTS/ folder structure instead of cluttered root
## [4.3.6] - 2025-08-13

### Fixed

- Remove duplicate models (hubert-base, hubert-soft) and invalid wav2vec2 model
- Fix failing download URLs for HuBERT models
- Consolidate to 5 authentic HuBERT variants from verified sources
## [4.3.5] - 2025-08-13

### Added

- Add HuBERT model selection to RVC Engine with 8 different model options
- Add auto-download support and language-specific recommendations for HuBERT models
- Add comprehensive tooltips explaining each HuBERT model

### Fixed

- Fix broken URLs for HuBERT model downloads
## [4.3.4] - 2025-08-13

### Added

- Add comprehensive workflow documentation including new unified RVC+ChatterBox workflow

### Changed

- Improve RVC Engine UI by removing duplicate pitch_detection parameter
- Improve RVC Pitch Options with enhanced sliders
- Better separation of concerns between engine configuration and detailed pitch extraction options
## [4.3.3] - 2025-08-13

### Fixed

- Fix RVC model dropdown to show both downloadable and local models like F5-TTS
- Load RVC Character Model node now properly displays both model types in same dropdown
- Add 'local:' prefix for local models to distinguish from downloadable models
- Match F5-TTS dropdown behavior for consistent user experience across engines
## [4.3.2] - 2025-08-13

### Added

- Add comprehensive RVC models setup guide to README documentation
- Add detailed section explaining RVC model auto-download system and folder structure
- Add guidance on available character models and download locations

### Fixed

- Fix Load RVC Character Model node to show correct downloadable models instead of generic fallbacks
## [4.3.1] - 2025-08-13

### Fixed

- Fix RVC vocal removal node failing to load due to missing ffmpeg-python dependency
- Replace ffmpeg-python package usage with direct subprocess calls to system ffmpeg binary
- Eliminate dependency requirement by matching existing SRT timing implementation approach
## [4.3.0] - 2025-08-12

### Added

- ## Version 4.3.0 - Architecture Modernization

🏗️ Core Architecture
- Universal Streaming Infrastructure: Complete architectural overhaul creating extensible framework for future engines
- Smart Model Loading: Prevents duplicate model instances, eliminating memory exhaustion when switching between processing modes
- Thread-Safe Design: Stateless wrapper architecture eliminates shared state corruption risks

⚙️ Engine Extensibility
- Universal Adapter System: Standardized framework ready for future TTS engines
- Modular Design: Clean separation between engine logic and processing coordination

📝 Performance Notes
- Sequential Processing Recommended: Despite parallel processing infrastructure, batch_size=0 (sequential) remains optimal for performance
- Memory Efficiency: Improved model sharing between traditional and streaming modes

🔧 Technical Improvements
- Centralized Caching: Content-based hashing system for reliable cache consistency
- Code Cleanup: Removed unused experimental processor code

⚠️ Known Issues
- Console Logging: Needs further cleanup (more verbose than previous versions)
- Parallel Processing: Available but slower than sequential - use batch_size=0 for best performance

This version focuses on architectural foundation and extensibility rather than immediate performance gains. The streaming infrastructure provides a robust base for future development while maintaining compatibility with existing workflows.
## [4.2.3] - 2025-08-08

### Fixed

- Fix character alias language resolution bug preventing character voice switching, improve console logging with caching to eliminate spam, fix pause tag parsing regex, and update workflow status in README
## [4.2.2] - 2025-08-08

### Fixed

- Remove IndicF5-Hindi model support due to architecture incompatibility

IndicF5 uses custom transformer layers that are incompatible with F5-TTS DiT architecture.
Alternative: F5-Hindi-Small remains available for Hindi TTS.
Other Indian languages now fall back to base F5TTS models.
## [4.2.2] - 2025-08-08

### Removed

- **REMOVED: IndicF5-Hindi model support** due to fundamental architecture incompatibility
  - IndicF5 uses custom transformer layers (time_embed.time_mlp, text_embed.text_blocks) that are incompatible with F5-TTS DiT architecture
  - Model weights cannot be loaded into standard F5-TTS pipeline due to layer structure differences
  - Attempting integration would require extensive custom DiT implementation beyond scope of F5-TTS integration
  - **Alternative**: F5-Hindi-Small (SPRINGLab/F5-Hindi-24KHz) remains available for Hindi TTS (632MB, fully compatible)
  - **Impact**: Indian languages (Assamese, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu) now fall back to base F5TTS models

### Technical Notes

- Removed IndicF5Engine, all related imports, and model configuration entries
- Updated language mapping to use F5-Hindi-Small for Hindi, base models for other Indian languages
- Cleaned up documentation and README references

## [4.2.1] - 2025-08-08

### Added

- Add comprehensive Hindi support with F5-Hindi-Small (632MB) and IndicF5-Hindi (1.4GB) models supporting 11 Indian languages
- Add support for all 11 Indian languages (Hindi, Assamese, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu) through IndicF5 multilingual model
- Add flexible Small vs Base architecture detection that works with any future Small model variants

### Fixed

- Fix language switching logic to properly detect explicit language tags in original text
- Fix model loading bugs with enhanced architecture detection for Small models (18 layers vs Base 22 layers)
- Fix language mapping system using existing language_mapper integration instead of hardcoded mappings
## [4.2.0] - 2025-08-08

### Added

- Add advanced RVC voice conversion and vocal separation capabilities

🎵 Enhanced RVC Features:
- Advanced RVC parameter controls matching Replay terminology (pitch, pitch_detection, index_ratio, etc.)
- Comprehensive audio stem extraction for voice, echo, and noise isolation
- New merge audio node for sophisticated audio blending
- Clean separation of concerns between RVC Engine and RVC Pitch Options nodes

🔧 Audio Processing Improvements:
- Vocal removal with adjustable aggressiveness parameter
- Enhanced model architecture detection and compatibility
- Robust audio format handling and processing pipeline
- Multiple new vocal separation models with performance ratings

⚠️ Experimental Features:
- SCNet SOTA architecture implementation (10.08 SDR) - **EXPERIMENTAL with audio buzzing artifacts**
- MDX23C/RoFormer model handling - **Some models have tensor alignment issues**
- Advanced chunked processing with overlap blending for large audio files

📋 Technical Enhancements:
- Fixed tensor reshaping and return type compatibility
- Enhanced UVR5 model compatibility with warnings
- Streamlined audio processing with standardized formats
- Improved error handling and user feedback
## [4.1.0] - 2025-08-08

### Fixed

- Add comprehensive RVC (Real-time Voice Conversion) integration

🎵 RVC Voice Conversion:
- New Load RVC Character Model node for .pth model loading
- RVC Engine support in unified Voice Changer
- Iterative refinement passes with smart caching system
- Official model download sources (RMVPE, content-vec-best)
- Automatic Faiss index loading for enhanced quality
- Integration with existing TTS workflow

🔧 Technical Improvements:
- Minimal reference wrapper for compatibility
- Cache system prevents recomputation of refinement passes
- Official download sources from lj1995 and lengyue233
- Added required dependencies: faiss-cpu, onnxruntime, torchcrepe

📋 Requirements:
- Updated requirements.txt with RVC dependencies
- Compatible with existing ChatterBox and F5-TTS workflows
## [4.0.0] - 2025-08-06

### 🚨 BREAKING CHANGES

- **Complete architectural transformation to TTS Audio Suite**
  - Project evolved from ChatterBox-focused implementation to universal multi-engine TTS system
  - **⚠️ WORKFLOW COMPATIBILITY BROKEN**: Existing workflows require migration to new unified node structure
  - New project name reflects expanded scope beyond ChatterBox to support multiple TTS engines

### Added

- **🏗️ MAJOR: Unified Multi-Engine Architecture**
  - Universal TTS nodes that work with any engine (TTS Text, TTS SRT, Voice Changer)
  - Engine configuration nodes for easy switching between ChatterBox, F5-TTS, and future engines
  - Modular engine adapter system for seamless engine integration
  - Character Voices node providing NARRATOR_VOICE outputs for any TTS node

- **🔧 Advanced Engine Management**
  - Engine-specific configuration nodes (ChatterBox Engine, F5-TTS Engine)
  - Engine adapter pattern for standardized interfaces
  - Separation of engine logic from user interface

- **📁 Complete Project Restructure**
  - Engine implementations in `engines/chatterbox/` and `engines/f5tts/`
  - Unified interface nodes in `nodes/unified/`
  - Engine adapters in `engines/adapters/`
  - Comprehensive utility systems in `utils/`
  - Clear separation between engine-agnostic and engine-specific functionality

### Changed

- **🎯 Node Architecture**
  - Text and SRT processing now handled by separate, engine-agnostic unified nodes
  - Consistent interface across all engines through unified nodes
  - Enhanced node categorization for better organization in ComfyUI

- **⚡ Performance Optimizations**
  - Cache-aware model loading system prevents unnecessary model reloads
  - Smart language grouping processes SRT files by language to reduce model switching overhead
  - Optimized memory usage through intelligent model lifecycle management


### Technical Details

- **Engine Adapter Pattern**: Standardized interface allowing easy addition of new TTS engines (RVC, Tortoise, etc.)
- **Unified Caching**: Consistent cache management across all engines with engine-specific keys
- **Modular Design**: Clear separation between engine implementations, adapters, and unified interface
- **Future-Proof Architecture**: Foundation for supporting additional TTS engines beyond ChatterBox and F5-TTS

### Migration Required

- **⚠️ Complete workflow migration required** - old workflows are incompatible with v4
- This is a new project (TTS Audio Suite) separate from the original ChatterBox project
- Users must recreate workflows using new unified node structure

This release represents a fundamental architectural transformation, evolving from a ChatterBox extension to a universal multi-engine TTS platform capable of supporting any TTS engine while maintaining the same user experience.

## [3.4.3] - 2025-08-05

### Fixed

- Fix language switching not working properly and add support for flexible language aliases like [German:], [Brazil:], [USA:]
## [3.4.2] - 2025-08-05

### Fixed

- Fix character tag removal bug in single character mode
  - Root cause: TTS nodes bypassed character parser in single character mode
  - Affected: Both ChatterBox TTS and F5-TTS nodes when text contains unrecognized character tags like [Alex]
  - Result: Character tags are now properly removed before TTS generation
  - Behavior: Text '[Alex] Hello world' now correctly generates 'Hello world' instead of 'Alex Hello world'
## [3.4.1] - 2025-08-03

### Changed

- **🏗️ Major Project Restructure** - Complete reorganization for better maintainability
  - Engine-centric architecture with separated `engines/chatterbox/` and `engines/f5tts/`
  - Organized nodes into `nodes/chatterbox/`, `nodes/f5tts/`, `nodes/audio/`, `nodes/base/`
  - Replaced `core/` with organized `utils/` structure (audio, text, voice, timing, models, system)
  - Self-documenting filenames for better code navigation
  - Scalable structure for future engine additions
  - All functionality preserved with full backward compatibility

- **📋 Developer Experience**
  - Enhanced version bump script with multiline changelog support
  - Improved project structure documentation
  - Better error handling and import management
## [3.4.0] - 2025-08-02

### Added

- **Major Feature: Language Switching with Bracket Syntax**
  - Introduced `[language:character]` syntax for inline language switching
  - Support for `[fr:Alice]`, `[de:Bob]`, `[es:]` patterns in text
  - Language codes automatically map to appropriate models (F5-DE, F5-FR, German, Norwegian, etc.)
  - Character alias system integration with language defaults
  - Automatic fallback to English model for unsupported languages with warnings

- **Language Support**
  - F5-TTS: English, German (de), Spanish (es), French (fr), Italian (it), Japanese (jp), Thai (th), Portuguese (pt)
  - ChatterBox: English, German (de), Norwegian (no/nb/nn)

- **Modular Architecture**
  - Modular multilingual engine architecture with engine-specific adapters
  - Unified audio cache system with engine-specific cache key generation

### Fixed

- Fixed character parser regex bug to support empty character names like `[fr:]`
- Character audio tuple handling fixes for ChatterBox engine

### Changed

- **Performance Optimizations**
  - Smart language loading: SRT nodes now analyze subtitles before model initialization
  - Eliminated wasteful default English model loading on startup
  - Language groups processed alphabetically (de→en→fr) for predictable behavior
  - Reduced model switching overhead in multilingual SRT processing

- **Technical Improvements**
  - Enhanced logging to distinguish SRT-level vs multilingual engine operations
## [3.3.0] - 2025-08-01

### Added

- Major Feature: Multilanguage ChatterBox Support
- 🌍 NEW: Multi-language ChatterBox TTS
- Added language parameter as second input in both TTS nodes
- All example workflows updated for new parameter structure

### Fixed

- Language dropdown for English, German, Norwegian models
- Automatic HuggingFace model download and management
- Local model prioritization for faster generation
- Safetensors format support with .pt backward compatibility
- Language-aware caching system to prevent model conflicts
- ChatterBox TTS Node: Full multilanguage support
- ChatterBox SRT TTS Node: SRT timing with multilanguage models
- Character switching works seamlessly with all supported languages
- Existing workflows need manual parameter adjustment
- Robust fallback system: local → HuggingFace → English fallback
- JaneDoe84's safetensors loading fix integrated safely
- Language-aware cache keys prevent cross-language conflicts

### Changed

- 🎯 Enhanced Nodes:
- ⚠️  BREAKING CHANGE: Workflow Compatibility
- 🔧 Technical Improvements:
- Enhanced model manager with language-specific loading
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