# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.13.2] - 2025-10-30

### Added

- Text-based emotion analysis now works with current transformers versions

### Fixed

- Fix IndexTTS-2 text emotion support with modern transformers (#90)
- Resolve QwenEmotion model loading error with transformers 4.50+
- Improved compatibility with various transformers library versions
## [4.13.1] - 2025-10-29

### Fixed

- Fix ChatterBox watermarking library import issue
- Resolve ModuleNotFoundError when perth is not installed in ComfyUI Manager
- Watermarking now gracefully handles missing optional dependency
- Improves compatibility with various ComfyUI installation methods
## [4.13.0] - 2025-10-29

### Added

- New Voice Fixer node for professional audio restoration (removes noise, reverberation, clipping)
- Support 3 restoration modes for different audio degradation levels

### Changed

- Add Voice Fixer audio restoration and improve audio processing
- Optimize dependencies by replacing torchlibrosa with librosa integration
- Improve Wave Analyzer to preserve stereo audio quality throughout pipeline
- Better organized model selection dropdown for vocal removal (grouped by category)

### Fixed

- Fix audio playback not updating when input changes
## [4.12.1] - 2025-10-29

### Added

- Higgs Audio now uses consistent fixed seed across segments
- Per-segment seed control available via segment parameters

### Changed

- Better cache consistency when editing or reordering subtitles

### Fixed

- Fix Higgs Audio SRT seed behavior
## [4.12.0] - 2025-10-29

### Added

- Add per-segment parameter switching across all TTS engines
- Override generation parameters (seed, temperature, CFG, speed, etc.) per segment
- Use inline tag syntax: [Alice|seed:42|temperature:0.5] for fine-grained control
- Works with all engines: ChatterBox, F5-TTS, Higgs Audio 2, VibeVoice, IndexTTS-2
- Apply parameters to individual SRT subtitle lines
- Order-independent parameter specification
- Engine-specific parameters automatically filtered
- Backward compatible with character and language switching
- Per-segment parameters override node defaults without affecting other segments
## [4.11.26] - 2025-10-28

### Added

- Add interrupt handling for all TTS engines
- All TTS engines now support operation cancellation:
- Users can stop SRT generation at any point
- Interruption checks at segment processing boundaries
- Character switching and chunking operations are interruptible
- Added documentation for future engine implementations

### Fixed

- Graceful error handling with informative messages
## [4.11.25] - 2025-10-27

### Changed

- Improve memory management with proper CUDA/CPU device handling across all engines

### Fixed

- Fix Clear VRAM button for all TTS engines
- Fix ChatterBox Classic, ChatterBox 23-Lang, F5-TTS, Higgs Audio, VibeVoice, RVC, and Index-TTS models not unloading from GPU when Clear VRAM is clicked
- Fix device mismatch errors after unloading and reloading TTS models
- Fix VRAM spike issues when reloading RVC and VibeVoice models
## [4.11.24] - 2025-10-27

### Added

- Ensure model components properly reload to GPU after VRAM optimization

### Fixed

- Fix ChatterBox Official 23-Lang device placement causing crashes after unloading
- Fix crashes on second generation after clicking 'Clear VRAM' button
- Fix device mismatch errors when models are moved between CPU and GPU
- Fix 'Clear VRAM' button not working correctly after model reload
## [4.11.23] - 2025-10-26

### Fixed

- Fix PyTorch installation failing on Windows Python 3.13
- Fix invalid pip requirement syntax preventing installation on all platforms
- Add graceful fallback if RVC dependencies unavailable (build tools missing)
## [4.11.22] - 2025-10-25

### Added

- Complete cleanup of deprecated transformers API usage

### Fixed

- Fix remaining torch_dtype deprecation warning in Higgs Audio

### Removed

- Remove final torch_dtype deprecation warning during model initialization
## [4.11.21] - 2025-10-25

### Fixed

- Fix Higgs Audio compatibility with transformers 4.57.1+
- Fix audio generation failing with transformers 4.57.1 and newer
- Maintain full backward compatibility with older transformers versions

### Removed

- Remove deprecated torch_dtype warnings during model loading
## [4.11.20] - 2025-10-23

### Added

- Clearer diagnostic messages during startup
- Add [TTS Audio Suite] prefix to character voice discovery messages
- Makes it easier to identify which extension is reporting information
## [4.11.19] - 2025-10-23

### Added

- Faster ComfyUI startup with lazy engine loading
- Reduce blocking operations during ComfyUI initialization
- Cache voice character discovery for faster repeated loading

### Changed

- Performance improvements:
- Optimize all 7 TTS engine nodes to load instantly on startup

### Fixed

- Move dependency validation to background thread
## [4.11.18] - 2025-10-23

### Added

- Removed verbose 'available' and 'loaded' success messages during initialization
- Keep important diagnostic information for troubleshooting
- Improved code clarity by removing complex message filtering attempts

### Changed

- Cleaner startup output with reduced logging spam
- Better user experience with less console noise on startup
## [4.11.17] - 2025-10-23

### Added

- Resolves issues when using rgthree Any Switch node

### Changed

- Reduces console spam with smart model download detection
- Improves user experience with cleaner logs

### Fixed

- Fix IndexTTS-2 and other engines failing on first run after reboot
- Fix engine parameters corrupted by ComfyUI JSON serialization
## [4.11.16] - 2025-10-22

### Changed

- Improve model reuse reliability after VRAM management operations

### Fixed

- Fix ChatterBox CUDA device mismatch error on subsequent generation runs
- Resolve 'tensors on different devices' error that occurs on second+ TTS generation with cached models
- Fix tokens not being moved to GPU when models are reloaded after CPU offloading
## [4.11.15] - 2025-10-22

### Changed

- Improve stability when switching between CPU and CUDA device placement

### Fixed

- Fix Higgs Audio cache errors in Memory Safe mode
- Fix AttributeError when resetting DynamicCache in Memory Safe mode (CUDA Graphs OFF)
- Fix cache not being recreated after model unload/reload cycle
- Better error handling and logging for cache operations
## [4.11.14] - 2025-10-22

### Added

- IndexTTS-2 uses fallback text processing on Windows (fully functional)
- Add helpful note in installation summary explaining why

### Changed

- Improve Windows install experience for IndexTTS-2
- User impact: Much cleaner install process on Windows with no confusing errors

### Fixed

- WeTextProcessing no longer shows confusing error messages on Windows install.
- Windows users get clean installation without error spam
## [4.11.13] - 2025-10-22

### Fixed

- Fix VibeVoice num_hidden_layers compatibility issue
- Fix VibeVoice generation error: 'VibeVoiceConfig' object has no attribute 'num_hidden_layers'
- Add compatibility patch for transformers 4.51.3+ DynamicCache initialization
- Implements fix from wildminder/ComfyUI-VibeVoice v1.5.1 for FushionHub fork
## [4.11.12] - 2025-10-21

### Added

- User impact: IndexTTS-2 now properly downloads and caches the CampPlus speaker embedding model

### Fixed

- Fix IndexTTS-2 CampPlus model download failure
- Resolves issue #144 where IndexTTS-2 fails to load with FileNotFoundError.
- CampPlus model now automatically downloads to the correct location.
## [4.11.11] - 2025-10-20

### Added

- First generation works fine
- Now ensures conditioning data moves with model to CUDA
- User impact: ChatterBox Official 23-Lang now works correctly when running multiple generations

### Fixed

- Fix ChatterBox Official 23-Lang CUDA device mismatch on repeated generations
- Resolves issue #143 where generation fails on second run after model detach.
- After VRAM management moves model to CPU, second run fails with device mismatch
## [4.11.10] - 2025-10-19

### Added

- Add optional SSL bypass for W2V-BERT model downloads (opt-in via TTS_DISABLE_SSL_VERIFY=1)
- SSL verification is temporarily disabled only during model loading, then restored

### Fixed

- Fix IndexTTS-2 SSL certificate errors for users in restricted networks
- Resolve certificate verification failures in China and corporate networks
## [4.11.9] - 2025-10-14

### Fixed

- Fix VibeVoice not detecting custom models in extra_model_paths.yaml
- Fix support for .safetensor file extension (some models use singular form)
- Fix custom model paths not being searched correctly
- Improve compatibility with user-organized model directories
## [4.11.8] - 2025-10-14

### Fixed

- Fix F5-TTS engine failing to load in ComfyUI portable
- Fix protobuf version mismatch preventing all TTS engines from loading
- Make wandb dependency optional (only needed for training, not inference)
- Resolves 'Detected incompatible Protobuf Gencode/Runtime versions' error in ComfyUI portable installations
## [4.11.7] - 2025-10-13

### Fixed

- Fix RVC HuBERT model loading issues
- Fix 'NoneType' object is not subscriptable error during voice conversion
- Improve compatibility with chinese-hubert-base.pt models
- Better error messages when model conversion fails
## [4.11.6] - 2025-10-10

### Added

- All IndexTTS-2 dependencies now download to TTS folder as intended

### Changed

- Better organization of model files following project standards

### Fixed

- Fix IndexTTS-2 model download policy violations
- Fix w2v-bert-2.0 model downloading to wrong directory
- Fix MaskGCT semantic codec downloading to wrong directory
- Fix BigVGAN vocoder downloading to wrong directory
## [4.11.5] - 2025-10-10

### Added

- HuBERT now downloads to TTS/hubert/hubert-base-ls960/ following TTS structure
- Consistent behavior with all other engines for model organization
- Properly respects extra_model_paths.yaml for custom model directories
- All engines now follow unified download policy: local > cache check > TTS download

### Fixed

- Fix Higgs Audio downloading dependencies to .cache instead of TTS folder
- Fix Higgs Audio HuBERT model downloading to .cache against our unified policy
## [4.11.4] - 2025-10-10

### Added

- Support multiple parent folder naming conventions (HiggsAudio/, higgs_audio/, etc.)
- Allow flexible subfolder naming like other TTS engines
- Models now properly detected from configured shared directories
- Consistent behavior with ChatterBox, F5-TTS, VibeVoice, and IndexTTS-2

### Fixed

- Fix Higgs Audio models not being detected from custom directories
- Fix Higgs Audio not respecting extra_model_paths.yaml configuration
## [4.11.3] - 2025-10-10

### Added

- Support both TTS/vibevoice/ and TTS/VibeVoice/ folder naming
- Allow flexible subfolder naming like other TTS engines
- Models now properly detected from configured shared directories

### Fixed

- Fix VibeVoice models not being detected from custom directories
- Fix VibeVoice not respecting extra_model_paths.yaml configuration
## [4.11.2] - 2025-10-10

### Added

- Phoneme Text Normalizer now handles Polish ł correctly
- Affects Unicode Decomposition method in experimental text preprocessing

### Fixed

- Fix Polish text normalization adding extra ł characters
- Fix Polish text showing doubled ł characters (słynie -> slłynie)
## [4.11.1] - 2025-10-08

### Added

- Add honest documentation about experimental nature of special tokens
- Note: v2 special tokens exist in vocabulary but may produce minimal effects

### Fixed

- Fix ChatterBox v2 special token conversion
- Fix v2 emotion tags not being converted from <emotion> to [emotion] format
- Fix incorrect token names in changelog and examples
- Link to ResembleAI issue showing lack of official documentation and limited effectiveness
## [4.11.0] - 2025-10-08

### Added

- Add ChatterBox Official 23-Lang v2 with emotion and sound effects
- New v2 model adds 30+ special tokens for expressive speech:
- Emotions: <giggle>, <laughter>, <sigh>, <gasp>, <cry>
- Sounds: <cough>, <sneeze>, <sniff>, <inhale>, <exhale>
- Vocal effects: <whisper>, <singing>, <mumble>
- Features:
- Select v1 or v2 model in ChatterBox engine configuration
- Auto-download v2 model files when selected
- Use angle brackets <emotion> to avoid conflicts with character switching
- Both v1 and v2 models can coexist in same installation
- See docs/CHATTERBOX_V2_SPECIAL_TOKENS.md for complete token list and usage examples

### Fixed

- Full compatibility with existing character voices and pause tags
- Bug fixes:
- Fix v1 and v2 models properly isolated (separate caches)
- Fix v2 model auto-download with correct file paths
- Fix v1 model loading with correct vocabulary size
## [4.10.16] - 2025-10-07

### Added

- Add chunking support for VibeVoice native multi-speaker mode
- Support chunk combination method selection from unified node

### Changed

- Improve text chunking to respect sentence boundaries

### Fixed

- Fix VibeVoice local model loading and chunking
- Fix local: prefix models failing with 'Unknown model' error
## [4.10.15] - 2025-09-29

### Changed

- Better fallback support for offline tokenizer downloads

### Fixed

- Fix VibeVoice tokenizer download errors
- Fix VibeVoice failing with 'download_huggingface_model' attribute error
- Improve error handling when HuggingFace cache is disabled
## [4.10.14] - 2025-09-27

### Changed

- Improve memory management to prevent unnecessary model reloading

### Fixed

- Fix VibeVoice performance issues with auto device mode
- Fix models being moved to CPU after first generation
- Resolve slow subsequent generations when using auto device setting
- Fix VibeVoice auto device detection staying on GPU consistently
## [4.10.13] - 2025-09-26

### Changed

- Improve cache accuracy for different VibeVoice configurations

### Fixed

- Fix VibeVoice cache not invalidating when changing attention modes
- Fix cache incorrectly reusing audio when switching between attention modes
## [4.10.12] - 2025-09-26

### Added

- Add accelerate version to startup diagnostics

### Changed

- Better support for different environment configurations

### Fixed

- Fix VibeVoice compatibility with accelerate library versions
- Fix VibeVoice failing to load with older accelerate versions
- Add smart compatibility patches for transformers + accelerate mismatches
- Improve error handling for missing accelerate functions
## [4.10.11] - 2025-09-26

### Added

- Add automatic fallback if SageAttention fails during generation
- Credits: SageAttention approach adapted from @Enemyx-net's implementation

### Changed

- Improve generation speed and reliability with SageAttention enabled

### Fixed

- Fix VibeVoice SageAttention generating infinite noise and corrupting model state
- Fix SageAttention generating 2+ minute audio with weird noise instead of proper speech
- Fix model state corruption requiring ComfyUI restart when switching attention modes
- Fix VibeVoice not stopping generation at proper endpoints
## [4.10.10] - 2025-09-26

### Changed

- Improve IndexTTS-2 reliability and user experience

### Fixed

- Fix IndexTTS-2 import conflicts with external installations
- Fix import errors when external IndexTTS packages are installed
- Add clear error messages with solution steps for conflicting installations
- Resolve 'No module named indextts.gpt.model_v2' errors reported by users
## [4.10.9] - 2025-09-26

### Fixed

- Fix severe VRAM spikes on Python 3.12 with automatic compatibility patch
- Fix multi-GB VRAM spikes occurring before and after TTS generation
- Add automatic ComfyUI compatibility system (no user action required)
- Fix affects all TTS engines: ChatterBox, F5-TTS, IndexTTS-2
- Resolve ComfyUI v0.3.57+ regression in Python 3.12 environments
- Patch only applies when needed, preserving performance on other Python versions
## [4.10.8] - 2025-09-25

### Added

- Add warnings when virtual environment Python differs from detected version

### Changed

- Better guidance for Windows users on using 'python' vs 'py' commands

### Fixed

- Fix installer Python version detection on Windows
- Fix install script showing wrong Python version when using 'py' command

### Removed

- Remove emojis from installer messages to prevent encoding errors
## [4.10.7] - 2025-09-25

### Added

- Add support for [1] [2] [3] [4] numeric character tags in native multi-speaker mode
- Support mixed usage like [1] [2] [Alice] mapping to Speakers 1, 2, 3

### Changed

- Improve voice generation quality with true zero-shot hybrid mode
- Increase CFG scale maximum from 5.0 to 10.0 for better control

### Fixed

- Add VibeVoice compatibility improvements with wildminder format
- Full backward compatibility with existing character tag systems
## [4.10.6] - 2025-09-23

### Added

- No changes needed for existing setups - all voice discovery now works correctly

### Changed

- Improve path comparison logic to properly detect duplicate directories

### Fixed

- Fix voice discovery not finding voices in extra_model_paths.yaml directories
- Fix voice dropdown not showing voices from shared directories configured in extra_model_paths.yaml
- Fix character voice switching not finding voices in configured shared directories
## [4.10.5] - 2025-09-23

### Added

- No functionality lost - features still work with appropriate fallbacks

### Changed

- Add FFmpeg availability check to startup logs for better troubleshooting
- Improve vocal separation to automatically fallback to WAV when FFmpeg unavailable

### Fixed

- Add graceful FFmpeg dependency handling with fallbacks
- Better error messages when FFmpeg missing instead of cryptic subprocess errors
## [4.10.4] - 2025-09-23

### Added

- No changes required for existing users

### Fixed

- Fix VibeVoice generation compatibility with transformers 4.56+
- Fix VibeVoice generation failing with newer transformers versions
- Maintain compatibility with older transformers versions
## [4.10.3] - 2025-09-23

### Changed

- Improve Japanese text handling in F5-TTS speech editing

### Fixed

- Fix F5-JP Japanese text processing issues
- Fix F5-TTS Japanese model incorrectly applying Chinese pronunciation
- Update F5-JP to use newer model checkpoint for better compatibility
## [4.10.2] - 2025-09-23

### Changed

- Improve stability for voice reference processing

### Fixed

- Fix ChatterBox Official 23-Lang intermittent device mismatch errors
- Fix random crashes during subsequent generations
- Resolve GPU/CPU tensor compatibility issues
## [4.10.1] - 2025-09-23

### Changed

- Add comprehensive tooltips to 📝 Phoneme Text Normalizer node for better user guidance
- Update PROJECT_INDEX.md with new Text Processing Nodes section
- Improve user experience with detailed parameter explanations

### Fixed

- Add missing F5-TTS workflow example and improve node documentation
- Add missing workflow file that was referenced in README and GitHub issues
## [4.10.0] - 2025-09-23

### Added

- Add universal multilingual text preprocessing for improved pronunciation
- Add 📝 Phoneme Text Normalizer node for multilingual TTS preprocessing
- Add IPA phonemization support with automatic language detection
- Add Unicode decomposition and character mapping fallbacks
- Add F5-Polish model auto-download for high-quality Polish synthesis
- Compatible with all TTS engines (F5-TTS, ChatterBox, Higgs Audio, etc.)

### Changed

- Improve pronunciation quality across all supported languages

### Fixed

- Fix Polish, German, French special character pronunciation in F5-TTS
## [4.9.25] - 2025-09-21

### Added

- Add clear warnings for users with outdated transformers versions
- Provide exact upgrade commands when version conflicts detected

### Fixed

- Fix transformers compatibility issues
- Remove problematic compatibility patches that masked upgrade requirements
- Better error messages for DynamicCache property setter issues
## [4.9.24] - 2025-09-21

### Added

- Unify fallback handling across Windows, Mac, and Linux systems
- Users can safely uninstall incorrect 'tn' package if previously installed

### Fixed

- Fix incorrect text normalization dependency causing confusion
- Remove wrong 'tn' package dependency (time utilities instead of text processing)
- Fix IndexTTS-2 text normalization to only require WeTextProcessing or wetext
- Better error messages when text processing packages unavailable
## [4.9.23] - 2025-09-18

### Added

- Add automatic fallback to wetext package if WeTextProcessing unavailable
- Implement graceful degradation to basic text processing as last resort
- Ensure IndexTTS-2 always works even without advanced normalization

### Fixed

- Fix IndexTTS-2 graceful fallback for missing text normalization
- Fix IndexTTS-2 crashes when WeTextProcessing fails to install
- Better error messages showing which text processing is available
## [4.9.22] - 2025-09-18

### Added

- Add automatic fallback to compatible wetext package
- Ensure text normalization always works with graceful degradation

### Fixed

- Fix IndexTTS-2 text processing installation issues
- Fix WeTextProcessing installation failures on Windows Python 3.13
- Better error messaging for text processing capabilities
## [4.9.21] - 2025-09-18

### Changed

- Better reliability for users with interrupted downloads or storage issues

### Fixed

- Fix IndexTTS-2 auto-download and handle partial/corrupted model files
- Fix IndexTTS-2 models now automatically download on first use
- Fix partial downloads are detected and completed automatically
- Fix corrupted or incomplete model files trigger smart re-download
- Improve error messages show exactly which files are missing
## [4.9.20] - 2025-09-18

### Added

- Based on updates from https://github.com/index-tts/index-tts (September 8-18, 2025):

### Changed

- Update IndexTTS-2 with latest upstream improvements from official repository
- Improve emotion vector logging shows normalization and alpha scaling clearly
- Better memory management eliminates VRAM accumulation during multiple generations

### Fixed

- Fix emotion vectors now auto-normalize to prevent voice identity loss (commit 8aa8064)
- Fix memory issues with automatic audio truncation to 15 seconds (commit 0828dcb)
- Fix persistent cache buildup during batch processing (commit 64cb31a)
- Fix CUDA performance via corrected BigVGAN kernel imports (commits ee23371, e409c4a)
## [4.9.19] - 2025-09-18

### Added

- Use proper PyPI package name for Chinese/English text normalization
- Maintain tn fallback for systems where WeTextProcessing fails

### Fixed

- Fix IndexTTS-2 text normalization dependency package name
- Fix incorrect wetext package reference to correct WeTextProcessing
- Improve IndexTTS-2 text processing compatibility across platforms
## [4.9.18] - 2025-09-18

### Added

- Add cumulative click counters showing total adjustments (+0.1, +0.2, etc.)

### Changed

- Improve IndexTTS-2 emotion radar dragging and click feedback
- Better visual feedback for both reducing all emotions and increasing individual ones
- Improve button text positioning in radar chart interface

### Fixed

- Fix emotions randomly switching when dragging near center point
- Fix emotions bouncing back when dragged past zero to opposite side
## [4.9.17] - 2025-09-17

### Added

- IndexTTS-2 now respects extra_model_paths.yaml configuration
- Downloads go to configured shared directories instead of ComfyUI folder
- Model discovery works across all configured paths
- Dropdown shows local: prefix for found models, model name for download

### Fixed

- Fix IndexTTS-2 extra_model_paths support
## [4.9.16] - 2025-09-17

### Added

- Ensure consistent GPU usage when CUDA is available

### Changed

- Better performance for multi-generation workflows

### Fixed

- Fix VibeVoice auto device mode performance
- Fix auto mode defaulting to CPU for subsequent generations
## [4.9.15] - 2025-09-17

### Added

- Ensure both primary and fallback text normalizers are available

### Fixed

- Fix IndexTTS-2 missing text normalization dependency on Linux
- Add missing tn package dependency for IndexTTS-2 text processing
- Fix IndexTTS-2 failing with tn.chinese module not found error
- Improve compatibility with Linux systems where wetext fails to install
## [4.9.14] - 2025-09-17

### Added

- Provide specific guidance when audio files are too long
- Recommend optimal audio length (under 30 seconds)
- Show clear solutions when generation fails due to memory limits

### Fixed

- Fix IndexTTS-2 crashes with large audio files
- Add graceful out-of-memory error handling
## [4.9.13] - 2025-09-17

### Added

- Ensure stable module loading for IndexTTS-2 initialization

### Changed

- Improve bundled module import handling to prevent conflicts

### Fixed

- Fix IndexTTS-2 KeyError import failure with bundled modules
- Fix IndexTTS-2 failing with KeyError indextts during module resolution
- Better compatibility with different Python import scenarios
## [4.9.12] - 2025-09-17

### Added

- Prevent module resolution conflicts during IndexTTS-2 initialization

### Fixed

- Fix IndexTTS-2 import errors caused by conflicting system packages
- Fix IndexTTS-2 failing to load when system indextts package is installed
- Ensure bundled IndexTTS-2 engine uses correct internal modules
- Improve compatibility with different Python environments
## [4.9.11] - 2025-09-17

### Added

- Handle faiss circular import issues gracefully during startup
- Prevent AttributeError about Float32 in containerized environments

### Fixed

- Fix startup crash in Docker environments due to faiss circular import
- Fix dependency checker crashing with faiss in Docker containers
- Improve compatibility with Docker and OpenSUSE setups
## [4.9.10] - 2025-09-17

### Added

- Consistent with other TTS engines temp file handling

### Changed

- Better file organization and cleanup

### Fixed

- Fix IndexTTS-2 using Windows temp folder instead of ComfyUI temp
- Fix IndexTTS-2 temp files now created in ComfyUI temp folder
## [4.9.9] - 2025-09-17

### Changed

- Better handling of package version conflicts during setup

### Fixed

- Fix installation issues with numpy downgrade and requirements parsing
- Fix opencv-python causing numpy version downgrades during installation
- Fix requirements.txt parsing errors with inline comments
- Improve dependency installation reliability and compatibility
## [4.9.8] - 2025-09-17

### Changed

- Preserve RVC GPU acceleration performance
- Better numpy version management during installation

### Fixed

- Fix install script downgrading numpy on Linux systems
- Fix faiss-gpu installation downgrading numpy from 2.x to 1.26.4 on Linux
- Install faiss-gpu with --no-deps to prevent dependency conflicts
## [4.9.7] - 2025-09-17

### Added

- Truncate emotion audio to 20 seconds to prevent memory crashes

### Changed

- Improve stability with large emotion reference files
- Better memory management for Wav2Vec2-BERT processing

### Fixed

- Fix IndexTTS-2 crashes with long emotion audio files
- Fix IndexTTS-2 OOM errors when using long emotion reference audio
## [4.9.6] - 2025-09-16

### Fixed

- Fix ChatterBox Official 23-Lang language resolution and narrator input issues
- Fix ChatterBox Official 23-Lang crashes due to language alias resolution errors
- Fix ChatterBox Official 23-Lang ignoring direct audio input from opt_narrator
- Fix similar language resolution errors in phonemizer and character parser
- Improve compatibility with direct audio input across all TTS engines
## [4.9.5] - 2025-09-16

### Fixed

- Fix missing julius dependency causing Higgs Audio failures
- Add julius dependency to install script for proper audio processing support
- Fix 'No module named julius' error when using Higgs Audio 2 engine
- Resolve dependency issue with RVC demucs components requiring julius
- Users should run install.py manually in their ComfyUI environment to get the fix
- Addresses issue #82 julius dependency error
## [4.9.4] - 2025-09-16

### Changed

- Improve auto device detection to maintain CUDA performance
- Better device handling for subsequent generations after model unload

### Fixed

- Fix VibeVoice stability and performance issues
- Fix crashes during garbage collection after VibeVoice generation
- Resolve UnboundLocalError with gc variable in model cleanup
- Addresses issue #81 garbage collection and device detection problems
## [4.9.3] - 2025-09-16

### Added

- Add clear explanation of .txt vs .reference.txt priority system

### Changed

- Improve node tooltips for better user experience
- Improve IndexTTS-2 emotion_control tooltip formatting and readability
- Better documentation of engine-specific requirements

### Fixed

- Fix Character Voices node tooltips to clearly explain .txt file requirements
- Addresses issue #69 tooltip clarity feedback
## [4.9.2] - 2025-09-16

### Fixed

- Fix VibeVoice output compatibility with RVC voice conversion
- Fix VibeVoice audio not working with RVC voice converter
- Resolve BFloat16 tensor format issues causing conversion failures
- Add defensive handling for future engine compatibility
- Resolves issue #47 reported audio processing errors
## [4.9.1] - 2025-09-16

### Added

- Add automatic recovery from CPU offloading corruption
- Add new IndexTTS-2 integration workflow example

### Changed

- Improve model stability for multiple generations

### Fixed

- Fix IndexTTS-2 device errors and add integration workflow
- Fix device mismatch error when reusing IndexTTS-2 after model unload
## [4.9.0] - 2025-09-16

### Added

- Add IndexTTS-2 TTS engine with advanced emotion control
- New IndexTTS-2 engine with sophisticated emotion control system
- Unified emotion control supporting multiple input methods (audio, text, vectors)
- Dynamic text emotion analysis with QwenEmotion AI and contextual templates
- Per-character emotion control using [Character:emotion_ref] syntax
- 8-emotion vector control (Happy, Angry, Sad, Surprised, Afraid, Disgusted, Calm, Melancholic)
- Audio reference emotion support including Character Voices integration
- Emotion intensity control from neutral to maximum dramatic expression
- Complete IndexTTS-2 Emotion Control Guide: docs/IndexTTS2_Emotion_Control_Guide.md

### Changed

- Advanced caching system for improved performance
## [4.8.38] - 2025-09-10

### Changed

- Cleaner installer output without warnings

### Fixed

- Fix installer deprecation warning
- Better compatibility across Python versions

### Removed

- Remove pkg_resources deprecation warning during installation
## [4.8.37] - 2025-09-10

### Changed

- Improve Higgs Audio reliability across different environments

### Fixed

- Fix Higgs Audio compatibility with transformers 4.56.0+
- Fix generation errors with newer transformers library versions
- Resolve 'missing device argument' TypeError during audio generation
- Maintain compatibility with older transformers versions
## [4.8.36] - 2025-09-10

### Changed

- Improve stability for character switching with minimal text

### Fixed

- Fix ChatterBox Official 23-Lang crashes with very short text segments
- Fix generation errors when processing single characters like 'A'
- Remove unnecessary crash protection padding for 23-lang model
## [4.8.35] - 2025-09-10

### Changed

- Better handling of existing PyTorch installations

### Fixed

- Fix critical Windows installation failures
- Fix Windows UTF-8 encoding errors preventing installation
- Fix PyTorch version parsing crashes on systems with CUDA builds
- Fix PyTorch 2.6.0 installation failures with automatic fallback
## [4.8.34] - 2025-09-10

### Added

- Add ComfyUI extra_model_paths.yaml support for shared models
- Support shared model directories across multiple ComfyUI installations
- Enable shared storage configurations as requested by users

### Changed

- Improve voice discovery with models/TTS/voices fallback support

### Fixed

- Fix model downloads always going to default ComfyUI models folder
- Reference: GitHub issue #63
## [4.8.33] - 2025-09-10

### Changed

- Improve Higgs Audio stability with latest dependencies

### Fixed

- Fix Higgs Audio compatibility with transformers 4.46.3+
- Fix 'property has no setter' errors when using Higgs Audio
- Fix StaticCache compatibility issues with newer transformers
- Reference: GitHub issue #64
## [4.8.32] - 2025-09-09

### Added

- Add automatic detection of manual Speaker format in text

### Changed

- Improve SRT subtitle processing with consistent character mapping
- Clean up console output for better user experience
- Better fallback handling for invalid voice connections

### Fixed

- Fix VibeVoice voice handling and improve multi-speaker reliability
- Fix voice references not loading correctly in Native Multi-Speaker mode
- Add helpful error messages when wrong connection types are used
## [4.8.31] - 2025-09-09

### Added

- Faster generation with optimized parameter combination

### Changed

- Optimize VibeVoice for better quality and consistency
- Improve default settings based on community testing (CFG 3.0, 3 inference steps)
- Better quality with less background noise and music hallucinations

### Fixed

- Fix voice generation producing inconsistent results between parameter configurations
## [4.8.30] - 2025-09-09

### Fixed

- Fix VibeVoice cache and compatibility issues
- Fix cache not updating when changing inference steps or CFG values
- Fix DynamicCache errors with newer transformers library versions
- Improve VibeVoice compatibility with latest transformers updates
## [4.8.29] - 2025-09-09

### Added

- Allow ultra-low inference steps (1-3) for fast generation experiments
- Add support for CFG 3.0 + 3 steps combination as reported by users

### Changed

- Improve VibeVoice generation flexibility and default quality
- Change default to Native Multi-Speaker mode for better text processing
- Better parameter flexibility for quality/speed experimentation

### Removed

- Remove CFG scale limit to allow testing high values like 3.0
## [4.8.28] - 2025-09-09

### Added

- Add support for 4-bit model quantization to reduce VRAM usage

### Changed

- Improve voice reference troubleshooting information

### Fixed

- Fix VibeVoice character voice loading and quantization issues
- Fix character voices not being found in VibeVoice engine
- Fix misleading completion messages when using character switching
- Better compatibility with audio-only engines like VibeVoice
## [4.8.27] - 2025-09-09

### Changed

- Performance improvements by only applying workarounds when needed

### Fixed

- Improve numba compatibility handling for Python 3.13
- Fix contradictory startup messages about JIT status
- Better handling of librosa/numba compatibility issues
- Cleaner startup process with single compatibility test
- Maintain full compatibility with all Python 3.13 environments
## [4.8.26] - 2025-09-08

### Fixed

- Fix Python 3.13 compatibility issues in conda environments
- Fix audio hashing errors with Python 3.13 + conda/miniconda
- Resolve 'get_call_template' compilation errors
- Improve numba compatibility across different Python environments
## [4.8.25] - 2025-09-07

### Added

- Note: System RAM still accumulates when switching modes

### Fixed

- Fix VibeVoice VRAM clearing
- Fix VRAM clearing to prevent out of memory errors
- Fix models not switching when changing attention modes
## [4.8.24] - 2025-09-07

### Changed

- Improve model caching to properly handle configuration changes

### Fixed

- Fix VibeVoice model reloading and improve stability
- Fix VibeVoice not reloading when changing attention mode settings
- Resolve transformers compatibility issues with VibeVoice engine
- Fix numba JIT disable logic showing wrong Python version messages

### Removed

- Remove excessive debug output for cleaner console logs
## [4.8.23] - 2025-09-07

### Added

- Add ComfyUI VRAM management support for ChatterBox Official 23-Lang
- Add ComfyUI VRAM management support (Clear VRAM button now works)

### Changed

- Improve memory efficiency for ChatterBox Official 23-Lang engine

### Fixed

- Fix model caching to prevent reloading on each generation
- Fix Python 3.13 compatibility issues with audio processing
## [4.8.22] - 2025-09-07

### Added

- Add ComfyUI VRAM management support (Clear VRAM button now works)

### Changed

- Improve model caching to reuse loaded models across generations
- Better memory efficiency for VibeVoice engine

### Fixed

- Fix VibeVoice memory issues and add VRAM management
- Fix model reloading on each generation causing out-of-memory errors
## [4.8.21] - 2025-09-06

### Added

- Add proper null checks for ChatterBox factory functions

### Fixed

- Fix remaining voice conversion import issues on Python 3.13
- Fix voice conversion still failing after import success
- Improve error handling in unified model interface
- Resolve edge case causing 'NoneType' errors despite successful imports
## [4.8.20] - 2025-09-06

### Changed

- Change default vocal separation model to faster, higher-quality option

### Fixed

- Fix Python 3.13 compatibility for RVC and vocal separation
- Fix RVC voice conversion working fully in Python 3.13 environments
- Fix vocal separation compatibility with most UVR models in Python 3.13
- Fix missing RVC f0 extraction methods (harvest, dio, fcpe) now available
- Fix model download and path resolution issues for specialty models
- Improve error messages with clear guidance instead of technical dumps
## [4.8.19] - 2025-09-06

### Added

- Restore complete RVC functionality that was working in previous Python versions

### Fixed

- Fix RVC voice conversion compatibility in Python 3.13
- Fix RVC voice conversion failing in Python 3.13 environments
- Fix missing f0 extraction methods (pm, harvest, dio, fcpe) that were broken
- Fix RVC model loading and pitch extraction reliability issues
- Fix Python 3.13 numba/librosa dependency conflicts causing crashes
## [4.8.18] - 2025-09-06

### Changed

- Improve ChatterBox support for incomplete language models
- Better fallback handling for ChatterBox multilingual TTS generation

### Fixed

- Fix ChatterBox French TTS model loading issues
- Fix ChatterBox French model failing to load due to missing tokenizer
## [4.8.17] - 2025-09-06

### Fixed

- Fix voice conversion crashes on Python 3.13 systems
- Fix voice changer failing with 'NoneType' object error
- Improve Python 3.13 compatibility for voice conversion
- Better error handling when watermarking library is unavailable
- Resolve custom model loading issues (e.g. local:German-KB-H2-Merge-v2)
## [4.8.16] - 2025-09-06

### Added

- Add manual implementations as fallbacks for removed librosa utilities
- RVC should now work with any librosa version including latest releases

### Fixed

- Improve RVC compatibility with all librosa library versions
- Fix additional librosa compatibility issues with newest versions
- Handle missing pad_center, tiny, and normalize functions individually
## [4.8.15] - 2025-09-06

### Added

- Support both old and new librosa library versions automatically
- RVC voice conversion now works regardless of librosa version

### Fixed

- Fix RVC voice conversion failing with newer librosa versions
- Fix RVC failing with 'cannot import name pad_center from librosa.util' error
- Resolve issues caused by manual dependency updates bypassing our install script
## [4.8.14] - 2025-09-06

### Added

- Both TTS engines now work reliably with latest transformers versions

### Fixed

- Fix compatibility with latest transformers library versions
- Fix VibeVoice failing with 'DynamicCache' object has no attribute 'key_cache' error
- Fix Higgs Audio 2 failing with 'StaticCache' object has no attribute 'key_cache' error
- Support transformers 4.56+ while maintain backward compatibility
- Resolve generation failures after clean ComfyUI installations with newer dependencies
## [4.8.13] - 2025-09-06

### Added

- Ensure all audio processing components work correctly with latest Python

### Fixed

- Fix Python 3.13 compatibility issues with ChatterBox Official 23-Lang engine
- Fix crashes when using ChatterBox Official 23-Lang with Python 3.13
- Resolve numba compilation errors that prevented voice generation
- Improve compatibility with modern Python environments
## [4.8.12] - 2025-09-06

### Changed

- Improve parameter flow consistency across all F5-TTS components

### Fixed

- Fix F5-TTS phonemization toggle not working properly
- Fix toggle not disabling phonemization for non-English models
- Fix cache not invalidating when changing phonemization setting

### Removed

- Remove debug messages when phonemization is disabled
## [4.8.11] - 2025-09-06

### Added

- Add support for RTX 50 series GPUs (Blackwell architecture)
- Enhanced input validation prevents generation failures
- Improved audio preprocessing reduces weird sound artifacts

### Changed

- Better handling of corrupted audio inputs that could cause noise generation

### Fixed

- Improve VibeVoice reliability and GPU compatibility
## [4.8.10] - 2025-09-06

### Added

- ChatterBox Official 23-Lang now recognizes existing safetensors files
- Prevents unnecessary re-downloads when safetensors already present

### Fixed

- Fix download detection for existing safetensors files
## [4.8.9] - 2025-09-06

### Added

- Support both .pt and safetensors formats automatically
- Add fallback download if official format unavailable

### Fixed

- Fix ChatterBox Official 23-Lang safetensors compatibility
- New users get official .pt files for guaranteed compatibility
- Users with safetensors get compatibility warnings but keep working
- Fix missing model files triggering proper re-downloads
## [4.8.8] - 2025-09-06

### Added

- Add phonemization control toggle in F5-TTS Engine for user customization
- Disable phonemization for Portuguese F5-PT-BR (was making quality worse)
- Add experimental phonemization with user feedback prompts for other languages

### Changed

- Better debugging messages when phonemization is actually used

### Fixed

- Fix F5-TTS multilingual support with phonemization control
- Fix Polish and other non-English F5-TTS model text processing
- Fix phonemization working consistently across both TTS and SRT workflows
## [4.8.7] - 2025-09-06

### Fixed

- Fix ChatterBox crashes with newer transformer versions
- Fix crashes when using ChatterBox with transformers 4.36+
- Fix ChatterBox Official 23-Lang model failing to generate
- Resolve attention implementation compatibility issues
## [4.8.6] - 2025-09-05

### Added

- VibeVoice now properly respects chunk_minutes setting
- chunk_minutes=0 now correctly disables all chunking (fixes #38)

### Fixed

- Fix VibeVoice chunking controls not working correctly
- Fix disabling chunking having no effect until restart
- Fix chunk size changes requiring ComfyUI restart
## [4.8.5] - 2025-09-05

### Added

- Support model names containing special characters like colons
- Voice conversion now works properly with all custom model names

### Fixed

- Fix voice conversion failing with custom local models
- Fix voice conversion crashes when using local models with custom names
- Better error handling for local model detection
## [4.8.4] - 2025-09-05

### Added

- Parameters now properly affect audio generation instead of using stale cached results
- SRT generation now responds correctly to all parameter changes in real-time

### Fixed

- Fix cache not updating when changing ChatterBox Official 23-Lang parameters
- Fix cache invalidation not working when changing TTS parameters like temperature or exaggeration
- Resolve issue where changing advanced parameters (repetition_penalty, min_p, top_p) had no effect
## [4.8.3] - 2025-09-05

### Changed

- Improve audio processing reliability for complex SRT files
- Better handling of multilingual character switching with pause tags

### Fixed

- Fix SRT overlapping and character voice issues
- Fix SRT overlapping mode not working properly (audio was being cut off)
- Fix cache not updating when changing ChatterBox Official 23-Lang parameters
- Fix character voices reverting to narrator after pause tags like [pause:10s]
## [4.8.2] - 2025-09-05

### Added

- All timing modes now work properly with character switching and overlaps

### Changed

- Better audio processing for pause tags and mixed audio formats

### Fixed

- Fix SRT overlapping and caching issues
- Fix overlapping subtitles not playing simultaneously across all TTS engines
- Fix audio parameter changes not invalidating cache in ChatterBox Official 23-Lang
- Fix crashes when using concatenate and stretch-to-fit timing modes
## [4.8.1] - 2025-09-05

### Added

- All TTS engines now handle overlapping dialogue correctly

### Changed

- Improve timing accuracy for conversations with multiple speakers
- Better audio synchronization for complex subtitle files

### Fixed

- Fix SRT audio overlapping not working properly
- Fix overlapping subtitles being cut off instead of playing simultaneously
## [4.8.0] - 2025-09-05

### Added

- Add ChatterBox Official 23-Lang multilingual TTS engine
- New advanced text-to-speech engine with support for 23 languages
- Generate high-quality speech in Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Swahili, and Turkish
- SRT subtitle timing support for synchronized audio generation
- Voice conversion capabilities supporting all 23 languages
- Improved audio quality and consistency across all features

### Changed

- Better voice reference processing and character switching

### Fixed

- Fix audio pitch issues that made voices sound unnaturally deep
## [4.7.2] - 2025-09-04

### Added

- Voice conversion now works properly with custom local models
- Fixes #36

### Fixed

- Fix voice conversion failing on Windows with local models
- Fix Windows path errors when using local ChatterBox models
- Better cross-platform compatibility for model loading
## [4.7.1] - 2025-09-04

### Added

- Add backup model sources for reliable downloads
- Add SageAttention GPU-optimized attention for 2-4x speed boost
- SageAttention implementation based on https://github.com/wildminder/ComfyUI-VibeVoice

### Changed

- Better GPU memory optimization with mixed-precision kernels

### Fixed

- Fix VibeVoice compatibility and performance improvements
- Fix compatibility with newer transformers library versions
- Fix tokenizer downloads to avoid cache conflicts
## [4.7.0] - 2025-09-04

### Added

- Add ChatterBox support for 9 new languages
- NEW: Italian bilingual model with automatic [it] prefix for Italian text
- NEW: French model with 1400+ hours training and voice cloning
- NEW: Russian complete model with training artifacts
- NEW: Armenian and Georgian models with unique architectures
- NEW: Japanese and Korean support with shared components
- NEW: Three German variants (Standard, Best quality, Expressive with emotion tags)
## [4.6.28] - 2025-09-03

### Added

- Add universal audio format support for VideoHelper LazyAudioMap objects
- All audio-processing nodes now work with VideoHelper audio without workarounds

### Fixed

- Fix VideoHelper Suite audio compatibility
- Fix Voice Changer not detecting audio from VideoHelper Suite Load Video nodes
- Improve compatibility with Character Voices and standard AUDIO inputs
## [4.6.27] - 2025-09-03

### Changed

- Improve installation reliability across different Python versions
- Better handling of audio separation dependencies

### Fixed

- Fix numpy compatibility issues with other ComfyUI nodes
- Support numpy 1.26.4 for better compatibility with other custom nodes
- Fix VibeVoice installation failing due to missing Microsoft repository (for whataver reason the repo is now gone)
## [4.6.26] - 2025-09-01

### Added

- Hardware settings grouped at top (device, quantization, attention mode)
- Quality controls placed together for intuitive workflow
- More logical parameter flow from hardware to generation settings

### Changed

- Improve VibeVoice Engine parameter layout
- Better parameter organization for easier configuration
## [4.6.25] - 2025-09-01

### Added

- Add 4-bit quantization option for VRAM savings (helpful when VRAM is limited)
- Add inference steps control for quality vs speed adjustment

### Changed

- Enhance VibeVoice engine with advanced parameters and fixes
- Add attention mode selection for performance optimization
- Realistic performance expectations in parameter descriptions

### Fixed

- Fix compatibility issues with newer Transformers library versions
- Better device handling and error messages for quantization
## [4.6.24] - 2025-09-01

### Added

- Maintain same functionality with improved model source
- Backward compatible with existing model installations

### Changed

- Update VibeVoice 7B to official Microsoft repository
- Use official Microsoft repository for better reliability
## [4.6.23] - 2025-09-01

### Fixed

- Fix VibeVoice engine showing as available when package not installed
- Fix misleading init messages for VibeVoice engine
- Improve dependency checking consistency across all engines
- Prevent engine from appearing in node list if dependencies missing
## [4.6.22] - 2025-09-01

### Changed

- Improve clarity when using custom downloaded language models
- Better support for unidentified local language models

### Fixed

- Fix local F5-TTS model language display issue
- Fix confusing language code display for local F5-TTS models
## [4.6.21] - 2025-09-01

### Added

- Add clear warnings that RVC is for voice conversion only (not text-to-speech)

### Changed

- Improve RVC voice conversion clarity and connections

### Fixed

- Fix RVC Engine connection suggestions in ComfyUI interface
- Improve node connection compatibility between RVC and Voice Changer
## [4.6.20] - 2025-09-01

### Changed

- Improve download reliability with fallback sources

### Fixed

- Fix RVC content-vec-best model failing to auto-download
- Fix voice conversion model download issues
- Add automatic format conversion for compatibility
## [4.6.19] - 2025-08-31

### Added

- Untagged text now correctly uses the engine's selected model

### Changed

- Only tagged characters or alias-mapped characters change models

### Fixed

- Fix F5-TTS not using selected model and language issues
- Fix F5-TTS generation failing due to variable scope error
- Fix F5-TTS SRT ignoring user's model selection (e.g., Hindi model)
## [4.6.18] - 2025-08-31

### Added

- Add automatic fallback to alternative audio processing when needed

### Fixed

- Fix VibeVoice failing with numba compatibility error
- Fix audio loading error preventing VibeVoice from working on some systems
- Improve compatibility with different Python environments
## [4.6.17] - 2025-08-30

### Changed

- Improve text processing for longer audio generation

### Fixed

- Fix VibeVoice audio generation failing with longer text
- Fix chunking error preventing VibeVoice from generating audio over 30 seconds
- Resolve issue where chunking would fail even when disabled
## [4.6.16] - 2025-08-30

### Added

- More reliable model unloading when using 'Clear VRAM' button
- Enhanced stability when switching between different models

### Changed

- Improve VibeVoice memory management and unloading
- Better integration with ComfyUI's memory management system
- Consistent architecture across all TTS engines for better maintainability
## [4.6.15] - 2025-08-30

### Added

- VibeVoice models now properly unload when clicking 'Unload Models' button
- VRAM is actually freed when unloading (not just cache cleared)
- Models are properly recreated fresh after unload

### Fixed

- Fix VibeVoice unload button not freeing VRAM
- Fix crash when generating after unloading models
## [4.6.14] - 2025-08-30

### Added

- Add support for existing VibeVoice installations and HuggingFace cache
- Eliminates need to restart ComfyUI between VibeVoice generations

### Changed

- VibeVoice models now stay loaded between uses for better performance

### Fixed

- Fix VibeVoice memory errors and model loading failures
- Fix allocation errors when generating audio multiple times
- Fix VibeVoice-7B model failing to download (404 errors)
## [4.6.13] - 2025-08-30

### Added

- Add support for Kijai's MelBandRoFormer models (.safetensors)

### Changed

- Improve model-specific output handling and inversion detection

### Fixed

- Fix MelBandRoFormer models now working properly
- Fix denoise_mel_band_roformer models (27.99 SDR) compatibility
- Fix architecture mismatch errors preventing model loading
## [4.6.12] - 2025-08-30

### Added

- Add VibeVoice legacy path and cache support
- Support models from original VibeVoice-ComfyUI extension
- Reuse HuggingFace cached VibeVoice models when available
- Avoid re-downloading models for users upgrading from other extensions

### Changed

- Improve model loading efficiency with multi-source fallback
## [4.6.11] - 2025-08-30

### Added

- Long text generation now works properly with Higgs Audio engine

### Fixed

- Fix Higgs Audio TTS chunking error with long text
- Fix crash when using Higgs Audio with text that requires chunking
- Resolve sample_rate parameter error in audio combination
## [4.6.10] - 2025-08-30

### Changed

- Update project description to mention all supported TTS engines.

### Removed

- Remove virtual environment files and update project info
- Remove accidentally committed virtual environment files.
## [4.6.9] - 2025-08-30

### Added

- Add comprehensive VibeVoice documentation to README.

### Changed

- Clean up debug output and improve documentation

### Removed

- Remove verbose debug logs for cleaner console output.
## [4.6.8] - 2025-08-30

### Added

- Implement comprehensive solution for module import issues
- Comprehensive solution for 'No module named' errors across all setups

### Changed

- Smart project root detection and multi-strategy import fallbacks
- Zero performance impact on working installations

### Fixed

- Add robust import system for cross-environment compatibility
- Fix module discovery issues in conda, Windows, and other environments
## [4.6.7] - 2025-08-30

### Added

- Add manual Speaker format support and voice priority system
- Support both [Character] tags and manual 'Speaker 1: Hello' format in Native Multi-Speaker mode
- Add voice priority system: connected speaker2/3/4 inputs override character aliases with warnings
- Support up to 4 speakers with flexible voice assignment options

### Changed

- Improve user experience with clear format explanations and debug logging

### Fixed

- Fix Speaker 1 voice mapping to use main narrator correctly
## [4.6.6] - 2025-08-29

### Added

- Complete VibeVoice SRT subtitle support
- Add Custom Character Switching mode for detailed character control
- Support all timing modes (stretch, pad, natural, concatenate)

### Changed

- Add Native Multi-Speaker mode for efficient multi-character subtitles
- Improve cache management when switching between modes

### Fixed

- Fix character voice cloning in SRT processing
## [4.6.5] - 2025-08-29

### Added

- Complete VibeVoice SRT implementation with character voices
- VibeVoice SRT now uses proper character voice files
- Alice and Bob characters now sound like their voice samples

### Changed

- Improved voice reference handling for better audio quality

### Fixed

- Fix character voices not working in SRT mode
## [4.6.4] - 2025-08-29

### Added

- Ensure cache invalidation triggers properly in all Python environments

### Fixed

- Fix module import errors in conda environments
- Fix ModuleNotFoundError not being caught in import fallback logic
- Improve conda environment compatibility for ChatterBox nodes
- Complete fix for 'No module named utils.audio.audio_hash' error
## [4.6.3] - 2025-08-29

### Changed

- Improve module import reliability for conda and isolated environments
- Improve robustness across different Python environment setups
- Zero performance impact on working installations

### Fixed

- Enhance fix for 'No module named utils.audio.audio_hash' error
- Add conda environment compatibility for import issues
- Better error handling for lazy imports in ChatterBox nodes
## [4.6.2] - 2025-08-29

### Added

- ChatterBox TTS and SRT nodes now work reliably for all installation types

### Changed

- Improve robustness of lazy imports across various Python environments

### Fixed

- Fix module import errors in ChatterBox TTS and SRT nodes
- Fix 'No module named utils.audio.audio_hash' error in TTS generation
- Resolve module discovery issues for users with different ComfyUI setups
## [4.6.1] - 2025-08-29

### Added

- Add support for .safetensors format in Voice Conversion
- German and Norwegian voice conversion now works properly

### Fixed

- Fix Voice Changer support for German and Norwegian models
- Fix Voice Changer failing to load German/Norwegian models
- Improve compatibility with multilingual ChatterBox models
## [4.6.0] - 2025-08-29

### Added

- Add VibeVoice TTS engine with long-form generation
- New VibeVoice TTS engine provides advanced multi-speaker synthesis:
- Support for Microsoft VibeVoice 1.5B and 7B models
- Generate up to 90 minutes of continuous audio
- Native multi-speaker mode supports up to 4 distinct voices
- Custom character switching with [CharacterName] tag support
- Time-based chunking interface for managing long-form content
- Pause tag support with [pause:Xs] syntax
- Seamless integration with existing Character Voices workflow
- Technical features:
- Unified cache system with proper character isolation
- High-quality audio processing matching official implementation
- Full integration with existing TTS Text and Character Voices nodes
- Note: This is an initial implementation focused on core functionality.
- SRT subtitle support and additional features will be added in upcoming releases.

### Fixed

- Automatic model downloading and dependency management
## [4.5.30] - 2025-08-28

### Added

- Support case variations in model names

### Fixed

- Fix F5-TTS model compatibility issues
- Fix F5TTS_V1_Base model name validation error
- Improve backward compatibility with existing workflows
## [4.5.29] - 2025-08-28

### Added

- Clearer messaging when local vs cloud models are used

### Changed

- Improve F5-TTS local model support and reduce Google dependencies
- Better local model detection for E2-TTS variants

### Fixed

- Fix F5-TTS to use local models when available without requiring google-cloud-storage
- Reduce dependency requirements for users with manual model downloads
## [4.5.28] - 2025-08-28

### Changed

- Better audio package installation for Mac users

### Fixed

- Fix Mac M1 compatibility and installation issues
- Fix architecture mismatch errors on M1/M2 Macs
- Resolve wandb circular import affecting all TTS nodes
- Add automatic system dependency checking for macOS
- Improve ARM64 package compatibility on Apple Silicon
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
