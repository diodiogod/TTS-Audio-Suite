# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-06-13
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