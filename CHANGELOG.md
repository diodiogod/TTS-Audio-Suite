# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.1] - 2025-06-11
### Fixed
- Resolved a tensor device mismatch error (`cuda:0` vs `cpu`) in the "ChatterBox SRT Voice TTS" node. This issue occurred when processing SRT files, particularly those with empty text entries, in "stretch_to_fit" and "pad_with_silence" timing modes. The fix ensures all audio tensors are consistently handled on the target processing device (`self.device`) throughout the audio generation and assembly pipeline.

## [1.1.0] - 2025-06-10
### Added
- Added the ability to handle subtitles with empty strings or silence in the SRT node.