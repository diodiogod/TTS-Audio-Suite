# ChatterBox SRT Subtitle Support Implementation

This document describes the SRT subtitle support implementation for the ChatterBox TTS ComfyUI node.

## Overview

The SRT implementation adds precise timing control to ChatterBox TTS, allowing you to generate audio that matches subtitle timing exactly. This is useful for:

- Dubbing videos with precise timing
- Creating synchronized audio for presentations
- Generating timed voiceovers for content
- Audio book production with chapter timing

## New Components

### 1. SRT Parser Module (`chatterbox/srt_parser.py`)

**Classes:**
- `SRTSubtitle`: Data class representing a single subtitle entry
- `SRTParser`: Main parser for SRT format with comprehensive error handling
- `SRTParseError`: Custom exception for parsing errors

**Key Features:**
- Parses standard SRT format (sequence, timestamps, text)
- Validates timing (no overlaps, reasonable durations)
- Handles multiple encodings (UTF-8, Latin-1, etc.)
- Removes HTML tags and normalizes text
- Provides timing statistics and analysis

```python
parser = SRTParser()
subtitles = parser.parse_srt_content(srt_text)
# or
subtitles = parser.parse_srt_file("subtitles.srt")
```

### 2. Audio Timing Utilities (`chatterbox/audio_timing.py`)

**Classes:**
- `AudioTimingUtils`: Basic timing conversion and audio manipulation utilities
- `PhaseVocoderTimeStretcher`: Advanced time-stretching using phase vocoder
- `TimedAudioAssembler`: Assembles audio segments with precise timing
- `AudioTimingError`: Custom exception for timing operations

### 3. New ComfyUI Node (`nodes.py`)

**New Node:** `🎤 ChatterBox SRT Voice TTS`

**Input Parameters:**
- `srt_content`: Multiline string containing SRT subtitle data
- `device`: Computation device (auto/cuda/cpu)
- `exaggeration`: Voice exaggeration level (0.25-2.0)
- `temperature`: Generation randomness (0.05-5.0)
- `cfg_weight`: Classifier-free guidance weight (0.0-1.0)
- `seed`: Random seed for reproducibility
- `timing_mode`: How to handle timing ("stretch_to_fit", "pad_with_silence", "natural_timing", "smart_natural")
- `reference_audio`: Optional reference audio for voice cloning
- `audio_prompt_path`: Path to reference audio file
- `max_stretch_ratio`: Maximum allowed time stretching (default: 2.0x)
- `min_stretch_ratio`: Minimum allowed time stretching (default: 0.5x)
- `fade_duration`: Crossfade duration for smooth transitions (default: 0.01s)

**Outputs:**
- `audio`: Generated audio matching SRT timing
- `generation_info`: Summary of generation process
- `timing_report`: Detailed timing analysis and adjustments
- `Adjusted_SRT`: Provides the actual, final SRT timings of the generated audio after all adjustments (shifts, stretches, shrinks, padding). This is a standard SRT multiline string.

## Timing Modes

### 1. Stretch to Fit (`stretch_to_fit`)
- **Default mode**
- Uses phase vocoder to time-stretch audio to match SRT timing exactly
- Preserves pitch while adjusting duration
- Best for precise synchronization

### 2. Pad with Silence (`pad_with_silence`)
- Generates audio at natural pace
- Adds silence gaps between subtitles to match timing
- Preserves natural speech rhythm
- Good for content where exact timing is critical

### 3. Smart Natural Timing (`smart_natural`)
- Prioritizes natural speech rhythm. It first attempts to accommodate longer natural audio by consuming existing gaps and then by shifting the start time of subsequent segments forward, utilizing any 'room' available in their natural audio duration, limited by `timing_tolerance`. If the segment still doesn't fit, it applies time stretching or shrinking within `max_stretch_ratio` and `min_stretch_ratio` limits. Finally, if necessary, it pads with silence or truncates the audio to fit the adjusted timeframe. This mode aims to preserve natural speech while adhering to timing constraints as much as possible.

## SRT Format Support

### Supported Format
```
1
00:00:01,000 --> 00:00:04,000
First subtitle text here

2
00:00:05,500 --> 00:00:08,200
Second subtitle text
can span multiple lines

3
00:00:10,000 --> 00:00:13,000
Third subtitle with <i>HTML tags</i> (automatically removed)
```

### Validation Rules
- Sequence numbers must be positive integers
- Timestamps in format `HH:MM:SS,mmm --> HH:MM:SS,mmm`
- Start time must be before end time
- No overlapping subtitles allowed
- Duration limits: 0.1s minimum, 30s maximum recommended
- Text cannot be empty after cleanup

### Error Handling
- Comprehensive parsing error messages
- Encoding detection (UTF-8, Latin-1, CP1252)
- HTML tag removal
- Whitespace normalization
- Timing validation with detailed error reporting

## Installation and Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage Examples

### Basic SRT TTS Generation
1. Load the ` ChatterBox SRT Voice TTS` node in ComfyUI
2. Paste your SRT content into the `srt_content` field
3. Choose your timing mode
4. Set voice parameters (exaggeration, temperature, etc.)
5. Optionally provide reference audio for voice cloning
6. Execute to generate timed audio

### Example SRT Content
```srt
1
00:00:01,000 --> 00:00:04,000
Welcome to our presentation about artificial intelligence.

2
00:00:05,000 --> 00:00:08,500
Today we'll explore how AI is transforming various industries.

3
00:00:10,000 --> 00:00:13,000
Let's start with the basics of machine learning.
```

### Advanced Usage
- Use `stretch_to_fit` for video dubbing where timing must be exact
- Use `pad_with_silence` for presentations with natural pacing
- Adjust `max_stretch_ratio` and `min_stretch_ratio` to control quality vs. timing accuracy
- Use `fade_duration` to smooth transitions between segments

## Technical Details

### Time Stretching Algorithm
- Uses phase vocoder (STFT-based) for high-quality time stretching
- Preserves pitch and formants while changing duration
- Fallback to simple interpolation if phase vocoder fails
- Configurable hop length and window size for quality tuning

### Sample Accuracy
- All timing calculations are sample-accurate
- Supports arbitrary sample rates
- Precise positioning within audio buffers
- No cumulative timing drift

### Memory Efficiency
- Processes audio segments individually
- Streaming assembly for large projects
- Minimal memory footprint for long content

### Segment-Level Caching
- The system now caches individual audio segments.
- Only modified segments are regenerated, significantly improving efficiency.
- Audio will *not* be regenerated if only the timings of a segment change (assuming the text and other generation parameters remain the same).
- This greatly reduces processing time for iterative adjustments.

### Error Recovery
- Graceful degradation when time stretching fails
- Comprehensive error messages for debugging
- Validation warnings for problematic timing

## Backward Compatibility

The implementation maintains full backward compatibility:
- Original `🎤 ChatterBox Voice TTS` node unchanged
- All existing workflows continue to work
- New SRT node is completely separate
- No changes to existing APIs

## Performance Considerations

### Time Stretching Performance
- Phase vocoder is CPU-intensive
- GPU acceleration not available for time stretching
- Consider using `pad_with_silence` for faster generation
- Batch processing recommended for large projects

### Memory Usage
- Memory usage scales with audio length
- Long subtitles may require more memory
- Consider splitting very long content

### Quality vs. Precision Trade-offs
- `stretch_to_fit`: Lower quality, precise timing, no overlap
- `pad_with_silence`: Best quality, precise timing, may have gaps
- `smart_natural`: Variable quality, adaptive timing, may truncate

## Troubleshooting

### Common Issues

**SRT Parsing Errors:**
- Check SRT format (sequence, timing, text structure)
- Verify timestamp format: `HH:MM:SS,mmm --> HH:MM:SS,mmm`
- Ensure no overlapping subtitles
- Check for empty text after HTML removal

**Time Stretching Issues:**
- Reduce `max_stretch_ratio` if audio quality is poor
- Use `pad_with_silence` mode for very aggressive timing
- Check that `scipy` is properly installed

**Memory Issues:**
- Split long SRT files into smaller segments

**Import Errors:**
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check that `chatterbox` modules are in the correct location
- Ensure ComfyUI can find the node files

### Debug Information
The node provides detailed timing reports including:
- Per-subtitle timing analysis
- Stretch factors and adjustments
- Quality warnings and recommendations
- Performance statistics

## Future Enhancements

Potential improvements for future versions:
- Improved audio quality controls for time stretching
- Batch processing for multiple SRT files

## API Reference

### SRTParser Methods
- `parse_srt_content(content: str) -> List[SRTSubtitle]`
- `parse_srt_file(file_path: str) -> List[SRTSubtitle]`
- `get_timing_info(subtitles: List[SRTSubtitle]) -> dict`

### AudioTimingUtils Methods
- `seconds_to_samples(seconds: float, sample_rate: int) -> int`
- `samples_to_seconds(samples: int, sample_rate: int) -> float`
- `get_audio_duration(audio: torch.Tensor, sample_rate: int) -> float`
- `create_silence(duration_seconds: float, sample_rate: int) -> torch.Tensor`

### TimedAudioAssembler Methods
- `assemble_timed_audio(audio_segments, target_timings, fade_duration) -> torch.Tensor`

## License and Credits

This SRT implementation extends the ChatterBox TTS ComfyUI node with subtitle timing support. It maintains the same license and attribution as the original project.

**Dependencies:**
- librosa: Audio processing
- scipy: Signal processing for phase vocoder
- torch/torchaudio: PyTorch audio operations
- numpy: Numerical computations