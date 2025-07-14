# ChatterBox SRT Subtitle Support Implementation

> **Complete guide to SRT subtitle timing support for ChatterBox TTS**

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Overview](#overview)
- [Quick Reference](#-quick-reference)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Technical Implementation](#technical-implementation)
- [API Reference](#api-reference)
- [Performance & Considerations](#performance--considerations)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

---

## 🚀 Quick Start

### Basic SRT TTS Generation
1. Load the `🎤 ChatterBox SRT Voice TTS` node in ComfyUI
2. Paste your SRT content into the `srt_content` field
3. Choose your timing mode (`stretch_to_fit` recommended for beginners)
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

---

## Overview

The SRT implementation adds precise timing control to ChatterBox TTS, allowing you to generate audio that matches subtitle timing exactly. This is useful for:

- **Dubbing videos** with precise timing
- **Creating synchronized audio** for presentations
- **Generating timed voiceovers** for content
- **Audio book production** with chapter timing

### Key Features
- ✅ **Precise timing control** with multiple modes
- ✅ **Smart caching** - only regenerates modified segments
- ✅ **Voice cloning support** with reference audio
- ✅ **Advanced time stretching** using phase vocoder
- ✅ **Comprehensive error handling** and validation
- ✅ **Multiple timing strategies** for different use cases

---

## 📖 Quick Reference

### Timing Modes Comparison

| Mode | Best For | Quality | Timing Accuracy | Speed |
|------|----------|---------|-----------------|-------|
| `stretch_to_fit` | Video dubbing | Lower | Perfect | Medium |
| `pad_with_silence` | Presentations | Best | Perfect | Fast |
| `smart_natural` | Natural speech | Variable | Adaptive | Medium |

### Key Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `exaggeration` | 0.25-2.0 | 0.5 | Voice emotion intensity |
| `temperature` | 0.05-5.0 | 0.7 | Generation randomness |
| `cfg_weight` | 0.0-1.0 | 0.5 | Classifier guidance strength |
| `max_stretch_ratio` | 1.0-3.0 | 2.0 | Maximum time stretching |
| `fade_duration` | 0.001-0.1 | 0.01 | Crossfade between segments |

---

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `librosa` - Audio processing
- `scipy` - Signal processing for phase vocoder
- `torch/torchaudio` - PyTorch audio operations
- `numpy` - Numerical computations

---

## Usage Examples

### Basic Usage in ComfyUI
1. **Add the node**: Search for `🎤 ChatterBox SRT Voice TTS` in ComfyUI
2. **Paste SRT content** into the `srt_content` field:
   ```srt
   1
   00:00:01,000 --> 00:00:04,000
   Hello world! This is my first subtitle.
   
   2
   00:00:05,000 --> 00:00:08,000
   This is the second subtitle with perfect timing.
   ```
3. **Select timing mode**: Choose `stretch_to_fit` for beginners
4. **Set voice parameters**: Adjust exaggeration (0.5), temperature (0.7), etc.
5. **Connect outputs**: Link `audio` output to PreviewAudio or SaveAudio node
6. **Execute** to generate timed audio

### Advanced Usage Examples

#### Video Dubbing (Precise Timing)
**Best for:** Exact timing synchronization with video
- **Timing Mode**: Select `stretch_to_fit`
- **Max Stretch Ratio**: Set to `1.5` (limits stretching for better quality)
- **Use Case**: When audio must match video frames exactly

#### Presentation Audio (Natural Pacing)
**Best for:** Natural-sounding narration with gaps
- **Timing Mode**: Select `pad_with_silence` 
- **Fade Duration**: Set to `0.02` for smooth transitions
- **Use Case**: Educational content, presentations, audiobooks

#### Voice Cloning with SRT
**Best for:** Personalized voice with subtitle timing
- **Timing Mode**: Select `smart_natural` (best balance)
- **Reference Audio**: Connect LoadAudio node with your voice sample
- **Use Case**: Creating personalized narration or dubbing

---

## Technical Implementation

### Core Components

#### 1. SRT Parser Module (`chatterbox/srt_parser.py`)
```python
class SRTSubtitle:
    """Data class representing a single subtitle entry"""
    sequence: int
    start_time: float
    end_time: float
    text: str

class SRTParser:
    """Main parser for SRT format with comprehensive error handling"""
    def parse_srt_content(content: str) -> List[SRTSubtitle]
    def parse_srt_file(file_path: str) -> List[SRTSubtitle]
```

**Features:**
- Parses standard SRT format (sequence, timestamps, text)
- Validates timing (no overlaps, reasonable durations)
- Handles multiple encodings (UTF-8, Latin-1, etc.)
- Removes HTML tags and normalizes text
- Provides timing statistics and analysis

#### 2. Audio Timing Utilities (`chatterbox/audio_timing.py`)
```python
class AudioTimingUtils:
    """Basic timing conversion and audio manipulation utilities"""

class PhaseVocoderTimeStretcher:
    """Advanced time-stretching using phase vocoder"""

class TimedAudioAssembler:
    """Assembles audio segments with precise timing"""
```

#### 3. ComfyUI Node Integration
**Node:** `🎤 ChatterBox SRT Voice TTS`

**Input Parameters:**
- `srt_content`: Multiline string containing SRT subtitle data
- `device`: Computation device (auto/cuda/cpu)
- `exaggeration`: Voice exaggeration level (0.25-2.0)
- `temperature`: Generation randomness (0.05-5.0)
- `cfg_weight`: Classifier-free guidance weight (0.0-1.0)
- `seed`: Random seed for reproducibility
- `timing_mode`: Timing strategy selection
- `reference_audio`: Optional reference audio for voice cloning
- `max_stretch_ratio`: Maximum allowed time stretching (default: 2.0x)
- `min_stretch_ratio`: Minimum allowed time stretching (default: 0.5x)
- `fade_duration`: Crossfade duration for smooth transitions (default: 0.01s)

**Outputs:**
- `audio`: Generated audio matching SRT timing
- `generation_info`: Summary of generation process
- `timing_report`: Detailed timing analysis and adjustments
- `adjusted_srt`: Actual final SRT timings after all adjustments

### Timing Modes Detailed

#### 1. Stretch to Fit (`stretch_to_fit`)
- **Best for:** Video dubbing requiring exact timing
- Uses phase vocoder to time-stretch audio to match SRT timing exactly
- Preserves pitch while adjusting duration
- May reduce audio quality with aggressive stretching

#### 2. Pad with Silence (`pad_with_silence`)
- **Best for:** Presentations with natural pacing
- Generates audio at natural pace
- Adds silence gaps between subtitles to match timing
- Preserves natural speech rhythm and quality

#### 3. Smart Natural Timing (`smart_natural`)
- **Best for:** Balanced natural speech with timing constraints
- Prioritizes natural speech rhythm
- Attempts to accommodate longer audio by consuming gaps
- Shifts subsequent segments forward when possible
- Falls back to time stretching or padding as needed

### Advanced Features

#### Segment-Level Caching
- Caches individual audio segments for efficiency
- Only regenerates modified segments
- Audio not regenerated if only timings change
- Significantly improves processing time for iterative adjustments

#### Sample Accuracy
- All timing calculations are sample-accurate
- Supports arbitrary sample rates
- Precise positioning within audio buffers
- No cumulative timing drift

#### Memory Efficiency
- Processes audio segments individually
- Streaming assembly for large projects
- Minimal memory footprint for long content

---

## API Reference

### SRTParser Methods
```python
def parse_srt_content(content: str) -> List[SRTSubtitle]:
    """Parse SRT content from string"""

def parse_srt_file(file_path: str) -> List[SRTSubtitle]:
    """Parse SRT content from file"""

def get_timing_info(subtitles: List[SRTSubtitle]) -> dict:
    """Get timing statistics and analysis"""
```

### AudioTimingUtils Methods
```python
def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Convert time to sample count"""

def samples_to_seconds(samples: int, sample_rate: int) -> float:
    """Convert samples to time"""

def get_audio_duration(audio: torch.Tensor, sample_rate: int) -> float:
    """Get audio duration in seconds"""

def create_silence(duration_seconds: float, sample_rate: int) -> torch.Tensor:
    """Create silence tensor"""
```

### TimedAudioAssembler Methods
```python
def assemble_timed_audio(audio_segments, target_timings, fade_duration) -> torch.Tensor:
    """Assemble audio segments with precise timing"""
```

---

## Performance & Considerations

### Performance Characteristics

#### Time Stretching Performance
- **CPU-intensive:** Phase vocoder requires significant processing
- **No GPU acceleration:** Time stretching runs on CPU only
- **Alternative:** Use `pad_with_silence` for faster generation
- **Recommendation:** Batch processing for large projects

#### Memory Usage
- Memory usage scales with audio length
- Long subtitles may require more memory
- Consider splitting very long content into chunks

#### Quality vs. Precision Trade-offs
- **`stretch_to_fit`:** Lower quality, precise timing, no overlap
- **`pad_with_silence`:** Best quality, precise timing, may have gaps  
- **`smart_natural`:** Variable quality, adaptive timing, may truncate

### SRT Format Support

#### Supported Format
```srt
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

#### Validation Rules
- ✅ Sequence numbers must be positive integers
- ✅ Timestamps in format `HH:MM:SS,mmm --> HH:MM:SS,mmm`
- ✅ Start time must be before end time
- ✅ No overlapping subtitles allowed
- ✅ Duration limits: 0.1s minimum, 30s maximum recommended
- ✅ Text cannot be empty after cleanup

#### Error Handling
- Comprehensive parsing error messages
- Encoding detection (UTF-8, Latin-1, CP1252)
- HTML tag removal and whitespace normalization
- Timing validation with detailed error reporting

---

## Troubleshooting

### Common Issues & Solutions

#### SRT Parsing Errors
**Problem:** Invalid SRT format
**Solutions:**
- Check SRT format (sequence, timing, text structure)
- Verify timestamp format: `HH:MM:SS,mmm --> HH:MM:SS,mmm`
- Ensure no overlapping subtitles
- Check for empty text after HTML removal

#### Time Stretching Issues
**Problem:** Poor audio quality with stretching
**Solutions:**
- Reduce `max_stretch_ratio` (try 1.5 instead of 2.0)
- Use `pad_with_silence` mode for very aggressive timing requirements
- Check that `scipy` is properly installed

#### Memory Issues
**Problem:** Out of memory errors
**Solutions:**
- Split long SRT files into smaller segments
- Reduce audio quality if possible
- Process in smaller batches

#### Import Errors
**Problem:** Module not found errors
**Solutions:**
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check that `chatterbox` modules are in the correct location
- Ensure ComfyUI can find the node files

### Debug Information
The node provides detailed timing reports including:
- Per-subtitle timing analysis
- Stretch factors and adjustments applied
- Quality warnings and recommendations
- Performance statistics and metrics

### Error Recovery
- Graceful degradation when time stretching fails
- Comprehensive error messages for debugging
- Validation warnings for problematic timing
- Automatic fallback to simpler methods when needed

---

## Future Enhancements

### Planned Improvements
- **Enhanced Quality Controls:** Better time stretching algorithms
- **Batch Processing:** Support for multiple SRT files
- **GPU Acceleration:** Explore GPU-based time stretching
- **Real-time Preview:** Live preview of timing adjustments
- **Advanced Caching:** More sophisticated caching strategies

### Contributing
This implementation maintains full backward compatibility with existing ChatterBox workflows. All existing nodes continue to work unchanged.

---

## License and Credits

This SRT implementation extends the ChatterBox TTS ComfyUI node with subtitle timing support. It maintains the same license and attribution as the original project.

**Key Dependencies:**
- [librosa](https://librosa.org/) - Audio processing
- [scipy](https://scipy.org/) - Signal processing for phase vocoder  
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [NumPy](https://numpy.org/) - Numerical computations

---

*For additional support and updates, please refer to the main project documentation.*