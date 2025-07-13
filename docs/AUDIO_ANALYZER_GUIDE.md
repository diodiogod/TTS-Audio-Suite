# Audio Analyzer - Complete User Guide

The Audio Analyzer is a sophisticated waveform visualization and timing extraction tool designed for precise audio editing workflows, especially useful for F5-TTS speech editing and audio segment analysis.

![Audio Analyzer Interface Overview](images/audio_analyzer_overview.png)

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Parameters](#core-parameters)
3. [Audio Analyzer Options Node](#audio-analyzer-options-node)
4. [Interactive Interface](#interactive-interface)
5. [Analysis Methods](#analysis-methods)
6. [Region Management](#region-management)
7. [Advanced Features](#advanced-features)
8. [Outputs Reference](#outputs-reference)
9. [Tips & Workflows](#tips--workflows)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Workflow
1. **Load Audio**: Drag audio file to interface OR set `audio_file` path OR connect audio input
2. **Choose Method**: Select analysis method (`silence`, `energy`, `peaks`, or `manual`)
3. **Click Analyze**: Process audio to detect timing regions
4. **Refine Regions**: Add/delete manual regions as needed
5. **Export**: Use timing data output for F5-TTS or other applications

![Quick Start Workflow](images/quick_start_workflow.png)

### First Time Setup
- Place audio files in ComfyUI's `input` directory for easy access
- For advanced settings, connect an **Audio Analyzer Options** node
- Recommended: Start with `silence` method for speech analysis

---

## Core Parameters

### Required Parameters

#### `audio_file` (STRING)
- **Purpose**: Path to audio file for analysis
- **Format**: File path or just filename if in ComfyUI input directory
- **Supported Formats**: WAV, MP3, OGG, FLAC, M4A, AAC
- **Priority**: If both `audio_file` and audio input are provided, audio input takes priority

```
Examples:
- "speech_sample.wav"
- "C:/Audio/my_voice.mp3"
- "voices/character_01.flac"
```

#### `analysis_method` (DROPDOWN)
- **silence**: Detects pauses between speech (best for clean speech)
- **energy**: Analyzes volume changes (good for music/noisy audio)
- **peaks**: Finds sharp audio spikes (useful for percussion/effects)
- **manual**: Uses only user-defined regions

![Analysis Methods Comparison](images/analysis_methods.png)

#### `precision_level` (DROPDOWN)
Controls output timing precision:
- **seconds**: Rounded to seconds (1.23s) - rough timing
- **milliseconds**: Precise to milliseconds (1.234s) - recommended
- **samples**: Raw sample numbers (27225 smp) - exact editing

#### `visualization_points` (INT: 500-10000)
Waveform detail level:
- **500-1000**: Smooth, fast rendering
- **2000-3000**: Balanced detail (recommended)
- **5000-10000**: Very detailed, slower but precise

### Optional Parameters

#### `audio` (AUDIO INPUT)
- Connect audio from other nodes (takes priority over `audio_file`)
- Useful for processing generated or processed audio

#### `options` (OPTIONS INPUT)
- Connect **Audio Analyzer Options** node for advanced settings
- If not connected, uses sensible defaults

#### `manual_regions` (MULTILINE STRING)
Define custom timing regions:
```
Format: start,end (one per line)
Examples:
1.5,3.2
4.0,6.8
8.1,10.5
```
- **Bidirectional Sync**: Interface ↔ text widget
- **Auto-sorting**: Regions sorted chronologically
- **Combined Mode**: Works with auto-detection methods

#### `region_labels` (MULTILINE STRING)
Custom labels for manual regions:
```
Examples:
Intro
Verse 1
Chorus
Bridge
```
- Must match number of manual regions
- Custom labels preserved during sorting
- Auto-generated labels (Region 1, Region 2) get renumbered

#### `export_format` (DROPDOWN)
- **f5tts**: Simple format for F5-TTS (start,end per line)
- **json**: Full data with confidence, labels, metadata
- **csv**: Spreadsheet-compatible format

---

## Audio Analyzer Options Node

For advanced control over analysis parameters, use the **Audio Analyzer Options** node.

![Audio Analyzer Options](images/options_node.png)

### Silence Detection Options

#### `silence_threshold` (0.001-1.000, step 0.001)
- **Low values (0.001-0.01)**: Detect very quiet passages
- **Medium values (0.01-0.1)**: Standard speech pauses
- **High values (0.1-1.0)**: Only detect significant silences

#### `silence_min_duration` (0.01-5.0s, step 0.01s)
Minimum silence length to detect:
- **0.01-0.05s**: Detect brief pauses (word boundaries)
- **0.1-0.5s**: Standard sentence breaks
- **0.5s+**: Only long pauses (paragraph breaks)

#### `invert_silence_regions` (BOOLEAN)
- **False**: Returns silence regions (pauses)
- **True**: Returns speech regions (inverted detection)
- **Use Case**: F5-TTS workflows where you need speech segments

![Silence Inversion Example](images/silence_inversion.png)

### Energy Detection Options

#### `energy_sensitivity` (0.1-2.0, step 0.1)
- **Low (0.1-0.5)**: Conservative, fewer boundaries
- **Medium (0.5-1.0)**: Balanced detection
- **High (1.0-2.0)**: Aggressive, more boundaries

### Peak Detection Options

#### `peak_threshold` (0.001-1.0, step 0.001)
Minimum amplitude for peak detection

#### `peak_min_distance` (0.01-1.0s, step 0.01s)
Minimum time between detected peaks

#### `peak_region_size` (0.01-1.0s, step 0.01s)
Size of region around each detected peak

### Advanced Options

#### `group_regions_threshold` (0.000-3.000s, step 0.001s)
Merge nearby regions within threshold:
- **0.000**: No grouping (default)
- **0.1-0.5s**: Merge very close regions
- **0.5-3.0s**: Aggressive merging

![Region Grouping](images/region_grouping.png)

---

## Interactive Interface

The Audio Analyzer provides a rich interactive interface for precise audio editing.

![Interface Components](images/interface_components.png)

### Waveform Display
- **Blue waveform**: Audio amplitude over time
- **Red RMS line**: Root Mean Square energy
- **Grid lines**: Time markers for navigation
- **Colored regions**: Detected/manual timing regions

### Mouse Controls

#### Selection & Navigation
- **Left click + drag**: Select audio region
- **Right click**: Clear selection
- **Double click**: Seek to position
- **Mouse wheel**: Zoom in/out
- **Middle mouse + drag**: Pan waveform
- **CTRL + left/right drag**: Pan waveform

#### Region Interaction
- **Left click on region**: Highlight region (green, persistent)
- **Alt + click region**: Multi-select for deletion (orange, toggle)
- **Alt + click empty**: Clear all multi-selections
- **Shift + left click**: Extend selection

#### Advanced Controls
- **Drag amplitude labels (±0.8)**: Scale waveform vertically
- **Drag loop markers**: Move start/end loop points

### Keyboard Shortcuts

#### Playback
- **Space**: Play/pause
- **Arrow keys**: Move playhead (±1s)
- **Shift + Arrow keys**: Move playhead (±10s)
- **Home/End**: Go to start/end

#### Editing
- **Enter**: Add selected region
- **Delete**: Delete highlighted/selected regions
- **Shift + Delete**: Clear all regions
- **Escape**: Clear selection

#### View
- **+/-**: Zoom in/out
- **0**: Reset zoom and amplitude scale

#### Looping
- **L**: Set loop from selection
- **Shift + L**: Toggle looping on/off
- **Shift + C**: Clear loop markers

### Speed Control

![Speed Control](images/speed_control.png)

The floating speed slider provides advanced playback control:

#### Normal Range (0.0x - 2.0x)
- Drag within slider for standard speed control
- Real-time audio playback with speed adjustment

#### Extended Range (Rubberband Effect)
- **Drag beyond edges**: Access extreme speeds (-8x to +8x)
- **Acceleration**: Further you drag, faster the speed increases
- **Negative speeds**: Silent backwards playhead movement

#### Visual Feedback
- Speed display shows actual value (e.g., "4.25x", "-2.50x")
- Thin gray track line for visual reference
- White vertical bar thumb for precise control

### Control Buttons

#### Audio Management
- **📁 Upload Audio**: Browse and upload files
- **🔍 Analyze**: Process audio with current settings

#### Region Management
- **➕ Add Region**: Add current selection as region
- **🗑️ Delete Region**: Remove highlighted/selected regions
- **🗑️ Clear All**: Remove all manual regions (keeps auto-detected)

#### Loop Controls
- **🔻 Set Loop**: Set loop markers from selection
- **🔄 Loop ON/OFF**: Toggle loop playback mode
- **🚫 Clear Loop**: Remove loop markers

#### View Controls
- **🔍+ / 🔍-**: Zoom in/out
- **🔄 Reset**: Reset zoom, amplitude, and speed to defaults
- **📋 Export Timings**: Copy timing data to clipboard

---

## Analysis Methods

### Silence Detection

**Best for**: Clean speech recordings, voice-overs, podcasts

#### How it works:
1. Analyzes amplitude levels across the audio
2. Identifies regions below silence threshold
3. Filters by minimum duration requirement
4. Optionally inverts to get speech regions

#### Settings Impact:
- **Lower threshold**: Detects quieter silences
- **Shorter min duration**: Finds brief pauses
- **Invert enabled**: Returns speech instead of silence

![Silence Detection](images/silence_method.png)

#### Use Cases:
- F5-TTS preparation (with invert enabled)
- Podcast chapter detection
- Speech segment isolation
- Automatic transcription alignment

### Energy Detection

**Best for**: Music, noisy audio, variable volume content

#### How it works:
1. Calculates RMS energy over time windows
2. Detects significant energy changes
3. Creates regions around transition points

#### Settings Impact:
- **Higher sensitivity**: More word boundaries detected
- **Lower sensitivity**: Only major transitions

![Energy Detection](images/energy_method.png)

#### Use Cases:
- Music beat detection
- Noisy speech processing
- Dynamic content analysis
- Volume-based segmentation

### Peak Detection

**Best for**: Percussion, sound effects, transient-rich audio

#### How it works:
1. Identifies sharp amplitude peaks
2. Creates regions around each peak
3. Filters by threshold and minimum distance

#### Settings Impact:
- **Lower threshold**: Detects smaller peaks
- **Smaller min distance**: Allows closer peaks
- **Larger region size**: Bigger regions around peaks

![Peak Detection](images/peak_method.png)

#### Use Cases:
- Drum hit isolation
- Sound effect extraction
- Transient analysis
- Rhythmic pattern detection

### Manual Mode

**Best for**: Precise custom timing, complex audio structures

#### How it works:
- Uses only user-defined regions
- No automatic detection performed
- Full manual control over timing

#### Features:
- Text widget input for precise timing
- Interactive region creation
- Custom labeling support
- Bidirectional sync between interface and text

![Manual Mode](images/manual_method.png)

#### Use Cases:
- Precise speech editing
- Custom audio segmentation
- Music arrangement timing
- Specific interval extraction

---

## Region Management

### Creating Regions

#### Automatic Detection
1. Choose analysis method (`silence`, `energy`, `peaks`)
2. Adjust settings via Options node (optional)
3. Click **Analyze** button
4. Regions appear automatically

#### Manual Creation
1. **Method 1**: Drag to select area → press **Enter** or click **Add Region**
2. **Method 2**: Type in `manual_regions` widget:
   ```
   1.5,3.2
   4.0,6.8
   ```
3. **Method 3**: Use manual mode exclusively

#### Combined Approach
- Use any auto-detection method
- Add manual regions on top
- Both types included in output
- Manual regions persist across analyses

![Creating Regions](images/creating_regions.png)

### Region Types & Colors

#### Manual Regions (Green)
- Created by user interaction
- Editable and persistent
- Always included in output
- Numbered sequentially (Region 1, Region 2, etc.)

#### Auto-detected Regions
- **Gray**: Silence regions
- **Forest Green**: Speech regions (inverted silence)
- **Yellow**: Energy/word boundaries
- **Blue**: Peak regions
- Color indicates detection method

#### Grouped Regions
- Maintain original type color
- Show grouping information in analysis report
- Created when group threshold > 0

### Editing Regions

#### Selection States
- **Green highlight**: Single region selected (click)
- **Orange highlight**: Multiple regions selected (Alt+click)
- **Yellow selection**: Current area selection

#### Deletion
- **Single deletion**: Click region → press Delete
- **Multi-deletion**: Alt+click multiple → press Delete
- **Clear all**: Shift+Delete or Clear All button

#### Modification
- **Move regions**: Edit `manual_regions` text widget
- **Rename regions**: Edit `region_labels` text widget
- **Re-analyze**: Adjust settings → click Analyze

![Editing Regions](images/editing_regions.png)

### Region Properties

#### Timing Information
- **Start time**: Region beginning
- **End time**: Region ending  
- **Duration**: Calculated length
- **Confidence**: Detection certainty (auto-regions)

#### Metadata
- **Type**: manual, silence, speech, energy, peaks
- **Source**: Detection method used
- **Grouping info**: If region was merged

#### Labels
- **Auto-generated**: Region 1, Region 2, etc.
- **Custom**: User-defined names
- **Detection-based**: silence, speech, peak_1, etc.

---

## Advanced Features

### Region Grouping

Automatically merge nearby regions to reduce fragmentation.

#### How it works:
1. Set `group_regions_threshold` > 0.000s in Options node
2. Regions within threshold distance get merged
3. Overlapping regions are combined
4. Metadata preserved from source regions

![Region Grouping Example](images/region_grouping_detail.png)

#### Benefits:
- Reduces over-segmentation
- Creates cleaner timing data
- Maintains original region information
- Improves F5-TTS results

### Silence Inversion

Convert silence detection to speech detection for F5-TTS workflows.

#### Process:
1. Normal silence detection finds pauses
2. Inversion calculates speech regions between pauses
3. Output contains only speech segments
4. Ideal for voice cloning preparation

![Silence Inversion Process](images/silence_inversion_process.png)

### Loop Functionality

Precise playback control for detailed editing.

#### Setting Loops:
1. Select region → press **L** or click **Set Loop**
2. Drag purple loop markers to adjust
3. Use **Shift+L** to toggle looping on/off

#### Visual Indicators:
- **Purple markers**: Loop start/end points
- **Loop status**: Shown in interface
- **Automatic repeat**: When looping enabled

### Bidirectional Sync

Seamless integration between interface and text widgets.

#### Text → Interface:
- Type regions in `manual_regions` widget
- Click back to interface
- Regions automatically appear

#### Interface → Text:
- Add regions via interface
- Text widgets update automatically
- Labels and timing stay synchronized

### Caching System

Intelligent performance optimization.

#### How it works:
- Analysis results cached based on audio + settings
- Instant results for repeated analyses
- Cache invalidated when parameters change
- Manual regions included in cache key

#### Benefits:
- Faster repeated processing
- Smooth parameter experimentation
- Reduced computation overhead

---

## Outputs Reference

The Audio Analyzer provides four outputs for different use cases:

![Outputs Overview](images/outputs_overview.png)

### 1. `processed_audio` (AUDIO)
- **Purpose**: Passthrough of original audio
- **Use Case**: Continue audio processing pipeline
- **Format**: Standard ComfyUI audio tensor
- **Notes**: Always first output for easy chaining

### 2. `timing_data` (STRING)
- **Purpose**: Main timing export for external use
- **Format**: Depends on `export_format` setting
- **Precision**: Respects `precision_level` setting

#### F5TTS Format:
```
1.500,3.200
4.000,6.800
8.100,10.500
```

#### JSON Format:
```json
[
  {
    "start": 1.500,
    "end": 3.200,
    "label": "speech",
    "confidence": 1.00,
    "metadata": {"type": "speech"}
  }
]
```

#### CSV Format:
```
start,end,label,confidence,duration
1.500,3.200,speech,1.00,1.700
4.000,6.800,speech,1.00,2.800
```

### 3. `analysis_info` (STRING)
- **Purpose**: Detailed analysis report
- **Content**: Statistics, settings, visualization summary
- **Use Case**: Documentation, debugging, analysis review

#### Example Report:
```
Audio Analysis Results
Duration: 10.789s
Sample Rate: 22050 Hz
Analysis Method: silence (inverted to speech regions)
Regions Found: 2

Region Grouping:
  Grouping Threshold: 0.250s
  Original Regions: 4
  Final Regions: 2 (1 grouped, 1 individual)
  Regions Merged: 2

Timing Regions:
  1. speech: 0.000s - 6.244s (duration: 6.244s, confidence: 1.00)
  2. speech: 6.847s - 10.789s (duration: 3.942s, confidence: 1.00) [grouped from 2 regions: speech, speech]

Visualization Summary:
  Waveform Points: 2000
  Duration: 10.789s
  Sample Rate: 22050 Hz
  RMS Data Points: 202
```

### 4. `segmented_audio` (AUDIO)
- **Purpose**: Audio containing only detected regions
- **Process**: Extracts and concatenates region audio
- **Use Case**: F5-TTS training, isolated speech extraction
- **Format**: Standard ComfyUI audio tensor

#### How it works:
1. Sort regions by start time
2. Extract audio for each region
3. Concatenate segments sequentially
4. Return as single audio tensor

![Segmented Audio Process](images/segmented_audio.png)

---

## Tips & Workflows

### F5-TTS Preparation Workflow

1. **Load clean speech audio**
2. **Connect Audio Analyzer Options** node:
   - Method: `silence`
   - Enable `invert_silence_regions`
   - Set appropriate `silence_threshold`
3. **Analyze** to get speech regions
4. **Fine-tune** by adding manual regions if needed
5. **Use outputs**:
   - `timing_data` → F5-TTS timing input
   - `segmented_audio` → F5-TTS audio input

![F5-TTS Workflow](images/f5tts_workflow.png)

### Music Analysis Workflow

1. **Load music track**
2. **Use `energy` method** for beat detection
3. **Adjust `energy_sensitivity`** to match dynamics
4. **Add manual regions** for specific sections
5. **Group regions** to merge close beats
6. **Export timing data** for synchronization

### Podcast Chapter Detection

1. **Load podcast audio**
2. **Use `silence` method** with:
   - Higher `silence_threshold` for speech gaps
   - Longer `silence_min_duration` for chapter breaks
3. **Manual refinement** for precise chapter boundaries
4. **Custom labels** for chapter names
5. **Export for media players**

### Quality Control Tips

#### Audio Preparation
- **Normalize volume** before analysis
- **Remove background noise** if possible
- **Use consistent recording conditions**
- **Check for clipping or distortion**

#### Parameter Tuning
- **Start with defaults** and adjust incrementally
- **Test with short audio samples** first
- **Use visual feedback** from waveform display
- **Compare different methods** for same audio

#### Verification
- **Listen to detected regions** using loop functionality
- **Check timing precision** with playhead
- **Verify region boundaries** at detailed zoom levels
- **Test output compatibility** with target applications

### Performance Optimization

#### For Large Files
- **Reduce `visualization_points`** for faster rendering
- **Use caching** - avoid changing parameters unnecessarily
- **Process in segments** if memory limited
- **Consider downsampling** for initial analysis

#### For Real-time Use
- **Pre-tune parameters** on representative samples
- **Use manual mode** for known timing patterns
- **Minimize UI interactions** during processing
- **Batch process** similar audio files

---

## Troubleshooting

### Common Issues

#### "No audio data received"
**Causes:**
- Audio file not found or corrupted
- Unsupported audio format
- Empty audio input connection
- Path resolution issues

**Solutions:**
- Verify file exists and path is correct
- Use supported formats (WAV, MP3, OGG, FLAC)
- Check audio input connections
- Try absolute file paths

#### "Fake test data" warning
**Causes:**
- Audio loading failed
- No audio source provided
- Network/file access issues

**Solutions:**
- Check audio file accessibility
- Verify ComfyUI input directory setup
- Ensure audio format compatibility
- Re-analyze with proper audio source

#### Regions not appearing
**Causes:**
- Detection thresholds too strict
- Audio too quiet/loud for method
- Incorrect analysis settings
- Empty manual regions

**Solutions:**
- Adjust detection thresholds in Options node
- Try different analysis methods
- Check audio amplitude levels
- Verify manual region format

#### Performance issues
**Causes:**
- Large audio files
- High visualization point count
- Complex region sets
- Frequent re-analysis

**Solutions:**
- Reduce visualization points
- Use appropriate zoom levels
- Optimize detection parameters
- Leverage caching system

### Interface Issues

#### Speed control not working
**Causes:**
- Audio not properly loaded
- Browser audio restrictions
- Conflicting audio processes

**Solutions:**
- Reload audio and re-analyze
- Check browser audio permissions
- Refresh ComfyUI interface

#### Visual artifacts or duplicates
**Causes:**
- Region synchronization issues
- Mixed manual/auto regions
- Caching problems

**Solutions:**
- Use Clear All to reset
- Re-analyze to refresh state
- Restart ComfyUI if persistent

#### Mouse controls unresponsive
**Causes:**
- Canvas focus issues
- Browser compatibility
- ComfyUI zoom conflicts

**Solutions:**
- Click on waveform to focus
- Try different browser
- Reset ComfyUI view zoom

### Audio Loading Issues

#### Supported formats not working
**Causes:**
- Missing audio codecs
- Corrupted files
- Encoding issues

**Solutions:**
- Install additional audio libraries
- Re-encode audio files
- Use WAV format for best compatibility

#### Path resolution problems
**Causes:**
- Relative vs absolute paths
- Special characters in paths
- Directory permissions

**Solutions:**
- Use full absolute paths
- Avoid special characters
- Check folder permissions
- Place files in ComfyUI input directory

### Analysis Problems

#### No regions detected
**Causes:**
- Thresholds too restrictive
- Audio characteristics don't match method
- Very short audio duration

**Solutions:**
- Lower detection thresholds
- Try different analysis methods
- Use manual mode for precise control
- Check audio actually contains target content

#### Too many regions detected
**Causes:**
- Thresholds too sensitive
- Noisy audio input
- Brief audio artifacts

**Solutions:**
- Raise detection thresholds
- Increase minimum duration settings
- Use region grouping to merge
- Pre-process audio to reduce noise

#### Inconsistent results
**Causes:**
- Variable audio quality
- Inconsistent recording conditions
- Parameter sensitivity

**Solutions:**
- Normalize audio levels
- Use consistent recording setup
- Fine-tune parameters per audio type
- Consider manual verification

---

## Advanced Configuration

### Custom Workflows

#### Multi-language Speech
- Adjust silence thresholds for language characteristics
- Use manual regions for complex pronunciation patterns
- Combine methods for different language sections

#### Music Production
- Use peak detection for drum isolation
- Energy method for dynamic sections
- Manual regions for precise arrangement timing

#### Podcast Enhancement
- Silence detection for automatic chapter breaks
- Manual refinement for sponsor segments
- Custom labels for content categorization

### Integration Examples

#### F5-TTS Pipeline
```
Audio File → Audio Analyzer → F5-TTS Edit Node
           ↓                 ↓
       Options Node      Timing Data
```

#### Batch Processing
```
Multiple Audio → Load Audio Node → Audio Analyzer → Export Timing
                                ↓
                            Options (shared settings)
```

#### Quality Control
```
Audio → Audio Analyzer → Preview → Manual Refinement → Final Export
                      ↓                               ↓
                  Visual Check                   Verified Timing
```

### Customization

#### Parameter Presets
Create Options node presets for common use cases:
- **Speech-optimized**: Low silence threshold, invert enabled
- **Music-focused**: Energy detection, higher sensitivity
- **Podcast-ready**: Longer silence duration, grouping enabled

#### Output Formatting
Choose export formats based on destination:
- **F5TTS**: Simple start,end format
- **Analysis**: Detailed JSON with metadata
- **External Tools**: CSV for spreadsheet compatibility

#### Performance Tuning
Optimize settings for your hardware:
- **Fast Preview**: Low visualization points, conservative settings
- **High Quality**: Maximum points, precise thresholds
- **Batch Mode**: Minimal UI updates, caching enabled

---

This comprehensive guide covers all aspects of the Audio Analyzer node. For additional support or feature requests, please refer to the main project documentation or community resources.