# 🌊 Audio Wave Analyzer - Complete User Guide

The Audio Wave Analyzer is a sophisticated waveform visualization and timing extraction tool designed primarily for **F5-TTS speech editing workflows**. It provides precise timing data for speech regions, making it ideal for preparing audio segments for F5-TTS voice cloning and editing.

### Why Use Audio Wave Analyzer for F5-TTS?

The F5-TTS Edit Node requires precise timing data to know which parts of audio to replace. The Audio Wave Analyzer excels at:

- **Speech Region Detection**: Automatically finds where speech occurs
- **Precise Timing**: Provides exact start/end times for each speech segment
- **Visual Verification**: Interactive waveform lets you verify and adjust regions
- **Clean Output**: Generates timing data in the exact format F5-TTS expects

![Audio Analyzer Interface Overview](images/audio_analyzer_overview.png)

## Table of Contents

1. [Quick Start](#quick-start)
2. [Node Parameters](#node-parameters)
3. [Audio Analyzer Options Node](#audio-analyzer-options-node)
4. [Interactive Interface](#interactive-interface)
5. [Interactive Buttons](#interactive-buttons)
6. [Analysis Methods Breakdown](#analysis-methods-breakdown)
7. [Region Management](#region-management)
8. [Advanced Features](#advanced-features)
9. [Outputs-Reference](#outputs-reference)

---

## Quick Start

### 🚀 Basic Workflow

1. **Load Audio**: Drag audio file to interface OR set `audio_file` path OR connect audio input
2. **Choose Method**: Select analysis method (`silence`, `energy`, `peaks`, or `manual`)
3. **Click Analyze**: Process audio to detect timing regions
4. **Refine Regions**: Add/delete manual regions as needed
5. **Export**: Use timing data output for F5-TTS or other applications

![Quick Start Workflow](images/quick_start_workflow.png)

## Node Parameters

### 🎵 Core Inputs

<details>
<summary><strong>audio_file</strong></summary>

- **Purpose**: Path to audio file for analysis

- **Format**: File path or just filename if in ComfyUI input directory

- **Supported Formats**: WAV, MP3, OGG, FLAC, M4A, AAC

```
Examples:
- "speech_sample.wav"
- "C:/Audio/my_voice.mp3"
- "voices/character_01.flac"
```

</details>

> [!IMPORTANT]
> If both `audio_file` and `audio input` are provided, **audio input takes priority**.

<details>
<summary><strong>analysis_method (DROPDOWN)</strong></summary>

- **silence**: Detects pauses between speech (best for clean speech) ⭐ Recommended for F5-TTS

- **energy**: Analyzes volume changes (good for music/noisy audio)

- **peaks**: Finds sharp audio spikes (useful for percussion/effects)

- **manual**: Uses only user-defined regions
  
  </details>

<details>
<summary><strong>precision_level & visualization_points</strong></summary>

**precision_level**: Output timing format

- **milliseconds**: `1.234s` ⭐ **Recommended**
- **seconds**: `1.23s` (rounded)
- **samples**: `27225 smp` (exact)

**visualization_points**: Waveform detail (500-10000)

- **2000-3000**: ⭐ **Recommended balance**
- **500-1000**: Faster, less detail
- **5000-10000**: Slower, more detail

</details>

### 🛠️ Optional Inputs

<details>
<summary><strong>audio</strong></summary>

- **Purpose**: Connect audio from other nodes (takes priority over `audio_file`)

- **Format**: Audio connection from upstream nodes

- **Use Case**: Processing generated or processed audio in workflows

```
Examples:
- Audio from TTS generation nodes
- Processed audio from effects chains
- Real-time audio input streams
```

</details>

<details>
<summary><strong>options</strong></summary>

- **Purpose**: Connect **Audio Wave Analyzer Options** extra node for advanced settings and custom threshold values

- **Default Behavior**: Uses sensible defaults if not connected

</details>

<details>
<summary><strong>manual_regions</strong></summary>

- **Purpose**: Define custom timing regions for analysis

- **Format**: `start,end` (one per line)

- **Features**: Bidirectional sync, auto-sorting, works with auto-detection

```
Examples:
1.5,3.2
4.0,6.8
8.1,10.5
```

</details>

<details>
<summary><strong>region_labels</strong></summary>

- **Purpose**: Custom labels for manual regions

- **Format**: One label per line (must match number of manual regions)

- **Behavior**: Custom labels preserved during sorting, auto-generated labels get renumbered

```
Examples:
Intro
Verse 1
Chorus
Bridge
```

</details>

<details>
<summary><strong>export_format</strong></summary>

- **f5tts**: Simple format for F5-TTS (start,end per line) ⭐ **Recommended for F5-TTS**

- **json**: Full data with confidence, labels, metadata

- **csv**: Spreadsheet-compatible format
  
  </details>

## Audio Analyzer Options Node

For advanced control over analysis parameters, use the **Audio Analyzer Options** node.

![Audio Analyzer Options](images/options_node.png)

### 🔇 Silence Detection Options

<details>
<summary><strong>silence_threshold</strong></summary>
(0.001-1.000, step 0.001)

- **Low values (0.001-0.01)**: Detect very quiet passages

- **Medium values (0.01-0.1)**: Standard speech pauses

- **High values (0.1-1.0)**: Only detect significant silences
  
  </details>

<details>
<summary><strong>silence_min_duration</strong></summary>
(0.01-5.0s, step 0.01s)

Minimum silence length to detect:

- **0.01-0.05s**: Detect brief pauses (word boundaries)

- **0.1-0.5s**: Standard sentence breaks

- **0.5s+**: Only long pauses (paragraph breaks)
  
  </details>

<details>
<summary><strong>invert_silence_regions</strong></summary>
(BOOLEAN)

- **False**: Returns silence regions (pauses)
- **True**: Returns speech regions (inverted detection)
- **Use Case**: F5-TTS workflows where you need speech segments

![Silence Inversion Example](images/silence_inversion.png)

</details>

### ⚡ Energy Detection Options

<details>
<summary><strong>energy_sensitivity</strong></summary>
(0.1-2.0, step 0.1)

- **Low (0.1-0.5)**: Conservative, fewer boundaries

- **Medium (0.5-1.0)**: Balanced detection

- **High (1.0-2.0)**: Aggressive, more boundaries
  
  </details>

### 🏔️ Peak Detection Options

<details>
<summary><strong>peak_threshold</strong></summary>
(0.001-1.0, step 0.001)

Minimum amplitude for peak detection

</details>

<details>
<summary><strong>peak_min_distance</strong></summary>
(0.01-1.0s, step 0.01s)

Minimum time between detected peaks

</details>

<details>
<summary><strong>peak_region_size</strong></summary>
(0.01-1.0s, step 0.01s)

Size of region around each detected peak

</details>

### ⚙️ Advanced Options

<details>
<summary><strong>group_regions_threshold</strong></summary>
(0.000-3.000s, step 0.001s)

Merge nearby regions within threshold:

- **0.000**: No grouping (default)
- **0.1-0.5s**: Merge very close regions
- **0.5-3.0s**: Aggressive merging

![Region Grouping](images/region_grouping.png)

</details>

## Interactive Interface

The Audio Analyzer provides a rich interactive interface for precise audio editing.

![Interface Components](images/interface_components.png)

### 🔷 Waveform Display

- **Blue waveform**: Audio amplitude over time
- **Red RMS line**: Root Mean Square energy
- **Grid lines**: Time markers for navigation
- **Colored regions**: Detected/manual timing regions

### 🔶 Mouse Controls

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

### 🔶 Keyboard Shortcuts

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

### 🔷 Speed Control

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



## Interactive Buttons

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

## Analysis Methods Breakdown

<details>
<summary><strong>🔇 Silence Detection</strong></summary>

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

</details>

<details>
<summary><strong>⚡ Energy Detection</strong></summary>

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

</details>

<details>
<summary><strong>🏔️ Peak Detection</strong></summary>

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

</details>

<details>
<summary><strong>🖐️ Manual Mode</strong></summary>

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

</details>

## Region Management

<details>
<summary><strong>➕ Creating Regions</strong></summary>

#### Automatic Detection

1. Choose analysis method (`silence`, `energy`, `peaks`)
2. Adjust settings via Options node (optional)
3. Click **Analyze** button
4. Regions appear automatically

#### Manual Creation

1. **Method 1**: Drag to select area → press **Enter** or click **Add Region**

2. **Method 2**: Type in `manual_regions` widget:
   1.5,3.2
   4.0,6.8

3. **Method 3**: Use manual mode exclusively

#### Combined Approach

- Use any auto-detection method
- Add manual regions on top
- Both types included in output
- Manual regions persist across analyses

![Creating Regions](images/creating_regions.png)

</details>

<details>
<summary><strong>🎨 Region Types & Colors</strong></summary>

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

</details>

<details>
<summary><strong>✏️ Editing Regions</strong></summary>

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

</details>

<details>
<summary><strong>🏷️ Region Properties</strong></summary>

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

</details>

## Advanced Features

<details>
<summary><strong>🔗 Region Grouping</strong></summary>

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

</details>

<details>
<summary><strong>🔇 Silence Inversion</strong></summary>

Convert silence detection to speech detection for F5-TTS workflows.

#### Process:

1. Normal silence detection finds pauses
2. Inversion calculates speech regions between pauses
3. Output contains only speech segments
4. Ideal for voice cloning preparation

![Silence Inversion Process](images/silence_inversion_process.png)

</details>

<details>
<summary><strong>🔁 Loop Functionality</strong></summary>

Precise playback control for detailed editing.

#### Setting Loops:

1. Select region → press **L** or click **Set Loop**
2. Drag purple loop markers to adjust
3. Use **Shift+L** to toggle looping on/off

#### Visual Indicators:

- **Purple markers**: Loop start/end points
- **Loop status**: Shown in interface
- **Automatic repeat**: When looping enabled

</details>

<details>
<summary><strong>🔀 Bidirectional Sync</strong></summary>

Seamless integration between interface and text widgets.

#### Text → Interface:

- Type regions in `manual_regions` widget
- Click back to interface
- Regions automatically appear

#### Interface → Text:

- Add regions via interface
- Text widgets update automatically
- Labels and timing stay synchronized

</details>

<details>
<summary><strong>💾 Caching System</strong></summary>

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

</details>

---

## Outputs Reference

The Audio Analyzer provides four outputs for different use cases:

![Outputs Overview](images/outputs_overview.png)

<details>
<summary>🔊 `processed_audio` (AUDIO)</summary>

- **Purpose**: Passthrough of original audio  
- **Use Case**: Continue audio processing pipeline  
- **Format**: Standard ComfyUI audio tensor  
- **Notes**: Always first output for easy chaining

</details>

<details>
<summary>🕒 `timing_data` (STRING)</summary>

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

</details>

<details>
<summary>📄 `analysis_info` (STRING)</summary>

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

</details>

<details>
<summary>✂️ `segmented_audio` (AUDIO)</summary>

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

</details>

---

This comprehensive guide covers all aspects of the Audio Analyzer node. For additional support or feature requests, please refer to the main project documentation or community resources.
