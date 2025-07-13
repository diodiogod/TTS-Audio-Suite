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

## $${\color{lightgreen}Quick \space Start}$$

<details>
<summary>$${\color{yellow}Basic \space Workflow \space Steps}$$</summary>

$${\color{lightgreen}1)}$$ **Load Audio**: Drag audio file to interface OR set `audio_file` path OR connect audio input

$${\color{lightgreen}2)}$$ **Choose Method**: Select analysis method (`silence`, `energy`, `peaks`, or `manual`)

$${\color{lightgreen}3)}$$ **Click Analyze**: Process audio to detect timing regions

$${\color{lightgreen}4)}$$ **Refine Regions**: Add/delete manual regions as needed

$${\color{lightgreen}5)}$$ **Export**: Use timing data output for F5-TTS or other applications

![Quick Start Workflow](images/quick_start_workflow.png)

</details>

<details>
<summary>$${\color{orange}First \space Time \space Setup}$$</summary>

- Place audio files in ComfyUI's `input` directory for easy access
- For advanced settings, connect an **Audio Analyzer Options** node
- $${\color{yellow}Recommended:}$$ Start with `silence` method for speech analysis

</details>

<p align="right">(<a href="#table-of-contents">back to top</a>)</p>

---

## $${\color{yellow}Core \space Parameters}$$

<details>
<summary>$${\color{lightgreen}Required \space Parameters}$$</summary>

#### $${\color{orange}audio\_file}$$ (STRING)
- **Purpose**: Path to audio file for analysis
- **Format**: File path or just filename if in ComfyUI input directory
- **Supported Formats**: $${\color{lightgreen}WAV, \space MP3, \space OGG, \space FLAC, \space M4A, \space AAC}$$
- **Priority**: If both `audio_file` and audio input are provided, audio input takes priority

```
Examples:
- "speech_sample.wav"
- "C:/Audio/my_voice.mp3"
- "voices/character_01.flac"
```

#### $${\color{orange}analysis\_method}$$ (DROPDOWN)
- $${\color{lightgreen}silence}$$: Detects pauses between speech (best for clean speech)
- $${\color{yellow}energy}$$: Analyzes volume changes (good for music/noisy audio)
- $${\color{orange}peaks}$$: Finds sharp audio spikes (useful for percussion/effects)
- $${\color{red}manual}$$: Uses only user-defined regions

![Analysis Methods Comparison](images/analysis_methods.png)

#### $${\color{orange}precision\_level}$$ (DROPDOWN)
Controls output timing precision:
- $${\color{yellow}seconds}$$: Rounded to seconds (1.23s) - rough timing
- $${\color{lightgreen}milliseconds}$$: Precise to milliseconds (1.234s) - $${\color{lightgreen}recommended}$$
- $${\color{orange}samples}$$: Raw sample numbers (27225 smp) - exact editing

#### $${\color{orange}visualization\_points}$$ (INT: 500-10000)
Waveform detail level:
- $${\color{lightgreen}500-1000}$$: Smooth, fast rendering
- $${\color{yellow}2000-3000}$$: Balanced detail ($${\color{lightgreen}recommended}$$)
- $${\color{orange}5000-10000}$$: Very detailed, slower but precise

</details>

<details>
<summary>$${\color{orange}Optional \space Parameters}$$</summary>

#### $${\color{orange}audio}$$ (AUDIO INPUT)
- Connect audio from other nodes (takes priority over `audio_file`)
- Useful for processing generated or processed audio

#### $${\color{orange}options}$$ (OPTIONS INPUT)
- Connect **Audio Analyzer Options** node for advanced settings
- If not connected, uses sensible defaults

#### $${\color{orange}manual\_regions}$$ (MULTILINE STRING)
Define custom timing regions:
```
Format: start,end (one per line)
Examples:
1.5,3.2
4.0,6.8
8.1,10.5
```
- $${\color{lightgreen}Bidirectional \space Sync}$$: Interface ↔ text widget
- $${\color{yellow}Auto-sorting}$$: Regions sorted chronologically
- $${\color{orange}Combined \space Mode}$$: Works with auto-detection methods

#### $${\color{orange}region\_labels}$$ (MULTILINE STRING)
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

#### $${\color{orange}export\_format}$$ (DROPDOWN)
- $${\color{lightgreen}f5tts}$$: Simple format for F5-TTS (start,end per line)
- $${\color{yellow}json}$$: Full data with confidence, labels, metadata
- $${\color{orange}csv}$$: Spreadsheet-compatible format

</details>

<p align="right">(<a href="#table-of-contents">back to top</a>)</p>

---

## $${\color{orange}Audio \space Analyzer \space Options \space Node}$$

For advanced control over analysis parameters, use the **Audio Analyzer Options** node.

![Audio Analyzer Options](images/options_node.png)

<details>
<summary>$${\color{lightgreen}Silence \space Detection \space Options}$$</summary>

#### $${\color{orange}silence\_threshold}$$ (0.001-1.000, step 0.001)
- $${\color{lightgreen}Low \space values \space (0.001-0.01)}$$: Detect very quiet passages
- $${\color{yellow}Medium \space values \space (0.01-0.1)}$$: Standard speech pauses
- $${\color{red}High \space values \space (0.1-1.0)}$$: Only detect significant silences

#### $${\color{orange}silence\_min\_duration}$$ (0.01-5.0s, step 0.01s)
Minimum silence length to detect:
- $${\color{lightgreen}0.01-0.05s}$$: Detect brief pauses (word boundaries)
- $${\color{yellow}0.1-0.5s}$$: Standard sentence breaks
- $${\color{red}0.5s+}$$: Only long pauses (paragraph breaks)

#### $${\color{orange}invert\_silence\_regions}$$ (BOOLEAN)
- $${\color{red}False}$$: Returns silence regions (pauses)
- $${\color{lightgreen}True}$$: Returns speech regions (inverted detection)
- $${\color{yellow}Use \space Case}$$: F5-TTS workflows where you need speech segments

![Silence Inversion Example](images/silence_inversion.png)

</details>

<details>
<summary>$${\color{yellow}Energy \space Detection \space Options}$$</summary>

#### $${\color{orange}energy\_sensitivity}$$ (0.1-2.0, step 0.1)
- $${\color{lightgreen}Low \space (0.1-0.5)}$$: Conservative, fewer boundaries
- $${\color{yellow}Medium \space (0.5-1.0)}$$: Balanced detection
- $${\color{red}High \space (1.0-2.0)}$$: Aggressive, more boundaries

</details>

<details>
<summary>$${\color{orange}Peak \space Detection \space Options}$$</summary>

#### $${\color{orange}peak\_threshold}$$ (0.001-1.0, step 0.001)
Minimum amplitude for peak detection

#### $${\color{orange}peak\_min\_distance}$$ (0.01-1.0s, step 0.01s)
Minimum time between detected peaks

#### $${\color{orange}peak\_region\_size}$$ (0.01-1.0s, step 0.01s)
Size of region around each detected peak

</details>

<details>
<summary>$${\color{red}Advanced \space Options}$$</summary>

#### $${\color{orange}group\_regions\_threshold}$$ (0.000-3.000s, step 0.001s)
Merge nearby regions within threshold:
- $${\color{red}0.000}$$: No grouping (default)
- $${\color{yellow}0.1-0.5s}$$: Merge very close regions
- $${\color{orange}0.5-3.0s}$$: Aggressive merging

![Region Grouping](images/region_grouping.png)

</details>

<p align="right">(<a href="#table-of-contents">back to top</a>)</p>

---

## $${\color{lightgreen}Interactive \space Interface}$$

The Audio Analyzer provides a rich interactive interface for precise audio editing.

![Interface Components](images/interface_components.png)

<details>
<summary>$${\color{yellow}Waveform \space Display}$$</summary>

- $${\color{lightblue}Blue \space waveform}$$: Audio amplitude over time
- $${\color{red}Red \space RMS \space line}$$: Root Mean Square energy
- $${\color{gray}Grid \space lines}$$: Time markers for navigation
- $${\color{lightgreen}Colored \space regions}$$: Detected/manual timing regions

</details>

<details>
<summary>$${\color{orange}Mouse \space Controls}$$</summary>

**$${\color{lightgreen}Selection \space \& \space Navigation}$$**
- **Left click + drag**: Select audio region
- **Right click**: Clear selection
- **Double click**: Seek to position
- **Mouse wheel**: Zoom in/out
- **Middle mouse + drag**: Pan waveform
- **CTRL + left/right drag**: Pan waveform

**$${\color{yellow}Region \space Interaction}$$**
- **Left click on region**: Highlight region (green, persistent)
- **Alt + click region**: Multi-select for deletion (orange, toggle)
- **Alt + click empty**: Clear all multi-selections
- **Shift + left click**: Extend selection

**$${\color{orange}Advanced \space Controls}$$**
- **Drag amplitude labels (±0.8)**: Scale waveform vertically
- **Drag loop markers**: Move start/end loop points

</details>

<details>
<summary>$${\color{red}Keyboard \space Shortcuts}$$</summary>

**$${\color{lightgreen}Playback}$$**
- **Space**: Play/pause
- **Arrow keys**: Move playhead (±1s)
- **Shift + Arrow keys**: Move playhead (±10s)
- **Home/End**: Go to start/end

**$${\color{yellow}Editing}$$**
- **Enter**: Add selected region
- **Delete**: Delete highlighted/selected regions
- **Shift + Delete**: Clear all regions
- **Escape**: Clear selection

**$${\color{orange}View}$$**
- **+/-**: Zoom in/out
- **0**: Reset zoom and amplitude scale

**$${\color{red}Looping}$$**
- **L**: Set loop from selection
- **Shift + L**: Toggle looping on/off
- **Shift + C**: Clear loop markers

</details>

<details>
<summary>$${\color{lightgreen}Speed \space Control}$$</summary>

![Speed Control](images/speed_control.png)

The floating speed slider provides advanced playback control:

**$${\color{yellow}Normal \space Range \space (0.0x \space - \space 2.0x)}$$**
- Drag within slider for standard speed control
- Real-time audio playback with speed adjustment

**$${\color{orange}Extended \space Range \space (Rubberband \space Effect)}$$**
- **Drag beyond edges**: Access extreme speeds (-8x to +8x)
- **Acceleration**: Further you drag, faster the speed increases
- **Negative speeds**: Silent backwards playhead movement

**$${\color{lightgreen}Visual \space Feedback}$$**
- Speed display shows actual value (e.g., "4.25x", "-2.50x")
- Thin gray track line for visual reference
- White vertical bar thumb for precise control

</details>

<details>
<summary>$${\color{orange}Control \space Buttons}$$</summary>

**$${\color{lightgreen}Audio \space Management}$$**
- **📁 Upload Audio**: Browse and upload files
- **🔍 Analyze**: Process audio with current settings

**$${\color{yellow}Region \space Management}$$**
- **➕ Add Region**: Add current selection as region
- **🗑️ Delete Region**: Remove highlighted/selected regions
- **🗑️ Clear All**: Remove all manual regions (keeps auto-detected)

**$${\color{orange}Loop \space Controls}$$**
- **🔻 Set Loop**: Set loop markers from selection
- **🔄 Loop ON/OFF**: Toggle loop playback mode
- **🚫 Clear Loop**: Remove loop markers

**$${\color{red}View \space Controls}$$**
- **🔍+ / 🔍-**: Zoom in/out
- **🔄 Reset**: Reset zoom, amplitude, and speed to defaults
- **📋 Export Timings**: Copy timing data to clipboard

</details>

<p align="right">(<a href="#table-of-contents">back to top</a>)</p>

---

## $${\color{yellow}Analysis \space Methods \space Deep \space Dive}$$

<details>
<summary>$${\color{lightgreen}Silence \space Detection}$$ - **Best for**: Clean speech recordings, voice-overs, podcasts</summary>

#### $${\color{orange}How \space it \space works:}$$
$${\color{lightgreen}1)}$$ Analyzes amplitude levels across the audio
$${\color{lightgreen}2)}$$ Identifies regions below silence threshold
$${\color{lightgreen}3)}$$ Filters by minimum duration requirement
$${\color{lightgreen}4)}$$ Optionally inverts to get speech regions

#### $${\color{yellow}Settings \space Impact:}$$
- **Lower threshold**: Detects quieter silences
- **Shorter min duration**: Finds brief pauses
- **Invert enabled**: Returns speech instead of silence

![Silence Detection](images/silence_method.png)

#### $${\color{lightgreen}Use \space Cases:}$$
- $${\color{orange}F5-TTS \space preparation}$$ (with invert enabled)
- Podcast chapter detection
- Speech segment isolation
- Automatic transcription alignment

</details>

<details>
<summary>$${\color{yellow}Energy \space Detection}$$ - **Best for**: Music, noisy audio, variable volume content</summary>

#### $${\color{orange}How \space it \space works:}$$
$${\color{lightgreen}1)}$$ Calculates RMS energy over time windows
$${\color{lightgreen}2)}$$ Detects significant energy changes
$${\color{lightgreen}3)}$$ Creates regions around transition points

#### $${\color{yellow}Settings \space Impact:}$$
- **Higher sensitivity**: More word boundaries detected
- **Lower sensitivity**: Only major transitions

![Energy Detection](images/energy_method.png)

#### $${\color{lightgreen}Use \space Cases:}$$
- Music beat detection
- Noisy speech processing
- Dynamic content analysis
- Volume-based segmentation

</details>

<details>
<summary>$${\color{orange}Peak \space Detection}$$ - **Best for**: Percussion, sound effects, transient-rich audio</summary>

#### $${\color{orange}How \space it \space works:}$$
$${\color{lightgreen}1)}$$ Identifies sharp amplitude peaks
$${\color{lightgreen}2)}$$ Creates regions around each peak
$${\color{lightgreen}3)}$$ Filters by threshold and minimum distance

#### $${\color{yellow}Settings \space Impact:}$$
- **Lower threshold**: Detects smaller peaks
- **Smaller min distance**: Allows closer peaks
- **Larger region size**: Bigger regions around peaks

![Peak Detection](images/peak_method.png)

#### $${\color{lightgreen}Use \space Cases:}$$
- Drum hit isolation
- Sound effect extraction
- Transient analysis
- Rhythmic pattern detection

</details>

<details>
<summary>$${\color{red}Manual \space Mode}$$ - **Best for**: Precise custom timing, complex audio structures</summary>

#### $${\color{orange}How \space it \space works:}$$
- Uses only user-defined regions
- No automatic detection performed
- Full manual control over timing

#### $${\color{yellow}Features:}$$
- Text widget input for precise timing
- Interactive region creation
- Custom labeling support
- Bidirectional sync between interface and text

![Manual Mode](images/manual_method.png)

#### $${\color{lightgreen}Use \space Cases:}$$
- Precise speech editing
- Custom audio segmentation
- Music arrangement timing
- Specific interval extraction

</details>

<p align="right">(<a href="#table-of-contents">back to top</a>)</p>

---

## $${\color{orange}Region \space Management}$$

<details>
<summary>$${\color{lightgreen}Creating \space Regions}$$ - Multiple methods available</summary>

**$${\color{yellow}Automatic \space Detection}$$**
$${\color{lightgreen}1)}$$ Choose analysis method (`silence`, `energy`, `peaks`)
$${\color{lightgreen}2)}$$ Adjust settings via Options node (optional)
$${\color{lightgreen}3)}$$ Click **Analyze** button
$${\color{lightgreen}4)}$$ Regions appear automatically

**$${\color{orange}Manual \space Creation}$$**
$${\color{lightgreen}1)}$$ **Method 1**: Drag to select area → press **Enter** or click **Add Region**
$${\color{lightgreen}2)}$$ **Method 2**: Type in `manual_regions` widget:
   ```
   1.5,3.2
   4.0,6.8
   ```
$${\color{lightgreen}3)}$$ **Method 3**: Use manual mode exclusively

**$${\color{red}Combined \space Approach}$$**
- Use any auto-detection method
- Add manual regions on top
- Both types included in output
- Manual regions persist across analyses

![Creating Regions](images/creating_regions.png)

</details>

<details>
<summary>$${\color{yellow}Region \space Types \space \& \space Colors}$$ - Visual identification system</summary>

**$${\color{lightgreen}Manual \space Regions \space (Green)}$$**
- Created by user interaction
- Editable and persistent
- Always included in output
- Numbered sequentially (Region 1, Region 2, etc.)

**$${\color{orange}Auto-detected \space Regions}$$**
- $${\color{gray}Gray}$$: Silence regions
- $${\color{lightgreen}Forest \space Green}$$: Speech regions (inverted silence)
- $${\color{yellow}Yellow}$$: Energy/word boundaries
- $${\color{lightblue}Blue}$$: Peak regions
- Color indicates detection method

**$${\color{red}Grouped \space Regions}$$**
- Maintain original type color
- Show grouping information in analysis report
- Created when group threshold > 0

</details>

<details>
<summary>$${\color{orange}Editing \space Regions}$$ - Selection and modification tools</summary>

**$${\color{lightgreen}Selection \space States}$$**
- $${\color{lightgreen}Green \space highlight}$$: Single region selected (click)
- $${\color{orange}Orange \space highlight}$$: Multiple regions selected (Alt+click)
- $${\color{yellow}Yellow \space selection}$$: Current area selection

**$${\color{red}Deletion}$$**
- **Single deletion**: Click region → press Delete
- **Multi-deletion**: Alt+click multiple → press Delete
- **Clear all**: Shift+Delete or Clear All button

**$${\color{yellow}Modification}$$**
- **Move regions**: Edit `manual_regions` text widget
- **Rename regions**: Edit `region_labels` text widget
- **Re-analyze**: Adjust settings → click Analyze

![Editing Regions](images/editing_regions.png)

</details>

<details>
<summary>$${\color{red}Region \space Properties}$$ - Technical details and metadata</summary>

**$${\color{lightgreen}Timing \space Information}$$**
- **Start time**: Region beginning
- **End time**: Region ending  
- **Duration**: Calculated length
- **Confidence**: Detection certainty (auto-regions)

**$${\color{orange}Metadata}$$**
- **Type**: manual, silence, speech, energy, peaks
- **Source**: Detection method used
- **Grouping info**: If region was merged

**$${\color{yellow}Labels}$$**
- **Auto-generated**: Region 1, Region 2, etc.
- **Custom**: User-defined names
- **Detection-based**: silence, speech, peak_1, etc.

</details>

<p align="right">(<a href="#table-of-contents">back to top</a>)</p>

---

## $${\color{red}Advanced \space Features}$$

<details>
<summary>$${\color{orange}Region \space Grouping}$$ - Merge nearby regions automatically</summary>

Automatically merge nearby regions to reduce fragmentation.

#### $${\color{lightgreen}How \space it \space works:}$$
$${\color{lightgreen}1)}$$ Set `group_regions_threshold` > 0.000s in Options node
$${\color{lightgreen}2)}$$ Regions within threshold distance get merged
$${\color{lightgreen}3)}$$ Overlapping regions are combined
$${\color{lightgreen}4)}$$ Metadata preserved from source regions

![Region Grouping Example](images/region_grouping_detail.png)

#### $${\color{yellow}Benefits:}$$
- Reduces over-segmentation
- Creates cleaner timing data
- Maintains original region information
- $${\color{orange}Improves \space F5-TTS \space results}$$

</details>

<details>
<summary>$${\color{lightgreen}Silence \space Inversion}$$ - Convert silence to speech detection</summary>

Convert silence detection to speech detection for F5-TTS workflows.

#### $${\color{orange}Process:}$$
$${\color{lightgreen}1)}$$ Normal silence detection finds pauses
$${\color{lightgreen}2)}$$ Inversion calculates speech regions between pauses
$${\color{lightgreen}3)}$$ Output contains only speech segments
$${\color{lightgreen}4)}$$ $${\color{yellow}Ideal \space for \space voice \space cloning \space preparation}$$

![Silence Inversion Process](images/silence_inversion_process.png)

</details>

<details>
<summary>$${\color{yellow}Loop \space Functionality}$$ - Precise playback control</summary>

Precise playback control for detailed editing.

#### $${\color{lightgreen}Setting \space Loops:}$$
$${\color{lightgreen}1)}$$ Select region → press **L** or click **Set Loop**
$${\color{lightgreen}2)}$$ Drag purple loop markers to adjust
$${\color{lightgreen}3)}$$ Use **Shift+L** to toggle looping on/off

#### $${\color{orange}Visual \space Indicators:}$$
- $${\color{purple}Purple \space markers}$$: Loop start/end points
- **Loop status**: Shown in interface
- **Automatic repeat**: When looping enabled

</details>

<details>
<summary>$${\color{orange}Bidirectional \space Sync}$$ - Interface ↔ Text widgets</summary>

Seamless integration between interface and text widgets.

#### $${\color{lightgreen}Text \space → \space Interface:}$$
- Type regions in `manual_regions` widget
- Click back to interface
- Regions automatically appear

#### $${\color{yellow}Interface \space → \space Text:}$$
- Add regions via interface
- Text widgets update automatically
- Labels and timing stay synchronized

</details>

<details>
<summary>$${\color{red}Caching \space System}$$ - Performance optimization</summary>

Intelligent performance optimization.

#### $${\color{orange}How \space it \space works:}$$
- Analysis results cached based on audio + settings
- Instant results for repeated analyses
- Cache invalidated when parameters change
- Manual regions included in cache key

#### $${\color{lightgreen}Benefits:}$$
- Faster repeated processing
- Smooth parameter experimentation
- Reduced computation overhead

</details>

<p align="right">(<a href="#table-of-contents">back to top</a>)</p>

---

## $${\color{lightgreen}Outputs \space Reference}$$

The Audio Analyzer provides four outputs for different use cases:

![Outputs Overview](images/outputs_overview.png)

<details>
<summary>$${\color{lightgreen}1. \space processed\_audio}$$ (AUDIO) - Passthrough for chaining</summary>

- **Purpose**: Passthrough of original audio
- **Use Case**: Continue audio processing pipeline
- **Format**: Standard ComfyUI audio tensor
- $${\color{yellow}Notes}$$: Always first output for easy chaining

</details>

<details>
<summary>$${\color{orange}2. \space timing\_data}$$ (STRING) - Main export output</summary>

- **Purpose**: Main timing export for external use
- **Format**: Depends on `export_format` setting
- **Precision**: Respects `precision_level` setting

**$${\color{lightgreen}F5TTS \space Format:}$$**
```
1.500,3.200
4.000,6.800
8.100,10.500
```

**$${\color{yellow}JSON \space Format:}$$**
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

**$${\color{orange}CSV \space Format:}$$**
```
start,end,label,confidence,duration
1.500,3.200,speech,1.00,1.700
4.000,6.800,speech,1.00,2.800
```

</details>

<details>
<summary>$${\color{yellow}3. \space analysis\_info}$$ (STRING) - Detailed analysis report</summary>

- **Purpose**: Detailed analysis report
- **Content**: Statistics, settings, visualization summary
- **Use Case**: Documentation, debugging, analysis review

**$${\color{lightgreen}Example \space Report:}$$**
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
<summary>$${\color{red}4. \space segmented\_audio}$$ (AUDIO) - Extracted region audio</summary>

- **Purpose**: Audio containing only detected regions
- **Process**: Extracts and concatenates region audio
- **Use Case**: $${\color{orange}F5-TTS \space training}$$, isolated speech extraction
- **Format**: Standard ComfyUI audio tensor

**$${\color{lightgreen}How \space it \space works:}$$**
$${\color{lightgreen}1)}$$ Sort regions by start time
$${\color{lightgreen}2)}$$ Extract audio for each region
$${\color{lightgreen}3)}$$ Concatenate segments sequentially
$${\color{lightgreen}4)}$$ Return as single audio tensor

![Segmented Audio Process](images/segmented_audio.png)

</details>

<p align="right">(<a href="#table-of-contents">back to top</a>)</p>

---

## $${\color{lightgreen}Tips \space \& \space Workflows}$$

<details>
<summary>$${\color{orange}F5-TTS \space Preparation \space Workflow}$$ - Voice cloning setup</summary>

$${\color{lightgreen}1)}$$ **Load clean speech audio**
$${\color{lightgreen}2)}$$ **Connect Audio Analyzer Options** node:
   - Method: `silence`
   - Enable `invert_silence_regions`
   - Set appropriate `silence_threshold`
$${\color{lightgreen}3)}$$ **Analyze** to get speech regions
$${\color{lightgreen}4)}$$ **Fine-tune** by adding manual regions if needed
$${\color{lightgreen}5)}$$ **Use outputs**:
   - `timing_data` → F5-TTS timing input
   - `segmented_audio` → F5-TTS audio input

![F5-TTS Workflow](images/f5tts_workflow.png)

</details>

<details>
<summary>$${\color{yellow}Music \space Analysis \space Workflow}$$ - Beat and rhythm detection</summary>

$${\color{lightgreen}1)}$$ **Load music track**
$${\color{lightgreen}2)}$$ **Use `energy` method** for beat detection
$${\color{lightgreen}3)}$$ **Adjust `energy_sensitivity`** to match dynamics
$${\color{lightgreen}4)}$$ **Add manual regions** for specific sections
$${\color{lightgreen}5)}$$ **Group regions** to merge close beats
$${\color{lightgreen}6)}$$ **Export timing data** for synchronization

</details>

<details>
<summary>$${\color{red}Podcast \space Chapter \space Detection}$$ - Chapter boundary identification</summary>

$${\color{lightgreen}1)}$$ **Load podcast audio**
$${\color{lightgreen}2)}$$ **Use `silence` method** with:
   - Higher `silence_threshold` for speech gaps
   - Longer `silence_min_duration` for chapter breaks
$${\color{lightgreen}3)}$$ **Manual refinement** for precise chapter boundaries
$${\color{lightgreen}4)}$$ **Custom labels** for chapter names
$${\color{lightgreen}5)}$$ **Export for media players**

</details>

<details>
<summary>$${\color{orange}Quality \space Control \space Tips}$$ - Best practices</summary>

**$${\color{lightgreen}Audio \space Preparation}$$**
- **Normalize volume** before analysis
- **Remove background noise** if possible
- **Use consistent recording conditions**
- **Check for clipping or distortion**

**$${\color{yellow}Parameter \space Tuning}$$**
- **Start with defaults** and adjust incrementally
- **Test with short audio samples** first
- **Use visual feedback** from waveform display
- **Compare different methods** for same audio

**$${\color{orange}Verification}$$**
- **Listen to detected regions** using loop functionality
- **Check timing precision** with playhead
- **Verify region boundaries** at detailed zoom levels
- **Test output compatibility** with target applications

</details>

<details>
<summary>$${\color{red}Performance \space Optimization}$$ - Speed and efficiency</summary>

**$${\color{lightgreen}For \space Large \space Files}$$**
- **Reduce `visualization_points`** for faster rendering
- **Use caching** - avoid changing parameters unnecessarily
- **Process in segments** if memory limited
- **Consider downsampling** for initial analysis

**$${\color{yellow}For \space Real-time \space Use}$$**
- **Pre-tune parameters** on representative samples
- **Use manual mode** for known timing patterns
- **Minimize UI interactions** during processing
- **Batch process** similar audio files

</details>

<p align="right">(<a href="#table-of-contents">back to top</a>)</p>

---

## $${\color{red}Troubleshooting}$$

<details>
<summary>$${\color{orange}Common \space Issues}$$ - Frequent problems and solutions</summary>

**$${\color{lightgreen}"No \space audio \space data \space received"}$$**
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

**$${\color{yellow}"Fake \space test \space data" \space warning}$$**
**Causes:**
- Audio loading failed
- No audio source provided
- Network/file access issues

**Solutions:**
- Check audio file accessibility
- Verify ComfyUI input directory setup
- Ensure audio format compatibility
- Re-analyze with proper audio source

**$${\color{orange}Regions \space not \space appearing}$$**
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

**$${\color{red}Performance \space issues}$$**
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

</details>

<details>
<summary>$${\color{yellow}Interface \space Issues}$$ - UI and interaction problems</summary>

**$${\color{lightgreen}Speed \space control \space not \space working}$$**
**Causes:**
- Audio not properly loaded
- Browser audio restrictions
- Conflicting audio processes

**Solutions:**
- Reload audio and re-analyze
- Check browser audio permissions
- Refresh ComfyUI interface

**$${\color{yellow}Visual \space artifacts \space or \space duplicates}$$**
**Causes:**
- Region synchronization issues
- Mixed manual/auto regions
- Caching problems

**Solutions:**
- Use Clear All to reset
- Re-analyze to refresh state
- Restart ComfyUI if persistent

**$${\color{orange}Mouse \space controls \space unresponsive}$$**
**Causes:**
- Canvas focus issues
- Browser compatibility
- ComfyUI zoom conflicts

**Solutions:**
- Click on waveform to focus
- Try different browser
- Reset ComfyUI view zoom

</details>

<details>
<summary>$${\color{lightgreen}Audio \space Loading \space Issues}$$ - File and format problems</summary>

**$${\color{yellow}Supported \space formats \space not \space working}$$**
**Causes:**
- Missing audio codecs
- Corrupted files
- Encoding issues

**Solutions:**
- Install additional audio libraries
- Re-encode audio files
- Use WAV format for best compatibility

**$${\color{orange}Path \space resolution \space problems}$$**
**Causes:**
- Relative vs absolute paths
- Special characters in paths
- Directory permissions

**Solutions:**
- Use full absolute paths
- Avoid special characters
- Check folder permissions
- Place files in ComfyUI input directory

</details>

<details>
<summary>$${\color{red}Analysis \space Problems}$$ - Detection and accuracy issues</summary>

**$${\color{lightgreen}No \space regions \space detected}$$**
**Causes:**
- Thresholds too restrictive
- Audio characteristics don't match method
- Very short audio duration

**Solutions:**
- Lower detection thresholds
- Try different analysis methods
- Use manual mode for precise control
- Check audio actually contains target content

**$${\color{yellow}Too \space many \space regions \space detected}$$**
**Causes:**
- Thresholds too sensitive
- Noisy audio input
- Brief audio artifacts

**Solutions:**
- Raise detection thresholds
- Increase minimum duration settings
- Use region grouping to merge
- Pre-process audio to reduce noise

**$${\color{orange}Inconsistent \space results}$$**
**Causes:**
- Variable audio quality
- Inconsistent recording conditions
- Parameter sensitivity

**Solutions:**
- Normalize audio levels
- Use consistent recording setup
- Fine-tune parameters per audio type
- Consider manual verification

</details>

<p align="right">(<a href="#table-of-contents">back to top</a>)</p>

---

## $${\color{orange}Advanced \space Configuration}$$

<details>
<summary>$${\color{lightgreen}Custom \space Workflows}$$ - Advanced usage patterns</summary>

**$${\color{yellow}Multi-language \space Speech}$$**
- Adjust silence thresholds for language characteristics
- Use manual regions for complex pronunciation patterns
- Combine methods for different language sections

**$${\color{orange}Music \space Production}$$**
- Use peak detection for drum isolation
- Energy method for dynamic sections
- Manual regions for precise arrangement timing

**$${\color{lightgreen}Podcast \space Enhancement}$$**
- Silence detection for automatic chapter breaks
- Manual refinement for sponsor segments
- Custom labels for content categorization

</details>

<details>
<summary>$${\color{yellow}Integration \space Examples}$$ - Workflow patterns</summary>

**$${\color{lightgreen}F5-TTS \space Pipeline}$$**
```
Audio File → Audio Analyzer → F5-TTS Edit Node
           ↓                 ↓
       Options Node      Timing Data
```

**$${\color{orange}Batch \space Processing}$$**
```
Multiple Audio → Load Audio Node → Audio Analyzer → Export Timing
                                ↓
                            Options (shared settings)
```

**$${\color{red}Quality \space Control}$$**
```
Audio → Audio Analyzer → Preview → Manual Refinement → Final Export
                      ↓                               ↓
                  Visual Check                   Verified Timing
```

</details>

<details>
<summary>$${\color{red}Customization}$$ - Advanced configuration options</summary>

**$${\color{lightgreen}Parameter \space Presets}$$**
Create Options node presets for common use cases:
- $${\color{orange}Speech-optimized}$$: Low silence threshold, invert enabled
- $${\color{yellow}Music-focused}$$: Energy detection, higher sensitivity
- $${\color{lightgreen}Podcast-ready}$$: Longer silence duration, grouping enabled

**$${\color{yellow}Output \space Formatting}$$**
Choose export formats based on destination:
- $${\color{orange}F5TTS}$$: Simple start,end format
- $${\color{lightgreen}Analysis}$$: Detailed JSON with metadata
- $${\color{yellow}External \space Tools}$$: CSV for spreadsheet compatibility

</details>

<p align="right">(<a href="#table-of-contents">back to top</a>)</p>

---

$${\color{lightgreen}🎉 \space Document \space transformation \space complete!}$$ This comprehensive guide now features:

- $${\color{orange}Colored \space headers \space and \space text}$$ for better visual hierarchy
- $${\color{yellow}Collapsible \space sections}$$ using `<details>` tags for cleaner organization  
- $${\color{lightgreen}"Back \space to \space top"}$$ links throughout for easy navigation
- $${\color{red}Consolidated \space content}$$ with reduced redundancy
- $${\color{lightblue}User-friendly \space structure}$$ for improved readability

For additional support or feature requests, please refer to the main project documentation or community resources.

<p align="right">(<a href="#table-of-contents">back to top</a>)</p>