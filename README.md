<a id="readme-top"></a>

[![](https://dcbadge.limes.pink/api/server/EwKE8KBDqD)](https://discord.gg/EwKE8KBDqD)
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Forks][forks-shield]][forks-url]
[![Dynamic TOML Badge][version-shield]][version-url]

# ComfyUI ChatterBox SRT Voice (diogod) v3.0.13

*This is a refactored node, originally created by [ShmuelRonen](https://github.com/ShmuelRonen/ComfyUI_ChatterBox_Voice).*

<div align="center">
  <img src="images/AllNodesShowcase.png" alt="ComfyUI ChatterBox Voice Nodes Showcase" />
</div>

An unofficial ComfyUI custom node integration for High-quality Text-to-Speech and Voice Conversion nodes for ComfyUI using ResembleAI's ChatterboxTTS with unlimited text length and with a node specially made for dealing with SRT timings.

## 🆕 What's New in my Project?

### SRT Timing and TTS Node

<img title="" src="images/srt.png" alt="SRT Node Screenshot" width="500" data-align="center">

The **"ChatterBox SRT Voice TTS"** node allows TTS generation by processing SRT content (SubRip Subtitle) files, ensuring precise timing and synchronization with your audio.

**Key SRT Features:**

* **SRT style Processing**: Uses SRT style to generate TTS, aligning audio with subtitle timings
* **`smart_natural` Timing Mode**: Intelligent shifting logic that prevents overlaps and ensures natural speech flow
* **`Adjusted_SRT` Output**: Provides actual timings for generated audio for accurate post-processing
* **Segment-Level Caching**: Only regenerates modified segments, significantly speeding up workflows

<details>
<summary>📖 More SRT Technical Details</summary>

For comprehensive technical information, refer to the [SRT_IMPLEMENTATION.md](docs/Dev%20reports/SRT_IMPLEMENTATION.md) file.

</details>

### 🆕 F5-TTS Integration and 🆕 Audio Analyzer

<img title="" src="images/waveanalgif.gif" alt="Audio Wave gif" width="500" data-align="center">

* **F5-TTS Voice Synthesis**: High-quality voice cloning with reference audio + text
* **Audio Wave Analyzer**: Interactive waveform visualization for precise timing extraction
* **Multi-language Support**: English, German, Spanish, French, Japanese models
* **Speech Editing Workflows**: Advanced F5-TTS editing capabilities

### 🎭 Character & Narrator Switching

**NEW in v3.0.13**: Seamless character switching for both F5TTS and ChatterBox engines!

* **Multi-Character Support**: Use `[CharacterName]` tags to switch between different voices
* **Voice Folder Integration**: Organized character voice management system
* **Robust Fallback**: Graceful handling when characters not found (no errors!)
* **Universal Compatibility**: Works with both F5TTS and ChatterBox TTS engines
* **SRT Integration**: Character switching within subtitle timing
* **Backward Compatible**: Existing workflows work unchanged

**📖 [Complete Character Switching Guide](docs/CHARACTER_SWITCHING_GUIDE.md)**

Example usage:
```
Hello! This is the narrator speaking.
[Alice] Hi there! I'm Alice, nice to meet you.
[Bob] And I'm Bob! Great to meet you both.
Back to the narrator for the conclusion.
``` 

## 🎥 Demo Video

<a href="https://youtu.be/VyOawMrCB1g?si=7BubljRhsudGqG3s">
  <img src="https://img.youtube.com/vi/VyOawMrCB1g/maxresdefault.jpg" width="400" alt="ChatterBox SRT Voice Demo">
</a>

**[▶️ Watch Full Demo](https://youtu.be/VyOawMrCB1g?si=7BubljRhsudGqG3s)**

### [ShmuelRonen](https://github.com/ShmuelRonen/ComfyUI_ChatterBox_Voice) ChatterBox TTS Nodes

<div align="center">
  <img src="https://github.com/user-attachments/assets/4197818c-8093-4da4-abd5-577943ac902c" width="45%" alt="ChatterBox TTS Nodes" />
  <img src="https://github.com/user-attachments/assets/701c219b-12ff-4567-b414-e58560594ffe" width="45%" alt="ChatterBox Voice Capture" />
</div>

* **Voice Recording**: Smart silence detection for voice capture
* **Enhanced Chunking**: Intelligent text splitting with multiple combination methods
* **Unlimited Text Length**: No character limits with smart processing

## Features

- 🎤 **ChatterBox TTS** - Generate speech from text with optional voice cloning
- 🎙️ **F5-TTS** - High-quality voice synthesis with reference audio + text cloning
- 🔄 **ChatterBox VC** - Convert voice from one speaker to another
- 🎙️ **ChatterBox Voice Capture** - Record voice input with smart silence detection
- ⚡ **Fast & Quality** - Production-grade TTS that outperforms ElevenLabs
- 🎭 **Emotion Control** - Unique exaggeration parameter for expressive speech
- 🌍 **Multi-language F5-TTS** - Support for English, German, Spanish, French, Japanese and more
- 📝 **Enhanced Chunking** - Intelligent text splitting for long content with multiple combination methods
- 📦 **Self-Contained** - Bundled ChatterBox for zero-installation-hassle experience
- 🎵 **Advanced Audio Processing** - Optional FFmpeg support for premium audio quality with graceful fallback
- 🌊 **Audio Wave Analyzer** - Interactive waveform visualization and precise timing extraction for F5-TTS workflows → **[📖 Complete Guide](docs/🌊_Audio_Wave_Analyzer-Complete_User_Guide.md)**

## 🚀 Quick Start

**Get running in 5 minutes:**

1. **Clone & Install**
   
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/diodiogod/ComfyUI_ChatterBox_SRT_Voice.git
   pip install -r ComfyUI_ChatterBox_SRT_Voice/requirements.txt
   ```

2. **Download Models** (Required)
   
   - Download from [HuggingFace ChatterBox](https://huggingface.co/ResembleAI/chatterbox/tree/main)
   - Place in `ComfyUI/models/chatterbox/`

3. **Try a Workflow**
   
   - Download: [ChatterBox Integration Workflow](example_workflows/Chatterbox%20integration.json)
   - Drag into ComfyUI and start generating!

4. **Restart ComfyUI** and look for 🎤 ChatterBox nodes

> **Need F5-TTS?** Also download F5-TTS models to `ComfyUI/models/F5-TTS/` from the links in the detailed installation below.

## Installation

> **Note:** There are multiple ChatterBox extensions available. This implementation focuses on simplicity, ComfyUI standards, and enhanced text processing capabilities.

### 1. Clone Repository

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/diodiogod/ComfyUI_ChatterBox_SRT_Voice.git
```

### 2. Install Dependencies

Some dependencies, particularly `s3tokenizer`, can occasionally cause installation issues on certain Python setups (e.g., Python 3.10, sometimes used by tools like Stability Matrix).

To minimize potential problems, it's highly recommended to first ensure your core packaging tools are up-to-date in your ComfyUI's virtual environment:

```bash
python -m pip install --upgrade pip setuptools wheel
```

After running the command above, install the node's specific requirements:

```bash
pip install -r requirements.txt
```

### 3. Optional: Install FFmpeg for Enhanced Audio Processing

ChatterBox Voice now supports FFmpeg for high-quality audio stretching. While not required, it's recommended for the best audio quality:

**Windows:**

```bash
winget install FFmpeg
# or with Chocolatey
choco install ffmpeg
```

**macOS:**

```bash
brew install ffmpeg
```

**Linux:**

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

If FFmpeg is not available, ChatterBox will automatically fall back to using the built-in phase vocoder method for audio stretching - your workflows will continue to work without interruption.

### 4. Download Models

**Download the ChatterboxTTS models** and place them in:

```
ComfyUI/models/chatterbox/
```

**Required files:**

- `conds.pt` (105 KB)
- `s3gen.pt` (~1 GB)
- `t3_cfg.pt` (~1 GB)  
- `tokenizer.json` (25 KB)
- `ve.pt` (5.5 MB)

**Download from:** https://huggingface.co/ResembleAI/chatterbox/tree/main

### 5. F5-TTS Models (Optional)

**For F5-TTS voice synthesis capabilities**, download F5-TTS models and place them in:

```
ComfyUI/models/F5-TTS/
```

**Available F5-TTS Models:**

| Model             | Language         | Download                                                                         | Size   |
| ----------------- | ---------------- | -------------------------------------------------------------------------------- | ------ |
| **F5TTS_Base**    | English          | [HuggingFace](https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_Base)         | ~1.2GB |
| **F5TTS_v1_Base** | English (v1)     | [HuggingFace](https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_v1_Base)      | ~1.2GB |
| **E2TTS_Base**    | English (E2-TTS) | [HuggingFace](https://huggingface.co/SWivid/E2-TTS/tree/main/E2TTS_Base)         | ~1.2GB |
| **F5-DE**         | German           | [HuggingFace](https://huggingface.co/aihpi/F5-TTS-German)                        | ~1.2GB |
| **F5-ES**         | Spanish          | [HuggingFace](https://huggingface.co/jpgallegoar/F5-Spanish)                     | ~1.2GB |
| **F5-FR**         | French           | [HuggingFace](https://huggingface.co/RASPIAUDIO/F5-French-MixedSpeakers-reduced) | ~1.2GB |
| **F5-JP**         | Japanese         | [HuggingFace](https://huggingface.co/Jmica/F5TTS)                                | ~1.2GB |

**Vocoder (Optional but Recommended):**

```
ComfyUI/models/F5-TTS/vocos/
├── config.yaml
├── pytorch_model.bin
└── vocab.txt
```

Download from: [Vocos Mel-24kHz](https://huggingface.co/charactr/vocos-mel-24khz)

**Complete Folder Structure:**

```
ComfyUI/models/F5-TTS/
├── F5TTS_Base/
│   ├── model_1200000.safetensors    ← Main model file
│   └── vocab.txt                    ← Vocabulary file
├── vocos/                           ← For offline vocoder
│   ├── config.yaml
│   └── pytorch_model.bin
└── F5TTS_v1_Base/
    ├── model_1250000.safetensors
    └── vocab.txt
```

**Required Files for Each Model:**

- `model_XXXXXX.safetensors` - The main model weights
- `vocab.txt` - Vocabulary/tokenizer file (download from same HuggingFace repo)

**Note:** F5-TTS uses internal config files, no config.yaml needed. Vocos vocoder doesn't need vocab.txt.

**Note:** F5-TTS models and vocoder will auto-download from HuggingFace if not found locally. The first generation may take longer while downloading (~1.2GB per model).

### 6. F5-TTS Voice References Setup

**For easy voice reference management**, create a dedicated voices folder:

```
ComfyUI/models/voices/
├── character1.wav
├── character1.reference.txt ← Contains: "Hello, I am character one speaking clearly."
├── character1.txt          ← Contains: "BBC Radio sample, licensed under CC3..."
├── narrator.wav
├── narrator.txt            ← Contains: "This is the narrator voice for storytelling."
├── my_voice.wav
└── my_voice.txt            ← Contains: "This is my personal voice sample."
```

**Voice Reference Requirements:**

- **Audio files**: WAV format, 5-30 seconds, clean speech, 24kHz recommended
- **Text files**: Exact transcription of what's spoken in the audio file
- **Naming**: `filename.wav` + `filename.reference.txt` (preferred) or `filename.txt` (fallback)

### 7. Restart ComfyUI

## Enhanced Features

### 📝 Intelligent Text Chunking (NEW!)

**Long text support with smart processing:**

- **Character-based limits** (100-1000 chars per chunk)
- **Sentence boundary preservation** - won't cut mid-sentence
- **Multiple combination methods**:
  - `auto` - Smart selection based on text length
  - `concatenate` - Simple joining
  - `silence_padding` - Add configurable silence between chunks
  - `crossfade` - Smooth audio blending
- **Comma-based splitting** for very long sentences
- **Backward compatible** - works with existing workflows

**Chunking Controls (all optional):**

- `enable_chunking` - Enable/disable smart chunking (default: True)
- `max_chars_per_chunk` - Chunk size limit (default: 400)
- `chunk_combination_method` - How to join audio (default: auto)
- `silence_between_chunks_ms` - Silence duration (default: 100ms)

**Auto-selection logic:**

- **Text > 1000 chars** → silence_padding (natural pauses)
- **Text > 500 chars** → crossfade (smooth blending)  
- **Text < 500 chars** → concatenate (simple joining)

### 📦 Smart Model Loading

**Priority-based model detection:**

1. **Bundled models** in node folder (self-contained)
2. **ComfyUI models** in standard location  
3. **HuggingFace download** with authentication

**Console output shows source:**

```
📦 Using BUNDLED ChatterBox (self-contained)
📦 Loading from bundled models: ./models/chatterbox
✅ ChatterboxTTS model loaded from bundled!
```

## Usage

### Voice Recording

1. Add **"🎤 ChatterBox Voice Capture"** node
2. Select your microphone from the dropdown
3. Adjust recording settings:
   - **Silence Threshold**: How quiet to consider "silence" (0.001-0.1)
   - **Silence Duration**: How long to wait before stopping (0.5-5.0 seconds)
   - **Sample Rate**: Audio quality (8000-96000 Hz, default 44100)
4. Change the **Trigger** value to start a new recording
5. Connect output to TTS (for voice cloning) or VC nodes

### Enhanced Text-to-Speech

1. Add **"🎤 ChatterBox Voice TTS"** node
2. Enter your text (any length - automatic chunking)
3. Optionally connect reference audio for voice cloning
4. Adjust TTS settings:
   - **Exaggeration**: Emotion intensity (0.25-2.0)
   - **Temperature**: Randomness (0.05-5.0)
   - **CFG Weight**: Guidance strength (0.0-1.0)
5. Configure chunking (optional):
   - **Enable Chunking**: For long texts
   - **Max Chars Per Chunk**: Chunk size (100-1000)
   - **Combination Method**: How to join chunks
   - **Silence Between Chunks**: Pause duration

### F5-TTS Voice Synthesis

1. Add **"🎤 F5-TTS Voice Generation"** node
2. Enter your target text (any length - automatic chunking)
3. **Required**: Connect reference audio for voice cloning
4. **Required**: Enter reference text that matches the reference audio exactly

<details>
<summary>📖 Voice Reference Setup Options</summary>

**Two ways to provide voice references:**

1. **Easy Method**: Select voice from `reference_audio_file` dropdown → text auto-detected from companion `.txt` file
2. **Manual Method**: Set `reference_audio_file` to "none" → connect `opt_reference_audio` + `opt_reference_text` inputs

</details>

5. Select F5-TTS model:
   - **F5TTS_Base**: English base model (recommended)
   - **F5TTS_v1_Base**: English v1 model
   - **E2TTS_Base**: E2-TTS model
   - **F5-DE**: German model
   - **F5-ES**: Spanish model
   - **F5-FR**: French model
   - **F5-JP**: Japanese model
6. Adjust F5-TTS settings:
   - **Temperature**: Voice variation (0.1-2.0, default: 0.8)
   - **Speed**: Speech speed (0.5-2.0, default: 1.0)
   - **CFG Strength**: Guidance strength (0.0-10.0, default: 2.0)
   - **NFE Step**: Quality vs speed (1-100, default: 32)

### Voice Conversion

1. Add **"🔄 ChatterBox Voice Conversion"** node
2. Connect source audio (voice to convert)
3. Connect target audio (voice style to copy)

## 📁 Example Workflows

**Ready-to-use ComfyUI workflows** - Download and drag into ComfyUI:

### Chatterbox Workflows

| Workflow                   | Description                                 | Files                                                      |
| -------------------------- | ------------------------------------------- | ---------------------------------------------------------- |
| **ChatterBox SRT**         | SRT subtitle timing and TTS generation      | [📁 JSON](example_workflows/📺%20Chatterbox%20SRT.json)    |
| **ChatterBox Integration** | General ChatterBox TTS and Voice Conversion | [📁 JSON](example_workflows/Chatterbox%20integration.json) |

### F5-TTS Workflows

| Workflow                                 | Description                                             | Files                                                                                |
| ---------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Audio Wave Analyzer + F5 Speech Edit** | Interactive waveform analysis for F5-TTS speech editing | [📁 JSON](example_workflows/👄%20F5-TTS%20Speech%20Editor%20Workflow.json)           |
| **F5-TTS SRT and Normal Generation**     | F5-TTS integration with SRT subtitle processing         | [📁 JSON](example_workflows/🎤%20📺%20F5-TTS%20SRT%20and%20Normal%20Generation.json) |

> **Note:** To use workflows, download the `.json` files and drag them directly into your ComfyUI interface. The workflows will automatically load with the proper node connections.

## Settings Guide

### Enhanced Chunking Settings

**For Long Articles/Books:**

- `max_chars_per_chunk=600`, `combination_method=silence_padding`, `silence_between_chunks_ms=200`

**For Natural Speech:**

- `max_chars_per_chunk=400`, `combination_method=auto` (default - works well)

**For Fast Processing:**

- `max_chars_per_chunk=800`, `combination_method=concatenate`

**For Smooth Audio:**

- `max_chars_per_chunk=300`, `combination_method=crossfade`

### Voice Recording Settings

**General Recording:**

- `silence_threshold=0.01`, `silence_duration=2.0` (default settings)

**Noisy Environment:**

- Higher `silence_threshold` (~0.05) to ignore background noise
- Longer `silence_duration` (~3.0) to avoid cutting off speech

**Quiet Environment:**

- Lower `silence_threshold` (~0.005) for sensitive detection
- Shorter `silence_duration` (~1.0) for quick stopping

### TTS Settings

**General Use:**

- `exaggeration=0.5`, `cfg_weight=0.5` (default settings work well)

**Expressive Speech:**

- Lower `cfg_weight` (~0.3) + higher `exaggeration` (~0.7)
- Higher exaggeration speeds up speech; lower CFG slows it down

## Text Processing Capabilities

### 📚 No Hard Text Limits!

Unlike many TTS systems:

- **OpenAI TTS**: 4096 character limit
- **ElevenLabs**: 2500 character limit  
- **ChatterBox**: No documented limits + intelligent chunking

### 🧠 Smart Text Splitting

**Sentence Boundary Detection:**

- Splits on `.!?` with proper spacing
- Preserves sentence integrity
- Handles abbreviations and edge cases

**Long Sentence Handling:**

- Splits on commas when sentences are too long
- Maintains natural speech patterns
- Falls back to character limits only when necessary

**Examples:**

```
Input: "This is a very long article about artificial intelligence and machine learning. It contains multiple sentences and complex punctuation, including lists, quotes, and technical terms. The enhanced chunking system will split this intelligently."

Output: 3 well-formed chunks with natural boundaries
```

## License

MIT License - Same as ChatterboxTTS

## Credits

- **ResembleAI** for ChatterboxTTS
- **ComfyUI** team for the amazing framework
- **sounddevice** library for audio recording functionality
- **[ShmuelRonen](https://github.com/ShmuelRonen/ComfyUI_ChatterBox_Voice)** for the Original ChatteBox Voice TTS node
- **[Diogod](https://github.com/diodiogod/ComfyUI_ChatterBox_SRT_Voice)** for the SRT Timing and TTS Node implementation

## 🔗 Links

- [Resemble AI ChatterBox](https://github.com/resemble-ai/chatterbox)
- [Model Downloads (Hugging Face)](https://huggingface.co/ResembleAI/chatterbox/tree/main) ⬅️ **Download models here**
- [ChatterBox Demo](https://resemble-ai.github.io/chatterbox_demopage/)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Resemble AI Official Site](https://www.resemble.ai/chatterbox/)

---

**Note**: The original ChatterBox model includes Resemble AI's Perth watermarking system for responsible AI usage. This ComfyUI integration includes the Perth dependency but has watermarking disabled by default to ensure maximum compatibility. Users can re-enable watermarking by modifying the code if needed, while maintaining the full quality and capabilities of the underlying TTS model.

<!-- MARKDOWN LINKS & IMAGES -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/diodiogod/ComfyUI_ChatterBox_SRT_Voice.svg?style=for-the-badge
[contributors-url]: https://github.com/diodiogod/ComfyUI_ChatterBox_SRT_Voice/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/diodiogod/ComfyUI_ChatterBox_SRT_Voice.svg?style=for-the-badge
[forks-url]: https://github.com/diodiogod/ComfyUI_ChatterBox_SRT_Voice/network/members
[stars-shield]: https://img.shields.io/github/stars/diodiogod/ComfyUI_ChatterBox_SRT_Voice.svg?style=for-the-badge
[stars-url]: https://github.com/diodiogod/ComfyUI_ChatterBox_SRT_Voice/stargazers
[issues-shield]: https://img.shields.io/github/issues/diodiogod/ComfyUI_ChatterBox_SRT_Voice.svg?style=for-the-badge
[issues-url]: https://github.com/diodiogod/ComfyUI_ChatterBox_SRT_Voice/issues
[license-shield]: https://img.shields.io/github/license/diodiogod/ComfyUI_ChatterBox_SRT_Voice.svg?style=for-the-badge
[license-url]: https://github.com/diodiogod/ComfyUI_ChatterBox_SRT_Voice/blob/master/LICENSE.txt

[version-shield]: https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdiodiogod%2FComfyUI_ChatterBox_SRT_Voice%2Fmain%2Fpyproject.toml&query=%24.project.version&label=Version&color=red&style=for-the-badge
[version-url]: pyproject.toml
