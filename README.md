<a id="readme-top"></a>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->

<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
[![Contributors][contributors-shield]][contributors-url]

-->

[![](https://dcbadge.limes.pink/api/server/EwKE8KBDqD)](https://discord.gg/EwKE8KBDqD)
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Forks][forks-shield]][forks-url]
[![Dynamic TOML Badge][version-shield]][version-url]

# ComfyUI ChatterBox SRT Voice (diogod) v2.0.2

*This is a refactored node, originally created by [ShmuelRonen](https://github.com/ShmuelRonen/ComfyUI_ChatterBox_Voice).*

An unofficial ComfyUI custom node integration for High-quality Text-to-Speech and Voice Conversion nodes for ComfyUI using ResembleAI's ChatterboxTTS with unlimited text length and with a node specially made for dealing with SRT timings.

NEW!: SRT Timing and TTS Node
![SRT Node Screenshot Placeholder](images/srt.png)

The **"ChatterBox SRT Voice TTS"** node allows TTS generation by processing SRT content (SubRip Subtitle) files, ensuring precise timing and synchronization with your audio.

### [YouTube Video](https://youtu.be/VyOawMrCB1g?si=7BubljRhsudGqG3s)

<a href="https://youtu.be/VyOawMrCB1g?si=7BubljRhsudGqG3">
  <img src="https://img.youtube.com/vi/VyOawMrCB1g/maxresdefault.jpg" width="400">
</a>

<details>

Key Features:

* **SRT style Processing**: uses SRT style to generate TTS, aligning audio with subtitle timings.
* **`smart_natural` Timing Mode**: Features flexible shifting logic that intelligently considers "room" in subsequent segments, preventing overlaps and ensuring natural speech flow.
* **`Adjusted_SRT` Output**: Provides actual timings for generated audio, allowing for accurate post-processing and integration.
* **Segment-Level Caching**: Only regenerates modified segments, significantly speeding up workflows. Timing-only changes do not trigger regeneration, optimizing resource usage.

For more detailed technical information, refer to the [SRT_IMPLEMENTATION.md](SRT_IMPLEMENTATION.md) file.

</details>

ChatterBox Text to Speech: 
![image](https://github.com/user-attachments/assets/4197818c-8093-4da4-abd5-577943ac902c)

NEW: Audio capture node
![image](https://github.com/user-attachments/assets/701c219b-12ff-4567-b414-e58560594ffe)

## Features

🎤 **ChatterBox TTS** - Generate speech from text with optional voice cloning
🔄 **ChatterBox VC** - Convert voice from one speaker to another
🎙️ **ChatterBox Voice Capture** - Record voice input with smart silence detection
⚡ **Fast & Quality** - Production-grade TTS that outperforms ElevenLabs
🎭 **Emotion Control** - Unique exaggeration parameter for expressive speech
📝 **Enhanced Chunking** - Intelligent text splitting for long content with multiple combination methods
📦 **Self-Contained** - Bundled ChatterBox for zero-installation-hassle experience
🎵 **Advanced Audio Processing** - Optional FFmpeg support for premium audio quality with graceful fallback

> **Note:** There are multiple ChatterBox extensions available. This implementation focuses on simplicity, ComfyUI standards, and enhanced text processing capabilities.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/diodiogod/ComfyUI_ChatterBox_SRT_Voice.git
```

#### 2.3. Install Additional Dependencies

Some dependencies, particularly `s3tokenizer`, can occasionally cause installation issues on certain Python setups (e.g., Python 3.10, sometimes used by tools like Stability Matrix).
Add comment
More actions

To minimize potential problems, it's highly recommended to first ensure your core packaging tools are up-to-date in your ComfyUI's virtual environment:

```bash
python -m pip install --upgrade pip setuptools wheel
```

After running the command above, install the node's specific requirements:

```bash
pip install -r requirements.txt
```

#### 2.4. Optional: Install FFmpeg for Enhanced Audio Processing

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

#### 2.4. Download Models

**Download the ChatterboxTTS models** and place them in:

```
ComfyUI/models/TTS/chatterbox/
```

**Required files:**

- `conds.pt` (105 KB)
- `s3gen.pt` (~1 GB)
- `t3_cfg.pt` (~1 GB)  
- `tokenizer.json` (25 KB)
- `ve.pt` (5.5 MB)

**Download from:** https://huggingface.co/ResembleAI/chatterbox/tree/main

### 3. Install Voice Recording Dependencies (Optional)

```bash
pip install sounddevice
```

### 4. Restart ComfyUI

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

### Voice Conversion

1. Add **"🔄 ChatterBox Voice Conversion"** node
2. Connect source audio (voice to convert)
3. Connect target audio (voice style to copy)

### Workflow Examples

**Long Text with Smart Chunking:**

```
Text Input (2000+ chars) → ChatterBox TTS (chunking enabled) → PreviewAudio
```

**Voice Cloning with Recording:**

```
🎤 Voice Capture → ChatterBox TTS (reference_audio) → PreviewAudio
```

**Voice Conversion Pipeline:**

```
🎤 Voice Capture (source) → ChatterBox VC ← 🎤 Voice Capture (target)
```

**Complete Advanced Pipeline:**

```
Long Text Input → ChatterBox TTS (with voice reference) → PreviewAudio
                ↘ ChatterBox VC ← 🎤 Target Voice Recording
```

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
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

[version-shield]: https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdiodiogod%2FComfyUI_ChatterBox_SRT_Voice%2Fmain%2Fpyproject.toml&query=%24.project.version&label=Version&color=red&style=for-the-badge
[version-url]: pyproject.toml
