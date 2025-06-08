# ComfyUI_ChatterBox_Voice
An unofficial ComfyUI custom node integration for High-quality Text-to-Speech and Voice Conversion nodes for ComfyUI using ResembleAI's ChatterboxTTS with unlimited text length!!!.

![image](https://github.com/user-attachments/assets/4197818c-8093-4da4-abd5-577943ac902c)

NEW: Audio capture node
![image](https://github.com/user-attachments/assets/701c219b-12ff-4567-b414-e58560594ffe)

NEW: SRT Timing and TTS Node
![SRT Node Screenshot Placeholder](images/srt.png)

The **"ChatterBox SRT Voice TTS"** node allows TTS generation by processing SRT content (SubRip Subtitle) files, ensuring precise timing and synchronization with your audio.

Key Features:
*   **SRT style Processing**: uses SRT style to generate TTS, aligning audio with subtitle timings.
*   **`smart_natural` Timing Mode**: Features flexible shifting logic that intelligently considers "room" in subsequent segments, preventing overlaps and ensuring natural speech flow.
*   **`Adjusted_SRT` Output**: Provides actual timings for generated audio, allowing for accurate post-processing and integration.
*   **Segment-Level Caching**: Only regenerates modified segments, significantly speeding up workflows. Timing-only changes do not trigger regeneration, optimizing resource usage.

For more detailed technical information, refer to the [SRT_IMPLEMENTATION.md](SRT_IMPLEMENTATION.md) file.

## Features

🎤 **ChatterBox TTS** - Generate speech from text with optional voice cloning  
🔄 **ChatterBox VC** - Convert voice from one speaker to another  
🎙️ **ChatterBox Voice Capture** - Record voice input with smart silence detection  
⚡ **Fast & Quality** - Production-grade TTS that outperforms ElevenLabs  
🎭 **Emotion Control** - Unique exaggeration parameter for expressive speech  
📝 **Enhanced Chunking** - Intelligent text splitting for long content with multiple combination methods  
📦 **Self-Contained** - Bundled ChatterBox for zero-installation-hassle experience  

> **Note:** There are multiple ChatterBox extensions available. This implementation focuses on simplicity, ComfyUI standards, and enhanced text processing capabilities.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ShmuelRonen/ComfyUI_ChatterBox.git
```

**Expected folder structure for bundled approach:**
```
ComfyUI_ChatterBox_Voice/
├── __init__.py
├── nodes.py
├── nodes_audio_recorder.py
├── chatterbox/
├── web/        

├── models/                  # ← Models bundled here (optional)
│   └── chatterbox/
│       ├── conds.pt
│       ├── s3gen.pt
│       ├── t3_cfg.pt
│       ├── tokenizer.json
│       └── ve.pt
└── README.md
```


#### 2.3. Install Additional Dependencies

```bash
pip install -r requirements.txt
```

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
- **Diogod** for the SRT Timing and TTS Node implementation

## 🔗 Links

- [Resemble AI ChatterBox](https://github.com/resemble-ai/chatterbox)
- [Model Downloads (Hugging Face)](https://huggingface.co/ResembleAI/chatterbox/tree/main) ⬅️ **Download models here**
- [ChatterBox Demo](https://resemble-ai.github.io/chatterbox_demopage/)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Resemble AI Official Site](https://www.resemble.ai/chatterbox/)

---

**Note**: The original ChatterBox model includes Resemble AI's Perth watermarking system for responsible AI usage. This ComfyUI integration includes the Perth dependency but has watermarking disabled by default to ensure maximum compatibility. Users can re-enable watermarking by modifying the code if needed, while maintaining the full quality and capabilities of the underlying TTS model.
