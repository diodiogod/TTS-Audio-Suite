# ComfyUI_ChatterBox
An unofficial ComfyUI custom node integration for High-quality Text-to-Speech and Voice Conversion nodes for ComfyUI using ResembleAI's ChatterboxTTS.

![image](https://github.com/user-attachments/assets/35639c75-8c00-4b81-a16c-be9567955db7)

## Features

🎤 **ChatterBox TTS** - Generate speech from text with optional voice cloning  
🔄 **ChatterBox VC** - Convert voice from one speaker to another  
⚡ **Fast & Quality** - Production-grade TTS that outperforms ElevenLabs  
🎭 **Emotion Control** - Unique exaggeration parameter for expressive speech  

> **Note:** There are multiple ChatterBox extensions available. This implementation focuses on simplicity and ComfyUI standards.  

## Installation

### 1. Install the Extension

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ShmuelRonen/ComfyUI_ChatterBox.git
```

### 2. Install ChatterboxTTS Package

**Copy the included package folders to your Python site-packages:**

**Windows Portable ComfyUI:**
```bash
cd D:\ComfyUI_windows\ComfyUI\custom_nodes\ComfyUI_ChatterBox
xcopy "put_contain_in_site_packages_folder\*" "..\..\..\python_embeded\Lib\site-packages\" /E /S
```

**WSL/Linux ComfyUI:**
```bash
cd ComfyUI/custom_nodes/ComfyUI_ChatterBox
cp -r put_contain_in_site_packages_folder/* ../../venv/lib/python3.11/site-packages/
```

**Other Python setups:**
```bash
# Find your site-packages location first:
python -c "import site; print(site.getsitepackages())"

# Then copy both folders:
cp -r put_contain_in_site_packages_folder/* /path/to/your/site-packages/
```

**This copies both required folders:**
- `chatterbox/` - The actual TTS package code
- `chatterbox_tts-0.1.1.dist-info/` - Package metadata for Python

### 3. Install Additional Dependencies

```bash
pip install -r requirements.txt
```

**Note:** `torch`, `torchaudio`, `numpy` should already be available in ComfyUI.

### 4. Download Models

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

**Manual download steps:**
1. Visit https://huggingface.co/ResembleAI/chatterbox/tree/main
2. Click each required file and download
3. Save all files to `ComfyUI/models/TTS/chatterbox/`
4. Folder should contain exactly 5 files as listed above

### 5. Restart ComfyUI

The ChatterBox nodes will appear in the **"ChatterBox"** category.

## Usage

### Text-to-Speech
1. Add **"ChatterBox Text-to-Speech"** node
2. Enter your text
3. Optionally connect reference audio for voice cloning
4. Adjust settings:
   - **Exaggeration**: Emotion intensity (0.25-2.0)
   - **Temperature**: Randomness (0.05-5.0)
   - **CFG Weight**: Guidance strength (0.0-1.0)
   
### ChatterBox TTS Text Limits
📝 No Official Hard Limit: Unlike some TTS systems (like OpenAI's TTS which has a 4096 character limit TTS model has a "hidden" 4096 characters limit - API - OpenAI Developer Community), ChatterBox TTS doesn't appear to have a documented hard character or word limit.

🔧 Practical Implementation: However, for optimal performance, the underlying model likely works best with shorter text segments.

### Voice Conversion  
1. Add **"ChatterBox Voice Conversion"** node
2. Connect source audio (voice to convert)
3. Connect target audio (voice style to copy)

## Settings Guide

**General Use:**
- `exaggeration=0.5`, `cfg_weight=0.5` (default settings work well)

**Expressive Speech:**
- Lower `cfg_weight` (~0.3) + higher `exaggeration` (~0.7)
- Higher exaggeration speeds up speech; lower CFG slows it down

## Installation Summary

1. **Clone extension** → `git clone https://github.com/your-username/ComfyUI_ChatterBox.git`
2. **Copy package** → Copy folders from `put_contain_in_site_packages_folder/` to site-packages
3. **Download models** → Get 5 files from HuggingFace to `ComfyUI/models/TTS/chatterbox/`
4. **Restart ComfyUI** → Nodes appear in "ChatterBox" category

**Why This Approach?**
- **No pip conflicts** - Avoids dependency issues with ComfyUI
- **Universal** - Works on Windows portable, WSL, Linux, conda, etc.
- **Offline** - No downloads during installation
- **Simple** - Just copy folders, no complex scripts

## Why Two Folders?

**`chatterbox/`** - Contains the actual Python code for the TTS engine  
**`chatterbox_tts-0.1.1.dist-info/`** - Contains package metadata (version, dependencies, etc.)  

Python's import system needs both folders to properly recognize and load the package. Missing either folder can cause import errors or version conflicts.

## Troubleshooting

**"ChatterboxTTS not available"** → Copy the package folders:
```bash
# Check if both folders exist in your site-packages:
# chatterbox/
# chatterbox_tts-0.1.1.dist-info/
```

**"No module named 'chatterbox'"** → Verify both folders copied correctly:
```bash
# Windows Portable
dir "python_embeded\Lib\site-packages\chatterbox"
dir "python_embeded\Lib\site-packages\chatterbox_tts-0.1.1.dist-info"

# WSL/Linux
ls venv/lib/python3.11/site-packages/chatterbox
ls venv/lib/python3.11/site-packages/chatterbox_tts-0.1.1.dist-info
```

**Models not found** → Download manually to `ComfyUI/models/TTS/chatterbox/`

**Wrong Python version** → Make sure you're copying to the same Python environment that ComfyUI uses

**Permission errors** → Run terminal as administrator (Windows) or use `sudo` (Linux)

## License

MIT License - Same as ChatterboxTTS

## Credits

- **ResembleAI** for ChatterboxTTS
- **ComfyUI** team for the amazing framework


## 🔗 Links

- [Resemble AI ChatterBox](https://github.com/resemble-ai/chatterbox)
- [Model Downloads (Hugging Face)](https://huggingface.co/ResembleAI/chatterbox/tree/main) ⬅️ **Download models here**
- [ChatterBox Demo](https://resemble-ai.github.io/chatterbox_demopage/)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Resemble AI Official Site](https://www.resemble.ai/chatterbox/)

---

**Note**: The original ChatterBox model includes Resemble AI's Perth watermarking system for responsible AI usage. This ComfyUI integration includes the Perth dependency but has watermarking disabled by default to ensure maximum compatibility. Users can re-enable watermarking by modifying the code if needed, while maintaining the full quality and capabilities of the underlying TTS model.
