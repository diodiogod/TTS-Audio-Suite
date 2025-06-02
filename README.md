# ComfyUI_ChatterBox Integration
An unofficial ComfyUI custom node integration for [Resemble AI's ChatterBox](https://github.com/resemble-ai/chatterbox) - a state-of-the-art open-source Text-to-Speech (TTS) model with voice cloning capabilities.

## 🎯 Features

- **High-Quality TTS**: Production-grade speech synthesis consistently preferred over ElevenLabs in blind evaluations
- **Voice Cloning**: Clone any voice from a short audio sample (7-20 seconds)
- **Emotion Control**: First open-source TTS with emotion exaggeration control
- **MIT Licensed**: Completely open-source and free for commercial use
- **GPU Accelerated**: Optimized for CUDA with automatic device handling
- **ComfyUI Integration**: Seamless workflow integration with preview and save capabilities

## 🚀 Installation

### 1. Clone the Repository

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ShmuelRonen/ComfyUI_ChatterBox.git
cd ComfyUI_ChatterBox
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `torch`
- `torchaudio`
- `librosa`
- `safetensor`
- `huggingface_hub`
- `conformer`
- `perth`


### 3. Model Setup

ChatterBox requires several model files that need to be downloaded and placed in the correct directory.

#### ⚠️ Manual Download Required

**Note**: Automatic download is currently not working. You must download the models manually from Hugging Face.

1. **Create model directory:**
   ```bash
   mkdir -p ComfyUI/models/TTS/chatterbox
   ```

2. **Download model files from Hugging Face:**
   
   Visit: **https://huggingface.co/ResembleAI/chatterbox/tree/main**
   
   Download the following files and place them in `ComfyUI/models/TTS/chatterbox/`:
   
   - **ve.safetensors** (Voice encoder model)
   - **t3_cfg.safetensors** (T3 text-to-speech model) 
   - **s3gen.safetensors** (S3Gen speech generation model)
   - **tokenizer.json** (English text tokenizer)
   - **conds.pt** (Built-in voice conditionals)

   **Alternative download methods:**
   
   Using `wget`:
   ```bash
   cd ComfyUI/models/TTS/chatterbox
   
   wget https://huggingface.co/ResembleAI/chatterbox/resolve/main/ve.safetensors
   wget https://huggingface.co/ResembleAI/chatterbox/resolve/main/t3_cfg.safetensors
   wget https://huggingface.co/ResembleAI/chatterbox/resolve/main/s3gen.safetensors
   wget https://huggingface.co/ResembleAI/chatterbox/resolve/main/tokenizer.json
   wget https://huggingface.co/ResembleAI/chatterbox/resolve/main/conds.pt
   ```
   
   Using `curl`:
   ```bash
   cd ComfyUI/models/TTS/chatterbox
   
   curl -L -o ve.safetensors https://huggingface.co/ResembleAI/chatterbox/resolve/main/ve.safetensors
   curl -L -o t3_cfg.safetensors https://huggingface.co/ResembleAI/chatterbox/resolve/main/t3_cfg.safetensors
   curl -L -o s3gen.safetensors https://huggingface.co/ResembleAI/chatterbox/resolve/main/s3gen.safetensors
   curl -L -o tokenizer.json https://huggingface.co/ResembleAI/chatterbox/resolve/main/tokenizer.json
   curl -L -o conds.pt https://huggingface.co/ResembleAI/chatterbox/resolve/main/conds.pt
   ```

3. **Verify files:**
   ```
   ComfyUI/models/TTS/chatterbox/
   ├── ve.safetensors          # Voice encoder model (~94MB)
   ├── t3_cfg.safetensors      # T3 text-to-speech model (~674MB)
   ├── s3gen.safetensors       # S3Gen speech generation model (~1.06GB)
   ├── tokenizer.json          # English text tokenizer (~2MB)
   └── conds.pt               # Built-in voice conditionals (~107KB)
   ```
   
   **Total download size**: ~3GB

### 4. Restart ComfyUI

After installation and model download, restart ComfyUI to load the new custom nodes.


## 📖 Usage

### Basic Workflow

1. **Add ChatterBox Generate Node**
   - Find `ChatterBox Generate` in the node menu under `ChatterBox` category
   - This node generates speech from text

2. **Configure Inputs**
   - **text**: The text you want to convert to speech
   - **model_path**: Path to your ChatterBox models directory
   - **reference_audio** (optional): Audio file for voice cloning
   - **exaggeration**: Emotion intensity (0.0-1.0, default: 0.5)
   - **cfg_weight**: Classifier-free guidance weight (0.0-1.0, default: 0.5)
   - **temperature**: Sampling temperature (0.1-1.0, default: 0.8)

3. **Connect Output**
   - Connect the `AUDIO` output to `PreviewAudio` node to hear the result
   - The generated audio is automatically saved to `ComfyUI/output/audio/`

### Voice Cloning

To clone a specific voice:

1. **Prepare Reference Audio**
   - Use a clean audio file (7-20 seconds recommended)
   - Supported formats: WAV, MP3, FLAC
   - Good quality recording with minimal background noise

2. **Connect Reference Audio**
   - Use `LoadAudio` node to load your reference audio
   - Connect it to the `reference_audio` input of ChatterBox Generate

3. **Generate Speech**
   - The model will clone the voice characteristics from your reference audio
   - Adjust `exaggeration` parameter to control emotion intensity

### Example Workflow

```
[LoadAudio] → [ChatterBox Generate] → [PreviewAudio]
                      ↑
              [Text Input: "Hello world!"]
```

## ⚙️ Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `text` | string | - | - | Text to synthesize |
| `model_path` | string | - | auto | Path to ChatterBox models |
| `reference_audio` | AUDIO | - | None | Reference audio for voice cloning |
| `exaggeration` | float | 0.0-1.0 | 0.5 | Emotion exaggeration intensity |
| `cfg_weight` | float | 0.0-1.0 | 0.5 | Classifier-free guidance weight |
| `temperature` | float | 0.1-1.0 | 0.8 | Sampling randomness |

## 🔧 Advanced Configuration


### Device Selection

The node automatically detects and uses the best available device:
- **CUDA**: If NVIDIA GPU with CUDA is available
- **MPS**: If Apple Silicon Mac
- **CPU**: Fallback option (slower)

### Output Location

Generated audio files are saved to:
```
ComfyUI/output/audio/chatterbox_output_{timestamp}.wav
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Model files not found"**
   - Ensure models are manually downloaded to `ComfyUI/models/TTS/chatterbox/`
   - Download from: https://huggingface.co/ResembleAI/chatterbox/tree/main
   - Verify all 5 required files are present and complete
   - Check file sizes match expected values (see installation section)

2. **"CUDA out of memory"**
   - Reduce batch size or use shorter text
   - Switch to CPU mode: Set device to "cpu" in code
   - Close other GPU-intensive applications

3. **"Device mismatch errors"**
   - Restart ComfyUI to reload models
   - Ensure PyTorch CUDA version matches your GPU drivers

4. **"Audio format not supported"**
   - Use WAV, MP3, or FLAC for reference audio
   - Ensure audio file is not corrupted

5. **"Download errors"**
   - Automatic download is not supported - download manually
   - Use stable internet connection for large model files
   - Verify downloaded files are not corrupted (check file sizes)

### Performance Tips

- **GPU Recommended**: CUDA significantly faster than CPU
- **Short Reference Audio**: 7-20 seconds optimal for voice cloning
- **Clean Audio**: Better reference audio = better voice cloning
- **Text Length**: Longer texts may require more memory

## 🏗️ Technical Details

### Model Architecture

ChatterBox uses a multi-stage architecture:

1. **Voice Encoder (VE)**: Extracts speaker embeddings from reference audio
2. **T3 Model**: Text-to-speech conversion with conditioning
3. **S3Gen**: High-quality speech generation and vocoding
4. **Tokenizers**: Text and speech token processing

### Audio Processing

- **Input Sample Rate**: Automatically resampled to 16kHz and 24kHz for different components
- **Output Sample Rate**: 24kHz high-quality audio
- **Format**: 32-bit float WAV files
- **Channels**: Mono output

### Memory Requirements

- **GPU**: 4GB+ VRAM recommended for optimal performance
- **RAM**: 8GB+ system RAM
- **Storage**: ~3GB for model files

## 📊 Benchmarks

According to Resemble AI's evaluations:
- **63.75%** of evaluators preferred ChatterBox over ElevenLabs
- Trained on **500K hours** of high-quality data
- Supports **emotion exaggeration control** (first open-source TTS)
- **MIT licensed** for commercial use

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.


## 📄 License

This integration is licensed under MIT License. 

The underlying ChatterBox model is also MIT licensed by Resemble AI.

## 🙏 Credits

- **Resemble AI**: For creating the excellent ChatterBox TTS model
- **Original ChatterBox**: https://github.com/resemble-ai/chatterbox
- **Model Downloads**: https://huggingface.co/ResembleAI/chatterbox/tree/main
- **ComfyUI**: For the amazing workflow platform

**Disclaimer**: This is an unofficial integration. All credit for the ChatterBox model goes to Resemble AI. This project simply provides ComfyUI compatibility.


## 🔗 Links

- [Resemble AI ChatterBox](https://github.com/resemble-ai/chatterbox)
- [Model Downloads (Hugging Face)](https://huggingface.co/ResembleAI/chatterbox/tree/main) ⬅️ **Download models here**
- [ChatterBox Demo](https://resemble-ai.github.io/chatterbox_demopage/)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Resemble AI Official Site](https://www.resemble.ai/chatterbox/)

---

**Note**: The original ChatterBox model includes Resemble AI's Perth watermarking system for responsible AI usage. This ComfyUI integration includes the Perth dependency but has watermarking disabled by default to ensure maximum compatibility. Users can re-enable watermarking by modifying the code if needed, while maintaining the full quality and capabilities of the underlying TTS model.
