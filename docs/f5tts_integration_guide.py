#!/usr/bin/env python3
"""
F5-TTS Integration Guide for ChatterBox Voice
Complete setup and usage instructions for F5-TTS integration
"""

# ===== INSTALLATION GUIDE =====

"""
F5-TTS Installation Steps:

1. Install F5-TTS from source (recommended):
   ```bash
   git clone https://github.com/SWivid/F5-TTS.git
   cd F5-TTS
   pip install -e .
   ```

2. Install F5-TTS specific dependencies:
   ```bash
   pip install -r requirements_f5tts.txt
   ```

3. Optional: Place F5-TTS models in ComfyUI models directory:
   ```
   ComfyUI/models/F5-TTS/
   ├── F5TTS_Base/
   │   ├── model_1200000.safetensors
   │   └── vocab.txt
   ├── F5TTS_v1_Base/
   │   ├── model_1250000.safetensors
   │   └── vocab.txt
   └── E2TTS_Base/
       ├── model_1200000.safetensors
       └── vocab.txt
   ```

4. Restart ComfyUI to load F5-TTS nodes
"""

# ===== USAGE EXAMPLES =====

# Basic F5-TTS Usage
f5tts_basic_example = {
    "text": "Hello! This is F5-TTS integrated with ChatterBox Voice. It provides high-quality text-to-speech with voice cloning capabilities.",
    "ref_text": "This is the reference text that matches the reference audio for voice cloning.",
    "reference_audio": "<audio_input>",  # ComfyUI audio input
    "model": "F5TTS_Base",
    "device": "auto",
    "temperature": 0.8,
    "speed": 1.0,
    "target_rms": 0.1,
    "nfe_step": 32,
    "cfg_strength": 2.0,
    "enable_chunking": True,
    "max_chars_per_chunk": 400,
    "chunk_combination_method": "auto"
}

# Long Text Processing with Chunking
f5tts_long_text_example = {
    "text": """This is a very long text that will be automatically chunked by the F5-TTS system. 
    The chunking system intelligently splits the text while preserving sentence boundaries and 
    ensuring natural flow. Each chunk is processed separately and then combined using the 
    specified method to create seamless audio output.""",
    "ref_text": "This is my reference voice sample.",
    "reference_audio": "<audio_input>",
    "model": "F5TTS_Base",
    "enable_chunking": True,
    "max_chars_per_chunk": 300,
    "chunk_combination_method": "crossfade",
    "silence_between_chunks_ms": 100
}

# Multi-language Support
f5tts_multilingual_examples = {
    "german": {
        "text": "Hallo! Dies ist F5-TTS mit deutscher Sprachunterstützung.",
        "ref_text": "Dies ist meine deutsche Referenzstimme.",
        "model": "F5-DE"
    },
    "spanish": {
        "text": "¡Hola! Este es F5-TTS con soporte para español.",
        "ref_text": "Esta es mi voz de referencia en español.",
        "model": "F5-ES"
    },
    "french": {
        "text": "Bonjour! Ceci est F5-TTS avec support français.",
        "ref_text": "Ceci est ma voix de référence française.",
        "model": "F5-FR"
    },
    "japanese": {
        "text": "こんにちは！これは日本語対応のF5-TTSです。",
        "ref_text": "これは日本語のリファレンス音声です。",
        "model": "F5-JP"
    }
}

# ===== INTEGRATION CODE SNIPPETS =====

# For developers extending the F5-TTS integration:

def example_f5tts_usage():
    """Example of using F5-TTS in custom code"""
    from chatterbox.f5tts import ChatterBoxF5TTS
    
    # Initialize F5-TTS
    f5tts = ChatterBoxF5TTS.from_pretrained("cuda", "F5TTS_Base")
    
    # Generate speech
    audio = f5tts.generate(
        text="Hello world!",
        ref_audio_path="reference.wav",
        ref_text="This is reference text",
        temperature=0.8,
        speed=1.0
    )
    
    return audio

def example_model_discovery():
    """Example of discovering available F5-TTS models"""
    from chatterbox.f5tts.f5tts import find_f5tts_models, get_f5tts_models
    
    # Find model paths
    model_paths = find_f5tts_models()
    print("Available model paths:", model_paths)
    
    # Get model names
    model_names = get_f5tts_models()
    print("Available models:", model_names)

def example_error_handling():
    """Example of proper error handling with F5-TTS"""
    try:
        from chatterbox.f5tts import ChatterBoxF5TTS, F5TTS_AVAILABLE
        
        if not F5TTS_AVAILABLE:
            print("F5-TTS not available - please install dependencies")
            return None
        
        f5tts = ChatterBoxF5TTS.from_pretrained("cuda", "F5TTS_Base")
        
        # Your F5-TTS code here
        
    except ImportError as e:
        print(f"F5-TTS import error: {e}")
        return None
    except Exception as e:
        print(f"F5-TTS error: {e}")
        return None

# ===== TROUBLESHOOTING =====

"""
Common Issues and Solutions:

1. F5-TTS not available:
   - Install F5-TTS from source: git clone https://github.com/SWivid/F5-TTS.git
   - Install dependencies: pip install -r requirements_f5tts.txt

2. Model loading errors:
   - Check model files are in correct format (.safetensors or .pt)
   - Verify vocab.txt file exists alongside model file
   - Try different model variants if one fails

3. Audio quality issues:
   - Ensure reference audio is high quality (24kHz recommended)
   - Reference text should accurately match reference audio
   - Adjust temperature and cfg_strength parameters

4. Memory issues:
   - Enable chunking for long texts
   - Reduce max_chars_per_chunk
   - Use CPU instead of GPU if needed

5. Performance optimization:
   - Use local models instead of downloading
   - Enable GPU acceleration if available
   - Use appropriate chunk sizes for your hardware
"""

# ===== CONFIGURATION =====

# F5-TTS Configuration Options
F5TTS_CONFIG = {
    # Model Selection
    "available_models": [
        "F5TTS_Base",           # English base model
        "F5TTS_v1_Base",        # English v1 model
        "E2TTS_Base",           # E2-TTS model
        "F5-DE",                # German model
        "F5-ES",                # Spanish model
        "F5-FR",                # French model
        "F5-JP",                # Japanese model
        "F5-IT",                # Italian model
        "F5-TH",                # Thai model
    ],
    
    # Audio Settings
    "sample_rate": 24000,
    "audio_format": "float32",
    
    # Generation Parameters
    "default_temperature": 0.8,
    "default_speed": 1.0,
    "default_target_rms": 0.1,
    "default_nfe_step": 32,
    "default_cfg_strength": 2.0,
    
    # Chunking Settings
    "default_chunk_size": 400,
    "chunk_methods": ["auto", "concatenate", "silence_padding", "crossfade"],
    
    # Model Paths
    "comfyui_models_dir": "ComfyUI/models/F5-TTS/",
    "fallback_to_huggingface": True,
}

if __name__ == "__main__":
    print("F5-TTS Integration Guide")
    print("=" * 50)
    print("This guide contains setup instructions and usage examples for F5-TTS integration.")
    print("Please read the comments in this file for detailed information.")
    print("=" * 50)