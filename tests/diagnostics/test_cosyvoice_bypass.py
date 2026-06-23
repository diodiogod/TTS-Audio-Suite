"""
Bypass test - save CosyVoice output DIRECTLY using torchaudio.
This tests if the issue is in CosyVoice model itself or in our wrapper layers.
"""
import sys
import os

# Add paths - Auto-detect based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
bundled_cosyvoice_path = os.path.join(project_root, "engines", "cosyvoice", "impl")
matcha_path = os.path.join(bundled_cosyvoice_path, "third_party", "Matcha-TTS")

if os.path.exists(bundled_cosyvoice_path) and bundled_cosyvoice_path not in sys.path:
    sys.path.insert(0, bundled_cosyvoice_path)
if os.path.exists(matcha_path) and matcha_path not in sys.path:
    sys.path.insert(0, matcha_path)

print(f"Using bundled CosyVoice from: {bundled_cosyvoice_path}")

import torch
import torchaudio
import soundfile as sf
import numpy as np

# Import CosyVoice AutoModel (official way)
from cosyvoice.cli.cosyvoice import AutoModel

# Model and voice paths - Accept from command line or use default
model_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\AiSymLink\TTS\CosyVoice\Fun-CosyVoice3-0.5B"
voice_path = os.path.join(project_root, "voices_examples", "David_Attenborough CC3.wav")
reference_text = "The first one who physical contact was with a female with her twins and she put a hand on the top of my head."

print(f"Loading model from: {model_path}")
print(f"Voice: {voice_path}")

# Load model exactly like official
cosyvoice = AutoModel(model_dir=model_path, fp16=True)
print(f"Model loaded! Sample rate: {cosyvoice.sample_rate}")

# Format prompt_text like official example
prompt_text = f"You are a helpful assistant.<|endofprompt|>{reference_text}"

# Use LONGER text to avoid the "too short" warning
test_text = """
Hello, this is a test of the CosyVoice three text to speech system. 
We are testing to verify that the audio output is clear and intelligible.
The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
Testing one two three four five six seven eight nine ten.
"""

print(f"TTS text length: {len(test_text)} chars")
print(f"Prompt text length: {len(reference_text)} chars")

# Create output directory
output_dir = os.path.join(project_root, "tests", "output")
os.makedirs(output_dir, exist_ok=True)

print("\n=== Generating audio directly ===")

# Generate and save DIRECTLY - no wrapper processing
all_audio = []
for i, output in enumerate(cosyvoice.inference_zero_shot(
    test_text.strip(),
    prompt_text,
    voice_path,
    stream=False
)):
    audio_chunk = output['tts_speech']
    print(f"  Chunk {i}: shape={audio_chunk.shape}, dtype={audio_chunk.dtype}, min={audio_chunk.min():.4f}, max={audio_chunk.max():.4f}")
    all_audio.append(audio_chunk)

# Combine all chunks
if all_audio:
    combined = torch.cat(all_audio, dim=-1)
    print(f"\nCombined: shape={combined.shape}, dtype={combined.dtype}")
    print(f"  min={combined.min():.4f}, max={combined.max():.4f}")
    print(f"  mean={combined.mean():.6f}, std={combined.std():.4f}")
    
    # Save using multiple methods to compare
    
    # Method 1: torchaudio (official way)
    wav_path1 = os.path.join(output_dir, "direct_test_torchaudio.wav")
    torchaudio.save(wav_path1, combined.cpu(), cosyvoice.sample_rate)
    print(f"\n✅ Saved via torchaudio: {wav_path1}")
    
    # Method 2: soundfile (alternative)
    wav_path2 = os.path.join(output_dir, "direct_test_soundfile.wav")
    audio_np = combined.squeeze().cpu().numpy()
    sf.write(wav_path2, audio_np, cosyvoice.sample_rate)
    print(f"✅ Saved via soundfile: {wav_path2}")
    
    print(f"\n👂 Please listen to BOTH files and compare!")
else:
    print("❌ No audio generated!")
