"""
Direct test of CosyVoice3 using official API pattern.
Run this outside ComfyUI to verify the model works correctly.

Usage:
    python tests/test_cosyvoice_direct.py <model_path> [voice_path] [reference_text]

Example:
    python tests/test_cosyvoice_direct.py "D:\\AiSymLink\\TTS\\CosyVoice\\Fun-CosyVoice3-0.5B"
"""
import sys
import os

# Parse arguments
if len(sys.argv) < 2:
    print("ERROR: Model path required")
    print("Usage: python tests/test_cosyvoice_direct.py <model_path> [voice_path] [reference_text]")
    print('Example: python tests/test_cosyvoice_direct.py "D:\\AiSymLink\\TTS\\CosyVoice\\Fun-CosyVoice3-0.5B"')
    sys.exit(1)

model_path = sys.argv[1]

if not os.path.exists(model_path):
    print(f"ERROR: Model path does not exist: {model_path}")
    sys.exit(1)

# Add paths (matching unified_model_interface.py)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bundled_cosyvoice_path = os.path.join(project_root, "engines", "cosyvoice", "impl")
matcha_path = os.path.join(bundled_cosyvoice_path, "third_party", "Matcha-TTS")

if os.path.exists(bundled_cosyvoice_path) and bundled_cosyvoice_path not in sys.path:
    sys.path.insert(0, bundled_cosyvoice_path)
if os.path.exists(matcha_path) and matcha_path not in sys.path:
    sys.path.insert(0, matcha_path)

print(f"Using bundled CosyVoice from: {bundled_cosyvoice_path}")

import torchaudio

# Import CosyVoice AutoModel (official way)
from cosyvoice.cli.cosyvoice import AutoModel

print(f"Loading model from: {model_path}")

# Load model exactly like official example
cosyvoice = AutoModel(model_dir=model_path, fp16=True)

print(f"Model loaded! Sample rate: {cosyvoice.sample_rate}")

# Voice sample and reference text
if len(sys.argv) >= 3:
    voice_path = sys.argv[2]
else:
    voice_path = os.path.join(project_root, "voices_examples", "David_Attenborough CC3.wav")

if len(sys.argv) >= 4:
    reference_text = sys.argv[3]
else:
    reference_text = "The first one who physical contact was with a female with her twins and she put a hand on the top of my head."

print(f"\n=== Testing zero_shot mode ===")
print(f"Voice sample: {voice_path}")
print(f"Reference text: {reference_text}")

# Format prompt_text EXACTLY like official example
prompt_text = f"You are a helpful assistant.<|endofprompt|>{reference_text}"

print(f"\nFormatted prompt_text: {prompt_text}")

# Test text - keeping it simple
test_text = "Hello, this is a test of the CosyVoice three text to speech system."
test_text_with_tag = "<|en|>" + test_text

print(f"TTS text: {test_text}")
print(f"TTS text with tag: {test_text_with_tag}")

# Test 1: zero_shot without <|en|> tag
print("\n=== Test 1: zero_shot WITHOUT language tag ===")
output_path_1 = os.path.join(project_root, "tests", "output", "zero_shot_NO_TAG.wav")

for i, output in enumerate(cosyvoice.inference_zero_shot(
    test_text,
    prompt_text,
    voice_path,
    stream=False,
    text_frontend=False
)):
    print(f"  Chunk {i}: shape={output['tts_speech'].shape}")
    torchaudio.save(output_path_1, output['tts_speech'], cosyvoice.sample_rate)

print(f"Saved: {output_path_1}")

# Test 2: zero_shot WITH <|en|> tag
print("\n=== Test 2: zero_shot WITH <|en|> tag ===")
output_path_2 = os.path.join(project_root, "tests", "output", "zero_shot_WITH_TAG.wav")

for i, output in enumerate(cosyvoice.inference_zero_shot(
    test_text_with_tag,
    prompt_text,
    voice_path,
    stream=False,
    text_frontend=False
)):
    print(f"  Chunk {i}: shape={output['tts_speech'].shape}")
    torchaudio.save(output_path_2, output['tts_speech'], cosyvoice.sample_rate)

print(f"Saved: {output_path_2}")

# Test 3: cross_lingual WITH <|en|> tag (official way)
print("\n=== Test 3: cross_lingual WITH <|en|> tag (OFFICIAL) ===")
output_path_3 = os.path.join(project_root, "tests", "output", "cross_lingual_EN_TAG.wav")

for i, output in enumerate(cosyvoice.inference_cross_lingual(
    test_text_with_tag,
    voice_path,
    stream=False
)):
    print(f"  Chunk {i}: shape={output['tts_speech'].shape}")
    torchaudio.save(output_path_3, output['tts_speech'], cosyvoice.sample_rate)

print(f"Saved: {output_path_3}")

print(f"\nDONE - Compare the three audio files:")
print(f"  1. zero_shot no tag:    {output_path_1}")
print(f"  2. zero_shot with tag:  {output_path_2}")
print(f"  3. cross_lingual (official): {output_path_3}")
print(f"\nListen to all three and see which one works for English!")
