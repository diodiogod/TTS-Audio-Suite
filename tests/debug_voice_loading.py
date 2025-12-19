"""Debug script to test CosyVoice3 generation directly."""
import sys
import os

# Add project root
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Test reference text loading specifically
from utils.voice.discovery import load_voice_reference

voice_file = "voices_examples/Sophie_Anderson CC3.wav"
audio_path, reference_text = load_voice_reference(voice_file)

print("=== Voice Reference Test ===")
print(f"Voice file: {voice_file}")
print(f"Audio path from cache: {audio_path}")
print(f"Reference text from cache: {reference_text}")

# Now test the fallback path
if not audio_path:
    fallback_path = os.path.join(project_root, voice_file)
    print(f"\nTrying fallback path: {fallback_path}")
    print(f"Fallback exists: {os.path.exists(fallback_path)}")
    
    if os.path.exists(fallback_path):
        ref_txt_path = os.path.splitext(fallback_path)[0] + ".reference.txt"
        print(f"Reference txt path: {ref_txt_path}")
        print(f"Reference txt exists: {os.path.exists(ref_txt_path)}")
        
        if os.path.exists(ref_txt_path):
            with open(ref_txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            print(f"Reference content: [{content}]")
