"""Test if model path exists."""
import os

model_path = r"C:\_stability_matrix\Data\Models\TTS\CosyVoice\Fun-CosyVoice3-0.5B"

print(f"Testing path: {model_path}")
print(f"Path exists: {os.path.exists(model_path)}")
print(f"Is directory: {os.path.isdir(model_path)}")

if os.path.exists(model_path):
    files = os.listdir(model_path)
    print(f"Files in dir: {len(files)}")
    for f in files[:5]:
        print(f"  - {f}")
