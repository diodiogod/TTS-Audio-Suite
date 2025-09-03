import os

paths_to_test = [
    "/mnt/j/stablediffusion1111s2/Data/Packages/ComfyUIPy129/ComfyUI/models/TTS/vibevoice/vibevoice-7B-Q8-gguf/model.gguf",
    "J:/stablediffusion1111s2/Data/Packages/ComfyUIPy129/ComfyUI/models/TTS/vibevoice/vibevoice-7B-Q8-gguf/model.gguf",
    "\\\\wsl.localhost\\Ubuntu\\mnt\\j\\stablediffusion1111s2\\Data\\Packages\\ComfyUIPy129\\ComfyUI\\models\\TTS\\vibevoice\\vibevoice-7B-Q8-gguf\\model.gguf"
]

for path in paths_to_test:
    exists = os.path.exists(path)
    print(f"Path: {path}")
    print(f"Exists: {exists}")
    if exists:
        print(f"Size: {os.path.getsize(path) / (1024*1024*1024):.1f} GB")
    print("-" * 80)