import torch
import os
from pathlib import Path

from ..chatterbox.tts import ChatterboxTTS

try:
    import folder_paths
    MODELS_DIR = folder_paths.models_dir
except ImportError:
    MODELS_DIR = str(Path.home() / "ComfyUI" / "models")

CHATTERBOX_MODEL_DIR = os.path.join(MODELS_DIR, "TTS", "chatterbox")
os.makedirs(CHATTERBOX_MODEL_DIR, exist_ok=True)

_MODEL_CACHE = {}

class ChatterboxModelLoader:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("CHATTERBOX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "ChatterBox"

    def load_model(self, device):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        cache_key = f"chatterbox_{device}"
        
        if cache_key in _MODEL_CACHE:
            print(f"Using cached model on {device}")
            return (_MODEL_CACHE[cache_key],)
        
        # Check if manual model files exist
        model_path = Path(CHATTERBOX_MODEL_DIR)
        required_files = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]
        
        missing_files = []
        for file in required_files:
            if not (model_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            error_msg = f"Missing model files in {CHATTERBOX_MODEL_DIR}:\n"
            for file in missing_files:
                error_msg += f"  - {file}\n"
            error_msg += "\nPlease download these files manually from:\n"
            error_msg += "https://huggingface.co/ResembleAI/chatterbox/tree/main\n"
            error_msg += f"and place them in: {CHATTERBOX_MODEL_DIR}"
            
            print(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            print(f"Loading model from {CHATTERBOX_MODEL_DIR}")
            print("All required files found, loading model...")
            
            model = ChatterboxTTS.from_local(model_path, device=device)
            
            _MODEL_CACHE[cache_key] = model
            print(f"Model loaded successfully on {device}")
            
            return (model,)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure all model files are properly downloaded and not corrupted.")
            raise e