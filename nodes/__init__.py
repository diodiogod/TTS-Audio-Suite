from .model_loader import ChatterboxModelLoader
from .generate import ChatterboxGenerate

NODE_CLASS_MAPPINGS = {
    "ChatterboxModelLoader": ChatterboxModelLoader,
    "ChatterboxGenerate": ChatterboxGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterboxModelLoader": "Chatterbox Model Loader",
    "ChatterboxGenerate": "Chatterbox Generate",
}