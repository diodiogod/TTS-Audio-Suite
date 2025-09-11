"""
ComfyUI Model Wrapper for TTS Audio Suite - Backward Compatibility Module

This module maintains backward compatibility while delegating to the new modular structure.
The actual implementation has been refactored into separate modules for maintainability.

Legacy 970-line monolith replaced with modular architecture:
- utils/models/comfyui_model_wrapper/
  ├── __init__.py                    # Main exports
  ├── base_wrapper.py               # Base ComfyUIModelWrapper class (~300 lines)
  ├── model_manager.py              # ComfyUITTSModelManager class (~250 lines)
  ├── cache_utils.py                # Cache invalidation utilities (~30 lines)
  └── engine_handlers/              # Engine-specific handling (~150 lines each)
      ├── __init__.py               # Handler factory
      ├── base_handler.py           # Abstract base class
      ├── generic_handler.py        # ChatterBox, F5-TTS, RVC, etc.
      ├── higgs_audio_handler.py    # CUDA graph management
      └── vibevoice_handler.py      # RAM cleanup and deletion

Each module is now ~150-300 lines, following the 500-600 line policy.
"""

# Import everything from the modular structure for backward compatibility
from .comfyui_model_wrapper import *

# Ensure global instance is available at module level
from .comfyui_model_wrapper import tts_model_manager