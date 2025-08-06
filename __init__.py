"""
TTS Audio Suite - Universal multi-engine TTS extension for ComfyUI
Unified architecture supporting ChatterBox, F5-TTS, and future engines like RVC:
• 🎤 TTS Text (unified text-to-speech)
• 📺 TTS SRT (unified SRT subtitle timing)  
• 🔄 Voice Changer (unified voice conversion)
• ⚙️ Engine nodes (ChatterBox, F5-TTS)
• 🎭 Character Voices (voice reference management)
"""

# Import from the main nodes.py file which handles the new unified architecture
import importlib.util
import os

# Get the path to the nodes.py file
nodes_py_path = os.path.join(os.path.dirname(__file__), "nodes.py")

# Load nodes.py as a module
spec = importlib.util.spec_from_file_location("nodes_main", nodes_py_path)
nodes_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes_module)

# Import constants and utilities
IS_DEV = nodes_module.IS_DEV
VERSION = nodes_module.VERSION
SEPARATOR = nodes_module.SEPARATOR
VERSION_DISPLAY = nodes_module.VERSION_DISPLAY

# The new unified architecture handles all node registration in nodes.py
# Just import the mappings that nodes.py creates
NODE_CLASS_MAPPINGS = nodes_module.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = nodes_module.NODE_DISPLAY_NAME_MAPPINGS

# Extension info
__version__ = VERSION_DISPLAY
__author__ = "TTS Audio Suite"
__description__ = "Universal multi-engine TTS extension for ComfyUI with unified architecture supporting ChatterBox, F5-TTS, and future engines like RVC"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Define web directory for JavaScript files
WEB_DIRECTORY = "./web"

# nodes.py already handles all the startup output and status reporting