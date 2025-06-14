"""
ComfyUI ChatterBox Voice Extension
High-quality Text-to-Speech and Voice Conversion nodes using ResembleAI's ChatterboxTTS:
• 🎤 ChatterBox Voice TTS
• 📺 ChatterBox SRT Voice TTS
• 🔄 ChatterBox Voice Conversion
• 🎙️ ChatterBox Voice Capture
"""

# Import from the main nodes.py file (not the nodes package)
# Use importlib to avoid naming conflicts
import importlib.util
import os

# Get the path to the nodes.py file
nodes_py_path = os.path.join(os.path.dirname(__file__), "nodes.py")

# Load nodes.py as a module
spec = importlib.util.spec_from_file_location("nodes_main", nodes_py_path)
nodes_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes_module)

# Import node classes
ChatterboxTTSNode = nodes_module.ChatterboxTTSNode
ChatterboxVCNode = nodes_module.ChatterboxVCNode

# Import constants and utilities
IS_DEV = nodes_module.IS_DEV
VERSION = nodes_module.VERSION
SEPARATOR = nodes_module.SEPARATOR
VERSION_DISPLAY = nodes_module.VERSION_DISPLAY
find_chatterbox_models = nodes_module.find_chatterbox_models

# Import SRT node if available
try:
    ChatterboxSRTTTSNode = nodes_module.ChatterboxSRTTTSNode
    SRT_SUPPORT_AVAILABLE = nodes_module.SRT_SUPPORT_AVAILABLE
except AttributeError:
    SRT_SUPPORT_AVAILABLE = False
    ChatterboxSRTTTSNode = None

# Import Audio Recorder node (now loaded from nodes.py)
try:
    ChatterBoxVoiceCapture = nodes_module.ChatterBoxVoiceCapture
    AUDIO_RECORDER_AVAILABLE = True
except AttributeError:
    AUDIO_RECORDER_AVAILABLE = False
    ChatterBoxVoiceCapture = None

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ChatterBoxVoiceTTS": ChatterboxTTSNode,
    "ChatterBoxVoiceVC": ChatterboxVCNode,
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterBoxVoiceTTS": "🎤 ChatterBox Voice TTS",
    "ChatterBoxVoiceVC": "🔄 ChatterBox Voice Conversion",
}

# Add SRT node if available
if SRT_SUPPORT_AVAILABLE:
    NODE_CLASS_MAPPINGS["ChatterBoxSRTVoiceTTS"] = ChatterboxSRTTTSNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxSRTVoiceTTS"] = "📺 ChatterBox SRT Voice TTS"

# Add Audio Recorder if available
if AUDIO_RECORDER_AVAILABLE and ChatterBoxVoiceCapture is not None:
    NODE_CLASS_MAPPINGS["ChatterBoxVoiceCapture"] = ChatterBoxVoiceCapture
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxVoiceCapture"] = "🎙️ ChatterBox Voice Capture"

# Extension info
__version__ = VERSION_DISPLAY
__author__ = "ComfyUI ChatterBox Voice Extension"
__description__ = "Enhanced ChatterBox TTS/VC with integrated voice recording and smart audio capture"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Define web directory for JavaScript files
WEB_DIRECTORY = "./web"

# Print final initialization with ALL nodes list
print(f"🚀 ChatterBox Voice Extension {VERSION_DISPLAY} loaded with {len(NODE_DISPLAY_NAME_MAPPINGS)} nodes:")
for node in sorted(NODE_DISPLAY_NAME_MAPPINGS.values()):
    print(f"   • {node}")
print(SEPARATOR)