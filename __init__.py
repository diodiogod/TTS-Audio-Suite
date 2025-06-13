"""
ComfyUI ChatterBox Voice Extension
High-quality Text-to-Speech and Voice Conversion nodes using ResembleAI's ChatterboxTTS:
• 🎤 ChatterBox Voice TTS
• 📺 ChatterBox SRT Voice TTS
• 🔄 ChatterBox Voice Conversion
• 🎙️ ChatterBox Voice Capture
"""

# Import from nodes.py
from .nodes import (
    ChatterboxTTSNode, ChatterboxVCNode, IS_DEV, VERSION,
    SEPARATOR, VERSION_DISPLAY, find_chatterbox_models
)

# Import SRT node if available
try:
    from .nodes import ChatterboxSRTTTSNode, SRT_SUPPORT_AVAILABLE
except ImportError:
    SRT_SUPPORT_AVAILABLE = False

# Import Audio Recorder node (with error handling)
try:
    from .nodes_audio_recorder import ChatterBoxVoiceCapture
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

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
if AUDIO_RECORDER_AVAILABLE:
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