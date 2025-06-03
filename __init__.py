"""
ComfyUI ChatterBox Extension
High-quality Text-to-Speech and Voice Conversion nodes using ResembleAI's ChatterboxTTS
Now with integrated audio recording and volume control!
"""

# Import ChatterBox TTS and VC nodes
from .nodes import ChatterboxTTSNode, ChatterboxVCNode

# Import Audio Recorder node (with error handling)
try:
    from .nodes_audio_recorder import ChatterBoxVoiceCapture
    AUDIO_RECORDER_AVAILABLE = True
    print("✅ Audio Recorder with Volume Control loaded!")
except ImportError as e:
    print(f"⚠️  Audio Recorder not available: {e}")
    print("📋 Install sounddevice: pip install sounddevice")
    AUDIO_RECORDER_AVAILABLE = False

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ChatterboxTTS": ChatterboxTTSNode,
    "ChatterboxVC": ChatterboxVCNode,
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterboxTTS": "ChatterBox Text-to-Speech",
    "ChatterboxVC": "ChatterBox Voice Conversion",
}

# Add Audio Recorder if available
if AUDIO_RECORDER_AVAILABLE:
    NODE_CLASS_MAPPINGS["ChatterBoxVoiceCapture"] = ChatterBoxVoiceCapture
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxVoiceCapture"] = "🎙️ ChatterBox Voice Capture"

# Extension info
__version__ = "1.1.0"
__author__ = "ComfyUI ChatterBox Extension"
__description__ = "High-quality TTS and VC nodes using ResembleAI's ChatterboxTTS with integrated audio recording"

# Check for required packages
print("🎯 Loading ChatterBox Extension...")

try:
    import chatterbox
    print("✅ ChatterBox TTS package found!")
except ImportError:
    print("❌ ChatterBox package not found!")
    print("📋 Copy folders from put_contain_in_site_packages_folder/ to your site-packages")
    print("📖 See README.md for installation instructions")

try:
    import sounddevice
    print("✅ SoundDevice available - Audio recording enabled!")
except ImportError:
    print("⚠️  SoundDevice not found - Audio recording disabled")
    print("📋 Run: pip install sounddevice")

# Summary
total_nodes = len(NODE_CLASS_MAPPINGS)
print(f"🚀 ChatterBox Extension v{__version__} loaded with {total_nodes} nodes:")
for node_class, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"   • {display_name}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Define web directory for JavaScript files
WEB_DIRECTORY = "./web"