"""
ComfyUI ChatterBox Voice Extension
High-quality Text-to-Speech and Voice Conversion nodes using ResembleAI's ChatterboxTTS
Enhanced with integrated audio recording and smart voice capture!
"""

# Import ChatterBox TTS and VC nodes
from .nodes import ChatterboxTTSNode, ChatterboxVCNode

# Import SRT node if available
try:
    from .nodes import ChatterboxSRTTTSNode, SRT_SUPPORT_AVAILABLE
    if SRT_SUPPORT_AVAILABLE:
        print("✅ SRT TTS node available!")
    else:
        print("⚠️  SRT TTS node not available - missing dependencies")
except ImportError as e:
    print(f"⚠️  SRT TTS node import failed: {e}")
    SRT_SUPPORT_AVAILABLE = False

# Import Audio Recorder node (with error handling)
try:
    from .nodes_audio_recorder import ChatterBoxVoiceCapture
    AUDIO_RECORDER_AVAILABLE = True
    print("✅ Audio Recorder with Volume Control loaded!")
except ImportError as e:
    print(f"⚠️  Audio Recorder not available: {e}")
    print("📋 Install sounddevice: pip install sounddevice")
    AUDIO_RECORDER_AVAILABLE = False

# Node class mappings for ComfyUI - UNIQUE NAMES TO AVOID CONFLICTS
NODE_CLASS_MAPPINGS = {
    "ChatterBoxVoiceTTS": ChatterboxTTSNode,
    "ChatterBoxVoiceVC": ChatterboxVCNode,
}

# Display names for the UI - CLEAR BRANDING
NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterBoxVoiceTTS": "🎤 ChatterBox Voice TTS",
    "ChatterBoxVoiceVC": "🔄 ChatterBox Voice Conversion",
}

# Add SRT node if available
if SRT_SUPPORT_AVAILABLE:
    NODE_CLASS_MAPPINGS["ChatterBoxSRTVoiceTTS"] = ChatterboxSRTTTSNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxSRTVoiceTTS"] = "📺 ChatterBox SRT Voice TTS"

# Add Audio Recorder if available - UNIQUE NAME
if AUDIO_RECORDER_AVAILABLE:
    NODE_CLASS_MAPPINGS["ChatterBoxVoiceCapture"] = ChatterBoxVoiceCapture
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxVoiceCapture"] = "🎙️ ChatterBox Voice Capture"

# Extension info - UPDATED BRANDING
__version__ = "1.1.0"
__author__ = "ComfyUI ChatterBox Voice Extension"
__description__ = "Enhanced ChatterBox TTS/VC with integrated voice recording and smart audio capture"

# Check for required packages
print("🎯 Loading ChatterBox Voice Extension...")

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
print(f"🚀 ChatterBox Voice Extension v{__version__} loaded with {total_nodes} nodes:")
for node_class, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"   • {display_name}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Define web directory for JavaScript files
WEB_DIRECTORY = "./web"