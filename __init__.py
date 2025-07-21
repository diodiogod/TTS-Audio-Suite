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

# Import F5-TTS node if available
try:
    F5TTSNode = nodes_module.F5TTSNode
    F5TTS_SUPPORT_AVAILABLE = nodes_module.F5TTS_SUPPORT_AVAILABLE
except AttributeError:
    F5TTS_SUPPORT_AVAILABLE = False
    F5TTSNode = None

# Import F5-TTS SRT node if available
try:
    F5TTSSRTNode = nodes_module.F5TTSSRTNode
    F5TTS_SRT_SUPPORT_AVAILABLE = nodes_module.F5TTS_SRT_SUPPORT_AVAILABLE
except AttributeError:
    F5TTS_SRT_SUPPORT_AVAILABLE = False
    F5TTSSRTNode = None

# Import F5-TTS Edit node if available
try:
    F5TTSEditNode = nodes_module.F5TTSEditNode
    F5TTS_EDIT_SUPPORT_AVAILABLE = nodes_module.F5TTS_EDIT_SUPPORT_AVAILABLE
except AttributeError:
    F5TTS_EDIT_SUPPORT_AVAILABLE = False
    F5TTSEditNode = None


# Import Audio Recorder node (now loaded from nodes.py)
try:
    ChatterBoxVoiceCapture = nodes_module.ChatterBoxVoiceCapture
    AUDIO_RECORDER_AVAILABLE = True
except AttributeError:
    AUDIO_RECORDER_AVAILABLE = False
    ChatterBoxVoiceCapture = None

# Import Audio Analyzer node if available
try:
    AudioAnalyzerNode = nodes_module.AudioAnalyzerNode
    AUDIO_ANALYZER_SUPPORT_AVAILABLE = nodes_module.AUDIO_ANALYZER_SUPPORT_AVAILABLE
except AttributeError:
    AUDIO_ANALYZER_SUPPORT_AVAILABLE = False
    AudioAnalyzerNode = None

# Import Audio Analyzer Options node if available
try:
    AudioAnalyzerOptionsNode = nodes_module.AudioAnalyzerOptionsNode
    AUDIO_ANALYZER_OPTIONS_SUPPORT_AVAILABLE = nodes_module.AUDIO_ANALYZER_OPTIONS_SUPPORT_AVAILABLE
except AttributeError:
    AUDIO_ANALYZER_OPTIONS_SUPPORT_AVAILABLE = False
    AudioAnalyzerOptionsNode = None

# Import F5-TTS Edit Options node if available
try:
    F5TTSEditOptionsNode = nodes_module.F5TTSEditOptionsNode
    F5TTS_EDIT_OPTIONS_SUPPORT_AVAILABLE = nodes_module.F5TTS_EDIT_OPTIONS_SUPPORT_AVAILABLE
except AttributeError:
    F5TTS_EDIT_OPTIONS_SUPPORT_AVAILABLE = False
    F5TTSEditOptionsNode = None

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ChatterBoxVoiceTTSDiogod": ChatterboxTTSNode,
    "ChatterBoxVoiceVCDiogod": ChatterboxVCNode,
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterBoxVoiceTTSDiogod": "🎤 ChatterBox Voice TTS (diogod)",
    "ChatterBoxVoiceVCDiogod": "🔄 ChatterBox Voice Conversion (diogod)",
}

# Add SRT node if available
if SRT_SUPPORT_AVAILABLE:
    NODE_CLASS_MAPPINGS["ChatterBoxSRTVoiceTTS"] = ChatterboxSRTTTSNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxSRTVoiceTTS"] = "📺 ChatterBox SRT Voice TTS"

# Add F5-TTS node if available
if F5TTS_SUPPORT_AVAILABLE and F5TTSNode is not None:
    NODE_CLASS_MAPPINGS["ChatterBoxF5TTSVoice"] = F5TTSNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxF5TTSVoice"] = "🎤 F5-TTS Voice Generation"

# Add F5-TTS SRT node if available
if F5TTS_SRT_SUPPORT_AVAILABLE and F5TTSSRTNode is not None:
    NODE_CLASS_MAPPINGS["ChatterBoxF5TTSSRTVoice"] = F5TTSSRTNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxF5TTSSRTVoice"] = "📺 F5-TTS SRT Voice Generation"

# Add F5-TTS Edit node if available
if F5TTS_EDIT_SUPPORT_AVAILABLE and F5TTSEditNode is not None:
    NODE_CLASS_MAPPINGS["ChatterBoxF5TTSEditVoice"] = F5TTSEditNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxF5TTSEditVoice"] = "👄 F5-TTS Speech Editor"


# Add Audio Recorder if available
if AUDIO_RECORDER_AVAILABLE and ChatterBoxVoiceCapture is not None:
    NODE_CLASS_MAPPINGS["ChatterBoxVoiceCaptureDiogod"] = ChatterBoxVoiceCapture
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxVoiceCaptureDiogod"] = "🎙️ ChatterBox Voice Capture (diogod)"

# Add Audio Analyzer if available
if AUDIO_ANALYZER_SUPPORT_AVAILABLE and AudioAnalyzerNode is not None:
    NODE_CLASS_MAPPINGS["AudioAnalyzerNode"] = AudioAnalyzerNode
    NODE_DISPLAY_NAME_MAPPINGS["AudioAnalyzerNode"] = "🌊 Audio Wave Analyzer"

# Add Audio Analyzer Options if available
if AUDIO_ANALYZER_OPTIONS_SUPPORT_AVAILABLE and AudioAnalyzerOptionsNode is not None:
    NODE_CLASS_MAPPINGS["AudioAnalyzerOptionsNode"] = AudioAnalyzerOptionsNode
    NODE_DISPLAY_NAME_MAPPINGS["AudioAnalyzerOptionsNode"] = "🔧 Audio Wave Analyzer Options"

# Add F5-TTS Edit Options if available
if F5TTS_EDIT_OPTIONS_SUPPORT_AVAILABLE and F5TTSEditOptionsNode is not None:
    NODE_CLASS_MAPPINGS["ChatterBoxF5TTSEditOptions"] = F5TTSEditOptionsNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxF5TTSEditOptions"] = "🔧 F5-TTS Edit Options"

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

# Report missing optional features
missing_features = []
if not F5TTS_SUPPORT_AVAILABLE:
    missing_features.append("F5-TTS nodes")
if not SRT_SUPPORT_AVAILABLE:
    missing_features.append("SRT nodes") 
if not AUDIO_ANALYZER_SUPPORT_AVAILABLE:
    missing_features.append("Audio Analyzer")
if not AUDIO_RECORDER_AVAILABLE:
    missing_features.append("Voice Recorder")

if missing_features:
    print(f"⚠️  Missing optional features: {', '.join(missing_features)}")
    print("   Install missing dependencies to enable all features")

print(SEPARATOR)