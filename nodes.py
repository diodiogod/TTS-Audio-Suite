# Version and constants
VERSION = "4.0.0"
IS_DEV = False  # Set to False for release builds
VERSION_DISPLAY = f"v{VERSION}" + (" (dev)" if IS_DEV else "")
SEPARATOR = "=" * 70

"""
TTS Audio Suite - Universal multi-engine TTS extension for ComfyUI
Unified architecture supporting ChatterBox, F5-TTS, and future engines like RVC
Features modular engine adapters, character voice management, and comprehensive audio processing
"""

import warnings
warnings.filterwarnings('ignore', message='.*PerthNet.*')
warnings.filterwarnings('ignore', message='.*LoRACompatibleLinear.*')
warnings.filterwarnings('ignore', message='.*requires authentication.*')

import os
import folder_paths

# Import unified node implementations
import sys
import os
import importlib.util

# Add current directory to path for absolute imports
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import nodes using direct file loading to avoid package path issues
def load_node_module(module_name, file_name):
    """Load a node module from the nodes directory"""
    module_path = os.path.join(current_dir, "nodes", file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    # Add to sys.modules to allow internal imports within the module
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load unified nodes
print("🔧 Loading TTS Audio Suite unified architecture...")

# Load engine nodes
try:
    chatterbox_engine_module = load_node_module("chatterbox_engine_node", "engines/chatterbox_engine_node.py")
    ChatterBoxEngineNode = chatterbox_engine_module.ChatterBoxEngineNode
    CHATTERBOX_ENGINE_AVAILABLE = True
    print("✓ ChatterBox TTS Engine loaded")
except Exception as e:
    print(f"❌ ChatterBox Engine failed: {e}")
    CHATTERBOX_ENGINE_AVAILABLE = False

try:
    f5tts_engine_module = load_node_module("f5tts_engine_node", "engines/f5tts_engine_node.py")
    F5TTSEngineNode = f5tts_engine_module.F5TTSEngineNode
    F5TTS_ENGINE_AVAILABLE = True
    print("✓ F5 TTS Engine loaded")
except Exception as e:
    print(f"❌ F5 TTS Engine failed: {e}")
    F5TTS_ENGINE_AVAILABLE = False

# Load shared nodes
try:
    character_voices_module = load_node_module("character_voices_node", "shared/character_voices_node.py")
    CharacterVoicesNode = character_voices_module.CharacterVoicesNode
    CHARACTER_VOICES_AVAILABLE = True
    print("✓ Character Voices node loaded")
except Exception as e:
    print(f"❌ Character Voices failed: {e}")
    CHARACTER_VOICES_AVAILABLE = False

# Load unified nodes
try:
    unified_text_module = load_node_module("unified_tts_text_node", "unified/tts_text_node.py")
    UnifiedTTSTextNode = unified_text_module.UnifiedTTSTextNode
    UNIFIED_TEXT_AVAILABLE = True
    print("✓ Unified TTS Text node loaded")
except Exception as e:
    print(f"❌ Unified TTS Text failed: {e}")
    UNIFIED_TEXT_AVAILABLE = False

try:
    unified_srt_module = load_node_module("unified_tts_srt_node", "unified/tts_srt_node.py")
    UnifiedTTSSRTNode = unified_srt_module.UnifiedTTSSRTNode
    UNIFIED_SRT_AVAILABLE = True
    print("✓ Unified TTS SRT node loaded")
except Exception as e:
    print(f"❌ Unified TTS SRT failed: {e}")
    UNIFIED_SRT_AVAILABLE = False

try:
    unified_vc_module = load_node_module("unified_voice_changer_node", "unified/voice_changer_node.py")
    UnifiedVoiceChangerNode = unified_vc_module.UnifiedVoiceChangerNode
    UNIFIED_VC_AVAILABLE = True
    print("✓ Unified Voice Changer node loaded")
except Exception as e:
    print(f"❌ Unified Voice Changer failed: {e}")
    UNIFIED_VC_AVAILABLE = False

# Load legacy support nodes (Audio Analyzer, Voice Recorder) that don't need refactoring
try:
    audio_recorder_module = load_node_module("chatterbox_audio_recorder_node", "audio/recorder_node.py")
    ChatterBoxVoiceCapture = audio_recorder_module.ChatterBoxVoiceCapture
    VOICE_CAPTURE_AVAILABLE = True
    print("✓ Voice Capture node loaded")
except Exception as e:
    print(f"❌ Voice Capture failed: {e}")
    VOICE_CAPTURE_AVAILABLE = False

# Load legacy audio analysis nodes (keep unchanged for compatibility)
try:
    audio_analyzer_module = load_node_module("chatterbox_audio_analyzer_node", "audio/analyzer_node.py")
    AudioAnalyzerNode = audio_analyzer_module.AudioAnalyzerNode
    AUDIO_ANALYZER_AVAILABLE = True
    print("✓ Audio Wave Analyzer loaded")
except Exception as e:
    print(f"❌ Audio Analyzer failed: {e}")
    AUDIO_ANALYZER_AVAILABLE = False

try:
    audio_analyzer_options_module = load_node_module("chatterbox_audio_analyzer_options_node", "audio/analyzer_options_node.py")
    AudioAnalyzerOptionsNode = audio_analyzer_options_module.AudioAnalyzerOptionsNode
    AUDIO_ANALYZER_OPTIONS_AVAILABLE = True
    print("✓ Audio Analyzer Options loaded")
except Exception as e:
    print(f"❌ Audio Analyzer Options failed: {e}")
    AUDIO_ANALYZER_OPTIONS_AVAILABLE = False

# Load F5-TTS Edit nodes (keep for specialized editing functionality)
try:
    f5tts_edit_module = load_node_module("chatterbox_f5tts_edit_node", "f5tts/f5tts_edit_node.py")
    F5TTSEditNode = f5tts_edit_module.F5TTSEditNode
    F5TTS_EDIT_AVAILABLE = True
    print("✓ F5-TTS Speech Editor loaded")
except Exception as e:
    print(f"❌ F5-TTS Edit failed: {e}")
    F5TTS_EDIT_AVAILABLE = False

try:
    f5tts_edit_options_module = load_node_module("chatterbox_f5tts_edit_options_node", "f5tts/f5tts_edit_options_node.py")
    F5TTSEditOptionsNode = f5tts_edit_options_module.F5TTSEditOptionsNode
    F5TTS_EDIT_OPTIONS_AVAILABLE = True
    print("✓ F5-TTS Edit Options loaded")
except Exception as e:
    print(f"❌ F5-TTS Edit Options failed: {e}")
    F5TTS_EDIT_OPTIONS_AVAILABLE = False

# Import foundation components for compatibility
from utils.system.import_manager import import_manager

# Legacy compatibility - keep these for existing workflows
GLOBAL_AUDIO_CACHE = {}
NODE_DIR = os.path.dirname(__file__)
BUNDLED_CHATTERBOX_DIR = os.path.join(NODE_DIR, "chatterbox")
BUNDLED_MODELS_DIR = os.path.join(NODE_DIR, "models", "chatterbox")

# Get availability status from import manager
availability = import_manager.get_availability_summary()
CHATTERBOX_TTS_AVAILABLE = availability["tts"]
CHATTERBOX_VC_AVAILABLE = availability["vc"]
CHATTERBOX_AVAILABLE = availability["any_chatterbox"]
USING_BUNDLED_CHATTERBOX = True  # Default assumption

def find_chatterbox_models():
    """Find ChatterBox model files in order of priority - Legacy compatibility function"""
    model_paths = []
    
    # 1. Check for bundled models in node folder
    bundled_model_path = os.path.join(BUNDLED_MODELS_DIR, "s3gen.pt")
    if os.path.exists(bundled_model_path):
        model_paths.append(("bundled", BUNDLED_MODELS_DIR))
        return model_paths  # Return immediately if bundled models found
    
    # 2. Check ComfyUI models folder - first check the standard location
    comfyui_model_path_standard = os.path.join(folder_paths.models_dir, "chatterbox", "s3gen.pt")
    if os.path.exists(comfyui_model_path_standard):
        model_paths.append(("comfyui", os.path.dirname(comfyui_model_path_standard)))
        return model_paths
    
    # 3. Check legacy location (TTS/chatterbox) for backward compatibility
    comfyui_model_path_legacy = os.path.join(folder_paths.models_dir, "TTS", "chatterbox", "s3gen.pt")
    if os.path.exists(comfyui_model_path_legacy):
        model_paths.append(("comfyui", os.path.dirname(comfyui_model_path_legacy)))
        return model_paths
    
    # 3. HuggingFace download as fallback (only if no local models found)
    model_paths.append(("huggingface", None))
    
    return model_paths

# Import SRT node conditionally
try:
    srt_module = load_node_module("chatterbox_srt_node", "chatterbox/chatterbox_srt_node.py")
    ChatterboxSRTTTSNode = srt_module.ChatterboxSRTTTSNode
    SRT_SUPPORT_AVAILABLE = True
except (ImportError, FileNotFoundError, AttributeError):
    SRT_SUPPORT_AVAILABLE = False
    
    # Create dummy SRT node for compatibility
    class ChatterboxSRTTTSNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"error": ("STRING", {"default": "SRT support not available"})}}
        
        RETURN_TYPES = ("STRING",)
        FUNCTION = "error"
        CATEGORY = "ChatterBox Voice"
        
        def error(self, error):
            raise ImportError("SRT support not available - missing required modules")

# Update SRT node availability based on import manager
try:
    success, modules, source = import_manager.import_srt_modules()
    if success:
        SRT_SUPPORT_AVAILABLE = True
        # Make SRT modules available for legacy compatibility if needed
        SRTParser = modules.get("SRTParser")
        SRTSubtitle = modules.get("SRTSubtitle")
        SRTParseError = modules.get("SRTParseError")
        AudioTimingUtils = modules.get("AudioTimingUtils")
        TimedAudioAssembler = modules.get("TimedAudioAssembler")
        calculate_timing_adjustments = modules.get("calculate_timing_adjustments")
        AudioTimingError = modules.get("AudioTimingError")
        PhaseVocoderTimeStretcher = modules.get("PhaseVocoderTimeStretcher")
        FFmpegTimeStretcher = modules.get("FFmpegTimeStretcher")
        
        if IS_DEV:
            print(f"✅ SRT TTS node available! (source: {source})")
    else:
        SRT_SUPPORT_AVAILABLE = False
        if IS_DEV:
            print("❌ SRT support not available")
except Exception:
    SRT_SUPPORT_AVAILABLE = False
    if IS_DEV:
        print("❌ SRT support initialization failed")

# Update F5-TTS node availability with detailed diagnostics
try:
    success, f5tts_class, source = import_manager.import_f5tts()
    if success:
        # F5-TTS is available - update global flag if needed
        if not F5TTS_SUPPORT_AVAILABLE:
            # This means the node loading failed earlier, but core F5-TTS is available
            if IS_DEV:
                print(f"⚠️ F5-TTS core available ({source}) but node loading failed - check node dependencies")
        else:
            if IS_DEV:
                print(f"✅ F5-TTS available! (source: {source})")
    else:
        # F5-TTS not available - get detailed error info
        from engines.f5tts.f5tts import F5TTS_IMPORT_ERROR
        F5TTS_SUPPORT_AVAILABLE = False
        F5TTS_SRT_SUPPORT_AVAILABLE = False  
        F5TTS_EDIT_SUPPORT_AVAILABLE = False
        F5TTS_EDIT_OPTIONS_SUPPORT_AVAILABLE = False
        # Always show F5-TTS errors to help with troubleshooting
        if F5TTS_IMPORT_ERROR:
            print(f"❌ F5-TTS not available: {F5TTS_IMPORT_ERROR}")
        else:
            print("❌ F5-TTS support not available")
except Exception as e:
    F5TTS_SUPPORT_AVAILABLE = False
    F5TTS_SRT_SUPPORT_AVAILABLE = False
    F5TTS_EDIT_SUPPORT_AVAILABLE = False
    F5TTS_EDIT_OPTIONS_SUPPORT_AVAILABLE = False
    # Always show critical F5-TTS errors
    print(f"❌ F5-TTS initialization failed: {str(e)}")

# Legacy compatibility: Remove old large SRT implementation - it's now in the new node

# Register unified nodes
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

print("🔧 Registering TTS Audio Suite nodes...")

# Register engine nodes
if CHATTERBOX_ENGINE_AVAILABLE:
    NODE_CLASS_MAPPINGS["ChatterBoxEngineNode"] = ChatterBoxEngineNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxEngineNode"] = "⚙️ ChatterBox TTS Engine"

if F5TTS_ENGINE_AVAILABLE:
    NODE_CLASS_MAPPINGS["F5TTSEngineNode"] = F5TTSEngineNode
    NODE_DISPLAY_NAME_MAPPINGS["F5TTSEngineNode"] = "⚙️ F5 TTS Engine"

# Register shared nodes
if CHARACTER_VOICES_AVAILABLE:
    NODE_CLASS_MAPPINGS["CharacterVoicesNode"] = CharacterVoicesNode
    NODE_DISPLAY_NAME_MAPPINGS["CharacterVoicesNode"] = "🎭 Character Voices"

# Register unified nodes
if UNIFIED_TEXT_AVAILABLE:
    NODE_CLASS_MAPPINGS["UnifiedTTSTextNode"] = UnifiedTTSTextNode
    NODE_DISPLAY_NAME_MAPPINGS["UnifiedTTSTextNode"] = "🎤 TTS Text"

if UNIFIED_SRT_AVAILABLE:
    NODE_CLASS_MAPPINGS["UnifiedTTSSRTNode"] = UnifiedTTSSRTNode
    NODE_DISPLAY_NAME_MAPPINGS["UnifiedTTSSRTNode"] = "📺 TTS SRT"

if UNIFIED_VC_AVAILABLE:
    NODE_CLASS_MAPPINGS["UnifiedVoiceChangerNode"] = UnifiedVoiceChangerNode
    NODE_DISPLAY_NAME_MAPPINGS["UnifiedVoiceChangerNode"] = "🔄 Voice Changer"

# Register legacy support nodes
if VOICE_CAPTURE_AVAILABLE:
    NODE_CLASS_MAPPINGS["ChatterBoxVoiceCapture"] = ChatterBoxVoiceCapture
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxVoiceCapture"] = "🎙️ Voice Capture"

if AUDIO_ANALYZER_AVAILABLE:
    NODE_CLASS_MAPPINGS["ChatterBoxAudioAnalyzer"] = AudioAnalyzerNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxAudioAnalyzer"] = "🌊 Audio Wave Analyzer"

if AUDIO_ANALYZER_OPTIONS_AVAILABLE:
    NODE_CLASS_MAPPINGS["ChatterBoxAudioAnalyzerOptions"] = AudioAnalyzerOptionsNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxAudioAnalyzerOptions"] = "🔧 Audio Analyzer Options"

if F5TTS_EDIT_AVAILABLE:
    NODE_CLASS_MAPPINGS["ChatterBoxF5TTSEditVoice"] = F5TTSEditNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxF5TTSEditVoice"] = "👄 F5-TTS Speech Editor"

if F5TTS_EDIT_OPTIONS_AVAILABLE:
    NODE_CLASS_MAPPINGS["ChatterBoxF5TTSEditOptions"] = F5TTSEditOptionsNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxF5TTSEditOptions"] = "🔧 F5-TTS Edit Options"

# Print startup banner
print(SEPARATOR)
print(f"🚀 TTS Audio Suite {VERSION_DISPLAY}")
print("Universal multi-engine TTS extension for ComfyUI")

# Show architecture status
unified_count = sum([UNIFIED_TEXT_AVAILABLE, UNIFIED_SRT_AVAILABLE, UNIFIED_VC_AVAILABLE])
engine_count = sum([CHATTERBOX_ENGINE_AVAILABLE, F5TTS_ENGINE_AVAILABLE])
support_count = sum([CHARACTER_VOICES_AVAILABLE, VOICE_CAPTURE_AVAILABLE, AUDIO_ANALYZER_AVAILABLE, F5TTS_EDIT_AVAILABLE])

print(f"📊 Architecture Status:")
print(f"   • Unified Nodes: {unified_count}/3 loaded")
print(f"   • Engine Nodes: {engine_count}/2 loaded") 
print(f"   • Support Nodes: {support_count} loaded")

# Check for local models (legacy compatibility)
try:
    model_paths = find_chatterbox_models()
    first_source = model_paths[0][0] if model_paths else None
    if first_source == "bundled":
        print("✓ Using bundled ChatterBox models")
    elif first_source == "comfyui":
        print("✓ Using ComfyUI ChatterBox models")
    else:
        print("⚠️ No local ChatterBox models found - will download from Hugging Face")
except:
    print("⚠️ ChatterBox model discovery not available")

# Check for system dependency issues (only show warnings if problems detected)
dependency_warnings = []

# Check PortAudio availability for voice recording
if VOICE_CAPTURE_AVAILABLE and hasattr(audio_recorder_module, 'SOUNDDEVICE_AVAILABLE') and not audio_recorder_module.SOUNDDEVICE_AVAILABLE:
    dependency_warnings.append("⚠️ PortAudio library not found - Voice recording disabled")
    dependency_warnings.append("   Install with: sudo apt-get install portaudio19-dev (Linux) or brew install portaudio (macOS)")

# Only show dependency section if there are warnings
if dependency_warnings:
    print("📋 System Dependencies:")
    for warning in dependency_warnings:
        print(f"   {warning}")

print(f"✅ TTS Audio Suite {VERSION_DISPLAY} loaded with {len(NODE_DISPLAY_NAME_MAPPINGS)} nodes")
print(SEPARATOR)
