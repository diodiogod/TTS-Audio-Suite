# Smart numba compatibility handling
try:
    from utils.compatibility import setup_numba_compatibility
    setup_numba_compatibility(quick_startup=True, verbose=False)
except ImportError:
    # Fallback: only apply workaround if not already handled
    import sys
    import os
    # Don't do anything - __init__.py already handled compatibility testing

# Version and constants
VERSION = "4.13.2"
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

import folder_paths
import importlib.util

# Add current directory to path for absolute imports
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Check transformers version compatibility
try:
    import transformers
    from packaging import version

    required_version = "4.51.3"
    current_version = transformers.__version__

    if version.parse(current_version) < version.parse(required_version):
        print(f"🚨 COMPATIBILITY WARNING:")
        print(f"   Transformers version {current_version} is too old (requires >={required_version})")
        print(f"   This WILL cause errors like 'DynamicCache property has no setter'")
        print(f"   📋 SOLUTION: Run this command to upgrade:")
        print(f"   pip install --upgrade transformers>={required_version}")
        print(f"   (Or use your environment's package manager)")
        print()
except Exception as e:
    print(f"⚠️ Could not check transformers version: {e}")
    print("   If you encounter DynamicCache errors, upgrade transformers to >=4.51.3")

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

# Load engine nodes
try:
    chatterbox_engine_module = load_node_module("chatterbox_engine_node", "engines/chatterbox_engine_node.py")
    ChatterBoxEngineNode = chatterbox_engine_module.ChatterBoxEngineNode
    CHATTERBOX_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"❌ ChatterBox Engine failed: {e}")
    CHATTERBOX_ENGINE_AVAILABLE = False

try:
    f5tts_engine_module = load_node_module("f5tts_engine_node", "engines/f5tts_engine_node.py")
    F5TTSEngineNode = f5tts_engine_module.F5TTSEngineNode
    F5TTS_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"❌ F5 TTS Engine failed: {e}")
    F5TTS_ENGINE_AVAILABLE = False

try:
    higgs_audio_engine_module = load_node_module("higgs_audio_engine_node", "engines/higgs_audio_engine_node.py")
    HiggsAudioEngineNode = higgs_audio_engine_module.HiggsAudioEngineNode
    HIGGS_AUDIO_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"❌ Higgs Audio Engine failed: {e}")
    HIGGS_AUDIO_ENGINE_AVAILABLE = False

try:
    vibevoice_engine_module = load_node_module("vibevoice_engine_node", "engines/vibevoice_engine_node.py")
    VibeVoiceEngineNode = vibevoice_engine_module.VibeVoiceEngineNode
    VIBEVOICE_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"❌ VibeVoice Engine failed: {e}")
    VIBEVOICE_ENGINE_AVAILABLE = False

try:
    chatterbox_official_23lang_engine_module = load_node_module("chatterbox_official_23lang_engine_node", "engines/chatterbox_official_23lang_engine_node.py")
    ChatterBoxOfficial23LangEngineNode = chatterbox_official_23lang_engine_module.ChatterBoxOfficial23LangEngineNode
    CHATTERBOX_OFFICIAL_23LANG_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"❌ ChatterBox Official 23-Lang Engine failed: {e}")
    CHATTERBOX_OFFICIAL_23LANG_ENGINE_AVAILABLE = False

try:
    index_tts_engine_module = load_node_module("index_tts_engine_node", "engines/index_tts_engine_node.py")
    IndexTTSEngineNode = index_tts_engine_module.IndexTTSEngineNode
    INDEX_TTS_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"❌ IndexTTS-2 Engine failed: {e}")
    INDEX_TTS_ENGINE_AVAILABLE = False

# IndexTTS-2 Emotion Options Node
try:
    index_tts_emotion_options_module = load_node_module("index_tts_emotion_options_node", "engines/index_tts_emotion_options_node.py")
    IndexTTSEmotionOptionsNode = index_tts_emotion_options_module.IndexTTSEmotionOptionsNode
    INDEX_TTS_EMOTION_OPTIONS_AVAILABLE = True
except Exception as e:
    print(f"❌ IndexTTS-2 Emotion Options failed: {e}")
    INDEX_TTS_EMOTION_OPTIONS_AVAILABLE = False

# QwenEmotion Text Analysis Node
try:
    qwen_emotion_module = load_node_module("qwen_emotion_node", "index_tts/qwen_emotion_node.py")
    QwenEmotionNode = qwen_emotion_module.QwenEmotionNode
    QWEN_EMOTION_AVAILABLE = True
except Exception as e:
    print(f"❌ QwenEmotion Text Analysis failed: {e}")
    QWEN_EMOTION_AVAILABLE = False

# Load shared nodes
try:
    character_voices_module = load_node_module("character_voices_node", "shared/character_voices_node.py")
    CharacterVoicesNode = character_voices_module.CharacterVoicesNode
    CHARACTER_VOICES_AVAILABLE = True
except Exception as e:
    print(f"❌ Character Voices failed: {e}")
    CHARACTER_VOICES_AVAILABLE = False

# Load unified nodes
try:
    unified_text_module = load_node_module("unified_tts_text_node", "unified/tts_text_node.py")
    UnifiedTTSTextNode = unified_text_module.UnifiedTTSTextNode
    UNIFIED_TEXT_AVAILABLE = True
except Exception as e:
    print(f"❌ Unified TTS Text failed: {e}")
    UNIFIED_TEXT_AVAILABLE = False

try:
    unified_srt_module = load_node_module("unified_tts_srt_node", "unified/tts_srt_node.py")
    UnifiedTTSSRTNode = unified_srt_module.UnifiedTTSSRTNode
    UNIFIED_SRT_AVAILABLE = True
except Exception as e:
    print(f"❌ Unified TTS SRT failed: {e}")
    UNIFIED_SRT_AVAILABLE = False

try:
    unified_vc_module = load_node_module("unified_voice_changer_node", "unified/voice_changer_node.py")
    UnifiedVoiceChangerNode = unified_vc_module.UnifiedVoiceChangerNode
    UNIFIED_VC_AVAILABLE = True
except Exception as e:
    print(f"❌ Unified Voice Changer failed: {e}")
    UNIFIED_VC_AVAILABLE = False

# Load support nodes
try:
    audio_recorder_module = load_node_module("chatterbox_audio_recorder_node", "audio/recorder_node.py")
    ChatterBoxVoiceCapture = audio_recorder_module.ChatterBoxVoiceCapture
    VOICE_CAPTURE_AVAILABLE = True
except Exception as e:
    print(f"❌ Voice Capture failed: {e}")
    VOICE_CAPTURE_AVAILABLE = False

# Load audio analysis nodes
try:
    audio_analyzer_module = load_node_module("chatterbox_audio_analyzer_node", "audio/analyzer_node.py")
    AudioAnalyzerNode = audio_analyzer_module.AudioAnalyzerNode
    AUDIO_ANALYZER_AVAILABLE = True
except Exception as e:
    print(f"❌ Audio Analyzer failed: {e}")
    AUDIO_ANALYZER_AVAILABLE = False

try:
    audio_analyzer_options_module = load_node_module("chatterbox_audio_analyzer_options_node", "audio/analyzer_options_node.py")
    AudioAnalyzerOptionsNode = audio_analyzer_options_module.AudioAnalyzerOptionsNode
    AUDIO_ANALYZER_OPTIONS_AVAILABLE = True
except Exception as e:
    print(f"❌ Audio Analyzer Options failed: {e}")
    AUDIO_ANALYZER_OPTIONS_AVAILABLE = False

# Load F5-TTS Edit nodes
try:
    f5tts_edit_module = load_node_module("chatterbox_f5tts_edit_node", "f5tts/f5tts_edit_node.py")
    F5TTSEditNode = f5tts_edit_module.F5TTSEditNode
    F5TTS_EDIT_AVAILABLE = True
except Exception as e:
    print(f"❌ F5-TTS Edit failed: {e}")
    F5TTS_EDIT_AVAILABLE = False

try:
    f5tts_edit_options_module = load_node_module("chatterbox_f5tts_edit_options_node", "f5tts/f5tts_edit_options_node.py")
    F5TTSEditOptionsNode = f5tts_edit_options_module.F5TTSEditOptionsNode
    F5TTS_EDIT_OPTIONS_AVAILABLE = True
except Exception as e:
    print(f"❌ F5-TTS Edit Options failed: {e}")
    F5TTS_EDIT_OPTIONS_AVAILABLE = False

# Load RVC nodes
try:
    rvc_engine_module = load_node_module("rvc_engine_node", "engines/rvc_engine_node.py")
    RVCEngineNode = rvc_engine_module.RVCEngineNode
    RVC_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"❌ RVC Engine failed: {e}")
    RVC_ENGINE_AVAILABLE = False

try:
    rvc_pitch_options_module = load_node_module("rvc_pitch_options_node", "audio/rvc_pitch_options_node.py")
    RVCPitchOptionsNode = rvc_pitch_options_module.RVCPitchOptionsNode
    RVC_PITCH_OPTIONS_AVAILABLE = True
except Exception as e:
    print(f"❌ RVC Pitch Options failed: {e}")
    RVC_PITCH_OPTIONS_AVAILABLE = False

try:
    vocal_removal_module = load_node_module("vocal_removal_node", "audio/vocal_removal_node.py")
    VocalRemovalNode = vocal_removal_module.VocalRemovalNode
    VOCAL_REMOVAL_AVAILABLE = True
except Exception as e:
    print(f"❌ Vocal/Noise Removal failed: {e}")
    VOCAL_REMOVAL_AVAILABLE = False

try:
    merge_audio_module = load_node_module("merge_audio_node", "audio/merge_audio_node.py")
    MergeAudioNode = merge_audio_module.MergeAudioNode
    MERGE_AUDIO_AVAILABLE = True
except Exception as e:
    print(f"❌ Merge Audio failed: {e}")
    MERGE_AUDIO_AVAILABLE = False

try:
    voice_fixer_module = load_node_module("voice_fixer_node", "audio/voice_fixer_node.py")
    VoiceFixerNode = voice_fixer_module.VoiceFixerNode
    VOICE_FIXER_AVAILABLE = True
except Exception as e:
    print(f"❌ Voice Fixer Audio Restoration failed: {e}")
    VOICE_FIXER_AVAILABLE = False

try:
    load_rvc_model_module = load_node_module("load_rvc_model_node", "models/load_rvc_model_node.py")
    LoadRVCModelNode = load_rvc_model_module.LoadRVCModelNode
    LOAD_RVC_MODEL_AVAILABLE = True
except Exception as e:
    print(f"❌ Load RVC Character Model failed: {e}")
    LOAD_RVC_MODEL_AVAILABLE = False

try:
    phoneme_text_normalizer_module = load_node_module("phoneme_text_normalizer_node", "text/phoneme_text_normalizer_node.py")
    PhonemeTextNormalizer = phoneme_text_normalizer_module.PhonemeTextNormalizer
    PHONEME_TEXT_NORMALIZER_AVAILABLE = True
except Exception as e:
    print(f"❌ Phoneme Text Normalizer failed: {e}")
    PHONEME_TEXT_NORMALIZER_AVAILABLE = False

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
    
    # 2. Check ComfyUI models folder - first check the new TTS organization
    comfyui_model_path_tts = os.path.join(folder_paths.models_dir, "TTS", "chatterbox", "s3gen.pt")
    if os.path.exists(comfyui_model_path_tts):
        model_paths.append(("comfyui", os.path.dirname(comfyui_model_path_tts)))
        return model_paths
    
    # 3. Check legacy location (direct chatterbox) for backward compatibility
    comfyui_model_path_legacy = os.path.join(folder_paths.models_dir, "chatterbox", "s3gen.pt")
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

# The new unified architecture handles engine availability internally

# Register unified nodes
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register engine nodes
if CHATTERBOX_ENGINE_AVAILABLE:
    NODE_CLASS_MAPPINGS["ChatterBoxEngineNode"] = ChatterBoxEngineNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxEngineNode"] = "⚙️ ChatterBox TTS Engine"

if F5TTS_ENGINE_AVAILABLE:
    NODE_CLASS_MAPPINGS["F5TTSEngineNode"] = F5TTSEngineNode
    NODE_DISPLAY_NAME_MAPPINGS["F5TTSEngineNode"] = "⚙️ F5 TTS Engine"

if HIGGS_AUDIO_ENGINE_AVAILABLE:
    NODE_CLASS_MAPPINGS["HiggsAudioEngineNode"] = HiggsAudioEngineNode
    NODE_DISPLAY_NAME_MAPPINGS["HiggsAudioEngineNode"] = "⚙️ Higgs Audio 2 Engine"

if VIBEVOICE_ENGINE_AVAILABLE:
    NODE_CLASS_MAPPINGS["VibeVoiceEngineNode"] = VibeVoiceEngineNode
    NODE_DISPLAY_NAME_MAPPINGS["VibeVoiceEngineNode"] = "⚙️ VibeVoice Engine"

if CHATTERBOX_OFFICIAL_23LANG_ENGINE_AVAILABLE:
    NODE_CLASS_MAPPINGS["ChatterBoxOfficial23LangEngineNode"] = ChatterBoxOfficial23LangEngineNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxOfficial23LangEngineNode"] = "⚙️ ChatterBox Official 23-Lang Engine"

if INDEX_TTS_ENGINE_AVAILABLE:
    NODE_CLASS_MAPPINGS["IndexTTSEngineNode"] = IndexTTSEngineNode
    NODE_DISPLAY_NAME_MAPPINGS["IndexTTSEngineNode"] = "⚙️ IndexTTS-2 Engine"

if INDEX_TTS_EMOTION_OPTIONS_AVAILABLE:
    NODE_CLASS_MAPPINGS["IndexTTSEmotionOptionsNode"] = IndexTTSEmotionOptionsNode
    NODE_DISPLAY_NAME_MAPPINGS["IndexTTSEmotionOptionsNode"] = "🌈 IndexTTS-2 Emotion Vectors"

if QWEN_EMOTION_AVAILABLE:
    NODE_CLASS_MAPPINGS["QwenEmotionNode"] = QwenEmotionNode
    NODE_DISPLAY_NAME_MAPPINGS["QwenEmotionNode"] = "🌈 IndexTTS-2 Text Emotion"

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

# Load video analysis nodes
try:
    mouth_movement_module = load_node_module("mouth_movement_analyzer_node", "video/mouth_movement_analyzer_node.py")
    MouthMovementAnalyzerNode = mouth_movement_module.MouthMovementAnalyzerNode
    MOUTH_MOVEMENT_AVAILABLE = True
except Exception as e:
    print(f"❌ Mouth Movement Analyzer failed: {e}")
    MOUTH_MOVEMENT_AVAILABLE = False

try:
    viseme_options_module = load_node_module("viseme_options_node", "video/viseme_options_node.py")
    VisemeDetectionOptionsNode = viseme_options_module.VisemeDetectionOptionsNode
    VISEME_OPTIONS_AVAILABLE = True
except Exception as e:
    print(f"❌ Viseme Options failed: {e}")
    VISEME_OPTIONS_AVAILABLE = False

# Register RVC nodes
if RVC_ENGINE_AVAILABLE:
    NODE_CLASS_MAPPINGS["RVCEngineNode"] = RVCEngineNode
    NODE_DISPLAY_NAME_MAPPINGS["RVCEngineNode"] = "⚙️ RVC Engine"

if RVC_PITCH_OPTIONS_AVAILABLE:
    NODE_CLASS_MAPPINGS["RVCPitchOptionsNode"] = RVCPitchOptionsNode
    NODE_DISPLAY_NAME_MAPPINGS["RVCPitchOptionsNode"] = "🔧 RVC Pitch Extraction Options"

if VOCAL_REMOVAL_AVAILABLE:
    NODE_CLASS_MAPPINGS["VocalRemovalNode"] = VocalRemovalNode
    NODE_DISPLAY_NAME_MAPPINGS["VocalRemovalNode"] = "🤐 Noise or Vocal Removal"

if MERGE_AUDIO_AVAILABLE:
    NODE_CLASS_MAPPINGS["MergeAudioNode"] = MergeAudioNode
    NODE_DISPLAY_NAME_MAPPINGS["MergeAudioNode"] = "🥪 Merge Audio"

if VOICE_FIXER_AVAILABLE:
    NODE_CLASS_MAPPINGS["VoiceFixerNode"] = VoiceFixerNode
    NODE_DISPLAY_NAME_MAPPINGS["VoiceFixerNode"] = "🤐 Voice Fixer"

if LOAD_RVC_MODEL_AVAILABLE:
    NODE_CLASS_MAPPINGS["LoadRVCModelNode"] = LoadRVCModelNode
    NODE_DISPLAY_NAME_MAPPINGS["LoadRVCModelNode"] = "🎭 Load RVC Character Model"

# Register text processing nodes
if PHONEME_TEXT_NORMALIZER_AVAILABLE:
    NODE_CLASS_MAPPINGS["PhonemeTextNormalizer"] = PhonemeTextNormalizer
    NODE_DISPLAY_NAME_MAPPINGS["PhonemeTextNormalizer"] = "📝 Phoneme Text Normalizer"

# Register video analysis nodes
if MOUTH_MOVEMENT_AVAILABLE:
    NODE_CLASS_MAPPINGS["MouthMovementAnalyzer"] = MouthMovementAnalyzerNode
    NODE_DISPLAY_NAME_MAPPINGS["MouthMovementAnalyzer"] = "🗣️ Silent Speech Analyzer"

if VISEME_OPTIONS_AVAILABLE:
    NODE_CLASS_MAPPINGS["VisemeDetectionOptionsNode"] = VisemeDetectionOptionsNode
    NODE_DISPLAY_NAME_MAPPINGS["VisemeDetectionOptionsNode"] = "🔧 Viseme Mouth Shape Options"

# Print startup banner
print(SEPARATOR)
print(f"🚀 TTS Audio Suite {VERSION_DISPLAY}")
print("Universal multi-engine TTS extension for ComfyUI")

# Show Python 3.13 compatibility status
if sys.version_info >= (3, 13):
    print(f"🐍 Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")
    # 🔬 NUMBA WORKAROUND: Commented out - testing if still needed with numba 0.61.2+ and librosa 0.11.0+
    # print("⚡ Numba JIT disabled for librosa compatibility")

# Removed legacy ChatterBox model discovery to improve startup performance
# Model discovery now happens lazily when needed by individual engines

# Import dependency checker
try:
    from utils.system.dependency_checker import DependencyChecker, AsyncDependencyChecker
    DEPENDENCY_CHECKER_AVAILABLE = True
except ImportError:
    DEPENDENCY_CHECKER_AVAILABLE = False
    AsyncDependencyChecker = None

# Check for immediate PortAudio issues (show during startup)
startup_warnings = []

# Check PortAudio availability for voice recording
if VOICE_CAPTURE_AVAILABLE and hasattr(audio_recorder_module, 'SOUNDDEVICE_AVAILABLE') and not audio_recorder_module.SOUNDDEVICE_AVAILABLE:
    startup_warnings.append("⚠️ PortAudio library not found - Voice recording disabled")
    startup_warnings.append("   Install with: sudo apt-get install portaudio19-dev (Linux) or brew install portaudio (macOS)")

# Only show startup dependencies if there are warnings
if startup_warnings:
    print("📋 System Dependencies:")
    for warning in startup_warnings:
        print(f"   {warning}")

# Start background dependency check (non-blocking)
# This will print warnings after ComfyUI loads if there are missing dependencies
if DEPENDENCY_CHECKER_AVAILABLE and AsyncDependencyChecker:
    try:
        async_checker = AsyncDependencyChecker()
        async_checker.start_background_check()
    except Exception:
        pass  # Silently fail - background check is optional

print(f"✅ TTS Audio Suite {VERSION_DISPLAY} loaded with {len(NODE_DISPLAY_NAME_MAPPINGS)} nodes:")
for node in sorted(NODE_DISPLAY_NAME_MAPPINGS.values()):
    print(f"   • {node}")
print(SEPARATOR)

# Trigger voice discovery initialization in background thread after ComfyUI startup
# This loads character cache or starts scanning in a non-blocking way
try:
    import threading
    from utils.voice.discovery import voice_discovery

    def _init_voice_discovery_background():
        """Initialize voice discovery in background after ComfyUI loads."""
        import time
        # Wait a moment for ComfyUI to fully initialize
        time.sleep(0.5)
        # Trigger initialization (loads cache or starts background scan)
        voice_discovery._ensure_initialized()

    discovery_thread = threading.Thread(target=_init_voice_discovery_background, daemon=True)
    discovery_thread.start()
except Exception:
    pass  # Silently fail - discovery will happen on first use
