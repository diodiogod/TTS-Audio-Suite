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
import sys

# Smart Numba Compatibility System - tests and applies fixes only when needed
try:
    from utils.compatibility import setup_numba_compatibility
    # Apply smart compatibility setup (fast startup test)
    compatibility_results = setup_numba_compatibility(quick_startup=True, verbose=True)
except ImportError:
    # Fallback to simple approach if compatibility module not found
    import sys
    import os
    if sys.version_info >= (3, 13):
        # Basic test: try librosa.stft and apply workaround if it fails
        try:
            import numpy as np
            import librosa
            test_audio = np.random.randn(512).astype(np.float32)
            _ = librosa.stft(test_audio, hop_length=256, n_fft=512)
            # Only show when there's a problem, not success
            # Mark that we've tested numba compatibility
            import sys
            sys.modules['__main__']._tts_numba_tested = True
        except Exception as e:
            if "'function' object has no attribute 'get_call_template'" in str(e):
                os.environ['NUMBA_DISABLE_JIT'] = '1'
                os.environ['NUMBA_ENABLE_CUDASIM'] = '1'
                try:
                    import numba
                    numba.config.DISABLE_JIT = True
                except ImportError:
                    pass
                print("🔧 Applied numba JIT workaround for Python 3.13 compatibility")
            else:
                print(f"⚠️ Librosa test failed with different error: {e}")
    else:
        # Only show warning when JIT is disabled (indicates a problem)
        pass

# Apply ComfyUI compatibility patches
try:
    from utils.comfyui_compatibility import apply_all_compatibility_patches
    apply_all_compatibility_patches()
except ImportError:
    pass

# Suppress torchaudio 2.9+ TorchCodec migration warnings (informational only, no action needed)
import warnings
import sys
import os

warnings.filterwarnings("ignore", message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec", category=UserWarning)
warnings.filterwarnings("ignore", message="In 2.9, this function's implementation will be changed to use torchaudio.save_with_torchcodec", category=UserWarning)

# Version disclosure for troubleshooting
def print_critical_versions():
    """Print versions of critical packages for troubleshooting"""
    critical_packages = [
        ('numpy', 'NumPy'),
        ('librosa', 'Librosa'),
        ('numba', 'Numba'),
        ('torch', 'PyTorch'),
        ('torchaudio', 'TorchAudio'),
        ('transformers', 'Transformers'),
        ('accelerate', 'Accelerate'),
        ('soundfile', 'SoundFile'),
    ]

    version_info = []
    for pkg_name, display_name in critical_packages:
        try:
            module = __import__(pkg_name)
            version = getattr(module, '__version__', 'unknown')
            version_info.append(f"{display_name} {version}")
        except ImportError:
            version_info.append(f"{display_name} not installed")

    print(f"ℹ️ Critical package versions: {', '.join(version_info)}")

def check_ffmpeg_availability():
    """Check ffmpeg availability and log status"""
    try:
        from utils.ffmpeg_utils import FFmpegUtils
        if FFmpegUtils.is_available():
            # Only show when unavailable (problem)
            pass
        else:
            print("⚠️ FFmpeg not found - using fallback audio processing (reduced quality)")
            print("💡 Install FFmpeg for optimal performance: https://ffmpeg.org/download.html")
    except ImportError:
        # Fallback check if utils not available yet
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                # Only show when unavailable (problem)
                pass
            else:
                print("⚠️ FFmpeg not found - using fallback audio processing (reduced quality)")
        except Exception:
            print("⚠️ FFmpeg not found - using fallback audio processing (reduced quality)")
            print("💡 Install FFmpeg for optimal performance: https://ffmpeg.org/download.html")

# Print versions and check dependencies immediately for troubleshooting
print_critical_versions()
check_ffmpeg_availability()

# Check for old ChatterBox extension conflict
def check_old_extension_conflict():
    """Check if the old ComfyUI_ChatterBox_SRT_Voice extension is installed"""
    try:
        import folder_paths
        custom_nodes_path = folder_paths.get_folder_paths("custom_nodes")[0]
        old_extension_path = os.path.join(custom_nodes_path, "ComfyUI_ChatterBox_SRT_Voice")
        
        if os.path.exists(old_extension_path):
            print("\n" + "="*80)
            print("⚠️  EXTENSION CONFLICT DETECTED ⚠️")
            print("="*80)
            print("❌ OLD EXTENSION FOUND: ComfyUI_ChatterBox_SRT_Voice")
            print("🆕 CURRENT EXTENSION: ComfyUI_TTS_Audio_Suite")
            print("")
            print("The old 'ComfyUI_ChatterBox_SRT_Voice' extension conflicts with this")
            print("new 'ComfyUI_TTS_Audio_Suite' extension and MUST be removed.")
            print("")
            print("REQUIRED ACTION:")
            print(f"1. Delete the old extension folder: {old_extension_path}")
            print("2. Restart ComfyUI")
            print("")
            print("The TTS Audio Suite is the evolved version with:")
            print("• Unified architecture supporting multiple TTS engines")
            print("• Better performance and stability")
            print("• All features from the old extension plus new capabilities")
            print("")
            print("Your workflows will be compatible - just update node names.")
            print("="*80)
            print("")
            return True
    except Exception as e:
        # Silently continue if we can't check (e.g., folder_paths not available yet)
        pass
    return False

# Perform conflict check
OLD_EXTENSION_CONFLICT = check_old_extension_conflict()

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