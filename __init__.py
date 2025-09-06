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

# Python 3.13 compatibility: Disable numba JIT for librosa compatibility
def setup_python313_compatibility():
    """Setup Python 3.13 compatibility fixes for numba/librosa issues"""
    if sys.version_info >= (3, 13):
        # Disable numba JIT compilation to fix librosa compatibility issues
        # This prevents numba compilation errors in librosa with Python 3.13
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        print(f"🔧 TTS Audio Suite: Python {sys.version_info.major}.{sys.version_info.minor} detected - disabled numba JIT for RVC compatibility")
        
        # Also try to disable it programmatically if numba is already loaded
        try:
            import numba
            numba.config.DISABLE_JIT = True
            print("🔧 TTS Audio Suite: Set numba.config.DISABLE_JIT = True")
        except ImportError:
            # numba not yet installed, the environment variable will handle it
            print("🔧 TTS Audio Suite: Numba not yet loaded - environment variable will handle it")
        
        # Additional compatibility environment variables
        os.environ['NUMBA_ENABLE_CUDASIM'] = '1'  # Enable CUDA simulation fallback
        
    else:
        print(f"🔧 TTS Audio Suite: Python {sys.version_info.major}.{sys.version_info.minor} detected - numba JIT enabled for optimal performance")

# Apply Python 3.13 compatibility fixes
setup_python313_compatibility()

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