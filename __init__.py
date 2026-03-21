"""
TTS Audio Suite - Universal multi-engine TTS extension for ComfyUI
Unified architecture supporting ChatterBox, F5-TTS, and future engines like RVC:
• 🎤 TTS Text (unified text-to-speech)
• 📺 TTS SRT (unified SRT subtitle timing)
• 🔄 Voice Changer (unified voice conversion)
• ⚙️ Engine nodes (ChatterBox, F5-TTS)
• 🎭 Character Voices (voice reference management)
"""

# Note: PYTORCH_ALLOC_CONF should be set in ComfyUI launch script if needed
# Setting it here causes "allocator mismatch" errors because ComfyUI already imported torch

# Import from the main nodes.py file which handles the new unified architecture
import importlib.util
import os
import sys
import time

# Note: PyTorch inductor patches removed - not needed for PyTorch 2.10+ with triton-windows 3.6+
# Qwen3-TTS torch.compile optimizations require:
# - PyTorch 2.10.0+ with CUDA 13.0
# - triton-windows 3.6.0+ (Windows) or triton 3.6.0+ (Linux)
# See docs/qwen3_tts_optimizations.md for installation instructions

# Enable TensorFloat32 for better performance on Ampere+ GPUs (RTX 30xx+)
try:
    import torch
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
except Exception:
    pass

# PyTorch patches solve TWO PyTorch 2.9 issues:
# 1. TorchCodec DLL incompatibility on Windows - Global patch uses scipy instead
# 2. PyTorch 2.9's changed torchaudio.load() returning raw int16 - safe_load_audio() normalizes
#
# Transformers patches solve:
# 1. Step Audio EditX tokenization bug in transformers 4.54+ (audio tokens not recognized)
# 2. Various model compatibility issues
try:
    # Load pytorch_patches directly by file path to avoid package import issues
    pytorch_patches_path = os.path.join(os.path.dirname(__file__), "utils", "compatibility", "pytorch_patches.py")
    spec = importlib.util.spec_from_file_location("pytorch_patches_module", pytorch_patches_path)
    pytorch_patches_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pytorch_patches_module)

    # Apply the patches (will only apply on PyTorch 2.9+, silently skip on older versions)
    pytorch_patches_module.apply_pytorch_patches(verbose=True)
except Exception as e:
    print(f"⚠️ Warning: Could not apply PyTorch patches: {e}")

# Transformers compatibility patches DEFERRED to first engine use.
# The patches module is deprecated (all patches are for old transformers versions),
# and importing it eagerly pulls in transformers (~1.3s).
# Patches will be applied lazily when an engine first imports transformers.
_transformers_patches_applied = False
def _apply_transformers_patches_once():
    """Apply transformers patches lazily, on first engine use."""
    global _transformers_patches_applied
    if _transformers_patches_applied:
        return
    _transformers_patches_applied = True
    try:
        transformers_patches_path = os.path.join(os.path.dirname(__file__), "utils", "compatibility", "transformers_patches.py")
        spec = importlib.util.spec_from_file_location("transformers_patches_module", transformers_patches_path)
        transformers_patches_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transformers_patches_module)
        transformers_patches_module.apply_transformers_patches(verbose=True)
    except Exception as e:
        print(f"⚠️ Warning: Could not apply Transformers patches: {e}")

# Numba/Librosa compatibility check at startup.
# Do NOT force NUMBA_DISABLE_JIT on Python 3.13 anymore:
# newer stacks (for example numba 0.64 + librosa 0.11) can work normally,
# and forcing the env var can itself trigger the get_call_template crash.
# For older Python + NumPy 2.x, keep the existing thorough compatibility test.
if sys.version_info < (3, 13):
    try:
        import numpy as _np
        if int(_np.__version__.split('.')[0]) >= 2:
            numba_compat_path = os.path.join(os.path.dirname(__file__), "utils", "compatibility", "numba_compat.py")
            _spec = importlib.util.spec_from_file_location("numba_compat_module", numba_compat_path)
            _numba_compat = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_numba_compat)
            _numba_compat.setup_numba_compatibility(quick_startup=False, verbose=True)
    except Exception:
        # If the compatibility test itself crashes, that means numba JIT is broken —
        # disable it and warn the user.
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        print("⚠️ TTS Audio Suite: Numba JIT crash detected at startup — disabling JIT (NUMBA_DISABLE_JIT=1)")

# TorchCodec note: Removed torchcodec dependency to eliminate FFmpeg system requirement
# torchaudio.load() works fine with fallback backends (soundfile, scipy)
import warnings
import sys
import os

def check_dependencies():
    """Fast check for critical dependencies without importing them into memory"""
    critical_packages = ['torch', 'torchaudio', 'transformers', 'librosa', 'numba', 'soundfile', 'accelerate']
    missing = []
    
    for pkg in critical_packages:
        # Avoid importlib.util.find_spec for namespace packages or if spec is None
        try:
            if importlib.util.find_spec(pkg) is None:
                missing.append(pkg)
        except Exception:
            missing.append(pkg)

    if missing:
        print(f"\n{'='*80}")
        print(f"⚠️  TTS AUDIO SUITE: CRITICAL DEPENDENCIES MISSING ⚠️")
        print(f"{'='*80}")
        print(f"The following required packages are missing: {', '.join(missing)}")
        print(f"")
        print(f"Please run the installation script or install them manually:")
        print(f"pip install -r requirements.txt")
        print(f"{'='*80}\n")

# Version disclosure for troubleshooting
def print_critical_versions():
    """Print versions of critical packages for troubleshooting.

    Uses importlib.metadata to read versions without importing the actual
    packages. This avoids pulling in transformers (~1.3s), librosa (~0.7s),
    and other heavy modules just to print a version string at startup.
    """
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

    from importlib.metadata import version as _pkg_version, PackageNotFoundError

    version_info = []
    for pkg_name, display_name in critical_packages:
        try:
            ver = _pkg_version(pkg_name)
            version_info.append(f"{display_name} {ver}")
        except PackageNotFoundError:
            version_info.append(f"{display_name} not installed")

    print(f"ℹ️ Critical package versions: {', '.join(version_info)}")

def warn_transformers_5_unsupported():
    """Warn when Transformers 5.x is installed (Qwen3-TTS tokenizer is incompatible).

    NOTE: This check uses sys.modules to avoid eagerly importing transformers (~1.3s).
    If transformers hasn't been imported yet (e.g. by the version-printing function above),
    we skip the check -- it will be caught later when an engine actually loads transformers.
    """
    try:
        # Only check if transformers is already loaded (avoids ~1.3s eager import)
        import sys as _sys
        if 'transformers' not in _sys.modules:
            return
        transformers = _sys.modules['transformers']
        try:
            from packaging.version import Version
            version = Version(transformers.__version__)
            is_5x = version >= Version("5.0.0")
        except Exception:
            parts = transformers.__version__.split(".")
            is_5x = int(parts[0]) >= 5 if parts and parts[0].isdigit() else False
        if is_5x:
            print("⚠️ Transformers 5.x detected: Qwen3-TTS tokenizer is incompatible.")
            print("   Please downgrade to transformers<=4.57.3 (see requirements.txt).")
    except Exception:
        pass

def check_ffmpeg_availability():
    """Check ffmpeg availability and log status"""
    try:
        # Load ffmpeg_utils directly by file path to avoid package import issues
        ffmpeg_utils_path = os.path.join(os.path.dirname(__file__), "utils", "ffmpeg_utils.py")
        spec = importlib.util.spec_from_file_location("ffmpeg_utils_module", ffmpeg_utils_path)
        ffmpeg_utils_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ffmpeg_utils_module)

        if ffmpeg_utils_module.FFmpegUtils.is_available():
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
check_dependencies()
print_critical_versions()
warn_transformers_5_unsupported()
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

# CRITICAL FIX FOR ISSUE #191: Clear poisoned utils from sys.modules
# Some custom nodes (e.g., LG_HotReload) have a utils.py file that gets loaded
# into sys.modules['utils'], shadowing our utils/ directory package.
# This causes "No module named 'utils.models'; 'utils' is not a package" errors
# when our code tries to import from utils submodules.
# We must clear it BEFORE loading nodes.py which imports from utils.
if 'utils' in sys.modules:
    utils_module = sys.modules['utils']
    # Check if it's a poisoned utils (single .py file, not a package directory)
    # Real packages have __path__ attribute, single files don't
    if not hasattr(utils_module, '__path__'):
        # It's a single .py file masquerading as utils - this will break our imports
        utils_file = getattr(utils_module, '__file__', 'unknown')
        print(f"\n{'='*80}")
        print(f"⚠️  UTILS NAMESPACE CONFLICT DETECTED")
        print(f"{'='*80}")
        print(f"Another custom node has a 'utils.py' file in sys.modules['utils']:")
        print(f"   Source: {utils_file}")
        print(f"")
        print(f"This conflicts with TTS Audio Suite's 'utils/' package directory.")
        print(f"Removing the conflicting module to allow TTS Audio Suite to load.")
        print(f"")
        print(f"If this causes issues with another custom node, that node should:")
        print(f"• Use relative imports (from .utils import X)")
        print(f"• Or use a unique name instead of 'utils'")
        print(f"{'='*80}\n")

        # Delete the poisoned utils module and any attempted submodules
        del sys.modules['utils']
        to_delete = [key for key in sys.modules.keys() if key.startswith('utils.')]
        for key in to_delete:
            del sys.modules[key]

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

# Define web directory for JavaScript files (settings UI)
WEB_DIRECTORY = "./web"


def _debug_timing_log(label: str, start_time: float, extra: str = ""):
    """Print consistent startup timing diagnostics for debug branches."""
    elapsed = time.perf_counter() - start_time
    suffix = f" | {extra}" if extra else ""
    print(f"⏱️ [startup-debug] {label}: {elapsed:.3f}s{suffix}")

# Register API endpoint for widget data
def setup_api_routes():
    """Setup API routes for widget communication"""
    try:
        import json
        from server import PromptServer
        from aiohttp import web

        @PromptServer.instance.routes.get("/api/tts-audio-suite/available-characters")
        async def get_available_characters_endpoint(request):
            """API endpoint to get available TTS character voices including aliases"""
            start_time = time.perf_counter()
            try:
                # Load voice discovery directly by file path to avoid package import issues
                voice_discovery_path = os.path.join(os.path.dirname(__file__), "utils", "voice", "discovery.py")
                spec = importlib.util.spec_from_file_location("voice_discovery_module", voice_discovery_path)
                voice_discovery_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(voice_discovery_module)

                characters = list(voice_discovery_module.get_available_characters())
                # Also get character aliases
                aliases = list(voice_discovery_module.voice_discovery._character_aliases.keys()) if hasattr(voice_discovery_module.voice_discovery, '_character_aliases') else []
                # Combine and deduplicate
                all_chars = sorted(set(characters + aliases))
                _debug_timing_log("/available-characters", start_time, f"count={len(all_chars)}")
                return web.json_response({"characters": all_chars})
            except Exception as e:
                print(f"⚠️ Error retrieving available characters: {e}")
                _debug_timing_log("/available-characters", start_time, f"error={type(e).__name__}")
                return web.json_response({"characters": [], "error": str(e)})

        @PromptServer.instance.routes.get("/api/tts-audio-suite/available-languages")
        async def get_available_languages_endpoint(request):
            """API endpoint to get available language codes from the canonical language mapper"""
            start_time = time.perf_counter()
            try:
                # Load language_mapper directly by file path to avoid package import issues
                language_mapper_path = os.path.join(os.path.dirname(__file__), "utils", "models", "language_mapper.py")
                spec = importlib.util.spec_from_file_location("language_mapper_module", language_mapper_path)
                language_mapper_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(language_mapper_module)

                # Get all unique canonical language codes (the values in LANGUAGE_ALIASES)
                languages = sorted(set(language_mapper_module.LANGUAGE_ALIASES.values()))
                _debug_timing_log("/available-languages", start_time, f"count={len(languages)}")
                return web.json_response({"languages": languages})
            except Exception as e:
                print(f"⚠️ Error retrieving available languages: {e}")
                _debug_timing_log("/available-languages", start_time, f"error={type(e).__name__}")
                # Fallback list
                return web.json_response({"languages": ["en", "de", "fr", "ja", "es", "it", "pt", "th", "no"], "error": str(e)})

        @PromptServer.instance.routes.get("/api/tts-audio-suite/voice-input-devices")
        async def get_voice_input_devices_endpoint(request):
            """Return input devices without risking a main-process PortAudio hang."""
            start_time = time.perf_counter()
            try:
                import json
                import subprocess

                probe_script = r"""
import json
import sounddevice as sd

devices = []
seen = set()
for device in sd.query_devices():
    try:
        max_input_channels = int(device.get("max_input_channels", 0))
    except Exception:
        max_input_channels = 0
    if max_input_channels <= 0:
        continue

    name = str(device.get("name", "")).strip()
    if not name or name in seen:
        continue

    seen.add(name)
    devices.append(name)

print(json.dumps({"devices": devices}))
"""

                result = subprocess.run(
                    [sys.executable, "-c", probe_script],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=False,
                )

                if result.returncode != 0:
                    stderr = (result.stderr or "").strip()
                    stdout = (result.stdout or "").strip()
                    error_message = stderr or stdout or f"device probe exited with code {result.returncode}"
                    return web.json_response({"devices": [], "error": error_message}, status=500)

                payload = json.loads(result.stdout or "{}")
                devices = payload.get("devices", [])
                if not isinstance(devices, list):
                    devices = []

                _debug_timing_log("/voice-input-devices", start_time, f"count={len(devices)}")
                return web.json_response({"devices": devices})
            except subprocess.TimeoutExpired:
                _debug_timing_log("/voice-input-devices", start_time, "timeout")
                return web.json_response(
                    {"devices": [], "error": "Timed out while probing audio input devices. Leaving the dropdown on system default avoids startup hangs."},
                    status=504,
                )
            except Exception as e:
                print(f"⚠️ Error retrieving voice input devices: {e}")
                _debug_timing_log("/voice-input-devices", start_time, f"error={type(e).__name__}")
                return web.json_response({"devices": [], "error": str(e)}, status=500)

        @PromptServer.instance.routes.post("/api/tts-audio-suite/settings")
        async def set_inline_tag_settings_endpoint(request):
            """API endpoint to receive settings from frontend for inline edit tags and restore VC"""
            start_time = time.perf_counter()
            print("🔧 Settings endpoint called")  # Immediate print to verify endpoint is reached
            try:
                data = await request.json()
                precision = data.get("precision", "auto")
                device = data.get("device", "auto")
                vc_engine = data.get("vc_engine", "chatterbox_23lang")
                cosyvoice_variant = data.get("cosyvoice_variant", "RL")

                print(f"🔧 Received settings: precision={precision}, device={device}, vc_engine={vc_engine}, cosyvoice_variant={cosyvoice_variant}")

                # Import edit_post_processor using normal import to ensure we get the same module instance
                # that will be used during workflow execution
                # CRITICAL: Must use the same module instance, not create a new one via importlib!
                try:
                    import_start = time.perf_counter()
                    from utils.audio import edit_post_processor as edit_post_processor_module
                    _debug_timing_log("/settings import edit_post_processor", import_start, "import_mode=normal")
                except ImportError:
                    # Fallback: Load directly by file path if normal import fails
                    edit_post_processor_path = os.path.join(os.path.dirname(__file__), "utils", "audio", "edit_post_processor.py")
                    spec = importlib.util.spec_from_file_location("utils.audio.edit_post_processor", edit_post_processor_path)
                    edit_post_processor_module = importlib.util.module_from_spec(spec)
                    sys.modules["utils.audio.edit_post_processor"] = edit_post_processor_module  # Register in sys.modules!
                    import_start = time.perf_counter()
                    spec.loader.exec_module(edit_post_processor_module)
                    _debug_timing_log("/settings import edit_post_processor", import_start, "import_mode=fallback")

                # Store in global settings that edit_post_processor can access
                edit_post_processor_module.set_inline_tag_settings(precision=precision, device=device, vc_engine=vc_engine, cosyvoice_variant=cosyvoice_variant)

                _debug_timing_log("/settings", start_time, "status=success")
                return web.json_response({"status": "success", "precision": precision, "device": device, "vc_engine": vc_engine, "cosyvoice_variant": cosyvoice_variant})
            except Exception as e:
                print(f"⚠️ Error setting inline tag settings: {e}")
                _debug_timing_log("/settings", start_time, f"error={type(e).__name__}")
                return web.json_response({"status": "error", "error": str(e)})

        @PromptServer.instance.routes.get("/api/tts-audio-suite/voice-preview")
        async def get_voice_preview_endpoint(request):
            """
            Stream selected Character Voices dropdown audio for browser preview playback.

            Query params:
            - voice_name: exact dropdown key from get_available_voices()
            """
            try:
                voice_name = request.query.get("voice_name", "").strip()
                if not voice_name or voice_name == "none":
                    return web.json_response({"error": "voice_name is required and cannot be 'none'"}, status=400)

                # Load voice discovery directly by file path to avoid package import issues
                voice_discovery_path = os.path.join(os.path.dirname(__file__), "utils", "voice", "discovery.py")
                spec = importlib.util.spec_from_file_location("voice_discovery_module", voice_discovery_path)
                voice_discovery_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(voice_discovery_module)

                # Use cached discovery for fast preview playback.
                voice_discovery_module.get_available_voices(force_refresh=False)
                audio_path, _ = voice_discovery_module.load_voice_reference(voice_name)

                if not audio_path or not os.path.exists(audio_path):
                    return web.json_response({"error": f"Voice file not found: {voice_name}"}, status=404)

                # Direct stream of resolved local audio file.
                response = web.FileResponse(path=audio_path)
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
                return response
            except Exception as e:
                print(f"⚠️ Error serving voice preview audio: {e}")
                return web.json_response({"error": str(e)}, status=500)
    except Exception as e:
        print(f"⚠️ Could not setup API routes: {e}")

# Setup API routes when extension loads
setup_api_routes()

# nodes.py already handles all the startup output and status reporting
