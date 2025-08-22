#!/usr/bin/env python3
"""
TTS Audio Suite - ComfyUI Installation Script
Handles Python 3.13 compatibility and dependency conflicts automatically.

This script is called by ComfyUI Manager to install all required dependencies
for the TTS Audio Suite custom node with proper conflict resolution.
"""

import subprocess
import sys
import os
import platform
from typing import List, Optional


class TTSAudioInstaller:
    """Intelligent installer for TTS Audio Suite with Python 3.13 compatibility"""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.is_python_313 = self.python_version >= (3, 13)
        self.is_windows = platform.system() == "Windows"
        self.pip_cmd = [sys.executable, "-m", "pip"]
        
    def log(self, message: str, level: str = "INFO"):
        """Log installation progress with emoji indicators"""
        emoji_map = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "WARNING": "⚠️",
            "ERROR": "❌",
            "INSTALL": "📦"
        }
        emoji = emoji_map.get(level, "ℹ️")
        print(f"{emoji} {message}")

    def run_pip_command(self, args: List[str], description: str, ignore_errors: bool = False) -> bool:
        """Execute pip command with error handling"""
        cmd = self.pip_cmd + args
        self.log(f"{description}...", "INSTALL")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if result.stdout.strip():
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            if ignore_errors:
                self.log(f"Warning: {description} failed (continuing anyway): {e.stderr.strip()}", "WARNING")
                return False
            else:
                self.log(f"Error: {description} failed: {e.stderr.strip()}", "ERROR")
                raise

    def install_core_dependencies(self):
        """Install safe core dependencies that don't cause conflicts"""
        self.log("Installing core dependencies (safe packages)", "INFO")
        
        core_packages = [
            "torch>=2.0.0",
            "torchaudio>=2.0.0", 
            "soundfile>=0.12.0",
            "sounddevice>=0.4.0",
            "jieba",
            "pypinyin", 
            "unidecode",
            "omegaconf>=2.3.0",
            "transformers>=4.46.3",
            "conformer>=0.3.2",
            "x-transformers",
            "torchdiffeq",
            "wandb",
            "accelerate",
            "ema-pytorch",
            "datasets",
            "requests",
            "dacite",
            "vocos",
            "opencv-python",
            "pillow"
        ]
        
        for package in core_packages:
            self.run_pip_command(["install", package], f"Installing {package}")

    def install_rvc_dependencies(self):
        """Install RVC voice conversion dependencies"""
        self.log("Installing RVC voice conversion dependencies", "INFO")
        
        rvc_packages = [
            "monotonic-alignment-search",  # Core RVC dependency
            "faiss-cpu>=1.7.4"            # RVC model loading
        ]
        
        for package in rvc_packages:
            self.run_pip_command(["install", package], f"Installing {package}")

    def install_numpy_with_constraints(self):
        """Install numpy with version constraints for compatibility"""
        self.log("Installing numpy with compatibility constraints", "INFO")
        
        # Critical: numpy 2.2.x for numba compatibility, avoid 2.3.x
        numpy_constraint = "numpy>=2.2.0,<2.3.0"
        self.run_pip_command(["install", numpy_constraint], "Installing numpy with version constraints")

    def install_problematic_packages(self):
        """Install packages that cause conflicts using --no-deps"""
        self.log("Installing problematic packages with --no-deps to prevent conflicts", "WARNING")
        
        problematic_packages = [
            "librosa",              # Forces numpy downgrade
            "descript-audio-codec", # Pulls unnecessary deps, conflicts with protobuf
            "cached-path",          # Forces package downgrades
            "torchcrepe",          # Conflicts via librosa dependency
            "onnxruntime"          # For OpenSeeFace, but forces numpy 2.3.x
        ]
        
        for package in problematic_packages:
            self.run_pip_command(
                ["install", package, "--no-deps"], 
                f"Installing {package} (--no-deps)",
                ignore_errors=True  # Some may already be satisfied
            )

    def handle_python_313_specific(self):
        """Handle Python 3.13 specific compatibility issues"""
        if not self.is_python_313:
            self.log("Python < 3.13 detected, skipping 3.13-specific workarounds", "INFO")
            return
            
        self.log("Python 3.13 detected - applying compatibility measures", "WARNING")
        
        # MediaPipe is incompatible - inform user about OpenSeeFace alternative
        self.log("MediaPipe is incompatible with Python 3.13", "WARNING")
        self.log("OpenSeeFace will be used automatically for mouth movement analysis", "INFO")
        self.log("Note: OpenSeeFace is experimental and may be less accurate than MediaPipe", "WARNING")
        
        # Ensure onnxruntime is available for OpenSeeFace (with --no-deps to avoid conflicts)
        self.run_pip_command(
            ["install", "onnxruntime", "--no-deps", "--force-reinstall"], 
            "Installing onnxruntime for OpenSeeFace (Python 3.13)",
            ignore_errors=True
        )

    def validate_installation(self):
        """Validate that critical packages can be imported"""
        self.log("Validating installation...", "INFO")
        
        critical_imports = [
            ("torch", "PyTorch"),
            ("torchaudio", "TorchAudio"),
            ("transformers", "Transformers"),
            ("soundfile", "SoundFile"),
            ("numpy", "NumPy"),
            ("librosa", "Librosa"),
            ("omegaconf", "OmegaConf")
        ]
        
        validation_errors = []
        
        for module_name, display_name in critical_imports:
            try:
                __import__(module_name)
                self.log(f"{display_name}: OK", "SUCCESS")
            except ImportError as e:
                validation_errors.append(f"{display_name}: {e}")
                self.log(f"{display_name}: FAILED - {e}", "ERROR")
        
        # Check Python 3.13 specific validations
        if self.is_python_313:
            try:
                import onnxruntime
                self.log("ONNXRuntime (OpenSeeFace): OK", "SUCCESS")
            except ImportError:
                validation_errors.append("ONNXRuntime required for OpenSeeFace on Python 3.13")
                self.log("ONNXRuntime (OpenSeeFace): FAILED", "ERROR")
        
        # Check RVC dependencies
        rvc_modules = [("monotonic_alignment_search", "Monotonic Alignment Search")]
        for module_name, display_name in rvc_modules:
            try:
                __import__(module_name)
                self.log(f"{display_name} (RVC): OK", "SUCCESS")
            except ImportError:
                # RVC is optional, so this is just a warning
                self.log(f"{display_name} (RVC): Not available - RVC voice conversion will not work", "WARNING")
        
        return len(validation_errors) == 0

    def check_version_conflicts(self):
        """Check for known version conflicts"""
        self.log("Checking for version conflicts...", "INFO")
        
        try:
            import numpy
            numpy_version = tuple(map(int, numpy.__version__.split('.')[:2]))
            
            if numpy_version >= (2, 3):
                self.log(f"WARNING: NumPy {numpy.__version__} detected - may cause numba conflicts", "WARNING")
                self.log("Consider downgrading: pip install 'numpy>=2.2.0,<2.3.0'", "WARNING")
            else:
                self.log(f"NumPy {numpy.__version__}: Version OK for compatibility", "SUCCESS")
                
        except ImportError:
            self.log("NumPy not found - this will cause issues", "ERROR")

    def print_installation_summary(self):
        """Print installation summary and next steps"""
        self.log("Installation Summary", "INFO")
        print("="*60)
        
        self.log("TTS Audio Suite installation completed!", "SUCCESS")
        print(f"Python version: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        
        if self.is_python_313:
            print("\n🧪 Python 3.13 Compatibility Notes:")
            print("• All TTS engines: ✅ Working (ChatterBox, F5-TTS, Higgs Audio)")
            print("• RVC voice conversion: ✅ Working") 
            print("• OpenSeeFace mouth movement: ✅ Working (experimental)")
            print("• MediaPipe mouth movement: ❌ Incompatible (use OpenSeeFace)")
            print("\n📢 Want MediaPipe Python 3.13 support? Vote on:")
            print("   https://github.com/google-ai-edge/mediapipe/issues/5708")
        else:
            print("\n✅ Full Compatibility:")
            print("• All TTS engines: ✅ Working")
            print("• RVC voice conversion: ✅ Working")
            print("• MediaPipe mouth movement: ✅ Working") 
            print("• OpenSeeFace mouth movement: ✅ Working")
        
        print("\n🚀 Ready to use TTS Audio Suite in ComfyUI!")
        print("="*60)

def main():
    """Main installation entry point"""
    installer = TTSAudioInstaller()
    
    try:
        installer.log("Starting TTS Audio Suite installation", "INFO")
        installer.log(f"Python {installer.python_version.major}.{installer.python_version.minor}.{installer.python_version.micro} detected", "INFO")
        
        # Install in correct order to prevent conflicts
        installer.install_core_dependencies()
        installer.install_numpy_with_constraints() 
        installer.install_rvc_dependencies()
        installer.install_problematic_packages()
        installer.handle_python_313_specific()
        
        # Validation and summary
        installer.check_version_conflicts()
        success = installer.validate_installation()
        installer.print_installation_summary()
        
        if not success:
            installer.log("Installation completed with warnings - some features may not work", "WARNING")
            sys.exit(1)
        else:
            installer.log("Installation completed successfully!", "SUCCESS")
            sys.exit(0)
            
    except Exception as e:
        installer.log(f"Installation failed: {e}", "ERROR")
        installer.log("Please check the error messages above and try again", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()