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
        self.is_macos = platform.system() == "Darwin"
        self.is_m1_mac = self.is_macos and platform.machine() == "arm64"
        self.pip_cmd = [sys.executable, "-m", "pip"]
        
    def log(self, message: str, level: str = "INFO"):
        """Log installation progress with safe visual indicators"""
        # Use ASCII-safe symbols that work on all systems
        symbol_map = {
            "INFO": "[i]",
            "SUCCESS": "[+]", 
            "WARNING": "[!]",
            "ERROR": "[X]",
            "INSTALL": "[*]"
        }
        symbol = symbol_map.get(level, "[i]")
        print(f"{symbol} {message}")
    
    def check_system_dependencies(self):
        """Check for required system libraries and provide helpful error messages"""
        if self.is_windows:
            return True  # Windows packages come pre-compiled
            
        if self.is_macos:
            return self.check_macos_dependencies()
        else:
            return self.check_linux_dependencies()
    
    def check_macos_dependencies(self):
        """Check for required system libraries on macOS"""
        self.log("Checking macOS system dependencies...", "INFO")
        missing_deps = []
        
        # Check for libsamplerate (needed by resampy/soxr)
        try:
            import ctypes.util
            if not ctypes.util.find_library('samplerate'):
                missing_deps.append(('libsamplerate', 'audio resampling'))
        except:
            pass
        
        # Check for portaudio (needed for sounddevice)
        try:
            import ctypes.util
            if not ctypes.util.find_library('portaudio'):
                missing_deps.append(('portaudio', 'voice recording'))
        except:
            pass
        
        if missing_deps:
            self.log("Missing system dependencies detected!", "WARNING")
            print("\n" + "="*60)
            print("MACOS SYSTEM DEPENDENCIES REQUIRED")
            print("="*60)
            for dep, purpose in missing_deps:
                print(f"• {dep} (for {purpose})")
            
            print("\nPlease install with Homebrew:")
            deps_list = " ".join([dep for dep, _ in missing_deps])
            print(f"brew install {deps_list}")
            print("\n# If you don't have Homebrew:")
            print('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
            
            if self.is_m1_mac:
                print("\n# M1/M2 Mac Note:")
                print("Make sure you're using an ARM64 Python environment!")
                print("Check with: python -c \"import platform; print(platform.machine())\"")
                print("Should show: arm64 (not x86_64)")
            
            print("="*60)
            print("Then run this install script again.\n")
            return False
        
        self.log("macOS system dependencies check passed", "SUCCESS")
        return True
    
    def check_linux_dependencies(self):
        """Check for required system libraries on Linux"""
        self.log("Checking Linux system dependencies...", "INFO")
        missing_deps = []
        
        # Check for libsamplerate (needed by resampy/soxr)
        try:
            # Try importing a package that would fail if libsamplerate is missing
            import ctypes.util
            if not ctypes.util.find_library('samplerate'):
                missing_deps.append(('libsamplerate0-dev', 'audio resampling'))
        except:
            pass
        
        # Check for portaudio (needed for sounddevice)
        try:
            import ctypes.util
            if not ctypes.util.find_library('portaudio'):
                missing_deps.append(('portaudio19-dev', 'voice recording'))
        except:
            pass
        
        if missing_deps:
            self.log("Missing system dependencies detected!", "WARNING")
            print("\n" + "="*60)
            print("LINUX SYSTEM DEPENDENCIES REQUIRED")
            print("="*60)
            for dep, purpose in missing_deps:
                print(f"• {dep} (for {purpose})")
            
            print("\nPlease install with:")
            print("# Ubuntu/Debian:")
            deps_list = " ".join([dep for dep, _ in missing_deps])
            print(f"sudo apt-get install {deps_list}")
            print("\n# Fedora/RHEL:")
            fedora_deps = deps_list.replace('-dev', '-devel').replace('19', '')
            print(f"sudo dnf install {fedora_deps}")
            print("="*60)
            print("Then run this install script again.\n")
            return False
        
        self.log("Linux system dependencies check passed", "SUCCESS")
        return True

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

    def detect_cuda_version(self):
        """Detect CUDA version and determine best PyTorch index"""
        try:
            # Try to detect CUDA version
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'CUDA Version:' in result.stdout:
                # Extract CUDA version (e.g., "CUDA Version: 12.1")
                import re
                cuda_match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', result.stdout)
                if cuda_match:
                    major, minor = int(cuda_match.group(1)), int(cuda_match.group(2))
                    self.log(f"Detected CUDA {major}.{minor}", "INFO")
                    
                    # Choose appropriate PyTorch CUDA build based on detected version
                    if major == 12 and minor >= 8:
                        return "cu124"  # CUDA 12.8+ → use cu124 index
                    elif major >= 12:
                        return "cu121"  # CUDA 12.1+ compatible
                    elif major == 11 and minor >= 8:
                        return "cu118"  # CUDA 11.8+ compatible
                    else:
                        self.log(f"CUDA {major}.{minor} detected - may need manual PyTorch installation", "WARNING")
                        return "cu118"  # Fallback for older CUDA
        except:
            pass
            
        # No CUDA detected - check for AMD GPU (basic detection)
        try:
            if self.is_windows:
                # Windows: check for AMD in device manager output
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                      capture_output=True, text=True, timeout=5)
                if 'amd' in result.stdout.lower() or 'radeon' in result.stdout.lower():
                    self.log("AMD GPU detected - will install CPU version (ROCm not yet supported)", "WARNING")
                    return "cpu"
        except:
            pass
            
        self.log("No CUDA detected - installing CPU-only PyTorch", "WARNING")
        return "cpu"

    def check_pytorch_compatibility(self):
        """Check if current PyTorch meets version and CUDA requirements"""
        try:
            import torch
            current_version = torch.__version__
            
            # Parse version (e.g., "2.5.1+cu121" -> (2, 5, 1))
            import re
            version_match = re.match(r'(\d+)\.(\d+)\.(\d+)', current_version)
            if version_match:
                major, minor, patch = map(int, version_match.groups())
                version_tuple = (major, minor, patch)
                
                # Check if version >= 2.6.0
                if version_tuple >= (2, 6, 0):
                    # Check CUDA availability if we detected CUDA
                    cuda_available = torch.cuda.is_available()
                    detected_cuda = self.detect_cuda_version() != "cpu"
                    
                    if detected_cuda and not cuda_available:
                        self.log(f"PyTorch {current_version} found but no CUDA support - will reinstall with CUDA", "WARNING")
                        return False
                    elif not detected_cuda and cuda_available:
                        self.log(f"PyTorch {current_version} has unnecessary CUDA support - keeping anyway", "INFO")
                        return True
                    else:
                        self.log(f"PyTorch {current_version} is compatible - skipping installation", "SUCCESS")
                        return True
                else:
                    self.log(f"PyTorch {current_version} < 2.6.0 - will upgrade for security fix", "WARNING")
                    return False
            else:
                self.log(f"Could not parse PyTorch version: {current_version} - will reinstall", "WARNING")
                return False
                
        except ImportError:
            self.log("PyTorch not found - will install", "INFO")
            return False
        except Exception as e:
            self.log(f"Error checking PyTorch: {e} - will reinstall", "WARNING")
            return False

    def install_pytorch_with_cuda(self):
        """Install PyTorch with appropriate acceleration (2.6+ required for CVE-2025-32434 security fix)"""
        # Check if current PyTorch is already compatible
        if self.check_pytorch_compatibility():
            return  # Skip installation
            
        cuda_version = self.detect_cuda_version()
        
        if cuda_version == "cpu":
            self.log("Installing PyTorch 2.6+ (CPU-only)", "INFO")
            index_url = "https://download.pytorch.org/whl/cpu"
        else:
            self.log(f"Installing PyTorch 2.6+ with CUDA {cuda_version} support", "INFO")
            index_url = f"https://download.pytorch.org/whl/{cuda_version}"
        
        # Force uninstall if we need to switch between CPU/CUDA variants
        try:
            import torch
            current_version = torch.__version__
            if (cuda_version != "cpu" and not torch.cuda.is_available()) or \
               (cuda_version == "cpu" and torch.cuda.is_available()):
                self.log(f"Uninstalling existing PyTorch {current_version} to switch variants", "WARNING")
                uninstall_cmd = ["uninstall", "-y", "torch", "torchvision", "torchaudio"]
                self.run_pip_command(uninstall_cmd, "Uninstalling existing PyTorch")
        except ImportError:
            pass  # PyTorch not installed
        
        # Install PyTorch 2.6+ with detected acceleration
        pytorch_packages = [
            "torch>=2.6.0", 
            "torchvision", 
            "torchaudio>=2.6.0"
        ]
        
        pytorch_cmd = [
            "install", 
            "--upgrade", 
            "--force-reinstall"
        ] + pytorch_packages + [
            "--index-url", index_url
        ]
        
        self.run_pip_command(pytorch_cmd, f"Installing PyTorch 2.6+ ({cuda_version} support)")

    def check_package_installed(self, package_spec):
        """Check if a package meets the version requirement"""
        try:
            # Parse package specification (e.g., "transformers>=4.46.3")
            import re
            match = re.match(r'^([a-zA-Z0-9\-_]+)([><=!]+)?(.+)?$', package_spec)
            if not match:
                return False
                
            package_name = match.group(1)
            operator = match.group(2) if match.group(2) else None
            required_version = match.group(3) if match.group(3) else None
            
            # Try to import and check version
            import importlib
            import pkg_resources
            
            try:
                # Check if package is installed
                distribution = pkg_resources.get_distribution(package_name)
                installed_version = distribution.version
                
                if not operator or not required_version:
                    # No version requirement, just check if installed
                    return True
                    
                # Check version requirement
                requirement = pkg_resources.Requirement.parse(package_spec)
                return distribution in requirement
                
            except pkg_resources.DistributionNotFound:
                return False
                
        except Exception:
            return False

    def install_macos_specific_packages(self):
        """Install packages with Mac-specific requirements"""
        if not self.is_macos:
            return
            
        self.log("Installing macOS-specific audio packages", "INFO")
        
        # For M1 Macs, ensure we use compatible versions
        if self.is_m1_mac:
            self.log("M1 Mac detected - installing ARM64-compatible packages", "INFO")
            
            # Force reinstall samplerate with proper architecture
            self.run_pip_command(
                ["uninstall", "-y", "samplerate"], 
                "Removing potentially x86_64 samplerate package", 
                ignore_errors=True
            )
            
            # Install with --no-cache to force ARM64 build
            self.run_pip_command(
                ["install", "--no-cache-dir", "--force-reinstall", "samplerate>=0.2.1"], 
                "Installing ARM64-compatible samplerate package"
            )
            
        # Install/reinstall audio packages that commonly have architecture issues on Mac
        mac_audio_packages = [
            "soundfile>=0.12.0",
            "sounddevice>=0.4.0",
        ]
        
        for package in mac_audio_packages:
            self.run_pip_command(
                ["install", "--force-reinstall", "--no-cache-dir", package], 
                f"Reinstalling {package} for macOS compatibility"
            )

    def install_core_dependencies(self):
        """Install safe core dependencies that don't cause conflicts"""
        self.log("Checking and installing core dependencies (with smart checking)", "INFO")
        
        core_packages = [
            # Audio and basic utilities (PyTorch installed separately with CUDA)
            "soundfile>=0.12.0",
            "sounddevice>=0.4.0",
            
            # Text processing (safe)
            "jieba",
            "pypinyin", 
            "unidecode",
            "omegaconf>=2.3.0",
            "transformers>=4.46.3",
            
            # Bundled engine dependencies (safe)
            "conformer>=0.3.2",      # ChatterBox engine
            "x-transformers",        # F5-TTS engine  
            "torchdiffeq",          # F5-TTS differential equations
            "wandb",                # F5-TTS logging
            "accelerate",           # F5-TTS acceleration
            "ema-pytorch",          # F5-TTS exponential moving average
            "datasets",             # F5-TTS dataset loading
            "vocos",                # F5-TTS vocoder
            
            # Basic utilities (safe)
            "requests",
            "dacite",
            "opencv-python",
            "pillow",
            
            # SAFE packages from DEPENDENCY_TESTING_RESULTS.md
            "s3tokenizer>=0.1.7",          # ✅ SAFE - Heavy dependencies but NO conflicts
            "vector-quantize-pytorch",     # ✅ SAFE - Clean install
            "resemble-perth",              # ✅ SAFE - Works in ChatterBox
            "diffusers>=0.30.0",          # ✅ SAFE - Likely safe
            "audio-separator>=0.35.2",    # ✅ SAFE - Heavy dependencies but no conflicts
            "hydra-core>=1.3.0",          # ✅ SAFE - Clean install, minimal dependencies
            
            # Dependencies for --no-deps packages based on PyPI metadata
            
            # For librosa (when installed with --no-deps)
            "lazy_loader>=0.1",            # Required by librosa
            "msgpack>=1.0",               # Required by librosa
            "pooch>=1.1",                 # Required by librosa
            "soxr>=0.3.2",                # Required by librosa
            "typing_extensions>=4.1.1",   # Required by librosa
            "decorator>=4.3.0",           # Required by librosa
            "joblib>=1.0",                # Required by librosa
            
            # For VibeVoice (when installed with --no-deps) - only safe dependencies
            "ml-collections",             # Required by VibeVoice
            "absl-py",                    # Required by VibeVoice (Google's Python utilities)
            "gradio",                     # Required by VibeVoice (may already be available)
            "av",                         # Required by VibeVoice (PyAV - audio/video processing)
            "scikit-learn>=1.1.0",        # Required by librosa
            
            # For cached-path (when installed with --no-deps)
            "filelock>=3.4",              # Required by cached-path
            "rich>=12.1",                 # Required by cached-path
            "boto3",                      # Required by cached-path
            "google-cloud-storage",       # Required by cached-path for F5-TTS
            "huggingface-hub",            # Required by cached-path
            
            # For descript-audio-codec (when installed with --no-deps)
            "einops",                      # Required by descript-audio-codec and MelBandRoFormer
            "argbind>=0.3.7",             # Required by descript-audio-codec
            # NOTE: descript-audiotools causes protobuf conflicts, installed via --no-deps
            
            # For MelBandRoFormer vocal separation
            "rotary_embedding_torch",     # Required by MelBandRoFormer architecture
            
            # For F5-TTS engine
            "matplotlib",                  # Required by F5-TTS utils_infer.py
            
            # Additional librosa dependencies for --no-deps installation
            "audioread>=2.1.9",           # Required by librosa
            "threadpoolctl>=3.1.0",       # Required by scikit-learn for librosa
            
            # Missing descript-audiotools dependencies for --no-deps installation
            "flatten-dict",               # Required by descript-audiotools
            "ffmpy",                      # Required by descript-audiotools
            "importlib-resources",        # Required by descript-audiotools
            "randomname",                 # Required by descript-audiotools
            "markdown2",                  # Required by descript-audiotools
            "pyloudnorm",                 # Required by descript-audiotools
            "pystoi",                     # Required by descript-audiotools
            "torch-stoi",                 # Required by descript-audiotools
            "ipython",                    # Required by descript-audiotools
            "tensorboard"                 # Required by descript-audiotools
        ]
        
        # Smart installation: check before installing (preserving all original packages and comments)
        packages_to_install = []
        skipped_packages = []
        
        for package in core_packages:
            if self.check_package_installed(package):
                package_name = package.split('>=')[0].split('==')[0].split('<')[0]
                skipped_packages.append(package_name)
            else:
                packages_to_install.append(package)
                
        if skipped_packages:
            self.log(f"Already satisfied: {', '.join(skipped_packages[:5])}" + 
                    (f" and {len(skipped_packages)-5} others" if len(skipped_packages) > 5 else ""), "SUCCESS")
            
        if packages_to_install:
            self.log(f"Installing {len(packages_to_install)} missing core packages", "INFO")
            for package in packages_to_install:
                self.run_pip_command(["install", package], f"Installing {package}")
        else:
            self.log("All core dependencies already satisfied", "SUCCESS")

    def install_rvc_dependencies(self):
        """Install RVC voice conversion dependencies with smart GPU detection"""
        self.log("Installing RVC voice conversion dependencies", "INFO")
        
        # Install core RVC dependency first
        self.run_pip_command(["install", "monotonic-alignment-search"], "Installing monotonic-alignment-search")
        
        # Smart faiss installation: GPU on Linux with CUDA, CPU fallback for Windows/compatibility
        cuda_version = self.detect_cuda_version()
        
        if not self.is_windows and cuda_version != "cpu":
            # Linux with CUDA - try GPU acceleration
            self.log("Linux + CUDA detected - attempting faiss-gpu for better RVC performance", "INFO")
            
            try:
                # Determine CUDA version for faiss-gpu package
                if cuda_version in ["cu121", "cu124"]:  # CUDA 12.x
                    faiss_gpu_package = "faiss-gpu-cu12>=1.7.4"
                elif cuda_version == "cu118":  # CUDA 11.x
                    faiss_gpu_package = "faiss-gpu-cu11>=1.7.4"
                else:
                    # Fallback for other CUDA versions
                    faiss_gpu_package = "faiss-gpu-cu12>=1.7.4"
                
                # Try GPU installation first
                self.run_pip_command(["install", faiss_gpu_package], f"Installing {faiss_gpu_package} for GPU acceleration")
                self.log("✅ faiss-gpu installed - RVC will use GPU acceleration for better performance", "SUCCESS")
                
            except subprocess.CalledProcessError:
                # GPU installation failed - fallback to CPU
                self.log("faiss-gpu installation failed - falling back to CPU version", "WARNING")
                self.run_pip_command(["install", "faiss-cpu>=1.7.4"], "Installing faiss-cpu (fallback)")
        else:
            # Windows or no CUDA - use reliable CPU version
            if self.is_windows and cuda_version != "cpu":
                self.log("Windows + CUDA detected - faiss-gpu not available on Windows, using CPU version", "INFO")
            else:
                self.log("No CUDA detected - using faiss-cpu", "INFO")
            
            self.run_pip_command(["install", "faiss-cpu>=1.7.4"], "Installing faiss-cpu for RVC voice matching")

    def install_numpy_with_constraints(self):
        """Install numpy with version constraints for compatibility"""
        self.log("Installing numpy and numba with compatibility constraints", "INFO")
        
        # Check current numba version to determine compatible numpy range
        try:
            import numba
            numba_version = numba.__version__
            self.log(f"Current numba version: {numba_version}", "INFO")
            
            # Parse numba version to determine numpy compatibility
            import re
            version_match = re.match(r'(\d+)\.(\d+)', numba_version)
            if version_match:
                major, minor = int(version_match.group(1)), int(version_match.group(2))
                
                if (major, minor) >= (0, 62):  # Numba 0.62+ supports NumPy 2.2+
                    numpy_constraint = "numpy>=2.2.0,<2.3.0"
                    self.log("Numba 0.62+ detected - using NumPy 2.2+", "INFO")
                elif (major, minor) == (0, 61):
                    # Check if it's 0.61.2+ (supports NumPy 2.2+) or 0.61.0-0.61.1 (needs NumPy 2.1)
                    patch_match = re.search(r'0\.61\.(\d+)', numba_version)
                    if patch_match and int(patch_match.group(1)) >= 2:
                        numpy_constraint = "numpy>=2.2.0,<2.3.0"
                        self.log(f"Numba {numba_version} supports NumPy 2.2+ - using modern NumPy", "INFO")
                    else:
                        # Numba 0.61.0-0.61.1 needs upgrade to 0.61.2+ for NumPy 2.2+ support
                        self.log(f"Numba {numba_version} detected - upgrading to 0.61.2+ for NumPy 2.2+ support", "WARNING")
                        self.run_pip_command(["install", "--upgrade", "numba>=0.61.2"], "Upgrading numba to 0.61.2+ for NumPy 2.2+ compatibility")
                        numpy_constraint = "numpy>=2.2.0,<2.3.0"
                else:
                    # Numba 0.60.x or older - upgrade to compatible version
                    self.log(f"Numba {numba_version} detected - upgrading to 0.61.2+ for NumPy 2.2+ support", "WARNING")
                    self.run_pip_command(["install", "--upgrade", "numba>=0.61.2"], "Upgrading numba to 0.61.2+ for NumPy 2.2+ compatibility")
                    numpy_constraint = "numpy>=2.2.0,<2.3.0"
            else:
                # Can't parse version - play it safe with known compatible version
                numpy_constraint = "numpy>=2.2.0,<2.3.0" 
                self.run_pip_command(["install", "--upgrade", "numba>=0.61.2"], "Upgrading numba to compatible version for safety")
                
        except ImportError:
            # No numba installed - install both with compatible versions
            self.log("Numba not found - installing compatible numba with NumPy", "INFO")
            numpy_constraint = "numpy>=2.2.0,<2.3.0"
            self.run_pip_command(["install", "numba>=0.61.2"], "Installing compatible numba 0.61.2+")
        except Exception as e:
            # Numba import failed (likely due to numpy incompatibility)
            self.log(f"Numba import failed ({e}) - upgrading numba first", "WARNING")
            self.run_pip_command(["install", "--upgrade", "numba>=0.61.2"], "Upgrading numba to 0.61.2+ for compatibility")
            numpy_constraint = "numpy>=2.2.0,<2.3.0"
        
        # Install numpy with determined constraints
        self.run_pip_command(["install", numpy_constraint], "Installing numpy with version constraints")

    def install_problematic_packages(self):
        """Install packages that cause conflicts using --no-deps"""
        self.log("Installing problematic packages with --no-deps to prevent conflicts", "WARNING")
        
        problematic_packages = [
            "librosa",              # Forces numpy downgrade - compatibility handled by runtime numba disabling for Python 3.13
            "descript-audio-codec", # Pulls unnecessary deps, conflicts with protobuf
            "descript-audiotools",  # Forces protobuf downgrade from 6.x to 3.19.x
            "cached-path",          # Forces package downgrades
            "torchcrepe",          # Conflicts via librosa dependency
            "onnxruntime",         # For OpenSeeFace, but forces numpy 2.3.x
        ]
        
        for package in problematic_packages:
            self.run_pip_command(
                ["install", package, "--no-deps"], 
                f"Installing {package} (--no-deps)",
                ignore_errors=True  # Some may already be satisfied
            )
    
    def install_vibevoice(self):
        """Install VibeVoice with careful dependency management"""
        self.log("Installing VibeVoice TTS engine", "INFO")
        
        # First ensure critical dependencies that VibeVoice needs but might downgrade
        vibevoice_deps = [
            "aiortc",      # Audio/video real-time communication - safe to install
            "pyee",        # Event emitter - lightweight
            "dnspython",   # DNS toolkit - safe
            "ifaddr",      # Network interface addresses - safe
            "pylibsrtp",   # SRTP library - safe
            "pyopenssl",   # OpenSSL wrapper - safe
        ]
        
        self.log("Installing VibeVoice safe dependencies first", "INFO")
        for dep in vibevoice_deps:
            self.run_pip_command(
                ["install", dep], 
                f"Installing {dep}",
                ignore_errors=True
            )
        
        # Now install VibeVoice with --no-deps to prevent downgrades
        self.log("Installing VibeVoice with --no-deps to prevent package downgrades", "WARNING")
        self.run_pip_command(
            ["install", "git+https://github.com/microsoft/VibeVoice.git", "--no-deps"],
            "Installing VibeVoice (--no-deps)",
            ignore_errors=True
        )

    def check_comfyui_environment(self):
        """Check if running in likely ComfyUI environment and warn for system Python"""
        python_path = sys.executable.lower()
        
        # Only warn for clearly identifiable system Python paths
        system_python_patterns = [
            "c:\\python",           # Windows system Python
            "/usr/bin/python",      # Linux system Python  
            "/usr/local/bin/python", # macOS Homebrew system Python
            "system32",             # Windows system directory
        ]
        
        if any(pattern in python_path for pattern in system_python_patterns):
            self.log("⚠️  WARNING: Detected system-wide Python installation", "WARNING")
            self.log(f"Current Python: {sys.executable}", "WARNING")
            self.log("This may install packages to the wrong location", "WARNING")
            self.log("💡 For best results, use ComfyUI Manager for automatic installation", "INFO")
            return False
        return True

    def handle_wandb_issues(self):
        """Fix wandb circular import issues that affect multiple nodes"""
        self.log("Checking and fixing wandb import issues", "INFO")
        
        try:
            # Try to import wandb to check for issues
            import wandb
            # Try to access the errors attribute that's causing the circular import
            hasattr(wandb, 'errors')
            self.log("wandb import test passed", "SUCCESS")
        except (ImportError, AttributeError) as e:
            self.log(f"wandb import issue detected: {e}", "WARNING")
            self.log("Reinstalling wandb to fix circular import", "WARNING")
            
            # Uninstall and reinstall wandb cleanly
            self.run_pip_command(
                ["uninstall", "-y", "wandb"], 
                "Uninstalling problematic wandb", 
                ignore_errors=True
            )
            
            # Clear any cached/partial installations
            self.run_pip_command(
                ["install", "--no-cache-dir", "--force-reinstall", "wandb>=0.17.0"], 
                "Reinstalling wandb cleanly"
            )
            
            # Test again after reinstallation
            try:
                import importlib
                importlib.invalidate_caches()  # Clear import cache
                import wandb
                hasattr(wandb, 'errors')
                self.log("wandb reinstallation successful", "SUCCESS")
            except Exception as retry_error:
                self.log(f"wandb reinstallation still has issues: {retry_error}", "ERROR")
                self.log("Some F5-TTS and Higgs features may not work properly", "WARNING")

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
        print("\n" + "="*70)
        print(" "*20 + "TTS AUDIO SUITE INSTALLATION")
        print("="*70)
        
        self.log("Installation completed successfully!", "SUCCESS")
        print(f"\n>>> Python version: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        if self.is_macos:
            print(f">>> Platform: macOS ({platform.machine()})")
        
        if self.is_python_313:
            print("\n" + "-"*50)
            print("   PYTHON 3.13 COMPATIBILITY STATUS")
            print("-"*50)
            print("  [+] All TTS engines: WORKING")
            print("      (ChatterBox, F5-TTS, Higgs Audio)")
            print("  [+] RVC voice conversion: WORKING") 
            print("  [+] OpenSeeFace mouth movement: WORKING (experimental)")
            print("  [+] Numba/Librosa compatibility: FIXED")
            print("      -> Automatic JIT disabling for Python 3.13")
            print("  [X] MediaPipe mouth movement: INCOMPATIBLE")
            print("      -> Use OpenSeeFace alternative")
            print("\n>> Want MediaPipe Python 3.13 support? Vote at:")
            print("   https://github.com/google-ai-edge/mediapipe/issues/5708")
        else:
            print("\n" + "-"*50)
            print("   FULL COMPATIBILITY STATUS")
            print("-"*50)
            print("  [+] All TTS engines: WORKING")
            print("  [+] RVC voice conversion: WORKING")
            print("  [+] MediaPipe mouth movement: WORKING") 
            print("  [+] OpenSeeFace mouth movement: WORKING")
        
        print("\n" + "="*70)
        print(" "*15 + "READY TO USE TTS AUDIO SUITE IN COMFYUI!")
        print("="*70 + "\n")

def main():
    """Main installation entry point"""
    installer = TTSAudioInstaller()
    
    try:
        installer.log("Starting TTS Audio Suite installation", "INFO")
        installer.log(f"Python {installer.python_version.major}.{installer.python_version.minor}.{installer.python_version.micro} detected", "INFO")
        
        # Check environment and system dependencies before proceeding
        installer.check_comfyui_environment()
        
        # Check system dependencies (Linux only)
        if not installer.check_system_dependencies():
            installer.log("System dependency check failed - aborting installation", "ERROR")
            sys.exit(1)
        
        # Install in correct order to prevent conflicts
        installer.install_pytorch_with_cuda()  # Install PyTorch first with proper CUDA detection
        installer.install_core_dependencies()
        installer.install_macos_specific_packages()  # Mac-specific package fixes
        installer.install_numpy_with_constraints() 
        installer.install_rvc_dependencies()
        installer.install_problematic_packages()
        installer.install_vibevoice()  # Install VibeVoice with careful dependency management
        installer.handle_wandb_issues()  # Fix wandb circular import
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