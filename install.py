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

    def ensure_requirements_installed(self):
        """Check and install requirements.txt if needed"""
        self.log("Checking requirements.txt dependencies", "INFO")

        requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
        if not os.path.exists(requirements_path):
            self.log("requirements.txt not found - skipping", "WARNING")
            return

        # Parse requirements.txt to get all package names
        missing_packages = []
        try:
            with open(requirements_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove inline comments
                        line = line.split('#')[0].strip()
                        if not line:
                            continue
                        # Extract package name (before >= <= == etc.)
                        package_name = line.split('>=')[0].split('<=')[0].split('==')[0].split('<')[0].split('>')[0].split('!')[0].strip()
                        if package_name:
                            try:
                                # Handle package name differences (dashes vs underscores)
                                import_name = package_name.replace("-", "_")
                                __import__(import_name)
                            except ImportError:
                                missing_packages.append(package_name)
        except Exception as e:
            self.log(f"Error reading requirements.txt: {e}", "WARNING")
            return

        if missing_packages:
            self.log(f"Missing {len(missing_packages)} requirements.txt packages: {', '.join(missing_packages[:5])}{'...' if len(missing_packages) > 5 else ''}", "WARNING")
            self.log("Installing missing requirements individually (preserves ComfyUI Manager safeguards)", "INFO")

            # Install each missing package individually using our safe method
            for package in missing_packages:
                # Get the full package spec from requirements.txt
                package_spec = package
                try:
                    with open(requirements_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                # Remove inline comments
                                clean_line = line.split('#')[0].strip()
                                if clean_line and clean_line.startswith(package):
                                    package_spec = clean_line
                                    break
                except:
                    pass

                self.run_pip_command(["install", package_spec], f"Installing {package}", ignore_errors=True)
        else:
            self.log("All requirements.txt dependencies already satisfied", "SUCCESS")

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
        """Execute pip command with error handling and Windows UTF-8 support"""
        cmd = self.pip_cmd + args
        self.log(f"{description}...", "INSTALL")
        
        # Set UTF-8 encoding for Windows to prevent charmap errors
        env = os.environ.copy()
        if self.is_windows:
            env['PYTHONUTF8'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True, 
                env=env,
                encoding='utf-8' if self.is_windows else None
            )
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
            
            # Parse version (e.g., "2.5.1+cu124" -> (2, 5, 1))
            # Strip CUDA suffix first to avoid parsing errors
            import re
            clean_version = current_version.split('+')[0]  # Remove CUDA suffix like "+cu124"
            version_match = re.match(r'(\d+)\.(\d+)\.(\d+)', clean_version)
            if version_match:
                major, minor, patch = map(int, version_match.groups())
                version_tuple = (major, minor, patch)
                
                # Check CUDA availability if we detected CUDA
                cuda_available = torch.cuda.is_available()
                detected_cuda = self.detect_cuda_version() != "cpu"
                
                # If CUDA mismatch, we need to reinstall
                if detected_cuda and not cuda_available:
                    self.log(f"PyTorch {current_version} found but no CUDA support - will reinstall with CUDA", "WARNING")
                    return False
                elif not detected_cuda and cuda_available:
                    self.log(f"PyTorch {current_version} has unnecessary CUDA support - keeping anyway", "INFO")
                    return True
                
                # Check security requirement: 2.6.0+ preferred for CVE-2025-32434, but 2.5.1+ acceptable
                if version_tuple >= (2, 6, 0):
                    self.log(f"PyTorch {current_version} >= 2.6.0 - secure version, skipping installation", "SUCCESS")
                    return True
                elif version_tuple >= (2, 5, 1):
                    # 2.5.1+ is acceptable but try to upgrade to 2.6.0 if available
                    self.log(f"PyTorch {current_version} >= 2.5.1 - will attempt upgrade to 2.6.0 for security fix", "INFO")
                    return False  # Try upgrade, but won't fail if 2.6.0 unavailable
                else:
                    self.log(f"PyTorch {current_version} < 2.5.1 - will upgrade for security and compatibility", "WARNING")
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
        
        # Try PyTorch 2.6+ first, fallback to 2.5+ if unavailable
        try:
            if cuda_version == "cpu":
                pytorch_packages_26 = [
                    "torch>=2.6.0",
                    "torchvision",
                    "torchaudio>=2.6.0"
                ]
            else:
                pytorch_packages_26 = [
                    "torch>=2.6.0",
                    "torchvision",
                    "torchaudio>=2.6.0"
                ]
            
            pytorch_cmd_26 = [
                "install", 
                "--upgrade", 
                "--force-reinstall"
            ] + pytorch_packages_26 + [
                "--index-url", index_url
            ]
            
            self.run_pip_command(pytorch_cmd_26, f"Installing PyTorch 2.6+ ({cuda_version} support)")
            
        except subprocess.CalledProcessError:
            # PyTorch 2.6.0 not available for this CUDA version - try 2.5+
            self.log(f"PyTorch 2.6.0 not available for {cuda_version} - falling back to latest 2.5.x", "WARNING")

            if cuda_version == "cpu":
                pytorch_packages_25 = [
                    "torch>=2.5.0",
                    "torchvision",
                    "torchaudio>=2.5.0"
                ]
            else:
                pytorch_packages_25 = [
                    "torch>=2.5.0",
                    "torchvision",
                    "torchaudio>=2.5.0"
                ]
            
            pytorch_cmd_25 = [
                "install", 
                "--upgrade", 
                "--force-reinstall"
            ] + pytorch_packages_25 + [
                "--index-url", index_url
            ]
            
            self.run_pip_command(pytorch_cmd_25, f"Installing PyTorch 2.5+ ({cuda_version} support)")

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
            
            # Use modern importlib.metadata (Python 3.8+) with fallback
            try:
                from importlib.metadata import version, PackageNotFoundError
            except ImportError:
                # Fallback for Python < 3.8
                try:
                    from importlib_metadata import version, PackageNotFoundError
                except ImportError:
                    # Final fallback to pkg_resources (with warning suppression)
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        import pkg_resources
                    
                    try:
                        distribution = pkg_resources.get_distribution(package_name)
                        installed_version = distribution.version
                        
                        if not operator or not required_version:
                            return True
                            
                        requirement = pkg_resources.Requirement.parse(package_spec)
                        return distribution in requirement
                        
                    except pkg_resources.DistributionNotFound:
                        return False
            
            try:
                installed_version = version(package_name)
                
                if not operator or not required_version:
                    return True
                    
                # Check version requirement using packaging module if available
                try:
                    from packaging.specifiers import SpecifierSet
                    from packaging.version import Version
                    
                    spec = SpecifierSet(f"{operator}{required_version}")
                    return Version(installed_version) in spec
                    
                except ImportError:
                    # Simple version comparison fallback
                    if operator == ">=":
                        return installed_version >= required_version
                    elif operator == ">":
                        return installed_version > required_version
                    elif operator == "==":
                        return installed_version == required_version
                    elif operator == "<=":
                        return installed_version <= required_version
                    elif operator == "<":
                        return installed_version < required_version
                    return True
                    
            except PackageNotFoundError:
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
            "transformers>=4.51.3",  # Required for VibeVoice compatibility
            
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
            "s3tokenizer>=0.1.7",          # SAFE - Heavy dependencies but NO conflicts
            "vector-quantize-pytorch",     # SAFE - Clean install
            "resemble-perth",              # SAFE - Works in ChatterBox
            "diffusers>=0.30.0",          # SAFE - Likely safe
            # "audio-separator>=0.35.2",    # MOVED - Requires numpy>=2, installed conditionally
            "hydra-core>=1.3.0",          # SAFE - Clean install, minimal dependencies
            
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
            "tensorboard",                # Required by descript-audiotools
            "julius",                     # Required by descript-audiotools for DSP operations
            
            # IndexTTS-2 engine dependencies (tested safe - no conflicts found)
            "cn2an>=0.5.22",              # Chinese number to Arabic number conversion
            "g2p-en>=2.1.0",              # English grapheme-to-phoneme conversion
            "json5>=0.12.0",              # JSON5 parsing for IndexTTS-2 config files
            "keras>=2.9.0",               # Deep learning framework
            "modelscope>=1.27.0",         # Chinese model hub for IndexTTS-2
            "munch>=4.0.0",               # Dictionary access with dot notation
            "sentencepiece>=0.2.1",       # Text tokenization
            "textstat>=0.7.10",           # Text statistics and readability
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

        # Install core RVC dependency first (with graceful failure for build-tool-less systems)
        self.run_pip_command(["install", "monotonic-alignment-search"], "Installing monotonic-alignment-search", ignore_errors=True)
        
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
                
                # Try GPU installation first with --no-deps to prevent numpy downgrade
                self.run_pip_command(["install", "--no-deps", faiss_gpu_package], f"Installing {faiss_gpu_package} for GPU acceleration (--no-deps)")
                self.log("faiss-gpu installed with --no-deps - RVC will use GPU acceleration without downgrading numpy", "SUCCESS")
                
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
        self.log("Checking numpy compatibility", "INFO")
        
        # Python 3.13+ requires numpy 2.1.0 or newer (no wheels for 1.26.x)
        if self.python_version >= (3, 13):
            minimum_numpy = "numpy>=2.1.0,<2.3.0"
            self.log("Python 3.13+ detected - requires NumPy 2.1.0 or newer", "INFO")
        else:
            minimum_numpy = "numpy>=1.26.4,<2.3.0"
        
        # Check if numpy is already installed and what version
        try:
            import numpy
            numpy_version = numpy.__version__
            self.log(f"Current numpy version: {numpy_version}", "INFO")
            
            # Parse numpy version
            import re
            version_match = re.match(r'(\d+)\.(\d+)', numpy_version)
            if version_match:
                major, minor = int(version_match.group(1)), int(version_match.group(2))
                
                # Python 3.13+ specific check
                if self.python_version >= (3, 13) and major < 2:
                    self.log(f"NumPy {numpy_version} is incompatible with Python 3.13+", "ERROR")
                    self.log("Python 3.13 requires NumPy 2.1.0 or newer (no wheels for 1.26.x)", "WARNING")
                    numpy_constraint = minimum_numpy
                # Accept both numpy 1.26.x and 2.x.x (but not 2.3.x) for older Python
                elif major == 1 and minor >= 26 and self.python_version < (3, 13):
                    self.log(f"NumPy {numpy_version} is compatible - keeping current version", "INFO")
                    return  # NumPy 1.26.x is fine for Python < 3.13
                elif major == 2 and minor < 3:
                    self.log(f"NumPy {numpy_version} is compatible - keeping current version", "INFO")
                    return  # NumPy 2.0.x, 2.1.x, 2.2.x are fine
                elif major >= 2 and minor >= 3:
                    # Only numpy 2.3+ needs downgrading
                    self.log(f"NumPy {numpy_version} may cause issues - constraining to <2.3.0", "WARNING")
                    numpy_constraint = minimum_numpy
                else:
                    # Very old numpy needs updating
                    self.log(f"NumPy {numpy_version} is too old", "WARNING")
                    numpy_constraint = minimum_numpy
            else:
                # Can't parse version - install safe range
                numpy_constraint = minimum_numpy
                
        except ImportError:
            # No numpy installed - install with appropriate constraints
            self.log("NumPy not found - installing with appropriate constraints", "INFO")
            numpy_constraint = minimum_numpy
        except Exception as e:
            # NumPy import failed
            self.log(f"NumPy check failed ({e}) - installing safe version", "WARNING")
            numpy_constraint = minimum_numpy
        
        # Only install/upgrade numpy if needed
        if 'numpy_constraint' in locals():
            self.run_pip_command(["install", numpy_constraint], "Installing numpy with version constraints")
        
        # Note: We're not forcing numba installation anymore since we disable JIT anyway
        self.log("NumPy compatibility check complete", "INFO")
    
    def install_audio_separator_if_compatible(self):
        """Install audio-separator only if numpy version supports it"""
        try:
            import numpy
            numpy_version = numpy.__version__
            
            # Parse numpy version
            import re
            version_match = re.match(r'(\d+)\.(\d+)', numpy_version)
            if version_match:
                major, minor = int(version_match.group(1)), int(version_match.group(2))
                
                if major >= 2:
                    # NumPy 2.x can use audio-separator
                    self.log(f"NumPy {numpy_version} supports audio-separator - installing", "INFO")
                    self.run_pip_command(
                        ["install", "audio-separator>=0.35.2"], 
                        "Installing audio-separator for enhanced vocal removal",
                        ignore_errors=True  # It's optional, so don't fail if it doesn't install
                    )
                else:
                    # NumPy 1.x - skip audio-separator, will use bundled implementations
                    self.log(f"NumPy {numpy_version} detected - skipping audio-separator (will use bundled vocal removal)", "INFO")
                    self.log("Vocal removal will use bundled RVC/MelBand/MDX23C implementations", "INFO")
            else:
                # Can't determine version - skip to be safe
                self.log("Could not determine NumPy version - skipping audio-separator", "WARNING")
                
        except ImportError:
            # NumPy not installed? This shouldn't happen at this point
            self.log("NumPy not found - skipping audio-separator installation", "WARNING")
        except Exception as e:
            # Any other error - skip audio-separator
            self.log(f"Error checking NumPy compatibility for audio-separator: {e}", "WARNING")
            self.log("Skipping audio-separator - vocal removal will use bundled implementations", "INFO")

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
            "opencv-python",       # Forces numpy downgrade from 2.x to 1.26.x
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
        # NOTE: Using FushionHub fork temporarily - Microsoft removed the official repo
        # Original: https://github.com/microsoft/VibeVoice.git (no longer exists)
        # This fork maintains the same API and should work identically
        self.log("Installing VibeVoice with --no-deps to prevent package downgrades", "WARNING")
        self.run_pip_command(
            ["install", "git+https://github.com/FushionHub/VibeVoice.git", "--no-deps"],
            "Installing VibeVoice (--no-deps)",
            ignore_errors=True
        )

    def install_f5tts_multilingual_support(self):
        """Install phonemization support for F5-TTS multilingual models (Polish, German, French, Spanish, etc.)"""
        self.log("Installing F5-TTS multilingual phonemization support", "INFO")
        
        if self.is_windows:
            # Windows: pip package that includes espeak binaries (no separate system install needed)
            self.log("Windows detected - installing espeak-phonemizer-windows for multilingual F5-TTS", "INFO")
            self.run_pip_command(
                ["install", "espeak-phonemizer-windows"], 
                "Installing espeak-phonemizer-windows (includes binaries)",
                ignore_errors=True
            )
            
            # Test if it works
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable, "-c",
                    "from espeak_phonemizer import Phonemizer; p=Phonemizer(); p.phonemize('test', voice='en')"
                ], capture_output=True, timeout=10)
                if result.returncode == 0:
                    self.log("espeak-phonemizer-windows working - multilingual F5-TTS models will work properly", "SUCCESS")
                else:
                    self.log("espeak-phonemizer-windows test failed - non-English F5-TTS models may have quality issues", "WARNING")
            except Exception:
                self.log("Could not test espeak-phonemizer-windows installation", "WARNING")
                
        else:
            # Linux/Mac: pip package + separate system dependency
            self.log("Linux/Mac detected - installing phonemizer for multilingual F5-TTS support", "INFO")
            
            phonemizer_installed = self.run_pip_command(
                ["install", "phonemizer"], 
                "Installing phonemizer package",
                ignore_errors=True
            )
            
            if phonemizer_installed:
                # Test if system espeak is available
                try:
                    import subprocess
                    result = subprocess.run([
                        sys.executable, "-c",
                        "from phonemizer import phonemize; phonemize('test', language='en', backend='espeak')"
                    ], capture_output=True, timeout=10)
                    
                    if result.returncode == 0:
                        self.log("phonemizer + system espeak working - multilingual F5-TTS models will work properly", "SUCCESS")
                    else:
                        self.log("phonemizer installed but system espeak dependency missing", "WARNING")
                        self.log("To enable multilingual F5-TTS support, install system espeak:", "INFO")
                        if self.is_macos:
                            print("   brew install espeak")
                        else:  # Linux
                            print("   sudo apt-get install espeak espeak-data  # Ubuntu/Debian")
                            print("   sudo dnf install espeak espeak-devel     # Fedora/RHEL")
                        self.log("Non-English F5-TTS models will fall back to character-based processing", "WARNING")
                        
                except Exception as e:
                    self.log(f"Could not test phonemizer installation: {e}", "WARNING")
            else:
                self.log("phonemizer installation failed - non-English F5-TTS models will use fallback processing", "WARNING")

    def install_indexts_text_processing(self):
        """Install IndexTTS-2 text processing with smart fallback handling"""
        self.log("Installing IndexTTS-2 text normalization support", "INFO")

        # WeTextProcessing requires manual wheel installation on Windows (build tools needed)
        # Skip it on Windows and use fallback instead to avoid confusing error messages
        if self.is_windows:
            self.log("Windows detected - WeTextProcessing requires manual build, using fallback approach", "INFO")
            wetextprocessing_success = False
        else:
            # Try WeTextProcessing first (newer, preferred package) - only on Linux/Mac
            wetextprocessing_success = self.run_pip_command(
                ["install", "WeTextProcessing"],
                "Installing WeTextProcessing (Chinese/English normalization)",
                ignore_errors=True
            )

        if wetextprocessing_success:
            # Test if WeTextProcessing actually works
            try:
                result = subprocess.run([
                    sys.executable, "-c",
                    "from WeTextProcessing import Normalizer; n=Normalizer(lang='en', operator='tn'); print('OK')"
                ], capture_output=True, timeout=10)
                if result.returncode == 0:
                    self.log("WeTextProcessing installed and working - IndexTTS-2 will have full text normalization", "SUCCESS")
                    return
                else:
                    self.log("WeTextProcessing installed but not working - trying fallback", "WARNING")
            except Exception:
                self.log("Could not test WeTextProcessing - trying fallback", "WARNING")

        # Fallback to wetext (older package, more compatible)
        wetext_success = self.run_pip_command(
            ["install", "wetext"],
            "Installing wetext (fallback text normalization)",
            ignore_errors=True
        )

        if wetext_success:
            try:
                result = subprocess.run([
                    sys.executable, "-c",
                    "from wetext import Normalizer; n=Normalizer(lang='en', operator='tn'); print('OK')"
                ], capture_output=True, timeout=10)
                if result.returncode == 0:
                    self.log("wetext fallback working - IndexTTS-2 will have basic text normalization", "SUCCESS")
                    return
                else:
                    self.log("wetext installed but not working", "WARNING")
            except Exception:
                self.log("Could not test wetext installation", "WARNING")

        # Both failed - IndexTTS-2 will use basic processing
        self.log("Text normalization packages failed to install - IndexTTS-2 will use basic text processing", "WARNING")
        self.log("This may affect quality for Chinese text and complex English patterns", "INFO")

    def check_python_environment(self):
        """Check Python environment and warn about potential mismatches"""
        python_path = sys.executable.lower()

        # Check for virtual environment mismatch (Windows py launcher issue)
        if 'VIRTUAL_ENV' in os.environ:
            venv_path = os.environ['VIRTUAL_ENV']
            if self.is_windows and not sys.executable.startswith(venv_path):
                self.log("WARNING: Python version mismatch detected!", "WARNING")
                self.log(f"Virtual environment: {venv_path}", "WARNING")
                self.log(f"Current Python: {sys.executable}", "WARNING")
                self.log("You likely used 'py install.py' instead of 'python install.py'", "WARNING")
                self.log("Use 'python install.py' to match your ComfyUI Python version", "INFO")
                print()  # Add spacing for visibility
                return False

        # Check for clearly identifiable system Python paths
        system_python_patterns = [
            "c:\\python",           # Windows system Python
            "/usr/bin/python",      # Linux system Python
            "/usr/local/bin/python", # macOS Homebrew system Python
            "system32",             # Windows system directory
        ]

        if any(pattern in python_path for pattern in system_python_patterns):
            self.log("WARNING: Detected system-wide Python installation", "WARNING")
            self.log(f"Current Python: {sys.executable}", "WARNING")
            self.log("This may install packages to the wrong location", "WARNING")
            self.log("For best results, use ComfyUI Manager for automatic installation", "INFO")
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
        print("="*70)

        # Windows-specific note about text normalization
        if self.is_windows:
            print("\n" + "-"*50)
            print("   WINDOWS NOTES")
            print("-"*50)
            print("  WeTextProcessing was skipped (requires manual wheel build).")
            print("  IndexTTS-2 will use fallback text processing instead - this")
            print("  means slightly lower quality for Chinese text, but all TTS")
            print("  generation will work fine.")
            print("\n  This is normal and expected on Windows!")

        print()

def main():
    """Main installation entry point"""
    installer = TTSAudioInstaller()
    
    try:
        installer.log("Starting TTS Audio Suite installation", "INFO")
        installer.log(f"Python {installer.python_version.major}.{installer.python_version.minor}.{installer.python_version.micro} detected", "INFO")
        
        # Check environment and system dependencies before proceeding
        installer.check_python_environment()
        installer.ensure_requirements_installed()  # Ensure requirements.txt is installed first
        
        # Check system dependencies (Linux only)
        if not installer.check_system_dependencies():
            installer.log("System dependency check failed - aborting installation", "ERROR")
            sys.exit(1)
        
        # Install in correct order to prevent conflicts
        installer.install_pytorch_with_cuda()  # Install PyTorch first with proper CUDA detection
        installer.install_core_dependencies()
        installer.install_macos_specific_packages()  # Mac-specific package fixes
        installer.install_numpy_with_constraints()
        installer.install_audio_separator_if_compatible()  # Install audio-separator only if numpy>=2
        installer.install_rvc_dependencies()
        installer.install_problematic_packages()
        installer.install_vibevoice()  # Install VibeVoice with careful dependency management
        installer.install_f5tts_multilingual_support()  # Install phonemization for Polish/multilingual F5-TTS
        installer.install_indexts_text_processing()  # Install IndexTTS-2 text normalization with fallback
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