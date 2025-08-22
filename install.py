#!/usr/bin/env python3
"""
TTS Audio Suite - Advanced Dependency Installation Script
Handles complex dependency conflicts and Python 3.13 compatibility issues
"""

import subprocess
import sys
import platform
import importlib.util

def run_pip_command(cmd, description="Installing packages"):
    """Run pip command with error handling"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_python_version():
    """Check Python version and return major.minor"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    return version.major, version.minor

def check_package_installed(package_name):
    """Check if a package is already installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def main():
    print("🚀 TTS Audio Suite - Advanced Installation")
    print("Handling dependency conflicts and Python 3.13 compatibility")
    
    major, minor = check_python_version()
    is_python313 = major == 3 and minor >= 13
    
    # Stage 1: Foundation packages that must be installed first
    foundation_packages = [
        "wheel",
        "setuptools",
        "numpy>=2.0.0,<2.3.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0"
    ]
    
    print("\n📦 Stage 1: Foundation packages")
    run_pip_command([
        sys.executable, "-m", "pip", "install", "--upgrade"
    ] + foundation_packages, "Installing foundation packages")
    
    # Stage 2: Conflict-prone packages with special flags
    conflict_packages = [
        "descript-audio-codec",
        "vector-quantize-pytorch",
    ]
    
    print("\n⚠️  Stage 2: Conflict-prone packages (with --no-deps)")
    for package in conflict_packages:
        run_pip_command([
            sys.executable, "-m", "pip", "install", 
            "--no-deps", "--force-reinstall", package
        ], f"Installing {package} (conflict resolution)")
    
    # Stage 3: MediaPipe Python 3.13 workaround
    if is_python313:
        print("\n🔧 Stage 3: Python 3.13 MediaPipe workaround")
        
        # Try to force install Python 3.12 wheel
        mediapipe_wheels = {
            "Windows": "https://files.pythonhosted.org/packages/c9/c7/40a1e1c274021ce5de2b7c57d0fc58b1b6db9ae6bb0cd5e0b78ab5e23a5b9/mediapipe-0.10.21-cp312-cp312-win_amd64.whl",
            "Darwin": "https://files.pythonhosted.org/packages/79/73/425ca52ea7b70f3b7c35ad6be2199bb0ba8ac6c67c2700cd71b09a067dde8/mediapipe-0.10.21-cp312-cp312-macosx_11_0_x86_64.whl",
            "Linux": "https://files.pythonhosted.org/packages/c6/11/cbb40647e36ac2adbef7f77b4a82b15b0d6ebedebffccfe0421a52c7b5b15/mediapipe-0.10.21-cp312-cp312-manylinux_2_28_x86_64.whl"
        }
        
        system = platform.system()
        if system in mediapipe_wheels:
            wheel_url = mediapipe_wheels[system]
            print(f"Attempting to force install MediaPipe wheel for {system}")
            
            success = run_pip_command([
                sys.executable, "-m", "pip", "install", 
                wheel_url, "--force-reinstall", "--no-deps"
            ], f"Force installing MediaPipe cp312 wheel on Python 3.13")
            
            if success:
                print("🎉 MediaPipe Python 3.13 workaround successful!")
            else:
                print("⚠️  MediaPipe workaround failed - video analysis will be unavailable")
        else:
            print(f"⚠️  No MediaPipe workaround available for {system}")
    else:
        print(f"\n✅ Stage 3: Standard MediaPipe install (Python {major}.{minor})")
        run_pip_command([
            sys.executable, "-m", "pip", "install", "mediapipe>=0.10.0"
        ], "Installing MediaPipe (standard)")
    
    # Stage 4: Remaining packages that depend on above
    remaining_packages = [
        "hydra-core>=1.3.0",
        "pillow>=9.0.0",
        "opencv-python>=4.8.0",
        "protobuf>=4.21.0",
    ]
    
    print("\n📦 Stage 4: Remaining packages")
    run_pip_command([
        sys.executable, "-m", "pip", "install"
    ] + remaining_packages, "Installing remaining packages")
    
    # Stage 5: Verification
    print("\n🔍 Stage 5: Verification")
    critical_packages = ["mediapipe", "hydra", "dac", "vector_quantize_pytorch"]
    
    for package in critical_packages:
        if check_package_installed(package):
            print(f"✅ {package} - Available")
        else:
            print(f"❌ {package} - Missing")
    
    print("\n🎉 TTS Audio Suite installation complete!")
    print("Check the verification results above for any missing features.")

if __name__ == "__main__":
    main()