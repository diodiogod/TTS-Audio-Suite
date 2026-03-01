# Dependency Management Guide

**TTS Audio Suite - Complete dependency compatibility documentation**

This document tracks all package dependencies, conflicts, solutions, and constraints implemented in the install script for universal compatibility across Python versions and platforms.

## Quick Reference

| Package | Version Constraint | Reason | Solution |
|---------|-------------------|---------|----------|
| `torch` | `>=2.6.0` for security | CVE-2025-32434 (Higgs Audio) | Smart CUDA detection + upgrade |
| `numpy` | `>=2.2.0,<2.3.0` | Numba compatibility | Version constraints |
| `numba` | `>=0.61.2` | NumPy 2.2+ support | Auto-upgrade from 0.61.0 |
| `librosa` | --no-deps | Forces NumPy downgrade | Isolated installation |
| `descript-audiotools` | --no-deps | Protobuf 3.19 conflict | Isolated installation |
| `faiss-cpu` | `>=1.7.4` | Cross-platform compatibility | Linux: try GPU, Windows: CPU |

---

## Critical Dependencies & Conflicts

### **Python 3.13 Audio-Separator + Resampy Issue**

**Problem:** Audio-Separator v0.36.1 claims Python 3.13 support, but dependency `resampy 0.4.3` fails
- Audio-Separator loads successfully ✅
- resampy compilation fails with numba in Python 3.13 ❌
- `NUMBA_DISABLE_JIT=1` doesn't prevent resampy from attempting compilation

**Impact:** Certain UVR models fail with Audio-Separator, force RVC fallback
```
Failed in nopython mode pipeline (step: nopython frontend)
Untyped global name '_resample_loop_s': Cannot determine Numba type of <class 'function'>
```

**Root Cause:** resampy 0.4.3 has incomplete Python 3.13 + numba support despite Audio-Separator's claims

**Current Solution:** 
- Audio-Separator fails → RVC fallback works for compatible models (UVR v2/v3 architecture)  
- Some models (e.g., UVR-DeNoise.pth) incompatible with both engines
- Clean error messages implemented to avoid confusion

**Model Compatibility:**
- ✅ **Working Models:** UVR-DeEcho-DeReverb.pth, 6_HP-Karaoke-UVR.pth (via RVC fallback)
- ❌ **Broken Models:** UVR-DeNoise.pth, models requiring Audio-Separator-specific architectures

**Future Fix:** Wait for resampy or Audio-Separator to fix Python 3.13 numba compilation

---

### 1. **NumPy + Numba Compatibility Crisis**

**Problem:** Different numba versions support different NumPy ranges
- `numba 0.61.0`: Supports NumPy ≤ 2.1.x ❌
- `numba 0.61.2+`: Supports NumPy 2.2.x ✅

**Impact:** Core engine loading failures
```
❌ ChatterboxTTS not available: Numba needs NumPy 2.1 or less. Got NumPy 2.2.
❌ F5-TTS not available: Numba needs NumPy 2.1 or less. Got NumPy 2.2.
```

**Solution:** Smart version detection and upgrade
```python
# Detect numba version and upgrade if needed
if numba_version == "0.61.0" or numba_version == "0.61.1":
    pip install --upgrade "numba>=0.61.2"
```

**Constraint:** `numpy>=2.2.0,<2.3.0` + `numba>=0.61.2`

---

### 2. **PyTorch Security Vulnerability (CVE-2025-32434)**

**Problem:** Higgs Audio engine requires PyTorch 2.6+ for security
- PyTorch < 2.6: transformers blocks torch.load() due to CVE
- PyTorch 2.6+: Security issue resolved

**Impact:** Higgs Audio fails to load models
```
ValueError: Due to a serious vulnerability issue in torch.load...
```

**Solution:** Smart CUDA detection and upgrade
- Detect system CUDA version (12.8 → cu124)
- Install PyTorch 2.6+ with correct CUDA support
- Force reinstall if switching CPU↔CUDA variants

**Constraint:** `torch>=2.6.0` with appropriate CUDA index

---

### 3. **Librosa Dependency Hell**

**Problem:** librosa forces NumPy downgrades
- librosa dependencies conflict with our NumPy constraints
- Causes cascade of package downgrades

**Impact:** Breaks carefully balanced dependency tree

**Solution:** `--no-deps` installation
```python
pip install librosa --no-deps
# Then manually install librosa's actual dependencies
```

**Dependencies manually installed:**
- `lazy_loader>=0.1`, `msgpack>=1.0`, `pooch>=1.1`
- `soxr>=0.3.2`, `decorator>=4.3.0`, `joblib>=1.0`
- `scikit-learn>=1.1.0`, `audioread>=2.1.9`

---

### 4. **descript-audiotools Protobuf Conflict**

**Problem:** Forces protobuf downgrade 6.x → 3.19.x
- Breaks other packages expecting modern protobuf
- Cascading compatibility issues

**Solution:** `--no-deps` installation with manual dependencies
```python
pip install descript-audiotools --no-deps
```

**Dependencies manually installed:**
- `flatten-dict`, `ffmpy`, `importlib-resources`
- `randomname`, `markdown2`, `pyloudnorm`
- `pystoi`, `torch-stoi`, `ipython`, `tensorboard`

---

### 5. **FAISS Platform Compatibility**

**Problem:** faiss-gpu availability varies by platform
- Linux: `faiss-gpu-cu12`, `faiss-gpu-cu11` available ✅
- Windows: No GPU version available via pip ❌

**Solution:** Platform-specific installation
```python
if not is_windows and cuda_detected:
    try:
        pip install faiss-gpu-cu12  # Linux + CUDA
    except:
        pip install faiss-cpu      # Fallback
else:
    pip install faiss-cpu          # Windows/CPU-only
```

---

## Python Version Compatibility

### Python 3.13 Specific Issues

| Package | Issue | Solution | Status |
|---------|-------|----------|--------|
| `MediaPipe` | Binary incompatible | OpenSeeFace fallback | ✅ Working |
| `numba` | JIT compilation issues | Disable JIT in Python 3.13 | ✅ Working |
| `audioop` packages | Not available | Remove from dependencies | ✅ Fixed |
| `resampy` | Numba compilation fails in Audio-Separator | RVC fallback works for compatible models | ⚠️ Partial |

### Python 3.12 Compatibility

| Package | Issue | Solution | Status |
|---------|-------|----------|--------|
| All engines | NumPy/numba mismatch | Smart version detection | ✅ Working |
| PyTorch CUDA | CPU vs CUDA variants | Force reinstall detection | ✅ Working |

---

## Install Script Logic Flow

### 1. **Environment Detection**
```python
- Python version (3.12, 3.13+)
- Platform (Windows, Linux, macOS)  
- CUDA availability (nvidia-smi detection)
- CUDA version (12.8 → cu124, 11.8 → cu118)
```

### 2. **PyTorch Installation**
```python
- Check existing version and CUDA support
- Upgrade if needed (security + compatibility)
- Platform-specific CUDA index selection
```

### 3. **Core Dependencies**
```python
- Smart package checking (skip if satisfied)
- Safe packages installed normally
- Sub-dependencies for --no-deps packages
```

### 4. **NumPy + Numba Resolution**  
```python
- Detect numba version compatibility
- Upgrade numba if needed (0.61.0 → 0.61.2+)
- Install constrained NumPy (>=2.2.0,<2.3.0)
```

### 5. **Problematic Package Isolation**
```python
- Install with --no-deps to prevent conflicts
- Manually install required sub-dependencies
- Maintain compatibility with main dependency tree
```

---

## Package Categories

### ✅ **SAFE Packages** (Normal installation)
- `transformers>=4.46.3`, `accelerate`, `datasets`
- `soundfile>=0.12.0`, `sounddevice>=0.4.0`
- `jieba`, `pypinyin`, `unidecode`
- `conformer>=0.3.2`, `x-transformers`, `vocos`
- `s3tokenizer>=0.1.7`, `resemble-perth`

### ⚠️ **PROBLEMATIC Packages** (--no-deps required)
- `librosa` - Forces NumPy downgrade
- `descript-audiotools` - Protobuf conflicts  
- `descript-audio-codec` - Dependency conflicts
- `cached-path` - Forces package downgrades
- `torchcrepe` - Conflicts via librosa
- `onnxruntime` - Forces NumPy 2.3.x (Python 3.13 only)

### 🎯 **VERSION-CRITICAL Packages**
- `torch>=2.6.0` - Security requirement
- `numpy>=2.2.0,<2.3.0` - Numba compatibility
- `numba>=0.61.2` - NumPy 2.2+ support
- `transformers>=4.46.3` - Higgs Audio compatibility

---

## Testing Matrix

### Verified Working Combinations

| Python | Platform | PyTorch | NumPy | Numba | Status |
|--------|----------|---------|-------|--------|--------|
| 3.12.6 | Windows | 2.7.1+cu126 | 2.2.6 | 0.61.2 | ✅ All engines |
| 3.13.x | Windows | 2.8.0+cpu | 2.2.6 | 0.61.2 | ✅ All engines |
| 3.12.6 | Linux | 2.6.0+cu121 | 2.2.6 | 0.61.2 | ✅ All engines |

### Known Broken Combinations

| Python | Platform | PyTorch | NumPy | Numba | Issue |
|--------|----------|---------|-------|--------|-------|
| 3.12.6 | Windows | Any | 2.2.6 | 0.61.0 | ❌ Engine loading |
| Any | Any | <2.6.0 | Any | Any | ❌ Higgs Audio CVE |

---

## Troubleshooting Guide

### Engine Loading Failures
```
❌ ChatterboxTTS not available: Numba needs NumPy 2.1 or less
```
**Cause:** numba 0.61.0 with NumPy 2.2+  
**Fix:** Run install script to upgrade numba to 0.61.2+

### Higgs Audio Model Loading
```
❌ CVE-2025-32434 vulnerability error
```
**Cause:** PyTorch < 2.6.0  
**Fix:** Run install script to upgrade PyTorch 2.6+

### Windows CUDA Issues  
```
AssertionError: Torch not compiled with CUDA enabled
```
**Cause:** CPU PyTorch on CUDA system  
**Fix:** Install script detects and reinstalls CUDA PyTorch

---

## Maintenance Notes

### When Adding New Dependencies
1. **Check conflicts** with existing constraints
2. **Test with problematic packages** (librosa, descript-*)  
3. **Update this documentation**
4. **Add to appropriate category** (Safe/Problematic/Critical)

### When Supporting New Python Versions
1. **Test numba compatibility** with NumPy constraints
2. **Check for binary package availability**
3. **Update compatibility matrix**
4. **Add version-specific workarounds if needed**

### When Updating Major Dependencies
1. **Review cascade effects** on sub-dependencies
2. **Test with --no-deps packages**
3. **Update version constraints**
4. **Verify cross-platform compatibility**

---

*Last updated: Version 4.8.19 - Python 3.13 Audio-Separator + resampy compatibility issues documented*