# Qwen3-TTS Optimization Guide

This document explains how to enable torch.compile optimizations for Qwen3-TTS to achieve ~1.5-2x speedup.

## Requirements

**CRITICAL**: torch.compile optimizations require specific versions tested with the reference implementation:

- **PyTorch 2.10.0+** with CUDA 13.0 support
- **triton-windows 3.6.0+** (Windows) or **triton 3.6.0+** (Linux)
- **Visual Studio C++ Build Tools** on Windows (a real `cl.exe` toolchain, not just stale PATH entries)
- Python 3.12 (recommended)

**Note**: PyTorch 2.8.x and 2.9.x have compatibility issues with triton on Windows. Only PyTorch 2.10+ is supported.

### Installation

#### Windows (Tested Configuration)
```bash
# Upgrade PyTorch to 2.10 with CUDA 13.0
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install triton-windows 3.6.x
pip install -U "triton-windows<3.7"
```

Also install **Visual Studio Build Tools** with the C++ toolchain:

- Workload: `Desktop development with C++`
- Or at minimum the MSVC x64/x86 build tools component

If `cl.exe` is missing, `torch.compile` in the isolated Qwen runtime will fall back to the non-compiled path.

**Verification:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); import triton; print(f'Triton: {triton.__version__}')"
```

Expected output:
```
PyTorch: 2.10.0+cu130
Triton: 3.6.0.post25
```

#### Linux
```bash
# Upgrade PyTorch to 2.10 with CUDA 13.0
pip install --upgrade torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130

# triton is usually bundled with PyTorch on Linux
# Verify version is 3.6.0+
python -c "import triton; print(triton.__version__)"
```

## Environment Configuration

**Main environment**: no special configuration beyond a working compiler toolchain.

**Shared / Dedicated Runtime isolation**:
- TTS Audio Suite now tries to detect Visual Studio automatically and inject a proper MSVC build environment into the isolated worker
- This is meant to preserve `torch.compile` parity with old direct Transformers 4 environments
- It still depends on the Build Tools actually being installed

**Note**: The `reduce-overhead` mode (which includes automatic CUDA graphs) may still have issues with cudaMallocAsync. Use `compile_mode="default"` for best compatibility.

## Usage in Qwen3-TTS Engine Node

Once the environment is configured:

1. Enable `use_torch_compile`: **True**
2. Keep `use_cuda_graphs`: **False** (not needed - reduce-overhead includes it)
3. Set `compile_mode`: **reduce-overhead** (recommended)

### Settings Explanation

- **use_torch_compile**: Enables torch.compile optimization (~1.4-3x speedup)
  - First generation will be slower (compilation time)
  - Subsequent generations are much faster

- **use_cuda_graphs**: Manual CUDA graph capture
  - **Keep FALSE** when using reduce-overhead mode (it already includes automatic CUDA graphs)
  - Only enable if using `compile_mode=default` and you want manual graph capture

- **compile_mode**:
  - `reduce-overhead` (RECOMMENDED): Includes automatic CUDA graphs, fast compilation
  - `max-autotune`: Longer compilation, marginal gains
  - `default`: No automatic CUDA graphs

## Troubleshooting

### Error: "'KernelNamespace' object has no attribute 'mm'"
**Cause**: Incompatible PyTorch and triton-windows versions.

**Solution**: Upgrade to PyTorch 2.10.0 + triton-windows 3.6.0 (see Installation section).

### Error: "operator torchvision::nms does not exist"
**Cause**: torchvision version doesn't match PyTorch version.

**Solution**:
```bash
pip install --upgrade torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Error: "cudaMallocAsync does not support checkPoolLiveAllocations"
**Cause**: Using `compile_mode="reduce-overhead"` which enables automatic CUDA graphs.

**Solution**: Use `compile_mode="default"` instead (recommended for Windows).

### Error: "Failed to find C compiler. Please specify via CC environment variable."
**Cause**: Triton/Inductor could not find a usable Windows C++ toolchain.

**Solution**:
1. Install or repair Visual Studio Build Tools with the C++ workload
2. Restart ComfyUI so the new environment is visible
3. If using Shared/Dedicated Runtime, let TTS Audio Suite auto-detect the toolchain

If detection still fails, your machine state is broken, not just the Python package stack.

### Warning: "TensorFloat32 tensor cores... not enabled"
**Info**: This is just a performance hint, not an error. TF32 is automatically enabled by the extension for better performance.

## Performance Expectations

**Tested on RTX 4090 with PyTorch 2.10.0 + triton-windows 3.6.0:**

| Configuration | Speed | Speedup | Notes |
|--------------|-------|---------|-------|
| **Baseline** (no optimizations) | 5.0 it/s | 1.0x | Standard inference |
| **torch.compile mode=default** | 8.5 it/s | 1.7x | ✅ RECOMMENDED for Windows |
| **+ manual CUDA graphs** | 8.6 it/s | 1.72x | Negligible gain (~0.1 it/s) |
| **mode=reduce-overhead** | FAILS | - | ❌ cudaMallocAsync error on Windows |
| **mode=max-autotune** | FAILS | - | ❌ cudaMallocAsync error on Windows |

### Key Findings

- **torch.compile alone provides ~1.7x speedup** with mode="default"
- **Manual CUDA graphs add almost nothing** on top of torch.compile
- **reduce-overhead and max-autotune fail on Windows** (work on Linux)
- **First generation is slower** (~30-60s compilation overhead)
- **Subsequent generations are fast** (compiled kernels cached)
- **Model unloading works correctly** even with CUDA graphs enabled
- **Shared Runtime should preserve these gains** if the Windows compiler toolchain is installed and detected correctly

## Notes

- The reference implementation (dffdeeq/Qwen3-TTS-streaming) uses the same approach
- torch.compile caches compiled kernels, so subsequent ComfyUI sessions start faster
- Model unloading works normally - no special handling needed (unlike manual CUDA graphs)
