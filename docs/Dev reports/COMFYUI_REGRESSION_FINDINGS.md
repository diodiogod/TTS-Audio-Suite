# ComfyUI VRAM Regression Analysis - Python 3.12

## Summary
ComfyUI versions after v0.3.48 introduce severe VRAM spikes specifically in Python 3.12 environments when using TTS Audio Suite. The issue affects model memory management and causes significant performance degradation.

## Affected Versions
- **Last Working**: ComfyUI v0.3.56 + ComfyUI-frontend v1.25.11
- **First Broken**: ComfyUI v0.3.57 + ComfyUI-frontend v1.25.11
- **Environment**: Python 3.12 only (Python 3.13 unaffected)
- **Root Cause**: Commit e2d1e5da "Enable Convolution AutoTuning" enables `torch.backends.cudnn.benchmark = True`

## Symptoms
- Massive VRAM spikes (multi-GB) occurring BEFORE and AFTER TTS inference, not during actual generation
- Spikes happen during model loading and audio processing/cleanup phases
- Affects all TTS engines: ChatterBox, ChatterBox 23-lang, IndexTTS-2
- Performance severely degraded on 24GB VRAM systems
- Issue does NOT occur during the TTS inference itself

## Testing Results

### ComfyUI v0.3.48 (Working)
```
Environment: Python 3.12
TTS Audio Suite: v4.10.8 (latest)
Result: Flat VRAM usage, no spikes
All engines tested: ✅ ChatterBox, ✅ ChatterBox 23-lang, ✅ IndexTTS-2
```

### ComfyUI v0.3.59 (Broken)
```
Environment: Python 3.12
TTS Audio Suite: v4.10.8 (latest)
Result: Severe VRAM spikes after inference
All engines affected: ❌ ChatterBox, ❌ ChatterBox 23-lang, ❌ IndexTTS-2
```

## Root Cause Analysis
**Specific Commit**: e2d1e5da "Enable Convolution AutoTuning" by contentis <lspindler@nvidia.com>

The commit enables `torch.backends.cudnn.benchmark = True` when `--fast autotune` is used. This CUDNN benchmarking:
1. Tests multiple convolution algorithms to find the fastest
2. **Allocates significant extra VRAM during algorithm benchmarking**
3. Appears to interact poorly with Python 3.12's garbage collection
4. Causes VRAM spikes during model operations, especially with TTS model wrappers

## Reproduction Steps
1. Install Python 3.12 environment
2. Install ComfyUI v0.3.59
3. Install TTS Audio Suite v4.10.8
4. Load any TTS engine workflow
5. Generate audio
6. Observe VRAM spikes before and after generation (not during the actual TTS inference)

## Workaround
Downgrade to ComfyUI v0.3.48 + ComfyUI-frontend v1.2.65

## Next Steps
1. Systematically test ComfyUI versions between v0.3.48 and v0.3.59
2. Identify exact commit that introduced the regression
3. Create detailed GitHub issue for ComfyUI project
4. Provide minimal reproduction case

## Technical Details
- Issue is environment-specific (Python 3.12 only)
- Affects ComfyUI's model wrapper system
- Related to PyTorch CUDA memory management integration
- No code changes needed in TTS Audio Suite - issue is upstream in ComfyUI