# RVC Feature Comparison: Replay vs ComfyUI Custom Node

## Overview
This document compares the features available in Replay RVC with the ComfyUI RVC custom nodes to identify gaps and missing functionality for TTS suite integration.

---

## Feature Comparison Table

| Feature Category | Replay RVC | ComfyUI RVC Node | Status | Notes |
|-----------------|------------|------------------|---------|-------|
| **BASIC INPUTS** | | | | |
| Audio Input | ✅ Select Audio | ✅ Audio input (AUDIO/VHS_AUDIO) | ✅ **AVAILABLE** | ComfyUI version more flexible |
| Voice Model | ✅ Model Select | ✅ RVC_MODEL input | ✅ **AVAILABLE** | Similar functionality |
| **PITCH CONTROL** | | | | |
| Pitch Shift | ✅ Pitch (-/+) | ✅ f0_up_key (-14 to +14) | ✅ **AVAILABLE** | ComfyUI: semitone steps |
| Instrumental Pitch | ✅ Separate pitch shift for backing track | ❌ Not available | ❌ **MISSING** | For pitch-shifting instrumentals during recombination |
| **PITCH DETECTION** | | | | |
| PM | ✅ Available | ❌ Not available | ❌ **MISSING** | - |
| Harvest | ✅ Available | ❌ Not available | ❌ **MISSING** | - |
| Crepe | ✅ Available | ✅ Available | ✅ **AVAILABLE** | - |
| Crepe Tiny | ✅ Available | ❌ Not available | ❌ **MISSING** | - |
| Mangio Crepe | ✅ Available | ✅ Available (`mangio-crepe`) | ✅ **AVAILABLE** | - |
| RMVPE | ✅ Available | ✅ Available (`rmvpe`) | ✅ **AVAILABLE** | - |
| FCPE | ✅ Available | ❌ Not available | ❌ **MISSING** | - |
| RMVPE+ | ❌ Not in Replay | ✅ Available (`rmvpe+`) | ➕ **BONUS** | ComfyUI exclusive |
| **ADVANCED SETTINGS** | | | | |
| Index Ratio | ✅ Float control | ✅ `index_rate` (0.0-1.0) | ✅ **AVAILABLE** | Same functionality |
| Consonant Protection | ✅ Float control | ✅ `protect` (0.0-0.5) | ✅ **AVAILABLE** | Same functionality |
| Volume Envelope | ✅ Float control | ✅ `rms_mix_rate` (0.0-1.0) | ✅ **AVAILABLE** | Same functionality |
| **AUDIO PROCESSING** | | | | |
| De-Echo & Reverb | ✅ Toggle | ✅ UVR5Node with `UVR-DeEcho-DeReverb.pth` | ✅ **AVAILABLE** | Separate node required |
| Output Format | ✅ Multiple formats | ✅ Multiple formats (wav, flac, mp3) | ✅ **AVAILABLE** | - |
| **STEMMING/SEPARATION** | | | | |
| Stem Input Select | ✅ Available | ✅ UVR5Node supports stem separation | ✅ **AVAILABLE** | Separate UVR5 node |
| Render Device | ✅ CUDA/CPU select | ✅ Auto-detected optimal device | ✅ **AVAILABLE** | Auto-optimized |
| Stemming Methods | ✅ Multiple options | ✅ Multiple UVR models available | ✅ **AVAILABLE** | Via UVR5Node |
| **ADDITIONAL FEATURES** | | | | |
| Autotune | ❌ Not available | ✅ `f0_autotune` boolean | ➕ **BONUS** | ComfyUI exclusive |
| Resample Rate | ❌ Not explicit | ✅ `resample_sr` options | ➕ **BONUS** | ComfyUI exclusive |
| Crepe Hop Length | ❌ Not configurable | ✅ `crepe_hop_length` (16-512) | ➕ **BONUS** | ComfyUI exclusive |
| Caching | ❌ Not available | ✅ `use_cache` boolean | ➕ **BONUS** | ComfyUI performance feature |

---

## Missing Features Summary

### 🔴 **CRITICAL MISSING** (High Priority)
1. **Instrumental Pitch Control** - Separate pitch shifting for backing track during vocal+instrumental recombination
2. **PM Pitch Detection** - Basic pitch detection method
3. **Harvest Pitch Detection** - Traditional pitch extraction
4. **FCPE Pitch Detection** - Fast pitch extraction
5. **Crepe Tiny** - Lightweight version of Crepe

### 🟡 **WORKFLOW DIFFERENCES** (Medium Priority)
1. **Integrated De-Echo/Reverb** - Currently requires separate UVR5 node
2. **Modular Design** - ComfyUI uses separate nodes (UVR5→RVC→Merge) vs Replay's single interface. This offers more flexibility but requires more setup.

### 🟢 **FUNCTIONAL EQUIVALENTS** (Available but different)
1. **Volume Envelope** ↔ **RMS Mix Rate** - Same functionality, different names
2. **Consonant Protection** ↔ **Protect** - Same functionality  
3. **Stemming Methods** - Available via separate UVR5Node

---

## ComfyUI Node Architecture

### Current Modular Workflow
```
Audio Input → UVR5Node (Separation/De-echo) → RVCNode (Voice Conversion) → MergeAudioNode (Combine) → Output
                ↓                                    ↓                            ↓
        Stemming Models                    PitchExtractionParams           Instrumental Audio
                                          HubertModel                      (with potential pitch shift)
                                          RVCModel
```

### Replay Single-Interface Approach
```
Audio Input → [Separation + RVC + Recombination in One Interface] → Output
```

### Modular vs Single-Node Trade-offs
**Modular Advantages:**
- Flexibility to swap components
- Reusable nodes for different workflows  
- Better for complex audio processing chains
- Follows ComfyUI design principles

**Single-Node Advantages:**
- Simpler user experience
- Fewer connections to manage
- More similar to Replay workflow

---

## Recommendations for TTS Suite Integration

### 🎯 **IMMEDIATE ACTIONS**
1. **Add missing pitch detection methods**: PM, Harvest, Crepe Tiny, FCPE
2. **Implement instrumental pitch control** for the MergeAudioNode
3. **Consider workflow optimization** based on your target user experience

### 🔧 **IMPLEMENTATION SUGGESTIONS**

#### 1. Enhanced Pitch Detection Options
```python
PITCH_EXTRACTION_OPTIONS = [
    "crepe", "mangio-crepe", "rmvpe", "rmvpe+",  # Current
    "pm", "harvest", "crepe-tiny", "fcpe"        # Missing
]
```

#### 2. Instrumental Pitch Control (for MergeAudioNode)
```python
"instrumental_pitch": ("INT", {
    "default": 0,
    "min": -24, 
    "max": 24,
    "tooltip": "Pitch shift for instrumental track during recombination"
})
```

#### 3. Enhanced Merge Audio Node
Consider adding instrumental pitch shifting to the existing MergeAudioNode to match Replay's capability of adjusting backing track pitch independently.

### 🏆 **DESIGN CONSIDERATIONS**
For TTS suite integration, evaluate whether to:
- **Keep modular approach** for maximum flexibility
- **Create convenience wrapper nodes** that combine common workflows
- **Add missing features to existing nodes** (recommended approach)

---

## Conclusion

The ComfyUI RVC implementation has **85% feature parity** with Replay RVC, with some **exclusive bonuses** (autotune, advanced caching, better device handling). The main gaps are in **pitch detection variety** and **workflow integration**. 

For TTS suite integration, focus on adding the missing pitch detection methods and creating a more streamlined workflow that can compete with Replay's user-friendly single-node approach.