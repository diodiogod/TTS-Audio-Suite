"""
F5-TTS Edit Options Node
Provides advanced configuration options for F5-TTS Speech Editor
Following Audio Analyzer pattern with separate options node
"""

class F5TTSEditOptionsNode:
    """
    🔧 F5-TTS Edit Options
    Advanced configuration options for F5-TTS Speech Editor
    
    Crossfade options are stable. Post-processing options are experimental.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "crossfade_duration_ms": ("INT", {
                    "default": 100, "min": 0, "max": 500, "step": 10,
                    "tooltip": "Crossfade duration in milliseconds for smooth transitions between segments"
                }),
                "crossfade_curve": (["linear", "cosine", "exponential"], {
                    "default": "cosine",
                    "tooltip": "Crossfade curve type: linear (constant), cosine (smooth), exponential (sharp)"
                }),
                "adaptive_crossfade": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically adjust crossfade duration based on segment size"
                }),
                "enable_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache F5-TTS generation to speed up subsequent runs with identical parameters"
                }),
                "cache_size_limit": ("INT", {
                    "default": 100, "min": 10, "max": 1000,
                    "tooltip": "Maximum number of cached audio segments to store in memory"
                }),
                "boundary_volume_matching": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "EXPERIMENTAL: Automatically match volume levels at segment boundaries to reduce clicks/pops"
                }),
                "full_segment_normalization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "EXPERIMENTAL: Normalize entire generated segments to match surrounding original audio RMS levels"
                }),
                "spectral_matching": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: Apply EQ to match spectral characteristics of original audio"
                }),
                "noise_floor_matching": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "EXPERIMENTAL: Add subtle noise to match the background noise level of original audio"
                }),
                "dynamic_range_compression": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "EXPERIMENTAL: Apply gentle compression to reduce volume spikes and make transitions smoother"
                }),
                "post_rms_normalization": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Post-processing RMS normalization level. Applied after F5-TTS generation to normalize generated segments volume (does not affect original segments)."
                }),
                "force_cache_clear": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force clear cache for debugging (will regenerate F5-TTS audio)"
                })
            }
        }
    
    RETURN_TYPES = ("F5TTS_EDIT_OPTIONS",)
    RETURN_NAMES = ("edit_options",)
    FUNCTION = "create_options"
    CATEGORY = "TTS Audio Suite/👄 F5-TTS"
    
    def create_options(self, crossfade_duration_ms=100, crossfade_curve="cosine", 
                      adaptive_crossfade=False, enable_cache=True, cache_size_limit=100,
                      boundary_volume_matching=True, full_segment_normalization=True,
                      spectral_matching=False, noise_floor_matching=False, 
                      dynamic_range_compression=True, post_rms_normalization=0.1, 
                      force_cache_clear=False):
        """Create F5-TTS edit options configuration"""
        
        options = {
            "crossfade_duration_ms": crossfade_duration_ms,
            "crossfade_curve": crossfade_curve,
            "adaptive_crossfade": adaptive_crossfade,
            "enable_cache": enable_cache,
            "cache_size_limit": cache_size_limit,
            "boundary_volume_matching": boundary_volume_matching,
            "full_segment_normalization": full_segment_normalization,
            "spectral_matching": spectral_matching,
            "noise_floor_matching": noise_floor_matching,
            "dynamic_range_compression": dynamic_range_compression,
            "post_rms_normalization": post_rms_normalization,
            "force_cache_clear": force_cache_clear
        }
        
        return (options,)

# Node export
NODE_CLASS_MAPPINGS = {
    "ChatterBoxF5TTSEditOptions": F5TTSEditOptionsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterBoxF5TTSEditOptions": "🔧 F5-TTS Edit Options"
}

# Export for ComfyUI registration
ChatterBoxF5TTSEditOptions = F5TTSEditOptionsNode

__all__ = ["ChatterBoxF5TTSEditOptions"]