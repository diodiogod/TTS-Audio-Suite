"""
F5-TTS Edit Options Node
Provides advanced configuration options for F5-TTS Speech Editor
Following Audio Analyzer pattern with separate options node
"""

class F5TTSEditOptionsNode:
    """
    🔧 F5-TTS Edit Options
    Advanced configuration options for F5-TTS Speech Editor
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "crossfade_duration_ms": ("INT", {
                    "default": 50, "min": 0, "max": 500, "step": 10,
                    "tooltip": "Crossfade duration in milliseconds for smooth transitions between segments"
                }),
                "crossfade_curve": (["linear", "cosine", "exponential"], {
                    "default": "linear",
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
                })
            }
        }
    
    RETURN_TYPES = ("F5TTS_EDIT_OPTIONS",)
    RETURN_NAMES = ("edit_options",)
    FUNCTION = "create_options"
    CATEGORY = "F5-TTS Voice"
    
    def create_options(self, crossfade_duration_ms=50, crossfade_curve="linear", 
                      adaptive_crossfade=False, enable_cache=True, cache_size_limit=100):
        """Create F5-TTS edit options configuration"""
        
        options = {
            "crossfade_duration_ms": crossfade_duration_ms,
            "crossfade_curve": crossfade_curve,
            "adaptive_crossfade": adaptive_crossfade,
            "enable_cache": enable_cache,
            "cache_size_limit": cache_size_limit
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