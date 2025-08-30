"""
Audio Analyzer Options Node - Configuration provider for Audio Analyzer
Provides advanced settings for different analysis methods through a separate optional node
"""

from typing import Dict, Any


class AudioAnalyzerOptionsNode:
    """
    Audio Analyzer Options Node for configuring analysis parameters.
    Outputs a configuration object that can be connected to the main Audio Analyzer node.
    """
    
    @classmethod
    def NAME(cls):
        return "🔧 Audio Analyzer Options"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                # Silence detection options
                "silence_threshold": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "How quiet audio must be to count as silence (0.001-0.1):\n• 0.001-0.005: Very sensitive, catches whispers as speech\n• 0.01: Default, good for most recordings\n• 0.05-0.1: Less sensitive, ignores background noise\nOnly used when analysis_method is 'silence'"
                }),
                "silence_min_duration": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Shortest pause to count as a break between words (0.01-2.0 seconds):\n• 0.01-0.05: Catches tiny pauses between syllables\n• 0.1: Default, good for word breaks\n• 0.5-2.0: Only long pauses between sentences\nOnly used when analysis_method is 'silence'"
                }),
                "invert_silence_regions": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert silence detection to get speech regions instead:\n• False: Returns silence regions (pauses between speech)\n• True: Returns speech regions (inverted silence detection)\nUseful for F5-TTS editing where you want speech segments\nOnly used when analysis_method is 'silence'"
                }),
                
                # Energy detection options
                "energy_sensitivity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "How sensitive to detect volume changes (0.0-1.0):\n• 0.0-0.3: Very sensitive, detects small volume changes\n• 0.5: Default, balanced detection\n• 0.7-1.0: Less sensitive, only major volume changes\nOnly used when analysis_method is 'energy'"
                }),
                
                # Peak detection options
                "peak_threshold": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.001,
                    "max": 0.5,
                    "step": 0.001,
                    "tooltip": "Minimum amplitude threshold for detecting peaks (0.001-0.5):\n• 0.001-0.01: Very sensitive, catches soft consonants and emphasis\n• 0.02: Default for speech, good for normal speaking volume\n• 0.05-0.1: Less sensitive, only strong emphasis or loud sounds\n• 0.2-0.5: Only very loud peaks\nOnly used when analysis_method is 'peaks'"
                }),
                "peak_min_distance": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum time between detected peaks in seconds (0.01-1.0):\n• 0.01-0.03: Very sensitive, catches rapid syllables\n• 0.05: Default for speech, good for normal speech pace\n• 0.1-0.2: Less sensitive, only distinct words/emphasis\n• 0.5-1.0: Only major speech events\nOnly used when analysis_method is 'peaks'"
                }),
                "peak_region_size": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.02,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Size of timing region around each peak in seconds (0.02-1.0):\n• 0.02-0.05: Tight regions for precise timing\n• 0.1: Default, good balance for speech editing\n• 0.2-0.5: Wider regions for context around peaks\n• 0.5-1.0: Very wide regions for phrase-level editing\nOnly used when analysis_method is 'peaks'"
                }),
                
                # Region grouping options
                "group_regions_threshold": ("FLOAT", {
                    "default": 0.000,
                    "min": 0.000,
                    "max": 3.000,
                    "step": 0.001,
                    "tooltip": "Group nearby detected regions into larger segments (0.000-3.000 seconds):\n• 0.000: No grouping, keep all regions separate\n• 0.001-0.050: Group very tiny gaps (remove micro-silences)\n• 0.100-0.200: Group small gaps (syllables → words)\n• 0.300-0.500: Group moderate gaps (words → phrases)\n• 1.000-3.000: Group large gaps (phrases → sentences)\nApplies to all detection methods after initial analysis"
                }),
            }
        }
    
    RETURN_TYPES = ("ADV_AUDIO_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "create_options"
    CATEGORY = "TTS Audio Suite/🎵 Audio Processing"
    
    def create_options(self, silence_threshold=0.01, silence_min_duration=0.1, invert_silence_regions=False,
                      energy_sensitivity=0.5, peak_threshold=0.02, peak_min_distance=0.05, 
                      peak_region_size=0.1, group_regions_threshold=0.000):
        """
        Create an options configuration object for the Audio Analyzer node.
        
        Args:
            silence_threshold: Threshold for silence detection
            silence_min_duration: Minimum silence duration
            invert_silence_regions: Whether to invert silence detection to get speech regions
            energy_sensitivity: Sensitivity for energy detection
            peak_threshold: Threshold for peak detection
            peak_min_distance: Minimum distance between peaks
            peak_region_size: Size of regions around peaks
            group_regions_threshold: Time threshold for grouping nearby regions
            
        Returns:
            Tuple containing the options dictionary
        """
        
        # Validate inputs
        options = {
            # Silence detection options
            "silence_threshold": max(0.001, min(0.1, silence_threshold)),
            "silence_min_duration": max(0.01, min(2.0, silence_min_duration)),
            "invert_silence_regions": bool(invert_silence_regions),
            
            # Energy detection options
            "energy_sensitivity": max(0.0, min(1.0, energy_sensitivity)),
            
            # Peak detection options  
            "peak_threshold": max(0.001, min(0.5, peak_threshold)),
            "peak_min_distance": max(0.01, min(1.0, peak_min_distance)),
            "peak_region_size": max(0.02, min(1.0, peak_region_size)),
            
            # Region grouping options
            "group_regions_threshold": max(0.000, min(3.000, group_regions_threshold)),
        }
        
        # Add metadata
        options["_node_type"] = "AudioAnalyzerOptions"
        options["_version"] = "1.0"
        
        return (options,)
    
    def validate_inputs(self, **inputs) -> Dict[str, Any]:
        """Validate node inputs."""
        validated = {}
        
        # Validate silence options
        validated["silence_threshold"] = max(0.001, min(0.1, inputs.get("silence_threshold", 0.01)))
        validated["silence_min_duration"] = max(0.01, min(2.0, inputs.get("silence_min_duration", 0.1)))
        validated["invert_silence_regions"] = bool(inputs.get("invert_silence_regions", False))
        
        # Validate energy options
        validated["energy_sensitivity"] = max(0.0, min(1.0, inputs.get("energy_sensitivity", 0.5)))
        
        # Validate peak options
        validated["peak_threshold"] = max(0.001, min(0.5, inputs.get("peak_threshold", 0.02)))
        validated["peak_min_distance"] = max(0.01, min(1.0, inputs.get("peak_min_distance", 0.05)))
        validated["peak_region_size"] = max(0.02, min(1.0, inputs.get("peak_region_size", 0.1)))
        
        # Validate grouping options
        validated["group_regions_threshold"] = max(0.000, min(3.000, inputs.get("group_regions_threshold", 0.000)))
        
        return validated


# Node class mappings for registration
NODE_CLASS_MAPPINGS = {
    "AudioAnalyzerOptionsNode": AudioAnalyzerOptionsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioAnalyzerOptionsNode": "🔧 Audio Analyzer Options"
}