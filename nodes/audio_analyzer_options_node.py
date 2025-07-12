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
        return "🎛️ Audio Analyzer Options"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_method": (["silence", "energy", "peaks", "manual"], {
                    "default": "silence",
                    "tooltip": "Which analysis method these options are for:\n• silence: Configure silence detection parameters\n• energy: Configure energy-based detection\n• peaks: Configure peak detection settings\n• manual: Configure manual region settings"
                }),
            },
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
                
                # Manual region options
                "manual_regions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Define your own timing regions manually.\nFormat: start,end (one per line)\nExample:\n1.5,3.2\n4.0,6.8\n8.1,10.5\n\nBidirectional sync:\n• Type/paste here → syncs to interface when you click back\n• Add regions on interface → automatically updates this text\n• Regions auto-sort chronologically by start time\n\nUse when analysis_method is 'manual' or to add extra regions."
                }),
                "region_labels": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional labels for each region (one per line).\nExample:\nIntro\nVerse 1\nChorus\n\nBidirectional sync:\n• Type/paste custom labels here → syncs to interface\n• Interface preserves custom labels when renumbering\n• Auto-generated labels (Region 1, Region 2) get renumbered\n• Custom labels stay unchanged during chronological sorting\n\nMust match the number of manual_regions lines."
                }),
                
                # Output formatting options
                "export_format": (["f5tts", "json", "csv"], {
                    "default": "f5tts",
                    "tooltip": "How to format the timing_data output:\n• f5tts: Simple format for F5-TTS (start,end per line)\n• json: Full data with confidence, labels, metadata\n• csv: Spreadsheet-compatible format for analysis\n\nAll formats respect the precision_level setting."
                }),
            }
        }
    
    RETURN_TYPES = ("OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "create_options"
    CATEGORY = "ChatterBox Audio"
    
    def create_options(self, analysis_method="silence", silence_threshold=0.01, silence_min_duration=0.1,
                      energy_sensitivity=0.5, peak_threshold=0.02, peak_min_distance=0.05, 
                      peak_region_size=0.1, manual_regions="", region_labels="", export_format="f5tts"):
        """
        Create an options configuration object for the Audio Analyzer node.
        
        Args:
            analysis_method: The analysis method these options are configured for
            silence_threshold: Threshold for silence detection
            silence_min_duration: Minimum silence duration
            energy_sensitivity: Sensitivity for energy detection
            peak_threshold: Threshold for peak detection
            peak_min_distance: Minimum distance between peaks
            peak_region_size: Size of regions around peaks
            manual_regions: Manual timing regions string
            region_labels: Labels for manual regions
            export_format: Output format selection
            
        Returns:
            Tuple containing the options dictionary
        """
        
        # Validate inputs
        options = {
            "analysis_method": analysis_method,
            
            # Silence detection options
            "silence_threshold": max(0.001, min(0.1, silence_threshold)),
            "silence_min_duration": max(0.01, min(2.0, silence_min_duration)),
            
            # Energy detection options
            "energy_sensitivity": max(0.0, min(1.0, energy_sensitivity)),
            
            # Peak detection options  
            "peak_threshold": max(0.001, min(0.5, peak_threshold)),
            "peak_min_distance": max(0.01, min(1.0, peak_min_distance)),
            "peak_region_size": max(0.02, min(1.0, peak_region_size)),
            
            # Manual region options
            "manual_regions": manual_regions,
            "region_labels": region_labels,
            
            # Output options
            "export_format": export_format if export_format in ["f5tts", "json", "csv"] else "f5tts",
        }
        
        # Add metadata
        options["_node_type"] = "AudioAnalyzerOptions"
        options["_version"] = "1.0"
        
        return (options,)
    
    def validate_inputs(self, **inputs) -> Dict[str, Any]:
        """Validate node inputs."""
        validated = {}
        
        # Validate analysis method
        analysis_method = inputs.get("analysis_method", "silence")
        if analysis_method not in ["silence", "energy", "peaks", "manual"]:
            analysis_method = "silence"
        validated["analysis_method"] = analysis_method
        
        # Validate silence options
        validated["silence_threshold"] = max(0.001, min(0.1, inputs.get("silence_threshold", 0.01)))
        validated["silence_min_duration"] = max(0.01, min(2.0, inputs.get("silence_min_duration", 0.1)))
        
        # Validate energy options
        validated["energy_sensitivity"] = max(0.0, min(1.0, inputs.get("energy_sensitivity", 0.5)))
        
        # Validate peak options
        validated["peak_threshold"] = max(0.001, min(0.5, inputs.get("peak_threshold", 0.02)))
        validated["peak_min_distance"] = max(0.01, min(1.0, inputs.get("peak_min_distance", 0.05)))
        validated["peak_region_size"] = max(0.02, min(1.0, inputs.get("peak_region_size", 0.1)))
        
        # Validate text inputs
        validated["manual_regions"] = inputs.get("manual_regions", "")
        validated["region_labels"] = inputs.get("region_labels", "")
        
        # Validate export format
        export_format = inputs.get("export_format", "f5tts")
        if export_format not in ["f5tts", "json", "csv"]:
            export_format = "f5tts"
        validated["export_format"] = export_format
        
        return validated


# Node class mappings for registration
NODE_CLASS_MAPPINGS = {
    "AudioAnalyzerOptionsNode": AudioAnalyzerOptionsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioAnalyzerOptionsNode": "🎛️ Audio Analyzer Options"
}