"""
Viseme Detection Options Node
Provides advanced viseme detection settings for mouth movement analysis
"""

import logging
from typing import Tuple, Dict, Any

# Add project root to path for imports
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)


class VisemeDetectionOptionsNode:
    """
    Configuration node for advanced viseme detection settings
    Connects to Mouth Movement Analyzer to enable vowel classification
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_viseme_detection": ("BOOLEAN", {
                    "default": True,
                    "label": "Enable Viseme Detection",
                    "tooltip": "Experimental mouth-shape classification (A, E, I, O, U). These geometric labels are not recognized phonemes or a transcript. Leave this node disconnected for phrase timing and dubbing."
                }),
                "viseme_sensitivity": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "How hard should I look for vowels?\n\nControls geometric thresholds for mouth shape classification:\n\n• 0.1-0.5: Very strict, only obvious vowel shapes\n• 0.8-1.2: Balanced detection (recommended)\n• 1.5-2.0: Lenient, detects subtle vowel variations\n\nHigher = more detections, may include false positives\nLower = fewer detections, higher accuracy"
                }),
                "viseme_confidence_threshold": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "How sure should I be before showing a vowel?\n\nMinimum confidence for valid viseme classification:\n\n• 0.0-0.2: Show all viseme attempts (noisy)\n• 0.3-0.5: Balanced filtering (recommended)\n• 0.6-0.8: Conservative, only clear vowels\n• 0.9-1.0: Ultra-strict, only perfect detections\n\nBelow threshold shows as 'neutral' instead of uncertain vowel."
                }),
                "viseme_smoothing": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Temporal smoothing to reduce viseme flickering:\n\n• 0.0: No smoothing (immediate response, may flicker)\n• 0.3: Light smoothing (recommended balance)\n• 0.7: Heavy smoothing (stable but slower response)\n• 1.0: Maximum smoothing (very stable, delayed)\n\nReduces rapid switching between visemes for cleaner sequences.\nUseful for noisy videos or subtle mouth movements."
                }),
                "enable_consonant_detection": ("BOOLEAN", {
                    "default": True,
                    "label": "Enable Consonant Detection",
                    "tooltip": "Experimental consonant-shaped mouth labels. Visually similar sounds such as B, P and M cannot be reliably distinguished from geometry alone. Not a speech recognizer."
                }),
                "enable_temporal_analysis": ("BOOLEAN", {
                    "default": True,
                    "label": "Enable Temporal Analysis",
                    "tooltip": "Use short frame windows to stabilize experimental consonant labels. This does not establish which consonant was spoken. Extra processing cost depends on the video."
                }),
                "enable_word_prediction": ("BOOLEAN", {
                    "default": False,
                    "label": "Enable Word Prediction",
                    "tooltip": "Experimental English dictionary guesses from mouth-shape labels. Guesses can be wrong even when the displayed geometric score is high. Use ASR or a supplied transcript for dubbing; leave prediction off for timing-only use."
                }),
            }
        }
    
    RETURN_TYPES = ("VISEME_OPTIONS",)
    RETURN_NAMES = ("viseme_options",)
    FUNCTION = "create_viseme_options"
    CATEGORY = "TTS Audio Suite/🎬 Video Analysis"
    
    def create_viseme_options(
        self,
        enable_viseme_detection: bool,
        viseme_sensitivity: float,
        viseme_confidence_threshold: float,
        viseme_smoothing: float,
        enable_consonant_detection: bool,
        enable_temporal_analysis: bool,
        enable_word_prediction: bool
    ) -> Tuple[Dict[str, Any]]:
        """
        Create viseme options configuration
        
        Returns:
            Dictionary containing all viseme detection settings
        """
        # Auto-enable temporal analysis when consonants are enabled for better accuracy
        if enable_consonant_detection:
            enable_temporal_analysis = True
            logger.info("Auto-enabled temporal analysis for consonant detection")

        viseme_options = {
            "enable_viseme_detection": enable_viseme_detection,
            "viseme_sensitivity": viseme_sensitivity,
            "viseme_confidence_threshold": viseme_confidence_threshold,
            "viseme_smoothing": viseme_smoothing,
            "enable_consonant_detection": enable_consonant_detection,
            "enable_temporal_analysis": enable_temporal_analysis,
            "enable_word_prediction": enable_word_prediction
        }
        
        logger.info(f"Viseme options created: enabled={enable_viseme_detection}, "
                   f"consonants={enable_consonant_detection}, temporal={enable_temporal_analysis}, "
                   f"words={enable_word_prediction}, sensitivity={viseme_sensitivity}, "
                   f"confidence={viseme_confidence_threshold}, smoothing={viseme_smoothing}")
        
        return (viseme_options,)


# Register the node class
NODE_CLASS_MAPPINGS = {
    "VisemeDetectionOptionsNode": VisemeDetectionOptionsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisemeDetectionOptionsNode": "🔧 Viseme Mouth Shape Options"
}
