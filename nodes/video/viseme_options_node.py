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
                    "tooltip": "Enable vowel classification (A, E, I, O, U) for precise lip-sync:\n\n• Analyzes mouth shape geometry beyond simple open/close\n• Detects vowel patterns in mouth movements\n• Adds ~20% processing time\n• Provides phoneme sequences for better TTS synchronization"
                }),
                "viseme_sensitivity": ("FLOAT", {
                    "default": 1.0,
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
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Temporal smoothing to reduce viseme flickering:\n\n• 0.0: No smoothing (immediate response, may flicker)\n• 0.3: Light smoothing (recommended balance)\n• 0.7: Heavy smoothing (stable but slower response)\n• 1.0: Maximum smoothing (very stable, delayed)\n\nReduces rapid switching between visemes for cleaner sequences.\nUseful for noisy videos or subtle mouth movements."
                }),
            }
        }
    
    RETURN_TYPES = ("VISEME_OPTIONS",)
    RETURN_NAMES = ("viseme_options",)
    FUNCTION = "create_viseme_options"
    CATEGORY = "image/video"
    
    def create_viseme_options(
        self,
        enable_viseme_detection: bool,
        viseme_sensitivity: float,
        viseme_confidence_threshold: float,
        viseme_smoothing: float
    ) -> Tuple[Dict[str, Any]]:
        """
        Create viseme options configuration
        
        Returns:
            Dictionary containing all viseme detection settings
        """
        viseme_options = {
            "enable_viseme_detection": enable_viseme_detection,
            "viseme_sensitivity": viseme_sensitivity,
            "viseme_confidence_threshold": viseme_confidence_threshold,
            "viseme_smoothing": viseme_smoothing
        }
        
        logger.info(f"Viseme options created: enabled={enable_viseme_detection}, "
                   f"sensitivity={viseme_sensitivity}, confidence={viseme_confidence_threshold}, "
                   f"smoothing={viseme_smoothing}")
        
        return (viseme_options,)


# Register the node class
NODE_CLASS_MAPPINGS = {
    "VisemeDetectionOptionsNode": VisemeDetectionOptionsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisemeDetectionOptionsNode": "🔧 Viseme Mouth Shape Options"
}