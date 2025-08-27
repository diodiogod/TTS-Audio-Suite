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
                    "tooltip": "Detect consonants (B, P, M, F, V, TH, etc.) in addition to vowels:\n\n• Vowels only: A, E, I, O, U, _\n• With consonants: A, E, I, O, U, B, P, M, F, V, TH, _, etc.\n• Adds ~10% processing time\n• Provides more detailed phoneme sequences\n• Better for advanced lip-sync and speech analysis\n\nLeave disabled for basic vowel-only detection."
                }),
                "enable_temporal_analysis": ("BOOLEAN", {
                    "default": True,
                    "label": "Enable Temporal Analysis",
                    "tooltip": "Advanced consonant burst detection using 5-frame windows:\n\n• Analyzes onset → peak → release patterns for true consonants\n• Dramatically improves B/P/M distinction accuracy\n• Detects rapid lip closure/release vs sustained patterns\n• Better coarticulation modeling (vowel context)\n• Adds ~50% processing time but much higher accuracy\n• Requires consonant detection enabled\n\nRecommended for: research, high-quality phonetic analysis\nSkip for: basic lip-sync, real-time processing"
                }),
                "enable_word_prediction": ("BOOLEAN", {
                    "default": False,
                    "label": "Enable Word Prediction",
                    "tooltip": "Predict words from detected phoneme sequences:\n\n• Uses 10,000 most common English words\n• Matches phoneme patterns to suggest likely words\n• Shows confidence with ?, (), markers\n• Example: 'AEIOU' → 'you', 'hey?', '(audio)'\n• Helpful for manual SRT editing\n• No processing time impact\n\nRequires viseme detection enabled to work."
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