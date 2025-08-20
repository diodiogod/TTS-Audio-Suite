"""
SRT Overlap Detection and Smart Natural Fallback Utility
Modularized overlap detection and automatic fallback logic for all TTS engines
"""

from typing import List, Tuple


class SRTOverlapHandler:
    """
    Utility class for detecting SRT subtitle overlaps and handling smart_natural mode fallback
    Used by ChatterBox, F5-TTS, and Higgs Audio engines for consistent behavior
    """
    
    @staticmethod
    def detect_overlaps(subtitles: List) -> bool:
        """
        Detect if subtitles have overlapping time ranges
        
        Args:
            subtitles: List of SRTSubtitle objects with start_time and end_time attributes
            
        Returns:
            bool: True if any overlaps are detected, False otherwise
        """
        for i in range(len(subtitles) - 1):
            current = subtitles[i]
            next_sub = subtitles[i + 1]
            if current.end_time > next_sub.start_time:
                return True
        return False
    
    @staticmethod
    def handle_smart_natural_fallback(timing_mode: str, has_overlaps: bool, engine_name: str) -> Tuple[str, bool]:
        """
        Handle automatic fallback from smart_natural to pad_with_silence when overlaps are detected
        
        Args:
            timing_mode: Original timing mode requested by user
            has_overlaps: Whether overlapping subtitles were detected
            engine_name: Name of the engine for logging (e.g., "ChatterBox SRT", "F5-TTS SRT")
            
        Returns:
            Tuple of (actual_timing_mode, mode_switched):
            - actual_timing_mode: The timing mode to actually use
            - mode_switched: Whether a mode switch occurred
        """
        if has_overlaps and timing_mode == "smart_natural":
            print(f"⚠️ {engine_name}: Overlapping subtitles detected, switching from smart_natural to pad_with_silence mode")
            return "pad_with_silence", True
        return timing_mode, False
    
    @staticmethod
    def get_overlap_summary(subtitles: List, has_overlaps: bool) -> str:
        """
        Generate a summary of overlap status for logging
        
        Args:
            subtitles: List of SRTSubtitle objects
            has_overlaps: Whether overlaps were detected
            
        Returns:
            str: Summary message for logging
        """
        if not has_overlaps:
            return f"No overlaps detected in {len(subtitles)} subtitles"
        
        # Count overlapping pairs
        overlap_count = 0
        for i in range(len(subtitles) - 1):
            current = subtitles[i]
            next_sub = subtitles[i + 1]
            if current.end_time > next_sub.start_time:
                overlap_count += 1
        
        return f"Overlaps detected: {overlap_count} overlapping pairs in {len(subtitles)} subtitles"