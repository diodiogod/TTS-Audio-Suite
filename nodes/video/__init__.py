"""
Video analysis nodes for ComfyUI TTS Audio Suite
"""

from .mouth_movement_analyzer_node import MouthMovementAnalyzerNode

NODE_CLASS_MAPPINGS = {
    "MouthMovementAnalyzer": MouthMovementAnalyzerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MouthMovementAnalyzer": "🗣️ Silent Speech Analyzer"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']