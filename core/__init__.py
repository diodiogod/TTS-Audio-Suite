"""
ChatterBox Voice Core Package
Core utilities and shared functionality for the ChatterBox Voice extension
"""

# Version info
__version__ = "3.0.13"
__author__ = "Diogod"

# Make imports available at package level
from .model_manager import ModelManager
from .import_manager import ImportManager
from .text_chunking import ImprovedChatterBoxChunker
from .audio_processing import AudioProcessingUtils

__all__ = [
    "ModelManager",
    "ImportManager", 
    "ImprovedChatterBoxChunker",
    "AudioProcessingUtils"
]