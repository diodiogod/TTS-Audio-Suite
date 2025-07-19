"""
ChatterBox Voice SRT Package
SRT subtitle processing and timing functionality
"""

# Version info
__version__ = "3.2.1"
__author__ = "Diogod"

# Import the new SRT modules
from .timing_engine import TimingEngine
from .audio_assembly import AudioAssemblyEngine
from .reporting import SRTReportGenerator

__all__ = [
    "TimingEngine",
    "AudioAssemblyEngine",
    "SRTReportGenerator"
]