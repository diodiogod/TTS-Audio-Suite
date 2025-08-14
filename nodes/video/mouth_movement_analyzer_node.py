"""
Mouth Movement Analyzer Node
Analyzes silent videos to extract precise mouth movement timing for TTS SRT synchronization
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum

import numpy as np
import folder_paths

import sys
import os

# Add the project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import base node using spec loading like in nodes.py
import importlib.util

def load_base_node():
    """Load base node module"""
    base_node_path = os.path.join(project_root, "nodes", "base", "base_node.py")
    spec = importlib.util.spec_from_file_location("base_node", base_node_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BaseChatterBoxNode

BaseNode = load_base_node()

logger = logging.getLogger(__name__)


class AnalysisProvider(Enum):
    """Available analysis providers"""
    MEDIAPIPE = "MediaPipe"
    OPENSEEFACE = "OpenSeeFace"
    DLIB = "dlib"


class OutputFormat(Enum):
    """Available output formats"""
    SRT = "SRT"
    JSON = "JSON"
    CSV = "CSV"
    TIMING_DATA = "TIMING_DATA"


class MouthMovementAnalyzerNode(BaseNode):
    """
    Analyzes videos to detect mouth movement timing for TTS synchronization
    Supports multiple computer vision providers and export formats
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "provider": ([p.value for p in AnalysisProvider], {
                    "default": AnalysisProvider.MEDIAPIPE.value
                }),
                "sensitivity": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "min_duration": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.05,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "number"
                }),
                "output_format": ([f.value for f in OutputFormat], {
                    "default": OutputFormat.SRT.value
                }),
            },
            "optional": {
                "preview_mode": ("BOOLEAN", {
                    "default": False,
                    "label": "Show preview with movement markers"
                }),
                "merge_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "label": "Merge gaps shorter than (seconds)"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("TIMING_DATA", "STRING", "LIST", "LIST", "VIDEO")
    RETURN_NAMES = ("timing_data", "srt_output", "movement_frames", "confidence_scores", "preview_video")
    
    FUNCTION = "analyze_mouth_movement"
    CATEGORY = "TTS_Audio_Suite/Video"
    
    def __init__(self):
        super().__init__()
        self.provider_registry = {}
        self._register_providers()
    
    def _register_providers(self):
        """Register available analysis providers"""
        # Import providers conditionally based on availability using spec loading
        try:
            mediapipe_path = os.path.join(project_root, "engines", "video", "providers", "mediapipe_provider.py")
            if os.path.exists(mediapipe_path):
                spec = importlib.util.spec_from_file_location("mediapipe_provider", mediapipe_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.provider_registry[AnalysisProvider.MEDIAPIPE.value] = module.MediaPipeProvider
                logger.info("MediaPipe provider registered")
        except Exception as e:
            logger.warning(f"MediaPipe provider not available: {e}")
        
        try:
            openseeface_path = os.path.join(project_root, "engines", "video", "providers", "openseeface_provider.py")
            if os.path.exists(openseeface_path):
                spec = importlib.util.spec_from_file_location("openseeface_provider", openseeface_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.provider_registry[AnalysisProvider.OPENSEEFACE.value] = module.OpenSeeFaceProvider
                logger.info("OpenSeeFace provider registered")
        except Exception as e:
            logger.warning(f"OpenSeeFace provider not available: {e}")
        
        try:
            dlib_path = os.path.join(project_root, "engines", "video", "providers", "dlib_provider.py")
            if os.path.exists(dlib_path):
                spec = importlib.util.spec_from_file_location("dlib_provider", dlib_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.provider_registry[AnalysisProvider.DLIB.value] = module.DlibProvider
                logger.info("dlib provider registered")
        except Exception as e:
            logger.warning(f"dlib provider not available: {e}")
    
    def analyze_mouth_movement(
        self,
        video,
        provider: str,
        sensitivity: float,
        min_duration: float,
        output_format: str,
        preview_mode: bool = False,
        merge_threshold: float = 0.2,
        confidence_threshold: float = 0.5,
        **kwargs
    ):
        """
        Main analysis function
        """
        logger.info(f"Starting mouth movement analysis with {provider} provider")
        
        # Validate provider availability
        if provider not in self.provider_registry:
            available = list(self.provider_registry.keys())
            if not available:
                raise RuntimeError("No analysis providers available. Please install required dependencies.")
            
            logger.warning(f"{provider} not available, falling back to {available[0]}")
            provider = available[0]
        
        # Initialize selected provider
        provider_class = self.provider_registry[provider]
        analyzer = provider_class(
            sensitivity=sensitivity,
            min_duration=min_duration,
            merge_threshold=merge_threshold,
            confidence_threshold=confidence_threshold
        )
        
        # Analyze video
        timing_data = analyzer.analyze_video(video, preview_mode=preview_mode)
        
        # Format outputs based on selected format
        srt_output = self._format_as_srt(timing_data) if output_format in [OutputFormat.SRT.value, OutputFormat.TIMING_DATA.value] else ""
        
        if output_format == OutputFormat.JSON.value:
            srt_output = self._format_as_json(timing_data)
        elif output_format == OutputFormat.CSV.value:
            srt_output = self._format_as_csv(timing_data)
        
        # Extract movement frames and confidence scores
        movement_frames = []
        confidence_scores = []
        
        for segment in timing_data.segments:
            movement_frames.extend(range(segment.start_frame, segment.end_frame + 1))
            confidence_scores.append(segment.confidence)
        
        # Handle preview video (if generated)
        if preview_mode:
            preview_path = analyzer.get_preview_video()
            if preview_path and os.path.exists(preview_path):
                # Convert back to ComfyUI video format
                # For now, return the original video since ComfyUI needs specific video format
                # TODO: Implement proper video format conversion
                preview_video = video
            else:
                preview_video = video
        else:
            preview_video = video
        
        logger.info(f"Analysis complete: {len(timing_data.segments)} segments detected")
        
        return (timing_data, srt_output, movement_frames, confidence_scores, preview_video)
    
    def _format_as_srt(self, timing_data) -> str:
        """Convert timing data to SRT format"""
        srt_lines = []
        
        for i, segment in enumerate(timing_data.segments, 1):
            start_time = self._seconds_to_srt_time(segment.start_time)
            end_time = self._seconds_to_srt_time(segment.end_time)
            
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(f"[Movement {i}]")  # Placeholder text
            srt_lines.append("")
        
        return "\n".join(srt_lines)
    
    def _format_as_json(self, timing_data) -> str:
        """Convert timing data to JSON format"""
        data = {
            "fps": timing_data.fps,
            "total_frames": timing_data.total_frames,
            "total_duration": timing_data.total_duration,
            "provider": timing_data.provider,
            "segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "start_frame": seg.start_frame,
                    "end_frame": seg.end_frame,
                    "confidence": seg.confidence,
                    "peak_mar": seg.peak_mar
                }
                for seg in timing_data.segments
            ],
            "metadata": timing_data.metadata
        }
        return json.dumps(data, indent=2)
    
    def _format_as_csv(self, timing_data) -> str:
        """Convert timing data to CSV format"""
        lines = ["start_time,end_time,start_frame,end_frame,confidence,peak_mar"]
        
        for seg in timing_data.segments:
            lines.append(f"{seg.start_time},{seg.end_time},{seg.start_frame},{seg.end_frame},{seg.confidence},{seg.peak_mar}")
        
        return "\n".join(lines)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


NODE_CLASS_MAPPINGS = {
    "MouthMovementAnalyzer": MouthMovementAnalyzerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MouthMovementAnalyzer": "🎥 Mouth Movement Analyzer"
}