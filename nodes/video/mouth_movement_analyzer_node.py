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
import nodes

try:
    import cv2
    import torch
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

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
                    "default": AnalysisProvider.MEDIAPIPE.value,
                    "tooltip": "Computer vision provider for mouth movement detection:\n\n• MediaPipe: Google's ML framework with 468 facial landmarks\n  - Fast, accurate, works on most hardware\n  - Best for general use and consistent results\n\n• OpenSeeFace: Real-time face tracking (coming soon)\n  - More detailed expression analysis\n  - Better for subtle movements\n\n• dlib: Traditional computer vision (coming soon)\n  - Lightweight, no ML dependencies\n  - Good for older hardware"
                }),
                "sensitivity": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Movement detection sensitivity (Mouth Aspect Ratio threshold):\n\n• Lower values (0.1-0.2): Detect only obvious mouth movements\n  - Good for clear speech, reduces false positives\n  - Use when video has noise or artifacts\n\n• Medium values (0.3-0.4): Balanced detection (recommended)\n  - Catches most speech movements reliably\n  - Good starting point for most videos\n\n• Higher values (0.5-1.0): Detect subtle movements\n  - Catches whispers, lip sync, small movements\n  - May include false positives from noise\n\nTip: Start with 0.3 and adjust based on results"
                }),
                "min_duration": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.05,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "number",
                    "tooltip": "Minimum duration for valid speech segments (in seconds):\n\n• Very short (0.05-0.08s): Include quick sounds, stutters\n  - Good for detailed analysis, rapid speech\n  - May include mouth noise, clicks\n\n• Short (0.1-0.15s): Normal speech segments (recommended)\n  - Filters out most noise and artifacts\n  - Good balance for TTS synchronization\n\n• Medium (0.2-0.5s): Only longer phrases\n  - Cleaner output, fewer segments\n  - Good for slow, deliberate speech\n\n• Long (0.5s+): Only extended speech\n  - Very clean, minimal segments\n  - May miss short words\n\nTip: 0.1s works well for most TTS applications"
                }),
                "output_format": ([f.value for f in OutputFormat], {
                    "default": OutputFormat.SRT.value,
                    "tooltip": "Output format for timing data:\n\n• SRT: SubRip subtitle format\n  - Standard timing format: 00:01:23,456 --> 00:01:25,789\n  - Compatible with most video players and TTS systems\n  - Best for TTS synchronization workflows\n\n• JSON: Structured data format\n  - Detailed analysis data with confidence scores\n  - Includes metadata and frame-level information\n  - Good for custom processing or debugging\n\n• CSV: Comma-separated values\n  - Simple spreadsheet format\n  - Easy to import into Excel or data analysis tools\n  - Good for statistical analysis\n\n• TIMING_DATA: Internal ComfyUI format\n  - For connecting to other TTS Audio Suite nodes\n  - Preserves all analysis details and metadata\n\nRecommended: Use SRT for TTS workflows, JSON for analysis"
                }),
            },
            "optional": {
                "preview_mode": ("BOOLEAN", {
                    "default": False,
                    "label": "Show preview with movement markers",
                    "tooltip": "Generate annotated video preview showing detected movements:\n\n• When enabled: Creates a preview video with visual markers\n  - Green overlay: Active mouth movement detected\n  - Red overlay: No movement detected\n  - Confidence scores and MAR values displayed\n  - Facial landmarks and mouth region highlighted\n\n• Performance impact:\n  - Uses 540p resolution for faster processing\n  - Fast VP9 encoding for quick generation\n  - Increases processing time by ~30-50%\n\n• Use cases:\n  - Debugging detection accuracy\n  - Visualizing analysis results\n  - Tuning sensitivity parameters\n  - Presentations and demonstrations\n\nTip: Enable for testing, disable for production workflows"
                }),
                "merge_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "label": "Merge gaps shorter than (seconds)",
                    "tooltip": "Merge nearby speech segments separated by short gaps:\n\n• No merging (0.0s): Keep all segments separate\n  - Preserves every detected pause\n  - May create many short segments\n  - Good for detailed timing analysis\n\n• Light merging (0.1-0.2s): Merge very short pauses (recommended)\n  - Combines words separated by brief pauses\n  - Reduces segment count while preserving speech patterns\n  - Good for natural TTS flow\n\n• Medium merging (0.3-0.5s): Merge longer pauses\n  - Creates phrase-level segments\n  - Smoother TTS synchronization\n  - May lose some natural speech rhythm\n\n• Heavy merging (0.6s+): Only major pauses separate segments\n  - Creates sentence-level segments\n  - Very smooth but may lose detail\n\nExample: Two segments [0-2s] and [2.1-4s] with 0.2s threshold\nbecome one segment [0-4s] because gap (0.1s) < threshold (0.2s)"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Minimum confidence score for including detected movements:\n\n• Low threshold (0.0-0.3): Include uncertain detections\n  - Catches more potential movements\n  - May include false positives from noise, shadows\n  - Good for subtle or unclear videos\n\n• Medium threshold (0.4-0.6): Balanced filtering (recommended)\n  - Good balance of accuracy vs completeness\n  - Filters out most false positives\n  - Retains clear mouth movements\n\n• High threshold (0.7-1.0): Only high-confidence detections\n  - Very clean results, minimal false positives\n  - May miss subtle or partially obscured movements\n  - Good for high-quality, well-lit videos\n\nConfidence is based on:\n• MediaPipe landmark detection quality\n• Face visibility and lighting\n• Motion clarity and consistency\n• Facial orientation and occlusion\n\nTip: Start with 0.5 and lower if missing movements,\nraise if getting too many false detections"
                }),
            }
        }
    
    RETURN_TYPES = ("VIDEO", "TIMING_DATA", "STRING", "LIST", "LIST")
    RETURN_NAMES = ("video", "timing_data", "srt_output", "movement_frames", "confidence_scores")
    
    FUNCTION = "analyze_mouth_movement"
    CATEGORY = "image/video"
    OUTPUT_NODE = True
    
    # Control animation widget behavior
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")  # Always re-execute for fresh animation
    
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
        
        logger.info(f"Analysis complete: {len(timing_data.segments)} segments detected")
        
        # Prepare UI data for video preview (combine Preview Bridge file handling with Save Video video display)
        ui_data = {}
        
        if preview_mode and CV2_AVAILABLE:
            # Get the preview video path 
            preview_path = analyzer.get_preview_video() if hasattr(analyzer, 'get_preview_video') else None
            if preview_path and os.path.exists(preview_path):
                try:
                    # Verify file exists and log details
                    if not os.path.exists(preview_path):
                        logger.error(f"Preview video file does not exist: {preview_path}")
                        return
                    
                    file_size = os.path.getsize(preview_path)
                    logger.info(f"Preview video file exists: {preview_path} ({file_size} bytes)")
                    
                    # Get just the filename from the full path
                    preview_filename = os.path.basename(preview_path)
                    
                    # Create UI data exactly like SaveAnimatedWEBP does (native ComfyUI)
                    results = [{
                        "filename": preview_filename,
                        "subfolder": "",
                        "type": "output"
                    }]
                    ui_data = {
                        "images": results,
                        "animated": (True,)  # This triggers native animation display
                    }
                    
                    # Verify the file was copied to output directory
                    output_dir = folder_paths.get_output_directory()
                    expected_output_path = os.path.join(output_dir, preview_filename)
                    
                    if os.path.exists(expected_output_path):
                        output_size = os.path.getsize(expected_output_path)
                        logger.info(f"Preview video ready: {preview_filename} in output directory ({output_size} bytes)")
                    else:
                        logger.error(f"Preview video not found in output directory: {expected_output_path}")
                        # Copy from original location to output directory
                        import shutil
                        shutil.copy2(preview_path, expected_output_path)
                        logger.info(f"Copied preview video to output directory: {expected_output_path}")
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare video preview: {e}")
        
        # Return the original video (preview is handled via UI data)
        output_video = video
        
        return {
            "ui": ui_data,
            "result": (output_video, timing_data, srt_output, movement_frames, confidence_scores)
        }
    
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