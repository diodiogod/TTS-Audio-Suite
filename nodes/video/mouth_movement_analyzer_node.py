"""
Silent Speech Analyzer Node
Detects and analyzes mouth movement in silent video frames to extract precise mouth movement timing for TTS SRT synchronization.
"""

import os
import json
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum

import numpy as np
import folder_paths
import nodes
import hashlib
import pickle

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

# Global cache for mouth movement analysis
MOUTH_MOVEMENT_CACHE = {}


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
    AUDIO_REGIONS = "AUDIO_REGIONS"


class SRTPlaceholderFormat(Enum):
    """Available SRT placeholder formats"""
    WORDS = "Words"
    SYLLABLES = "Syllables"
    CHARACTERS = "Characters"
    UNDERSCORES = "Underscores"
    DURATION_INFO = "Duration + Length"


class MouthMovementAnalyzerNode(BaseNode):
    """
    Analyzes videos to detect mouth movement timing for TTS synchronization
    Supports multiple computer vision providers and export formats
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        import sys
        
        # Smart default provider selection based on Python version
        default_provider = AnalysisProvider.OPENSEEFACE.value if sys.version_info >= (3, 13) else AnalysisProvider.MEDIAPIPE.value
        
        return {
            "required": {
                "video": ("VIDEO",),
                "provider": ([p.value for p in AnalysisProvider], {
                    "default": default_provider,
                    "tooltip": "Computer vision provider for mouth movement detection:\n\n• MediaPipe: Google's ML framework (preferred, incompatible with Python 3.13)\n  - Fast, accurate, works on most hardware\n  - Best for general use and consistent results\n\n• OpenSeeFace: Real-time face tracking (experimental, Python 3.13 compatible)\n  - Alternative for newer Python versions\n  - Results may be less accurate than MediaPipe\n\n• dlib: Traditional computer vision (coming soon)\n  - Lightweight, no ML dependencies"
                }),
                "sensitivity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.05,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Ultra-fine movement detection sensitivity (exponential scaling):\n\n• 0.05-0.2: Only obvious mouth movements (conservative)\n• 0.3-0.4: Clear speech detection (balanced)\n• 0.5-0.6: Most speech including soft talking\n• 0.7-0.8: Sensitive, catches subtle movements\n• 0.9-1.0: Ultra-sensitive, includes whispers and micro-movements\n\nExponential scaling provides fine control at higher values.\nStart with 0.5, then fine-tune in 0.01 increments."
                }),
                "min_duration": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Minimum duration for valid speech segments (in seconds):\n\nLower values: Include quick sounds and short words, more segments\nHigher values: Only longer phrases, cleaner but may miss short words\n\nRecommended: 0.1s for balanced filtering, 0.05s for detailed analysis"
                }),
                "output_format": ([f.value for f in OutputFormat], {
                    "default": OutputFormat.SRT.value,
                    "tooltip": "Output format for timing data:\n\n• SRT: Standard subtitle format, best for TTS synchronization\n• JSON: Detailed data with confidence scores for analysis\n• CSV: Spreadsheet format for data processing\n• AUDIO_REGIONS: start,end format compatible with Audio Analyzer\n\nRecommended: SRT for TTS workflows, AUDIO_REGIONS for audio analysis"
                }),
                "srt_placeholder_format": ([f.value for f in SRTPlaceholderFormat], {
                    "default": SRTPlaceholderFormat.WORDS.value,
                    "tooltip": "SRT placeholder format - adapts to viseme detection:\n\n🔌 WITHOUT Viseme Options:\n• Words: [word word word] - estimated word placeholders\n• Syllables: [syl-la-ble syl-la-ble] - estimated syllable patterns\n• Characters: [••••••••••••••••••••] - character count\n\n🔗 WITH Viseme Options connected:\n• Words: [A EI OU] - vowels grouped as words\n• Syllables: [AE-I-OU] - vowels grouped as syllables\n• Characters: [AEIOU] - raw vowel sequence\n\nFormat = presentation style, visemes = actual vowel content!"
                }),
            },
            "optional": {
                "viseme_options": ("VISEME_OPTIONS", {
                    "tooltip": "Connect Viseme Mouth Shape Options node for vowel detection (A, E, I, O, U):\n\n• When connected: Enables precise vowel classification\n• When not connected: Basic speech/no-speech detection only\n\nViseme detection adds ~20% processing time but provides detailed mouth shape analysis for lip-sync."
                }),
                "preview_mode": ("BOOLEAN", {
                    "default": True,
                    "label": "Show preview with movement markers",
                    "tooltip": "Generate annotated video preview with movement markers:\n\nShows green/red overlays for detected/undetected movements with confidence scores and facial landmarks.\n\nPerformance: Uses 540p resolution, increases processing time by ~40%\n\nUse for: Debugging detection accuracy and tuning parameters"
                }),
                "clean_subtitle_output": ("BOOLEAN", {
                    "default": False,
                    "label": "Clean Subtitle Output",
                    "tooltip": "Remove all brackets, confidence scores, and metadata from output:\n\n• OFF: [hello world] (confidence: 74.2%, 1.1s)\n• ON: hello world\n\nUse when you want clean subtitles ready for direct use without technical information."
                }),
                "merge_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "display": "slider",
                    "label": "Merge gaps shorter than (seconds)",
                    "tooltip": "Merge nearby speech segments separated by short gaps:\n\nLower values: Keep more segments separate, preserve natural pauses\nHigher values: Merge more segments together, smoother but less detailed\n\nRecommended: 0.2s for natural flow, 1.0s+ for sentence-level segments"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Minimum confidence score for including detected movements:\n\n• 0.0-0.2: Include all detections (may include noise)\n• 0.3-0.4: Balanced filtering (recommended start)\n• 0.5-0.7: Conservative, only clear movements\n• 0.8-1.0: Ultra-strict, only highest confidence\n\nConfidence based on landmark quality, face visibility, lighting.\nWith new exponential sensitivity, lower values work better."
                }),
            }
        }
    
    RETURN_TYPES = ("VIDEO", "STRING", "TIMING_DATA", "LIST", "LIST") 
    RETURN_NAMES = ("video", "srt_output", "timing_data", "movement_frames", "confidence_scores")
    
    FUNCTION = "analyze_mouth_movement"
    CATEGORY = "TTS Audio Suite/🎬 Video Analysis"
    OUTPUT_NODE = True
    
    
    def __init__(self):
        super().__init__()
        self.provider_registry = {}
        self._register_providers()
    
    def _generate_cache_key(self, video_input, provider: str, sensitivity: float, 
                           min_duration: float, merge_threshold: float, 
                           confidence_threshold: float, preview_mode: bool,
                           enable_viseme: bool = False, viseme_sensitivity: float = 1.0,
                           viseme_confidence_threshold: float = 0.04, 
                           viseme_smoothing: float = 0.3,
                           enable_consonant_detection: bool = False) -> str:
        """Generate cache key for mouth movement analysis (excludes post-processing parameters)"""
        # Get video source path for cache key
        if hasattr(video_input, 'get_stream_source'):
            video_path = video_input.get_stream_source()
        elif hasattr(video_input, '_VideoFromFile__file'):
            video_path = video_input._VideoFromFile__file
        elif hasattr(video_input, 'video_path'):
            video_path = video_input.video_path()
        elif hasattr(video_input, 'path'):
            video_path = video_input.path
        elif hasattr(video_input, 'file_path'):
            video_path = video_input.file_path
        elif isinstance(video_input, str):
            video_path = video_input
        else:
            video_path = str(video_input)
        
        # Get video file stats for cache invalidation
        try:
            import os
            if os.path.exists(video_path):
                file_stats = os.stat(video_path)
                file_hash = f"{file_stats.st_size}_{file_stats.st_mtime}"
            else:
                file_hash = "unknown_file"
        except:
            file_hash = "unknown_file"
        
        # Create cache data (excludes post-processing parameters like merge_threshold, min_duration, confidence_threshold)
        cache_data = {
            'video_path': video_path,
            'file_hash': file_hash,
            'provider': provider,
            'sensitivity': sensitivity,
            'preview_mode': preview_mode,
            'enable_viseme': enable_viseme,
            'viseme_sensitivity': viseme_sensitivity,
            'viseme_confidence_threshold': viseme_confidence_threshold,
            'viseme_smoothing': viseme_smoothing,
            'enable_consonant_detection': enable_consonant_detection
        }
        
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_analysis(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached mouth movement analysis"""
        return MOUTH_MOVEMENT_CACHE.get(cache_key)
    
    def _cache_analysis(self, cache_key: str, timing_data, movement_frames: List[int], 
                       confidence_scores: List[float], preview_path: Optional[str] = None):
        """Cache mouth movement analysis results"""
        cache_data = {
            'timing_data': timing_data,
            'movement_frames': movement_frames,
            'confidence_scores': confidence_scores,
            'preview_path': preview_path
        }
        MOUTH_MOVEMENT_CACHE[cache_key] = cache_data
        logger.info(f"💾 Cached mouth movement analysis: {cache_key[:8]}...")
    
    def _create_filtered_preview(self, analyzer, cached_timing_data, filtered_segments) -> Optional[str]:
        """Create preview video showing only filtered segments using cached frame data"""
        frame_data = cached_timing_data.metadata.get('frame_data')
        if not frame_data:
            logger.warning("No frame data available for preview generation")
            return None
        
        fps = cached_timing_data.fps
        width = cached_timing_data.metadata.get('video_width')
        height = cached_timing_data.metadata.get('video_height')
        
        if not all([fps, width, height]):
            logger.error("Missing video properties for preview generation")
            return None
        
        # Convert filtered segments to frame-based movement mapping
        movement_by_frame = {}
        for segment in filtered_segments:
            start_frame = int(segment.start_time * fps)
            end_frame = int(segment.end_time * fps)
            for frame_num in range(start_frame, end_frame + 1):
                movement_by_frame[frame_num] = True
        
        # Generate annotated frames using filtered segments
        preview_frames = []
        for frame_info in frame_data:
            frame = frame_info['frame']
            landmarks = frame_info['landmarks']
            frame_number = frame_info['frame_number']
            
            # Determine if this frame should show movement based on filtered segments
            is_moving = movement_by_frame.get(frame_number, False)
            confidence = frame_info['confidence'] if is_moving else 0.0
            
            annotated = analyzer.annotate_frame(
                frame, landmarks, is_moving, confidence,
                current_viseme=frame_info['current_viseme'],
                viseme_confidence=frame_info['viseme_confidence'],
                geometric_features=frame_info['geometric_features'],
                frame_number=frame_number,
                consonant_scores=frame_info['consonant_scores'],
                analyzer_method=frame_info['analyzer_method']
            )
            preview_frames.append(annotated)
        
        # Use the existing _create_preview_video method 
        analyzer._create_preview_video(preview_frames, fps, width, height)
        return analyzer.get_preview_video()
    
    
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
        srt_placeholder_format: str,
        viseme_options=None,
        preview_mode: bool = False,
        clean_subtitle_output: bool = False,
        merge_threshold: float = 0.2,
        confidence_threshold: float = 0.5,
        **kwargs
    ):
        """
        Main analysis function with caching
        """
        logger.info(f"Starting mouth movement analysis with {provider} provider")
        
        # Extract viseme settings from options or use defaults
        if viseme_options is not None:
            enable_viseme_detection = viseme_options.get("enable_viseme_detection", False)
            viseme_sensitivity = viseme_options.get("viseme_sensitivity", 1.0)
            viseme_confidence_threshold = viseme_options.get("viseme_confidence_threshold", 0.04)
            viseme_smoothing = viseme_options.get("viseme_smoothing", 0.3)
            enable_consonant_detection = viseme_options.get("enable_consonant_detection", False)
            enable_word_prediction = viseme_options.get("enable_word_prediction", False)
        else:
            # No viseme options connected - basic speech detection only
            enable_viseme_detection = False
            viseme_sensitivity = 1.0
            viseme_confidence_threshold = 0.04
            viseme_smoothing = 0.3
            enable_consonant_detection = False
            enable_word_prediction = False
        
        # Generate cache key for analysis (excludes format parameters)
        cache_key = self._generate_cache_key(
            video, provider, sensitivity, 0, 0, 0, preview_mode,
            enable_viseme_detection, viseme_sensitivity,
            viseme_confidence_threshold, viseme_smoothing,
            enable_consonant_detection
        )
        
        # First check analysis cache (without post-processing parameters)
        analysis_cached_result = self._get_cached_analysis(cache_key)
        if analysis_cached_result:
            logger.info(f"💾 CACHE HIT for analysis: {cache_key[:8]}...")
            
            # Check if we have filtered result cached with current parameters
            full_cache_key = cache_key + f"_filter_{min_duration}_{merge_threshold}_{confidence_threshold}"
            filtered_cached_result = self._get_cached_analysis(full_cache_key)
            
            if filtered_cached_result:
                logger.info(f"💾 CACHE HIT for filtered result")
                timing_data = filtered_cached_result['timing_data']
                movement_frames = filtered_cached_result['movement_frames'] 
                confidence_scores = filtered_cached_result['confidence_scores']
                preview_path = filtered_cached_result.get('preview_path')
            else:
                logger.info(f"🔄 Re-filtering cached analysis with new parameters...")
                # Get unfiltered segments from cached analysis
                cached_timing_data = analysis_cached_result['timing_data']
                unfiltered_segments = cached_timing_data.metadata.get('unfiltered_segments', [])
                
                if unfiltered_segments:
                    # Re-apply filtering with current parameters
                    provider_class = self.provider_registry[provider]
                    temp_analyzer = provider_class(
                        min_duration=min_duration,
                        merge_threshold=merge_threshold,
                        confidence_threshold=confidence_threshold
                    )
                    filtered_segments = temp_analyzer.filter_segments(unfiltered_segments)
                    
                    # Create new timing data with filtered segments
                    timing_data = cached_timing_data
                    timing_data.segments = filtered_segments
                    timing_data.metadata.update({
                        "total_segments_after_filter": len(filtered_segments)
                    })
                    
                    # Reuse cached movement frames and confidence scores
                    movement_frames = analysis_cached_result['movement_frames']
                    confidence_scores = analysis_cached_result['confidence_scores'] 
                    
                    # Generate new preview with filtered segments if preview mode enabled
                    preview_path = None
                    if preview_mode:
                        logger.info("🎬 Regenerating preview with filtered segments...")
                        preview_path = self._create_filtered_preview(temp_analyzer, cached_timing_data, filtered_segments)
                        if preview_path:
                            logger.info(f"Preview created: {preview_path}")
                    
                    # Cache the filtered result
                    self._cache_analysis(full_cache_key, timing_data, movement_frames, confidence_scores, preview_path)
                    
                    logger.info(f"Re-filtered: {len(filtered_segments)} segments from {len(unfiltered_segments)} unfiltered")
                else:
                    # Fallback: no unfiltered segments available
                    logger.warning("No unfiltered segments in cache, using cached filtered result")
                    timing_data = cached_timing_data
                    movement_frames = analysis_cached_result['movement_frames']
                    confidence_scores = analysis_cached_result['confidence_scores']
                    preview_path = analysis_cached_result.get('preview_path')
        else:
            logger.info(f"🔍 CACHE MISS - analyzing video: {cache_key[:8]}...")
            
            # Generate full cache key for storing results
            full_cache_key = cache_key + f"_filter_{min_duration}_{merge_threshold}_{confidence_threshold}"
            
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
                confidence_threshold=confidence_threshold,
                viseme_sensitivity=viseme_sensitivity,
                viseme_confidence_threshold=viseme_confidence_threshold,
                viseme_smoothing=viseme_smoothing,
                enable_consonant_detection=enable_consonant_detection
            )
            
            # Analyze video with viseme detection if enabled
            if hasattr(analyzer, 'analyze_video'):
                # Check if provider supports viseme detection
                import inspect
                sig = inspect.signature(analyzer.analyze_video)
                if 'enable_viseme' in sig.parameters:
                    # Pass full viseme_options dictionary to provider
                    if 'viseme_options' in sig.parameters:
                        timing_data = analyzer.analyze_video(video, preview_mode=preview_mode, enable_viseme=enable_viseme_detection, viseme_options=viseme_options)
                    else:
                        timing_data = analyzer.analyze_video(video, preview_mode=preview_mode, enable_viseme=enable_viseme_detection)
                else:
                    timing_data = analyzer.analyze_video(video, preview_mode=preview_mode)
                    if enable_viseme_detection:
                        logger.warning(f"{provider} provider doesn't support viseme detection yet")
            else:
                timing_data = analyzer.analyze_video(video, preview_mode=preview_mode)
            
            # Extract movement frames and confidence scores
            movement_frames = []
            confidence_scores = []
            
            for segment in timing_data.segments:
                movement_frames.extend(range(segment.start_frame, segment.end_frame + 1))
                confidence_scores.append(segment.confidence)
            
            # Get preview path if generated
            preview_path = analyzer.get_preview_video() if hasattr(analyzer, 'get_preview_video') else None
            
            # Cache both analysis results (for reuse) and filtered results (for this specific parameter set)
            self._cache_analysis(cache_key, timing_data, movement_frames, confidence_scores, preview_path)  # Analysis cache
            self._cache_analysis(full_cache_key, timing_data, movement_frames, confidence_scores, preview_path)  # Filtered cache
        
        # Format outputs based on selected format
        srt_output = self._format_as_srt(timing_data, srt_placeholder_format, enable_word_prediction, clean_subtitle_output) if output_format == OutputFormat.SRT.value else ""
        
        if output_format == OutputFormat.JSON.value:
            srt_output = self._format_as_json(timing_data)
        elif output_format == OutputFormat.CSV.value:
            srt_output = self._format_as_csv(timing_data)
        elif output_format == OutputFormat.AUDIO_REGIONS.value:
            srt_output = self._format_as_audio_regions(timing_data)
        
        logger.info(f"Analysis complete: {len(timing_data.segments)} segments detected")
        
        # Prepare UI data for video preview (combine Preview Bridge file handling with Save Video video display)
        ui_data = {}
        
        if preview_mode and CV2_AVAILABLE and preview_path:
            if os.path.exists(preview_path):
                try:
                    # Verify file exists and log details
                    if not os.path.exists(preview_path):
                        logger.error(f"Preview video file does not exist: {preview_path}")
                        return
                    
                    file_size = os.path.getsize(preview_path)
                    logger.info(f"Preview video file exists: {preview_path} ({file_size} bytes)")
                    
                    # Store preview in ComfyUI's temp directory instead of output directory
                    try:
                        # Try ComfyUI temp directory first
                        temp_dir = folder_paths.get_temp_directory()
                    except:
                        # Fallback to system temp if ComfyUI doesn't have get_temp_directory
                        import tempfile
                        temp_dir = tempfile.gettempdir()
                    
                    preview_filename = f"viseme_preview_{cache_key[:8]}_{int(time.time())}.webm"
                    preview_temp_path = os.path.join(temp_dir, preview_filename)
                    
                    # Copy preview to temp directory
                    import shutil
                    shutil.copy2(preview_path, preview_temp_path)
                    
                    # Create UI data for ComfyUI display using temp location
                    results = [{
                        "filename": preview_filename,
                        "subfolder": "",
                        "type": "temp"
                    }]
                    ui_data = {
                        "images": results,
                        "animated": (True,)  # This triggers native animation display
                    }
                    
                    # Verify temp file exists
                    if os.path.exists(preview_temp_path):
                        temp_size = os.path.getsize(preview_temp_path)
                        logger.info(f"Preview video ready in ComfyUI temp: {preview_temp_path} ({temp_size} bytes)")
                    else:
                        logger.error(f"Preview video not found in temp: {preview_temp_path}")
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare video preview: {e}")
        
        # Return the original video (preview is handled via UI data)
        output_video = video
        
        return {
            "ui": ui_data,
            "result": (output_video, srt_output, timing_data, movement_frames, confidence_scores)
        }
    
    def _format_as_srt(self, timing_data, placeholder_format: str, enable_word_prediction: bool = False, clean_subtitle_output: bool = False) -> str:
        """Convert timing data to SRT format with user-selected placeholder format"""
        srt_lines = []
        
        # Initialize word prediction if enabled
        phoneme_matcher = None
        if enable_word_prediction:
            try:
                # Import here to avoid dependency issues
                import sys
                import os
                utils_path = os.path.join(project_root, "utils")
                if utils_path not in sys.path:
                    sys.path.insert(0, utils_path)
                
                from utils.phoneme_matcher import get_phoneme_matcher
                phoneme_matcher = get_phoneme_matcher()
                logger.info("Word prediction enabled with phoneme matcher")
            except Exception as e:
                logger.warning(f"Could not initialize word prediction: {e}")
                enable_word_prediction = False
        
        # Sort segments by start time to ensure proper chronological order
        sorted_segments = sorted(timing_data.segments, key=lambda s: s.start_time)
        
        for i, segment in enumerate(sorted_segments, 1):
            start_time = self._seconds_to_srt_time(segment.start_time)
            end_time = self._seconds_to_srt_time(segment.end_time)
            
            # Calculate segment duration
            duration = segment.end_time - segment.start_time
            
            # Check if we have viseme data for this segment
            has_visemes = (hasattr(segment, 'viseme_sequence') and 
                          segment.viseme_sequence and 
                          len(segment.viseme_sequence) > 0)
            
            # Generate placeholder based on selected format
            if placeholder_format == SRTPlaceholderFormat.WORDS.value:
                if has_visemes:
                    # Estimate word boundaries in viseme sequence (using pauses and timing)
                    visemes = segment.viseme_sequence
                    words_per_second = 3.5  # Standard speech rate
                    estimated_words = max(1, round(duration * words_per_second))
                    
                    # Split visemes into estimated word chunks, but limit chunk size
                    if len(visemes) <= estimated_words:
                        # Each vowel is a separate word
                        word_chunks = list(visemes)
                    else:
                        # Distribute vowels across estimated words, but cap chunk size
                        chunk_size = len(visemes) // estimated_words
                        remainder = len(visemes) % estimated_words
                        
                        # Limit chunk size to avoid very long phoneme sequences
                        max_chunk_size = 8  # Reasonable limit for readability
                        if chunk_size > max_chunk_size:
                            # Recalculate with more words to keep chunks smaller
                            estimated_words = max(estimated_words, len(visemes) // max_chunk_size)
                            chunk_size = len(visemes) // estimated_words
                            remainder = len(visemes) % estimated_words
                        
                        word_chunks = []
                        chunk_start = 0
                        for w in range(estimated_words):
                            # Add extra vowel to some words if remainder
                            size = chunk_size + (1 if w < remainder else 0)
                            size = max(1, min(size, max_chunk_size))  # Ensure reasonable size
                            if chunk_start < len(visemes):
                                word_chunks.append(visemes[chunk_start:chunk_start+size])
                                chunk_start += size
                    
                    # Apply word prediction if enabled
                    if enable_word_prediction and phoneme_matcher:
                        predicted_words = []
                        prediction_used = False
                        
                        for chunk in word_chunks:
                            suggestions = phoneme_matcher.get_word_suggestions_for_segment(chunk)
                            if suggestions and suggestions[0] != chunk:  # Only use if different from phonemes
                                predicted_words.append(suggestions[0])
                                prediction_used = True
                            else:
                                predicted_words.append(chunk)  # Fallback to phonemes
                        
                        # Apply repetition penalty to avoid "it is it is it is"
                        predicted_words = self._reduce_word_repetition(predicted_words)
                        
                        placeholder = " ".join(predicted_words)
                        avg_confidence = sum(segment.viseme_confidences) / len(segment.viseme_confidences) if segment.viseme_confidences else 0
                        
                        if prediction_used:
                            info = f"(predicted, confidence: {avg_confidence:.1%}, {duration:.1f}s)"
                        else:
                            info = f"(confidence: {avg_confidence:.1%}, {duration:.1f}s)"
                    else:
                        # Convert word_chunks (lists of visemes) to strings
                        chunk_strings = []
                        for chunk in word_chunks:
                            if isinstance(chunk, list):
                                chunk_strings.append("".join(chunk))  # Join visemes in chunk
                            else:
                                chunk_strings.append(str(chunk))
                        
                        placeholder = " ".join(chunk_strings)
                        avg_confidence = sum(segment.viseme_confidences) / len(segment.viseme_confidences) if segment.viseme_confidences else 0
                        info = f"(confidence: {avg_confidence:.1%}, {duration:.1f}s)"
                else:
                    # Fallback to estimated words
                    estimated_words = max(1, int(duration * 3.5))
                    placeholder = " ".join(["word"] * estimated_words)
                    info = f"({estimated_words} word{'s' if estimated_words != 1 else ''}, {duration:.1f}s)"
                
            elif placeholder_format == SRTPlaceholderFormat.SYLLABLES.value:
                if has_visemes:
                    # Create syllable structure with word boundaries like "syl-la-ble syl-la-ble"
                    visemes = segment.viseme_sequence
                    words_per_second = 3.5
                    syllables_per_word = 2.5  # Average syllables per word
                    estimated_words = max(1, round(duration * words_per_second))
                    
                    # Split visemes into word groups first
                    if len(visemes) <= estimated_words:
                        word_groups = [[v] for v in visemes]  # Each vowel is a word
                    else:
                        # Distribute vowels across estimated words
                        chunk_size = len(visemes) // estimated_words
                        remainder = len(visemes) % estimated_words
                        
                        word_groups = []
                        group_start = 0
                        for w in range(estimated_words):
                            size = chunk_size + (1 if w < remainder else 0)
                            size = max(1, size)
                            word_groups.append(list(visemes[group_start:group_start+size]))
                            group_start += size
                    
                    # Convert each word group to syllable format (split into 1-2 vowel chunks)
                    word_syllables = []
                    for word_vowels in word_groups:
                        if len(word_vowels) <= 2:
                            # Short word - keep as single syllable
                            word_syllables.append("".join(word_vowels))
                        else:
                            # Split into syllables of 1-2 vowels
                            syllables = []
                            syl_idx = 0
                            while syl_idx < len(word_vowels):
                                syl_size = min(2, len(word_vowels) - syl_idx)
                                syllables.append("".join(word_vowels[syl_idx:syl_idx+syl_size]))
                                syl_idx += syl_size
                            word_syllables.append("-".join(syllables))
                    
                    # Apply word prediction for syllables if enabled
                    if enable_word_prediction and phoneme_matcher:
                        predicted_syllables = []
                        for syllable_group in word_syllables:
                            # Try to predict words from syllable patterns
                            clean_group = syllable_group.replace('-', '')
                            suggestions = phoneme_matcher.get_word_suggestions_for_segment(clean_group)
                            if suggestions:
                                # Convert predicted word back to syllable format
                                word = suggestions[0].replace('?', '').replace('(', '').replace(')', '')
                                if len(word) > 3:
                                    # Split longer words into syllable-like chunks
                                    mid = len(word) // 2
                                    predicted_syllables.append(f"{word[:mid]}-{word[mid:]}")
                                else:
                                    predicted_syllables.append(word)
                            else:
                                predicted_syllables.append(syllable_group)
                        placeholder = " ".join(predicted_syllables)
                        avg_confidence = sum(segment.viseme_confidences) / len(segment.viseme_confidences) if segment.viseme_confidences else 0
                        info = f"(predicted syllables, confidence: {avg_confidence:.1%}, {duration:.1f}s)"
                    else:
                        placeholder = " ".join(word_syllables)  # Space separates words, hyphens separate syllables
                        avg_confidence = sum(segment.viseme_confidences) / len(segment.viseme_confidences) if segment.viseme_confidences else 0
                        info = f"(confidence: {avg_confidence:.1%}, {duration:.1f}s)"
                else:
                    # Fallback to estimated syllables
                    estimated_syllables = max(1, int(duration * 4.5))
                    placeholder = " ".join(["syl-la-ble"] * (estimated_syllables // 3 + 1))[:estimated_syllables * 4]
                    info = f"({estimated_syllables} syllable{'s' if estimated_syllables != 1 else ''}, {duration:.1f}s)"
                
            elif placeholder_format == SRTPlaceholderFormat.CHARACTERS.value:
                if has_visemes:
                    # Apply word prediction for character-level if enabled
                    if enable_word_prediction and phoneme_matcher:
                        # Try to predict from entire sequence
                        suggestions = phoneme_matcher.get_word_suggestions_for_segment(segment.viseme_sequence)
                        if suggestions and suggestions[0] != segment.viseme_sequence:
                            # Show best suggestion only
                            placeholder = f"{suggestions[0]} ({segment.viseme_sequence})"
                        else:
                            placeholder = segment.viseme_sequence
                        avg_confidence = sum(segment.viseme_confidences) / len(segment.viseme_confidences) if segment.viseme_confidences else 0
                        info = f"(phoneme→word, confidence: {avg_confidence:.1%}, {duration:.1f}s)"
                    else:
                        # Show raw viseme sequence as characters
                        placeholder = segment.viseme_sequence
                        avg_confidence = sum(segment.viseme_confidences) / len(segment.viseme_confidences) if segment.viseme_confidences else 0
                        info = f"(confidence: {avg_confidence:.1%}, {duration:.1f}s)"
                else:
                    # Fallback to estimated characters
                    estimated_chars = max(1, int(duration * 20))
                    placeholder = "•" * estimated_chars
                    info = f"({estimated_chars} char{'s' if estimated_chars != 1 else ''}, {duration:.1f}s)"
                
            elif placeholder_format == SRTPlaceholderFormat.UNDERSCORES.value:
                # Visual word slots with underscores
                estimated_words = max(1, int(duration * 3.5))
                placeholder = " ".join(["_"] * estimated_words)
                info = f"({estimated_words} slot{'s' if estimated_words != 1 else ''}, {duration:.1f}s)"
                
            elif placeholder_format == SRTPlaceholderFormat.DURATION_INFO.value:
                # Duration with visual length indicator
                estimated_chars = max(1, int(duration * 20))
                placeholder = f"{duration:.1f}s: " + "_" * min(estimated_chars, 40)  # Cap at 40 chars for readability
                info = f"({estimated_chars} chars max)"
                
            else:
                # Fallback to words format
                estimated_words = max(1, int(duration * 3.5))
                placeholder = " ".join(["word"] * estimated_words)
                info = f"({estimated_words} word{'s' if estimated_words != 1 else ''}, {duration:.1f}s)"
            
            # Debug: Log what number we're assigning
            if i <= 3:  # Only log first few
                logger.debug(f"SRT segment {i}: {start_time} -> {end_time}")
            
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            # Apply clean subtitle output toggle
            if clean_subtitle_output:
                srt_lines.append(placeholder)  # Clean output - no brackets or metadata
            else:
                srt_lines.append(f"[{placeholder}] {info}")  # Standard output with metadata
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
    
    def _format_as_audio_regions(self, timing_data) -> str:
        """Convert timing data to Audio Analyzer compatible format (start,end)"""
        lines = []
        
        for seg in timing_data.segments:
            lines.append(f"{seg.start_time:.3f},{seg.end_time:.3f}")
        
        return "\n".join(lines)
    
    def _reduce_word_repetition(self, words: List[str]) -> List[str]:
        """
        Reduce repetitive words in predictions to avoid "it is it is it is"
        
        Args:
            words: List of predicted words
            
        Returns:
            List with reduced repetition
        """
        if len(words) <= 1:
            return words
        
        result = []
        for i, word in enumerate(words):
            # Check if this word appeared recently (within last 2 positions)
            recent_positions = max(0, len(result) - 2)
            recent_words = result[recent_positions:]
            
            if word in recent_words:
                # Word is repetitive, try to find phonetically similar alternatives
                found_alternative = False
                
                # Try phonetic alternatives using the phoneme matcher
                try:
                    from utils.phoneme_matcher import get_phoneme_matcher
                    matcher = get_phoneme_matcher()
                    
                    # Create a simple pattern from the word (approximate reverse mapping)
                    if len(word) <= 4:
                        # For short words, try to find phonetic alternatives
                        # Use word's first/dominant characters as pattern
                        dominant_chars = []
                        for char in word.lower():
                            if char in 'aeiou':
                                dominant_chars.append(char.upper())
                            elif char in 'bpmfvtdnkgrlw':
                                dominant_chars.append(char.upper())
                        
                        if dominant_chars:
                            # Try pattern matching for alternatives
                            pattern = ''.join(dominant_chars[:3])  # Use first 3 chars as pattern
                            alternatives = matcher.match_phonemes_to_words(pattern, max_suggestions=5)
                            
                            for alt_word, confidence in alternatives:
                                if alt_word != word and alt_word not in recent_words and len(alt_word) <= 6:
                                    result.append(alt_word)
                                    found_alternative = True
                                    break
                except Exception:
                    pass  # Fall back to hardcoded alternatives if phoneme matcher fails
                
                # Fallback to hardcoded alternatives for common short words
                if not found_alternative and len(word) <= 3:
                    alternatives = {
                        'it': ['is', 'in', 'if'],
                        'is': ['it', 'in', 'as'], 
                        'at': ['an', 'as', 'ah'],
                        'to': ['so', 'do', 'go'],
                        'he': ['we', 'me', 'by'],
                        'be': ['by', 'me', 'we', 'boy', 'big', 'bay'],  # More B alternatives
                        'oh': ['uh', 'ah', 'eh'],
                        'eh': ['oh', 'uh', 'ah']
                    }
                    
                    if word in alternatives:
                        # Try each alternative
                        for alt in alternatives[word]:
                            if alt not in recent_words:
                                result.append(alt)
                                found_alternative = True
                                break
                
                # Last resort: keep original word rather than blank it out
                if not found_alternative:
                    result.append(word)  # Keep original rather than losing phonetic info
                else:
                    # Longer words - just use placeholder to avoid repetition
                    result.append('_')
            else:
                # Word is not repetitive, use as-is
                result.append(word)
        
        return result


NODE_CLASS_MAPPINGS = {
    "MouthMovementAnalyzer": MouthMovementAnalyzerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MouthMovementAnalyzer": "🗣️ Silent Speech Analyzer"
}