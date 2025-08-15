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
    TIMING_DATA = "TIMING_DATA"


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
        return {
            "required": {
                "video": ("VIDEO",),
                "provider": ([p.value for p in AnalysisProvider], {
                    "default": AnalysisProvider.MEDIAPIPE.value,
                    "tooltip": "Computer vision provider for mouth movement detection:\n\n• MediaPipe: Google's ML framework with 468 facial landmarks\n  - Fast, accurate, works on most hardware\n  - Best for general use and consistent results\n\n• OpenSeeFace: Real-time face tracking (coming soon)\n  - More detailed expression analysis\n  - Better for subtle movements\n\n• dlib: Traditional computer vision (coming soon)\n  - Lightweight, no ML dependencies\n  - Good for older hardware"
                }),
                "sensitivity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.05,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Ultra-fine movement detection sensitivity (exponential scaling):\n\n• 0.05-0.2: Only obvious mouth movements (conservative)\n• 0.3-0.4: Clear speech detection (balanced)\n• 0.5-0.6: Most speech including soft talking\n• 0.7-0.8: Sensitive, catches subtle movements\n• 0.9-1.0: Ultra-sensitive, includes whispers and micro-movements\n\nExponential scaling provides fine control at higher values.\nStart with 0.5, then fine-tune in 0.01 increments."
                }),
                "min_duration": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Minimum duration for valid speech segments (in seconds):\n\nLower values: Include quick sounds and short words, more segments\nHigher values: Only longer phrases, cleaner but may miss short words\n\nRecommended: 0.1s for balanced filtering, 0.05s for detailed analysis"
                }),
                "output_format": ([f.value for f in OutputFormat], {
                    "default": OutputFormat.SRT.value,
                    "tooltip": "Output format for timing data:\n\n• SRT: Standard subtitle format, best for TTS synchronization\n• JSON: Detailed data with confidence scores for analysis\n• CSV: Spreadsheet format for data processing\n• TIMING_DATA: Internal format for connecting to other nodes\n\nRecommended: SRT for TTS workflows, JSON for debugging"
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
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Minimum confidence score for including detected movements:\n\n• 0.0-0.2: Include all detections (may include noise)\n• 0.3-0.4: Balanced filtering (recommended start)\n• 0.5-0.7: Conservative, only clear movements\n• 0.8-1.0: Ultra-strict, only highest confidence\n\nConfidence based on landmark quality, face visibility, lighting.\nWith new exponential sensitivity, lower values work better."
                }),
            }
        }
    
    RETURN_TYPES = ("VIDEO", "TIMING_DATA", "STRING", "LIST", "LIST")
    RETURN_NAMES = ("video", "timing_data", "srt_output", "movement_frames", "confidence_scores")
    
    FUNCTION = "analyze_mouth_movement"
    CATEGORY = "image/video"
    OUTPUT_NODE = True
    
    
    def __init__(self):
        super().__init__()
        self.provider_registry = {}
        self._register_providers()
    
    def _generate_cache_key(self, video_input, provider: str, sensitivity: float, 
                           min_duration: float, merge_threshold: float, 
                           confidence_threshold: float, preview_mode: bool,
                           enable_viseme: bool = False, viseme_sensitivity: float = 1.0,
                           viseme_confidence_threshold: float = 0.4, 
                           viseme_smoothing: float = 0.3,
                           enable_consonant_detection: bool = False) -> str:
        """Generate cache key for mouth movement analysis (excludes SRT format)"""
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
        
        # Create cache data (excludes output_format and srt_placeholder_format)
        cache_data = {
            'video_path': video_path,
            'file_hash': file_hash,
            'provider': provider,
            'sensitivity': sensitivity,
            'min_duration': min_duration,
            'merge_threshold': merge_threshold,
            'confidence_threshold': confidence_threshold,
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
    
    @classmethod
    def IS_CHANGED(cls, video, provider, sensitivity, min_duration, output_format, 
                   srt_placeholder_format, viseme_options=None, preview_mode=False, 
                   merge_threshold=0.2, confidence_threshold=0.5, **kwargs):
        """Cache invalidation - only invalidate for analysis parameters, not format"""
        # Extract viseme settings from options or use defaults
        if viseme_options is not None:
            enable_viseme_detection = viseme_options.get("enable_viseme_detection", False)
            viseme_sensitivity = viseme_options.get("viseme_sensitivity", 1.0)
            viseme_confidence_threshold = viseme_options.get("viseme_confidence_threshold", 0.4)
            viseme_smoothing = viseme_options.get("viseme_smoothing", 0.3)
            enable_consonant_detection = viseme_options.get("enable_consonant_detection", False)
        else:
            enable_viseme_detection = False
            viseme_sensitivity = 1.0
            viseme_confidence_threshold = 0.4
            viseme_smoothing = 0.3
            enable_consonant_detection = False
        
        # Create temporary instance for cache key generation
        temp_instance = cls()
        cache_key = temp_instance._generate_cache_key(
            video, provider, sensitivity, min_duration, 
            merge_threshold, confidence_threshold, preview_mode,
            enable_viseme_detection, viseme_sensitivity, 
            viseme_confidence_threshold, viseme_smoothing,
            enable_consonant_detection
        )
        
        # Return cache key - ComfyUI will only re-execute if this changes
        return cache_key
    
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
            viseme_confidence_threshold = viseme_options.get("viseme_confidence_threshold", 0.4)
            viseme_smoothing = viseme_options.get("viseme_smoothing", 0.3)
            enable_consonant_detection = viseme_options.get("enable_consonant_detection", False)
            enable_word_prediction = viseme_options.get("enable_word_prediction", False)
        else:
            # No viseme options connected - basic speech detection only
            enable_viseme_detection = False
            viseme_sensitivity = 1.0
            viseme_confidence_threshold = 0.4
            viseme_smoothing = 0.3
            enable_consonant_detection = False
            enable_word_prediction = False
        
        # Generate cache key for analysis (excludes format parameters)
        cache_key = self._generate_cache_key(
            video, provider, sensitivity, min_duration,
            merge_threshold, confidence_threshold, preview_mode,
            enable_viseme_detection, viseme_sensitivity,
            viseme_confidence_threshold, viseme_smoothing,
            enable_consonant_detection
        )
        
        # Check cache first
        cached_result = self._get_cached_analysis(cache_key)
        if cached_result:
            logger.info(f"💾 CACHE HIT for mouth movement analysis: {cache_key[:8]}...")
            timing_data = cached_result['timing_data']
            movement_frames = cached_result['movement_frames']
            confidence_scores = cached_result['confidence_scores']
            preview_path = cached_result.get('preview_path')
        else:
            logger.info(f"🔍 CACHE MISS - analyzing video: {cache_key[:8]}...")
            
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
            
            # Cache the analysis results
            self._cache_analysis(cache_key, timing_data, movement_frames, confidence_scores, preview_path)
        
        # Format outputs based on selected format
        srt_output = self._format_as_srt(timing_data, srt_placeholder_format, enable_word_prediction) if output_format in [OutputFormat.SRT.value, OutputFormat.TIMING_DATA.value] else ""
        
        if output_format == OutputFormat.JSON.value:
            srt_output = self._format_as_json(timing_data)
        elif output_format == OutputFormat.CSV.value:
            srt_output = self._format_as_csv(timing_data)
        
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
    
    def _format_as_srt(self, timing_data, placeholder_format: str, enable_word_prediction: bool = False) -> str:
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
                          len(segment.viseme_sequence.strip()) > 0)
            
            # Generate placeholder based on selected format
            if placeholder_format == SRTPlaceholderFormat.WORDS.value:
                if has_visemes:
                    # Estimate word boundaries in viseme sequence (using pauses and timing)
                    visemes = segment.viseme_sequence
                    words_per_second = 3.5  # Standard speech rate
                    estimated_words = max(1, round(duration * words_per_second))
                    
                    # Split visemes into estimated word chunks
                    if len(visemes) <= estimated_words:
                        # Each vowel is a separate word
                        word_chunks = list(visemes)
                    else:
                        # Distribute vowels across estimated words
                        chunk_size = len(visemes) // estimated_words
                        remainder = len(visemes) % estimated_words
                        
                        word_chunks = []
                        i = 0
                        for w in range(estimated_words):
                            # Add extra vowel to some words if remainder
                            size = chunk_size + (1 if w < remainder else 0)
                            size = max(1, size)  # Ensure at least 1 vowel per word
                            word_chunks.append(visemes[i:i+size])
                            i += size
                    
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
                        placeholder = " ".join(word_chunks)
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
                        i = 0
                        for w in range(estimated_words):
                            size = chunk_size + (1 if w < remainder else 0)
                            size = max(1, size)
                            word_groups.append(list(visemes[i:i+size]))
                            i += size
                    
                    # Convert each word group to syllable format (split into 1-2 vowel chunks)
                    word_syllables = []
                    for word_vowels in word_groups:
                        if len(word_vowels) <= 2:
                            # Short word - keep as single syllable
                            word_syllables.append("".join(word_vowels))
                        else:
                            # Split into syllables of 1-2 vowels
                            syllables = []
                            i = 0
                            while i < len(word_vowels):
                                syl_size = min(2, len(word_vowels) - i)
                                syllables.append("".join(word_vowels[i:i+syl_size]))
                                i += syl_size
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
            
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(f"[{placeholder}] {info}")
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
                # Word is repetitive, try alternatives
                if len(word) <= 3:  # Short words - use alternatives
                    alternatives = {
                        'it': ['is', 'in', 'if'],
                        'is': ['it', 'in', 'as'], 
                        'at': ['an', 'as', 'ah'],
                        'to': ['so', 'do', 'go'],
                        'he': ['we', 'me', 'be'],
                        'oh': ['uh', 'ah', 'eh'],
                        'eh': ['oh', 'uh', 'ah']
                    }
                    
                    if word in alternatives:
                        # Try each alternative
                        for alt in alternatives[word]:
                            if alt not in recent_words:
                                result.append(alt)
                                break
                        else:
                            # No good alternative, use original phonemes or simplified
                            result.append('_')
                    else:
                        result.append('_')  # Unknown word, use placeholder
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
    "MouthMovementAnalyzer": "🎥 Mouth Movement Analyzer"
}