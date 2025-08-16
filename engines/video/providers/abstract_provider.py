"""
Abstract base class for mouth movement analysis providers
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class MovementSegment:
    """Represents a detected mouth movement segment"""
    start_time: float  # in seconds
    end_time: float    # in seconds
    start_frame: int
    end_frame: int
    confidence: float  # 0.0 to 1.0
    peak_mar: float    # peak mouth aspect ratio
    viseme_sequence: Optional[str] = None  # e.g., "AAEEIIOOUUA"
    viseme_confidences: Optional[List[float]] = None  # confidence per viseme

@dataclass
class VisemeFrame:
    """Viseme detection for a single frame"""
    frame_index: int
    viseme: str  # 'A', 'E', 'I', 'O', 'U', or 'neutral'
    confidence: float
    geometric_features: Dict[str, float]  # lip_width, lip_height, etc.
    metadata: Optional[Dict[str, Any]] = None  # Additional debug info (scores, etc.)

@dataclass
class TimingData:
    """Standardized timing data format"""
    segments: List[MovementSegment]
    fps: float
    total_frames: int
    total_duration: float
    provider: str
    metadata: Dict[str, Any]
    viseme_frames: Optional[List[VisemeFrame]] = None  # Frame-by-frame viseme data

logger = logging.getLogger(__name__)


class AbstractProvider(ABC):
    """
    Base class for all mouth movement analysis providers
    Defines the standard interface that all providers must implement
    """
    
    def __init__(
        self,
        sensitivity: float = 0.3,
        min_duration: float = 0.1,
        merge_threshold: float = 0.2,
        confidence_threshold: float = 0.5,
        viseme_sensitivity: float = 1.0,
        viseme_confidence_threshold: float = 0.04,
        viseme_smoothing: float = 0.3,
        enable_consonant_detection: bool = False
    ):
        """
        Initialize provider with common parameters
        
        Args:
            sensitivity: Movement detection threshold (0.1-1.0)
            min_duration: Minimum movement duration in seconds
            merge_threshold: Gap threshold for merging nearby movements
            confidence_threshold: Minimum confidence for valid detection
        """
        self.sensitivity = sensitivity
        self.min_duration = min_duration
        self.merge_threshold = merge_threshold
        self.confidence_threshold = confidence_threshold
        self.viseme_sensitivity = viseme_sensitivity
        self.viseme_confidence_threshold = viseme_confidence_threshold
        self.viseme_smoothing = viseme_smoothing
        self.enable_consonant_detection = enable_consonant_detection
        self.preview_video = None
        self.previous_visemes = []  # For smoothing
        
        # Provider-specific initialization
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize provider-specific components"""
        pass
    
    @abstractmethod
    def analyze_video(self, video_input, preview_mode: bool = False) -> TimingData:
        """
        Analyze video and extract mouth movement timing
        
        Args:
            video_input: Video input (ComfyUI video object or file path)
            preview_mode: Whether to generate preview with markers
            
        Returns:
            TimingData object containing all timing information
        """
        pass
    
    @abstractmethod
    def detect_movement(self, frame: np.ndarray) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Detect mouth movement in a single frame
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            Tuple of (is_moving, confidence, landmarks)
        """
        pass
    
    @abstractmethod
    def calculate_mar(self, landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio from landmarks
        
        Args:
            landmarks: Facial landmarks array
            
        Returns:
            MAR value
        """
        pass
    
    def filter_segments(self, segments: List[MovementSegment]) -> List[MovementSegment]:
        """
        Apply filtering to movement segments
        
        - Remove segments shorter than min_duration
        - Merge segments closer than merge_threshold
        - Remove low confidence segments
        """
        if not segments:
            return segments
        
        # Filter by minimum duration
        filtered = [s for s in segments if (s.end_time - s.start_time) >= self.min_duration]
        
        # Filter by confidence
        filtered = [s for s in filtered if s.confidence >= self.confidence_threshold]
        
        # Merge nearby segments
        merged = []
        for segment in filtered:
            if not merged:
                merged.append(segment)
            else:
                last = merged[-1]
                gap = segment.start_time - last.end_time
                
                if gap <= self.merge_threshold:
                    # Merge segments
                    last.end_time = segment.end_time
                    last.end_frame = segment.end_frame
                    last.confidence = max(last.confidence, segment.confidence)
                    last.peak_mar = max(last.peak_mar, segment.peak_mar)
                else:
                    merged.append(segment)
        
        return merged
    
    def smooth_confidence_scores(
        self,
        confidence_scores: List[float],
        window_size: int = 5
    ) -> List[float]:
        """
        Apply smoothing to confidence scores to reduce noise
        
        Args:
            confidence_scores: List of raw confidence values
            window_size: Size of smoothing window
            
        Returns:
            Smoothed confidence scores
        """
        if len(confidence_scores) <= window_size:
            return confidence_scores
        
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(confidence_scores)):
            start = max(0, i - half_window)
            end = min(len(confidence_scores), i + half_window + 1)
            window = confidence_scores[start:end]
            smoothed.append(sum(window) / len(window))
        
        return smoothed
    
    def frames_to_segments(
        self,
        movement_frames: List[bool],
        confidence_scores: List[float],
        mar_values: List[float],
        fps: float
    ) -> List[MovementSegment]:
        """
        Convert frame-by-frame detection to movement segments
        
        Args:
            movement_frames: Boolean list of movement detection per frame
            confidence_scores: Confidence score per frame
            mar_values: MAR value per frame
            fps: Video frames per second
            
        Returns:
            List of MovementSegment objects
        """
        segments = []
        in_segment = False
        start_frame = 0
        segment_confidence = []
        segment_mar = []
        
        for i, is_moving in enumerate(movement_frames):
            if is_moving and not in_segment:
                # Start new segment
                in_segment = True
                start_frame = i
                segment_confidence = [confidence_scores[i]]
                segment_mar = [mar_values[i]]
                
            elif is_moving and in_segment:
                # Continue segment
                segment_confidence.append(confidence_scores[i])
                segment_mar.append(mar_values[i])
                
            elif not is_moving and in_segment:
                # End segment
                in_segment = False
                
                segments.append(MovementSegment(
                    start_time=start_frame / fps,
                    end_time=i / fps,
                    start_frame=start_frame,
                    end_frame=i - 1,
                    confidence=sum(segment_confidence) / len(segment_confidence),
                    peak_mar=max(segment_mar)
                ))
        
        # Handle segment that extends to end of video
        if in_segment:
            segments.append(MovementSegment(
                start_time=start_frame / fps,
                end_time=len(movement_frames) / fps,
                start_frame=start_frame,
                end_frame=len(movement_frames) - 1,
                confidence=sum(segment_confidence) / len(segment_confidence),
                peak_mar=max(segment_mar)
            ))
        
        return segments
    
    def get_preview_frames(self) -> Optional[List[np.ndarray]]:
        """
        Get the preview frames with movement markers if generated
        
        Returns:
            List of preview frames or None
        """
        return getattr(self, 'preview_frames', None)
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray],
        is_moving: bool,
        confidence: float
    ) -> np.ndarray:
        """
        Add visual annotations to frame for preview
        
        Args:
            frame: Original frame
            landmarks: Detected landmarks (if any)
            is_moving: Whether movement is detected
            confidence: Detection confidence
            
        Returns:
            Annotated frame
        """
        import cv2
        annotated = frame.copy()
        
        # Add movement indicator
        color = (0, 255, 0) if is_moving else (255, 0, 0)
        text = f"Moving: {confidence:.2f}" if is_moving else "Still"
        cv2.putText(
            annotated,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )
        
        # Draw landmarks if available
        if landmarks is not None and len(landmarks) > 0:
            for point in landmarks:
                if len(point) >= 2:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(annotated, (x, y), 2, (0, 255, 255), -1)
        
        return annotated
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider"""
        pass
    
    @property
    def is_available(self) -> bool:
        """Check if this provider is available/installed"""
        try:
            self._check_dependencies()
            return True
        except ImportError:
            return False
    
    @abstractmethod
    def _check_dependencies(self):
        """Check if required dependencies are installed"""
        pass