"""
MediaPipe provider for mouth movement analysis
High-performance facial landmark detection with 468 3D face landmarks
"""

import logging
import os
from typing import Optional, Tuple, List, Any
import numpy as np

try:
    import cv2
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

import sys
import os

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import using spec loading to avoid relative import issues
import importlib.util
abstract_provider_path = os.path.join(current_dir, "abstract_provider.py")
spec = importlib.util.spec_from_file_location("abstract_provider", abstract_provider_path)
abstract_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abstract_module)

AbstractProvider = abstract_module.AbstractProvider
TimingData = abstract_module.TimingData
MovementSegment = abstract_module.MovementSegment

logger = logging.getLogger(__name__)


class MediaPipeProvider(AbstractProvider):
    """
    MediaPipe-based mouth movement detection provider
    Uses Google's MediaPipe Face Mesh for accurate facial landmark tracking
    """
    
    # Mouth landmark indices for MediaPipe Face Mesh (468 landmarks)
    # Upper lip landmarks
    UPPER_LIP_INDICES = [61, 84, 17, 314, 405, 308, 415, 310, 311, 312, 13, 82, 81, 80, 78]
    # Lower lip landmarks  
    LOWER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    # Inner mouth landmarks for more accurate MAR calculation
    INNER_MOUTH_INDICES = [13, 14, 269, 270, 267, 271, 272, 78, 80, 81, 82, 87, 88, 89, 90, 91, 95, 96, 178, 179, 180, 181, 185, 191, 308, 310, 311, 312, 317, 318, 319, 320, 321, 324, 325, 402, 403, 404, 405, 415]
    
    def _initialize(self):
        """Initialize MediaPipe components"""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is not installed. Please install with: pip install mediapipe opencv-python")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face mesh with optimized settings
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enables iris landmarks and better accuracy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MAR threshold based on sensitivity
        self.mar_threshold = 0.15 * (2.0 - self.sensitivity)  # Higher sensitivity = lower threshold
        
        logger.info(f"MediaPipe provider initialized with MAR threshold: {self.mar_threshold:.3f}")
    
    def analyze_video(self, video_input, preview_mode: bool = False) -> TimingData:
        """
        Analyze video using MediaPipe Face Mesh
        """
        # Handle ComfyUI video input format
        logger.debug(f"Video input type: {type(video_input)}")
        logger.debug(f"Video input attributes: {dir(video_input)}")
        
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
            raise ValueError(f"Cannot extract file path from video input of type {type(video_input)}. Available attributes: {dir(video_input)}")
        
        logger.info(f"Analyzing video with MediaPipe: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames, {total_duration:.2f}s")
        
        # Process video frame by frame
        movement_frames = []
        confidence_scores = []
        mar_values = []
        preview_frames = [] if preview_mode else None
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect movement in frame
            is_moving, confidence, landmarks = self.detect_movement(frame)
            
            movement_frames.append(is_moving)
            confidence_scores.append(confidence)
            
            # Calculate MAR if landmarks detected
            if landmarks is not None:
                mar = self.calculate_mar(landmarks)
                mar_values.append(mar)
            else:
                mar_values.append(0.0)
            
            # Generate preview frame if requested
            if preview_mode:
                annotated = self.annotate_frame(frame, landmarks, is_moving, confidence)
                preview_frames.append(annotated)
            
            frame_count += 1
            
            # Progress logging
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                logger.debug(f"Processing: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        cap.release()
        
        # Smooth confidence scores to reduce noise
        confidence_scores = self.smooth_confidence_scores(confidence_scores)
        
        # Convert frame-by-frame detection to segments
        segments = self.frames_to_segments(movement_frames, confidence_scores, mar_values, fps)
        
        # Apply filtering
        filtered_segments = self.filter_segments(segments)
        
        # Create preview video if requested
        if preview_mode and preview_frames:
            self._create_preview_video(preview_frames, fps, width, height)
        
        # Create timing data
        timing_data = TimingData(
            segments=filtered_segments,
            fps=fps,
            total_frames=total_frames,
            total_duration=total_duration,
            provider=self.provider_name,
            metadata={
                "mar_threshold": self.mar_threshold,
                "sensitivity": self.sensitivity,
                "video_resolution": f"{width}x{height}",
                "total_segments_before_filter": len(segments),
                "total_segments_after_filter": len(filtered_segments)
            }
        )
        
        logger.info(f"Analysis complete: {len(filtered_segments)} segments detected (filtered from {len(segments)})")
        
        return timing_data
    
    def detect_movement(self, frame: np.ndarray) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Detect mouth movement in a single frame using MediaPipe
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return False, 0.0, None
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array
        h, w = frame.shape[:2]
        landmarks = np.array([
            [landmark.x * w, landmark.y * h, landmark.z * w]
            for landmark in face_landmarks.landmark
        ])
        
        # Calculate MAR
        mar = self.calculate_mar(landmarks)
        
        # Determine if mouth is moving based on MAR threshold
        is_moving = mar > self.mar_threshold
        
        # Calculate confidence based on how far MAR is from threshold
        if is_moving:
            confidence = min(1.0, (mar - self.mar_threshold) / self.mar_threshold)
        else:
            confidence = max(0.0, 1.0 - (self.mar_threshold - mar) / self.mar_threshold)
        
        return is_moving, confidence, landmarks
    
    def calculate_mar(self, landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) from MediaPipe landmarks
        
        MAR = (|p2-p8| + |p3-p7| + |p4-p6|) / (3 * |p1-p5|)
        where p1-p8 are specific mouth landmarks
        """
        if landmarks is None or len(landmarks) < 468:
            return 0.0
        
        try:
            # Get mouth corner landmarks (left and right)
            left_corner = landmarks[61][:2]  # Left mouth corner
            right_corner = landmarks[291][:2]  # Right mouth corner
            
            # Get upper and lower lip landmarks
            upper_lip_top = landmarks[13][:2]  # Top of upper lip
            lower_lip_bottom = landmarks[14][:2]  # Bottom of lower lip
            
            # Additional vertical measurements for better accuracy
            upper_inner = landmarks[82][:2]  # Inner upper lip
            lower_inner = landmarks[87][:2]  # Inner lower lip
            
            # Calculate horizontal distance (mouth width)
            horizontal_dist = np.linalg.norm(right_corner - left_corner)
            
            # Calculate vertical distances
            vertical_dist1 = np.linalg.norm(upper_lip_top - lower_lip_bottom)
            vertical_dist2 = np.linalg.norm(upper_inner - lower_inner)
            
            # Average vertical distance
            vertical_dist = (vertical_dist1 + vertical_dist2) / 2
            
            # Calculate MAR
            if horizontal_dist > 0:
                mar = vertical_dist / horizontal_dist
            else:
                mar = 0.0
            
            return mar
            
        except (IndexError, TypeError) as e:
            logger.warning(f"Error calculating MAR: {e}")
            return 0.0
    
    def _create_preview_video(self, frames: List[np.ndarray], fps: float, width: int, height: int):
        """Create preview video with movement annotations"""
        if not frames:
            return
        
        # Create temporary output path
        import tempfile
        output_path = tempfile.mktemp(suffix='.mp4')
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        # Store path for retrieval
        self.preview_video = output_path
        logger.info(f"Preview video created: {output_path}")
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray],
        is_moving: bool,
        confidence: float
    ) -> np.ndarray:
        """
        Enhanced frame annotation with MediaPipe landmarks
        """
        annotated = frame.copy()
        
        # Add movement indicator
        color = (0, 255, 0) if is_moving else (255, 0, 0)
        status_text = f"SPEAKING: {confidence:.2f}" if is_moving else f"SILENT: {confidence:.2f}"
        
        # Add background rectangle for better text visibility
        cv2.rectangle(annotated, (5, 5), (250, 35), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            status_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        # Draw mouth landmarks if available
        if landmarks is not None and len(landmarks) >= 468:
            # Draw mouth contour
            mouth_indices = self.INNER_MOUTH_INDICES
            
            for i in mouth_indices:
                if i < len(landmarks):
                    x, y = int(landmarks[i][0]), int(landmarks[i][1])
                    # Color based on movement
                    point_color = (0, 255, 0) if is_moving else (0, 0, 255)
                    cv2.circle(annotated, (x, y), 2, point_color, -1)
            
            # Draw MAR value
            mar = self.calculate_mar(landmarks)
            mar_text = f"MAR: {mar:.3f}"
            cv2.putText(
                annotated,
                mar_text,
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated
    
    @property
    def provider_name(self) -> str:
        """Return the name of this provider"""
        return "MediaPipe"
    
    def _check_dependencies(self):
        """Check if required dependencies are installed"""
        import mediapipe
        import cv2