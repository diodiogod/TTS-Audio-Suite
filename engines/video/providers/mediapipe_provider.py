"""
MediaPipe provider for mouth movement analysis
High-performance facial landmark detection with 468 3D face landmarks
"""

import logging
import os
import sys
from typing import Optional, Tuple, List, Any, Dict
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
VisemeFrame = abstract_module.VisemeFrame

# Import modular viseme analysis system
try:
    # Add analysis path to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_path = os.path.join(current_dir, "..", "analysis")
    if analysis_path not in sys.path:
        sys.path.insert(0, analysis_path)
    
    from viseme_analysis_factory import VisemeAnalysisFactory
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    try:
        # Alternative: Try from project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        analysis_path = os.path.join(project_root, "engines", "video", "analysis")
        if analysis_path not in sys.path:
            sys.path.insert(0, analysis_path)
        
        from viseme_analysis_factory import VisemeAnalysisFactory
        ANALYSIS_AVAILABLE = True
    except ImportError:
        print(f"Warning: Modular analysis system not available: {e}")
        ANALYSIS_AVAILABLE = False

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
        import sys
        
        if not MEDIAPIPE_AVAILABLE:
            if sys.version_info >= (3, 13):
                raise ImportError(
                    "MediaPipe is not compatible with Python 3.13. "
                    "OpenSeeFace is available as an experimental alternative, "
                    "but results may be less accurate.\n\n"
                    "📢 Help request Python 3.13 support! Vote/comment on:\n"
                    "• https://github.com/google-ai-edge/mediapipe/issues/5708\n"
                    "• https://github.com/google-ai-edge/mediapipe/issues/6025\n"
                    "The more users request it, the higher priority it gets!"
                )
            else:
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
        
        # Enhanced MAR threshold with exponential scaling for ultra-fine control
        # Exponential mapping for much more granular sensitivity control:
        # At sensitivity 0.1: threshold ≈ 0.25 (very conservative, only obvious movements)
        # At sensitivity 0.3: threshold ≈ 0.12 (moderate, good for clear speech)
        # At sensitivity 0.5: threshold ≈ 0.08 (balanced, catches most speech)
        # At sensitivity 0.7: threshold ≈ 0.05 (sensitive, includes subtle movements)
        # At sensitivity 0.9: threshold ≈ 0.025 (very sensitive, catches whispers)
        # At sensitivity 1.0: threshold ≈ 0.015 (ultra-sensitive, may include micro-movements)
        
        # Use exponential decay for ultra-fine control at high sensitivity
        normalized_sensitivity = max(0.0, min(1.0, self.sensitivity))  # Clamp to [0,1]
        
        # Exponential formula: threshold = base_max * exp(-sensitivity_factor * sensitivity)
        # This gives much finer control at higher sensitivities
        import math
        base_max = 0.25  # Maximum threshold at sensitivity 0
        sensitivity_factor = 4.0  # Controls how steep the exponential curve is
        
        self.mar_threshold = base_max * math.exp(-sensitivity_factor * normalized_sensitivity)
        
        # Ensure minimum threshold to prevent unrealistic detection
        self.mar_threshold = max(0.01, self.mar_threshold)
        
        # Initialize viseme detection mode
        self.enable_viseme_detection = False  # Will be enabled via node parameters
        
        logger.info(f"MediaPipe provider initialized with MAR threshold: {self.mar_threshold:.3f}")
    
    def analyze_video(self, video_input, preview_mode: bool = False, enable_viseme: bool = False, viseme_options: Dict[str, Any] = None) -> TimingData:
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
        
        # Store viseme options for modular analysis - ensure defaults are set
        self.viseme_options = viseme_options or {}
        
        # Ensure critical options have defaults for modular analysis
        if 'enable_consonant_detection' not in self.viseme_options:
            self.viseme_options['enable_consonant_detection'] = False
        if 'enable_temporal_analysis' not in self.viseme_options:
            self.viseme_options['enable_temporal_analysis'] = False
        if 'viseme_sensitivity' not in self.viseme_options:
            self.viseme_options['viseme_sensitivity'] = 1.0
        if 'viseme_confidence_threshold' not in self.viseme_options:
            self.viseme_options['viseme_confidence_threshold'] = 0.04
        
        # Extract consonant detection setting from options
        self.enable_consonant_detection = self.viseme_options.get('enable_consonant_detection', False)
        
        print(f"[DEBUG] MediaPipe viseme_options: {self.viseme_options}")
        print(f"[DEBUG] ANALYSIS_AVAILABLE: {ANALYSIS_AVAILABLE}")
        
        logger.info(f"Analyzing video with MediaPipe: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get original video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_duration = total_frames / fps if fps > 0 else 0
        
        # Calculate optimal processing dimensions for MediaPipe
        # Target: 720p max for analysis, 540p for preview (good balance of speed vs quality)
        max_dimension = 720
        preview_max_dimension = 540 if preview_mode else 720  # Good balance for preview
        # Use smaller dimensions for preview mode to speed up processing
        target_dimension = preview_max_dimension if preview_mode else max_dimension
        
        if max(original_width, original_height) > target_dimension:
            if original_width > original_height:
                width = target_dimension
                height = int((original_height / original_width) * target_dimension)
            else:
                height = target_dimension
                width = int((original_width / original_height) * target_dimension)
            
            # Ensure dimensions are even (required for some video codecs)
            width = width - (width % 2)
            height = height - (height % 2)
            
            if preview_mode:
                logger.info(f"Resizing video from {original_width}x{original_height} to {width}x{height} for fast preview generation")
            else:
                logger.info(f"Resizing video from {original_width}x{original_height} to {width}x{height} for optimal MediaPipe performance")
        else:
            width, height = original_width, original_height
            logger.info(f"Video resolution {width}x{height} is optimal, no resizing needed")
        
        logger.info(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames, {total_duration:.2f}s")
        
        # Process video frame by frame
        movement_frames = []
        confidence_scores = []
        mar_values = []
        viseme_frames = [] if enable_viseme else None
        # Store frame data for preview generation
        frame_data = [] if preview_mode else None
        
        # Store scaling factors for later coordinate conversion
        scale_x = original_width / width
        scale_y = original_height / height
        needs_scaling = scale_x != 1.0 or scale_y != 1.0
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for optimal MediaPipe processing
            if needs_scaling:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            
            # Detect movement in frame
            is_moving, confidence, landmarks = self.detect_movement(frame)
            
            movement_frames.append(is_moving)
            confidence_scores.append(confidence)
            
            # Calculate MAR and extract features if landmarks detected
            if landmarks is not None:
                mar = self.calculate_mar(landmarks)
                mar_values.append(mar)
                
                # Extract viseme features if enabled
                if enable_viseme:
                    features = self.extract_geometric_features(landmarks)
                    
                    # Use modular analysis system if available
                    metadata = None
                    if ANALYSIS_AVAILABLE and self.viseme_options:
                        # Create analyzer only once per video, not per frame!
                        if not hasattr(self, '_temporal_analyzer'):
                            self._temporal_analyzer = VisemeAnalysisFactory.create_analyzer(self.viseme_options)
                            print(f"[DEBUG] Created persistent analyzer: {type(self._temporal_analyzer).__name__}")
                        
                        result = self._temporal_analyzer.classify_viseme(features, self.enable_consonant_detection)
                        viseme, viseme_conf = result.viseme, result.confidence
                        metadata = result.metadata  # Capture classifier metadata
                    else:
                        # Fallback to built-in method
                        print(f"[DEBUG] Using built-in method: ANALYSIS_AVAILABLE={ANALYSIS_AVAILABLE}, viseme_options={bool(self.viseme_options)}")
                        viseme, viseme_conf = self.classify_viseme(features, self.enable_consonant_detection)
                    
                    viseme_frames.append(VisemeFrame(
                        frame_index=frame_count,
                        viseme=viseme,
                        confidence=viseme_conf,
                        geometric_features=features,
                        metadata=metadata
                    ))
            else:
                mar_values.append(0.0)
                if enable_viseme:
                    viseme_frames.append(VisemeFrame(
                        frame_index=frame_count,
                        viseme='neutral',
                        confidence=0.0,
                        geometric_features={},
                        metadata=None
                    ))
            
            # Store frame data for preview generation if requested
            if preview_mode:
                # Get current viseme and geometric features if available
                current_viseme = None
                viseme_confidence = 0.0
                geometric_features = None
                consonant_scores = None
                analyzer_method = 'basic'  # Default analyzer method
                if enable_viseme and viseme_frames and frame_count < len(viseme_frames):
                    current_viseme = viseme_frames[-1].viseme  # Last added viseme
                    viseme_confidence = viseme_frames[-1].confidence
                    geometric_features = viseme_frames[-1].geometric_features
                    # Extract debug info from metadata
                    if hasattr(viseme_frames[-1], 'metadata') and viseme_frames[-1].metadata:
                        metadata = viseme_frames[-1].metadata
                        consonant_scores = metadata.get('raw_scores', {})
                        # Filter for consonant scores only
                        consonant_scores = {k: v for k, v in consonant_scores.items()
                                          if k in ['B', 'P', 'M', 'F', 'V', 'TH', 'T', 'D', 'N', 'K', 'G'] and v > 0.0}

                        # Extract analyzer method info
                        analyzer_method = metadata.get('method', 'unknown')
                    else:
                        analyzer_method = 'basic'
                
                # Store frame data for later preview generation
                frame_data.append({
                    'frame': frame.copy(),  # Store copy of frame
                    'landmarks': landmarks,
                    'is_moving': is_moving,
                    'confidence': confidence,
                    'frame_number': frame_count,
                    'current_viseme': current_viseme,
                    'viseme_confidence': viseme_confidence,
                    'geometric_features': geometric_features,
                    'consonant_scores': consonant_scores,
                    'analyzer_method': analyzer_method
                })
            
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
        
        # Store frame data for preview generation (done separately)
        if preview_mode and frame_data:
            self.frame_data = frame_data
        else:
            self.frame_data = None
        
        # Add viseme sequences to segments if enabled
        if enable_viseme and viseme_frames:
            self._add_viseme_sequences_to_segments(filtered_segments, viseme_frames, fps)
        
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
                "video_width": width,
                "video_height": height,
                "total_segments_before_filter": len(segments),
                "total_segments_after_filter": len(filtered_segments),
                "viseme_detection_enabled": enable_viseme,
                "unfiltered_segments": segments,  # Store unfiltered segments for re-filtering
                "frame_data": frame_data if preview_mode else None  # Store frame data for preview generation
            },
            viseme_frames=viseme_frames
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
    
    def extract_geometric_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive geometric features for viseme detection
        
        Returns:
            Dictionary of geometric features including:
            - lip_width: Horizontal distance between mouth corners
            - lip_height: Vertical distance between upper and lower lips
            - mouth_area: Approximate area of mouth opening
            - lip_ratio: Width/height ratio for shape classification
            - upper_lip_curvature: Curvature of upper lip
            - lower_lip_curvature: Curvature of lower lip
            - mouth_roundedness: How circular the mouth shape is
            - lip_protrusion: 3D depth information from MediaPipe
        """
        if landmarks is None or len(landmarks) < 468:
            return {}
        
        try:
            # Key landmark points for mouth geometry
            left_corner = landmarks[61][:2]  # Left mouth corner
            right_corner = landmarks[291][:2]  # Right mouth corner
            upper_lip_top = landmarks[13][:2]  # Top of upper lip
            lower_lip_bottom = landmarks[14][:2]  # Bottom of lower lip
            upper_inner = landmarks[82][:2]  # Inner upper lip
            lower_inner = landmarks[87][:2]  # Inner lower lip
            
            # Additional points for detailed analysis
            upper_lip_left = landmarks[84][:2]  # Upper lip left
            upper_lip_right = landmarks[314][:2]  # Upper lip right
            lower_lip_left = landmarks[91][:2]  # Lower lip left
            lower_lip_right = landmarks[321][:2]  # Lower lip right
            
            # Calculate basic measurements
            lip_width = np.linalg.norm(right_corner - left_corner)
            lip_height_outer = np.linalg.norm(upper_lip_top - lower_lip_bottom)
            lip_height_inner = np.linalg.norm(upper_inner - lower_inner)
            lip_height = (lip_height_outer + lip_height_inner) / 2
            
            # Aspect ratio for shape classification
            lip_ratio = lip_width / lip_height if lip_height > 0 else 0
            
            # Approximate mouth area (simplified)
            mouth_area = lip_width * lip_height * 0.7  # Elliptical approximation
            
            # Calculate lip curvature (simplified)
            upper_lip_center = (upper_lip_left + upper_lip_top + upper_lip_right) / 3
            lower_lip_center = (lower_lip_left + lower_lip_bottom + lower_lip_right) / 3
            
            upper_lip_curvature = np.linalg.norm(upper_lip_top - upper_lip_center)
            lower_lip_curvature = np.linalg.norm(lower_lip_bottom - lower_lip_center)
            
            # Calculate roundedness using a robust geometric approach
            # MediaPipe mouth landmarks: outer lip contour (lips) + inner lip area
            mouth_center = np.array([(left_corner[0] + right_corner[0]) / 2, 
                                   (upper_lip_top[1] + lower_lip_bottom[1]) / 2])
            
            # Use correct MediaPipe mouth landmark indices for outer lip contour
            # These form a closed loop around the mouth perimeter
            outer_lip_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
            
            # Extract outer lip contour points
            mouth_contour = []
            for idx in outer_lip_indices:
                if idx < len(landmarks):
                    mouth_contour.append(landmarks[idx][:2])
            
            if len(mouth_contour) >= 6:  # Need enough points for meaningful analysis
                # Method 1: Circularity via perimeter-to-area ratio
                # Calculate contour area and perimeter
                contour_area = 0.0
                perimeter = 0.0
                
                for i in range(len(mouth_contour)):
                    j = (i + 1) % len(mouth_contour)
                    # Shoelace formula for polygon area
                    contour_area += mouth_contour[i][0] * mouth_contour[j][1]
                    contour_area -= mouth_contour[j][0] * mouth_contour[i][1]
                    # Perimeter calculation
                    perimeter += np.linalg.norm(np.array(mouth_contour[j]) - np.array(mouth_contour[i]))
                
                contour_area = abs(contour_area) / 2.0
                
                if perimeter > 0 and contour_area > 0:
                    # Circularity: 4π * area / perimeter²
                    # Perfect circle = 1.0, elongated shapes < 1.0
                    circularity = (4 * np.pi * contour_area) / (perimeter ** 2)
                    roundedness = min(1.0, circularity)  # Clamp to max 1.0
                else:
                    # Fallback: Method 2: Distance variance from center
                    distances = [np.linalg.norm(np.array(point) - mouth_center) for point in mouth_contour]
                    if distances:
                        mean_dist = np.mean(distances)
                        std_dist = np.std(distances)
                        # Low coefficient of variation = high roundedness
                        cv = std_dist / mean_dist if mean_dist > 0 else 1.0
                        roundedness = max(0.0, 1.0 - cv * 2.0)  # Less aggressive scaling
                    else:
                        roundedness = 0.0
            else:
                # Fallback: Simple aspect ratio based roundedness
                if lip_height > 0:
                    aspect_ratio = lip_width / lip_height
                    # Circular shape has aspect ratio around 1.0-1.5
                    if 0.8 <= aspect_ratio <= 2.0:
                        roundedness = 1.0 - abs(aspect_ratio - 1.2) / 2.0  # Peak at 1.2 ratio
                        roundedness = max(0.0, min(1.0, roundedness))
                    else:
                        roundedness = 0.0
                else:
                    roundedness = 0.0
            
            # Extract 3D depth information if available
            lip_protrusion = 0.0
            if landmarks.shape[1] >= 3:
                # Average Z-coordinate of lip landmarks
                lip_z_coords = [
                    landmarks[61][2], landmarks[291][2],
                    landmarks[13][2], landmarks[14][2],
                    landmarks[82][2], landmarks[87][2]
                ]
                lip_protrusion = np.mean(lip_z_coords)
            
            # Calculate MAR for backward compatibility
            mar = lip_height / lip_width if lip_width > 0 else 0
            
            # Enhanced features for consonant detection
            # Lip contact detection (for B, P, M)
            lip_contact = 1.0 - (lip_height / lip_width) if lip_width > 0 else 1.0
            
            # Teeth visibility (for F, V, TH) - using inner lip landmarks
            teeth_gap = np.linalg.norm(landmarks[13][:2] - landmarks[14][:2])  # Inner gap
            teeth_visibility = teeth_gap / lip_width if lip_width > 0 else 0
            
            # Lip compression (for stops: P, B, T, D, K, G)
            lip_compression = (upper_lip_curvature + lower_lip_curvature) / lip_width if lip_width > 0 else 0
            
            # Nose flare detection (for nasals: M, N, NG) - using nose landmarks
            left_nostril = landmarks[31][:2] if len(landmarks) > 31 else left_corner
            right_nostril = landmarks[35][:2] if len(landmarks) > 35 else right_corner
            nose_width = np.linalg.norm(right_nostril - left_nostril)
            nose_flare = nose_width / lip_width if lip_width > 0 else 0
            
            # Tongue visibility (for TH, L, R) - approximate using inner mouth
            tongue_space = np.linalg.norm(landmarks[17][:2] - landmarks[18][:2]) if len(landmarks) > 18 else 0
            tongue_visibility = tongue_space / lip_width if lip_width > 0 else 0
            
            return {
                'lip_width': lip_width,
                'lip_height': lip_height,
                'mouth_area': mouth_area,
                'lip_ratio': lip_ratio,
                'upper_lip_curvature': upper_lip_curvature,
                'lower_lip_curvature': lower_lip_curvature,
                'roundedness': roundedness,
                'lip_protrusion': lip_protrusion,
                'mar': mar,
                # New consonant features
                'lip_contact': lip_contact,
                'teeth_visibility': teeth_visibility,
                'lip_compression': lip_compression,
                'nose_flare': nose_flare,
                'tongue_visibility': tongue_visibility
            }
            
        except (IndexError, TypeError) as e:
            logger.warning(f"Error extracting geometric features: {e}")
            return {}
    
    def classify_viseme(self, features: Dict[str, float], enable_consonants: bool = False) -> Tuple[str, float]:
        """
        Classify mouth shape into viseme categories based on geometric features
        
        Args:
            features: Dictionary of geometric features
            enable_consonants: Whether to detect consonants in addition to vowels
            
        Returns:
            Tuple of (viseme_label, confidence)
            Vowels: 'A', 'E', 'I', 'O', 'U', 'neutral'
            Consonants (if enabled): 'B', 'P', 'M', 'F', 'V', 'TH', etc.
        """
        if not features:
            return 'neutral', 0.0
        
        lip_ratio = features.get('lip_ratio', 0)
        roundedness = features.get('roundedness', 0)
        mouth_area = features.get('mouth_area', 0)
        mar = features.get('mar', 0)
        
        # Normalize mouth area (rough approximation)
        normalized_area = min(1.0, mouth_area / 1000.0)  # Adjust scale as needed
        
        # Initialize scoring for vowels and optionally consonants
        viseme_scores = {
            'A': 0.0,
            'E': 0.0,
            'I': 0.0,
            'O': 0.0,
            'U': 0.0,
            'neutral': 0.0
        }
        
        # Add consonant scores if enabled
        if enable_consonants:
            consonant_scores = {
                'B': 0.0, 'P': 0.0, 'M': 0.0,  # Bilabial stops/nasals
                'F': 0.0, 'V': 0.0,             # Labiodental fricatives
                'TH': 0.0,                      # Dental fricatives
                'T': 0.0, 'D': 0.0, 'N': 0.0,  # Alveolar stops/nasals
                'K': 0.0, 'G': 0.0              # Velar stops
            }
            viseme_scores.update(consonant_scores)
        
        # Get consonant features for classification
        lip_contact = features.get('lip_contact', 0)
        teeth_visibility = features.get('teeth_visibility', 0)
        lip_compression = features.get('lip_compression', 0)
        nose_flare = features.get('nose_flare', 0)
        
        # Check if mouth is open enough for vowel (complete closure = neutral)
        if mar < self.mar_threshold * 0.5:  # Half threshold for complete closure
            return 'neutral', 0.8
        
        # Vowel classification first (PRIMARY DETECTION)
        # Vowels are 60-80% of speech - they should dominate classification
        
        # Apply viseme sensitivity scaling to thresholds
        sens_factor = self.viseme_sensitivity
        
        # A: Wide open mouth, high aperture
        if mar > (0.25 / sens_factor) and lip_ratio < (3.5 * sens_factor):
            viseme_scores['A'] = (mar / 0.5) * (1.0 + normalized_area) * sens_factor
        
        # E: Spread lips, moderate opening
        if (0.1 / sens_factor) < mar < (0.3 / sens_factor) and lip_ratio > (2.5 / sens_factor):
            viseme_scores['E'] = (1.0 - abs(mar - 0.2) * 5.0) * min(1.0, lip_ratio / 3.5) * sens_factor
        
        # I: Narrow vertical, wide horizontal (smile-like)
        if mar < (0.15 / sens_factor) and lip_ratio > (3.0 / sens_factor):
            viseme_scores['I'] = (1.0 - mar * 6.0) * min(1.0, lip_ratio / 4.0) * sens_factor
        
        # O: Rounded, moderate opening
        if (0.15 / sens_factor) < mar < (0.35 / sens_factor) and roundedness > (0.5 / sens_factor):
            viseme_scores['O'] = roundedness * (1.0 - abs(mar - 0.25) * 4.0) * sens_factor
        
        # U: Small rounded opening
        if mar < (0.2 / sens_factor) and roundedness > (0.6 / sens_factor):
            viseme_scores['U'] = roundedness * (1.0 - mar * 5.0) * sens_factor
        
        # Find best vowel match first
        best_vowel = max([(k, v) for k, v in viseme_scores.items() if k in ['A', 'E', 'I', 'O', 'U', 'neutral']], 
                        key=lambda x: x[1])
        
        # Consonant classification (SECONDARY - compete with vowels fairly)
        # Consonants are RARE (15-25% total) and BRIEF interruptions of vowel flow
        if enable_consonants:  # Always check consonants, let them compete fairly
            
            # MODERATELY RAISED THRESHOLDS (balanced for realistic detection)
            # B sounds occur ~2-5% of speech - rare but not impossible
            
            # Bilabial stops and nasals (lips clearly closed but achievable)
            if lip_contact > (0.92 / sens_factor):  # Was 0.98, now 0.92 - still restrictive but achievable
                # Distinguish between B/P/M based on other features
                if nose_flare > (0.6 / sens_factor):  # Nasal detection
                    viseme_scores['M'] = lip_contact * nose_flare * sens_factor
                elif lip_compression > (0.75 / sens_factor):  # Voiceless stop
                    viseme_scores['P'] = lip_contact * lip_compression * sens_factor
                else:  # Voiced stop
                    viseme_scores['B'] = lip_contact * sens_factor
            
            # Labiodental fricatives (teeth on lip)
            if teeth_visibility > (0.75 / sens_factor) and lip_contact > (0.6 / sens_factor):
                if lip_compression > (0.6 / sens_factor):
                    viseme_scores['F'] = teeth_visibility * lip_compression * sens_factor
                else:
                    viseme_scores['V'] = teeth_visibility * lip_contact * sens_factor
            
            # Dental fricatives (tongue visible between teeth)
            if teeth_visibility > (0.8 / sens_factor) and mar > (0.15 / sens_factor):
                viseme_scores['TH'] = teeth_visibility * mar * sens_factor
            
            # Alveolar and velar stops (compression patterns)
            if lip_compression > (0.8 / sens_factor) and lip_contact < (0.4 / sens_factor):
                if mar < (0.08 / sens_factor):
                    viseme_scores['T'] = lip_compression * (1.0 - mar) * sens_factor
                elif nose_flare > (0.5 / sens_factor):
                    viseme_scores['N'] = lip_compression * nose_flare * sens_factor
                else:
                    viseme_scores['D'] = lip_compression * mar * sens_factor
            
            # Velar stops (back of tongue - harder to detect)
            if lip_compression > (0.7 / sens_factor) and roundedness < (0.2 / sens_factor):
                if mar < (0.05 / sens_factor):
                    viseme_scores['K'] = lip_compression * (1.0 - roundedness) * sens_factor * 0.8
                else:
                    viseme_scores['G'] = lip_compression * (1.0 - roundedness) * mar * sens_factor * 0.8
        
        # Find best match
        best_viseme = max(viseme_scores.items(), key=lambda x: x[1])
        
        # Normalize confidence to 0-1 range
        confidence = min(1.0, best_viseme[1] / 2.0)
        
        # Apply viseme confidence threshold
        if confidence < self.viseme_confidence_threshold:
            return 'neutral', confidence
        
        # Apply temporal smoothing if enabled
        if self.viseme_smoothing > 0.0 and hasattr(self, 'previous_visemes'):
            # Add current detection to history
            self.previous_visemes.append((best_viseme[0], confidence))
            
            # Keep only recent history for smoothing
            history_length = max(1, int(5 * self.viseme_smoothing))  # 1-5 frames
            self.previous_visemes = self.previous_visemes[-history_length:]
            
            # Calculate weighted average (more recent = higher weight)
            viseme_weights = {}
            total_weight = 0.0
            
            for i, (v, c) in enumerate(self.previous_visemes):
                weight = (i + 1) * c  # Recent frames + confidence weighting
                viseme_weights[v] = viseme_weights.get(v, 0.0) + weight
                total_weight += weight
            
            if total_weight > 0:
                # Find most weighted viseme
                smoothed_viseme = max(viseme_weights.items(), key=lambda x: x[1])
                smoothed_confidence = min(1.0, smoothed_viseme[1] / total_weight)
                return smoothed_viseme[0], smoothed_confidence
        
        return best_viseme[0], confidence
    
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
    
    def _add_viseme_sequences_to_segments(
        self,
        segments: List[MovementSegment],
        viseme_frames: List[VisemeFrame],
        fps: float
    ):
        """
        Add viseme sequences to movement segments with temporal filtering
        
        Args:
            segments: List of movement segments
            viseme_frames: List of viseme detections per frame
            fps: Video framerate
        """
        for segment in segments:
            # Get visemes for this segment's frame range
            raw_visemes = []
            raw_confidences = []
            
            for frame_idx in range(segment.start_frame, segment.end_frame + 1):
                if frame_idx < len(viseme_frames):
                    vf = viseme_frames[frame_idx]
                    # Include neutral visemes as underscores for better timing information
                    if vf.viseme == 'neutral':
                        raw_visemes.append('_')
                        raw_confidences.append(vf.confidence)
                    else:
                        raw_visemes.append(vf.viseme)
                        raw_confidences.append(vf.confidence)
            
            # Apply temporal filtering for consonants (CRITICAL FIX)
            if raw_visemes:
                filtered_visemes, filtered_confidences = self._apply_temporal_filtering(
                    raw_visemes, raw_confidences, fps
                )
                segment.viseme_sequence = ''.join(filtered_visemes)
                segment.viseme_confidences = filtered_confidences
            else:
                segment.viseme_sequence = ''
                segment.viseme_confidences = []
    
    def _apply_temporal_filtering(self, visemes: List[str], confidences: List[float], fps: float) -> Tuple[List[str], List[float]]:
        """
        Apply temporal filtering to remove unrealistic consonant patterns
        
        Consonants should be BRIEF (1-3 frames max) and RARE in natural speech
        
        Args:
            visemes: List of detected visemes
            confidences: List of confidence scores
            fps: Video framerate
            
        Returns:
            Tuple of (filtered_visemes, filtered_confidences)
        """
        if len(visemes) < 2:
            return visemes, confidences
            
        consonants = {'B', 'P', 'M', 'F', 'V', 'TH', 'T', 'D', 'N', 'K', 'G'}
        vowels = {'A', 'E', 'I', 'O', 'U'}
        
        filtered_visemes = visemes[:]
        filtered_confidences = confidences[:]
        
        # 1. TEMPORAL FILTERING: Remove sustained consonants (>3 frames)
        i = 0
        while i < len(filtered_visemes):
            if filtered_visemes[i] in consonants:
                # Count consecutive frames of this consonant
                consonant_start = i
                current_consonant = filtered_visemes[i]
                
                # Find end of consonant sequence
                while (i < len(filtered_visemes) and 
                       filtered_visemes[i] == current_consonant):
                    i += 1
                
                consonant_length = i - consonant_start
                
                # CRITICAL: Consonants longer than 3 frames are impossible
                # Convert them to the most appropriate vowel or neutral
                if consonant_length > 3:
                    # Find best replacement (prefer previous/next vowel or neutral)
                    replacement = '_'  # Default to neutral
                    
                    # Look for neighboring vowels to use as replacement
                    if consonant_start > 0 and filtered_visemes[consonant_start - 1] in vowels:
                        replacement = filtered_visemes[consonant_start - 1]
                    elif i < len(filtered_visemes) and filtered_visemes[i] in vowels:
                        replacement = filtered_visemes[i]
                    
                    # Replace the sustained consonant with appropriate vowel/neutral
                    for j in range(consonant_start, i):
                        filtered_visemes[j] = replacement
                        # Reduce confidence since this was a correction
                        filtered_confidences[j] *= 0.3
            else:
                i += 1
        
        # 2. FREQUENCY FILTERING: Enforce natural consonant frequency limits
        total_frames = len(filtered_visemes)
        consonant_count = sum(1 for v in filtered_visemes if v in consonants)
        consonant_frequency = consonant_count / total_frames if total_frames > 0 else 0
        
        # If consonant frequency > 25%, demote lowest confidence consonants
        if consonant_frequency > 0.25:
            # Get consonant indices with their confidence scores
            consonant_indices = [(i, filtered_confidences[i]) for i, v in enumerate(filtered_visemes) if v in consonants]
            
            # Sort by confidence (lowest first)
            consonant_indices.sort(key=lambda x: x[1])
            
            # Calculate how many to convert
            target_consonants = int(total_frames * 0.25)  # Max 25%
            excess_count = len(consonant_indices) - target_consonants
            
            # Convert lowest confidence consonants to neutral
            for i in range(excess_count):
                idx = consonant_indices[i][0]
                filtered_visemes[idx] = '_'
                filtered_confidences[idx] *= 0.2  # Very low confidence
        
        # 3. SPECIFIC B-FREQUENCY FILTER: B should be ~0.5% of frames, not 70%!
        b_count = sum(1 for v in filtered_visemes if v == 'B')
        b_frequency = b_count / total_frames if total_frames > 0 else 0
        
        if b_frequency > 0.05:  # Max 5% B sounds
            # Get B indices with confidence scores
            b_indices = [(i, filtered_confidences[i]) for i, v in enumerate(filtered_visemes) if v == 'B']
            b_indices.sort(key=lambda x: x[1])  # Lowest confidence first
            
            # Keep only highest confidence B's (max 5% of segment)
            max_b_count = max(1, int(total_frames * 0.05))
            excess_b_count = len(b_indices) - max_b_count
            
            for i in range(excess_b_count):
                idx = b_indices[i][0]
                filtered_visemes[idx] = '_'  # Convert to neutral
                filtered_confidences[idx] *= 0.1
        
        return filtered_visemes, filtered_confidences
    
    
    def get_preview_video(self) -> Optional[str]:
        """Get the preview video file path if generated"""
        return getattr(self, 'preview_video', None)
    
    def _create_preview_video(self, frames: List[np.ndarray], fps: float, width: int, height: int):
        """Create preview video with movement annotations - create both MP4 and WebP for compatibility"""
        if not frames:
            return
        
        # Create output path in ComfyUI output directory like Save Video does
        import folder_paths
        import os
        
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename
        import time
        timestamp = int(time.time())
        filename_mp4 = f"mouth_preview_{timestamp}.mp4"
        filename_webp = f"mouth_preview_{timestamp}.webp"
        output_path_mp4 = os.path.join(output_dir, filename_mp4)
        output_path_webp = os.path.join(output_dir, filename_webp)
        
        # Create MP4 video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path_mp4, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Failed to open video writer for {output_path_mp4}")
            return
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        # Create WEBM video for native ComfyUI display (like SaveWEBM - better performance)
        try:
            import av
            from fractions import Fraction
            
            # Convert BGR frames to RGB
            rgb_frames = []
            for frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frames.append(rgb_frame)
            
            # Create WEBM using av library like SaveWEBM does - optimized for fast preview
            container = av.open(output_path_webp.replace('.webp', '.webm'), mode="w")
            stream = container.add_stream("libvpx-vp9", rate=Fraction(round(fps * 1000), 1000))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            # Fast encoding options for preview (lower quality but much faster)
            stream.options = {
                "crf": "45",  # Higher CRF = lower quality but faster encoding
                "speed": "8",  # Fastest encoding speed
                "cpu-used": "8"  # Maximum CPU efficiency mode
            }
            
            for rgb_frame in rgb_frames:
                av_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)
            
            # Flush encoder
            for packet in stream.encode():
                container.mux(packet)
            
            container.close()
            
            webm_path = output_path_webp.replace('.webp', '.webm')
            self.preview_video = webm_path
            logger.info(f"Preview WEBM created: {webm_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create WEBM, falling back to MP4: {e}")
            self.preview_video = output_path_mp4
            logger.info(f"Preview MP4 created: {output_path_mp4}")
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray],
        is_moving: bool,
        confidence: float,
        current_viseme: Optional[str] = None,
        viseme_confidence: float = 0.0,
        geometric_features: Optional[dict] = None,
        frame_number: Optional[int] = None,
        consonant_scores: Optional[dict] = None,
        analyzer_method: str = 'basic'
    ) -> np.ndarray:
        """
        Enhanced frame annotation with MediaPipe landmarks and viseme display
        """
        annotated = frame.copy()
        
        # Add movement indicator
        color = (0, 255, 0) if is_moving else (255, 0, 0)
        status_text = f"SPEAKING: {confidence:.2f}" if is_moving else f"SILENT: {confidence:.2f}"
        
        # Add semi-transparent background for better text visibility
        overlay = annotated.copy()
        cv2.rectangle(overlay, (5, 5), (350, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        # Add frame number if provided - position it in top-right area
        if frame_number is not None:
            cv2.putText(
                annotated,
                f"Fr: {frame_number}",
                (200, 25),  # Top-right area, same height as status text
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,  # Smaller font
                (0, 255, 255),  # Bright yellow for visibility
                1  # Thinner stroke
            )

        # Add analyzer method info
        method_color = (255, 255, 0) if 'temporal' in analyzer_method.lower() else (128, 128, 128)
        cv2.putText(
            annotated,
            f"Analyzer: {analyzer_method}",
            (200, 50),  # Below frame number
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            method_color,
            1
        )
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
            # Draw mouth contour with viseme-based coloring
            mouth_indices = self.INNER_MOUTH_INDICES
            
            # Color coding for visemes
            viseme_colors = {
                # Vowels
                'A': (255, 100, 100),  # Red-ish
                'E': (100, 255, 100),  # Green-ish
                'I': (100, 100, 255),  # Blue-ish
                'O': (255, 255, 100),  # Yellow-ish
                'U': (255, 100, 255),  # Magenta-ish
                # Consonants (cooler colors)
                'B': (150, 75, 0),     # Brown
                'P': (150, 150, 0),    # Olive
                'M': (75, 150, 75),    # Dark green
                'F': (200, 50, 150),   # Pink
                'V': (150, 50, 200),   # Purple
                'TH': (100, 200, 200), # Cyan
                'T': (200, 150, 50),   # Orange
                'D': (150, 100, 50),   # Bronze
                'N': (50, 150, 100),   # Teal
                'K': (200, 100, 0),    # Dark orange
                'G': (100, 150, 200),  # Light blue
                'neutral': (128, 128, 128)  # Gray
            }
            
            # Get viseme color or default
            if current_viseme and current_viseme in viseme_colors:
                point_color = viseme_colors[current_viseme]
            else:
                point_color = (0, 255, 0) if is_moving else (0, 0, 255)
            
            for i in mouth_indices:
                if i < len(landmarks):
                    x, y = int(landmarks[i][0]), int(landmarks[i][1])
                    cv2.circle(annotated, (x, y), 2, point_color, -1)
            
            # Draw MAR value
            mar = self.calculate_mar(landmarks)
            mar_text = f"MAR: {mar:.3f}"
            cv2.putText(
                annotated,
                mar_text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw Viseme detection if available
            if current_viseme:
                # Draw large viseme character
                viseme_display = current_viseme if current_viseme != 'neutral' else '_'
                viseme_color = viseme_colors.get(current_viseme, (255, 255, 255))
                
                # Large viseme character display
                cv2.putText(
                    annotated,
                    viseme_display,
                    (250, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,  # Large font
                    viseme_color,
                    3
                )
                
                # Viseme label and confidence
                viseme_text = f"Viseme: {current_viseme}"
                cv2.putText(
                    annotated,
                    viseme_text,
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    viseme_color,
                    2
                )
                
                conf_text = f"Conf: {viseme_confidence:.1%}"
                cv2.putText(
                    annotated,
                    conf_text,
                    (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                # Add consonant scores debugging
                if consonant_scores:
                    score_text = " ".join([f"{k}:{v:.2f}" for k, v in consonant_scores.items()])
                    cv2.putText(
                        annotated,
                        f"Consonants: {score_text}",
                        (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,  # Smaller font for consonant scores
                        (255, 100, 255),  # Purple color for consonant scores
                        1
                    )
        
        # Add geometric features display if available
        if geometric_features:
            # Extend overlay for additional metrics
            overlay = annotated.copy()
            cv2.rectangle(overlay, (5, 135), (400, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            
            # Display key geometric features
            roundedness = geometric_features.get('roundedness', 0.0)
            lip_ratio = geometric_features.get('lip_ratio', 0.0)
            mar = geometric_features.get('mar', 0.0)
            
            # Line 1: Roundedness (most important for U/O)
            round_text = f"Roundedness: {roundedness:.3f}"
            cv2.putText(
                annotated,
                round_text,
                (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),  # Yellow for roundedness
                1
            )
            
            # Line 2: Lip Ratio (width/height)
            ratio_text = f"Lip Ratio: {lip_ratio:.2f}"
            cv2.putText(
                annotated,
                ratio_text,
                (10, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),  # Magenta for lip ratio
                1
            )
            
            # Line 3: MAR for reference
            mar_text = f"MAR: {mar:.3f}"
            cv2.putText(
                annotated,
                mar_text,
                (10, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White for MAR
                1
            )

            # Add consonant features for debugging consonant detection
            lip_contact = geometric_features.get('lip_contact', 0.0)
            teeth_visibility = geometric_features.get('teeth_visibility', 0.0)
            lip_compression = geometric_features.get('lip_compression', 0.0)

            # Extend the black background to accommodate consonant features
            overlay = annotated.copy()
            cv2.rectangle(overlay, (5, 200), (400, 260), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

            # Line 4: Lip Contact (for B, P, M)
            contact_text = f"LipContact: {lip_contact:.3f}"
            cv2.putText(
                annotated,
                contact_text,
                (10, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),  # Green for lip contact
                1
            )

            # Line 5: Teeth Visibility (for F, V, TH)
            teeth_text = f"TeethVis: {teeth_visibility:.3f}"
            cv2.putText(
                annotated,
                teeth_text,
                (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 200, 255),  # Orange for teeth visibility
                1
            )

            # Line 6: Lip Compression (for various consonants)
            compression_text = f"LipComp: {lip_compression:.3f}"
            cv2.putText(
                annotated,
                compression_text,
                (10, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 200, 0),  # Cyan for lip compression
                1
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