"""
OpenSeeFace provider for mouth movement analysis
Robust facial tracking with strong performance in challenging conditions
"""

import logging
import os
import sys
from typing import Optional, Tuple, List, Any, Dict
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

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

# Try to import bundled OpenSeeFace components
OPENSEEFACE_AVAILABLE = False
try:
    # Import from bundled OpenSeeFace
    openseeface_path = os.path.join(current_dir, "..", "openseeface")
    if openseeface_path not in sys.path:
        sys.path.insert(0, openseeface_path)
    
    from tracker import Tracker
    from model_downloader import openseeface_downloader
    OPENSEEFACE_AVAILABLE = True
    print("Using bundled OpenSeeFace components")
except ImportError as e:
    try:
        # Fallback: Try user's environment or common paths
        possible_paths = [
            os.path.join(os.path.expanduser("~"), "OpenSeeFace"),
            os.path.join(os.getcwd(), "OpenSeeFace"),
            "/opt/OpenSeeFace",
            "OpenSeeFace"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                sys.path.insert(0, path)
                try:
                    from tracker import Tracker
                    OPENSEEFACE_AVAILABLE = True
                    print(f"Using OpenSeeFace from: {path}")
                    break
                except ImportError:
                    continue
                    
        if not OPENSEEFACE_AVAILABLE:
            print("OpenSeeFace not available. Install with: pip install onnxruntime opencv-python pillow numpy")
    except Exception as e:
        print(f"OpenSeeFace initialization failed: {e}")

logger = logging.getLogger(__name__)


class OpenSeeFaceProvider(AbstractProvider):
    """
    OpenSeeFace provider for facial landmark detection
    
    Advantages:
    - Excellent stability in challenging lighting conditions
    - CPU-optimized for resource-constrained environments  
    - Multiple model options (speed vs accuracy trade-off)
    - Works well with partial face occlusion
    
    Best for:
    - Poor lighting conditions
    - Unstable camera setups
    - Real-time processing requirements
    - Fallback when MediaPipe struggles
    """
    
    def __init__(
        self,
        sensitivity: float = 0.3,
        min_duration: float = 0.1,
        merge_threshold: float = 0.2,
        confidence_threshold: float = 0.5,
        viseme_sensitivity: float = 1.0,
        viseme_confidence_threshold: float = 0.4,
        viseme_smoothing: float = 0.3,
        enable_consonant_detection: bool = False
    ):
        """
        Initialize OpenSeeFace provider
        
        Args:
            sensitivity: Movement detection sensitivity (0.1-1.0)
            min_duration: Minimum movement duration in seconds
            merge_threshold: Threshold for merging nearby segments
            confidence_threshold: Minimum confidence for valid detection
            viseme_sensitivity: Viseme detection sensitivity
            viseme_confidence_threshold: Minimum confidence for viseme detection
            viseme_smoothing: Temporal smoothing factor for visemes
            enable_consonant_detection: Whether to detect consonants
        """
        if not OPENSEEFACE_AVAILABLE:
            raise RuntimeError(
                "OpenSeeFace is not available. Please install OpenSeeFace:\n"
                "git clone https://github.com/emilianavt/OpenSeeFace.git\n"
                "cd OpenSeeFace && pip install onnxruntime opencv-python pillow numpy"
            )
        
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV is required for OpenSeeFace provider. Install with: pip install opencv-python")
        
        # Call parent constructor with all parameters
        super().__init__(
            sensitivity=sensitivity,
            min_duration=min_duration,
            merge_threshold=merge_threshold,
            confidence_threshold=confidence_threshold,
            viseme_sensitivity=viseme_sensitivity,
            viseme_confidence_threshold=viseme_confidence_threshold,
            viseme_smoothing=viseme_smoothing,
            enable_consonant_detection=enable_consonant_detection
        )
        
        # Calculate MAR threshold using exponential sensitivity mapping
        import math
        base_max = 0.20  # OpenSeeFace is more stable, can use lower thresholds
        sensitivity_factor = 3.5
        normalized_sensitivity = max(0.0, min(1.0, sensitivity))
        
        self.mar_threshold = base_max * math.exp(-sensitivity_factor * normalized_sensitivity)
        self.mar_threshold = max(0.008, self.mar_threshold)  # Lower minimum for better detection
        
        # Initialize tracking parameters
        self.tracker = None
        
        logger.info(f"OpenSeeFace provider initialized with MAR threshold: {self.mar_threshold:.3f}")
        
        # Initialize provider
        self._initialize()
    
    def _check_dependencies(self):
        """Check if required dependencies are installed"""
        if not OPENSEEFACE_AVAILABLE:
            raise RuntimeError("OpenSeeFace is not available")
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV is not available")
    
    def _initialize(self):
        """Initialize provider-specific components"""
        self._check_dependencies()
        # Additional initialization done in analyze_video when we have video dimensions
    
    @property
    def provider_name(self) -> str:
        """Return the name of this provider"""
        return "OpenSeeFace"
    
    def detect_movement(self, frame: np.ndarray) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Detect mouth movement in a single frame
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (movement_detected, confidence, landmarks)
        """
        if self.tracker is None:
            return False, 0.0, None
        
        try:
            faces = self.tracker.predict(frame)
            
            if len(faces) > 0:
                face = faces[0]
                landmarks = face.landmarks
                confidence = face.conf
                
                if confidence > 0.7:  # Conservative threshold
                    mar = self._calculate_mar_openseeface(landmarks)
                    movement_detected = mar > self.mar_threshold
                    return movement_detected, confidence, landmarks
            
            return False, 0.0, None
            
        except Exception as e:
            logger.warning(f"OpenSeeFace movement detection failed: {e}")
            return False, 0.0, None
    
    def calculate_mar(self, landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio from landmarks
        
        Args:
            landmarks: Facial landmarks array
            
        Returns:
            Mouth aspect ratio value
        """
        return self._calculate_mar_openseeface(landmarks)
    
    def _get_model_filename(self, model_type: int) -> str:
        """Get model filename for given model type"""
        model_map = {
            0: 'lm_model0_opt.onnx',
            1: 'lm_model1_opt.onnx', 
            2: 'lm_model2_opt.onnx',
            3: 'lm_model3_opt.onnx',
            4: 'lm_model4_opt.onnx',
            -1: 'lm_modelT_opt.onnx',
            -2: 'lm_modelV_opt.onnx',
            -3: 'lm_modelU_opt.onnx'
        }
        return model_map.get(model_type, 'lm_model3_opt.onnx')  # Default to model 3
    
    @staticmethod
    def is_available() -> bool:
        """Check if OpenSeeFace is available"""
        return OPENSEEFACE_AVAILABLE and OPENCV_AVAILABLE
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get provider information and capabilities"""
        return {
            'name': 'OpenSeeFace',
            'description': 'Robust facial tracking for challenging conditions',
            'version': '1.20.0',
            'advantages': [
                'Excellent stability in poor lighting',
                'CPU-optimized performance', 
                'Multiple quality/speed models',
                'Works with partial occlusion'
            ],
            'best_for': [
                'Poor lighting conditions',
                'Unstable camera setups', 
                'Real-time processing',
                'MediaPipe fallback scenarios'
            ],
            'requirements': ['onnxruntime', 'opencv-python', 'pillow', 'numpy'],
            'model_options': {
                0: 'Fastest (lowest quality)',
                1: 'Fast (basic quality)', 
                2: 'Balanced (good quality)',
                3: 'High quality (slower)',
                4: 'Wink-optimized (special purpose)'
            }
        }
    
    def analyze_video(self, video_input, preview_mode: bool = False, enable_viseme: bool = False, viseme_options: Dict[str, Any] = None) -> TimingData:
        """
        Analyze video using OpenSeeFace tracking
        
        Args:
            video_input: Video input (various ComfyUI formats supported)
            preview_mode: Enable visual feedback during processing
            enable_viseme: Enable advanced viseme detection
            viseme_options: Configuration for viseme detection
            
        Returns:
            TimingData with movement segments and optional viseme information
        """
        # Handle ComfyUI video input format (same as MediaPipe)
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
            raise ValueError(f"Cannot extract file path from video input of type {type(video_input)}")
        
        # Store viseme options for modular analysis
        self.viseme_options = viseme_options or {}
        
        logger.info(f"Analyzing video with OpenSeeFace: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video properties: {original_width}x{original_height}, {fps:.2f} FPS, {total_frames} frames")
        
        # Initialize OpenSeeFace tracker with model management
        model_type = self.viseme_options.get('openseeface_model', 2)
        detection_threshold = 0.6
        tracking_threshold = 0.7  # More conservative for stability
        
        # Ensure required models are available
        if 'openseeface_downloader' in globals():
            # Check and download required models
            required_model = self._get_model_filename(model_type)
            model_path = openseeface_downloader.get_model_path(required_model)
            
            if not model_path:
                logger.info(f"Downloading OpenSeeFace model: {required_model}")
                model_path = openseeface_downloader.download_model(required_model)
                
                if not model_path:
                    logger.warning(f"Failed to download {required_model}, falling back to bundled model")
                    model_type = 0  # Use bundled basic model
            
            # Ensure basic models are available
            if not openseeface_downloader.ensure_basic_models():
                raise RuntimeError("OpenSeeFace basic models not available")
            
            # Use custom model directory
            model_dir = openseeface_downloader.organized_models_dir
        else:
            model_dir = None  # Use default model directory
        
        self.tracker = Tracker(
            width=original_width,
            height=original_height,
            model_type=model_type,
            detection_threshold=detection_threshold,
            threshold=tracking_threshold,
            max_faces=1,  # Single face tracking for mouth analysis
            discard_after=10,  # Keep tracking longer for stability
            scan_every=5,  # Less frequent scanning for performance
            max_threads=4,
            silent=True,  # Reduce console output
            no_gaze=True,  # Disable gaze for mouth-only analysis
            model_dir=model_dir  # Use our organized model directory
        )
        
        # Process video frames
        mar_values = []
        viseme_frames = []
        frame_count = 0
        
        logger.info("Starting OpenSeeFace tracking...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Track faces using OpenSeeFace
                faces = self.tracker.predict(frame)
                
                if len(faces) > 0:
                    # Use the first detected face
                    face = faces[0]
                    landmarks = face.landmarks
                    confidence = face.conf
                    
                    if confidence > tracking_threshold:
                        # Calculate MAR from OpenSeeFace landmarks
                        mar = self._calculate_mar_openseeface(landmarks)
                        mar_values.append(mar)
                        
                        # Extract viseme features if enabled
                        if enable_viseme:
                            features = self._extract_geometric_features_openseeface(landmarks)
                            
                            # Use modular analysis system if available
                            if ANALYSIS_AVAILABLE and hasattr(self, 'viseme_options'):
                                analyzer = VisemeAnalysisFactory.create_analyzer(self.viseme_options)
                                result = analyzer.classify_viseme(features, self.viseme_options.get('enable_consonant_detection', False))
                                viseme, viseme_conf = result.viseme, result.confidence
                            else:
                                # Fallback to built-in method
                                viseme, viseme_conf = self._classify_viseme_basic(features)
                            
                            viseme_frames.append(VisemeFrame(
                                frame_index=frame_count,
                                viseme=viseme,
                                confidence=viseme_conf,
                                geometric_features=features
                            ))
                    else:
                        # Low confidence tracking
                        mar_values.append(0.0)
                        if enable_viseme:
                            viseme_frames.append(VisemeFrame(
                                frame_index=frame_count,
                                viseme='neutral',
                                confidence=0.0,
                                geometric_features={}
                            ))
                else:
                    # No face detected
                    mar_values.append(0.0)
                    if enable_viseme:
                        viseme_frames.append(VisemeFrame(
                            frame_index=frame_count,
                            viseme='neutral', 
                            confidence=0.0,
                            geometric_features={}
                        ))
            
            except Exception as e:
                logger.warning(f"OpenSeeFace tracking failed on frame {frame_count}: {e}")
                mar_values.append(0.0)
                if enable_viseme:
                    viseme_frames.append(VisemeFrame(
                        frame_index=frame_count,
                        viseme='neutral',
                        confidence=0.0,
                        geometric_features={}
                    ))
            
            frame_count += 1
            
            # Progress reporting
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"OpenSeeFace tracking progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        logger.info(f"OpenSeeFace tracking completed: {frame_count} frames processed")
        
        # Detect movement segments
        movement_segments = self._detect_movement_segments(mar_values, fps)
        
        return TimingData(
            provider_name="OpenSeeFace",
            total_duration=total_duration,
            fps=fps,
            total_frames=total_frames,
            movement_segments=movement_segments,
            mar_values=mar_values,
            mar_threshold=self.mar_threshold,
            viseme_frames=viseme_frames if enable_viseme else []
        )
    
    def _calculate_mar_openseeface(self, landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio from OpenSeeFace landmarks
        
        OpenSeeFace provides 68 facial landmarks in a different format than MediaPipe.
        We need to map to the mouth region (landmarks 48-67 in 68-point format).
        """
        try:
            if landmarks is None or len(landmarks) < 68:
                return 0.0
            
            # Convert to standard format if needed
            if landmarks.shape[0] == 2:  # If landmarks are in [2, N] format
                landmarks = landmarks.T  # Convert to [N, 2]
            
            # OpenSeeFace mouth landmarks (48-67 in 68-point model)
            # Inner mouth landmarks for MAR calculation
            top_lip = landmarks[51]     # Top center
            bottom_lip = landmarks[57]  # Bottom center  
            left_corner = landmarks[48] # Left corner
            right_corner = landmarks[54] # Right corner
            
            # Calculate mouth aspect ratio
            vertical_dist = np.linalg.norm(top_lip - bottom_lip)
            horizontal_dist = np.linalg.norm(left_corner - right_corner)
            
            if horizontal_dist > 0:
                mar = vertical_dist / horizontal_dist
            else:
                mar = 0.0
            
            return min(mar, 1.0)  # Clamp to reasonable values
            
        except (IndexError, ValueError, TypeError) as e:
            logger.warning(f"MAR calculation failed: {e}")
            return 0.0
    
    def _extract_geometric_features_openseeface(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract geometric features from OpenSeeFace landmarks for viseme detection
        """
        try:
            if landmarks is None or len(landmarks) < 68:
                return {}
            
            # Convert format if needed
            if landmarks.shape[0] == 2:
                landmarks = landmarks.T
            
            # Mouth region landmarks (48-67)
            mouth_landmarks = landmarks[48:68]
            
            # Calculate basic features
            features = {}
            
            # Mouth aspect ratio (vertical opening)
            features['mar'] = self._calculate_mar_openseeface(landmarks)
            
            # Lip ratio (width vs height relationship)
            left_corner = landmarks[48]
            right_corner = landmarks[54] 
            top_lip = landmarks[51]
            bottom_lip = landmarks[57]
            
            width = np.linalg.norm(right_corner - left_corner)
            height = np.linalg.norm(top_lip - bottom_lip)
            features['lip_ratio'] = width / max(height, 0.001)
            
            # Roundedness (lip curvature approximation)
            # Use distance from corners to center points
            center_top = landmarks[51]
            center_bottom = landmarks[57]
            mouth_center = (center_top + center_bottom) / 2
            
            corner_to_center_dist = (np.linalg.norm(left_corner - mouth_center) + 
                                   np.linalg.norm(right_corner - mouth_center)) / 2
            features['roundedness'] = min(1.0, height / max(corner_to_center_dist, 0.001))
            
            # Mouth area approximation
            features['mouth_area'] = width * height
            
            # Lip contact (how close upper and lower lips are)
            features['lip_contact'] = 1.0 - min(1.0, features['mar'] / 0.1)
            
            # Teeth visibility (approximation from mouth opening)
            features['teeth_visibility'] = min(1.0, features['mar'] / 0.05)
            
            # Lip compression (pressure/tension approximation)
            # Based on mouth width vs relaxed state
            relaxed_width = np.linalg.norm(landmarks[48] - landmarks[54]) 
            current_width = width
            features['lip_compression'] = max(0.0, 1.0 - (current_width / max(relaxed_width, 0.001)))
            
            # Nose flare (limited detection from available landmarks)
            # Approximate from nostril landmarks if available
            if len(landmarks) > 31:
                left_nostril = landmarks[31]
                right_nostril = landmarks[35] 
                nostril_width = np.linalg.norm(right_nostril - left_nostril)
                features['nose_flare'] = min(1.0, nostril_width / 20.0)  # Rough normalization
            else:
                features['nose_flare'] = 0.0
            
            return features
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return {}
    
    def _classify_viseme_basic(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Basic viseme classification fallback when modular system unavailable
        """
        if not features:
            return 'neutral', 0.0
        
        mar = features.get('mar', 0)
        lip_ratio = features.get('lip_ratio', 0)
        roundedness = features.get('roundedness', 0)
        
        # Simple vowel classification
        if mar > 0.15:
            return 'A', min(1.0, mar * 2.0)
        elif mar > 0.08 and roundedness > 0.6:
            return 'O', roundedness
        elif lip_ratio > 3.0:
            return 'E', min(1.0, lip_ratio / 4.0)
        elif roundedness > 0.7:
            return 'U', roundedness
        elif mar > 0.05:
            return 'I', mar * 3.0
        else:
            return 'neutral', 0.3
    
    def _detect_movement_segments(self, mar_values: List[float], fps: float) -> List[MovementSegment]:
        """
        Detect movement segments from MAR values using OpenSeeFace-optimized thresholds
        """
        segments = []
        
        if not mar_values or fps <= 0:
            return segments
        
        # OpenSeeFace is more stable, so we can use more sensitive detection
        threshold = self.mar_threshold * 0.8  # Slightly lower threshold
        min_duration_frames = max(2, int(fps * 0.1))  # Minimum 0.1 second segments
        
        in_movement = False
        movement_start = 0
        
        for i, mar in enumerate(mar_values):
            if not in_movement and mar > threshold:
                # Start of movement
                in_movement = True
                movement_start = i
            elif in_movement and mar <= threshold:
                # End of movement
                movement_duration = i - movement_start
                
                if movement_duration >= min_duration_frames:
                    start_time = movement_start / fps
                    end_time = i / fps
                    
                    # Calculate confidence based on average MAR in segment
                    segment_mars = mar_values[movement_start:i]
                    avg_mar = sum(segment_mars) / len(segment_mars)
                    confidence = min(1.0, (avg_mar - threshold) / threshold)
                    
                    segments.append(MovementSegment(
                        start_time=start_time,
                        end_time=end_time,
                        confidence=confidence,
                        peak_value=max(segment_mars),
                        frame_start=movement_start,
                        frame_end=i
                    ))
                
                in_movement = False
        
        # Handle case where movement continues to end of video
        if in_movement:
            movement_duration = len(mar_values) - movement_start
            if movement_duration >= min_duration_frames:
                start_time = movement_start / fps
                end_time = len(mar_values) / fps
                
                segment_mars = mar_values[movement_start:]
                avg_mar = sum(segment_mars) / len(segment_mars)
                confidence = min(1.0, (avg_mar - threshold) / threshold)
                
                segments.append(MovementSegment(
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                    peak_value=max(segment_mars),
                    frame_start=movement_start,
                    frame_end=len(mar_values)
                ))
        
        logger.info(f"OpenSeeFace detected {len(segments)} movement segments")
        return segments