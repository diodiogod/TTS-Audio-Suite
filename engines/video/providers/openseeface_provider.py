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
        viseme_confidence_threshold: float = 0.04,
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
        import sys
        
        # Inform user about Python 3.13 fallback context
        if sys.version_info >= (3, 13):
            print("🧪 Using OpenSeeFace provider (experimental alternative for Python 3.13)")
            print("⚠️ Note: Results may be less accurate than MediaPipe on older Python versions")
            print("📢 Want MediaPipe Python 3.13 support? Add your voice:")
            print("   https://github.com/google-ai-edge/mediapipe/issues/5708")
        
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
        
        # Calculate MAR threshold for sophisticated multi-point OpenSeeFace calculation
        # New MAR range: 0.0-1.5 with enhanced precision using inner+outer mouth points
        # Closed mouth: ~0.0-0.3, Speaking: ~0.4-0.8, Wide open: ~0.9-1.5
        
        normalized_sensitivity = max(0.0, min(1.0, sensitivity))
        
        # Threshold mapping for sophisticated MAR calculation:
        # High sensitivity (1.0) = 0.25 (detects subtle movements)
        # Medium sensitivity (0.5) = 0.45 (normal speech detection)  
        # Low sensitivity (0.0) = 0.65 (only clear speech)
        base_threshold = 0.45  # Good baseline for speech detection
        sensitivity_range = 0.20  # ±0.20 range for fine control
        
        # Higher sensitivity = lower threshold (more sensitive to movement)
        self.mar_threshold = base_threshold + sensitivity_range - (normalized_sensitivity * 2 * sensitivity_range)
        
        # Clamp to reasonable bounds for sophisticated MAR (0.2 to 0.8)
        self.mar_threshold = max(0.20, min(0.80, self.mar_threshold))
        
        # Initialize tracking parameters
        self.tracker = None
        
        logger.info(f"OpenSeeFace provider initialized with MAR threshold: {self.mar_threshold:.3f} (sensitivity: {sensitivity:.2f})")
        
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
                
                # Handle different face object structures
                if hasattr(face, 'landmarks'):
                    landmarks = face.landmarks
                elif hasattr(face, 'lms'):
                    landmarks = face.lms
                elif hasattr(face, 'pts'):
                    landmarks = face.pts
                else:
                    landmarks = None
                
                if hasattr(face, 'conf'):
                    confidence = face.conf
                elif hasattr(face, 'confidence'):
                    confidence = face.confidence
                else:
                    confidence = 1.0
                
                if confidence > 0.7 and landmarks is not None:  # Conservative threshold
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
        
        # Calculate optimal processing dimensions for OpenSeeFace
        # Target: 720p max for analysis, 540p for preview (good balance of speed vs quality)
        max_dimension = 720
        preview_max_dimension = 540 if preview_mode else 720
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
                logger.info(f"Resizing video from {original_width}x{original_height} to {width}x{height} for optimal OpenSeeFace performance")
        else:
            width, height = original_width, original_height
            logger.info(f"Video resolution {width}x{height} is optimal, no resizing needed")
        
        logger.info(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
        
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
            
            # Download RetinaFace detection model and config if not available
            retinaface_model = 'retinaface_640x640_opt.onnx'
            priorbox_config = 'priorbox_640x640.json'
            
            if not openseeface_downloader.get_model_path(retinaface_model):
                logger.info(f"Downloading required detection model: {retinaface_model}")
                openseeface_downloader.download_model(retinaface_model)
            
            if not openseeface_downloader.get_model_path(priorbox_config):
                logger.info(f"Downloading required config file: {priorbox_config}")
                openseeface_downloader.download_model(priorbox_config)
            
            # Ensure basic models are available
            if not openseeface_downloader.ensure_basic_models():
                raise RuntimeError("OpenSeeFace basic models not available")
            
            # Use custom model directory
            model_dir = openseeface_downloader.organized_models_dir
            
            # For bundled models, use the bundled directory
            if model_type == 0:
                bundled_dir = os.path.join(current_dir, "..", "openseeface", "models")
                if os.path.exists(bundled_dir):
                    model_dir = bundled_dir
        else:
            # Use bundled model directory
            model_dir = os.path.join(current_dir, "..", "openseeface", "models")
        
        # Check if we have RetinaFace model available
        has_retinaface = False
        if model_dir:
            retinaface_path = os.path.join(model_dir, 'retinaface_640x640_opt.onnx')
            priorbox_path = os.path.join(model_dir, 'priorbox_640x640.json')
            has_retinaface = os.path.exists(retinaface_path) and os.path.exists(priorbox_path)
        
        try:
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
                use_retinaface=has_retinaface,  # Only use RetinaFace if models are available
                model_dir=model_dir  # Use our organized model directory
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenSeeFace tracker: {e}")
            # If we have model issues, try with basic bundled model
            if model_type != 0:
                logger.info("Retrying with bundled basic model...")
                try:
                    bundled_model_dir = os.path.join(current_dir, "..", "openseeface", "models")
                    self.tracker = Tracker(
                        width=original_width,
                        height=original_height,
                        model_type=0,  # Use bundled basic model
                        detection_threshold=detection_threshold,
                        threshold=tracking_threshold,
                        max_faces=1,
                        discard_after=10,
                        scan_every=5,
                        max_threads=4,
                        silent=True,
                        no_gaze=True,
                        use_retinaface=False,  # Disable RetinaFace for bundled basic model
                        model_dir=bundled_model_dir  # Use bundled directory
                    )
                    logger.info("Successfully initialized with basic bundled model")
                except Exception as e2:
                    logger.error(f"Failed to initialize even with basic model: {e2}")
                    raise RuntimeError(f"OpenSeeFace initialization failed: {e2}")
            else:
                raise RuntimeError(f"OpenSeeFace initialization failed: {e}")
        
        # Process video frames
        mar_values = []
        viseme_frames = []
        preview_frames = []
        frame_count = 0
        
        # Store frame dimensions for coordinate scaling in visualization
        self.original_width = original_width
        self.original_height = original_height
        self.processed_width = width
        self.processed_height = height
        
        logger.info("Starting OpenSeeFace tracking...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for optimal OpenSeeFace processing
            if self.processed_width != original_width or self.processed_height != original_height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            
            # Store frame for preview if requested
            if preview_mode:  # Store all frames - optimizations make this fast enough now
                # (skip frames for faster preview)
                # if preview_mode and frame_count % 3 == 0:  # Only store every 3rd frame
                # Create annotated frame for preview
                annotated_frame = self._create_annotated_frame(
                    frame.copy(), 
                    faces if 'faces' in locals() else [],
                    mar_values[-1] if mar_values else 0.0,
                    viseme_frames[-1] if viseme_frames else None,
                    frame_count
                )
                preview_frames.append(annotated_frame)
            
            try:
                # Track faces using OpenSeeFace
                faces = self.tracker.predict(frame)
                
                if len(faces) > 0:
                    # Use the first detected face
                    face = faces[0]
                    
                    # Debug: check what attributes the face object actually has
                    if not hasattr(face, 'landmarks'):
                        logger.debug(f"Face object attributes: {dir(face)}")
                        # Try common alternative attribute names
                        if hasattr(face, 'lms'):
                            landmarks = face.lms
                        elif hasattr(face, 'pts'):
                            landmarks = face.pts
                        elif hasattr(face, 'points'):
                            landmarks = face.points
                        else:
                            logger.warning(f"No landmarks found in face object with attributes: {dir(face)}")
                            landmarks = None
                    else:
                        landmarks = face.landmarks
                    
                    # Debug: log landmarks info when no MAR is calculated
                    if landmarks is not None and frame_count < 5:  # Only log first few frames
                        logger.info(f"Frame {frame_count}: landmarks shape={np.array(landmarks).shape}, first few points={np.array(landmarks)[:5] if len(landmarks) > 5 else landmarks}")
                    
                    # Get confidence
                    if hasattr(face, 'conf'):
                        confidence = face.conf
                    elif hasattr(face, 'confidence'):
                        confidence = face.confidence
                    elif hasattr(face, 'score'):
                        confidence = face.score
                    else:
                        confidence = 1.0  # Default if no confidence available
                    
                    if confidence > tracking_threshold and landmarks is not None:
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
        
        # Create preview video if requested
        if preview_mode and len(preview_frames) > 0:
            logger.info("Creating preview video with OpenSeeFace tracking annotations...")
            provider_dir = os.path.dirname(os.path.abspath(__file__))
            utils_dir = os.path.join(os.path.dirname(provider_dir), "utils")
            if utils_dir not in sys.path:
                sys.path.insert(0, utils_dir)
            from preview_creator import PreviewVideoCreator
            preview_path = PreviewVideoCreator.create_preview_video(
                preview_frames, fps, width, height, "OpenSeeFace"
            )
            if preview_path:
                self.preview_video = preview_path
        
        # Detect movement segments
        movement_segments = self._detect_movement_segments(mar_values, fps)
        
        # Add viseme sequences to movement segments if viseme detection was enabled
        if enable_viseme and viseme_frames:
            for segment in movement_segments:
                # Get visemes for this segment's frame range
                segment_visemes = []
                segment_confidences = []
                
                for frame_idx in range(segment.start_frame, segment.end_frame):
                    if frame_idx < len(viseme_frames):
                        vf = viseme_frames[frame_idx]
                        if hasattr(vf, 'viseme'):
                            segment_visemes.append(vf.viseme)
                            segment_confidences.append(vf.confidence)
                        elif hasattr(vf, 'detected_viseme'):
                            segment_visemes.append(vf.detected_viseme)
                            segment_confidences.append(getattr(vf, 'confidence', 1.0))
                
                # Add viseme data to segment
                if segment_visemes:
                    segment.viseme_sequence = segment_visemes
                    segment.viseme_confidences = segment_confidences
        
        return TimingData(
            segments=movement_segments,
            fps=fps,
            total_frames=total_frames,
            total_duration=total_duration,
            provider="OpenSeeFace",
            metadata={
                "mar_values": mar_values,
                "mar_threshold": self.mar_threshold
            },
            viseme_frames=viseme_frames if enable_viseme else []
        )
    
    def _calculate_mar_openseeface(self, landmarks: np.ndarray) -> float:
        """
        Sophisticated Mouth Aspect Ratio calculation using multiple landmark points
        
        OpenSeeFace provides 68 facial landmarks following the standard dlib model.
        Uses both inner and outer mouth contours for enhanced precision:
        - Outer mouth (48-59) for overall mouth shape
        - Inner mouth (60-67) for precise opening detection
        """
        try:
            if landmarks is None or len(landmarks) < 68:
                logger.info(f"OpenSeeFace MAR: Invalid landmarks - None={landmarks is None}, len={len(landmarks) if landmarks is not None else 'N/A'}")
                return 0.0
            
            # Debug: log that MAR calculation is starting (disabled for production)
            # logger.info(f"OpenSeeFace MAR: Starting calculation with {len(landmarks)} landmarks")
            
            # Convert to standard format if needed
            if landmarks.shape[0] == 2:  # If landmarks are in [2, N] format
                landmarks = landmarks.T  # Convert to [N, 2]
            
            # Apply coordinate transformation (keep existing working transformation)
            def transform_point(pt):
                # Handle 3D landmarks from OpenSeeFace (X, Y, Z) - use only X, Y
                if len(pt) >= 2:
                    return np.array([pt[1], pt[0]])  # Swap X and Y coordinates, ignore Z
                else:
                    return np.array([0, 0])  # Fallback for invalid points
            
            # === OUTER MOUTH MEASUREMENTS (Primary structure) ===
            # Outer lip contour provides mouth width and general shape
            outer_left = transform_point(landmarks[48])      # Left corner
            outer_right = transform_point(landmarks[54])     # Right corner  
            outer_top = transform_point(landmarks[51])       # Top center
            outer_bottom = transform_point(landmarks[57])    # Bottom center
            
            # Additional outer points for enhanced measurement
            outer_top_left = transform_point(landmarks[49])   # Top left
            outer_top_right = transform_point(landmarks[53])  # Top right
            outer_bottom_left = transform_point(landmarks[59]) # Bottom left
            outer_bottom_right = transform_point(landmarks[55]) # Bottom right
            
            # Calculate outer height for fallback inner bottom calculation
            outer_height_center = np.linalg.norm(outer_top - outer_bottom)
            
            # === INNER MOUTH MEASUREMENTS (Precise opening) ===
            # Inner lip contour provides accurate opening detection
            inner_left = transform_point(landmarks[60])      # Inner left
            inner_right = transform_point(landmarks[64])     # Inner right
            inner_top = transform_point(landmarks[62])       # Inner top center
            
            # Handle missing landmark 66 in bundled model - disable inner calculations
            if np.allclose(landmarks[66], [0, 0, 0]) or len(landmarks[66]) < 2:
                # Fallback: Use outer mouth only when inner landmarks are invalid
                inner_bottom = outer_bottom  # Use outer bottom as placeholder
                use_inner_mouth = False  # Flag to disable inner mouth calculations
            else:
                inner_bottom = transform_point(landmarks[66])  # Inner bottom center
                use_inner_mouth = True  # Use normal inner mouth calculations
            
            # Check for invalid coordinates and debug landmark issues
            key_points = [outer_left, outer_right, outer_top, outer_bottom,
                         inner_left, inner_right, inner_top, inner_bottom]
            
            # Debug: log all key points to see their actual values (disabled for production)
            # logger.info(f"OpenSeeFace key points: outer_left={outer_left}, outer_right={outer_right}")
            # logger.info(f"OpenSeeFace key points: outer_top={outer_top}, outer_bottom={outer_bottom}")
            # logger.info(f"OpenSeeFace key points: inner_left={inner_left}, inner_right={inner_right}")
            # logger.info(f"OpenSeeFace key points: inner_top={inner_top}, inner_bottom={inner_bottom}")
            
            if any(np.allclose(pt, [0, 0]) for pt in key_points):
                # Debug: log which points are zero to diagnose the issue (disabled for production)
                zero_points = [i for i, pt in enumerate(key_points) if np.allclose(pt, [0, 0])]
                # logger.info(f"OpenSeeFace MAR: Zero coordinates detected at points {zero_points}")
                return 0.0
            
            # === MULTI-POINT DISTANCE CALCULATIONS ===
            
            # Outer mouth dimensions (overall structure)
            outer_width = np.linalg.norm(outer_right - outer_left)
            outer_height_center = np.linalg.norm(outer_top - outer_bottom)
            
            # Multiple vertical measurements for robustness
            outer_height_left = np.linalg.norm(outer_top_left - outer_bottom_left)
            outer_height_right = np.linalg.norm(outer_top_right - outer_bottom_right)
            outer_height_avg = (outer_height_center + outer_height_left + outer_height_right) / 3
            
            # Inner mouth dimensions (precise opening)
            inner_width = np.linalg.norm(inner_right - inner_left)
            inner_height = np.linalg.norm(inner_top - inner_bottom)
            
            # === SOPHISTICATED MAR CALCULATION ===
            
            # Debug: log the actual measurements to understand the issue (limited to first 5 frames)
            debug_measurements = getattr(self, '_debug_frame_count', 0) < 5
            if debug_measurements:
                logger.info(f"OpenSeeFace measurements: outer_width={outer_width:.3f}, outer_height_avg={outer_height_avg:.3f}")
                logger.info(f"OpenSeeFace measurements: inner_width={inner_width:.3f}, inner_height={inner_height:.3f}")
                self._debug_frame_count = getattr(self, '_debug_frame_count', 0) + 1
            
            # Primary MAR using outer mouth for stability
            if outer_width > 0:
                outer_mar = outer_height_avg / outer_width
            else:
                outer_mar = 0.0
            
            # Secondary MAR using inner mouth for precision
            if inner_width > 0:
                inner_mar = inner_height / inner_width
            else:
                inner_mar = 0.0
            
            # Debug: log MAR components (limited to first 5 frames)
            if debug_measurements:
                logger.info(f"OpenSeeFace MAR components: outer_mar={outer_mar:.6f}, inner_mar={inner_mar:.6f}")
            
            # === HYBRID MAR COMBINATION ===
            # Combine outer (stability) and inner (precision) measurements
            
            if use_inner_mouth:
                # Normal calculation with both outer and inner mouth
                outer_weight = 0.7  # Primary structural measurement
                inner_weight = 0.3  # Precision opening adjustment
                
                # Combined MAR calculation
                if inner_height > 0.5:  # When mouth is clearly open, emphasize inner precision
                    combined_mar = (outer_weight * outer_mar) + (inner_weight * inner_mar * 2.0)
                else:  # When mouth is closed/barely open, rely more on outer structure
                    combined_mar = (0.9 * outer_mar) + (0.1 * inner_mar)
                
                # Enhanced opening detection
                opening_bonus = 0.0
                if inner_height > 0.2 and inner_width > 2.0:  # Detectable inner opening
                    opening_bonus = min(0.1, inner_height / 10.0)  # Small bonus for opening
                
                final_mar = combined_mar + opening_bonus
            else:
                # Fallback: Use only outer mouth when inner landmarks are invalid
                final_mar = outer_mar
            
            # Clamp to reasonable values (0.0-1.5 range for sophisticated detection)
            result = max(0.0, min(1.5, final_mar))
            
            return result
            
        except (IndexError, ValueError, TypeError) as e:
            logger.warning(f"Sophisticated MAR calculation failed: {e}")
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
            
            # Apply coordinate transformation for all feature calculations
            def transform_point(pt):
                return np.array([pt[1], pt[0]])  # Swap X and Y coordinates
            
            # Multi-point sophisticated analysis using both inner and outer lip landmarks
            
            # Outer lip contour (48-59)
            outer_left = transform_point(landmarks[48])    # Outer left corner
            outer_right = transform_point(landmarks[54])   # Outer right corner
            outer_top = transform_point(landmarks[51])     # Outer top center
            outer_bottom = transform_point(landmarks[57])  # Outer bottom center
            
            # Inner lip contour (60-67) for detailed mouth shape
            inner_left = transform_point(landmarks[60])    # Inner left
            inner_right = transform_point(landmarks[64])   # Inner right
            inner_top = transform_point(landmarks[62])     # Inner top center
            inner_bottom = transform_point(landmarks[66])  # Inner bottom center
            
            # Additional outer lip points for shape analysis
            outer_top_left = transform_point(landmarks[49])   # Top left
            outer_top_right = transform_point(landmarks[53])  # Top right
            outer_bottom_left = transform_point(landmarks[59]) # Bottom left
            outer_bottom_right = transform_point(landmarks[55]) # Bottom right
            
            # Check for invalid coordinates
            key_points = [outer_left, outer_right, outer_top, outer_bottom, 
                         inner_left, inner_right, inner_top, inner_bottom]
            if any(np.allclose(pt, [0, 0]) for pt in key_points):
                return {}
            
            # Multi-dimensional mouth measurements
            outer_width = np.linalg.norm(outer_right - outer_left)
            outer_height = np.linalg.norm(outer_top - outer_bottom)
            inner_width = np.linalg.norm(inner_right - inner_left)
            inner_height = np.linalg.norm(inner_top - inner_bottom)
            
            # Primary measurements (using outer for main ratio)
            width = outer_width
            height = outer_height
            features['lip_ratio'] = width / max(height, 0.001)
            
            # Inner lip analysis for mouth opening precision
            features['inner_opening'] = inner_height / max(inner_width, 0.001)
            features['outer_to_inner_ratio'] = outer_height / max(inner_height, 0.001)
            
            # Lip thickness analysis
            features['lip_thickness'] = (outer_height - inner_height) / max(outer_height, 0.001)
            
            # Advanced roundedness using multiple points
            mouth_center = (outer_top + outer_bottom) / 2
            
            # Distance from corners to center
            corner_to_center_dist = (np.linalg.norm(outer_left - mouth_center) + 
                                   np.linalg.norm(outer_right - mouth_center)) / 2
            features['roundedness'] = min(1.0, height / max(corner_to_center_dist, 0.001))
            
            # Lip curvature analysis using additional points
            # Top lip curvature (how curved vs straight)
            top_left_dist = np.linalg.norm(outer_top_left - outer_top)
            top_right_dist = np.linalg.norm(outer_top_right - outer_top)
            top_curve = (top_left_dist + top_right_dist) / max(outer_width * 0.5, 0.001)
            features['top_lip_curvature'] = min(1.0, top_curve)
            
            # Bottom lip curvature
            bottom_left_dist = np.linalg.norm(outer_bottom_left - outer_bottom)
            bottom_right_dist = np.linalg.norm(outer_bottom_right - outer_bottom)
            bottom_curve = (bottom_left_dist + bottom_right_dist) / max(outer_width * 0.5, 0.001)
            features['bottom_lip_curvature'] = min(1.0, bottom_curve)
            
            # Mouth area approximation
            features['mouth_area'] = width * height
            
            # Lip contact (how close upper and lower lips are)
            features['lip_contact'] = 1.0 - min(1.0, features['mar'] / 0.1)
            
            # Teeth visibility (approximation from mouth opening)
            features['teeth_visibility'] = min(1.0, features['mar'] / 0.05)
            
            # Lip compression using inner vs outer width ratio
            features['lip_compression'] = max(0.0, 1.0 - (inner_width / max(outer_width, 0.001)))
            
            # Mouth pursing (how much lips are pushed forward/inward)
            features['mouth_pursing'] = inner_width / max(outer_width, 0.001)
            
            # Enhanced mouth area calculations
            features['outer_mouth_area'] = outer_width * outer_height
            features['inner_mouth_area'] = inner_width * inner_height
            features['mouth_area_ratio'] = features['inner_mouth_area'] / max(features['outer_mouth_area'], 0.001)
            
            # Lip asymmetry analysis
            left_lip_height = np.linalg.norm(outer_top_left - outer_bottom_left)
            right_lip_height = np.linalg.norm(outer_top_right - outer_bottom_right)
            features['lip_asymmetry'] = abs(left_lip_height - right_lip_height) / max((left_lip_height + right_lip_height) * 0.5, 0.001)
            
            # Nose flare (limited detection from available landmarks)
            if len(landmarks) > 31:
                left_nostril = transform_point(landmarks[31])
                right_nostril = transform_point(landmarks[35])
                nostril_width = np.linalg.norm(right_nostril - left_nostril)
                features['nose_flare'] = min(1.0, nostril_width / max(outer_width, 0.001))
            else:
                features['nose_flare'] = 0.0
            
            return features
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return {}
    
    def _classify_viseme_basic(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Enhanced viseme classification for OpenSeeFace (adapted from MediaPipe logic)
        """
        if not features:
            return 'neutral', 0.0
        
        # Extract all sophisticated features
        mar = features.get('mar', 0)
        lip_ratio = features.get('lip_ratio', 0)
        roundedness = features.get('roundedness', 0)
        inner_opening = features.get('inner_opening', 0)
        lip_thickness = features.get('lip_thickness', 0)
        mouth_pursing = features.get('mouth_pursing', 0)
        top_lip_curvature = features.get('top_lip_curvature', 0)
        bottom_lip_curvature = features.get('bottom_lip_curvature', 0)
        mouth_area_ratio = features.get('mouth_area_ratio', 0)
        lip_asymmetry = features.get('lip_asymmetry', 0)
        
        # OpenSeeFace has different MAR scale (0.0-1.0), adjust thresholds accordingly
        # Convert to MediaPipe-like scale for consistent logic
        normalized_mar = mar * 0.3  # Scale down from OpenSeeFace range
        
        # A - Wide mouth opening with significant inner opening
        if (normalized_mar > 0.15 and inner_opening > 0.3 and 
            lip_ratio < 2.5 and mouth_area_ratio > 0.4):
            confidence = min(1.0, (normalized_mar + inner_opening) * 1.0)
            return 'A', confidence
        
        # O - Round mouth shape with high pursing and roundedness
        elif (roundedness > 0.6 and mouth_pursing < 0.8 and 
              normalized_mar > 0.05 and lip_ratio < 2.2 and
              top_lip_curvature > 0.3 and bottom_lip_curvature > 0.3):
            confidence = min(1.0, (roundedness + (1.0 - mouth_pursing)) * 0.7)
            return 'O', confidence
        
        # E - Wide, stretched mouth with minimal curvature
        elif (lip_ratio > 2.3 and normalized_mar < 0.12 and 
              top_lip_curvature < 0.5 and mouth_pursing > 0.7):
            confidence = min(1.0, lip_ratio / 3.5)
            return 'E', confidence
        
        # U - Pursed lips with high roundedness and small opening
        elif (mouth_pursing < 0.6 and roundedness > 0.7 and 
              normalized_mar < 0.10 and inner_opening < 0.4 and
              lip_thickness > 0.3):
            confidence = min(1.0, (roundedness + (1.0 - mouth_pursing)) * 0.8)
            return 'U', confidence
        
        # I - Small opening with moderate stretch
        elif (normalized_mar > 0.03 and normalized_mar < 0.08 and 
              lip_ratio > 1.8 and inner_opening < 0.5 and
              mouth_area_ratio > 0.2):
            confidence = min(1.0, normalized_mar * 5.0)
            return 'I', confidence
        
        # Neutral - closed or unclear mouth
        else:
            confidence = max(0.3, 1.0 - normalized_mar)
            return 'neutral', confidence
    
    def _detect_movement_segments(self, mar_values: List[float], fps: float) -> List[MovementSegment]:
        """
        Detect movement segments from MAR values using OpenSeeFace-optimized thresholds
        """
        segments = []
        
        if not mar_values or fps <= 0:
            return segments
        
        # Use the calculated threshold directly for OpenSeeFace
        threshold = self.mar_threshold
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
                        start_frame=movement_start,
                        end_frame=i,
                        confidence=confidence,
                        peak_mar=max(segment_mars)
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
                    start_frame=movement_start,
                    end_frame=len(mar_values),
                    confidence=confidence,
                    peak_mar=max(segment_mars)
                ))
        
        # Log MAR statistics for threshold optimization
        if mar_values:
            min_mar = min(mar_values)
            max_mar = max(mar_values)
            avg_mar = sum(mar_values) / len(mar_values)
            above_threshold = sum(1 for v in mar_values if v > threshold)
            below_threshold = len(mar_values) - above_threshold
            logger.info(f"OpenSeeFace MAR stats: min={min_mar:.3f}, max={max_mar:.3f}, avg={avg_mar:.3f}")
            logger.info(f"Threshold analysis: {threshold:.3f} -> {above_threshold} above, {below_threshold} below")
        
        logger.info(f"OpenSeeFace detected {len(segments)} movement segments")
        return segments
    
    def get_preview_video(self) -> Optional[str]:
        """Get the preview video file path if generated"""
        return getattr(self, 'preview_video', None)
    
    
    def _create_annotated_frame(self, frame: np.ndarray, faces: list, mar_value: float, viseme_frame, frame_number: int) -> np.ndarray:
        """Create annotated frame with OpenSeeFace tracking visualizations"""
        try:
            import cv2
            
            annotated = frame.copy()
            height, width = frame.shape[:2]
            
            # Create semi-transparent overlay for status
            overlay = annotated.copy()
            cv2.rectangle(overlay, (5, 5), (300, 105), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            
            # Status text
            status_text = f"OpenSeeFace | Frame: {frame_number}"
            cv2.putText(annotated, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw face detection and landmarks if available
            if len(faces) > 0:
                face = faces[0]
                
                # Get landmarks (try different attribute names)
                landmarks = None
                if hasattr(face, 'landmarks'):
                    landmarks = face.landmarks
                elif hasattr(face, 'lms'):
                    landmarks = face.lms
                elif hasattr(face, 'pts'):
                    landmarks = face.pts
                
                if landmarks is not None:
                    # Debug and fix OpenSeeFace landmark coordinate mapping
                    try:
                        landmarks = np.array(landmarks)
                        
                        # Debug: Log the actual landmark format
                        # OpenSeeFace landmarks successfully detected and processed
                        logger.debug(f"OpenSeeFace landmarks: {landmarks.shape}, Frame: {width}x{height}")
                        
                        # Convert format if needed
                        if landmarks.shape[0] == 2:  # If landmarks are in [2, N] format
                            landmarks = landmarks.T  # Convert to [N, 2]
                        
                        # Ensure we have at least 68 points for standard face model
                        if len(landmarks) >= 68:
                            # Draw mouth landmarks (48-67) with different colors for visibility
                            mouth_indices = list(range(48, 68))
                            
                            for i, idx in enumerate(mouth_indices):
                                if idx < len(landmarks):
                                    pt = landmarks[idx]
                                    if len(pt) >= 2:
                                        # Calculate direct pixel coordinates for OpenSeeFace
                                        x_direct = int(pt[0])
                                        y_direct = int(pt[1])
                                        
                                        # Try coordinate transformations to fix rotation
                                        # The landmarks appear to be rotated - try swapping X and Y
                                        x = y_direct  # Use Y as X (swap coordinates)
                                        y = x_direct  # Use X as Y (swap coordinates)
                                        
                                        # Log coordinate transformation (reduced logging)
                                        if i == 0:
                                            logger.debug(f"OpenSeeFace coordinate transformation: ({pt[0]:.1f}, {pt[1]:.1f}) -> ({x}, {y})")
                                        
                                        # Ensure coordinates are within bounds
                                        x = max(0, min(width-1, x))
                                        y = max(0, min(height-1, y))
                                        
                                        coord_method = "swapped_xy"
                                        
                                        if i == 0:  # Log coordinate method once
                                            logger.debug(f"Using {coord_method} coordinates: ({x}, {y}) from raw ({pt[0]:.2f}, {pt[1]:.2f})")
                                        
                                        # Color coding for mouth landmarks:
                                        if 48 <= idx <= 59:  # Outer lip contour
                                            color = (0, 255, 0)  # Green
                                        elif 60 <= idx <= 67:  # Inner lip contour
                                            color = (0, 0, 255)  # Red
                                        else:
                                            color = (255, 255, 255)  # White
                                        
                                        cv2.circle(annotated, (x, y), 3, color, -1)
                                        # Add point index for debugging
                                        cv2.putText(annotated, str(idx), (x+3, y-3), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            
                            # Individual landmarks provide sufficient visualization
                            # Bounding box removed for cleaner appearance
                    
                    except Exception as e:
                        logger.error(f"Landmark visualization failed: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Draw MAR value
            mar_text = f"MAR: {mar_value:.4f} ({'OPEN' if mar_value > self.mar_threshold else 'CLOSED'})"
            cv2.putText(annotated, mar_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw viseme information if available
            if viseme_frame is not None:
                # Handle different attribute names for viseme
                viseme_value = 'neutral'
                if hasattr(viseme_frame, 'viseme'):
                    viseme_value = viseme_frame.viseme
                elif hasattr(viseme_frame, 'detected_viseme'):
                    viseme_value = viseme_frame.detected_viseme
                
                viseme_text = f"Viseme: {viseme_value}"
                cv2.putText(annotated, viseme_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                if hasattr(viseme_frame, 'confidence'):
                    conf_text = f"Conf: {viseme_frame.confidence:.2f}"
                    cv2.putText(annotated, conf_text, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            return annotated
            
        except Exception as e:
            logger.warning(f"Failed to create annotated frame: {e}")
            return frame  # Return original frame if annotation fails