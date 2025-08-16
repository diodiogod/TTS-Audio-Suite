"""
OpenSeeFace Model Downloader
Downloads advanced OpenSeeFace models to ComfyUI models directory
"""

import os
import sys
from typing import Optional, Dict, Any, List
import logging

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils.downloads.unified_downloader import unified_downloader
    UNIFIED_DOWNLOADER_AVAILABLE = True
except ImportError:
    UNIFIED_DOWNLOADER_AVAILABLE = False

logger = logging.getLogger(__name__)


class OpenSeeFaceModelDownloader:
    """
    Manages downloading of OpenSeeFace models to organized ComfyUI models directory
    
    Model organization: models/TTS/OpenSeeFace/models/
    """
    
    # OpenSeeFace model definitions with GitHub raw URLs
    MODELS = {
        # Basic models (faster, lower quality)
        'lm_model0_opt.onnx': {
            'size_mb': 1.9,
            'description': 'Fastest model (lowest quality) - Bundled',
            'quality': 'basic',
            'speed': 'fastest',
            'bundled': True
        },
        'lm_model1_opt.onnx': {
            'size_mb': 4.8,
            'description': 'Fast model (basic quality)',
            'quality': 'basic',
            'speed': 'fast',
            'bundled': False
        },
        
        # Balanced models
        'lm_model2_opt.onnx': {
            'size_mb': 8.5,
            'description': 'Balanced model (good quality)',
            'quality': 'good',
            'speed': 'medium',
            'bundled': False
        },
        
        # High quality models
        'lm_model3_opt.onnx': {
            'size_mb': 13.5,
            'description': 'High quality model (slower) - Default',
            'quality': 'high',
            'speed': 'slow',
            'bundled': False
        },
        'lm_model4_opt.onnx': {
            'size_mb': 13.5,
            'description': 'Wink-optimized model',
            'quality': 'high',
            'speed': 'slow',
            'bundled': False
        },
        
        # Alternative models
        'lm_modelT_opt.onnx': {
            'size_mb': 13.4,
            'description': 'Temporal model variant',
            'quality': 'high',
            'speed': 'slow',
            'bundled': False
        },
        'lm_modelU_opt.onnx': {
            'size_mb': 4.8,
            'description': 'Ultra-fast variant',
            'quality': 'basic',
            'speed': 'ultra_fast',
            'bundled': False
        },
        'lm_modelV_opt.onnx': {
            'size_mb': 13.5,
            'description': 'Video-optimized model',
            'quality': 'high',
            'speed': 'slow',
            'bundled': False
        },
        
        # Detection and auxiliary models
        'mnv3_detection_opt.onnx': {
            'size_mb': 0.6,
            'description': 'Face detection model - Bundled',
            'quality': 'standard',
            'speed': 'fast',
            'bundled': True
        },
        'retinaface_640x640_opt.onnx': {
            'size_mb': 1.7,
            'description': 'RetinaFace detection (robust)',
            'quality': 'high',
            'speed': 'medium',
            'bundled': False
        },
        'mnv3_gaze32_split_opt.onnx': {
            'size_mb': 3.9,
            'description': 'Gaze tracking model',
            'quality': 'standard',
            'speed': 'medium',
            'bundled': False
        },
        
        # Configuration files
        'priorbox_640x640.json': {
            'size_mb': 1.3,
            'description': 'Detection configuration - Bundled',
            'quality': 'standard',
            'speed': 'n/a',
            'bundled': True
        }
    }
    
    # GitHub raw URLs for OpenSeeFace models
    BASE_URL = "https://github.com/emilianavt/OpenSeeFace/raw/master/models/"
    
    def __init__(self):
        """Initialize OpenSeeFace model downloader"""
        if not UNIFIED_DOWNLOADER_AVAILABLE:
            logger.warning("Unified downloader not available, falling back to basic downloads")
        
        # Get bundled models directory
        self.bundled_models_dir = os.path.join(
            os.path.dirname(__file__), "models"
        )
        
        # Get organized models directory in ComfyUI
        if UNIFIED_DOWNLOADER_AVAILABLE:
            self.organized_models_dir = unified_downloader.get_organized_path("OpenSeeFace", "models")
        else:
            # Fallback path
            import folder_paths
            self.organized_models_dir = os.path.join(
                folder_paths.models_dir, "TTS", "OpenSeeFace", "models"
            )
    
    def get_model_path(self, model_filename: str) -> Optional[str]:
        """
        Get the path to a model, checking both bundled and downloaded locations
        
        Args:
            model_filename: Name of the model file (e.g., 'lm_model3_opt.onnx')
            
        Returns:
            Full path to model if found, None otherwise
        """
        # Check bundled location first
        bundled_path = os.path.join(self.bundled_models_dir, model_filename)
        if os.path.exists(bundled_path):
            return bundled_path
        
        # Check organized download location
        organized_path = os.path.join(self.organized_models_dir, model_filename)
        if os.path.exists(organized_path):
            return organized_path
        
        return None
    
    def is_model_available(self, model_filename: str) -> bool:
        """Check if a model is available (bundled or downloaded)"""
        return self.get_model_path(model_filename) is not None
    
    def download_model(self, model_filename: str) -> Optional[str]:
        """
        Download a specific OpenSeeFace model
        
        Args:
            model_filename: Name of the model file to download
            
        Returns:
            Path to downloaded model if successful, None otherwise
        """
        if model_filename not in self.MODELS:
            logger.error(f"Unknown OpenSeeFace model: {model_filename}")
            return None
        
        model_info = self.MODELS[model_filename]
        
        # Check if already available
        existing_path = self.get_model_path(model_filename)
        if existing_path:
            logger.info(f"OpenSeeFace model already available: {model_filename}")
            return existing_path
        
        if model_info['bundled']:
            logger.warning(f"Model {model_filename} should be bundled but not found")
            return None
        
        # Download using unified system
        if UNIFIED_DOWNLOADER_AVAILABLE:
            url = f"{self.BASE_URL}{model_filename}"
            target_path = os.path.join(self.organized_models_dir, model_filename)
            
            logger.info(f"Downloading OpenSeeFace model: {model_filename} ({model_info['size_mb']:.1f}MB)")
            success = unified_downloader.download_file(
                url=url,
                target_path=target_path,
                description=f"OpenSeeFace {model_info['description']}"
            )
            
            if success:
                return target_path
            else:
                logger.error(f"Failed to download OpenSeeFace model: {model_filename}")
                return None
        else:
            logger.error("Unified downloader not available, cannot download models")
            return None
    
    def download_recommended_models(self) -> Dict[str, Optional[str]]:
        """
        Download recommended models for typical usage
        
        Returns:
            Dictionary mapping model names to their paths (or None if failed)
        """
        recommended_models = [
            'lm_model3_opt.onnx',  # High quality default
            'retinaface_640x640_opt.onnx',  # Robust detection
            'lm_model1_opt.onnx'  # Fast alternative
        ]
        
        results = {}
        for model in recommended_models:
            logger.info(f"Downloading recommended OpenSeeFace model: {model}")
            path = self.download_model(model)
            results[model] = path
        
        return results
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available models with their status
        
        Returns:
            Dictionary with model info including availability status
        """
        model_status = {}
        
        for model_name, model_info in self.MODELS.items():
            status = {
                **model_info,
                'available': self.is_model_available(model_name),
                'path': self.get_model_path(model_name)
            }
            model_status[model_name] = status
        
        return model_status
    
    def get_model_recommendations(self, use_case: str = "balanced") -> List[str]:
        """
        Get model recommendations for different use cases
        
        Args:
            use_case: One of 'fast', 'balanced', 'quality', 'research'
            
        Returns:
            List of recommended model filenames
        """
        recommendations = {
            'fast': [
                'lm_model0_opt.onnx',  # Bundled fast option
                'lm_model1_opt.onnx',  # Better fast option
                'mnv3_detection_opt.onnx'  # Bundled detection
            ],
            
            'balanced': [
                'lm_model2_opt.onnx',  # Good quality/speed balance
                'lm_model3_opt.onnx',  # High quality default
                'retinaface_640x640_opt.onnx'  # Robust detection
            ],
            
            'quality': [
                'lm_model3_opt.onnx',  # High quality
                'lm_model4_opt.onnx',  # Wink optimized
                'lm_modelT_opt.onnx',  # Temporal variant
                'retinaface_640x640_opt.onnx',  # Best detection
                'mnv3_gaze32_split_opt.onnx'  # Gaze tracking
            ],
            
            'research': [
                'lm_model3_opt.onnx',  # Primary model
                'lm_model4_opt.onnx',  # Wink optimized
                'lm_modelT_opt.onnx',  # Temporal
                'lm_modelV_opt.onnx',  # Video optimized
                'retinaface_640x640_opt.onnx',  # Advanced detection
                'mnv3_gaze32_split_opt.onnx'  # Gaze analysis
            ]
        }
        
        return recommendations.get(use_case, recommendations['balanced'])
    
    def ensure_basic_models(self) -> bool:
        """
        Ensure basic models are available for out-of-box functionality
        
        Returns:
            True if basic models are available, False otherwise
        """
        basic_models = ['lm_model0_opt.onnx', 'mnv3_detection_opt.onnx', 'priorbox_640x640.json']
        
        for model in basic_models:
            if not self.is_model_available(model):
                logger.error(f"Basic OpenSeeFace model missing: {model}")
                return False
        
        return True


# Global instance for easy access
openseeface_downloader = OpenSeeFaceModelDownloader()