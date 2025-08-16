"""
Factory for creating appropriate viseme analysis systems
"""

from typing import Dict, Any

try:
    from .abstract_viseme_classifier import AbstractVisemeClassifier
    from .basic_viseme_classifier import BasicVisemeClassifier
    from .temporal_consonant_analyzer import TemporalConsonantAnalyzer
except ImportError:
    # Fallback to absolute imports
    from abstract_viseme_classifier import AbstractVisemeClassifier
    from basic_viseme_classifier import BasicVisemeClassifier
    from temporal_consonant_analyzer import TemporalConsonantAnalyzer


class VisemeAnalysisFactory:
    """
    Factory for creating appropriate viseme analysis systems based on configuration.
    
    Chooses between basic frame-by-frame analysis and advanced temporal analysis
    based on user preferences and performance requirements.
    """
    
    @staticmethod
    def create_analyzer(config: Dict[str, Any]) -> AbstractVisemeClassifier:
        """
        Create appropriate viseme analyzer based on configuration
        
        Args:
            config: Configuration dictionary with analysis options
            
        Returns:
            Configured viseme analyzer instance
        """
        # Extract configuration options
        enable_consonants = config.get('enable_consonant_detection', False)
        enable_temporal = config.get('enable_temporal_analysis', False)
        sensitivity = config.get('viseme_sensitivity', 1.0)
        confidence_threshold = config.get('viseme_confidence_threshold', 0.04)
        
        # Choose analyzer type based on configuration
        if enable_consonants and enable_temporal:
            # Advanced temporal analysis for accurate consonant detection
            analyzer = TemporalConsonantAnalyzer(
                sensitivity=sensitivity,
                confidence_threshold=confidence_threshold,
                window_size=config.get('temporal_window_size', 5),
                burst_threshold=config.get('consonant_burst_threshold', 0.4)
            )
            print(f"[ANALYZER] Created TemporalConsonantAnalyzer (sensitivity={sensitivity}, window={config.get('temporal_window_size', 5)})")
        else:
            # Basic frame-by-frame analysis (faster, good for vowels)
            analyzer = BasicVisemeClassifier(
                sensitivity=sensitivity,
                confidence_threshold=confidence_threshold,
                mar_threshold=config.get('mar_threshold', 0.05)
            )
            print(f"[ANALYZER] Created BasicVisemeClassifier (sensitivity={sensitivity})")
        
        return analyzer
    
    @staticmethod
    def get_recommended_config(use_case: str) -> Dict[str, Any]:
        """
        Get recommended configuration for common use cases
        
        Args:
            use_case: One of 'fast', 'balanced', 'accurate', 'research'
            
        Returns:
            Recommended configuration dictionary
        """
        configs = {
            'fast': {
                'enable_consonant_detection': False,
                'enable_temporal_analysis': False,
                'viseme_sensitivity': 1.0,
                'viseme_confidence_threshold': 0.5,
                'description': 'Fast vowel-only detection, good for basic lip-sync'
            },
            
            'balanced': {
                'enable_consonant_detection': True,
                'enable_temporal_analysis': False,
                'viseme_sensitivity': 1.2,
                'viseme_confidence_threshold': 0.04,
                'description': 'Good balance of speed and accuracy with basic consonants'
            },
            
            'accurate': {
                'enable_consonant_detection': True,
                'enable_temporal_analysis': True,
                'viseme_sensitivity': 1.5,
                'viseme_confidence_threshold': 0.3,
                'temporal_window_size': 5,
                'consonant_burst_threshold': 0.4,
                'description': 'High accuracy with temporal consonant burst detection'
            },
            
            'research': {
                'enable_consonant_detection': True,
                'enable_temporal_analysis': True,
                'viseme_sensitivity': 2.0,
                'viseme_confidence_threshold': 0.2,
                'temporal_window_size': 7,
                'consonant_burst_threshold': 0.3,
                'description': 'Maximum sensitivity for research and detailed analysis'
            }
        }
        
        return configs.get(use_case, configs['balanced'])
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize configuration options
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Validated and sanitized configuration
        """
        validated = {}
        
        # Boolean options
        validated['enable_consonant_detection'] = bool(config.get('enable_consonant_detection', False))
        validated['enable_temporal_analysis'] = bool(config.get('enable_temporal_analysis', False))
        
        # Numeric options with bounds
        validated['viseme_sensitivity'] = max(0.1, min(2.0, float(config.get('viseme_sensitivity', 1.0))))
        validated['viseme_confidence_threshold'] = max(0.0, min(1.0, float(config.get('viseme_confidence_threshold', 0.04))))
        validated['viseme_smoothing'] = max(0.0, min(1.0, float(config.get('viseme_smoothing', 0.3))))
        
        # Temporal analysis options
        if validated['enable_temporal_analysis']:
            validated['temporal_window_size'] = max(3, min(7, int(config.get('temporal_window_size', 5))))
            validated['consonant_burst_threshold'] = max(0.1, min(1.0, float(config.get('consonant_burst_threshold', 0.4))))
        
        # Basic classifier options
        validated['mar_threshold'] = max(0.01, min(0.2, float(config.get('mar_threshold', 0.05))))
        
        return validated
    
    @staticmethod
    def get_performance_estimate(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate performance characteristics for given configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Performance estimate with processing time and accuracy info
        """
        base_time = 1.0  # Baseline processing time unit
        
        # Calculate relative processing time
        processing_time = base_time
        
        if config.get('enable_consonant_detection', False):
            processing_time *= 1.3  # ~30% overhead for consonant detection
        
        if config.get('enable_temporal_analysis', False):
            window_size = config.get('temporal_window_size', 5)
            processing_time *= (1.0 + window_size * 0.1)  # Scales with window size
        
        # Estimate accuracy characteristics
        vowel_accuracy = 'high'  # Vowels are generally well-detected
        
        consonant_accuracy = 'none'
        if config.get('enable_consonant_detection', False):
            if config.get('enable_temporal_analysis', False):
                consonant_accuracy = 'high'
            else:
                consonant_accuracy = 'medium'
        
        return {
            'relative_processing_time': round(processing_time, 2),
            'vowel_accuracy': vowel_accuracy,
            'consonant_accuracy': consonant_accuracy,
            'temporal_context': config.get('enable_temporal_analysis', False),
            'recommended_for': VisemeAnalysisFactory._get_use_case_recommendation(config)
        }
    
    @staticmethod
    def _get_use_case_recommendation(config: Dict[str, Any]) -> str:
        """Get recommended use case for configuration"""
        if not config.get('enable_consonant_detection', False):
            return 'Basic lip-sync, fast processing'
        elif not config.get('enable_temporal_analysis', False):
            return 'Good lip-sync with basic consonants'
        else:
            return 'High-quality phonetic analysis, research'