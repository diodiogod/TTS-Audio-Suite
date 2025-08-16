"""
Abstract base class for viseme classification systems
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class VisemeResult:
    """Result of viseme classification"""
    viseme: str
    confidence: float
    metadata: Dict = None


class AbstractVisemeClassifier(ABC):
    """
    Abstract base class for viseme classification systems.
    
    Provides common interface for both basic frame-by-frame classification
    and advanced temporal analysis systems.
    """
    
    def __init__(self, sensitivity: float = 1.0, confidence_threshold: float = 0.04):
        """
        Initialize classifier
        
        Args:
            sensitivity: Detection sensitivity (0.1-2.0)
            confidence_threshold: Minimum confidence for valid detection (0.0-1.0)
        """
        self.sensitivity = sensitivity
        self.confidence_threshold = confidence_threshold
    
    @abstractmethod
    def classify_viseme(self, features: Dict[str, float], enable_consonants: bool = False) -> VisemeResult:
        """
        Classify viseme from geometric features
        
        Args:
            features: Dictionary of geometric features from mouth landmarks
            enable_consonants: Whether to detect consonants in addition to vowels
            
        Returns:
            VisemeResult with classification and confidence
        """
        pass
    
    @abstractmethod
    def supports_temporal_analysis(self) -> bool:
        """Return True if this classifier uses temporal context"""
        pass
    
    def get_supported_visemes(self, include_consonants: bool = False) -> List[str]:
        """
        Get list of visemes this classifier can detect
        
        Args:
            include_consonants: Whether to include consonant visemes
            
        Returns:
            List of supported viseme labels
        """
        vowels = ['A', 'E', 'I', 'O', 'U', 'neutral']
        if include_consonants:
            consonants = ['B', 'P', 'M', 'F', 'V', 'TH', 'T', 'D', 'N', 'K', 'G']
            return vowels + consonants
        return vowels
    
    def update_sensitivity(self, sensitivity: float):
        """Update detection sensitivity"""
        self.sensitivity = max(0.1, min(2.0, sensitivity))
    
    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))