"""
Basic Viseme Classifier - Frame-by-frame geometric analysis
"""

import numpy as np
from typing import Dict, Tuple
from .abstract_viseme_classifier import AbstractVisemeClassifier, VisemeResult


class BasicVisemeClassifier(AbstractVisemeClassifier):
    """
    Basic frame-by-frame viseme classifier using geometric mouth features.
    
    Analyzes individual frames without temporal context for fast classification.
    Good for basic vowel detection and simple consonant identification.
    """
    
    def __init__(self, sensitivity: float = 1.0, confidence_threshold: float = 0.4, 
                 mar_threshold: float = 0.05):
        """
        Initialize basic classifier
        
        Args:
            sensitivity: Detection sensitivity (0.1-2.0)
            confidence_threshold: Minimum confidence for valid detection 
            mar_threshold: Mouth aspect ratio threshold for neutral detection
        """
        super().__init__(sensitivity, confidence_threshold)
        self.mar_threshold = mar_threshold
        self.previous_visemes = []  # For smoothing
    
    def supports_temporal_analysis(self) -> bool:
        """Basic classifier uses single frames"""
        return False
    
    def classify_viseme(self, features: Dict[str, float], enable_consonants: bool = False) -> VisemeResult:
        """
        Classify mouth shape into viseme categories based on geometric features
        
        Args:
            features: Dictionary of geometric features
            enable_consonants: Whether to detect consonants in addition to vowels
            
        Returns:
            VisemeResult with (viseme_label, confidence, metadata)
        """
        if not features:
            return VisemeResult('neutral', 0.0, {'reason': 'no_features'})
        
        # Extract basic features
        lip_ratio = features.get('lip_ratio', 0)
        roundedness = features.get('roundedness', 0)
        mouth_area = features.get('mouth_area', 0)
        mar = features.get('mar', 0)
        
        # Get consonant features
        lip_contact = features.get('lip_contact', 0)
        teeth_visibility = features.get('teeth_visibility', 0)
        lip_compression = features.get('lip_compression', 0)
        nose_flare = features.get('nose_flare', 0)
        
        # Check if mouth is open enough for vowel (complete closure = neutral)
        if mar < self.mar_threshold * 0.5:
            return VisemeResult('neutral', 0.8, {'reason': 'mouth_closed'})
        
        # Initialize scoring
        viseme_scores = {
            'A': 0.0, 'E': 0.0, 'I': 0.0, 'O': 0.0, 'U': 0.0, 'neutral': 0.0
        }
        
        # Add consonant scores if enabled
        if enable_consonants:
            consonant_scores = {
                'B': 0.0, 'P': 0.0, 'M': 0.0,  # Bilabial
                'F': 0.0, 'V': 0.0,             # Labiodental
                'TH': 0.0,                      # Dental
                'T': 0.0, 'D': 0.0, 'N': 0.0,  # Alveolar
                'K': 0.0, 'G': 0.0              # Velar
            }
            viseme_scores.update(consonant_scores)
        
        # Normalize mouth area
        normalized_area = min(1.0, mouth_area / 1000.0)
        sens_factor = self.sensitivity
        
        # VOWEL CLASSIFICATION (PRIMARY)
        self._classify_vowels(viseme_scores, mar, lip_ratio, roundedness, 
                             normalized_area, sens_factor)
        
        # CONSONANT CLASSIFICATION (SECONDARY)
        if enable_consonants:
            self._classify_consonants(viseme_scores, lip_contact, teeth_visibility,
                                    lip_compression, nose_flare, mar, roundedness, sens_factor)
        
        # Find best match
        best_viseme = max(viseme_scores.items(), key=lambda x: x[1])
        
        # Normalize confidence to 0-1 range
        confidence = min(1.0, best_viseme[1] / 2.0)
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            return VisemeResult('neutral', confidence, {
                'reason': 'low_confidence',
                'raw_confidence': confidence,
                'threshold': self.confidence_threshold
            })
        
        # Apply temporal smoothing if enabled
        smoothed_viseme, smoothed_confidence = self._apply_smoothing(best_viseme[0], confidence)
        
        return VisemeResult(smoothed_viseme, smoothed_confidence, {
            'method': 'basic_geometric',
            'raw_scores': {k: v for k, v in viseme_scores.items() if v > 0.1},
            'features_used': list(features.keys())
        })
    
    def _classify_vowels(self, scores: Dict[str, float], mar: float, lip_ratio: float,
                        roundedness: float, normalized_area: float, sens_factor: float):
        """Classify vowel visemes based on geometric features"""
        
        # A: Wide open mouth, high aperture
        if mar > (0.25 / sens_factor) and lip_ratio < (3.5 * sens_factor):
            scores['A'] = (mar / 0.5) * (1.0 + normalized_area) * sens_factor
        
        # E: Spread lips, moderate opening
        if (0.1 / sens_factor) < mar < (0.3 / sens_factor) and lip_ratio > (2.5 / sens_factor):
            scores['E'] = (1.0 - abs(mar - 0.2) * 5.0) * min(1.0, lip_ratio / 3.5) * sens_factor
        
        # I: Narrow vertical, wide horizontal (smile-like)
        if mar < (0.15 / sens_factor) and lip_ratio > (3.0 / sens_factor):
            scores['I'] = (1.0 - mar * 6.0) * min(1.0, lip_ratio / 4.0) * sens_factor
        
        # O: Rounded, moderate opening
        if (0.15 / sens_factor) < mar < (0.35 / sens_factor) and roundedness > (0.5 / sens_factor):
            scores['O'] = roundedness * (1.0 - abs(mar - 0.25) * 4.0) * sens_factor
        
        # U: Small rounded opening
        if mar < (0.2 / sens_factor) and roundedness > (0.6 / sens_factor):
            scores['U'] = roundedness * (1.0 - mar * 5.0) * sens_factor
    
    def _classify_consonants(self, scores: Dict[str, float], lip_contact: float,
                           teeth_visibility: float, lip_compression: float,
                           nose_flare: float, mar: float, roundedness: float, sens_factor: float):
        """Classify consonant visemes based on geometric features"""
        
        # Bilabial stops and nasals (lips clearly closed but achievable)
        if lip_contact > (0.92 / sens_factor):
            if nose_flare > (0.6 / sens_factor):  # Nasal
                scores['M'] = lip_contact * nose_flare * sens_factor
            elif lip_compression > (0.75 / sens_factor):  # Voiceless stop
                scores['P'] = lip_contact * lip_compression * sens_factor
            else:  # Voiced stop
                scores['B'] = lip_contact * sens_factor
        
        # Labiodental fricatives (teeth on lip)
        if teeth_visibility > (0.75 / sens_factor) and lip_contact > (0.6 / sens_factor):
            if lip_compression > (0.6 / sens_factor):
                scores['F'] = teeth_visibility * lip_compression * sens_factor
            else:
                scores['V'] = teeth_visibility * lip_contact * sens_factor
        
        # Dental fricatives (tongue visible between teeth)
        if teeth_visibility > (0.8 / sens_factor) and mar > (0.15 / sens_factor):
            scores['TH'] = teeth_visibility * mar * sens_factor
        
        # Alveolar and velar stops (compression patterns)
        if lip_compression > (0.8 / sens_factor) and lip_contact < (0.4 / sens_factor):
            if mar < (0.08 / sens_factor):
                scores['T'] = lip_compression * (1.0 - mar) * sens_factor
            elif nose_flare > (0.5 / sens_factor):
                scores['N'] = lip_compression * nose_flare * sens_factor
            else:
                scores['D'] = lip_compression * mar * sens_factor
        
        # Velar stops (back of tongue - harder to detect)
        if lip_compression > (0.7 / sens_factor) and roundedness < (0.2 / sens_factor):
            if mar < (0.05 / sens_factor):
                scores['K'] = lip_compression * (1.0 - roundedness) * sens_factor * 0.8
            else:
                scores['G'] = lip_compression * (1.0 - roundedness) * mar * sens_factor * 0.8
    
    def _apply_smoothing(self, viseme: str, confidence: float, smoothing_factor: float = 0.3) -> Tuple[str, float]:
        """Apply temporal smoothing to reduce flickering"""
        if smoothing_factor <= 0.0:
            return viseme, confidence
        
        # Add current detection to history
        self.previous_visemes.append((viseme, confidence))
        
        # Keep only recent history for smoothing
        history_length = max(1, int(5 * smoothing_factor))  # 1-5 frames
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
        
        return viseme, confidence
    
    def reset_smoothing(self):
        """Reset smoothing history (call between videos)"""
        self.previous_visemes = []