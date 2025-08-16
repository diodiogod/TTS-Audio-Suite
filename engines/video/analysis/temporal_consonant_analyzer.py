"""
Temporal Consonant Analyzer - Advanced 5-frame window analysis for true phonetic detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass

try:
    from .basic_viseme_classifier import BasicVisemeClassifier, VisemeResult
except ImportError:
    from basic_viseme_classifier import BasicVisemeClassifier, VisemeResult


@dataclass
class FrameFeatures:
    """Features for a single frame with temporal context"""
    features: Dict[str, float]
    frame_index: int
    timestamp: float


@dataclass
class ConsonantBurst:
    """Detected consonant burst pattern"""
    consonant_type: str
    onset_frame: int
    peak_frame: int
    release_frame: int
    confidence: float
    burst_strength: float


class TemporalConsonantAnalyzer(BasicVisemeClassifier):
    """
    Advanced temporal consonant analyzer using 5-frame sliding windows.
    
    Detects true consonant bursts by analyzing:
    - Onset patterns (rapid feature changes)
    - Hold phases (brief stable periods) 
    - Release patterns (rapid return to vowel)
    - Coarticulation context (surrounding vowels)
    
    Provides dramatically more accurate consonant detection than frame-by-frame.
    """
    
    def __init__(self, sensitivity: float = 1.0, confidence_threshold: float = 0.04,
                 window_size: int = 5, burst_threshold: float = 0.4):
        """
        Initialize temporal analyzer
        
        Args:
            sensitivity: Detection sensitivity (0.1-2.0)
            confidence_threshold: Minimum confidence for valid detection
            window_size: Number of frames to analyze for temporal patterns (3-7)
            burst_threshold: Minimum burst strength for consonant detection
        """
        super().__init__(sensitivity, confidence_threshold)
        self.window_size = max(3, min(7, window_size))  # 3-7 frame window
        self.burst_threshold = burst_threshold
        
        # Temporal analysis state
        self.frame_history: deque = deque(maxlen=self.window_size)
        self.detected_bursts: List[ConsonantBurst] = []
        self.vowel_baseline: Dict[str, float] = {}
    
    def supports_temporal_analysis(self) -> bool:
        """Temporal analyzer uses multi-frame context"""
        return True
    
    def classify_viseme(self, features: Dict[str, float], enable_consonants: bool = False) -> VisemeResult:
        """
        Classify viseme using temporal context analysis
        
        Args:
            features: Current frame geometric features
            enable_consonants: Whether to perform consonant burst detection
            
        Returns:
            VisemeResult with temporal analysis results
        """
        if not features:
            return VisemeResult('neutral', 0.0, {'reason': 'no_features'})
        
        # Add current frame to history
        current_frame = FrameFeatures(
            features=features,
            frame_index=len(self.frame_history),
            timestamp=0.0  # Will be set by caller if needed
        )
        self.frame_history.append(current_frame)
        
        # If we don't have enough frames yet, use basic classification
        if len(self.frame_history) < self.window_size:
            return super().classify_viseme(features, enable_consonants)
        
        # NEW APPROACH: Additive consonant+vowel detection
        consonant_detected = None
        
        if enable_consonants:
            # First check for direct consonant bursts
            consonant_result = self._analyze_consonant_bursts()
            if consonant_result:
                consonant_detected = consonant_result.viseme
            else:
                # Retroactive burst detection - check if current vowel suggests recent consonant
                retroactive_result = self._analyze_retroactive_bursts()
                if retroactive_result:
                    consonant_detected = retroactive_result.viseme
        
        # Always analyze vowel patterns (don't skip vowels!)
        vowel_result = self._analyze_vowel_patterns()
        
        # Combine consonant + vowel if both detected
        if consonant_detected and vowel_result.viseme != 'neutral':
            combined_viseme = consonant_detected + vowel_result.viseme
            combined_confidence = min(0.8, (vowel_result.confidence + 0.6) / 2)  # Average with consonant confidence
            
            return VisemeResult(
                combined_viseme,
                combined_confidence,
                {
                    'method': 'additive_consonant_vowel',
                    'consonant': consonant_detected,
                    'vowel': vowel_result.viseme,
                    'consonant_source': 'retroactive_burst' if consonant_detected else None,
                    'vowel_confidence': vowel_result.confidence,
                    'raw_scores': {consonant_detected: 0.6, vowel_result.viseme: vowel_result.confidence}  # Show detected components
                }
            )
        
        # Return vowel if no consonant detected
        return vowel_result
    
    def _analyze_consonant_bursts(self) -> Optional[VisemeResult]:
        """
        Analyze frame history for consonant burst patterns
        
        Returns:
            VisemeResult if consonant burst detected, None otherwise
        """
        frames = list(self.frame_history)
        center_idx = len(frames) // 2  # Focus on center frame
        
        # Extract feature trajectories
        trajectories = self._extract_feature_trajectories(frames)
        
        # Detect different types of consonant bursts
        burst_candidates = []
        
        # Bilabial bursts (B, P, M)
        bilabial_burst = self._detect_bilabial_burst(trajectories, center_idx)
        if bilabial_burst:
            burst_candidates.append(bilabial_burst)
        
        # Labiodental bursts (F, V)  
        labiodental_burst = self._detect_labiodental_burst(trajectories, center_idx)
        if labiodental_burst:
            burst_candidates.append(labiodental_burst)
        
        # Dental/alveolar bursts (TH, T, D, N)
        dental_burst = self._detect_dental_burst(trajectories, center_idx)
        if dental_burst:
            burst_candidates.append(dental_burst)
        
        # Select best burst candidate
        if burst_candidates:
            best_burst = max(burst_candidates, key=lambda b: b.confidence * b.burst_strength)
            
            if best_burst.confidence > self.confidence_threshold:
                return VisemeResult(
                    best_burst.consonant_type,
                    best_burst.confidence,
                    {
                        'method': 'temporal_burst_analysis',
                        'burst_strength': best_burst.burst_strength,
                        'onset_frame': best_burst.onset_frame,
                        'peak_frame': best_burst.peak_frame,
                        'release_frame': best_burst.release_frame,
                        'window_size': self.window_size
                    }
                )
        
        return None
    
    def _detect_bilabial_burst(self, trajectories: Dict[str, List[float]], center_idx: int) -> Optional[ConsonantBurst]:
        """Detect B/P/M consonant bursts (lip closure → release pattern)"""
        lip_contact = trajectories.get('lip_contact', [0] * self.window_size)
        nose_flare = trajectories.get('nose_flare', [0] * self.window_size)
        lip_compression = trajectories.get('lip_compression', [0] * self.window_size)
        
        # Look for rapid increase → peak → rapid decrease in lip_contact
        burst_pattern = self._analyze_burst_pattern(lip_contact, min_peak=0.8 / self.sensitivity)
        
        if burst_pattern and burst_pattern['peak_frame'] == center_idx:
            # Determine B vs P vs M based on secondary features
            peak_nose = nose_flare[center_idx] if center_idx < len(nose_flare) else 0
            peak_compression = lip_compression[center_idx] if center_idx < len(lip_compression) else 0
            
            # Classification logic
            if peak_nose > (0.6 / self.sensitivity):
                consonant_type = 'M'  # Nasal
                type_confidence = peak_nose
            elif peak_compression > (0.75 / self.sensitivity):
                consonant_type = 'P'  # Voiceless stop (more compression)
                type_confidence = peak_compression
            else:
                consonant_type = 'B'  # Voiced stop
                type_confidence = 0.8
            
            return ConsonantBurst(
                consonant_type=consonant_type,
                onset_frame=burst_pattern['onset_frame'],
                peak_frame=burst_pattern['peak_frame'],
                release_frame=burst_pattern['release_frame'],
                confidence=burst_pattern['confidence'] * type_confidence,
                burst_strength=burst_pattern['burst_strength']
            )
        
        return None
    
    def _detect_labiodental_burst(self, trajectories: Dict[str, List[float]], center_idx: int) -> Optional[ConsonantBurst]:
        """Detect F/V consonant bursts (teeth contact → release pattern)"""
        teeth_visibility = trajectories.get('teeth_visibility', [0] * self.window_size)
        lip_contact = trajectories.get('lip_contact', [0] * self.window_size)
        lip_compression = trajectories.get('lip_compression', [0] * self.window_size)
        
        # Look for teeth-on-lip pattern
        teeth_pattern = self._analyze_burst_pattern(teeth_visibility, min_peak=0.6 / self.sensitivity)
        lip_pattern = self._analyze_burst_pattern(lip_contact, min_peak=0.5 / self.sensitivity)
        
        # Both patterns should align
        if (teeth_pattern and lip_pattern and 
            abs(teeth_pattern['peak_frame'] - lip_pattern['peak_frame']) <= 1 and
            teeth_pattern['peak_frame'] == center_idx):
            
            peak_compression = lip_compression[center_idx] if center_idx < len(lip_compression) else 0
            
            # F vs V distinction
            if peak_compression > (0.6 / self.sensitivity):
                consonant_type = 'F'  # Voiceless (more compression)
                type_confidence = peak_compression
            else:
                consonant_type = 'V'  # Voiced
                type_confidence = 0.7
            
            combined_confidence = (teeth_pattern['confidence'] + lip_pattern['confidence']) / 2
            
            return ConsonantBurst(
                consonant_type=consonant_type,
                onset_frame=min(teeth_pattern['onset_frame'], lip_pattern['onset_frame']),
                peak_frame=center_idx,
                release_frame=max(teeth_pattern['release_frame'], lip_pattern['release_frame']),
                confidence=combined_confidence * type_confidence,
                burst_strength=(teeth_pattern['burst_strength'] + lip_pattern['burst_strength']) / 2
            )
        
        return None
    
    def _detect_dental_burst(self, trajectories: Dict[str, List[float]], center_idx: int) -> Optional[ConsonantBurst]:
        """Detect TH/T/D/N consonant bursts (compression patterns)"""
        lip_compression = trajectories.get('lip_compression', [0] * self.window_size)
        teeth_visibility = trajectories.get('teeth_visibility', [0] * self.window_size)
        nose_flare = trajectories.get('nose_flare', [0] * self.window_size)
        mar = trajectories.get('mar', [0] * self.window_size)
        
        # Look for compression burst
        compression_pattern = self._analyze_burst_pattern(lip_compression, min_peak=0.7 / self.sensitivity)
        
        if compression_pattern and compression_pattern['peak_frame'] == center_idx:
            peak_teeth = teeth_visibility[center_idx] if center_idx < len(teeth_visibility) else 0
            peak_nose = nose_flare[center_idx] if center_idx < len(nose_flare) else 0
            peak_mar = mar[center_idx] if center_idx < len(mar) else 0
            
            # Classification logic
            if peak_teeth > (0.8 / self.sensitivity) and peak_mar > (0.15 / self.sensitivity):
                consonant_type = 'TH'  # Dental fricative
                type_confidence = peak_teeth
            elif peak_nose > (0.5 / self.sensitivity):
                consonant_type = 'N'   # Nasal
                type_confidence = peak_nose
            elif peak_mar < (0.08 / self.sensitivity):
                consonant_type = 'T'   # Voiceless stop (tight closure)
                type_confidence = 1.0 - peak_mar
            else:
                consonant_type = 'D'   # Voiced stop
                type_confidence = peak_mar
            
            return ConsonantBurst(
                consonant_type=consonant_type,
                onset_frame=compression_pattern['onset_frame'],
                peak_frame=compression_pattern['peak_frame'],
                release_frame=compression_pattern['release_frame'],
                confidence=compression_pattern['confidence'] * type_confidence,
                burst_strength=compression_pattern['burst_strength']
            )
        
        return None
    
    def _analyze_burst_pattern(self, feature_trajectory: List[float], min_peak: float) -> Optional[Dict]:
        """
        Analyze feature trajectory for consonant burst pattern
        
        Looks for: low → rapid rise → peak → rapid fall → low
        
        Args:
            feature_trajectory: List of feature values over time
            min_peak: Minimum peak value to consider
            
        Returns:
            Dict with burst pattern info or None
        """
        if len(feature_trajectory) < 3:
            return None
        
        # Find peak
        peak_idx = np.argmax(feature_trajectory)
        peak_value = feature_trajectory[peak_idx]
        
        if peak_value < min_peak:
            return None
        
        # Analyze onset (before peak)
        onset_idx = 0
        for i in range(peak_idx - 1, -1, -1):
            if feature_trajectory[i] < peak_value * 0.3:  # 30% of peak
                onset_idx = i
                break
        
        # Analyze release (after peak)
        release_idx = len(feature_trajectory) - 1
        for i in range(peak_idx + 1, len(feature_trajectory)):
            if feature_trajectory[i] < peak_value * 0.3:  # 30% of peak
                release_idx = i
                break
        
        # Calculate burst characteristics
        onset_speed = (peak_value - feature_trajectory[onset_idx]) / max(1, peak_idx - onset_idx)
        release_speed = (peak_value - feature_trajectory[release_idx]) / max(1, release_idx - peak_idx)
        burst_duration = release_idx - onset_idx
        
        # Consonants should be brief (1-3 frames) with rapid onset/release
        if burst_duration <= 3 and onset_speed > 0.2 and release_speed > 0.2:
            burst_strength = (onset_speed + release_speed) / 2
            confidence = min(1.0, peak_value * burst_strength)
            
            return {
                'onset_frame': onset_idx,
                'peak_frame': peak_idx,
                'release_frame': release_idx,
                'burst_strength': burst_strength,
                'confidence': confidence,
                'duration': burst_duration
            }
        
        return None
    
    def _extract_feature_trajectories(self, frames: List[FrameFeatures]) -> Dict[str, List[float]]:
        """Extract feature trajectories across the frame window"""
        trajectories = {}
        
        # Get all unique feature names
        all_features = set()
        for frame in frames:
            all_features.update(frame.features.keys())
        
        # Extract trajectory for each feature
        for feature_name in all_features:
            trajectory = []
            for frame in frames:
                value = frame.features.get(feature_name, 0.0)
                trajectory.append(value)
            trajectories[feature_name] = trajectory
        
        return trajectories
    
    def _analyze_retroactive_bursts(self) -> Optional[VisemeResult]:
        """
        Retroactive consonant detection: Detect vowels that suggest recent consonant bursts
        
        Logic: If current frame shows vowel + high LipContact, look backward for closure
        Example: Detect P in "Papa" when seeing A with high LipContact
        
        Returns:
            VisemeResult if retroactive consonant detected, None otherwise
        """
        frames = list(self.frame_history)
        if len(frames) < 3:  # Need at least 3 frames for retroactive analysis
            return None
            
        current_frame = frames[-1]  # Current frame
        prev_frames = frames[-3:-1]  # Look back 2 frames
        
        # Get current frame features
        current_features = current_frame.features
        mar = current_features.get('mar', 0)
        lip_contact = current_features.get('lip_contact', 0)
        roundedness = current_features.get('roundedness', 0)
        lip_ratio = current_features.get('lip_ratio', 0)
        
        # Check if current frame is a vowel with consonant evidence
        is_vowel = mar > 0.1  # Mouth is open (vowel)
        
        # Different thresholds for different consonant types
        has_bilabial_evidence = lip_contact > 0.7  # P, B, M (high closure)
        has_fricative_evidence = lip_contact > 0.4 and current_features.get('teeth_visibility', 0) > 0.4  # F, V, TH (teeth+lip)
        
        if not (is_vowel and (has_bilabial_evidence or has_fricative_evidence)):
            return None
            
        # Look backward for the actual closure moment
        for i, prev_frame in enumerate(reversed(prev_frames)):
            prev_features = prev_frame.features
            prev_mar = prev_features.get('mar', 0)
            prev_lip_contact = prev_features.get('lip_contact', 0)
            prev_lip_compression = prev_features.get('lip_compression', 0)
            prev_nose_flare = prev_features.get('nose_flare', 0)
            
            # Check for different consonant patterns
            was_bilabial_closed = prev_mar < 0.05 and prev_lip_contact > 0.9  # P, B, M
            was_fricative_active = prev_features.get('teeth_visibility', 0) > 0.5 and prev_lip_contact > 0.3  # F, V, TH
            
            if was_bilabial_closed or was_fricative_active:
                # Determine consonant type based on characteristics
                consonant_type = 'P'  # Default
                confidence = 0.6
                
                if was_fricative_active:
                    # Fricative consonants (F, V, TH)
                    teeth_vis = prev_features.get('teeth_visibility', 0)
                    if teeth_vis > 0.7:
                        consonant_type = 'F'  # Strong teeth visibility = F
                        confidence = 0.7
                    elif teeth_vis > 0.5:
                        consonant_type = 'TH'  # Moderate teeth = TH
                        confidence = 0.6
                    else:
                        consonant_type = 'V'  # Lower teeth = V
                        confidence = 0.5
                elif was_bilabial_closed:
                    # Bilabial consonants (P, B, M)
                    if prev_nose_flare > 0.6:
                        consonant_type = 'M'  # Nasal
                        confidence = 0.7
                    elif prev_lip_compression > 0.8:
                        consonant_type = 'P'  # Voiceless stop
                        confidence = 0.8
                    else:
                        consonant_type = 'B'  # Voiced stop
                        confidence = 0.6
                
                # Return the detected consonant
                return VisemeResult(
                    consonant_type,
                    confidence,
                    {
                        'method': 'retroactive_burst_detection',
                        'closure_frame_offset': -(i + 1),  # How many frames back
                        'current_vowel_evidence': {
                            'mar': mar,
                            'lip_contact': lip_contact,
                            'roundedness': roundedness
                        },
                        'closure_evidence': {
                            'prev_mar': prev_mar,
                            'prev_lip_contact': prev_lip_contact,
                            'prev_lip_compression': prev_lip_compression,
                            'prev_nose_flare': prev_nose_flare
                        }
                    }
                )
                
        return None
    
    def _analyze_vowel_patterns(self) -> VisemeResult:
        """Analyze vowel patterns with temporal smoothing"""
        # Use the most recent frame but with enhanced smoothing
        current_features = self.frame_history[-1].features
        
        # Get basic vowel classification
        basic_result = super().classify_viseme(current_features, enable_consonants=False)
        
        # Enhance with temporal context
        vowel_stability = self._calculate_vowel_stability()
        enhanced_confidence = basic_result.confidence * vowel_stability
        
        return VisemeResult(
            basic_result.viseme,
            enhanced_confidence,
            {
                'method': 'temporal_vowel_analysis',
                'stability': vowel_stability,
                'basic_confidence': basic_result.confidence,
                'window_size': self.window_size
            }
        )
    
    def _calculate_vowel_stability(self) -> float:
        """Calculate how stable vowel features are across the window"""
        if len(self.frame_history) < 2:
            return 1.0
        
        # Check stability of key vowel features
        features_to_check = ['mar', 'lip_ratio', 'roundedness']
        stability_scores = []
        
        for feature_name in features_to_check:
            values = [frame.features.get(feature_name, 0) for frame in self.frame_history]
            if values:
                std_dev = np.std(values)
                mean_val = np.mean(values)
                # Stability = 1 - (coefficient of variation)
                stability = 1.0 - (std_dev / max(mean_val, 0.1))
                stability_scores.append(max(0.0, stability))
        
        return np.mean(stability_scores) if stability_scores else 1.0
    
    def reset_temporal_state(self):
        """Reset temporal analysis state (call between videos)"""
        self.frame_history.clear()
        self.detected_bursts.clear()
        self.vowel_baseline.clear()
        self.reset_smoothing()