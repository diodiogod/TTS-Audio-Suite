"""
Phoneme-to-word matching utility for viseme sequences
Provides simple word prediction from detected phoneme patterns
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
import re

logger = logging.getLogger(__name__)

class PhonemeWordMatcher:
    """
    Simple phoneme-to-word matching for viseme sequences
    Uses pattern matching to suggest words from detected phonemes
    """
    
    def __init__(self, dictionary_path: Optional[str] = None):
        self.dictionary_path = dictionary_path or self._get_default_dictionary_path()
        self.word_list = []
        self.phoneme_patterns = {}
        self._load_dictionary()
        self._build_phoneme_patterns()
    
    def _get_default_dictionary_path(self) -> str:
        """Get path to default dictionary file"""
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Try CMU dictionary first (has phoneme data), fallback to word list
        cmu_path = os.path.join(current_dir, "data", "dictionaries", "cmudict.txt")
        if os.path.exists(cmu_path):
            return cmu_path
        
        moby_path = os.path.join(current_dir, "data", "dictionaries", "moby_words.txt")
        if os.path.exists(moby_path):
            return moby_path
            
        return os.path.join(current_dir, "data", "dictionaries", "common_words.txt")
    
    def _load_dictionary(self):
        """Load word list from dictionary file"""
        try:
            if os.path.exists(self.dictionary_path):
                self.word_list = []
                self.cmu_phonemes = {}  # word -> phoneme sequence
                
                with open(self.dictionary_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        # Check if this is CMU format (word PHONEME1 PHONEME2...)
                        if ' ' in line and any(c.isupper() for c in line):
                            parts = line.split()
                            if len(parts) > 1:
                                word = parts[0].lower()
                                # Remove variant markers like (2), (3)
                                if '(' in word:
                                    word = word.split('(')[0]
                                
                                phonemes = parts[1:]
                                self.word_list.append(word)
                                self.cmu_phonemes[word] = phonemes
                        else:
                            # Simple word list format
                            word = line.lower()
                            if word and word.isalpha():  # Only alphabetic words
                                self.word_list.append(word)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_words = []
                for word in self.word_list:
                    if word not in seen:
                        seen.add(word)
                        unique_words.append(word)
                self.word_list = unique_words
                
                logger.info(f"Loaded {len(self.word_list)} words from dictionary")
                if self.cmu_phonemes:
                    logger.info(f"Found CMU phoneme data for {len(self.cmu_phonemes)} words")
            else:
                logger.warning(f"Dictionary not found: {self.dictionary_path}")
                # Fallback to basic words
                self.word_list = [
                    "the", "and", "you", "that", "was", "for", "are", "with", "his", "they",
                    "have", "this", "from", "had", "she", "but", "not", "what", "all", "can"
                ]
                self.cmu_phonemes = {}
                logger.info(f"Using fallback dictionary with {len(self.word_list)} words")
        except Exception as e:
            logger.error(f"Error loading dictionary: {e}")
            self.word_list = ["the", "and", "you"]  # Minimal fallback
            self.cmu_phonemes = {}
    
    def _build_phoneme_patterns(self):
        """
        Build simplified phoneme patterns for common words
        Maps common vowel/consonant patterns to phonemes
        """
        # Simple vowel pattern mapping (very approximate)
        vowel_map = {
            'A': ['a', 'ah', 'ay'],      # cat, father, day
            'E': ['e', 'eh', 'ee'],      # bet, bed, see  
            'I': ['i', 'ih', 'eye'],     # bit, bid, I
            'O': ['o', 'oh', 'aw'],      # got, go, saw
            'U': ['u', 'uh', 'oo']       # but, book, too
        }
        
        # Simple consonant approximations (visible mouth shapes)
        consonant_map = {
            'B': ['b'],     'P': ['p'],     'M': ['m'],
            'F': ['f'],     'V': ['v'],     'TH': ['th'],
            'T': ['t'],     'D': ['d'],     'N': ['n'],
            'K': ['k', 'c'], 'G': ['g']
        }
        
        # Build patterns for each word
        words_to_process = self.word_list[:5000] if len(self.word_list) > 5000 else self.word_list  # Use more words now
        
        for word in words_to_process:
            # Use CMU phoneme data if available, otherwise fall back to spelling approximation
            if hasattr(self, 'cmu_phonemes') and word in self.cmu_phonemes:
                pattern = self._cmu_to_viseme_pattern(self.cmu_phonemes[word])
            else:
                pattern = self._word_to_phoneme_pattern(word)
            
            if pattern and len(pattern) > 0:
                if pattern not in self.phoneme_patterns:
                    self.phoneme_patterns[pattern] = []
                self.phoneme_patterns[pattern].append(word)
    
    def _cmu_to_viseme_pattern(self, cmu_phonemes: List[str]) -> str:
        """
        Convert CMU phonemes to viseme pattern
        Maps CMU phoneme notation to our viseme symbols
        """
        pattern = ""
        
        # CMU to viseme mapping
        cmu_to_viseme = {
            # Vowels
            'AA': 'A', 'AE': 'A', 'AH': 'A', 'AO': 'O', 'AW': 'A',  # A-like sounds
            'AY': 'A', 'EH': 'E', 'ER': 'E', 'EY': 'E', 'IH': 'I',  # E-like sounds  
            'IY': 'I', 'OW': 'O', 'OY': 'O', 'UH': 'U', 'UW': 'U',  # I, O, U sounds
            
            # Consonants (visually detectable)
            'B': 'B', 'P': 'P', 'M': 'M',           # Bilabial
            'F': 'F', 'V': 'V',                      # Labiodental  
            'TH': 'TH', 'DH': 'TH',                  # Dental
            'T': 'T', 'D': 'D', 'N': 'N',           # Alveolar
            'K': 'K', 'G': 'G', 'NG': 'N',          # Velar
            
            # Other consonants (less visually distinctive) - skip or approximate
            'S': '', 'Z': '', 'SH': '', 'ZH': '',    # Sibilants (hard to see)
            'CH': 'T', 'JH': 'D',                    # Affricates (approximate)
            'L': '', 'R': '', 'W': 'U', 'Y': 'I',   # Liquids/glides (approximate)
            'HH': '',                                 # Aspirant (not visible)
        }
        
        for phoneme in cmu_phonemes:
            # Remove stress markers (0, 1, 2)
            clean_phoneme = ''.join(c for c in phoneme if not c.isdigit())
            
            if clean_phoneme in cmu_to_viseme:
                viseme = cmu_to_viseme[clean_phoneme]
                if viseme:  # Only add non-empty visemes
                    pattern += viseme
        
        return pattern
    
    def _word_to_phoneme_pattern(self, word: str) -> str:
        """
        Convert word to simplified phoneme pattern
        Very rough approximation based on spelling
        """
        pattern = ""
        word = word.lower()
        i = 0
        
        while i < len(word):
            char = word[i]
            
            # Vowels
            if char in 'aeiou':
                if char == 'a':
                    # Simple heuristics
                    if i + 1 < len(word) and word[i + 1] in 'wy':
                        pattern += 'E'  # ay sound
                    else:
                        pattern += 'A'
                elif char == 'e':
                    if i == len(word) - 1:  # silent e
                        pass
                    elif i + 1 < len(word) and word[i + 1] == 'e':
                        pattern += 'E'  # ee sound
                        i += 1
                    else:
                        pattern += 'E'
                elif char == 'i':
                    pattern += 'I'
                elif char == 'o':
                    if i + 1 < len(word) and word[i + 1] == 'o':
                        pattern += 'U'  # oo sound
                        i += 1
                    else:
                        pattern += 'O'
                elif char == 'u':
                    pattern += 'U'
            
            # Consonants (only visually detectable ones)
            elif char in 'bpm':
                if char == 'b': pattern += 'B'
                elif char == 'p': pattern += 'P'
                elif char == 'm': pattern += 'M'
            elif char in 'fv':
                if char == 'f': pattern += 'F'
                elif char == 'v': pattern += 'V'
            elif char in 'td':
                if char == 't': pattern += 'T'
                elif char == 'd': pattern += 'D'
            elif char in 'kg':
                if char == 'k': pattern += 'K'
                elif char == 'g': pattern += 'G'
            elif char == 'n':
                pattern += 'N'
            elif word[i:i+2] == 'th':
                pattern += 'TH'
                i += 1
            # Skip other consonants (not easily detectable from mouth shape)
            
            i += 1
        
        return pattern
    
    def match_phonemes_to_words(self, phoneme_sequence: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """
        Match phoneme sequence to potential words
        
        Args:
            phoneme_sequence: String of phonemes like "AEIOU" or "B_A_T"
            max_suggestions: Maximum number of word suggestions
            
        Returns:
            List of (word, confidence) tuples
        """
        if not phoneme_sequence or phoneme_sequence.replace('_', '') == '':
            return []
        
        # Clean sequence (remove underscores for matching)
        clean_sequence = phoneme_sequence.replace('_', '')
        
        suggestions = []
        
        # Exact pattern matches
        if clean_sequence in self.phoneme_patterns:
            for word in self.phoneme_patterns[clean_sequence][:max_suggestions]:
                suggestions.append((word, 0.9))
        
        # Partial matches (subsequence matching)
        if len(suggestions) < max_suggestions:
            for pattern, words in self.phoneme_patterns.items():
                if len(pattern) >= 2:  # Only consider patterns of reasonable length
                    # Check if clean_sequence is a subsequence of pattern or vice versa
                    if self._is_subsequence(clean_sequence, pattern) or self._is_subsequence(pattern, clean_sequence):
                        confidence = self._calculate_similarity(clean_sequence, pattern)
                        if confidence > 0.3:  # Minimum similarity threshold
                            for word in words[:2]:  # Limit per pattern
                                if not any(w == word for w, _ in suggestions):
                                    suggestions.append((word, confidence))
                                    if len(suggestions) >= max_suggestions:
                                        break
                    if len(suggestions) >= max_suggestions:
                        break
        
        # Sort by confidence and return
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]
    
    def _is_subsequence(self, s: str, t: str) -> bool:
        """Check if s is a subsequence of t"""
        i = 0
        for char in t:
            if i < len(s) and s[i] == char:
                i += 1
        return i == len(s)
    
    def _calculate_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate similarity between two phoneme sequences"""
        if not seq1 or not seq2:
            return 0.0
        
        # Simple similarity based on common characters
        common = sum(1 for c in seq1 if c in seq2)
        max_len = max(len(seq1), len(seq2))
        
        # Bonus for subsequence relationships
        subseq_bonus = 0.2 if (self._is_subsequence(seq1, seq2) or self._is_subsequence(seq2, seq1)) else 0
        
        return (common / max_len) + subseq_bonus
    
    def get_word_suggestions_for_segment(self, viseme_sequence: str) -> List[str]:
        """
        Get word suggestions for a viseme sequence, formatted for SRT display
        
        Args:
            viseme_sequence: String like "AEIOU" or "B_A_T_"
            
        Returns:
            List of suggested words (max 1, only high confidence)
        """
        matches = self.match_phonemes_to_words(viseme_sequence, max_suggestions=3)
        
        # Use more lenient confidence for common patterns and prefer shorter, common words
        good_matches = []
        for word, confidence in matches:
            min_confidence = 0.4 if len(word) <= 4 else 0.6  # Lower threshold for short words
            if confidence > min_confidence and len(word) <= 8 and word.isalpha():
                # Strongly prefer shorter words and common words
                word_score = confidence + (0.3 if len(word) <= 3 else 0.1 if len(word) <= 4 else 0)
                good_matches.append((word, word_score))
        
        if good_matches:
            # Sort by score and return only the best match
            good_matches.sort(key=lambda x: x[1], reverse=True)
            best_word = good_matches[0][0]
            return [best_word]
        
        # If no good matches, try simplified patterns for common words
        clean_sequence = viseme_sequence.replace('_', '').strip()
        
        # Special handling for common repeated patterns
        if clean_sequence == 'I' or 'III' in clean_sequence:
            return ['it']  # Common "I" sound words
        elif clean_sequence == 'A' or 'AAA' in clean_sequence:
            return ['at']  # Common "A" sound words  
        elif clean_sequence == 'E' or 'EEE' in clean_sequence:
            return ['eh']  # Common "E" sound
        elif clean_sequence == 'O' or 'OOO' in clean_sequence:
            return ['oh']  # Common "O" sound
        elif clean_sequence == 'U' or 'UUU' in clean_sequence:
            return ['uh']  # Common "U" sound
        
        # For very short sequences, just return cleaned
        if len(clean_sequence) <= 2:
            return [clean_sequence]
        else:
            return [viseme_sequence]  # Show original with underscores


# Global instance for reuse
_phoneme_matcher = None

def get_phoneme_matcher() -> PhonemeWordMatcher:
    """Get global phoneme matcher instance"""
    global _phoneme_matcher
    if _phoneme_matcher is None:
        _phoneme_matcher = PhonemeWordMatcher()
    return _phoneme_matcher