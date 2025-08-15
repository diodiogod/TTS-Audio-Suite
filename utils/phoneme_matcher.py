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
        
        # Build patterns prioritizing common words first
        # Use Google's common words first, then CMU dictionary
        if hasattr(self, 'cmu_phonemes') and len(self.cmu_phonemes) > 1000:
            # If we have CMU data, prioritize common English words
            common_words = [
                'the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they',
                'have', 'this', 'from', 'had', 'she', 'but', 'not', 'what', 'all', 'can',
                'he', 'we', 'me', 'be', 'it', 'is', 'at', 'to', 'so', 'do', 'go', 'no',
                'my', 'by', 'up', 'if', 'an', 'as', 'or', 'on', 'in', 'of', 'a', 'i',
                'hello', 'world', 'yes', 'way', 'day', 'say', 'may', 'see', 'get', 'let'
            ]
            # Filter common words that exist in our dictionary
            priority_words = [w for w in common_words if w in self.word_list]
            # Add short words (3-5 chars) as high priority since they're often important
            remaining_words = [w for w in self.word_list if w not in priority_words]
            short_words = [w for w in remaining_words if 3 <= len(w) <= 5]
            longer_words = [w for w in remaining_words if len(w) > 5 or len(w) < 3]
            # Process: common words + short words + first 1500 longer words
            words_to_process = priority_words + short_words + longer_words[:1500]
        else:
            words_to_process = self.word_list[:3000] if len(self.word_list) > 3000 else self.word_list
        
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
        
        # Keep underscores for wildcard matching, but also create clean version
        wildcard_sequence = phoneme_sequence  # Keep _ as wildcards
        clean_sequence = phoneme_sequence.replace('_', '')  # Remove for exact matching
        
        suggestions = []
        
        # Exact pattern matches (no wildcards)
        if clean_sequence in self.phoneme_patterns:
            for word in self.phoneme_patterns[clean_sequence][:max_suggestions]:
                suggestions.append((word, 0.9))
        
        # Wildcard matching (if sequence contains underscores)
        if '_' in wildcard_sequence and len(suggestions) < max_suggestions:
            # Standard wildcard matching (exact length)
            for pattern, words in self.phoneme_patterns.items():
                if self._wildcard_match(wildcard_sequence, pattern):
                    confidence = self._calculate_wildcard_similarity(wildcard_sequence, pattern)
                    if confidence > 0.4:  # Higher threshold for wildcard matches
                        for word in words[:2]:
                            if not any(w == word for w, _ in suggestions):
                                suggestions.append((word, confidence))
                                if len(suggestions) >= max_suggestions:
                                    break
                if len(suggestions) >= max_suggestions:
                    break
            
            # Flexible wildcard matching (for patterns like B_B_B -> BABA)
            if len(suggestions) < max_suggestions:
                for pattern, words in self.phoneme_patterns.items():
                    if self._flexible_wildcard_match(wildcard_sequence, pattern):
                        confidence = self._calculate_flexible_similarity(wildcard_sequence, pattern)
                        if confidence > 0.4:
                            for word in words[:2]:
                                if not any(w == word for w, _ in suggestions):
                                    suggestions.append((word, confidence))
                                    if len(suggestions) >= max_suggestions:
                                        break
                    if len(suggestions) >= max_suggestions:
                        break
        
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
    
    def _wildcard_match(self, wildcard_pattern: str, target: str) -> bool:
        """
        Check if wildcard pattern matches target, where _ matches any single character
        
        Args:
            wildcard_pattern: Pattern with _ as wildcards (e.g., "A_E_O")
            target: Target string to match (e.g., "ABEIO")
            
        Returns:
            True if pattern matches
        """
        if len(wildcard_pattern) != len(target):
            return False
        
        for i, (w_char, t_char) in enumerate(zip(wildcard_pattern, target)):
            if w_char != '_' and w_char != t_char:
                return False
        
        return True
    
    def _calculate_wildcard_similarity(self, wildcard_pattern: str, target: str) -> float:
        """Calculate similarity score for wildcard matches"""
        if not self._wildcard_match(wildcard_pattern, target):
            return 0.0
        
        # Count non-wildcard matches
        exact_matches = sum(1 for w, t in zip(wildcard_pattern, target) if w != '_' and w == t)
        total_chars = len(wildcard_pattern)
        wildcards = wildcard_pattern.count('_')
        
        # Score based on exact matches vs wildcards
        if total_chars == 0:
            return 0.0
        
        # Prefer patterns with more exact matches and fewer wildcards
        exact_ratio = exact_matches / total_chars
        wildcard_penalty = wildcards / total_chars
        
        return max(0.0, exact_ratio - (wildcard_penalty * 0.3))
    
    def _flexible_wildcard_match(self, wildcard_pattern: str, target: str) -> bool:
        """
        Flexible wildcard matching where B_B_B can match BABA, etc.
        Uses phonetic similarity rather than strict character counting.
        """
        # Extract non-wildcard characters from pattern
        pattern_chars = [c for c in wildcard_pattern if c != '_']
        
        if not pattern_chars:
            return False
        
        # Count character frequencies
        from collections import Counter
        pattern_count = Counter(pattern_chars)
        target_count = Counter(target)
        
        # For patterns with single repeated character (like B_B_B)
        if len(set(pattern_chars)) == 1:
            char = pattern_chars[0]
            target_char_count = target_count[char]
            pattern_char_count = len(pattern_chars)
            
            # Flexible matching for repeated patterns:
            # B_B_B (3 Bs) can match BABA (2 Bs) if B is dominant in target
            # But target must have at least half the pattern count, and be the dominant char
            min_required = max(1, pattern_char_count // 2)  # At least half
            
            if target_char_count >= min_required:
                # Check if this character is dominant in target (>= 40% of target)
                dominance_ratio = target_char_count / len(target)
                if dominance_ratio >= 0.4:  # Character is prominent in target
                    return True
            
            # Fallback: exact count matching
            return target_char_count >= pattern_char_count
        
        # For mixed patterns, use stricter matching
        # Check if target has at least as many of each character as pattern
        for char, count in pattern_count.items():
            if target_count[char] < count:
                return False
        
        # Check character order for mixed patterns
        char_positions = {}
        for char in set(pattern_chars):
            char_positions[char] = [i for i, c in enumerate(target) if c == char]
        
        # Try to match characters in order
        used_positions = set()
        last_pos = -1
        
        for char in pattern_chars:
            available_positions = [pos for pos in char_positions[char] 
                                 if pos not in used_positions and pos > last_pos]
            
            if not available_positions:
                # Allow some flexibility for mixed patterns
                available_positions = [pos for pos in char_positions[char] 
                                     if pos not in used_positions]
                
            if available_positions:
                chosen_pos = min(available_positions)
                used_positions.add(chosen_pos)
                last_pos = chosen_pos
            else:
                return False
        
        return True
    
    def _calculate_flexible_similarity(self, wildcard_pattern: str, target: str) -> float:
        """Calculate similarity score for flexible wildcard matches"""
        if not self._flexible_wildcard_match(wildcard_pattern, target):
            return 0.0
        
        # Count matching characters
        pattern_chars = [c for c in wildcard_pattern if c != '_']
        target_chars = list(target)
        
        # Count how many pattern chars appear in target
        matches = 0
        target_remaining = target_chars[:]
        for char in pattern_chars:
            if char in target_remaining:
                target_remaining.remove(char)
                matches += 1
        
        # Base similarity on character matches
        char_similarity = matches / len(wildcard_pattern)
        
        # Bonus for length similarity (prefer similar length matches)
        length_similarity = 1.0 - abs(len(wildcard_pattern) - len(target)) / max(len(wildcard_pattern), len(target))
        
        return (char_similarity * 0.7) + (length_similarity * 0.3)
    
    def get_word_suggestions_for_segment(self, viseme_sequence: str) -> List[str]:
        """
        Get word suggestions for a viseme sequence, formatted for SRT display
        
        Args:
            viseme_sequence: String like "AEIOU" or "B_A_T_"
            
        Returns:
            List of suggested words (max 1, only high confidence)
        """
        matches = self.match_phonemes_to_words(viseme_sequence, max_suggestions=5)
        
        # Use more lenient confidence and prioritize phonetic accuracy
        good_matches = []
        # Define most common words for boosting
        super_common = {'the', 'and', 'you', 'it', 'is', 'at', 'to', 'he', 'we', 'me', 'be', 'my', 'by', 'up', 'if', 'an', 'as', 'or', 'on', 'in', 'of', 'a', 'i'}
        
        # Count phonemes in input sequence for phonetic accuracy scoring
        input_phonemes = len(viseme_sequence.replace('_', ''))
        
        for word, confidence in matches:
            # More lenient confidence thresholds
            min_confidence = 0.3 if len(word) <= 6 else 0.4
            if confidence > min_confidence and len(word) <= 10 and word.isalpha():
                # Calculate word score prioritizing phonetic accuracy over brevity
                word_score = confidence
                
                # MAJOR BONUS: Phonetic accuracy (prioritize words that match input phoneme count)
                word_phonemes = len(word)  # Approximate phoneme count
                phoneme_accuracy = 1.0 - abs(input_phonemes - word_phonemes) / max(input_phonemes, word_phonemes, 1)
                
                # Give massive bonus for phonetic accuracy - this should dominate scoring
                if phoneme_accuracy > 0.8:  # Very close match
                    word_score += 1.0  # Huge bonus for phonetically accurate words
                elif phoneme_accuracy > 0.6:  # Good match
                    word_score += 0.7  # Large bonus
                elif phoneme_accuracy > 0.4:  # Reasonable match  
                    word_score += 0.4  # Moderate bonus
                
                # Reduce length bias - only tiny preference for shorter words
                if len(word) <= 3:
                    word_score += 0.05  # Very small bonus
                elif len(word) <= 4:
                    word_score += 0.02  # Minimal bonus
                
                # Common word bonus (small boost for super common words)
                if word in super_common:
                    word_score += 0.1  # Small boost for common words
                
                good_matches.append((word, word_score))
        
        if good_matches:
            # Sort by score and return only the best match
            good_matches.sort(key=lambda x: x[1], reverse=True)
            best_word = good_matches[0][0]
            return [best_word]
        
        # If no good matches, try simplified patterns for common words
        clean_sequence = viseme_sequence.replace('_', '').strip()
        
        # Special handling for common repeated patterns and long sequences
        # Simplify very long repeated patterns
        if len(clean_sequence) >= 4:
            # Count dominant characters
            char_counts = {}
            for char in clean_sequence:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # Find most common character
            dominant_char = max(char_counts.items(), key=lambda x: x[1])[0]
            dominant_ratio = char_counts[dominant_char] / len(clean_sequence)
            
            # If one character dominates (>60%), treat as that sound
            if dominant_ratio > 0.6:
                vowel_words = {
                    'I': 'it', 'A': 'at', 'E': 'eh', 'O': 'oh', 'U': 'uh'
                }
                consonant_words = {
                    'B': 'be', 'P': 'pah', 'M': 'mmm', 'F': 'fff', 'V': 'vvv',
                    'T': 'ttt', 'D': 'ddd', 'K': 'kay', 'G': 'ggg', 'N': 'nnn'
                }
                if dominant_char in vowel_words:
                    return [vowel_words[dominant_char]]
                elif dominant_char in consonant_words:
                    return [consonant_words[dominant_char]]
        
        # Handle shorter repeated patterns
        if clean_sequence == 'I' or clean_sequence in ['II', 'III', 'IIII']:
            return ['it']  # Common "I" sound words
        elif clean_sequence == 'A' or clean_sequence in ['AA', 'AAA', 'AAAA']:
            return ['at']  # Common "A" sound words  
        elif clean_sequence == 'E' or clean_sequence in ['EE', 'EEE', 'EEEE']:
            return ['eh']  # Common "E" sound
        elif clean_sequence == 'O' or clean_sequence in ['OO', 'OOO', 'OOOO']:
            return ['oh']  # Common "O" sound
        elif clean_sequence == 'U' or clean_sequence in ['UU', 'UUU', 'UUUU']:
            return ['uh']  # Common "U" sound
        elif clean_sequence == 'B' or clean_sequence in ['BB', 'BBB', 'BBBB', 'BBBBB']:
            return ['be']  # Common "B" sound words
        elif clean_sequence == 'P' or clean_sequence in ['PP', 'PPP', 'PPPP']:
            return ['pah']  # Common "P" sound
        elif clean_sequence == 'M' or clean_sequence in ['MM', 'MMM', 'MMMM']:
            return ['mmm']  # Common "M" sound
        
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