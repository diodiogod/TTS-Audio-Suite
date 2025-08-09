"""
Character Grouper Module

Groups segments by character WITHIN each language group.
Integrates with existing language grouping system - does not replace it.

Fixes the issue where segments like:
- Segment 3: crestfallen_original  
- Segment 5: crestfallen_original
Were processed separately instead of being batched together.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CharacterSegment:
    """Represents a single segment for a specific character within a language."""
    original_idx: int
    character: str
    segment_text: str
    segment_lang: str
    
    def __str__(self):
        return f"Segment({self.original_idx+1}: {self.character})"


class CharacterGroup:
    """A group of segments for the same character within a language."""
    
    def __init__(self, character: str):
        self.character = character
        self.segments: List[CharacterSegment] = []
    
    def add_segment(self, segment: CharacterSegment):
        """Add a segment to this character group."""
        self.segments.append(segment)
    
    def can_batch(self) -> bool:
        """Check if this group has enough segments to warrant batch processing."""
        return len(self.segments) > 1
    
    def get_segment_data(self) -> List[Tuple[int, str, str, str]]:
        """Get segment data in the format expected by existing code: (original_idx, character, segment_text, segment_lang)"""
        return [(seg.original_idx, seg.character, seg.segment_text, seg.segment_lang) for seg in self.segments]
    
    def __len__(self):
        return len(self.segments)
    
    def __str__(self):
        indices = [str(seg.original_idx + 1) for seg in self.segments]  # 1-based for display
        batch_indicator = "🚀" if self.can_batch() else "→"
        return f"{batch_indicator} {self.character}({len(self.segments)}): #{','.join(indices)}"


class CharacterGrouper:
    """
    Groups segments by character WITHIN a language group.
    
    This works with the existing language grouping system:
    
    EXISTING: language_groups[lang] = [(idx, char, text, lang), ...]
    NEW:      character_groups = group_by_character(lang_segments)
    
    So we go from:
    lang_segments = [(1,'narrator',...), (2,'david',...), (3,'narrator',...)]
    
    To:
    character_groups = {
        'narrator': CharacterGroup([segment1, segment3]),
        'david': CharacterGroup([segment2])
    }
    """
    
    @staticmethod
    def group_by_character(lang_segments: List[Tuple[int, str, str, str]]) -> Dict[str, CharacterGroup]:
        """
        Group segments from a language by character.
        
        Args:
            lang_segments: List of (original_idx, character, segment_text, segment_lang)
                          This comes from the existing language grouping system
        
        Returns:
            Dict mapping character_name -> CharacterGroup
        """
        character_groups = {}
        
        for original_idx, character, segment_text, segment_lang in lang_segments:
            if character not in character_groups:
                character_groups[character] = CharacterGroup(character)
            
            segment = CharacterSegment(
                original_idx=original_idx,
                character=character,
                segment_text=segment_text,
                segment_lang=segment_lang
            )
            
            character_groups[character].add_segment(segment)
        
        return character_groups
    
    @staticmethod
    def print_character_grouping_summary(language: str, character_groups: Dict[str, CharacterGroup]):
        """Print summary of character grouping for a language."""
        total_segments = sum(len(group) for group in character_groups.values())
        batchable_segments = sum(len(group) for group in character_groups.values() if group.can_batch())
        
        group_summaries = [str(group) for group in character_groups.values()]
        
        print(f"🎭 {language} CHARACTER GROUPS ({batchable_segments}/{total_segments} batchable):")
        print(f"   {' | '.join(group_summaries)}")
    
    @staticmethod
    def validate_character_grouping(original_lang_segments: List[Tuple[int, str, str, str]], 
                                   character_groups: Dict[str, CharacterGroup]):
        """Validate that character grouping preserves all segments."""
        original_count = len(original_lang_segments)
        grouped_count = sum(len(group) for group in character_groups.values())
        
        if original_count != grouped_count:
            raise RuntimeError(f"Character grouping lost segments: {original_count} -> {grouped_count}")
        
        # Validate all original indices are preserved
        original_indices = {idx for idx, _, _, _ in original_lang_segments}
        grouped_indices = set()
        
        for group in character_groups.values():
            for seg in group.segments:
                grouped_indices.add(seg.original_idx)
        
        if original_indices != grouped_indices:
            missing = original_indices - grouped_indices
            extra = grouped_indices - original_indices
            raise RuntimeError(f"Character grouping index mismatch. Missing: {missing}, Extra: {extra}")
        
        print(f"   ✅ Character grouping validated: {grouped_count} segments preserved")