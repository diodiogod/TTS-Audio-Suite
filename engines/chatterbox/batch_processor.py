"""
Batch Processor Module

Handles actual batch processing of character groups.
Works with character_grouper.py to process multiple segments simultaneously.
"""

from typing import Dict, List, Any
import torch
from .character_grouper import CharacterGroup


class BatchProcessor:
    """
    Processes character groups using batch inference.
    
    This is where the actual batching magic happens - instead of:
    1. Process segment 3 (crestfallen_original)
    2. Process segment 5 (crestfallen_original)
    
    We do:
    1. Batch process segments [3, 5] for crestfallen_original simultaneously
    """
    
    def __init__(self, tts_model):
        self.tts_model = tts_model
    
    def process_character_group_batch(
        self,
        character_group: CharacterGroup,
        inputs: Dict[str, Any],
        voice_refs: Dict[str, str],
        chunker,
        language: str
    ) -> Dict[int, torch.Tensor]:
        """
        Process all segments for a character using TRUE batch processing.
        
        Args:
            character_group: Group of segments for same character
            inputs: Node inputs (exaggeration, temperature, etc.)
            voice_refs: Character voice reference mapping
            chunker: Text chunking utility
            language: Language code for logging
        
        Returns:
            Dict mapping original_idx -> generated_audio_tensor
        """
        character = character_group.character
        segments = character_group.segments
        
        print(f"🚀 BATCH PROCESSING: {character} - {len(segments)} segments in {language}")
        
        # Collect all texts for batching
        batch_texts = []
        segment_chunk_counts = []  # Track how many chunks each segment produces
        
        for segment in segments:
            # Apply chunking if needed
            if inputs["enable_chunking"] and len(segment.segment_text) > inputs["max_chars_per_chunk"]:
                chunks = chunker.split_into_chunks(segment.segment_text, inputs["max_chars_per_chunk"])
            else:
                chunks = [segment.segment_text]
            
            # Process each chunk (apply crash protection, etc.)
            processed_chunks = []
            for chunk_text in chunks:
                processed_chunk = self._apply_crash_protection(chunk_text, inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"))
                processed_chunks.append(processed_chunk)
            
            batch_texts.extend(processed_chunks)
            segment_chunk_counts.append(len(processed_chunks))
        
        # Get voice reference for this character
        char_audio_prompt = voice_refs[character]
        
        # THE ACTUAL BATCH PROCESSING - this is the key improvement
        print(f"⚡ Batch generating {len(batch_texts)} chunks for {character}")
        print(f"🔧 DEBUG: batch_size from inputs = {inputs.get('batch_size', 4)}")
        try:
            batch_audio = self.tts_model.generate_batch(
                texts=batch_texts,
                audio_prompt_path=char_audio_prompt,
                exaggeration=inputs["exaggeration"],
                cfg_weight=inputs["cfg_weight"],
                temperature=inputs["temperature"],
                batch_size=inputs.get("batch_size", 4),
                max_workers=inputs.get("batch_size", 4),  # Use batch_size as max_workers  
                enable_adaptive_batching=(inputs.get("batch_size", 1) > 1)  # Auto-enable based on batch_size
            )
            
            print(f"✅ BATCH SUCCESS: Generated {len(batch_audio)} audio chunks")
            
        except Exception as e:
            print(f"❌ BATCH FAILED: {e}")
            raise  # Let calling code handle fallback
        
        # Map batch results back to original segments
        results = {}
        batch_idx = 0
        
        for i, segment in enumerate(segments):
            chunk_count = segment_chunk_counts[i]
            
            # Collect all chunks for this segment
            segment_chunks = []
            for _ in range(chunk_count):
                segment_chunks.append(batch_audio[batch_idx])
                batch_idx += 1
            
            # Combine chunks if multiple
            if len(segment_chunks) == 1:
                segment_audio = segment_chunks[0]
            else:
                segment_audio = torch.cat(segment_chunks, dim=-1)
            
            results[segment.original_idx] = segment_audio
        
        return results
    
    def process_character_group_sequential(
        self,
        character_group: CharacterGroup,
        inputs: Dict[str, Any],
        voice_refs: Dict[str, str],
        required_language: str,
        total_segments: int,
        language: str,
        stable_audio_component: str,
        chunker,
        generate_tts_method
    ) -> Dict[int, torch.Tensor]:
        """
        Fallback: Process character group sequentially.
        Used when batch processing fails or for single segments.
        """
        character = character_group.character
        segments = character_group.segments
        
        print(f"→ SEQUENTIAL: {character} - {len(segments)} segments in {language}")
        
        results = {}
        char_audio_prompt = voice_refs[character]
        
        for segment in segments:
            segment_display_idx = segment.original_idx + 1  # 1-based for display
            
            print(f"🎤 Generating segment {segment_display_idx}/{total_segments} for '{character}' (lang: {language})")
            
            # Apply chunking
            if inputs["enable_chunking"] and len(segment.segment_text) > inputs["max_chars_per_chunk"]:
                chunks = chunker.split_into_chunks(segment.segment_text, inputs["max_chars_per_chunk"])
            else:
                chunks = [segment.segment_text]
            
            # Process each chunk sequentially
            segment_audio_chunks = []
            for chunk_i, chunk_text in enumerate(chunks):
                if len(chunks) > 1:
                    print(f"  → Chunk {chunk_i+1}/{len(chunks)}")
                
                chunk_audio = generate_tts_method(
                    chunk_text, char_audio_prompt, inputs["exaggeration"],
                    inputs["temperature"], inputs["cfg_weight"], required_language,
                    True, character=character, seed=inputs["seed"],
                    enable_cache=inputs.get("enable_audio_cache", True),
                    crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                    stable_audio_component=stable_audio_component
                )
                segment_audio_chunks.append(chunk_audio)
            
            # Combine chunks for this segment
            if len(segment_audio_chunks) == 1:
                segment_audio = segment_audio_chunks[0]
            else:
                segment_audio = torch.cat(segment_audio_chunks, dim=-1)
            
            results[segment.original_idx] = segment_audio
        
        return results
    
    def _apply_crash_protection(self, text: str, padding_template: str, min_length: int = 15) -> str:
        """Apply crash protection padding to short texts."""
        if len(text.strip()) < min_length:
            return padding_template.format(seg=text)
        return text