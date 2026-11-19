"""
Text Chunking - Enhanced text chunker for ChatterBox Voice
Extracted from the original nodes.py implementation
"""

import re
import textwrap
import torch
from typing import List


MAX_CHARS_PER_CHUNK = 20_000


class ImprovedChatterBoxChunker:
    """Enhanced text chunker inspired by Orpheus TTS approach"""
    
    @staticmethod
    def split_into_chunks(text: str, max_chars: int = 400) -> List[str]:
        """
        Split text into chunks with better sentence boundary handling.
        Uses character-based limits like Orpheus TTS for more predictable chunk sizes.
        
        Args:
            text: Input text to split
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
            
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # If text is short enough, return as single chunk
        if len(text) <= max_chars:
            return [text]
        
        # Split into sentences using robust regex (same as Orpheus)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check oversized sentences before packing them after an existing chunk.
            if len(sentence) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                parts = re.split(r'(?<=,)\s+', sentence)
            else:
                parts = [sentence]

            for part in parts:
                if len(part) > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    # Prefer word boundaries; only split a word if it exceeds the limit itself.
                    sub_chunks = textwrap.wrap(part, width=max_chars, break_on_hyphens=False)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1]
                elif current_chunk and len(current_chunk) + len(part) + 1 > max_chars:
                    chunks.append(current_chunk)
                    current_chunk = part
                else:
                    # Comma-separated parts already retain their punctuation.
                    current_chunk = current_chunk + " " + part if current_chunk else part
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @staticmethod
    def add_silence_padding(audio: torch.Tensor, duration_ms: int = 50, sample_rate: int = 22050) -> torch.Tensor:
        """
        Add brief silence between chunks to improve naturalness
        
        Args:
            audio: Input audio tensor
            duration_ms: Duration of silence in milliseconds
            sample_rate: Sample rate of the audio
            
        Returns:
            Audio tensor with silence padding added
        """
        silence_samples = int(duration_ms * sample_rate / 1000)
        
        # Create silence tensor with same shape as audio tensor
        if audio.dim() == 1:
            # 1D audio tensor [samples]
            silence = torch.zeros(silence_samples, device=audio.device, dtype=audio.dtype)
        elif audio.dim() == 2:
            # 2D audio tensor [channels, samples]
            silence = torch.zeros(audio.shape[0], silence_samples, device=audio.device, dtype=audio.dtype)
        else:
            # Fallback - just match the last dimension
            silence_shape = list(audio.shape)
            silence_shape[-1] = silence_samples
            silence = torch.zeros(*silence_shape, device=audio.device, dtype=audio.dtype)
        
        return torch.cat([audio, silence], dim=-1)
    
    @staticmethod
    def estimate_chunk_count(text: str, max_chars: int = 400) -> int:
        """
        Estimate how many chunks the text will be split into without actually splitting.
        Useful for progress tracking.
        
        Args:
            text: Input text to estimate
            max_chars: Maximum characters per chunk
            
        Returns:
            Estimated number of chunks
        """
        if not text.strip():
            return 0
        
        text = re.sub(r'\s+', ' ', text.strip())
        
        if len(text) <= max_chars:
            return 1
        
        # Quick estimation based on average chunk size
        # This is approximate but good enough for progress estimation
        return max(1, len(text) // max_chars + (1 if len(text) % max_chars > 0 else 0))
    
    @staticmethod
    def get_chunk_stats(chunks: List[str]) -> dict:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_chars": 0,
                "avg_chars_per_chunk": 0,
                "min_chars": 0,
                "max_chars": 0,
                "shortest_chunk": "",
                "longest_chunk": ""
            }
        
        chunk_lengths = [len(chunk) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_chars": sum(chunk_lengths),
            "avg_chars_per_chunk": sum(chunk_lengths) / len(chunks),
            "min_chars": min(chunk_lengths),
            "max_chars": max(chunk_lengths),
            "shortest_chunk": min(chunks, key=len),
            "longest_chunk": max(chunks, key=len)
        }
    
    @classmethod
    def validate_chunking_params(cls, max_chars: int) -> int:
        """
        Validate and normalize chunking parameters.
        
        Args:
            max_chars: Maximum characters per chunk
            
        Returns:
            Validated max_chars value
        """
        # Ensure reasonable bounds
        if max_chars < 100:
            max_chars = 100
        elif max_chars > MAX_CHARS_PER_CHUNK:
            max_chars = MAX_CHARS_PER_CHUNK
        
        return max_chars
    
    @classmethod
    def chunk_with_overlap(cls, text: str, max_chars: int = 400, overlap_chars: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better context preservation.
        
        Args:
            text: Input text to split
            max_chars: Maximum characters per chunk
            overlap_chars: Number of characters to overlap between chunks
            
        Returns:
            List of overlapping text chunks
        """
        if not text.strip():
            return []
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:].strip())
                break
            
            # Try to find a good break point (sentence boundary)
            chunk_text = text[start:end]
            
            # Look for sentence endings near the end of the chunk
            sentence_breaks = [m.end() for m in re.finditer(r'[.!?]\s+', chunk_text)]
            if sentence_breaks:
                # Use the last sentence break as the end point
                actual_end = start + sentence_breaks[-1]
                chunks.append(text[start:actual_end].strip())
                start = actual_end - overlap_chars
            else:
                # No good break point found, split at max_chars
                chunks.append(chunk_text.strip())
                start = end - overlap_chars
            
            # Ensure we don't go backwards
            if start < 0:
                start = 0
        
        return [chunk for chunk in chunks if chunk.strip()]