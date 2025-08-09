"""
Overlapping Batch Processor for ChatterBox TTS

Handles advanced parallel processing with overlapping batches instead of sequential batching.
This allows maximum throughput by keeping all workers busy continuously.

Modular design following CLAUDE.md guidelines - under 500 lines, focused responsibility.
"""

import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any


class OverlappingBatchProcessor:
    """
    Advanced parallel processing that doesn't wait for batches to complete sequentially.
    
    Instead of processing texts in batches of N and waiting for each batch to complete,
    this processor maintains up to max_workers threads running simultaneously at all times.
    
    This provides true overlapping parallel processing for maximum throughput.
    """
    
    def __init__(self, tts_model):
        """
        Initialize with reference to the TTS model.
        
        Args:
            tts_model: ChatterboxTTS instance with generate() method
        """
        self.tts_model = tts_model
    
    def process_texts_with_overlap(
        self, 
        texts: List[str], 
        temperature: float, 
        cfg_weight: float, 
        exaggeration: float, 
        max_workers: int
    ) -> List[torch.Tensor]:
        """
        Process all texts using overlapping workers for maximum throughput.
        
        Args:
            texts: List of text strings to generate audio for
            temperature: Sampling temperature
            cfg_weight: CFG weight  
            exaggeration: Emotion exaggeration factor
            max_workers: Maximum number of simultaneous workers
            
        Returns:
            List of audio tensors in original order
        """
        print(f"🔥 OVERLAPPING PROCESSOR: {len(texts)} texts with {max_workers} workers")
        
        def _process_single_text(text_index: int, text: str) -> tuple:
            """Process a single text with full error handling and logging."""
            try:
                print(f"  🧵 Worker {text_index+1}: Starting: {text[:30]}...")
                
                # Use the TTS model's generate method with pre-loaded conditioning
                audio = self.tts_model.generate(
                    text=text,
                    audio_prompt_path=None,  # Use already loaded conditionals
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
                
                print(f"  ✅ Worker {text_index+1}: Completed successfully")
                return text_index, audio
                
            except Exception as e:
                print(f"  ❌ Worker {text_index+1}: Failed - {e}")
                # Return empty tensor as fallback
                return text_index, torch.zeros(1, 1000)
        
        # Process all texts with true overlapping parallel execution
        results = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit ALL texts for processing - this enables overlap!
            # Unlike sequential batching, workers start immediately as others complete
            futures = [
                executor.submit(_process_single_text, i, text)
                for i, text in enumerate(texts)
            ]
            
            # Collect results as they complete (preserves original order)
            completed_count = 0
            for future in as_completed(futures):
                text_index, audio = future.result()
                results[text_index] = audio
                completed_count += 1
                
                # Progress reporting
                progress_percent = int(100 * completed_count / len(texts))
                print(f"  📊 Progress: {completed_count}/{len(texts)} completed ({progress_percent}%)")
        
        print(f"✅ OVERLAPPING PROCESSING COMPLETED: {len(results)} audio segments")
        return results
    
    def should_use_overlapping(self, num_texts: int, max_workers: int) -> bool:
        """
        Determine if overlapping processing should be used.
        
        Args:
            num_texts: Number of texts to process
            max_workers: Available workers
            
        Returns:
            True if overlapping processing would be beneficial
        """
        # Use overlapping when we have more texts than workers
        # This ensures workers stay busy instead of waiting for batch completion
        return num_texts > max_workers
    
    def estimate_performance_improvement(self, num_texts: int, max_workers: int) -> str:
        """
        Estimate performance improvement over sequential batching.
        
        Args:
            num_texts: Number of texts
            max_workers: Number of workers
            
        Returns:
            Human-readable performance estimate
        """
        if num_texts <= max_workers:
            return "Single batch - no improvement needed"
        
        sequential_batches = (num_texts + max_workers - 1) // max_workers
        overlapping_efficiency = min(num_texts, max_workers * 2)  # Rough estimate
        
        improvement = overlapping_efficiency / sequential_batches
        
        if improvement > 1.5:
            return f"~{improvement:.1f}x faster with overlapping processing"
        elif improvement > 1.2:
            return f"~{improvement:.1f}x faster with overlapping processing"
        else:
            return "Minimal improvement expected"


class BatchingStrategy:
    """
    Determines the optimal batching strategy based on workload characteristics.
    """
    
    @staticmethod
    def choose_strategy(num_texts: int, max_workers: int) -> str:
        """
        Choose the optimal processing strategy.
        
        Args:
            num_texts: Number of texts to process
            max_workers: Available workers
            
        Returns:
            Strategy name: "single_batch", "overlapping", or "sequential"
        """
        if num_texts <= max_workers:
            return "single_batch"
        elif num_texts <= max_workers * 2:
            return "overlapping"  # Sweet spot for overlapping
        else:
            return "overlapping"  # Still better than sequential for large workloads
    
    @staticmethod
    def get_strategy_description(strategy: str) -> str:
        """Get human-readable description of the chosen strategy."""
        descriptions = {
            "single_batch": "All texts processed simultaneously in one batch",
            "overlapping": "Continuous parallel processing with overlapping workers", 
            "sequential": "Traditional sequential batch processing (fallback only)"
        }
        return descriptions.get(strategy, "Unknown strategy")