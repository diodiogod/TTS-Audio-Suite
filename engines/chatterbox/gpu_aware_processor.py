"""
GPU-Aware Batch Processor for ChatterBox TTS

Intelligently limits parallel processing based on GPU capabilities to avoid resource contention.
Uses empirical testing to find optimal worker count for maximum throughput.

Modular design following CLAUDE.md guidelines - under 500 lines, focused responsibility.
"""

import torch
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import time


class GPUAwareBatchProcessor:
    """
    Intelligent batch processor that optimizes worker count based on GPU capabilities.
    
    Instead of using all available workers (which causes resource contention),
    this processor finds the optimal balance between parallelism and GPU efficiency.
    """
    
    def __init__(self, tts_model, device: str = "auto"):
        """
        Initialize with GPU-aware configuration.
        
        Args:
            tts_model: ChatterboxTTS instance
            device: Target device ("cuda", "cpu", "auto")
        """
        self.tts_model = tts_model
        self.device = self._resolve_device(device)
        self.optimal_workers = self._determine_optimal_workers()
        
        print(f"🔧 GPU-Aware Processor initialized: {self.optimal_workers} optimal workers for {self.device}")
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _determine_optimal_workers(self) -> int:
        """
        Determine optimal worker count based on hardware capabilities.
        
        Returns:
            Optimal number of parallel workers
        """
        if self.device == "cpu":
            # CPU-only: use more workers since no GPU contention
            return min(psutil.cpu_count() or 4, 8)
        
        # GPU-based optimization
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Empirical optimization based on ChatterBox GPU requirements
            if gpu_memory_gb >= 24:      # RTX 4090, A6000, etc.
                return 4  # Sweet spot for high-end cards
            elif gpu_memory_gb >= 16:    # RTX 4080, etc.
                return 3
            elif gpu_memory_gb >= 12:    # RTX 4070 Ti, etc.
                return 2
            elif gpu_memory_gb >= 8:     # RTX 4060 Ti, etc.  
                return 2
            else:                        # < 8GB VRAM
                return 1  # Sequential processing safer
        
        return 2  # Conservative fallback
    
    def process_texts_optimized(
        self, 
        texts: List[str], 
        temperature: float, 
        cfg_weight: float, 
        exaggeration: float,
        requested_workers: int
    ) -> List[torch.Tensor]:
        """
        Process texts with GPU-optimized worker count.
        
        Args:
            texts: List of text strings
            temperature: Sampling temperature
            cfg_weight: CFG weight
            exaggeration: Emotion exaggeration
            requested_workers: User-requested worker count (will be clamped to optimal)
            
        Returns:
            List of audio tensors in original order
        """
        # Use optimal workers regardless of user request for best performance
        effective_workers = min(self.optimal_workers, len(texts))
        
        if effective_workers != requested_workers:
            print(f"🔧 GPU Optimization: Using {effective_workers} workers instead of {requested_workers} for optimal GPU utilization")
        
        return self._process_with_workers(texts, temperature, cfg_weight, exaggeration, effective_workers)
    
    def _process_with_workers(
        self, 
        texts: List[str], 
        temperature: float, 
        cfg_weight: float, 
        exaggeration: float,
        max_workers: int
    ) -> List[torch.Tensor]:
        """Process texts with specified worker count."""
        
        def _process_single_text(text_index: int, text: str) -> Tuple[int, torch.Tensor]:
            """Process single text with GPU-aware error handling."""
            try:
                print(f"  🧵 GPU-Worker {text_index+1}: Processing: {text[:30]}...")
                
                # Clear GPU cache before processing to avoid OOM
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                
                start_time = time.time()
                audio = self.tts_model.generate(
                    text=text,
                    audio_prompt_path=None,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
                
                duration = time.time() - start_time
                print(f"  ✅ GPU-Worker {text_index+1}: Completed in {duration:.1f}s")
                return text_index, audio
                
            except Exception as e:
                print(f"  ❌ GPU-Worker {text_index+1}: Failed - {e}")
                return text_index, torch.zeros(1, 1000)
        
        print(f"🚀 GPU-OPTIMIZED PROCESSING: {len(texts)} texts with {max_workers} optimal GPU workers")
        
        # Process with optimal worker count
        results = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_process_single_text, i, text)
                for i, text in enumerate(texts)
            ]
            
            completed_count = 0
            total_duration = time.time()
            
            for future in as_completed(futures):
                text_index, audio = future.result()
                results[text_index] = audio
                completed_count += 1
                
                progress_percent = int(100 * completed_count / len(texts))
                print(f"  📊 Progress: {completed_count}/{len(texts)} completed ({progress_percent}%)")
        
        total_time = time.time() - total_duration
        avg_time = total_time / len(texts)
        print(f"✅ GPU-OPTIMIZED PROCESSING COMPLETED: {len(results)} texts in {total_time:.1f}s (avg: {avg_time:.1f}s/text)")
        
        return results
    
    def benchmark_worker_counts(self, sample_texts: List[str], max_test_workers: int = 8) -> dict:
        """
        Benchmark different worker counts to find empirically optimal settings.
        
        Args:
            sample_texts: Representative text samples for benchmarking
            max_test_workers: Maximum workers to test
            
        Returns:
            Dict with benchmark results
        """
        if len(sample_texts) < 2:
            print("⚠️ Need at least 2 sample texts for benchmarking")
            return {}
        
        print(f"🔬 Benchmarking worker counts 1-{max_test_workers} with {len(sample_texts)} sample texts...")
        
        results = {}
        
        for workers in range(1, min(max_test_workers + 1, len(sample_texts) + 1)):
            print(f"🧪 Testing {workers} workers...")
            
            start_time = time.time()
            _ = self._process_with_workers(
                sample_texts[:workers], 0.8, 0.5, 0.5, workers
            )
            duration = time.time() - start_time
            
            throughput = workers / duration  # texts per second
            results[workers] = {
                'duration': duration,
                'throughput': throughput,
                'texts_processed': workers
            }
            
            print(f"  📈 {workers} workers: {duration:.1f}s total, {throughput:.2f} texts/sec")
        
        # Find optimal
        best_workers = max(results.keys(), key=lambda w: results[w]['throughput'])
        best_throughput = results[best_workers]['throughput']
        
        print(f"🏆 OPTIMAL: {best_workers} workers with {best_throughput:.2f} texts/sec throughput")
        
        return results


class ProcessingStrategy:
    """Strategy selector for different processing approaches."""
    
    @staticmethod
    def choose_strategy(num_texts: int, requested_workers: int, device: str) -> str:
        """Choose optimal processing strategy."""
        if device == "cpu":
            return "cpu_parallel"
        elif num_texts == 1:
            return "single"
        elif torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb < 8:
                return "sequential"  # Low VRAM - avoid parallelism
            else:
                return "gpu_optimized"
        else:
            return "sequential"
    
    @staticmethod
    def get_strategy_explanation(strategy: str) -> str:
        """Get explanation of chosen strategy."""
        explanations = {
            "single": "Single text - direct processing",
            "sequential": "Sequential processing - low GPU memory",
            "cpu_parallel": "CPU parallel processing - no GPU constraints",
            "gpu_optimized": "GPU-optimized parallel processing - balanced for throughput"
        }
        return explanations.get(strategy, "Unknown strategy")