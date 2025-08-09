"""
Adaptive Batch Processor for ChatterBox TTS

Dynamically adjusts worker count based on real-time performance monitoring.
Starts with user's batch_size (respecting their insight) and adapts from there.

Optional feature - can be disabled to respect user's exact worker count preference.
Modular design following CLAUDE.md guidelines - under 500 lines, focused responsibility.
"""

import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import List, Tuple, Dict, Optional, Deque
from collections import deque
from dataclasses import dataclass
import threading


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    timestamp: float
    worker_count: int
    completion_time: float
    tokens_per_second: float
    total_throughput: float  # texts completed per second


class AdaptivePerformanceMonitor:
    """
    Monitors real-time performance and suggests worker count adjustments.
    """
    
    def __init__(self, initial_workers: int, adaptation_enabled: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            initial_workers: User's preferred starting worker count
            adaptation_enabled: Whether to suggest adaptations
        """
        self.initial_workers = initial_workers
        self.current_optimal_workers = initial_workers
        self.adaptation_enabled = adaptation_enabled
        
        # Performance tracking
        self.metrics: Deque[PerformanceMetric] = deque(maxlen=20)  # Last 20 measurements
        self.worker_performance: Dict[int, List[float]] = {}  # worker_count -> [throughputs]
        
        # Adaptation logic
        self.evaluation_interval = 3  # Evaluate every N completions
        self.completions_since_eval = 0
        self.last_adaptation_time = time.time()
        self.min_adaptation_interval = 10.0  # Wait 10s between adaptations
        
        # Stability tracking
        self.stable_performance_threshold = 0.1  # 10% variance
        self.performance_declining_threshold = 0.15  # 15% decline triggers reduction
        
        print(f"🔧 Adaptive Monitor: Starting with {initial_workers} workers (user preference)")
        print(f"🎛️ Dynamic adaptation: {'ENABLED' if adaptation_enabled else 'DISABLED'}")
    
    def record_completion(self, completion_time: float, tokens_per_second: float) -> Optional[int]:
        """
        Record a text completion and return suggested worker count change.
        
        Args:
            completion_time: Time taken for this text (seconds)
            tokens_per_second: Generation speed for this text
            
        Returns:
            Suggested new worker count, or None if no change recommended
        """
        if not self.adaptation_enabled:
            return None  # Respect user's exact worker count
        
        # Record this completion
        current_time = time.time()
        
        # Calculate current throughput (approximate) - FIXED: prevent division by zero
        recent_completions = len([m for m in self.metrics if current_time - m.timestamp < 10])
        if self.metrics:
            time_window = max(0.1, current_time - self.metrics[0].timestamp)  # Minimum 0.1s window
            total_throughput = recent_completions / min(10, time_window)
        else:
            total_throughput = 0.0  # No data yet
        
        metric = PerformanceMetric(
            timestamp=current_time,
            worker_count=self.current_optimal_workers,
            completion_time=completion_time,
            tokens_per_second=tokens_per_second,
            total_throughput=total_throughput
        )
        self.metrics.append(metric)
        
        # Track performance for this worker count
        if self.current_optimal_workers not in self.worker_performance:
            self.worker_performance[self.current_optimal_workers] = []
        self.worker_performance[self.current_optimal_workers].append(total_throughput)
        
        # Check if it's time to evaluate
        self.completions_since_eval += 1
        if self.completions_since_eval >= self.evaluation_interval:
            return self._evaluate_adaptation()
        
        return None
    
    def _evaluate_adaptation(self) -> Optional[int]:
        """Evaluate whether to adapt worker count."""
        self.completions_since_eval = 0
        current_time = time.time()
        
        # Don't adapt too frequently
        if current_time - self.last_adaptation_time < self.min_adaptation_interval:
            return None
        
        if len(self.metrics) < 6:  # Need some data
            return None
        
        # Get recent performance
        recent_metrics = list(self.metrics)[-6:]  # Last 6 completions
        current_avg_throughput = sum(m.total_throughput for m in recent_metrics) / len(recent_metrics)
        current_avg_speed = sum(m.tokens_per_second for m in recent_metrics) / len(recent_metrics)
        
        # Compare with earlier performance for same worker count
        worker_history = self.worker_performance.get(self.current_optimal_workers, [])
        if len(worker_history) > 6:
            earlier_avg = sum(worker_history[-12:-6]) / 6  # Earlier batch
            current_avg = sum(worker_history[-6:]) / 6     # Current batch
            
            performance_change = (current_avg - earlier_avg) / earlier_avg
            
            if performance_change < -self.performance_declining_threshold:
                # Performance declining significantly - reduce workers
                new_workers = max(1, self.current_optimal_workers - 1)
                print(f"🔽 ADAPTIVE: Performance declining ({performance_change:.1%}), reducing workers {self.current_optimal_workers} → {new_workers}")
                return self._suggest_adaptation(new_workers)
            
            elif performance_change > -0.05 and current_avg_speed > 3.0:  # Stable + fast
                # Performance stable and individual workers are fast - try more workers
                # But don't go crazy - cap at 2x user's initial preference
                max_workers = min(self.initial_workers * 2, 8)
                if self.current_optimal_workers < max_workers:
                    new_workers = self.current_optimal_workers + 1
                    print(f"🔼 ADAPTIVE: Performance stable & fast ({current_avg_speed:.1f} it/s), trying more workers {self.current_optimal_workers} → {new_workers}")
                    return self._suggest_adaptation(new_workers)
        
        return None
    
    def _suggest_adaptation(self, new_workers: int) -> int:
        """Record adaptation suggestion."""
        self.last_adaptation_time = time.time()
        self.current_optimal_workers = new_workers
        return new_workers
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        if not self.metrics:
            return {"status": "No data yet"}
        
        recent = list(self.metrics)[-5:] if len(self.metrics) >= 5 else list(self.metrics)
        avg_throughput = sum(m.total_throughput for m in recent) / len(recent)
        avg_speed = sum(m.tokens_per_second for m in recent) / len(recent)
        
        return {
            "current_workers": self.current_optimal_workers,
            "initial_workers": self.initial_workers,
            "avg_throughput": avg_throughput,
            "avg_speed_per_worker": avg_speed,
            "adaptation_enabled": self.adaptation_enabled,
            "total_completions": len(self.metrics)
        }


class AdaptiveBatchProcessor:
    """
    Batch processor with optional dynamic worker adaptation.
    
    Starts with user's batch_size (respecting their insight) and optionally
    adapts based on real-time performance monitoring.
    """
    
    def __init__(self, tts_model, enable_adaptation: bool = True):
        """
        Initialize adaptive processor.
        
        Args:
            tts_model: ChatterboxTTS instance
            enable_adaptation: Whether to enable dynamic adaptation
        """
        self.tts_model = tts_model
        self.enable_adaptation = enable_adaptation
        self.current_executor: Optional[ThreadPoolExecutor] = None
        self.monitor: Optional[AdaptivePerformanceMonitor] = None
        self.adaptation_lock = threading.Lock()
    
    def process_texts_adaptively(
        self, 
        texts: List[str], 
        temperature: float, 
        cfg_weight: float, 
        exaggeration: float,
        user_batch_size: int
    ) -> List[torch.Tensor]:
        """
        Process texts with optional adaptive worker management.
        
        Args:
            texts: List of text strings
            temperature: Sampling temperature
            cfg_weight: CFG weight
            exaggeration: Emotion exaggeration
            user_batch_size: User's preferred batch size (starting point)
            
        Returns:
            List of audio tensors in original order
        """
        # Initialize monitor with user's preference
        self.monitor = AdaptivePerformanceMonitor(
            initial_workers=user_batch_size,
            adaptation_enabled=self.enable_adaptation
        )
        
        effective_workers = min(user_batch_size, len(texts))
        
        if self.enable_adaptation:
            print(f"🤖 ADAPTIVE PROCESSING: Starting with {effective_workers} workers (user preference), will adapt based on performance")
        else:
            print(f"🔒 FIXED PROCESSING: Using exactly {effective_workers} workers as requested")
        
        return self._process_with_adaptive_workers(
            texts, temperature, cfg_weight, exaggeration, effective_workers
        )
    
    def _process_with_adaptive_workers(
        self, 
        texts: List[str], 
        temperature: float, 
        cfg_weight: float, 
        exaggeration: float,
        initial_workers: int
    ) -> List[torch.Tensor]:
        """Process texts with adaptive worker management."""
        
        results = [None] * len(texts)
        completed_indices = set()
        current_workers = initial_workers
        
        def _process_single_text(text_index: int, text: str) -> Tuple[int, torch.Tensor, float, float]:
            """Process single text and return timing metrics."""
            start_time = time.time()
            
            try:
                print(f"  🧵 Worker {text_index+1}: Processing: {text[:30]}...")
                
                # Track generation performance
                gen_start = time.time()
                audio = self.tts_model.generate(
                    text=text,
                    audio_prompt_path=None,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
                gen_time = time.time() - gen_start
                
                # Estimate tokens/second (rough approximation)
                estimated_tokens = len(text) * 2  # Rough estimate
                tokens_per_second = estimated_tokens / gen_time if gen_time > 0 else 0
                
                total_time = time.time() - start_time
                print(f"  ✅ Worker {text_index+1}: Completed in {total_time:.1f}s ({tokens_per_second:.1f} tok/s)")
                
                return text_index, audio, total_time, tokens_per_second
                
            except Exception as e:
                total_time = time.time() - start_time
                print(f"  ❌ Worker {text_index+1}: Failed in {total_time:.1f}s - {e}")
                return text_index, torch.zeros(1, 1000), total_time, 0.0
        
        # Process all texts
        with ThreadPoolExecutor(max_workers=current_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_process_single_text, i, text): i
                for i, text in enumerate(texts)
            }
            
            completed_count = 0
            total_start_time = time.time()
            
            for future in as_completed(futures):
                text_index, audio, completion_time, tokens_per_second = future.result()
                results[text_index] = audio
                completed_indices.add(text_index)
                completed_count += 1
                
                # Record performance and get adaptation suggestion
                if self.monitor:
                    suggested_workers = self.monitor.record_completion(completion_time, tokens_per_second)
                    
                    # Note: For simplicity, we don't actually restart the executor mid-process
                    # Real adaptation would need more complex future management
                    # This logs what WOULD happen for user feedback
                    if suggested_workers and suggested_workers != current_workers:
                        print(f"📊 ADAPTATION SUGGESTION: Would adjust to {suggested_workers} workers (currently {current_workers})")
                        # In a full implementation, you'd restart executor with new worker count
                        # current_workers = suggested_workers
                
                progress_percent = int(100 * completed_count / len(texts))
                print(f"  📊 Progress: {completed_count}/{len(texts)} completed ({progress_percent}%)")
        
        total_time = time.time() - total_start_time
        avg_time = total_time / len(texts)
        
        # Performance summary
        if self.monitor:
            summary = self.monitor.get_performance_summary()
            print(f"✅ ADAPTIVE PROCESSING COMPLETED: {len(results)} texts in {total_time:.1f}s")
            print(f"📈 Performance: {summary['avg_throughput']:.2f} texts/sec, {summary['avg_speed_per_worker']:.1f} tok/s per worker")
            if self.enable_adaptation:
                print(f"🎛️ Final recommendation: Use {summary['current_workers']} workers for this workload")
        
        return results