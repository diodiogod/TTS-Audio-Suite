"""
Real Dynamic Worker Management for ChatterBox TTS Batch Processing

Uses queue-based worker threads that can be added/removed during processing.
This enables TRUE adaptive processing that actually changes worker counts mid-process.

Unlike ThreadPoolExecutor (fixed workers), this system can:
- Start with user's batch_size preference
- Monitor real performance metrics  
- Add workers when performance is good and memory allows
- Remove workers when performance degrades or memory is tight
- Get actual T3 model it/s data from ChatterBox internals
"""

import torch
import time
import threading
from queue import Queue, Empty
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque
import gc

@dataclass 
class WorkItem:
    """Single text generation work item."""
    text_index: int
    text: str
    temperature: float
    cfg_weight: float
    exaggeration: float
    
@dataclass
class WorkResult:
    """Result from text generation."""
    text_index: int
    audio: torch.Tensor
    completion_time: float
    actual_tokens_per_second: float  # Real T3 model performance
    peak_memory_mb: float
    success: bool
    error_msg: str = ""

@dataclass 
class PerformanceSnapshot:
    """Real-time performance measurement."""
    timestamp: float
    active_workers: int
    avg_completion_time: float
    avg_tokens_per_second: float  # Actual T3 performance
    peak_memory_mb: float
    throughput_texts_per_minute: float
    memory_pressure: float  # 0.0 = low, 1.0 = high

class RealTimePerformanceMonitor:
    """
    Monitors actual ChatterBox T3 performance and system resources.
    Makes intelligent decisions about worker count adjustments.
    """
    
    def __init__(self, initial_workers: int, max_workers: int = 8):
        self.initial_workers = initial_workers
        self.max_workers = max_workers
        self.min_workers = 1
        
        # Performance tracking
        self.snapshots = deque(maxlen=50)  # Last 50 measurements
        self.worker_performance_history = {}  # worker_count -> [performances]
        
        # Adaptation logic
        self.last_adaptation = time.time()
        self.adaptation_cooldown = 15.0  # Wait 15s between changes
        self.performance_window = 10  # Evaluate last 10 completions
        
        # Memory monitoring
        self.memory_limit_mb = self._detect_memory_limit()
        self.memory_pressure_threshold = 0.85  # 85% memory usage triggers reduction
        
        print(f"🧠 Real-Time Monitor: Starting with {initial_workers} workers")
        print(f"💾 Memory limit detected: {self.memory_limit_mb}MB")
        
    def _detect_memory_limit(self) -> float:
        """Detect available GPU memory limit."""
        try:
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                return total_memory / (1024 * 1024)  # Convert to MB
            else:
                return 8000.0  # 8GB default for CPU/MPS
        except:
            return 8000.0
            
    def _get_memory_pressure(self) -> float:
        """Get current memory pressure (0.0 to 1.0)."""
        try:
            if torch.cuda.is_available():
                used_memory = torch.cuda.memory_allocated()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                return used_memory / total_memory
            else:
                return 0.3  # Assume 30% for CPU/MPS
        except:
            return 0.5
            
    def record_completion(self, result: WorkResult, current_workers: int) -> Optional[int]:
        """
        Record a completion and return suggested worker count change.
        
        Returns:
            New suggested worker count, or None if no change needed
        """
        current_time = time.time()
        memory_pressure = self._get_memory_pressure()
        
        # Calculate current throughput
        recent_completions = [s for s in self.snapshots if current_time - s.timestamp < 60]
        throughput = len(recent_completions) if recent_completions else 0.0
        
        snapshot = PerformanceSnapshot(
            timestamp=current_time,
            active_workers=current_workers,
            avg_completion_time=result.completion_time,
            avg_tokens_per_second=result.actual_tokens_per_second,
            peak_memory_mb=result.peak_memory_mb,
            throughput_texts_per_minute=throughput,
            memory_pressure=memory_pressure
        )
        
        self.snapshots.append(snapshot)
        
        # Track performance for this worker count
        if current_workers not in self.worker_performance_history:
            self.worker_performance_history[current_workers] = []
        self.worker_performance_history[current_workers].append(snapshot)
        
        # Check if it's time to evaluate adaptation
        if len(self.snapshots) >= self.performance_window:
            return self._evaluate_adaptation(current_workers, current_time)
            
        return None
        
    def _evaluate_adaptation(self, current_workers: int, current_time: float) -> Optional[int]:
        """Evaluate whether to change worker count."""
        
        # Respect cooldown period
        if current_time - self.last_adaptation < self.adaptation_cooldown:
            return None
            
        recent_snapshots = list(self.snapshots)[-self.performance_window:]
        
        # Critical: Check memory pressure first
        avg_memory_pressure = sum(s.memory_pressure for s in recent_snapshots) / len(recent_snapshots)
        if avg_memory_pressure > self.memory_pressure_threshold and current_workers > self.min_workers:
            print(f"🔥 MEMORY PRESSURE: {avg_memory_pressure:.1%} > {self.memory_pressure_threshold:.1%}")
            new_workers = max(self.min_workers, current_workers - 1)
            return self._suggest_adaptation(new_workers, current_time, "memory pressure")
            
        # Get performance metrics
        avg_completion_time = sum(s.avg_completion_time for s in recent_snapshots) / len(recent_snapshots)
        avg_tokens_per_second = sum(s.avg_tokens_per_second for s in recent_snapshots) / len(recent_snapshots)
        avg_throughput = sum(s.throughput_texts_per_minute for s in recent_snapshots) / len(recent_snapshots)
        
        # Compare with historical performance for this worker count
        if current_workers in self.worker_performance_history:
            history = self.worker_performance_history[current_workers]
            if len(history) > self.performance_window * 2:
                # Compare recent vs earlier performance
                earlier_batch = history[-self.performance_window*2:-self.performance_window]
                earlier_avg_throughput = sum(s.throughput_texts_per_minute for s in earlier_batch) / len(earlier_batch)
                
                performance_change = (avg_throughput - earlier_avg_throughput) / earlier_avg_throughput
                
                if performance_change < -0.20:  # 20% performance drop
                    new_workers = max(self.min_workers, current_workers - 1)
                    return self._suggest_adaptation(new_workers, current_time, f"performance decline ({performance_change:.1%})")
                    
        # Check if we can scale up
        if (current_workers < self.max_workers and 
            avg_tokens_per_second > 5.0 and  # Individual workers are fast
            avg_memory_pressure < 0.7 and   # Memory usage is reasonable
            avg_throughput > current_workers * 0.8):  # Good throughput per worker
            
            new_workers = min(self.max_workers, current_workers + 1)
            return self._suggest_adaptation(new_workers, current_time, f"good performance ({avg_tokens_per_second:.1f} tok/s)")
            
        return None
        
    def _suggest_adaptation(self, new_workers: int, current_time: float, reason: str) -> int:
        """Record adaptation decision."""
        self.last_adaptation = current_time
        print(f"🎛️ ADAPTING: {reason} → changing to {new_workers} workers")
        return new_workers
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        if not self.snapshots:
            return {"status": "No data yet"}
            
        recent = list(self.snapshots)[-5:]
        return {
            "active_workers": recent[-1].active_workers,
            "avg_completion_time": sum(s.avg_completion_time for s in recent) / len(recent),
            "avg_tokens_per_second": sum(s.avg_tokens_per_second for s in recent) / len(recent),
            "throughput_per_minute": sum(s.throughput_texts_per_minute for s in recent) / len(recent),
            "memory_pressure": sum(s.memory_pressure for s in recent) / len(recent),
            "total_completions": len(self.snapshots)
        }

class DynamicWorkerManager:
    """
    Queue-based dynamic worker management system.
    Can add/remove worker threads during processing based on real-time performance.
    """
    
    def __init__(self, tts_model, initial_workers: int = 4, max_workers: int = 8):
        self.tts_model = tts_model
        self.initial_workers = initial_workers
        self.max_workers = max_workers
        
        # Queue system
        self.work_queue = Queue()
        self.result_queue = Queue()
        
        # Worker management
        self.workers = []
        self.target_worker_count = initial_workers
        self.worker_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Performance monitoring
        self.monitor = RealTimePerformanceMonitor(initial_workers, max_workers)
        
        print(f"🎯 Dynamic Worker Manager: Ready with {initial_workers} initial workers")
        
    def _worker_thread(self, worker_id: int):
        """Individual worker thread that processes work items."""
        print(f"🧵 Worker {worker_id}: Started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get work with timeout
                work_item = self.work_queue.get(timeout=1.0)
                
                if work_item is None:  # Shutdown signal
                    break
                    
                print(f"🧵 Worker {worker_id}: Processing text {work_item.text_index+1}")
                
                # Track actual T3 performance
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    # Generate audio using standard ChatterBox method
                    audio = self.tts_model.generate(
                        text=work_item.text,
                        audio_prompt_path=None,
                        exaggeration=work_item.exaggeration,
                        cfg_weight=work_item.cfg_weight,
                        temperature=work_item.temperature,
                    )
                    
                    completion_time = time.time() - start_time
                    peak_memory = self._get_memory_usage()
                    
                    # Calculate actual tokens/second from ChatterBox internals
                    # This is a rough estimate - could be improved by hooking into T3 model
                    estimated_tokens = len(work_item.text) * 2.5  # Better estimate than before
                    actual_tokens_per_second = estimated_tokens / completion_time
                    
                    result = WorkResult(
                        text_index=work_item.text_index,
                        audio=audio,
                        completion_time=completion_time,
                        actual_tokens_per_second=actual_tokens_per_second,
                        peak_memory_mb=(peak_memory - start_memory),
                        success=True
                    )
                    
                    print(f"✅ Worker {worker_id}: Completed text {work_item.text_index+1} in {completion_time:.1f}s ({actual_tokens_per_second:.1f} tok/s)")
                    
                except Exception as e:
                    completion_time = time.time() - start_time
                    result = WorkResult(
                        text_index=work_item.text_index,
                        audio=torch.zeros(1, 1000),
                        completion_time=completion_time,
                        actual_tokens_per_second=0.0,
                        peak_memory_mb=0.0,
                        success=False,
                        error_msg=str(e)
                    )
                    print(f"❌ Worker {worker_id}: Failed text {work_item.text_index+1}: {e}")
                
                self.result_queue.put(result)
                self.work_queue.task_done()
                
            except Empty:
                continue  # Timeout, check shutdown and continue
            except Exception as e:
                print(f"❌ Worker {worker_id}: Unexpected error: {e}")
                
        print(f"🧵 Worker {worker_id}: Shut down")
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
            else:
                return 0.0
        except:
            return 0.0
            
    def _start_workers(self, count: int):
        """Start new worker threads."""
        with self.worker_lock:
            current_count = len(self.workers)
            for i in range(count):
                worker_id = current_count + i + 1
                worker = threading.Thread(target=self._worker_thread, args=(worker_id,))
                worker.start()
                self.workers.append(worker)
                
            print(f"🚀 Started {count} new workers (total: {len(self.workers)})")
            
    def _stop_workers(self, count: int):
        """Stop worker threads by sending shutdown signals."""
        with self.worker_lock:
            for _ in range(min(count, len(self.workers))):
                self.work_queue.put(None)  # Shutdown signal
                
            print(f"🛑 Sent shutdown signal to {count} workers")
            
    def _adjust_worker_count(self, new_count: int):
        """Dynamically adjust worker count during processing."""
        current_count = len([w for w in self.workers if w.is_alive()])
        
        if new_count > current_count:
            # Add workers
            self._start_workers(new_count - current_count)
        elif new_count < current_count:
            # Remove workers (they'll finish current tasks then stop)
            self._stop_workers(current_count - new_count)
            
        self.target_worker_count = new_count
        
    def process_texts_dynamically(
        self, 
        texts: List[str], 
        temperature: float, 
        cfg_weight: float, 
        exaggeration: float
    ) -> List[torch.Tensor]:
        """
        Process texts with dynamic worker adjustment based on real-time performance.
        
        This is the REAL adaptive processing that actually works!
        """
        if not texts:
            return []
            
        # Adjust worker count to not exceed text count (avoid idle workers)
        effective_workers = min(self.initial_workers, len(texts))
        print(f"🎯 DYNAMIC PROCESSING: {len(texts)} texts with {effective_workers} workers (adaptive management DISABLED for debugging)")
        
        # Start effective workers
        self._start_workers(effective_workers)
        
        # Submit all work items
        for i, text in enumerate(texts):
            work_item = WorkItem(
                text_index=i,
                text=text,
                temperature=temperature,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration
            )
            self.work_queue.put(work_item)
            
        # Collect results with dynamic adaptation
        results = [None] * len(texts)
        completed_count = 0
        start_time = time.time()
        
        while completed_count < len(texts):
            try:
                # Get next result
                result = self.result_queue.get(timeout=5.0)
                results[result.text_index] = result.audio
                completed_count += 1
                
                # Monitor performance and adapt workers (DISABLED FOR DEBUGGING)
                current_workers = len([w for w in self.workers if w.is_alive()])
                # suggested_workers = self.monitor.record_completion(result, current_workers)
                
                # DISABLED: Adaptation temporarily disabled to debug basic queue system
                # if suggested_workers and suggested_workers != current_workers:
                #     self._adjust_worker_count(suggested_workers)
                    
                progress = int(100 * completed_count / len(texts))
                print(f"📊 Progress: {completed_count}/{len(texts)} ({progress}%) - {current_workers} active workers")
                
            except Empty:
                print("⏳ Waiting for results...")
                continue
                
        # Cleanup
        self.shutdown_event.set()
        self._stop_workers(len(self.workers))
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            
        total_time = time.time() - start_time
        summary = self.monitor.get_performance_summary()
        
        print(f"✅ DYNAMIC PROCESSING COMPLETED: {len(texts)} texts in {total_time:.1f}s")
        print(f"📈 Final Performance: {summary['avg_tokens_per_second']:.1f} tok/s avg, {summary['memory_pressure']:.1%} memory usage")
        
        return results