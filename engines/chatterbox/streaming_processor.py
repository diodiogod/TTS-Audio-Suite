"""
True Streaming Dynamic Worker Processor for ChatterBox TTS

This implements REAL dynamic processing where:
- Workers continuously pull from a shared work queue
- No waiting for entire batches to complete  
- Can add/remove workers mid-stream based on performance
- Workers that finish immediately grab the next available work
- True resource utilization and scalability

This is what the user actually wanted - not fake dynamic processing.
"""

import torch
import time
import threading
from queue import Queue, Empty
from typing import List, Optional, Callable
from dataclasses import dataclass
from collections import deque
import gc

@dataclass
class StreamingWorkItem:
    """Single text generation work item for streaming processing."""
    text_index: int
    text: str
    temperature: float
    cfg_weight: float
    exaggeration: float

@dataclass 
class StreamingResult:
    """Result from streaming text generation."""
    text_index: int
    audio: torch.Tensor
    completion_time: float
    worker_id: int
    success: bool
    error_msg: str = ""

class StreamingPerformanceMonitor:
    """
    Monitors real-time performance for streaming workers.
    Makes decisions about scaling workers up/down during processing.
    """
    
    def __init__(self, initial_workers: int, max_workers: int = 12):
        self.initial_workers = initial_workers
        self.current_target_workers = initial_workers
        self.max_workers = max_workers
        self.min_workers = 1
        
        # Performance tracking
        self.completion_times = deque(maxlen=20)  # Last 20 completions
        self.last_scale_decision = time.time()
        self.scale_cooldown = 3.0  # Wait 3s between scaling decisions (AGGRESSIVE FOR TESTING)
        
        # Memory monitoring (AGGRESSIVE FOR TESTING)
        self.memory_pressure_threshold = 0.65  # 65% triggers scale-down
        self.performance_decline_threshold = -0.15  # 15% decline triggers scale-down
        
        print(f"📊 Streaming Monitor: Target {initial_workers} workers, max {max_workers}")
        
    def record_completion(self, completion_time: float, active_workers: int, total_workload: int) -> Optional[int]:
        """
        Record completion and return new target worker count if scaling needed.
        
        Args:
            completion_time: Time to complete this work item
            active_workers: Number of workers currently active
            total_workload: Total number of texts in this batch (for context)
        
        Returns:
            New target worker count, or None if no change
        """
        current_time = time.time()
        
        # Calculate workload utilization ratio (how well-utilized were workers)
        utilization_ratio = min(1.0, total_workload / max(active_workers, 1))
        
        self.completion_times.append((current_time, completion_time, active_workers, utilization_ratio, total_workload))
        
        # SMART: Don't make scaling decisions on tiny batches with low utilization
        if utilization_ratio < 0.5:  # Less than 50% worker utilization
            print(f"🚫 SKIPPING SCALING: Low utilization ({utilization_ratio:.1%}) - not enough workload to judge performance")
            return None
            
        if total_workload < 5:  # Less than 5 texts total
            print(f"🚫 SKIPPING SCALING: Workload too small ({total_workload} texts) - performance data unreliable")
            return None
        
        # Don't scale too frequently
        if current_time - self.last_scale_decision < self.scale_cooldown:
            return None
            
        if len(self.completion_times) < 2:  # Need minimal data
            return None
            
        # Check memory pressure first (ALWAYS CHECK)
        memory_pressure = self._get_memory_pressure()
        print(f"💾 Memory pressure: {memory_pressure:.1%} (threshold: {self.memory_pressure_threshold:.1%})")
        if memory_pressure > self.memory_pressure_threshold:
            if self.current_target_workers > self.min_workers:
                self.current_target_workers = max(self.min_workers, self.current_target_workers - 1)
                self.last_scale_decision = current_time
                print(f"🔥 MEMORY PRESSURE {memory_pressure:.1%} → Scale DOWN to {self.current_target_workers} workers")
                return self.current_target_workers
                
        # Check performance trends - only use well-utilized data points
        well_utilized_data = [(t, ct, aw, ur, tw) for t, ct, aw, ur, tw in list(self.completion_times) if ur > 0.7]
        
        if len(well_utilized_data) >= 6:
            recent_times = [ct for _, ct, _, _, _ in well_utilized_data[-3:]]  # Last 3 well-utilized
            earlier_times = [ct for _, ct, _, _, _ in well_utilized_data[-6:-3]]  # Earlier 3 well-utilized
            
            recent_avg = sum(recent_times) / len(recent_times)
            earlier_avg = sum(earlier_times) / len(earlier_times)
            performance_change = (recent_avg - earlier_avg) / earlier_avg
            
            recent_utilization = sum(ur for _, _, _, ur, _ in well_utilized_data[-3:]) / 3
            print(f"📊 Performance trend: {performance_change:.1%} change, {recent_utilization:.1%} avg utilization")
            
            # STABILITY ZONE: Performance is acceptable - don't change anything
            if -0.10 <= performance_change <= 0.05 and 15.0 <= recent_avg <= 35.0:
                print(f"✅ STABLE PERFORMANCE: {performance_change:.1%} change, {recent_avg:.1f}s avg - keeping {self.current_target_workers} workers")
                return None  # Happy with current worker count!
            
            # Performance declining significantly with good utilization  
            elif performance_change > self.performance_decline_threshold and self.current_target_workers > self.min_workers:
                self.current_target_workers -= 1
                self.last_scale_decision = current_time
                print(f"📉 PERFORMANCE DECLINE {performance_change:.1%} → Scale DOWN to {self.current_target_workers} workers")
                return self.current_target_workers
                
            # Performance excellent and fast with high utilization - can we scale up?
            elif performance_change < -0.05 and recent_avg < 18.0 and memory_pressure < 0.7 and recent_utilization > 0.85:
                if self.current_target_workers < self.max_workers:
                    self.current_target_workers += 1  
                    self.last_scale_decision = current_time
                    print(f"📈 EXCELLENT PERFORMANCE ({recent_avg:.1f}s avg, {recent_utilization:.1%} util) → Scale UP to {self.current_target_workers} workers")
                    return self.current_target_workers
            
            # In between - not bad enough to scale down, not good enough to scale up
            else:
                print(f"🔄 ACCEPTABLE PERFORMANCE: {performance_change:.1%} change, {recent_avg:.1f}s avg - no change needed")
        else:
            print(f"📊 Not enough well-utilized data points ({len(well_utilized_data)}/6 needed) - no scaling decision")
                    
        return None
        
    def _get_memory_pressure(self) -> float:
        """Get current GPU memory pressure (0.0 to 1.0)."""
        try:
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory  
                return used / total
            return 0.4  # Assume 40% for CPU/MPS
        except:
            return 0.5

class StreamingWorker:
    """
    Individual streaming worker that continuously processes work from shared queue.
    Can be dynamically started/stopped without disrupting other workers.
    """
    
    def __init__(self, worker_id: int, work_queue: Queue, result_queue: Queue, 
                 tts_model, shutdown_event: threading.Event):
        self.worker_id = worker_id
        self.work_queue = work_queue
        self.result_queue = result_queue  
        self.tts_model = tts_model
        self.shutdown_event = shutdown_event
        self.thread = None
        self.is_active = False
        self.should_stop = False
        
    def start(self):
        """Start the worker thread."""
        if not self.is_active:
            self.should_stop = False
            self.thread = threading.Thread(target=self._work_loop)
            self.thread.start()
            self.is_active = True
            print(f"🚀 Worker {self.worker_id}: Started and ready for streaming work")
            
    def stop(self):
        """Signal worker to stop after current work (graceful shutdown)."""
        self.should_stop = True
        print(f"🛑 Worker {self.worker_id}: Marked for shutdown after current work")
        
    def join(self, timeout=5.0):
        """Wait for worker to finish current work and shutdown."""
        if self.thread:
            self.thread.join(timeout)
            self.is_active = False
            
    def _work_loop(self):
        """Main worker loop - continuously pulls work from queue."""
        print(f"🧵 Worker {self.worker_id}: Entering streaming work loop")
        
        while not self.shutdown_event.is_set() and not self.should_stop:
            try:
                # Pull next work item (with timeout to check shutdown)
                work_item = self.work_queue.get(timeout=1.0)
                
                if work_item is None:  # Poison pill - shutdown signal
                    break
                    
                # Process the work
                print(f"🔄 Worker {self.worker_id}: Processing text {work_item.text_index+1}: {work_item.text[:40]}...")
                
                start_time = time.time()
                try:
                    audio = self.tts_model.generate(
                        text=work_item.text,
                        audio_prompt_path=None,
                        exaggeration=work_item.exaggeration,
                        cfg_weight=work_item.cfg_weight,
                        temperature=work_item.temperature,
                    )
                    
                    completion_time = time.time() - start_time
                    result = StreamingResult(
                        text_index=work_item.text_index,
                        audio=audio,
                        completion_time=completion_time,
                        worker_id=self.worker_id,
                        success=True
                    )
                    
                    print(f"✅ Worker {self.worker_id}: Completed text {work_item.text_index+1} in {completion_time:.1f}s")
                    
                except Exception as e:
                    completion_time = time.time() - start_time
                    result = StreamingResult(
                        text_index=work_item.text_index,
                        audio=torch.zeros(1, 1000),
                        completion_time=completion_time,
                        worker_id=self.worker_id,
                        success=False,
                        error_msg=str(e)
                    )
                    print(f"❌ Worker {self.worker_id}: Failed text {work_item.text_index+1}: {e}")
                    
                # Send result back
                self.result_queue.put(result)
                self.work_queue.task_done()
                
                # Check if we should stop (graceful shutdown)
                if self.should_stop:
                    print(f"🛑 Worker {self.worker_id}: Gracefully stopping after completing work")
                    break
                    
            except Empty:
                # Timeout - check shutdown conditions and continue
                continue
            except Exception as e:
                print(f"❌ Worker {self.worker_id}: Unexpected error: {e}")
                
        print(f"🧵 Worker {self.worker_id}: Exiting work loop")

class TrueStreamingProcessor:
    """
    True streaming dynamic processor with real-time worker scaling.
    
    Workers continuously pull from shared work queue - no batch waiting.
    Can dynamically add/remove workers based on performance and memory.
    This provides true parallel efficiency and resource utilization.
    """
    
    def __init__(self, tts_model, initial_workers: int = 4, max_workers: int = 12):
        self.tts_model = tts_model
        self.initial_workers = initial_workers
        self.max_workers = max_workers
        
        # Streaming infrastructure
        self.work_queue = Queue()
        self.result_queue = Queue()
        self.shutdown_event = threading.Event()
        
        # Worker management
        self.workers = {}  # worker_id -> StreamingWorker
        self.next_worker_id = 1
        self.target_workers = initial_workers
        
        # Performance monitoring
        self.monitor = StreamingPerformanceMonitor(initial_workers, max_workers)
        
        print(f"🎯 True Streaming Processor: Ready for dynamic processing")
        
    def process_texts_streaming(
        self,
        texts: List[str],
        temperature: float,
        cfg_weight: float,
        exaggeration: float
    ) -> List[torch.Tensor]:
        """
        Process texts with true streaming dynamic workers.
        
        This is REAL dynamic processing:
        - Workers continuously pull from shared queue
        - No waiting for batch completion
        - Can scale workers up/down mid-process
        - Maximum resource utilization
        """
        if not texts:
            return []
            
        print(f"🌊 STREAMING PROCESSING: {len(texts)} texts with dynamic worker scaling")
        
        # Reset for new processing session
        self.shutdown_event.clear()
        self.target_workers = min(self.initial_workers, len(texts))
        
        # Fill work queue with all texts
        for i, text in enumerate(texts):
            work_item = StreamingWorkItem(
                text_index=i,
                text=text,
                temperature=temperature,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration
            )
            self.work_queue.put(work_item)
            
        print(f"📥 Queued {len(texts)} work items")
        
        # Start initial workers
        self._scale_to_target_workers()
        
        # Collect results with dynamic scaling
        results = [None] * len(texts)
        completed_count = 0
        start_time = time.time()
        
        while completed_count < len(texts):
            try:
                # Get next completion
                result = self.result_queue.get(timeout=3.0)
                results[result.text_index] = result.audio
                completed_count += 1
                
                # Monitor performance and scale workers if needed
                active_workers = len([w for w in self.workers.values() if w.is_active])
                new_target = self.monitor.record_completion(result.completion_time, active_workers, len(texts))
                
                if new_target and new_target != self.target_workers:
                    self.target_workers = new_target
                    self._scale_to_target_workers()
                    
                # Progress update
                progress = int(100 * completed_count / len(texts))
                active_count = len([w for w in self.workers.values() if w.is_active])
                print(f"📊 Progress: {completed_count}/{len(texts)} ({progress}%) - {active_count} active workers - Queue: {self.work_queue.qsize()} remaining")
                
            except Empty:
                print("⏳ Waiting for streaming results...")
                continue
                
        # Cleanup
        self._shutdown_all_workers()
        
        total_time = time.time() - start_time
        print(f"✅ STREAMING PROCESSING COMPLETED: {len(texts)} texts in {total_time:.1f}s")
        print(f"🎯 Average throughput: {len(texts) / total_time:.2f} texts/second")
        
        return results
        
    def _scale_to_target_workers(self):
        """Scale active workers to match target."""
        active_workers = [w for w in self.workers.values() if w.is_active]
        active_count = len(active_workers)
        
        if active_count < self.target_workers:
            # Need to add workers
            workers_to_add = self.target_workers - active_count
            for _ in range(workers_to_add):
                self._add_worker()
                
        elif active_count > self.target_workers:
            # Need to remove workers (gracefully)
            workers_to_remove = active_count - self.target_workers
            for _ in range(workers_to_remove):
                self._remove_worker()
                
    def _add_worker(self):
        """Add a new streaming worker."""
        worker_id = self.next_worker_id
        self.next_worker_id += 1
        
        worker = StreamingWorker(
            worker_id=worker_id,
            work_queue=self.work_queue,
            result_queue=self.result_queue,
            tts_model=self.tts_model,
            shutdown_event=self.shutdown_event
        )
        
        self.workers[worker_id] = worker
        worker.start()
        
    def _remove_worker(self):
        """Remove a worker gracefully (after current work)."""
        # Find an active worker to remove
        for worker in self.workers.values():
            if worker.is_active:
                worker.stop()  # Graceful stop after current work
                break
                
    def _shutdown_all_workers(self):
        """Shutdown all workers and clean up."""
        print("🛑 Shutting down all streaming workers...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Stop all workers gracefully
        for worker in self.workers.values():
            if worker.is_active:
                worker.stop()
                
        # Wait for workers to finish
        for worker in self.workers.values():
            worker.join(timeout=3.0)
            
        # Clear workers
        self.workers.clear()
        
        # Clear any remaining queue items
        while not self.work_queue.empty():
            try:
                self.work_queue.get_nowait()
            except Empty:
                break
                
        print("✅ All streaming workers shut down cleanly")