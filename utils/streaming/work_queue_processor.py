"""
Universal Work Queue Processor

Engine-agnostic parallel processing system that replaces engine-specific
streaming implementations with a unified approach.
"""

import torch
import time
import threading
import gc
from queue import Queue, Empty
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .streaming_types import StreamingSegment, StreamingResult, StreamingConfig, CharacterGroup
from .streaming_interface import StreamingEngineAdapter


class StreamingWorker:
    """
    Individual streaming worker that processes segments from a queue.
    
    Engine-agnostic worker that uses the adapter pattern to process segments
    with any TTS engine.
    """
    
    def __init__(
        self,
        worker_id: int,
        work_queue: Queue,
        result_queue: Queue,
        adapter: StreamingEngineAdapter,
        shutdown_event: threading.Event
    ):
        """
        Initialize streaming worker.
        
        Args:
            worker_id: Unique worker identifier
            work_queue: Queue to pull work from
            result_queue: Queue to send results to
            adapter: Engine adapter for processing segments
            shutdown_event: Event to signal shutdown
        """
        self.worker_id = worker_id
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.adapter = adapter
        self.shutdown_event = shutdown_event
        
        # Statistics
        self.segments_processed = 0
        self.total_processing_time = 0.0
        
    def process_segment(self, segment: StreamingSegment, **kwargs) -> StreamingResult:
        """
        Process a single segment using the engine adapter.
        
        Args:
            segment: StreamingSegment to process
            **kwargs: Additional processing parameters
            
        Returns:
            StreamingResult with generated audio
        """
        start_time = time.time()
        
        try:
            # Validate segment with adapter
            if not self.adapter.validate_segment(segment):
                raise ValueError(f"Invalid segment for {self.adapter.engine_name}")
            
            # Process segment through adapter
            result = self.adapter.process_segment(segment, **kwargs)
            
            # Update result with worker info
            result.worker_id = self.worker_id
            result.processing_time = time.time() - start_time
            
            self.segments_processed += 1
            self.total_processing_time += result.processing_time
            
            return result
            
        except Exception as e:
            # Return error result
            processing_time = time.time() - start_time
            return StreamingResult(
                index=segment.index,
                audio=torch.zeros(1, 1000),  # Empty audio fallback
                duration=0.0,
                character=segment.character,
                language=segment.language,
                processing_time=processing_time,
                worker_id=self.worker_id,
                success=False,
                error_msg=str(e)
            )
    
    def run(self):
        """Main worker loop - continuously processes segments from queue."""
        print(f"🧵 Worker {self.worker_id}: Started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get next work item with timeout
                work_item = self.work_queue.get(timeout=1.0)
                
                if work_item is None:  # Shutdown signal
                    break
                
                # Unpack work item
                segment, kwargs = work_item
                
                # Process segment
                print(f"🔄 Worker {self.worker_id}: Processing segment {segment.index} ({segment.character} in {segment.language})")
                result = self.process_segment(segment, **kwargs)
                
                # Send result back
                self.result_queue.put(result)
                self.work_queue.task_done()
                
                # Log progress every 5 segments
                if self.segments_processed % 5 == 0:
                    avg_time = self.total_processing_time / self.segments_processed
                    print(f"👷 Worker {self.worker_id}: {self.segments_processed} segments processed (avg {avg_time:.2f}s)")
                
            except Empty:
                continue  # Timeout, check shutdown and continue
            except Exception as e:
                print(f"❌ Worker {self.worker_id}: Unexpected error: {e}")
                
        print(f"🛑 Worker {self.worker_id}: Shutdown complete ({self.segments_processed} segments)")


class UniversalWorkQueueProcessor:
    """
    Universal work queue processor for parallel segment processing.
    
    Replaces all engine-specific streaming processors with a single,
    engine-agnostic implementation.
    """
    
    def __init__(
        self,
        adapter: StreamingEngineAdapter,
        max_workers: int = 4,
        timeout: float = 300.0
    ):
        """
        Initialize processor with engine adapter.
        
        Args:
            adapter: Engine adapter for processing segments
            max_workers: Maximum number of parallel workers
            timeout: Worker timeout in seconds
        """
        self.adapter = adapter
        self.max_workers = max_workers
        self.timeout = timeout
        
        # Queue system
        self.work_queue = Queue()
        self.result_queue = Queue()
        self.shutdown_event = threading.Event()
        
        # Worker management
        self.workers = []
        self.worker_threads = []
        
        print(f"🌊 UniversalWorkQueueProcessor: Ready with {max_workers} workers for {adapter.engine_name}")
    
    def build_work_queue(
        self,
        segments: List[StreamingSegment],
        language_character_groups: Dict[str, Dict[str, CharacterGroup]],
        **kwargs
    ) -> int:
        """
        Build work queue maintaining optimal language->character ordering.
        
        Args:
            segments: List of segments to process
            language_character_groups: Segments grouped by language and character
            **kwargs: Additional processing parameters
            
        Returns:
            Total number of work items queued
        """
        total_items = 0
        
        # Maintain optimal ordering: language -> character -> segments
        for lang_code, char_groups in language_character_groups.items():
            for character, char_group in char_groups.items():
                for segment in char_group.segments:
                    # Queue segment with processing parameters
                    work_item = (segment, kwargs)
                    self.work_queue.put(work_item)
                    total_items += 1
        
        print(f"📥 Queued {total_items} segments across {len(language_character_groups)} languages")
        return total_items
    
    def start_workers(self):
        """Start all worker threads."""
        self.shutdown_event.clear()
        
        for worker_id in range(self.max_workers):
            worker = StreamingWorker(
                worker_id=worker_id + 1,
                work_queue=self.work_queue,
                result_queue=self.result_queue,
                adapter=self.adapter,
                shutdown_event=self.shutdown_event
            )
            
            worker_thread = threading.Thread(target=worker.run)
            worker_thread.start()
            
            self.workers.append(worker)
            self.worker_threads.append(worker_thread)
        
        print(f"🚀 Started {self.max_workers} streaming workers")
    
    def collect_results(self, expected_count: int) -> Dict[int, StreamingResult]:
        """
        Collect results from workers as they complete.
        
        Args:
            expected_count: Number of results to collect
            
        Returns:
            Dict mapping segment index -> StreamingResult
        """
        results = {}
        completed_count = 0
        start_time = time.time()
        last_progress_time = start_time
        
        while completed_count < expected_count:
            try:
                # Get next result with timeout
                result = self.result_queue.get(timeout=5.0)
                
                # Store result by original index
                results[result.index] = result
                completed_count += 1
                
                # Progress reporting every 10% or 5 seconds
                current_time = time.time()
                progress_percent = int(100 * completed_count / expected_count)
                
                should_report = (
                    completed_count % max(1, expected_count // 10) == 0 or
                    current_time - last_progress_time > 5.0
                )
                
                if should_report:
                    elapsed = current_time - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    remaining = (expected_count - completed_count) / rate if rate > 0 else 0
                    
                    print(f"📊 Progress: {completed_count}/{expected_count} ({progress_percent}%) "
                          f"- {rate:.1f} segments/sec - ETA: {remaining:.0f}s")
                    last_progress_time = current_time
                
            except Empty:
                print("⏳ Waiting for streaming results...")
                # Check if workers are still alive
                alive_workers = sum(1 for t in self.worker_threads if t.is_alive())
                if alive_workers == 0:
                    print("❌ All workers have stopped unexpectedly")
                    break
                continue
        
        return results
    
    def shutdown_workers(self):
        """Shutdown all workers gracefully."""
        print("🛑 Shutting down streaming workers...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Send shutdown signals to queue
        for _ in range(self.max_workers):
            self.work_queue.put(None)
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        # Clear workers
        self.workers.clear()
        self.worker_threads.clear()
        
        # Clear any remaining queue items
        while not self.work_queue.empty():
            try:
                self.work_queue.get_nowait()
            except Empty:
                break
                
        print("✅ All streaming workers shut down")
    
    def process_segments(
        self,
        segments: List[StreamingSegment],
        language_character_groups: Dict[str, Dict[str, CharacterGroup]],
        config: StreamingConfig,
        **kwargs
    ) -> Dict[int, StreamingResult]:
        """
        Main processing method - processes segments with parallel workers.
        
        Args:
            segments: List of segments to process
            language_character_groups: Segments grouped by language and character
            config: StreamingConfig with processing settings
            **kwargs: Additional processing parameters
            
        Returns:
            Dict mapping segment index -> StreamingResult
        """
        try:
            # Build work queue
            total_items = self.build_work_queue(segments, language_character_groups, **kwargs)
            
            # Start workers
            self.start_workers()
            
            # Collect results
            print(f"🌊 Processing {total_items} segments with {self.max_workers} workers...")
            start_time = time.time()
            
            results = self.collect_results(total_items)
            
            total_time = time.time() - start_time
            throughput = total_items / total_time if total_time > 0 else 0
            
            print(f"✅ Streaming complete: {total_items} segments in {total_time:.1f}s ({throughput:.2f} segments/sec)")
            
            return results
            
        finally:
            # Always cleanup workers
            self.shutdown_workers()
            
            # Force garbage collection to free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()