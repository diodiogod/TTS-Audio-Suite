"""
Continuous Worker Processor - True Non-Stop Parallelization for ChatterBox

This implements what the user actually wanted:
- Workers continuously process work without waiting for batches
- No idle workers when user sets high worker counts
- Maintains the beautiful logging structure from the original
- Falls back gracefully to sequential processing for 0-1 workers
"""

import torch
import time
import threading
from queue import Queue, Empty
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import gc

@dataclass
class ContinuousWorkItem:
    """Single work item for continuous processing."""
    text_index: int
    text: str
    temperature: float
    cfg_weight: float
    exaggeration: float

@dataclass
class ContinuousResult:
    """Result from continuous processing."""
    text_index: int
    audio: torch.Tensor
    processing_time: float
    worker_id: int
    success: bool = True
    error_msg: str = ""

class ContinuousWorkerProcessor:
    """
    True continuous worker processor that maintains workers always busy.
    Falls back to sequential processing gracefully for 0-1 workers.
    Maintains beautiful logging structure.
    """
    
    def __init__(self, tts_model):
        self.tts_model = tts_model
        
    def process_texts_continuously(
        self,
        texts: List[str],
        temperature: float,
        cfg_weight: float, 
        exaggeration: float,
        max_workers: int
    ) -> List[torch.Tensor]:
        """
        Process texts with continuous workers or fallback to sequential.
        
        Args:
            texts: List of texts to process
            temperature: Sampling temperature
            cfg_weight: CFG weight
            exaggeration: Exaggeration factor
            max_workers: Number of workers (0-1 = sequential, >1 = continuous)
        
        Returns:
            List of audio tensors
        """
        if not texts:
            return []
            
        # SMART FALLBACK: Use sequential processing for 0-1 workers
        if max_workers <= 1:
            print(f"→ SEQUENTIAL FALLBACK: {len(texts)} texts (workers={max_workers})")
            return self._process_sequential(texts, temperature, cfg_weight, exaggeration)
        
        # Use continuous worker processing for >1 workers
        print(f"🌊 CONTINUOUS PROCESSING: {len(texts)} texts with {max_workers} workers")
        return self._process_with_continuous_workers(texts, temperature, cfg_weight, exaggeration, max_workers)
    
    def _process_sequential(
        self,
        texts: List[str], 
        temperature: float,
        cfg_weight: float,
        exaggeration: float
    ) -> List[torch.Tensor]:
        """Sequential processing - clean and organized like the original."""
        results = []
        
        for i, text in enumerate(texts):
            print(f"  🎤 Sequential {i+1}/{len(texts)}: {text[:40]}...")
            
            try:
                audio = self.tts_model.generate(
                    text=text,
                    audio_prompt_path=None,  # Already prepared
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    exaggeration=exaggeration
                )
                results.append(audio)
                print(f"  ✅ Sequential {i+1}: Completed")
                
            except Exception as e:
                print(f"  ❌ Sequential {i+1}: Failed - {e}")
                results.append(torch.zeros(1, 1000))
        
        print(f"✅ SEQUENTIAL COMPLETE: {len(texts)} texts processed")
        return results
    
    def _process_with_continuous_workers(
        self,
        texts: List[str],
        temperature: float,
        cfg_weight: float, 
        exaggeration: float,
        max_workers: int
    ) -> List[torch.Tensor]:
        """
        Continuous worker processing - workers never idle.
        Maintains beautiful logging structure.
        """
        # Build work queue
        work_queue = Queue()
        result_queue = Queue()
        shutdown_event = threading.Event()
        
        # Fill queue with all work
        for i, text in enumerate(texts):
            work_item = ContinuousWorkItem(
                text_index=i,
                text=text,
                temperature=temperature,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration
            )
            work_queue.put(work_item)
        
        print(f"📥 Work queue filled: {len(texts)} items")
        
        # Start workers
        workers = []
        worker_threads = []
        
        for worker_id in range(max_workers):
            worker = ContinuousWorker(
                worker_id=worker_id + 1,
                work_queue=work_queue,
                result_queue=result_queue,
                tts_model=self.tts_model,
                shutdown_event=shutdown_event
            )
            
            thread = threading.Thread(target=worker.run)
            thread.start()
            
            workers.append(worker)
            worker_threads.append(thread)
        
        print(f"🚀 Started {max_workers} continuous workers")
        
        # Collect results
        results = [None] * len(texts)
        completed_count = 0
        start_time = time.time()
        
        while completed_count < len(texts):
            try:
                result = result_queue.get(timeout=5.0)
                results[result.text_index] = result.audio
                completed_count += 1
                
                # Beautiful progress logging (like original)
                if completed_count % max(1, len(texts) // 10) == 0:
                    progress = int(100 * completed_count / len(texts))
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    print(f"📊 Continuous progress: {completed_count}/{len(texts)} ({progress}%) - {rate:.1f}/sec")
                
            except Empty:
                print("⏳ Waiting for continuous workers...")
                continue
        
        # Shutdown workers
        shutdown_event.set()
        
        # Send shutdown signals
        for _ in range(max_workers):
            work_queue.put(None)
        
        # Wait for threads
        for thread in worker_threads:
            thread.join(timeout=3.0)
        
        total_time = time.time() - start_time
        throughput = len(texts) / total_time if total_time > 0 else 0
        
        print(f"✅ CONTINUOUS COMPLETE: {len(texts)} texts in {total_time:.1f}s ({throughput:.2f}/sec)")
        
        return results

class ContinuousWorker:
    """Individual continuous worker that processes from shared queue."""
    
    def __init__(self, worker_id: int, work_queue: Queue, result_queue: Queue, 
                 tts_model, shutdown_event: threading.Event):
        self.worker_id = worker_id
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.tts_model = tts_model
        self.shutdown_event = shutdown_event
        self.processed_count = 0
        
    def run(self):
        """Main worker loop - continuously processes work."""
        print(f"  🧵 Worker {self.worker_id}: Ready for continuous work")
        
        while not self.shutdown_event.is_set():
            try:
                # Get next work item
                work_item = self.work_queue.get(timeout=1.0)
                
                if work_item is None:  # Shutdown signal
                    break
                
                print(f"  🧵 Worker {self.worker_id}: Processing: {work_item.text[:30]}...")
                
                # Process work item
                start_time = time.time()
                try:
                    audio = self.tts_model.generate(
                        text=work_item.text,
                        audio_prompt_path=None,  # Already prepared
                        temperature=work_item.temperature,
                        cfg_weight=work_item.cfg_weight,
                        exaggeration=work_item.exaggeration
                    )
                    
                    processing_time = time.time() - start_time
                    self.processed_count += 1
                    
                    result = ContinuousResult(
                        text_index=work_item.text_index,
                        audio=audio,
                        processing_time=processing_time,
                        worker_id=self.worker_id,
                        success=True
                    )
                    
                    print(f"  ✅ Worker {self.worker_id}: Completed in {processing_time:.1f}s")
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    result = ContinuousResult(
                        text_index=work_item.text_index,
                        audio=torch.zeros(1, 1000),
                        processing_time=processing_time,
                        worker_id=self.worker_id,
                        success=False,
                        error_msg=str(e)
                    )
                    
                    print(f"  ❌ Worker {self.worker_id}: Failed - {e}")
                
                # Send result back
                self.result_queue.put(result)
                self.work_queue.task_done()
                
            except Empty:
                # Timeout - check shutdown and continue
                continue
            except Exception as e:
                print(f"  ❌ Worker {self.worker_id}: Unexpected error - {e}")
        
        print(f"  🛑 Worker {self.worker_id}: Shutdown ({self.processed_count} processed)")