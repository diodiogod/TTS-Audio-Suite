"""
Streaming Work Queue Processor for ChatterBox TTS

Implements efficient work queue where workers continuously pull work across characters and languages.
Maintains optimal language->character ordering while maximizing worker utilization.

Key improvements:
- All workers stay busy (no idle workers when user sets 24 workers)
- No waiting for character groups to complete
- Smart model/voice switching
- Memory management based on worker count
"""

import torch
import time
import threading
import gc
from queue import Queue, Empty
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class WorkItem:
    """Single work item for the streaming queue."""
    segment_id: str
    text: str
    character: str
    language: str
    voice_path: str
    original_index: int
    exaggeration: float
    temperature: float
    cfg_weight: float

@dataclass
class WorkResult:
    """Result from processing a work item."""
    original_index: int
    audio: torch.Tensor
    worker_id: int
    character: str
    language: str
    processing_time: float
    success: bool = True
    error_msg: str = ""

class MultiModelTTSPool:
    """
    Manages multiple TTS model instances for different languages.
    Handles memory-efficient model loading/unloading based on worker count.
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.model_cache = {}  # language -> TTS model instance
        self.model_memory_usage = {}  # language -> estimated memory MB
        self.available_languages = set()
        self.memory_threshold = 0.7  # Use max 70% of available memory
        
    def estimate_available_memory(self) -> float:
        """Estimate available GPU memory in MB."""
        try:
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                used_memory = torch.cuda.memory_allocated()
                available = (total_memory - used_memory) / (1024 * 1024)  # Convert to MB
                return available * self.memory_threshold
            else:
                return 8000.0  # 8GB default for CPU/MPS
        except:
            return 4000.0  # 4GB conservative fallback
    
    def should_load_all_models(self, languages: List[str]) -> bool:
        """DON'T pre-load models - use existing model management system."""
        print(f"🎯 Using existing model management - no pre-loading of {len(languages)} models")
        return False
    
    def load_model(self, language: str, device: str = "auto"):
        """DON'T load models here - this should never be called now."""
        raise NotImplementedError("Model loading should be handled by existing ChatterBox node model management")
    
    def get_model(self, language: str, device: str = "auto"):
        """Get model for language, loading if necessary."""
        if language not in self.model_cache:
            return self.load_model(language, device)
        return self.model_cache[language]
    
    def unload_model(self, language: str):
        """Unload a specific model to free memory."""
        if language in self.model_cache:
            print(f"🗑️ Unloading {language} model to free memory")
            del self.model_cache[language]
            if language in self.model_memory_usage:
                del self.model_memory_usage[language]
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def cleanup(self):
        """Clean up all loaded models."""
        print("🧹 Cleaning up all TTS models")
        for language in list(self.model_cache.keys()):
            self.unload_model(language)

class StreamingWorker:
    """
    Simplified streaming worker that uses existing TTS model and voice management.
    Just pulls work from queue and processes using the node's existing systems.
    """
    
    def __init__(self, worker_id: int, work_queue: Queue, result_queue: Queue, 
                 tts_node, shutdown_event: threading.Event):
        self.worker_id = worker_id
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.tts_node = tts_node  # The ChatterBox TTS node instance
        self.shutdown_event = shutdown_event
        
        # Statistics
        self.segments_processed = 0
        
    def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Process a single work item using existing node's segment processing."""
        start_time = time.time()
        
        try:
            print(f"🧵 Worker {self.worker_id}: Processing {work_item.segment_id}")
            
            # Use the existing node's segment processing method
            # This delegates to all the existing model loading, voice management, etc.
            audio = self.tts_node._process_single_segment_for_streaming(
                original_idx=work_item.original_index,
                character=work_item.character,
                segment_text=work_item.text,
                language=work_item.language,
                voice_path=work_item.voice_path,
                inputs={
                    "exaggeration": work_item.exaggeration,
                    "temperature": work_item.temperature,
                    "cfg_weight": work_item.cfg_weight,
                    "enable_chunking": False,  # Don't chunk in streaming - already handled
                    "enable_audio_cache": True,
                    "crash_protection_template": "hmm ,, {seg} hmm ,,",
                    "device": "auto"
                }
            )
            
            processing_time = time.time() - start_time
            self.segments_processed += 1
            
            return WorkResult(
                original_index=work_item.original_index,
                audio=audio,
                worker_id=self.worker_id,
                character=work_item.character,
                language=work_item.language,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Worker {self.worker_id}: Failed processing {work_item.segment_id}: {e}")
            
            return WorkResult(
                original_index=work_item.original_index,
                audio=torch.zeros(1, 1000),  # Empty audio fallback
                worker_id=self.worker_id,
                character=work_item.character,
                language=work_item.language,
                processing_time=processing_time,
                success=False,
                error_msg=str(e)
            )
    
    def run(self):
        """Main worker loop - continuously processes work from queue."""        
        while not self.shutdown_event.is_set():
            try:
                # Get next work item
                work_item = self.work_queue.get(timeout=1.0)
                
                if work_item is None:  # Shutdown signal
                    break
                
                # Process work item
                result = self.process_work_item(work_item)
                
                # Send result back
                self.result_queue.put(result)
                self.work_queue.task_done()
                
                # Log progress
                if self.segments_processed % 5 == 0:  # Every 5 segments
                    print(f"👷 Worker {self.worker_id}: {self.segments_processed} segments processed")
                
            except Empty:
                continue  # Timeout, check shutdown and continue
            except Exception as e:
                print(f"❌ Worker {self.worker_id}: Unexpected error: {e}")
                
        print(f"🛑 Worker {self.worker_id}: Shutdown complete ({self.segments_processed} segments)")

class StreamingWorkQueueProcessor:
    """
    Main streaming processor that manages the work queue and worker pool.
    Implements the efficient streaming approach with optimal worker utilization.
    """
    
    def __init__(self, max_workers: int = 4, tts_node=None):
        self.max_workers = max_workers
        self.tts_node = tts_node  # The ChatterBox TTS node instance
        
        # Queue system
        self.work_queue = Queue()
        self.result_queue = Queue()
        self.shutdown_event = threading.Event()
        
        # Worker management
        self.workers = []
        self.worker_threads = []
        
        print(f"🌊 StreamingWorkQueueProcessor: Ready with {max_workers} workers using existing model management")
    
    def build_work_queue(
        self,
        language_groups: Dict[str, Any],
        character_groups_by_lang: Dict[str, Dict[str, Any]],
        voice_refs: Dict[str, str],
        inputs: Dict[str, Any]
    ) -> int:
        """
        Build the streaming work queue maintaining optimal language->character ordering.
        """
        total_items = 0
        
        # Maintain the optimal ordering: language -> character -> segments
        for lang_code, lang_segments in language_groups.items():
            character_groups = character_groups_by_lang[lang_code]
            
            for character, character_group in character_groups.items():
                for segment in character_group.segments:
                    work_item = WorkItem(
                        segment_id=f"{lang_code}_{character}_{segment.original_idx}",
                        text=segment.segment_text,
                        character=character,
                        language=lang_code,
                        voice_path=voice_refs.get(character, "none"),
                        original_index=segment.original_idx,
                        exaggeration=inputs.get("exaggeration", 0.5),
                        temperature=inputs.get("temperature", 0.8),
                        cfg_weight=inputs.get("cfg_weight", 0.5)
                    )
                    
                    self.work_queue.put(work_item)
                    total_items += 1
        
        print(f"📥 Built streaming work queue: {total_items} work items across {len(language_groups)} languages")
        return total_items
    
    def start_workers(self):
        """Start all worker threads."""
        self.shutdown_event.clear()
        
        for worker_id in range(self.max_workers):
            worker = StreamingWorker(
                worker_id=worker_id + 1,
                work_queue=self.work_queue,
                result_queue=self.result_queue,
                tts_node=self.tts_node,
                shutdown_event=self.shutdown_event
            )
            
            worker_thread = threading.Thread(target=worker.run)
            worker_thread.start()
            
            self.workers.append(worker)
            self.worker_threads.append(worker_thread)
        
        print(f"🚀 All {self.max_workers} streaming workers started and ready")
    
    def collect_results(self, expected_count: int) -> Dict[int, torch.Tensor]:
        """Collect results from workers as they complete."""
        results = {}
        completed_count = 0
        start_time = time.time()
        
        while completed_count < expected_count:
            try:
                result = self.result_queue.get(timeout=5.0)
                results[result.original_index] = result.audio
                completed_count += 1
                
                # Progress reporting
                if completed_count % max(1, expected_count // 10) == 0:  # Every 10%
                    progress = int(100 * completed_count / expected_count)
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    remaining = (expected_count - completed_count) / rate if rate > 0 else 0
                    
                    print(f"📊 Streaming progress: {completed_count}/{expected_count} ({progress}%) "
                          f"- {rate:.1f} segments/sec - ETA: {remaining:.0f}s")
                
            except Empty:
                print("⏳ Waiting for streaming results...")
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
        
        print("✅ All streaming workers shut down")
    
    def process_streaming(
        self,
        language_groups: Dict[str, Any],
        character_groups_by_lang: Dict[str, Dict[str, Any]],
        voice_refs: Dict[str, str],
        inputs: Dict[str, Any]
    ) -> Dict[int, torch.Tensor]:
        """
        Main processing method - implements the efficient streaming approach.
        """
        try:
            # No pre-loading - use existing model management
            languages = list(language_groups.keys())
            print(f"🎯 Using existing model management for {len(languages)} languages: {languages}")
            
            # Build work queue
            total_items = self.build_work_queue(language_groups, character_groups_by_lang, voice_refs, inputs)
            
            # Start workers
            self.start_workers()
            
            # Collect results
            print(f"🌊 Starting streaming processing with {self.max_workers} workers...")
            start_time = time.time()
            
            results = self.collect_results(total_items)
            
            total_time = time.time() - start_time
            throughput = total_items / total_time if total_time > 0 else 0
            
            print(f"✅ Streaming processing complete: {total_items} segments in {total_time:.1f}s ({throughput:.2f} segments/sec)")
            
            return results
            
        finally:
            # Always cleanup workers
            self.shutdown_workers()