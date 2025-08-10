"""
Universal Streaming Coordinator

Replaces all format-specific routers (like SRTBatchProcessingRouter) with a 
single universal coordinator that works with any node type through the 
StreamingSegment interface.
"""

import torch
import time
from typing import List, Dict, Any, Optional, Tuple
from .streaming_types import (
    StreamingSegment, StreamingResult, StreamingConfig, 
    StreamingMode, StreamingMetrics
)
from .streaming_interface import StreamingEngineAdapter
from .work_queue_processor import UniversalWorkQueueProcessor


class StreamingCoordinator:
    """
    Universal coordinator for streaming and traditional processing.
    
    Replaces all node-specific routers with a single, engine-agnostic coordinator
    that works with universal StreamingSegment format.
    """
    
    @staticmethod
    def should_use_streaming(config: StreamingConfig, segment_count: int) -> bool:
        """
        Determine whether to use streaming or traditional processing.
        
        Args:
            config: StreamingConfig with batch_size and threshold settings
            segment_count: Number of segments to process
            
        Returns:
            True for streaming mode, False for traditional mode
        """
        # User explicitly disabled streaming
        if config.batch_size <= 0:
            return False
            
        # Not enough segments to benefit from streaming
        if segment_count < config.streaming_threshold:
            return False
            
        # Batch size > 1 means user wants streaming
        return config.batch_size > 1
    
    @staticmethod
    def process(
        segments: List[StreamingSegment],
        adapter: StreamingEngineAdapter,
        config: StreamingConfig,
        **kwargs
    ) -> Tuple[Dict[int, torch.Tensor], StreamingMetrics, bool]:
        """
        Process segments using streaming or traditional mode.
        
        This is the main entry point that replaces all format-specific routers.
        
        Args:
            segments: List of StreamingSegments to process
            adapter: Engine adapter implementing StreamingEngineAdapter
            config: StreamingConfig with processing settings
            **kwargs: Additional processing parameters
            
        Returns:
            Tuple of (results_dict, metrics, success)
            - results_dict: Dict mapping segment index -> audio tensor
            - metrics: StreamingMetrics with performance data
            - success: True if processing succeeded
        """
        if not segments:
            return {}, StreamingMetrics(), True
            
        # Initialize metrics
        metrics = StreamingMetrics()
        metrics.total_segments = len(segments)
        
        # Decide processing mode
        use_streaming = StreamingCoordinator.should_use_streaming(config, len(segments))
        
        if use_streaming:
            print(f"🌊 StreamingCoordinator: Processing {len(segments)} segments with {config.batch_size} workers")
            return StreamingCoordinator._process_streaming(
                segments, adapter, config, metrics, **kwargs
            )
        else:
            print(f"📖 StreamingCoordinator: Processing {len(segments)} segments sequentially")
            return StreamingCoordinator._process_traditional(
                segments, adapter, config, metrics, **kwargs
            )
    
    @staticmethod
    def _process_streaming(
        segments: List[StreamingSegment],
        adapter: StreamingEngineAdapter,
        config: StreamingConfig,
        metrics: StreamingMetrics,
        **kwargs
    ) -> Tuple[Dict[int, torch.Tensor], StreamingMetrics, bool]:
        """
        Process segments using streaming parallel workers.
        
        Args:
            segments: List of segments to process
            adapter: Engine adapter for processing
            config: StreamingConfig settings
            metrics: Metrics object to update
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (results_dict, metrics, success)
        """
        try:
            # Group segments by language for optimal processing
            language_character_groups = adapter.group_segments_by_character(segments)
            
            # Pre-load models for all languages if enabled
            if config.enable_model_preloading:
                languages = list(language_character_groups.keys())
                print(f"🚀 Pre-loading models for {len(languages)} languages: {languages}")
                device = kwargs.get('device', 'auto')
                adapter.preload_models(languages, device)
            
            # Initialize streaming processor
            processor = UniversalWorkQueueProcessor(
                adapter=adapter,
                max_workers=config.batch_size,
                timeout=config.worker_timeout
            )
            
            # Process all segments with streaming
            start_time = time.time()
            results = processor.process_segments(
                segments=segments,
                language_character_groups=language_character_groups,
                config=config,
                **kwargs
            )
            
            # Collect metrics
            for result in results.values():
                if isinstance(result, StreamingResult):
                    metrics.add_result(result)
            
            # Convert StreamingResults to audio tensors
            audio_results = {}
            for idx, result in results.items():
                if isinstance(result, StreamingResult):
                    audio_results[idx] = result.audio
                elif isinstance(result, torch.Tensor):
                    audio_results[idx] = result
            
            total_time = time.time() - start_time
            print(f"✅ Streaming completed: {len(audio_results)}/{len(segments)} segments in {total_time:.1f}s")
            
            return audio_results, metrics, True
            
        except Exception as e:
            print(f"❌ Streaming failed: {e}")
            
            # Fall back to traditional if configured
            if config.fallback_to_traditional:
                print("📖 Falling back to traditional processing...")
                return StreamingCoordinator._process_traditional(
                    segments, adapter, config, metrics, **kwargs
                )
            else:
                return {}, metrics, False
    
    @staticmethod
    def _process_traditional(
        segments: List[StreamingSegment],
        adapter: StreamingEngineAdapter,
        config: StreamingConfig,
        metrics: StreamingMetrics,
        **kwargs
    ) -> Tuple[Dict[int, torch.Tensor], StreamingMetrics, bool]:
        """
        Process segments sequentially using traditional method.
        
        Args:
            segments: List of segments to process
            adapter: Engine adapter for processing
            config: StreamingConfig settings
            metrics: Metrics object to update
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (results_dict, metrics, success)
        """
        results = {}
        current_language = None
        start_time = time.time()
        
        for i, segment in enumerate(segments):
            try:
                # Switch language model if needed
                if segment.language != current_language:
                    print(f"🔄 Loading {adapter.engine_name} model for {segment.language}")
                    adapter.load_model_for_language(segment.language, kwargs.get('device', 'auto'))
                    current_language = segment.language
                
                # Process segment
                segment_start = time.time()
                result = adapter.process_segment(segment, **kwargs)
                
                if result.success:
                    results[segment.index] = result.audio
                    metrics.add_result(result)
                    
                    # Progress update
                    progress = int(100 * (i + 1) / len(segments))
                    print(f"📊 Progress: {i+1}/{len(segments)} ({progress}%) - {segment.character} in {segment.language}")
                else:
                    print(f"❌ Failed to process segment {segment.index}: {result.error_msg}")
                    
            except Exception as e:
                print(f"❌ Error processing segment {segment.index}: {e}")
                continue
        
        total_time = time.time() - start_time
        print(f"✅ Traditional processing completed: {len(results)}/{len(segments)} segments in {total_time:.1f}s")
        
        return results, metrics, len(results) > 0
    
    @staticmethod
    def convert_node_data_to_segments(
        node_type: str,
        data: Any,
        voice_refs: Dict[str, str],
        **kwargs
    ) -> List[StreamingSegment]:
        """
        Convert node-specific data to universal StreamingSegments.
        
        This eliminates the need for format-specific routers.
        
        Args:
            node_type: Type of node ('tts', 'srt', 'vc')
            data: Node-specific data to convert
            voice_refs: Character -> voice path mapping
            **kwargs: Additional conversion parameters
            
        Returns:
            List of StreamingSegments
        """
        segments = []
        
        if node_type == 'tts':
            # Convert TTS text chunks to segments
            # Data is list of (idx, character, text, language) tuples
            for idx, character, text, language in data:
                segments.append(StreamingSegment(
                    index=idx,
                    text=text,
                    character=character,
                    language=language,
                    voice_path=voice_refs.get(character, 'none'),
                    metadata={'source': 'tts'}
                ))
                
        elif node_type == 'srt':
            # Convert SRT subtitles to segments
            # Data is list of (idx, subtitle, character_segments) tuples
            for idx, subtitle, character_segments in data:
                for character, text, language in character_segments:
                    segments.append(StreamingSegment(
                        index=idx,
                        text=text,
                        character=character,
                        language=language,
                        voice_path=voice_refs.get(character, 'none'),
                        metadata={
                            'source': 'srt',
                            'start_time': subtitle.start_time if hasattr(subtitle, 'start_time') else None,
                            'end_time': subtitle.end_time if hasattr(subtitle, 'end_time') else None,
                            'duration': subtitle.duration if hasattr(subtitle, 'duration') else None
                        }
                    ))
                    
        elif node_type == 'vc':
            # Convert voice conversion data to segments
            # Future implementation for VC nodes
            pass
            
        return segments
    
    @staticmethod
    def convert_results_to_node_format(
        node_type: str,
        results: Dict[int, torch.Tensor],
        original_data: Any,
        **kwargs
    ) -> Any:
        """
        Convert universal results back to node-specific format.
        
        Args:
            node_type: Type of node ('tts', 'srt', 'vc')
            results: Dict mapping index -> audio tensor
            original_data: Original node data for reference
            **kwargs: Additional conversion parameters
            
        Returns:
            Node-specific result format
        """
        if node_type == 'tts':
            # TTS nodes expect list of audio tensors in order
            audio_list = []
            for i in sorted(results.keys()):
                audio_list.append(results.get(i))
            return audio_list
            
        elif node_type == 'srt':
            # SRT nodes expect (audio_segments, natural_durations, any_cached)
            audio_segments = []
            natural_durations = []
            
            for i in sorted(results.keys()):
                audio = results.get(i)
                if audio is not None:
                    audio_segments.append(audio)
                    # Calculate duration from audio tensor
                    sample_rate = kwargs.get('sample_rate', 22050)
                    duration = float(audio.shape[-1]) / sample_rate
                    natural_durations.append(duration)
                else:
                    audio_segments.append(None)
                    natural_durations.append(0.0)
                    
            any_cached = kwargs.get('enable_audio_cache', False) and len(results) > 0
            return audio_segments, natural_durations, any_cached
            
        elif node_type == 'vc':
            # Future implementation for VC nodes
            return results
            
        return results