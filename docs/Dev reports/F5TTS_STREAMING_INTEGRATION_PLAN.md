# F5-TTS Streaming Integration Plan

*Unified streaming architecture to eliminate spaghetti code and enable F5-TTS streaming*

## Current Spaghetti Architecture Analysis

**ChatterBox SRT Streaming Currently Works Through:**
1. **ChatterBox SRT Node** checks `batch_size > 0` → creates `SRTBatchProcessingRouter` 
2. **SRTBatchProcessingRouter** converts SRT data to TTS format → calls `StreamingWorkQueueProcessor`
3. **StreamingWorkQueueProcessor** processes via TTS node's `_process_single_segment_for_streaming()`
4. **Router converts back** streaming results to SRT format

**The Spaghetti Problem:**
- **Data conversion hell**: SRT → TTS format → streaming → back to SRT format
- **Hardcoded ChatterBox dependencies**: Router only works with ChatterBox SRT nodes
- **Duplicated streaming integration**: Every node type needs its own router
- **Bridge pattern messiness**: Each node type needs format conversion bridges

## Unified Streaming Architecture Solution

### Phase 1: Create Universal Streaming Interface
**Replace format conversion with universal streaming contracts:**

1. **New `utils/streaming/` module:**
   ```
   utils/streaming/
   ├── __init__.py
   ├── streaming_coordinator.py    # Universal coordinator (replaces all routers)
   ├── streaming_interface.py      # Abstract interface for engines  
   ├── work_queue_processor.py     # Engine-agnostic processor
   └── streaming_types.py          # Universal data structures
   ```

2. **Universal `StreamingCoordinator`** (replaces `SRTBatchProcessingRouter`):
   - Works with ANY node type (TTS, SRT, Voice Conversion)  
   - NO format conversion needed - works with universal segment interface
   - Single decision logic: streaming vs traditional
   - Engine-agnostic through adapter pattern

### Phase 2: Universal Segment Interface
**Eliminate data conversion spaghetti:**

1. **Universal `StreamingSegment` structure:**
   ```python
   @dataclass
   class StreamingSegment:
       index: int
       text: str  
       character: str
       language: str
       voice_path: str
       metadata: Dict[str, Any]  # Node-specific data (SRT timings, etc.)
   ```

2. **All nodes produce same format:**
   - TTS nodes: Create segments from text chunks
   - SRT nodes: Create segments from subtitles (with timing metadata)
   - Future VC nodes: Create segments from audio inputs
   - **No format conversion between nodes and streaming system**

### Phase 3: Engine Streaming Adapters  
**Make engines streaming-compatible:**

1. **Streaming adapters implement universal interface:**
   - `chatterbox_streaming_adapter.py` - Current `StreamingWorkQueueProcessor` logic
   - `f5tts_streaming_adapter.py` - New F5-TTS streaming support
   - Each handles engine-specific optimizations (language switching, caching)

2. **Adapters work with universal segments:**
   - No need to know if segments came from TTS or SRT nodes
   - Engine-specific processing through `process_segment(StreamingSegment)` method

### Phase 4: Node Streaming Integration
**Clean integration without spaghetti:**

1. **Universal `StreamingMixin` for all node types:**
   ```python
   class StreamingMixin:
       def enable_streaming(self, segments, batch_size, **kwargs):
           if self.should_stream(batch_size, len(segments)):
               return StreamingCoordinator.process(segments, self.engine_adapter)
           else:
               return self.traditional_process(segments, **kwargs)
   ```

2. **Clean node integration:**
   ```python
   # ChatterBox SRT Node
   segments = self.convert_srt_to_segments(subtitles)  # Simple conversion
   results = self.enable_streaming(segments, batch_size, **kwargs)
   
   # F5-TTS TTS Node  
   segments = self.convert_text_to_segments(text)      # Simple conversion
   results = self.enable_streaming(segments, batch_size, **kwargs)
   ```

### Phase 5: F5-TTS Streaming Implementation
**Add F5-TTS support through unified system:**

1. **F5-TTS streaming adapter:**
   - Implements language model switching (F5-DE, F5-FR, F5-ES, etc.)
   - Handles F5-TTS longer inference times appropriately  
   - Uses same `StreamingWorkQueueProcessor` pattern as ChatterBox

2. **F5-TTS node updates:**
   - Add `StreamingMixin` to F5-TTS nodes
   - Implement segment conversion methods
   - **Zero changes to existing functionality**

## Key Benefits Over Current Spaghetti

1. **Eliminates format conversion**: Universal segment interface means no SRT→TTS→SRT conversion
2. **No router duplication**: Single `StreamingCoordinator` works for all node types  
3. **Engine extensibility**: New engines just implement the adapter interface
4. **Clean node integration**: Nodes just convert to segments and call `enable_streaming()`
5. **Preserves all functionality**: Existing ChatterBox SRT streaming preserved during migration

## Migration Strategy (Preserving ChatterBox SRT)

1. **Phase 1**: Create universal streaming core alongside existing system
2. **Phase 2**: Migrate ChatterBox TTS to universal system (SRT keeps working)
3. **Phase 3**: Migrate ChatterBox SRT to universal system (remove router spaghetti)
4. **Phase 4**: Add F5-TTS streaming through universal system  
5. **Phase 5**: Clean up old ChatterBox-specific streaming code

## Technical Implementation Notes

### Current ChatterBox SRT Router Analysis
The current `SRTBatchProcessingRouter` performs these conversions:
- `_convert_srt_to_streaming_format()` - Converts SRT subtitles to TTS-compatible tuples
- `_convert_streaming_results_to_srt_format()` - Converts streaming results back to SRT format
- These conversions are **pure overhead** that can be eliminated with universal segments

### Universal Streaming Interface Design
```python
# Instead of format-specific routers, one coordinator handles all:
class StreamingCoordinator:
    @staticmethod
    def process(segments: List[StreamingSegment], engine_adapter: StreamingEngineAdapter):
        # Universal streaming logic works with any segment type
        # No format conversion needed
```

### Backward Compatibility
- Existing ChatterBox SRT streaming will continue working during migration
- Migration can be done incrementally without breaking functionality
- Old routers can be deprecated after universal system is proven

This eliminates the current "bridge pattern hell" where every node type needs its own router and format conversion, replacing it with a clean universal interface that any engine can plug into.