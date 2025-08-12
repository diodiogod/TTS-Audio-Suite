# Stateless ChatterBox Wrapper Implementation Plan

*Solving the shared model state corruption issue in streaming parallel processing*

## Executive Summary

The current streaming implementation causes crashes even on large segments due to **shared model state corruption** when multiple workers modify `self.conds` simultaneously. This plan outlines implementing a **stateless wrapper** that eliminates shared state, enabling true parallel processing without crashes.

## Problem Analysis

### Current State Corruption Issue

**The Bug:**
```python
# Multiple workers share same model instance
# Worker A and B both call model.generate() simultaneously
# Both modify self.conds causing tensor dimension mismatches
```

**Why Current Solutions Don't Work:**
1. **Pre-loaded models** - Still share the same `self.conds` state
2. **Model switching** - Doesn't prevent concurrent state modification  
3. **Reference implementation** - Has the exact same bug (never solved it)

### Root Cause

The ChatterBox `generate()` method modifies instance state:
```python
# Line 345-356 in tts.py
if audio_prompt_path:
    self.prepare_conditionals(audio_prompt_path, exaggeration)  # MODIFIES self.conds!
    
# Line 350-356: Updates exaggeration
if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
    self.conds.t3 = T3Cond(...)  # MODIFIES self.conds!
```

## Solution: Stateless Wrapper Architecture

### Core Concept

Create a wrapper that **never stores state** - all conditions are calculated fresh per call and passed as parameters.

### Implementation Strategy

#### 1. Create Stateless Wrapper Class
Location: `/engines/chatterbox/stateless_wrapper.py`

```python
class StatelessChatterBoxWrapper:
    """Thread-safe stateless wrapper for ChatterBox TTS."""
    
    def __init__(self, chatterbox_model):
        self.model = chatterbox_model
        # NO self.conds stored!
        
    def generate_stateless(self, text, audio_prompt_path=None, 
                          exaggeration=0.5, cfg_weight=0.5, 
                          temperature=0.8, seed=42):
        """Generate audio without modifying shared state."""
        
        # 1. Prepare conditions locally (no state modification)
        local_conds = self._prepare_conditions_locally(
            audio_prompt_path, exaggeration
        )
        
        # 2. Process text
        text_tokens = self._prepare_text_tokens(text)
        
        # 3. Generate with local conditions (thread-safe)
        with torch.inference_mode():
            speech_tokens = self.model.t3.inference(
                t3_cond=local_conds.t3,  # Fresh, isolated
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight
            )
            
            wav, _ = self.model.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=local_conds.gen  # Fresh, isolated
            )
            
        return self._post_process_audio(wav)
```

#### 2. Implement Local Condition Preparation
Extract from existing `prepare_conditionals()` but return instead of storing:

```python
def _prepare_conditions_locally(self, wav_fpath, exaggeration):
    """Prepare conditions without storing in self."""
    if not wav_fpath:
        return self._get_default_conditions(exaggeration)
        
    # Same logic as prepare_conditionals() but returns Conditionals
    s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
    # ... (rest of preparation logic)
    
    return Conditionals(t3_cond, s3gen_ref_dict)  # RETURN, don't store
```

#### 3. Update Streaming Processor
Modify `streaming_processor.py` to use stateless wrapper:

```python
class StreamingWorker:
    def __init__(self, worker_id, work_queue, result_queue, 
                 stateless_model, shutdown_event):
        self.stateless_model = stateless_model  # Use wrapper
        
    def _work_loop(self):
        while not self.shutdown_event.is_set():
            work_item = self.work_queue.get(timeout=1.0)
            
            # Use stateless generation (thread-safe!)
            audio = self.stateless_model.generate_stateless(
                text=work_item.text,
                audio_prompt_path=work_item.audio_prompt_path,
                exaggeration=work_item.exaggeration,
                cfg_weight=work_item.cfg_weight,
                temperature=work_item.temperature
            )
```

#### 4. Integration Points

**Streaming Model Manager Update:**
```python
def get_stateless_model_for_language(self, language_code):
    """Get stateless wrapper for language."""
    base_model = self.preloaded_models.get(language_code)
    if base_model:
        return StatelessChatterBoxWrapper(base_model)
    return None
```

**Node Integration:**
```python
def _process_single_segment_for_streaming(self, ...):
    # Get stateless wrapper instead of raw model
    stateless_model = self._streaming_model_manager.get_stateless_model_for_language(language)
    
    # Generate using stateless method
    audio = stateless_model.generate_stateless(...)
```

## Implementation Phases

### Phase 1: Core Stateless Wrapper (2-3 hours)
1. Create `stateless_wrapper.py` with `StatelessChatterBoxWrapper` class
2. Implement `_prepare_conditions_locally()` method
3. Implement `generate_stateless()` method
4. Add helper methods for text token preparation

### Phase 2: Streaming Integration (1-2 hours)
1. Update `streaming_processor.py` to use stateless wrapper
2. Modify `StreamingWorker` to call stateless methods
3. Update `TrueStreamingProcessor` initialization

### Phase 3: Model Manager Updates (1 hour)
1. Update `streaming_model_manager.py` to provide stateless wrappers
2. Ensure compatibility with existing pre-loading system
3. Add wrapper caching to avoid recreation

### Phase 4: Node Updates (1 hour)
1. Update `_process_single_segment_for_streaming()` in TTS/SRT nodes
2. Ensure backward compatibility with non-streaming paths
3. Update any direct model.generate() calls

### Phase 5: Testing & Validation (2 hours)
1. Test with multiple workers on short segments
2. Test with multiple workers on large segments  
3. Verify no state corruption with concurrent generation
4. Performance benchmarking vs current implementation

## Benefits

1. **Eliminates Crashes**: No shared state = no corruption
2. **True Parallelism**: Workers can run simultaneously without locks
3. **Same Memory Usage**: Still uses single model instance per language
4. **Clean Architecture**: Separation of concerns between model and state
5. **Future Proof**: Easy to extend to other engines (F5-TTS, etc.)

## Risk Mitigation

1. **Backward Compatibility**: Keep existing methods intact
2. **Gradual Rollout**: Test with streaming first, then expand
3. **Performance**: Cache prepared conditions where possible
4. **Fallback**: Easy to revert if issues arise

## Success Metrics

1. **No crashes** with 10+ workers on short segments
2. **No crashes** with sequential generation after parallel
3. **Performance improvement** of at least 2x with 4 workers
4. **Memory usage** remains constant (no model duplication)

## Timeline

**Total Estimated Time: 6-8 hours**

- Day 1: Implement core wrapper and basic integration
- Day 2: Complete integration and testing
- Day 3: Performance optimization and documentation

## Next Steps

1. Review and approve this plan
2. Create feature branch for implementation
3. Implement Phase 1 (core wrapper)
4. Test basic functionality
5. Continue with remaining phases

## Appendix: Technical Details

### Current Shared State Points

1. `self.conds` - Main shared state causing corruption
2. `self.prepare_conditionals()` - Modifies self.conds
3. Exaggeration update (lines 350-356) - Modifies self.conds.t3
4. No synchronization between workers

### Stateless Principles

1. **No instance variables** for generation state
2. **All state passed as parameters**
3. **Immutable conditions** during generation
4. **Thread-safe by design**

### Compatibility Matrix

| Component | Current | With Wrapper | Changes Needed |
|-----------|---------|--------------|----------------|
| ChatterBox TTS | ✅ | ✅ | Add wrapper class |
| Streaming Processor | ❌ Crashes | ✅ | Use wrapper |
| Traditional Mode | ✅ | ✅ | No changes |
| Caching | ✅ | ✅ | Works as-is |
| Pre-loading | ✅ | ✅ | Return wrappers |

This plan provides a clear path to solving the state corruption issue while maintaining all existing functionality and improving performance.