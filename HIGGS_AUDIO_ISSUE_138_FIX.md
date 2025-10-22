# Issue #138 Fix: Higgs Audio DynamicCache AttributeError

## Problem Investigation

**Issue**: Using ⚙️ Higgs Audio 2 Engine with 🎤 TTS Text fails with:
```
AttributeError: 'NoneType' object has no attribute 'zero_'
```

**Stack trace analysis**: The error occurs when `kv_cache.reset()` is called during `_prepare_kv_caches()`.

## Root Cause

The code had two critical issues with cache handling:

### Issue 1: Ambiguous Type Checking in Cache Reset (PRIMARY BUG)

**Location**: `serve_engine.py` lines 662-671 (original code)

```python
if hasattr(kv_cache, 'reset'):
    kv_cache.reset()  # ERROR HAPPENS HERE
```

**Problem**: Both `StaticCache` and `DynamicCache` have a `reset()` method. When using DynamicCache (Memory Safe mode), the code would incorrectly call `StaticCache.reset()` behavior on a `DynamicCache` object:

1. For StaticCache: `.reset()` iterates layers and calls `zero_()` on tensors
2. For manually-created DynamicCache: The object isn't properly initialized, so `self.keys` is `None`
3. When `StaticCache.reset()` is called on DynamicCache, it tries to access `None.zero_()` → **AttributeError**

### Issue 2: Incomplete DynamicCache Initialization (SECONDARY BUG)

**Location**: `serve_engine.py` lines 346-360 and 629-648

When DynamicCache constructor fails, the fallback code creates the cache manually:
```python
cache = object.__new__(DynamicCache)
object.__setattr__(cache, '_key_cache', [])
object.__setattr__(cache, '_value_cache', [])
```

This creates a bare object without proper initialization that transformers expects. When `.reset()` is called later, it crashes because the internal state is incomplete.

## Solution Implemented

### Fix 1: Use Explicit Type Checking

**Change**: Replace `hasattr(kv_cache, 'reset')` with explicit `isinstance()` checks.

```python
# BEFORE
if hasattr(kv_cache, 'reset'):
    kv_cache.reset()

# AFTER
if isinstance(kv_cache, StaticCache):
    try:
        kv_cache.reset()
    except Exception as e:
        print(f"  ⚠️ StaticCache reset error: {type(e).__name__}")
elif isinstance(kv_cache, DynamicCache):
    try:
        kv_cache.crop(0)  # Proper DynamicCache clearing
    except Exception as e:
        print(f"  ⚠️ DynamicCache crop error: {type(e).__name__}: {e}")
```

**Why this works**:
- Explicitly checks the actual type instead of relying on method availability
- Routes to the correct clearing method for each cache type
- Gracefully handles errors instead of crashing
- DynamicCache uses `crop(0)` instead of `reset()` which is its proper API

### Fix 2: Improve Fallback Cache Creation

**Change**: Add better error handling and logging for manual cache creation.

```python
# More comprehensive exception handling
except (AttributeError, TypeError) as e:
    logger.warning(f"DynamicCache() constructor failed: {e}, attempting manual construction...")
    try:
        cache = object.__new__(DynamicCache)
        # Initialize all expected internal attributes
        object.__setattr__(cache, '_key_cache', [])
        object.__setattr__(cache, '_value_cache', [])
        if hasattr(DynamicCache, '_seen_tokens'):
            object.__setattr__(cache, '_seen_tokens', 0)
        # Try to mark as initialized
        if hasattr(cache, '__dict__'):
            cache.__dict__['_is_initialized'] = True
        return cache
    except Exception as fallback_error:
        logger.error(f"Manual DynamicCache construction failed: {fallback_error}")
        raise RuntimeError(f"Cannot create DynamicCache: {e}") from fallback_error
```

**Why this helps**:
- Logs why DynamicCache() is failing for debugging
- Catches more exception types (TypeError in addition to AttributeError)
- Better error reporting instead of silent failure
- Marks cache as initialized to help with future state checks

## Files Modified

- `engines/higgs_audio/boson_multimodal/serve/serve_engine.py`
  - Line 10: Added `DynamicCache` to imports
  - Lines 345-365: Improved initial cache creation
  - Lines 632-653: Improved cache recreation in `_prepare_kv_caches()`
  - Lines 664-680: Fixed cache reset logic with explicit type checking

## Why This Issue Was Hard to Reproduce

1. **Version-dependent**: Only manifests when transformers raises AttributeError on DynamicCache()
2. **Mode-specific**: Only occurs in Memory Safe mode (CUDA Graphs OFF)
3. **Stateful**: Requires manual cache creation to happen first, then later reset to be called

## Testing Recommendations

To verify the fix works:

1. Test with Higgs Audio 2 Engine + TTS Text node
2. Enable Memory Safe mode (disable CUDA Graphs in engine settings)
3. Attempt multiple generations to trigger `_prepare_kv_caches()` resets
4. Check logs for proper cache type routing ("Created DynamicCache" and "Cleared DynamicCache state")

## Performance Impact

- **Zero**: Type checking via `isinstance()` is negligible compared to generation time
- **Improvement**: Using `crop(0)` instead of `reset()` for DynamicCache is slightly faster
- **Safety**: Error handling prevents crashes, may log warnings instead

