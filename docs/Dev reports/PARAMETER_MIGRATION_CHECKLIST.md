# Parameter Migration Checklist

**Moving parameters between main node inputs and advanced options (edit_options) in F5-TTS Edit Node**

## ⚠️ CRITICAL: This process affects multiple files and cache systems. Follow ALL steps in order.

---

## 📋 Pre-Migration Checklist

### 1. **Identify Parameter Type**
- [ ] **Generation Parameter**: Affects F5-TTS model inference (stays in main inputs)
- [ ] **Post-Processing Parameter**: Applied after generation (move to edit_options)
- [ ] **Cache Dependency**: Does changing this parameter require regeneration?

### 2. **Document Current Usage**
- [ ] Find all files referencing the parameter: `grep -r "parameter_name" .`
- [ ] List all method signatures that include the parameter
- [ ] Note cache key dependencies
- [ ] Document tooltip/description for reference

---

## 🔄 Migration Steps (Moving FROM main inputs TO edit_options)

### Step 1: **Remove from Main Node** (`nodes/f5tts_edit_node.py`)
- [ ] Remove parameter from `INPUT_TYPES` definition
- [ ] Remove parameter from `edit_speech()` method signature
- [ ] Remove parameter from `validate_inputs()` call
- [ ] Remove parameter from `edit_engine.perform_f5tts_edit()` call

### Step 2: **Add to Edit Options Node** (`nodes/f5tts_edit_options_node.py`)
- [ ] Add parameter to `INPUT_TYPES` definition with proper tooltip
- [ ] Add parameter to `create_options()` method signature with default value
- [ ] Add parameter to options dictionary in `create_options()` method

### Step 3: **Update Edit Engine** (`core/f5tts_edit_engine.py`)
- [ ] Remove parameter from `perform_f5tts_edit()` method signature
- [ ] Add parameter extraction from `edit_options` with `edit_options.get("param_name", default_value)`
- [ ] Update all usage of parameter to use the extracted variable

### Step 4: **Update Cache System** (`core/f5tts_edit_cache.py`)
**⚠️ CRITICAL: Only do this if parameter is post-processing and shouldn't affect cache**
- [ ] Remove parameter from `_generate_cache_key()` method signature
- [ ] Remove parameter from cache key dictionary
- [ ] Remove parameter from `get()` method signature
- [ ] Remove parameter from `put()` method signature

### Step 5: **Update Cache Calls** (`core/f5tts_edit_engine.py`)
- [ ] Remove parameter from `cache.get()` call
- [ ] Remove parameter from `cache.put()` call

---

## 🔄 Migration Steps (Moving FROM edit_options TO main inputs)

### Step 1: **Remove from Edit Options Node** (`nodes/f5tts_edit_options_node.py`)
- [ ] Remove parameter from `INPUT_TYPES` definition
- [ ] Remove parameter from `create_options()` method signature
- [ ] Remove parameter from options dictionary in `create_options()` method

### Step 2: **Add to Main Node** (`nodes/f5tts_edit_node.py`)
- [ ] Add parameter to `INPUT_TYPES` definition with proper tooltip
- [ ] Add parameter to `edit_speech()` method signature
- [ ] Add parameter to `validate_inputs()` call
- [ ] Add parameter to `edit_engine.perform_f5tts_edit()` call

### Step 3: **Update Edit Engine** (`core/f5tts_edit_engine.py`)
- [ ] Add parameter to `perform_f5tts_edit()` method signature
- [ ] Remove parameter extraction from `edit_options`
- [ ] Update all usage to use the method parameter directly

### Step 4: **Update Cache System** (`core/f5tts_edit_cache.py`)
**⚠️ CRITICAL: Only do this if parameter affects generation and should trigger cache regeneration**
- [ ] Add parameter to `_generate_cache_key()` method signature
- [ ] Add parameter to cache key dictionary
- [ ] Add parameter to `get()` method signature
- [ ] Add parameter to `put()` method signature

### Step 5: **Update Cache Calls** (`core/f5tts_edit_engine.py`)
- [ ] Add parameter to `cache.get()` call
- [ ] Add parameter to `cache.put()` call

---

## 🧪 Testing Checklist

### After Each Step:
- [ ] Run syntax check: `python -m py_compile filename.py`
- [ ] Test node loads in ComfyUI without errors
- [ ] Test parameter appears in correct location (main inputs vs edit_options)

### Final Testing:
- [ ] Test with default values
- [ ] Test with extreme values (min/max)
- [ ] Test cache behavior (if applicable)
- [ ] Test that parameter actually affects the intended behavior
- [ ] Test backwards compatibility with existing workflows

---

## 📝 Common Gotchas

### 1. **Method Signature Mismatches**
- Always update method signatures in the same order: definition → calls
- Check for positional vs keyword arguments

### 2. **Cache Key Consistency**
- `_generate_cache_key()` method signature must match the dictionary keys
- All cache calls (`get()`, `put()`) must have matching signatures

### 3. **Edit Options Extraction**
- Always provide default values: `edit_options.get("param_name", default_value)`
- Use the same default as the INPUT_TYPES definition

### 4. **Parameter Naming**
- Consider renaming when moving to clarify purpose (e.g., `target_rms` → `post_rms_normalization`)
- Update ALL references to the new name

### 5. **Cache vs Post-Processing**
- **Generation parameters**: Include in cache key (changes require regeneration)
- **Post-processing parameters**: Exclude from cache key (applied after cached generation)

---

## 🔍 Verification Commands

### Find All Parameter References:
```bash
grep -r "parameter_name" . --include="*.py"
```

### Check Method Signatures:
```bash
grep -r "def.*parameter_name" . --include="*.py"
```

### Verify Cache Key Consistency:
```bash
grep -A 10 -B 10 "_generate_cache_key" core/f5tts_edit_cache.py
```

---

## 📁 Files Typically Affected

### Always Check:
- `nodes/f5tts_edit_node.py` - Main node definition
- `nodes/f5tts_edit_options_node.py` - Advanced options
- `core/f5tts_edit_engine.py` - Core logic
- `core/f5tts_edit_cache.py` - Cache system (if cache-affecting)

### Sometimes Check:
- `nodes/base_node.py` - If base validation changes
- `core/audio_compositing.py` - If audio processing changes
- Other nodes that might reference the parameter

---

## 🚨 Emergency Rollback

If migration fails:
1. Revert all files to previous state
2. Test that original functionality works
3. Review checklist for missed steps
4. Try migration again step-by-step

---

## 📋 Example Migration Template

```python
# BEFORE (main input):
"parameter_name": ("TYPE", {
    "default": value, "min": min, "max": max,
    "tooltip": "Description"
})

# AFTER (edit_options):
# 1. Remove from main node INPUT_TYPES
# 2. Add to edit_options INPUT_TYPES
# 3. Extract in engine: param = edit_options.get("parameter_name", default_value)
# 4. Update cache if necessary
```

---

**💡 Remember: Test after each step, not just at the end!**