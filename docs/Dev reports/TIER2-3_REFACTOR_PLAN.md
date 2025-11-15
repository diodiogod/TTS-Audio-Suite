# Model Loading Tier 2-3 Refactor Plan

## Overview
Complete the model loading architecture refactor by:
1. **Tier 2**: Consolidate caching, standardize error handling
2. **Tier 3**: Standardize factory signatures, implement capability registry

## TIER 2: Consolidation & Standardization

### 2.1 Remove SmartModelLoader Usage

**Current State**: SmartModelLoader still used in:
- `nodes/chatterbox/chatterbox_tts_node.py` (lines with smart_model_loader)
- `engines/chatterbox/streaming_model_manager.py`
- `engines/chatterbox_official_23lang/streaming_model_manager.py`
- `engines/adapters/f5tts_streaming_adapter.py`

**Why Remove**:
- Creates third caching system (after unified interface + ComfyUI manager)
- Causes cache synchronization issues
- Delegates to manager.load_tts_model() anyway (redundant)

**Approach**:
1. ChatterBox TTS node: Replace smart_model_loader call with direct unified interface
2. Streaming processors: Use unified interface for model loading
3. Verify no other code depends on SmartModelLoader returns

**Risk**: Medium - streaming processors have complex state management

---

### 2.2 Standardize Error Handling

**Current State**: 5 different error patterns across factories

**Target State**: Single exception hierarchy:
```python
class TTSModelLoadingError(Exception):
    """Base exception for model loading"""
    pass

class TTSModelNotFoundError(TTSModelLoadingError):
    """Model not found locally or on HuggingFace"""
    pass

class TTSDeviceError(TTSModelLoadingError):
    """Device-related issues"""
    pass

class TTSModelFormatError(TTSModelLoadingError):
    """Model format compatibility issues"""
    pass
```

**Files to Update**:
- Create `utils/models/exceptions.py` with exception classes
- Update all factories to use consistent exceptions
- Update node error handlers to catch standardized exceptions

**Risk**: Low - backwards compatible

---

### 2.3 Unified Cache Key Generation

**Current State**: Multiple cache key formats:
- Unified interface: string concatenation
- SmartModelLoader: MD5 hashing
- Nodes: MD5 of dict items

**Target State**: Single hash-based cache key function

```python
def generate_model_cache_key(
    engine_name: str,
    model_name: str,
    device: str,
    language: Optional[str] = None,
    **extra_params
) -> str:
    """Generate standardized cache key"""
```

**Files to Create/Update**:
- `utils/models/cache_key_generator.py` (new)
- Update unified_model_interface.py to use it
- Update all node caching to use it

**Risk**: Low

---

## TIER 3: Architecture Standardization

### 3.1 Factory Parameter Dataclass

**Current State**: Each factory has different parameters

**Target State**: All factories accept standardized dataclass:

```python
@dataclass
class ModelLoadConfig:
    """Standard configuration for all model factories"""
    # Required
    device: str  # "auto", "cuda", "cpu", etc.
    model_name: str

    # Optional (engine-specific)
    language: Optional[str] = None
    model_path: Optional[str] = None
    repo_id: Optional[str] = None

    # Additional params (for quantization, optimization, etc.)
    additional_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Resolve device before storing
        from utils.device import resolve_torch_device
        self.device = resolve_torch_device(self.device)
```

**Factory Signature Becomes**:
```python
def engine_factory(config: ModelLoadConfig) -> Any:
    """All factories use same signature"""
```

**Files to Create/Update**:
- `utils/models/factory_config.py` (new)
- All factories in unified_model_interface.py
- Unified interface load_model() method

**Risk**: High - changes all factory signatures

---

### 3.2 Engine Capability Registry

**Current State**: Hardcoded engine-specific logic in model_manager.py

**Target State**: Data-driven engine capabilities:

```python
ENGINE_CAPABILITIES = {
    "chatterbox": {
        "supports_voice_conversion": True,
        "has_fallback": True,
        "can_corrupt": False,
        "recovery_handler": None
    },
    "higgs_audio": {
        "supports_voice_conversion": False,
        "has_fallback": False,
        "can_corrupt": True,
        "recovery_handler": reset_higgs_cuda_graphs,
        "requires_special_init": True
    },
    # ... all 7 engines
}
```

**Files to Create**:
- `utils/models/engine_registry.py` (new) - define capabilities
- Remove hardcoded engine checks from model_manager.py
- Use registry for generic recovery

**Risk**: Medium - affects error recovery paths

---

### 3.3 Unified Fallback Handler

**Current State**: ChatterBox has sophisticated fallback, others don't

**Target State**: Generic fallback utility any engine can use:

```python
class ModelFallbackHandler:
    """Generic fallback chain for model loading"""

    def __init__(self, primary_loader: Callable):
        self.primary_loader = primary_loader
        self.fallback_chain = []

    def add_fallback(self, fallback_loader: Callable, condition: Callable):
        """Add fallback with condition"""
        pass

    def load(self) -> Any:
        """Execute fallback chain"""
        pass
```

**Files to Create**:
- `utils/models/fallback_handler.py` (new)

**Risk**: Low - new utility, no breaking changes

---

## Implementation Order

1. **First**: Create exception classes (no breaking changes)
2. **Second**: Create cache key generator (parallel with tier 1 utilities)
3. **Third**: Create factory config dataclass
4. **Fourth**: Update all factories to use ModelLoadConfig
5. **Fifth**: Remove SmartModelLoader usage
6. **Sixth**: Create engine registry
7. **Seventh**: Implement unified fallback handler
8. **Eighth**: Update model_manager.py to use registry

---

## Risk Assessment

| Task | Risk | Dependencies | Effort |
|------|------|--------------|--------|
| Exception classes | Low | None | 1 hr |
| Cache key generator | Low | Exceptions | 1 hr |
| Factory config dataclass | Medium | None | 2 hrs |
| Update factories | High | Config, exceptions | 4 hrs |
| Remove SmartModelLoader | Medium | Updated factories | 2 hrs |
| Engine registry | Medium | Updated factories | 2 hrs |
| Fallback handler | Low | Exceptions | 2 hrs |
| Integration testing | High | All above | 3 hrs |

**Total**: ~17 hours of work

---

## Success Criteria

- [ ] All factories use consistent error handling
- [ ] Cache key generation centralized and consistent
- [ ] SmartModelLoader completely removed
- [ ] Engine capabilities defined in registry
- [ ] All 7 engines working without issues
- [ ] Cache hits work correctly across device switches
- [ ] Error messages are consistent and helpful
- [ ] No VRAM spikes or device mismatches
