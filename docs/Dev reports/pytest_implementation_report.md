# Pytest Testing Framework Implementation Report

**Date:** 2024-12-18  
**Author:** AI Assistant  
**Status:** ✅ Working (60/64 tests passing)

---

## Executive Summary

Implementing pytest for a ComfyUI custom node is **non-trivial** due to the way Python imports work with packages that have complex `__init__.py` files. This document summarizes the approaches tried, failures encountered, and the final working solution.

---

## The Core Problem

When pytest discovers and imports test files, Python's import system also imports the parent package's `__init__.py`. For TTS-Audio-Suite, this `__init__.py`:

1. Applies compatibility patches (numba, PyTorch, transformers)
2. Imports `nodes.py` which imports `folder_paths` (ComfyUI-only module)
3. Loads all node modules dynamically
4. Registers API endpoints with ComfyUI server

**Result:** Running `pytest tests/unit/` triggers the entire ComfyUI node loading chain, which fails because ComfyUI modules (`folder_paths`, `comfy`, `server`) don't exist in the test context.

---

## Approaches Tried (and Why They Failed)

### ❌ Approach 1: Environment Variable Guard in `__init__.py`

**Idea:** Check `os.environ['COMFYUI_TESTING']` at the top of `__init__.py` and skip loading.

```python
# __init__.py
if os.environ.get('COMFYUI_TESTING') == '1':
    NODE_CLASS_MAPPINGS = {}
    # ... skip rest of file
```

**Why it failed:** Would require wrapping 300+ lines of code in an `if/else` block, causing massive indentation changes. Python doesn't have a "skip rest of file" statement.

---

### ❌ Approach 2: `collect_ignore` in conftest.py

**Idea:** Tell pytest to ignore `__init__.py` during collection.

```python
# conftest.py
collect_ignore = ["__init__.py", "nodes.py"]
```

**Why it failed:** `collect_ignore` prevents pytest from *collecting tests* from those files, but doesn't prevent Python from *importing* them when test modules import utilities.

---

### ❌ Approach 3: Moving conftest.py to tests/ directory

**Idea:** If conftest.py is in `tests/`, pytest won't treat the parent as a package.

**Why it failed:** Pytest still finds `pytest.ini` in the parent and treats the directory structure as packages. The import chain is triggered when test files do `from utils.timing.parser import ...`.

---

### ❌ Approach 4: Using `--import-mode=importlib`

**Idea:** Pytest's alternative import mode might avoid `__init__.py` loading.

```bash
pytest --import-mode=importlib tests/
```

**Why it failed:** Same underlying issue - Python still imports parent packages.

---

### ❌ Approach 5: Using importlib.util in test files

**Idea:** Bypass package imports entirely by loading modules directly.

```python
# test_srt_parser.py
import importlib.util
parser_path = custom_node_root / "utils" / "timing" / "parser.py"
spec = importlib.util.spec_from_file_location("parser_module", parser_path)
parser_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser_module)
```

**Why it failed:** Conftest.py is still loaded first by pytest, and conftest.py's mere existence in the package triggers `__init__.py` import.

---

### ✅ Approach 6: Mock ComfyUI Modules at conftest.py Module Level (WORKING)

**The Solution:** Mock all ComfyUI modules *before* pytest imports anything else.

```python
# tests/conftest.py - MUST be at very top
import os
import sys
from unittest.mock import MagicMock

os.environ['COMFYUI_TESTING'] = '1'
os.environ['PYTEST_CURRENT_TEST'] = 'true'

# Mock ComfyUI modules BEFORE any other imports
_MOCK_MODULES = [
    'comfy', 'comfy.model_management', 'comfy.utils',
    'nodes', 'folder_paths', 'server', 'execution', 'comfy_extras',
]

for _module_name in _MOCK_MODULES:
    if _module_name not in sys.modules:
        sys.modules[_module_name] = MagicMock()

# NOW safe to import pytest, etc.
import pytest
```

**Why it works:** When `__init__.py` executes `import folder_paths`, Python finds a MagicMock already in `sys.modules` and uses that instead of trying to find the real module.

---

## Additional Bug Found

During testing, discovered a **real bug** in `nodes.py`:

```python
# BEFORE (buggy)
try:
    from utils.compatibility import setup_numba_compatibility
    setup_numba_compatibility(...)
except ImportError:
    import sys  # ← Only imported in except block!
    import os   # ← Only imported in except block!
    
# Later in file...
current_dir = os.path.dirname(__file__)  # ← NameError if try succeeded!
```

**Fix:** Move `import sys` and `import os` to the top of the file, outside the try/except block.

---

## Guidance for Future Developers

### 1. Test File Structure

```
TTS-Audio-Suite/
├── __init__.py           # Complex, avoid importing during tests
├── nodes.py              # Node registration  
├── pytest.ini            # Test configuration
├── tests/
│   ├── conftest.py       # CRITICAL: Mock ComfyUI here
│   ├── unit/
│   │   ├── test_srt_parser.py
│   │   └── test_audio_processing.py
│   └── integration/
│       └── test_workflows.py
```

### 2. conftest.py Template

```python
"""Mock ComfyUI at module level BEFORE any imports"""
import os
import sys
from unittest.mock import MagicMock

# Environment flags
os.environ['COMFYUI_TESTING'] = '1'

# Mock ComfyUI modules
for mod in ['folder_paths', 'comfy', 'server', 'execution']:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Now safe for normal imports
import pytest
# ... rest of fixtures
```

### 3. Unit Test Pattern

```python
# test_my_utility.py
import importlib.util
from pathlib import Path

custom_node_root = Path(__file__).parent.parent.parent
module_path = custom_node_root / "utils" / "my_utility.py"

spec = importlib.util.spec_from_file_location("my_utility", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Now use module.MyClass, module.my_function, etc.
```

### 4. Running Tests

```bash
# Unit tests (fast, no server)
python -m pytest tests/unit/ -m unit -v

# Integration tests (requires ComfyUI running)
python -m pytest tests/integration/ -m integration -v
```

### 5. Key Gotchas

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: folder_paths` | Add to mock list in conftest.py |
| `NameError: os not defined` | Check all imports are at module top level |
| Tests pass locally but fail in CI | Ensure conftest.py loads first |
| `__init__.py` still runs | Mock must happen at conftest.py *module level*, not in a function |

---

## Conclusion

The working solution requires:
1. **Mocking at conftest.py module level** (not in `pytest_configure` hook - too late)
2. **Ensuring `os`/`sys` imports are at top level** in production code
3. **Using importlib for utility imports** in test files (optional but cleaner)

This approach allows unit tests to run without ComfyUI while integration tests can still use the real server.
