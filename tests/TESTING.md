# TTS Audio Suite - Testing Guide

This guide covers running tests for the TTS Audio Suite custom node.

## Quick Start

For a split installation, first create the ignored `tests/.env.local` file described under [Path Configuration & Local Overrides](#path-configuration--local-overrides). The wrapper cannot infer paths when ComfyUI and this repository live in separate directory trees.

```bash
cd /path/to/ComfyUI/custom_nodes/TTS-Audio-Suite

# Run all tests
python tests/run_tests.py -v

# Run only fast unit tests (no server needed)
python tests/run_tests.py unit/ -m unit -v

# Run integration tests (requires ComfyUI running)
python tests/run_tests.py integration/ -m integration -v
```

Always use `tests/run_tests.py`; it selects the configured Python, changes into the isolated test directory, and ensures the local pytest configuration wins.

---

## Test Categories

### Unit Tests (64 tests)
Fast tests that validate utility functions without ComfyUI.

| Module | Tests | Description |
|--------|-------|-------------|
| `test_srt_parser.py` | 23 | SRT timestamp parsing, validation, subtitle handling |
| `test_pause_processor.py` | 19 | Pause tag detection, silence generation |
| `test_audio_processing.py` | 22 | Time conversions, silence creation, audio normalization |

### Integration Tests (23 tests)
Tests that run against a live ComfyUI server.

| Category | Tests | Description |
|----------|-------|-------------|
| Node Registration | 10 | Verify all TTS nodes are registered |
| Engine Config | 4 | Validate engine configuration workflows |
| Workflow Fixtures | 4 | Load and validate workflow JSON files |
| Server Health | 3 | API endpoints, node counts |
| E2E Generation | 2 | Full TTS generation with audio output |

---

## Test Markers

```bash
# Run by marker
python tests/run_tests.py -m unit           # Fast, no server
python tests/run_tests.py -m integration    # Requires ComfyUI
python tests/run_tests.py -m slow           # Full generation tests
python tests/run_tests.py -m cosyvoice      # CosyVoice3 specific
```

---

## Running Integration Tests

Integration tests require ComfyUI to be running:

```bash
# Terminal 1: Start ComfyUI
cd /path/to/ComfyUI
/path/to/comfyui/python main.py

# Terminal 2: Run tests
cd /path/to/ComfyUI/custom_nodes/TTS-Audio-Suite
python tests/run_tests.py integration/ -m integration -v
```

---

## FL-MCP-Assisted Live Validation

[ComfyUI_FL-MCP](https://github.com/filliptm/ComfyUI_FL-MCP) can drive and inspect a live ComfyUI session from an MCP-capable coding client. It complements pytest by exposing workflow state, browser screenshots, queue execution, history, errors, and output files. It is optional and is not a dependency of TTS Audio Suite.

Use it for new-engine acceptance testing after the implementation and automated tests are complete. Follow [FL_MCP_VALIDATION.md](FL_MCP_VALIDATION.md) for prerequisites, the standard smoke sequence, required evidence, safety boundaries, and the manual fallback.

New custom-node Python code is loaded only when ComfyUI starts. During implementation, follow the runbook's mandatory edit–restart–browser reconnect–validate loop. Keeping an old ComfyUI process open does not test newly written engine code.

REST availability alone is insufficient for canvas inspection. Before live validation, bind the MCP client to the browser's persistent FL-MCP session ID and verify one bridge session reports both frontend and MCP connections.

FL-MCP does not determine whether generated speech sounds good. A human must assess pronunciation, voice similarity, artifacts, pacing, and other subjective audio qualities.

---

## Test Fixtures

Test workflow files are in `tests/fixtures/workflows/`:

| File | Purpose |
|------|---------|
| `test_chatterbox_e2e.json` | Full ChatterBox generation pipeline |
| `test_cosyvoice_e2e.json` | Full CosyVoice3 generation pipeline |
| `test_chatterbox_engine.json` | Engine config validation |
| `test_cosyvoice_engine.json` | Engine config validation |
| `test_f5tts_engine.json` | F5-TTS config validation |
| `test_indextts_engine.json` | IndexTTS config validation |

---

## Path Configuration & Local Overrides

By default, the tests attempt to locate ComfyUI and its python virtual environment using default relative paths (`../../../`).

If you use a custom setup (such as stability-matrix or a different venv location), you can override these paths locally without editing any tracked git files:

1. Create a file named `tests/.env.local` (this file is ignored by git).
2. Define the path overrides:

```env
# Root directory of your ComfyUI installation
TTS_SUITE_TEST_COMFY_ROOT=/path/to/ComfyUI

# Path to the python executable inside the virtual environment
TTS_SUITE_TEST_VENV_PYTHON=/path/to/ComfyUI/venv/bin/python
```

These overrides will be automatically picked up by `tests/env_config.py` when you run tests.

---

## For Developers

### Adding Unit Tests

1. Create test file in `tests/unit/test_*.py`
2. Mark tests with `@pytest.mark.unit`
3. Import modules using `importlib` to avoid triggering `__init__.py`

```python
import pytest
from pathlib import Path
import sys

# Add custom node root before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.timing.parser import SRTParser

@pytest.mark.unit
class TestMyFeature:
    def test_something(self):
        assert True
```

### Adding Integration Tests

1. Create test in `tests/integration/test_*.py`
2. Mark with `@pytest.mark.integration`
3. Use provided fixtures: `api_client`, `workflow_fixtures_path`

```python
@pytest.mark.integration
def test_my_workflow(self, api_client, workflow_fixtures_path):
    workflow = load_workflow(workflow_fixtures_path / "my_test.json")
    result = api_client.queue_prompt(workflow)
    assert "prompt_id" in result
```

### Test Isolation

The test suite mocks ComfyUI modules (`folder_paths`, `comfy`, etc.) in `tests/conftest.py` to allow unit tests to run without a server. See `docs/Dev reports/pytest_implementation_report.md` for technical details.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: folder_paths` | Ensure `tests/conftest.py` is present (it mocks ComfyUI) |
| Integration tests hang | Check ComfyUI is running at `http://127.0.0.1:8188` |
| E2E tests fail with model errors | Models auto-download on first run, may take time |
| `ImportError` in unit tests | Don't import from package root, use direct module paths |

---

## Architectural Roadblocks & Solutions

This section documents critical issues encountered when setting up pytest for ComfyUI custom nodes:

### 1. Pytest Package Discovery vs. ComfyUI
ComfyUI packages use relative imports in `__init__.py` (e.g., `from .nodes import ...`). When pytest runs from the project root, it tries to import the package, which fails with `ImportError: attempted relative import with no known parent package`.

**Solution:** The `run_tests.py` script changes directory to `tests/` before running pytest (`os.chdir(os.path.join(script_dir, 'tests'))`). This prevents pytest from treating the root as a package.

### 2. Parent pytest.ini Interference
ComfyUI's root `pytest.ini` has `pythonpath = .` which adds the package root to `sys.path`, triggering package discovery issues.

**Solution:** Use a local `tests/pytest.ini` with `pythonpath = ` (empty) to override parent settings.

### 3. Module Dependency Isolation
Node modules typically depend on ComfyUI (`folder_paths`, `comfy`, etc.). Only modules in `utils/` subdirectories that are free of ComfyUI dependencies can be safely unit tested standalone.

### 4. Key Takeaways
- **Use `run_tests.py`**: Always run tests via the wrapper script, not directly with pytest.
- **Don't modify `__init__.py`**: Adding `COMFYUI_TESTING` checks to skip imports breaks node registration in the actual server.
- **Isolate testable code**: Keep utility functions in `utils/` subdirectories separate from node definitions.
