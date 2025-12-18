# TTS Audio Suite - Testing Guide

This guide covers running tests for the TTS Audio Suite custom node.

## Quick Start

```bash
cd C:\_stability_matrix\Data\Packages\Comfy-new\custom_nodes\TTS-Audio-Suite

# Run all tests
..\..\venv\Scripts\python -m pytest tests/ -v

# Run only fast unit tests (no server needed)
..\..\venv\Scripts\python -m pytest tests/unit/ -m unit -v

# Run integration tests (requires ComfyUI running)
..\..\venv\Scripts\python -m pytest tests/integration/ -m integration -v
```

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
pytest -m unit           # Fast, no server
pytest -m integration    # Requires ComfyUI
pytest -m slow           # Full generation tests
pytest -m cosyvoice      # CosyVoice3 specific
```

---

## Running Integration Tests

Integration tests require ComfyUI to be running:

```bash
# Terminal 1: Start ComfyUI
cd C:\_stability_matrix\Data\Packages\Comfy-new
venv\Scripts\python main.py

# Terminal 2: Run tests
cd custom_nodes\TTS-Audio-Suite
..\..\venv\Scripts\python -m pytest tests/integration/ -m integration -v
```

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
