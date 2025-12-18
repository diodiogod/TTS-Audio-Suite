"""
TTS-Audio-Suite Test Configuration

Session-scoped fixtures for ComfyUI server management and API testing.
"""

# ============================================================================
# CRITICAL: Mock ComfyUI modules BEFORE anything else
# This prevents import errors when pytest discovers and imports test modules
# ============================================================================
import os
import sys
from unittest.mock import MagicMock

# Set testing environment
os.environ['COMFYUI_TESTING'] = '1'
os.environ['PYTEST_CURRENT_TEST'] = 'true'

# Mock ComfyUI modules at module level - BEFORE any imports can trigger __init__.py
_MOCK_MODULES = [
    'comfy',
    'comfy.model_management',
    'comfy.utils',
    'nodes',
    'folder_paths',
    'server',
    'execution',
    'comfy_extras',
]

for _module_name in _MOCK_MODULES:
    if _module_name not in sys.modules:
        sys.modules[_module_name] = MagicMock()

# Now safe to import pytest and other modules
import pytest
import subprocess
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# Tell pytest to ignore certain files during collection
collect_ignore = ["__init__.py", "nodes.py"]

# Path configuration
CUSTOM_NODE_ROOT = Path(__file__).parent.parent  # tests/ -> TTS-Audio-Suite/
COMFY_ROOT = CUSTOM_NODE_ROOT.parent.parent  # Navigate to Comfy-new
VENV_PYTHON = COMFY_ROOT / "venv" / "Scripts" / "python.exe"  # Windows


class ComfyUIAPIClient:
    """Helper class for interacting with ComfyUI API during tests"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client_id = "pytest-test-client"
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system stats to verify server is running"""
        response = requests.get(f"{self.base_url}/system_stats", timeout=5)
        response.raise_for_status()
        return response.json()
    
    def get_object_info(self) -> Dict[str, Any]:
        """Get all registered node info - useful for verifying nodes loaded"""
        response = requests.get(f"{self.base_url}/object_info", timeout=30)
        response.raise_for_status()
        return response.json()
    
    def queue_prompt(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a workflow for execution"""
        response = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get execution history for a prompt"""
        response = requests.get(
            f"{self.base_url}/history/{prompt_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 120) -> Dict[str, Any]:
        """Wait for workflow to complete"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                history = self.get_history(prompt_id)
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    status = history[prompt_id].get("status", {})
                    
                    # Check if completed
                    if status.get("completed", False):
                        return history[prompt_id]
                    
                    # Check for error
                    status_str = status.get("status_str", "")
                    if status_str == "error":
                        raise RuntimeError(
                            f"Workflow execution failed: {status}"
                        )
            except requests.RequestException:
                pass  # Server might be busy
            
            time.sleep(0.5)
        
        raise TimeoutError(
            f"Workflow did not complete within {timeout}s (prompt_id: {prompt_id})"
        )
    
    def execute_workflow(self, workflow: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        """Queue workflow and wait for completion"""
        result = self.queue_prompt(workflow)
        prompt_id = result["prompt_id"]
        return self.wait_for_completion(prompt_id, timeout)
    
    def node_exists(self, node_class: str) -> bool:
        """Check if a node class is registered"""
        try:
            object_info = self.get_object_info()
            return node_class in object_info
        except Exception:
            return False


@pytest.fixture(scope="session")
def comfyui_server():
    """
    Start ComfyUI server for the entire test session.
    Automatically shuts down when tests complete.
    
    Yields:
        dict with 'url' and 'process' keys
    """
    print("\n🚀 Starting ComfyUI server for integration tests...")
    
    # Check if server is already running
    server_url = "http://127.0.0.1:8188"
    try:
        response = requests.get(f"{server_url}/system_stats", timeout=2)
        if response.status_code == 200:
            print(f"✅ ComfyUI server already running at {server_url}")
            yield {"url": server_url, "process": None, "external": True}
            return
    except (requests.ConnectionError, requests.Timeout):
        pass  # Server not running, we'll start it
    
    # Start ComfyUI in subprocess
    process = subprocess.Popen(
        [
            str(VENV_PYTHON),
            "main.py",
            "--listen", "127.0.0.1",
            "--port", "8188",
            "--disable-auto-launch",
            "--cpu"  # Use CPU for faster startup in tests
        ],
        cwd=str(COMFY_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Wait for server to be ready
    max_retries = 120  # 2 minutes timeout for model loading
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.get(f"{server_url}/system_stats", timeout=2)
            if response.status_code == 200:
                print(f"✅ ComfyUI server ready at {server_url}")
                break
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(1)
            retry_count += 1
            if retry_count % 15 == 0:
                print(f"⏳ Waiting for server... ({retry_count}s)")
    
    if retry_count >= max_retries:
        # Try to get any error output
        process.terminate()
        try:
            stdout, _ = process.communicate(timeout=5)
            error_lines = stdout[-2000:] if stdout else "No output"
        except Exception:
            error_lines = "Could not capture output"
        
        raise RuntimeError(
            f"❌ ComfyUI server failed to start within {max_retries}s\n"
            f"Last output:\n{error_lines}"
        )
    
    yield {"url": server_url, "process": process, "external": False}
    
    # Teardown
    if process and process.poll() is None:
        print("\n🛑 Shutting down ComfyUI server...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        print("✅ Server stopped")


@pytest.fixture
def api_client(comfyui_server) -> ComfyUIAPIClient:
    """Get API client connected to running ComfyUI server"""
    return ComfyUIAPIClient(comfyui_server["url"])


@pytest.fixture
def workflow_fixtures_path() -> Path:
    """Path to workflow test fixtures"""
    return CUSTOM_NODE_ROOT / "tests" / "fixtures" / "workflows"


@pytest.fixture
def sample_voice_path() -> Optional[Path]:
    """Path to a sample voice file for testing (if available)"""
    voices_dir = CUSTOM_NODE_ROOT / "voices_examples"
    if voices_dir.exists():
        # Find any .wav file
        for wav_file in voices_dir.rglob("*.wav"):
            return wav_file
    return None


# ============================================================================
# Unit test fixtures (no server required)
# ============================================================================

@pytest.fixture
def sample_srt_content() -> str:
    """Sample SRT content for parser testing"""
    return """1
00:00:01,000 --> 00:00:04,000
Hello, this is the first subtitle.

2
00:00:05,000 --> 00:00:08,500
And this is the second one.

3
00:00:10,000 --> 00:00:12,000
Final subtitle here.
"""


@pytest.fixture
def sample_text_with_pauses() -> str:
    """Sample text with pause tags for testing"""
    return "Hello! [pause:2] How are you today? [wait:500ms] I hope you're doing well."


@pytest.fixture
def sample_text_with_characters() -> str:
    """Sample text with character tags for testing"""
    return """[Alice]
Hello everyone! Welcome to the show.

[Bob]
Thanks Alice! Great to be here.

[Alice]
Let's get started!
"""
