#!/usr/bin/env python
"""
Universal ComfyUI Custom Node Test Runner.

Handles working directory isolation, cross-platform venv detection,
and forwards execution to pytest.

Usage (from the custom node root):
    python tests/run_tests.py                 # all tests
    python tests/run_tests.py unit/           # unit only
    python tests/run_tests.py integration/    # integration only
    python tests/run_tests.py -m unit         # via marker
"""

import os
import sys
import subprocess

# Flag for code paths that need to know they're under test
os.environ["COMFYUI_TESTING"] = "1"

# ---------------------------------------------------------------------------
# Path routing
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Works whether invoked from repo root or inside tests/
TEST_DIR = SCRIPT_DIR if SCRIPT_DIR.endswith("tests") else os.path.join(SCRIPT_DIR, "tests")
COMFYUI_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", "..", ".."))
if not os.path.isdir(os.path.join(COMFYUI_ROOT, "venv")):
    # Fallback to two levels up if venv is not found (e.g. if run from a different layout)
    COMFYUI_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", ".."))

# Cross-platform venv resolution
if os.name == "nt":
    PYTHON_EXE = os.path.join(COMFYUI_ROOT, "venv", "Scripts", "python.exe")
else:
    PYTHON_EXE = os.path.join(COMFYUI_ROOT, "venv", "bin", "python")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    # Isolate to tests/ so the local pytest.ini wins over ComfyUI's root config
    os.chdir(TEST_DIR)

    # Check if a path argument is explicitly provided
    has_path = False
    for arg in args:
        if not arg.startswith("-") and (os.path.exists(arg) or "/" in arg or arg.endswith(".py")):
            has_path = True
            break

    pytest_args = [PYTHON_EXE, "-m", "pytest"]
    if not has_path:
        # If no explicit path is given, we let pytest.ini's testpaths control collection
        # rather than forcing "." which overrides testpaths and collects root scripts
        pass
    pytest_args.extend(args)
    print(f"\n📋 Executing: {' '.join(pytest_args)}")

    result = subprocess.run(pytest_args, env=os.environ)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
