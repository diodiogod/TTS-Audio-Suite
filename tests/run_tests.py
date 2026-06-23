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

# Import shared configuration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import env_config
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
    import env_config

TEST_DIR = env_config.TESTS_DIR
COMFYUI_ROOT = env_config.COMFYUI_ROOT
PYTHON_EXE = env_config.VENV_PYTHON


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
    print(f"\n[TESTS] Executing: {' '.join(pytest_args)}")

    result = subprocess.run(pytest_args, env=os.environ)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

