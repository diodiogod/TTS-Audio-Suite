#!/usr/bin/env python
"""
TTS-Audio-Suite Test Runner

This script runs pytest with proper environment isolation to prevent
the main TTS-Audio-Suite __init__.py from being loaded during unit tests.

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py -m unit            # Run only unit tests
    python tests/run_tests.py -m integration     # Run only integration tests
"""

import os
import sys
import subprocess

# Set testing environment BEFORE any imports
os.environ['COMFYUI_TESTING'] = '1'

# Get paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

def check_pytest(python_path):
    if not os.path.exists(python_path):
        return False
    try:
        res = subprocess.run([python_path, '-c', 'import pytest'], capture_output=True)
        return res.returncode == 0
    except Exception:
        return False

# Detect environments
if os.name == 'nt':
    local_venv = os.path.join(project_dir, '.venv', 'Scripts', 'python.exe')
    parent_venv_2 = os.path.join(script_dir, '..', '..', 'venv', 'Scripts', 'python.exe')
    parent_venv_3 = os.path.join(script_dir, '..', '..', '..', 'venv', 'Scripts', 'python.exe')
else:
    local_venv = os.path.join(project_dir, '.venv', 'bin', 'python')
    parent_venv_2 = os.path.join(script_dir, '..', '..', 'venv', 'bin', 'python')
    parent_venv_3 = os.path.join(script_dir, '..', '..', '..', 'venv', 'bin', 'python')

if check_pytest(local_venv):
    venv_python = local_venv
elif check_pytest(parent_venv_2):
    venv_python = parent_venv_2
elif check_pytest(parent_venv_3):
    venv_python = parent_venv_3
else:
    venv_python = sys.executable

# Build pytest command
pytest_args = [
    venv_python,
    '-m', 'pytest',
    script_dir,
    '-v',
    '--tb=short',
]

# Add any additional arguments passed to this script
pytest_args.extend(sys.argv[1:])

# Run from tests directory to avoid package import
os.chdir(script_dir)

print(f"Running: {' '.join(pytest_args)}")
result = subprocess.run(pytest_args, env=os.environ)
sys.exit(result.returncode)
