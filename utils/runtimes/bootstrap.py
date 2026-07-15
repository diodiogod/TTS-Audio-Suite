from __future__ import annotations

"""
Bootstrap isolated Python runtimes on demand.
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .profiles import RuntimeProfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "runtimes"
BOOTSTRAP_STRATEGY = "profile_packages_v6"


def _venv_python_path(runtime_dir: Path) -> Path:
    if os.name == "nt":
        return runtime_dir / "Scripts" / "python.exe"
    return runtime_dir / "bin" / "python"


def resolve_runtime_dir(profile: RuntimeProfile) -> Path:
    hint = Path(profile.python_path_hint or "")
    if hint.parts and hint.parts[0] == "runtimes" and len(hint.parents) >= 2:
        return PROJECT_ROOT / hint.parent.parent
    return RUNTIME_ROOT / profile.name


def resolve_runtime_python(profile: RuntimeProfile) -> Path:
    return _venv_python_path(resolve_runtime_dir(profile))


def _venv_site_packages_path(runtime_dir: Path) -> Path:
    if os.name == "nt":
        return runtime_dir / "Lib" / "site-packages"
    major = sys.version_info.major
    minor = sys.version_info.minor
    return runtime_dir / "lib" / f"python{major}.{minor}" / "site-packages"


def _detect_base_runtime_context(source_python: str) -> Optional[dict]:
    probe = (
        "import json\n"
        "import os\n"
        "import site\n"
        "try:\n"
        " import torch\n"
        "except Exception:\n"
        " torch = None\n"
        "try:\n"
        " import torchaudio\n"
        "except Exception:\n"
        " torchaudio = None\n"
        "print(json.dumps({"
        "'torch': getattr(torch, '__version__', None), "
        "'torchaudio': getattr(torchaudio, '__version__', None), "
        "'site_packages': site.getsitepackages()"
        "}))\n"
    )
    try:
        result = subprocess.run(
            [source_python, "-c", probe],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout.strip())
        return payload
    except Exception:
        return None


def _inherit_base_site_packages(runtime_dir: Path, source_python: str) -> Optional[str]:
    context = _detect_base_runtime_context(source_python)
    if not context:
        return None

    site_packages = context.get("site_packages") or []
    if not site_packages:
        return None

    base_site_packages = None
    for candidate in site_packages:
        if candidate and os.path.exists(os.path.join(candidate, "torch")):
            base_site_packages = candidate
            break
    if not base_site_packages:
        base_site_packages = next((candidate for candidate in site_packages if candidate), None)
    if not base_site_packages:
        return None

    venv_site_packages = _venv_site_packages_path(runtime_dir)
    venv_site_packages.mkdir(parents=True, exist_ok=True)
    pth_path = venv_site_packages / "_tts_audio_suite_base_runtime.pth"
    pth_path.write_text(base_site_packages + "\n", encoding="utf-8")

    print(f"🔧 Reusing base runtime site-packages: {base_site_packages}")
    return base_site_packages


def _run_bootstrap_command(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )


def _ensure_virtualenv_available(source_python: str) -> None:
    probe = subprocess.run(
        [source_python, "-m", "virtualenv", "--version"],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return

    print("🔧 Installing virtualenv in base runtime for isolated-runtime bootstrap fallback")
    install = subprocess.run(
        [source_python, "-m", "pip", "install", "virtualenv"],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        stderr = (install.stderr or "").strip()
        stdout = (install.stdout or "").strip()
        details = stderr or stdout or "unknown error"
        raise RuntimeError(
            "Failed to install virtualenv for isolated-runtime bootstrap fallback. "
            f"Command: {source_python} -m pip install virtualenv\n{details}"
        )


def _create_runtime_environment(runtime_dir: Path, source_python: str) -> None:
    venv_command = [source_python, "-m", "venv", str(runtime_dir)]
    venv_result = _run_bootstrap_command(venv_command)
    if venv_result.returncode == 0:
        return

    print("⚠️ Standard venv bootstrap failed; trying virtualenv fallback")
    shutil.rmtree(runtime_dir, ignore_errors=True)
    _ensure_virtualenv_available(source_python)

    virtualenv_command = [source_python, "-m", "virtualenv", str(runtime_dir)]
    virtualenv_result = _run_bootstrap_command(virtualenv_command)
    if virtualenv_result.returncode == 0:
        return

    stderr = (virtualenv_result.stderr or "").strip()
    stdout = (virtualenv_result.stdout or "").strip()
    fallback_details = stderr or stdout or "unknown error"

    original_stderr = (venv_result.stderr or "").strip()
    original_stdout = (venv_result.stdout or "").strip()
    original_details = original_stderr or original_stdout or "unknown error"

    raise RuntimeError(
        "Failed to create isolated runtime environment.\n"
        f"Primary command: {' '.join(venv_command)}\n"
        f"Primary error: {original_details}\n"
        f"Fallback command: {' '.join(virtualenv_command)}\n"
        f"Fallback error: {fallback_details}"
    )


def ensure_runtime(
    profile: RuntimeProfile,
    *,
    base_python: Optional[str] = None,
) -> Path:
    runtime_dir = resolve_runtime_dir(profile)
    python_path = _venv_python_path(runtime_dir)
    metadata_path = runtime_dir / "runtime_metadata.json"

    if python_path.exists():
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    metadata.get("bootstrap_strategy") == BOOTSTRAP_STRATEGY
                    and metadata.get("install_complete") is True
                    and metadata.get("profile") == asdict(profile)
                ):
                    return python_path
            except Exception:
                pass

        print(
            f"🔧 Rebuilding isolated runtime '{profile.name}' because it was created "
            f"with an older bootstrap strategy or profile"
        )
        shutil.rmtree(runtime_dir, ignore_errors=True)

    runtime_dir.parent.mkdir(parents=True, exist_ok=True)

    source_python = base_python or sys.executable
    print(
        f"🔧 Creating isolated runtime '{profile.name}' at {runtime_dir} "
        f"(reuses heavy base packages like PyTorch from the main runtime when configured)"
    )

    _create_runtime_environment(runtime_dir, source_python)

    subprocess.run(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    inherited_site_packages = None
    if profile.inherit_base_site_packages:
        inherited_site_packages = _inherit_base_site_packages(runtime_dir, source_python)
        if not inherited_site_packages:
            print("⚠️ Failed to inherit base site-packages; isolated runtime will rely only on local installs")

    if not profile.pip_packages:
        raise RuntimeError(
            f"Runtime profile '{profile.name}' has no package list. "
            f"Do not route engines into isolated mode until that profile is defined."
        )

    print(f"🔧 Installing profile-specific dependencies for '{profile.name}'")
    subprocess.run(
        [str(python_path), "-m", "pip", "install", *profile.pip_packages],
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    if profile.pip_packages_no_deps:
        print(
            f"🔧 Installing source packages without dependency resolution for "
            f"'{profile.name}'"
        )
        subprocess.run(
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "--no-deps",
                *profile.pip_packages_no_deps,
            ],
            cwd=str(PROJECT_ROOT),
            check=True,
        )

    metadata = {
        "profile": asdict(profile),
        "bootstrap_strategy": BOOTSTRAP_STRATEGY,
        "install_complete": True,
        "inherited_base_site_packages": inherited_site_packages,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_python": source_python,
    }
    (runtime_dir / "runtime_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    return python_path
