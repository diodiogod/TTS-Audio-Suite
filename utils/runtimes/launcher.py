from __future__ import annotations

"""
Subprocess launcher for isolated engine workers.

This is scaffolding only. Engines are not routed through this path until an
engine-specific worker entrypoint is added.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from .profiles import RuntimeProfile
from .protocol import RuntimeJobRequest, RuntimeJobResponse


class IsolatedRuntimeLauncher:
    def __init__(self, runtime_root: Optional[str] = None):
        self.runtime_root = Path(runtime_root) if runtime_root else None

    def resolve_python_path(self, profile: RuntimeProfile, explicit_python: Optional[str] = None) -> str:
        if explicit_python:
            return explicit_python

        if profile.python_path_hint is None:
            raise RuntimeError(f"Runtime profile '{profile.name}' has no Python path hint")

        if self.runtime_root is None:
            return profile.python_path_hint

        return str(self.runtime_root / profile.python_path_hint)

    def build_env(self, profile: RuntimeProfile) -> Dict[str, str]:
        env = dict(os.environ)
        self._sanitize_pythonhashseed(env)
        current_pythonpath = env.get("PYTHONPATH", "")
        inherited_paths = [
            path for path in sys.path
            if path and self._should_inherit_pythonpath_entry(path)
        ]
        if current_pythonpath:
            inherited_paths.extend(
                path
                for path in current_pythonpath.split(os.pathsep)
                if path and self._should_inherit_pythonpath_entry(path)
            )
        deduped_paths = []
        seen = set()
        for path in inherited_paths:
            normalized = os.path.normcase(os.path.normpath(path))
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped_paths.append(path)
        if deduped_paths:
            env["PYTHONPATH"] = os.pathsep.join(deduped_paths)
        env["PYTHONUNBUFFERED"] = "1"
        env = self._augment_windows_toolchain_env(env)
        env.update(profile.env_vars)
        return env

    @staticmethod
    def _sanitize_pythonhashseed(env: Dict[str, str]) -> None:
        """Prevent invalid inherited hash seeds from crashing worker startup."""
        value = env.get("PYTHONHASHSEED")
        if value is None or value == "random":
            return

        try:
            valid = 0 <= int(value) <= 4294967295
        except (TypeError, ValueError):
            valid = False

        if not valid:
            env.pop("PYTHONHASHSEED", None)

    @classmethod
    def _augment_windows_toolchain_env(cls, env: Dict[str, str]) -> Dict[str, str]:
        if os.name != "nt":
            return env

        current_path = env.get("PATH", "")
        if shutil.which("cl", path=current_path) and env.get("INCLUDE") and env.get("LIB"):
            return env

        msvc_env = cls._get_windows_msvc_env()
        if not msvc_env:
            return env

        merged = dict(env)
        merged.update(msvc_env)
        cl_path = shutil.which("cl", path=merged.get("PATH", ""))
        if cl_path:
            merged.setdefault("CC", cl_path)
            merged.setdefault("CXX", cl_path)
        return merged

    @classmethod
    @lru_cache(maxsize=1)
    def _get_windows_msvc_env(cls) -> Optional[Dict[str, str]]:
        if os.name != "nt":
            return None

        vcvars_path = cls._find_vcvars64()
        if vcvars_path is None:
            return None

        try:
            with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False, encoding="utf-8") as temp_cmd:
                temp_cmd.write("@echo off\n")
                temp_cmd.write(f"call \"{vcvars_path}\" >nul\n")
                temp_cmd.write("set\n")
                temp_cmd_path = temp_cmd.name
            try:
                completed = subprocess.run(
                    ["cmd.exe", "/d", "/c", temp_cmd_path],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=dict(os.environ),
                )
            finally:
                try:
                    os.unlink(temp_cmd_path)
                except OSError:
                    pass
        except Exception:
            return None

        if completed.returncode != 0:
            return None

        resolved_env: Dict[str, str] = {}
        for line in completed.stdout.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if not key:
                continue
            resolved_env[key] = value

        if not resolved_env.get("PATH"):
            return None

        if not shutil.which("cl", path=resolved_env["PATH"]):
            return None

        return resolved_env

    @classmethod
    def _find_vcvars64(cls) -> Optional[str]:
        if os.name != "nt":
            return None

        vswhere_path = cls._find_vswhere()
        if vswhere_path:
            try:
                completed = subprocess.run(
                    [
                        vswhere_path,
                        "-latest",
                        "-products",
                        "*",
                        "-requires",
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                        "-property",
                        "installationPath",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                installation_path = completed.stdout.strip()
                if installation_path:
                    candidate = Path(installation_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                    if candidate.exists():
                        return str(candidate)
            except Exception:
                pass

        candidate_roots = [
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio",
        ]
        discovered = []
        for root in candidate_roots:
            if not root.exists():
                continue
            discovered.extend(root.glob("*/*/VC/Auxiliary/Build/vcvars64.bat"))

        if not discovered:
            return None

        discovered.sort(key=lambda path: str(path), reverse=True)
        return str(discovered[0])

    @staticmethod
    def _find_vswhere() -> Optional[str]:
        if os.name != "nt":
            return None

        installer_root = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer"
        candidate = installer_root / "vswhere.exe"
        if candidate.exists():
            return str(candidate)
        return None

    @staticmethod
    def _should_inherit_pythonpath_entry(path: str) -> bool:
        normalized = os.path.normcase(os.path.normpath(path))
        if not normalized:
            return False

        lower = normalized.lower()
        if "site-packages" in lower or "dist-packages" in lower:
            return False

        try:
            prefix = os.path.normcase(os.path.normpath(sys.prefix))
            if normalized.startswith(prefix):
                return False
        except Exception:
            pass

        try:
            base_prefix = os.path.normcase(os.path.normpath(sys.base_prefix))
            if normalized.startswith(base_prefix):
                return False
        except Exception:
            pass

        return True

    def run_json_worker(
        self,
        *,
        profile: RuntimeProfile,
        worker_script: str,
        request: RuntimeJobRequest,
        explicit_python: Optional[str] = None,
    ) -> RuntimeJobResponse:
        python_path = self.resolve_python_path(profile, explicit_python=explicit_python)

        with tempfile.TemporaryDirectory(prefix="tts_runtime_") as temp_dir:
            temp_path = Path(temp_dir)
            request_path = temp_path / "request.json"
            response_path = temp_path / "response.json"
            request_path.write_text(json.dumps(request.to_dict(), ensure_ascii=True), encoding="utf-8")

            command = [
                python_path,
                worker_script,
                "--request",
                str(request_path),
                "--response",
                str(response_path),
            ]

            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=self.build_env(profile),
            )

            if not response_path.exists():
                return RuntimeJobResponse(
                    ok=False,
                    error=(
                        f"Isolated runtime worker failed before writing a response "
                        f"(exit_code={completed.returncode})"
                    ),
                    logs=[
                        completed.stdout.strip(),
                        completed.stderr.strip(),
                    ],
                    request_id=request.request_id,
                )

            data = json.loads(response_path.read_text(encoding="utf-8"))
            response = RuntimeJobResponse(**data)

            if completed.stdout.strip():
                response.logs.append(completed.stdout.strip())
            if completed.stderr.strip():
                response.logs.append(completed.stderr.strip())

            return response
