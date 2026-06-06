from __future__ import annotations

"""
Subprocess launcher for isolated engine workers.

This is scaffolding only. Engines are not routed through this path until an
engine-specific worker entrypoint is added.
"""

import json
import os
import subprocess
import sys
import tempfile
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
        env.update(profile.env_vars)
        return env

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
