from __future__ import annotations

"""
Persistent JSON-line worker session for isolated runtimes.
"""

import json
import subprocess
import threading
from pathlib import Path
from typing import Dict, Optional

from .protocol import RuntimeJobRequest, RuntimeJobResponse


class JsonLineWorkerSession:
    def __init__(
        self,
        python_path: str,
        worker_script: str,
        env: Optional[Dict[str, str]] = None,
    ):
        self.python_path = python_path
        self.worker_script = worker_script
        self.env = env
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        self._process = subprocess.Popen(
            [self.python_path, self.worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=self.env,
            cwd=str(Path(self.worker_script).resolve().parents[3]),
        )

    def request(self, request: RuntimeJobRequest) -> RuntimeJobResponse:
        self.start()
        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None

        with self._lock:
            if self._process.poll() is not None:
                return RuntimeJobResponse(
                    ok=False,
                    error=f"Worker exited unexpectedly with code {self._process.returncode}",
                    request_id=request.request_id,
                )

            self._process.stdin.write(json.dumps(request.to_dict(), ensure_ascii=True) + "\n")
            self._process.stdin.flush()

            stdout_noise = []
            while True:
                line = self._process.stdout.readline()
                if not line:
                    error = "Worker closed the response stream unexpectedly"
                    if self._process.poll() is not None:
                        error = f"{error} (exit_code={self._process.returncode})"
                    if stdout_noise:
                        error = f"{error}\n" + "\n".join(stdout_noise)
                    return RuntimeJobResponse(
                        ok=False,
                        error=error,
                        logs=stdout_noise,
                        request_id=request.request_id,
                    )

                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    payload = json.loads(stripped)
                except Exception:
                    stdout_noise.append(stripped)
                    continue

                response = RuntimeJobResponse(**payload)
                if stdout_noise:
                    response.logs.extend(stdout_noise)
                return response

    def close(self) -> None:
        if self._process is None:
            return

        if self._process.poll() is None:
            try:
                self.request(
                    RuntimeJobRequest(
                        engine_name="runtime",
                        action="shutdown",
                        model_name="",
                        device="cpu",
                    )
                )
            except Exception:
                pass

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)

        self._process = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
