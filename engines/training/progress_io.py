"""
Best-effort JSON progress writes for long-running training jobs.

Windows can transiently deny ``os.replace`` when another process briefly holds
the target file open. Progress telemetry must never crash training, so these
writes retry a few times and then fail quietly.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any, Callable


def write_json_progress_file(
    path: str,
    payload: dict[str, Any],
    *,
    default: Callable[[Any], Any] | None = None,
    retries: int = 12,
    retry_delay_sec: float = 0.05,
) -> bool:
    if not path:
        return False

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    basename = os.path.basename(path)

    for attempt in range(max(int(retries), 1)):
        temp_path = os.path.join(
            directory,
            f".{basename}.{os.getpid()}.{int(time.time() * 1000)}.{attempt}.tmp",
        )
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, default=default)
            os.replace(temp_path, path)
            return True
        except OSError:
            with contextlib.suppress(OSError):
                os.remove(temp_path)
            if attempt + 1 >= max(int(retries), 1):
                break
            time.sleep(retry_delay_sec * (attempt + 1))

    return False

