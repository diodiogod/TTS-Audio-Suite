"""SHA-256 hex digests of files and byte strings."""

from __future__ import annotations

import hashlib
import os


def sha256_file(path: str | os.PathLike, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
