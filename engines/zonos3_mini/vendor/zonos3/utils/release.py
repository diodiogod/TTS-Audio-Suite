"""hf:// locations for the ZONOS3 Mini release checkpoints.

Registries, configs and the server's default --model-path may name files and folders in a Hugging Face repo as
``hf://<owner>/<repo>/<path inside the repo>``. ``resolve_file`` / ``resolve_dir_or_local`` fetch them into the local HF cache
(``HF_HOME``; ``HF_TOKEN`` or a stored login for private repos) and return local paths; plain local paths pass through
unchanged, so the cluster servers keep running from their local registries. Default repo: ``ZONOS3_RELEASE_REPO``
(Zyphra/ZONOS3-Mini, public) and its language model ``lm/zonos3_mini``.
"""

from __future__ import annotations

import os
# TTS Audio Suite Patch: hf:// locations resolve inside a local model root (set_local_root); nothing is downloaded here.
from functools import lru_cache
from pathlib import Path

PREFIX = "hf://"
RELEASE_REPO = os.environ.get("ZONOS3_RELEASE_REPO", "Zyphra/ZONOS3-Mini")
RELEASE_LM = os.environ.get("ZONOS3_RELEASE_LM", f"{PREFIX}{RELEASE_REPO}/lm/zonos3_mini")
RELEASE_SPEAKER_REGISTRY = f"{PREFIX}{RELEASE_REPO}/registries/speaker_encoder_registry.json"
# The release LM's phoneme, start and pause rows (zonos-text-extra/1), loaded by default with RELEASE_LM.
RELEASE_TEXT_ROWS = os.environ.get(
    "ZONOS3_RELEASE_TEXT_ROWS", f"{PREFIX}{RELEASE_REPO}/lm/zonos3_mini/text_rows/pause_tokens_s5000.pt"
)


def is_hf(path) -> bool:
    return isinstance(path, str) and path.startswith(PREFIX)


def split(uri: str) -> tuple[str, str]:
    """hf://owner/repo/a/b -> ('owner/repo', 'a/b'); hf://owner/repo -> ('owner/repo', '')."""
    if not is_hf(uri):
        raise ValueError(f"not an hf:// location: {uri!r}")
    parts = uri[len(PREFIX) :].strip("/").split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"hf:// location needs owner/repo: {uri!r}")
    return f"{parts[0]}/{parts[1]}", (parts[2] if len(parts) > 2 else "")


def hf_join(base, *parts) -> str:
    """Join path parts onto a base that may be an hf:// location (pathlib would collapse the '//')."""
    if is_hf(base):
        return base.rstrip("/") + "".join("/" + str(p).strip("/") for p in parts if str(p))
    return str(Path(str(base), *[str(p) for p in parts]))


_LOCAL_ROOT: str | None = None


def set_local_root(path: str | None) -> None:
    """TTS Audio Suite Patch: the folder holding the Zyphra/ZONOS3-Mini repo layout that hf:// locations map into."""
    global _LOCAL_ROOT
    _LOCAL_ROOT = None if path is None else str(path)
    resolve_file.cache_clear()
    resolve_dir_or_local.cache_clear()


def _local(path: str) -> str:
    repo_id, inside = split(path)
    if repo_id != RELEASE_REPO:
        raise ValueError(f"ZONOS3-Mini: unexpected model repository {repo_id}")
    if _LOCAL_ROOT is None:
        raise RuntimeError("ZONOS3-Mini: local model root is not set")
    local = os.path.join(_LOCAL_ROOT, inside) if inside else _LOCAL_ROOT
    if not os.path.exists(local):
        raise FileNotFoundError(f"ZONOS3-Mini model file missing: {local}")
    return local


@lru_cache(maxsize=None)
def resolve_file(path) -> str:
    """Local path for a file: hf:// locations map into the local model root, others pass through."""
    if path is None:
        return path
    path = str(path)
    if not is_hf(path):
        return os.path.expanduser(path)
    if not split(path)[1]:
        raise ValueError(f"hf:// file location needs a path inside the repo: {path!r}")
    return _local(path)


@lru_cache(maxsize=None)
def resolve_dir_or_local(path) -> str:
    """Local directory for a folder: hf:// locations map into the local model root, others pass through."""
    if path is None:
        return path
    path = str(path)
    if not is_hf(path):
        return os.path.expanduser(path)
    return _local(path)


def resolve_tokenizer_path(path, model_dir) -> str:
    """The text tokenizer file of a checkpoint: hf://, absolute, or relative to the checkpoint folder. A cluster-absolute
    path that is missing here falls back to tokenizer/<basename> next to params.json (the release layout)."""
    if path is None:
        return path
    path = str(path)
    if is_hf(path):
        return resolve_file(path)
    model_dir = Path(str(model_dir)) if model_dir is not None else None
    p = Path(os.path.expanduser(path))
    if p.is_absolute():
        if p.exists() or model_dir is None:
            return str(p)
        alt = model_dir / "tokenizer" / p.name
        return str(alt) if alt.exists() else str(p)
    return str(model_dir / p) if model_dir is not None else str(p)
