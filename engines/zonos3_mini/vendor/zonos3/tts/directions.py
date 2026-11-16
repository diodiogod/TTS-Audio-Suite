"""Direction sets: the vectors a steering bundle (zonos3.tts.steering) serves as controls.

A set directory holds ``manifest.json`` (``dim``, ``space`` and ``directions: {name: {"file": ..., "kind": ...}}``),
one ``.npy`` vector per direction and optionally ``calibration.json`` (per-direction doses chosen offline).

``space`` selects where a set's directions apply:

* ``"raw"`` -- the speaker-embedding space the encoder produces; the delta moves the speaker embedding.
* ``"attnres_source"`` -- the model's hidden space. The delta is added, on every generated frame, to source
  ``source_index`` of the Full AttnRes bank (0 = embedding, 2i+1 = attention/GDN output of layer i, 2i+2 = FFN/MoE
  output of layer i) before later sublayers and the head route from it.

Vectors are stored as mean-difference vectors (not unit-normalised). This module does no GPU work.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
CALIBRATION_NAME = "calibration.json"

RAW_SPACE = "raw"
HIDDEN_SPACE = "attnres_source"
SPACES = (RAW_SPACE, HIDDEN_SPACE)

# Names treated as bipolar axes rather than named directions (besides manifest entries of kind "axis").
AXIS_NAMES = ("valence", "arousal")


@dataclass
class Calibration:
    """Per-speaker, per-direction raw dose chosen offline (a set's ``calibration.json``).

    ``strength(speaker_key, name)`` falls back per direction to ``default`` and finally to ``global_default`` for
    unknown speakers/directions. A dose of 0 disables the direction for that speaker. Bipolar axes may carry a
    separate dose for the negative pole under the key ``"<name>-"``; ``"<name>-": 0`` disables only the negative pole.
    """

    by_speaker: Dict[str, Dict[str, float]] = field(default_factory=dict)
    default: Dict[str, float] = field(default_factory=dict)
    global_default: float = 1.0

    def strength(self, speaker_key: str | None, name: str) -> float:
        s = self._lookup(speaker_key, name)
        return float(self.global_default) if s is None else s

    def _lookup(self, speaker_key: str | None, key: str) -> float | None:
        if speaker_key:
            s = self.by_speaker.get(speaker_key, {}).get(key)
            if s is not None:
                return float(s)
        s = self.default.get(key)
        return None if s is None else float(s)

    def disabled(self, speaker_key: str | None, names: Iterable[str]) -> list[str]:
        """Directions whose calibrated dose is 0 for this speaker. A negative pole disabled on its own is listed as
        ``"<name>-"``."""
        out = []
        for name in names:
            if self.strength(speaker_key, name) == 0.0:
                out.append(name)
            elif self._lookup(speaker_key, name + "-") == 0.0:
                out.append(name + "-")
        return out

    def document(self) -> dict:
        """The calibration as served by ``/v2/controls`` (schema ``ControlCalibration``)."""
        return {
            "default": {k: float(v) for k, v in self.default.items()},
            "by_speaker": {s: {k: float(v) for k, v in d.items()} for s, d in self.by_speaker.items()},
            "global_default": float(self.global_default),
        }

    @classmethod
    def load(cls, directory: str | Path) -> "Calibration | None":
        path = Path(directory).expanduser() / CALIBRATION_NAME
        if not path.is_file():
            return None
        return cls.from_document(json.loads(path.read_text()))

    @classmethod
    def from_document(cls, data: dict) -> "Calibration":
        """A parsed calibration.json."""
        return cls(
            by_speaker={k: {e: float(v) for e, v in d.items()} for k, d in data.get("by_speaker", {}).items()},
            default={e: float(v) for e, v in data.get("default", {}).items()},
            global_default=float(data.get("global_default", 1.0)),
        )


@dataclass
class DirectionSet:
    """One direction set: 1-D float32 CPU vectors of ``dim`` elements, split into named directions and axes."""

    dim: int
    named: Dict[str, torch.Tensor] = field(default_factory=dict)
    axes: Dict[str, torch.Tensor] = field(default_factory=dict)
    space: str = RAW_SPACE
    source_index: int | None = None  # Full AttnRes bank source of a HIDDEN_SPACE set
    calibration: Calibration | None = None

    @property
    def names(self) -> list[str]:
        return [*self.named, *self.axes]

    def is_empty(self) -> bool:
        return not self.named and not self.axes

    @classmethod
    def load(cls, directory: str | Path) -> "DirectionSet | None":
        """Load a set from a directory containing ``manifest.json`` (and optionally ``calibration.json``).

        Returns ``None`` when the directory or manifest is missing; raises ValueError for a space not in ``SPACES``.
        """
        root = Path(directory).expanduser()
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text())
        dim = int(manifest["dim"])
        space = str(manifest.get("space", RAW_SPACE))
        if space not in SPACES:
            raise ValueError(f"direction space {space!r} is not one of {list(SPACES)} ({manifest_path})")
        named: Dict[str, torch.Tensor] = {}
        axes: Dict[str, torch.Tensor] = {}
        for name, entry in manifest.get("directions", {}).items():
            vec = _load_vector(root / entry["file"], expected_dim=dim, name=name)
            if str(entry.get("kind")) == "axis" or name in AXIS_NAMES:
                axes[name] = vec
            else:
                named[name] = vec
        source_index = None
        if space == HIDDEN_SPACE:
            source_index = int(manifest["source_index"])
            if source_index < 0:
                raise ValueError(f"source_index must be >= 0, got {source_index} ({manifest_path})")
        directions = cls(
            dim=dim, named=named, axes=axes, space=space, source_index=source_index, calibration=Calibration.load(root)
        )
        if directions.is_empty():
            logger.warning("Direction set manifest at %s contains no directions.", manifest_path)
        return directions


def _load_vector(path: Path, *, expected_dim: int, name: str) -> torch.Tensor:
    arr = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_dim:
        raise ValueError(f"Direction '{name}' has dim {arr.shape[0]}, expected {expected_dim} ({path}).")
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=torch.float32, device="cpu")
