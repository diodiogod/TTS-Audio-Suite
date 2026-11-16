"""Extra text rows (lingua ``apps.zonos.text_extra``, ``zonos-text-extra/1``): phoneme, start and pause tokens.

The rows were trained on a frozen base (``text_extra_only``), so a model gains them by appending them to its text
table, after the padding row and after any extra rows it already has (a speech-edit vocabulary's marker rows stay at
``text_vocab + 1 + j``). Text that never reads them runs exactly as before. The rows are the phoneme rows, then the
start row, then the pause rows (each block optional); they add ``text_tokenizer_spec.phonemes`` (group markers, ids,
symbols), ``start`` (id, marker) and ``pauses`` (ids, markers, seconds) with ids shifted by the rows already there.

``tools/add_text_extra_rows.py`` applies them to a checkpoint on disk; the servers and the reference apply them to a
loaded model (``--phoneme-rows``; the Space and the release defaults with the release rows).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

FORMAT = "zonos-text-extra/1"
_PHONEME_KEYS = ("open_id", "close_id", "symbols", "open_marker", "close_marker", "inventory_sha256")


@dataclass(frozen=True)
class TextExtraRows:
    """One parsed export: float32 ``rows`` [n, dim] trained after a ``text_vocab`` + padding table, and the phoneme,
    start and pause blocks they encode (ids relative to a table without other extra rows), if any."""

    rows: torch.Tensor
    text_vocab: int
    phonemes: dict | None
    start: dict | None
    pauses: dict | None
    run: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "TextExtraRows":
        from zonos3.tts.superbpe import START_MARKER
        from zonos3.utils.release import resolve_file

        path = resolve_file(str(path))
        extra = torch.load(path, map_location="cpu", weights_only=True)
        if extra.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} export")
        text_vocab = int(extra["text_vocab"])
        first = text_vocab + 1
        phonemes = start = pauses = None
        if "phonemes" in extra:
            phonemes = {k: extra["phonemes"][k] for k in _PHONEME_KEYS}
            if (phonemes["open_id"], phonemes["close_id"]) != (first, first + 1):
                raise ValueError(f"{path}: exported phoneme rows must start at the padding row + 1")
        n_ph = 0 if phonemes is None else 2 + len(phonemes["symbols"])
        if "start" in extra:
            start = {"id": int(extra["start"]["id"]), "marker": extra["start"]["marker"]}
            if start != {"id": first + n_ph, "marker": START_MARKER}:
                raise ValueError(f"{path}: the start row must follow the phoneme rows")
        if "pauses" in extra:
            p = extra["pauses"]
            pauses = {"ids": [int(i) for i in p["ids"]], "markers": list(p["markers"]), "seconds": list(p["seconds"])}
            lo, n = first + n_ph + int(start is not None), len(pauses["ids"])
            if (
                n == 0
                or pauses["ids"] != list(range(lo, lo + n))
                or pauses["markers"] != [chr(0xE003 + j) for j in range(n)]
                or len(pauses["seconds"]) != n
            ):
                raise ValueError(f"{path}: the pause rows must follow the start row")
        n_pauses = 0 if pauses is None else len(pauses["ids"])
        rows = int(extra["rows"].shape[0])
        if (phonemes, start, pauses) != (None, None, None) and n_ph + int(start is not None) + n_pauses != rows:
            raise ValueError(
                f"{path}: {rows} rows for {n_ph} phoneme, {int(start is not None)} start and {n_pauses} pause rows"
            )
        return cls(extra["rows"].float(), text_vocab, phonemes, start, pauses, extra.get("run"))

    @property
    def count(self) -> int:
        return int(self.rows.shape[0])

    def spec(self, spec: dict | None, *, text_vocab: int, existing: int) -> dict | None:
        """``spec`` (a ``text_tokenizer_spec``, or None) of a model with ``existing`` extra rows once these rows follow
        them: they add ``phonemes`` / ``start`` / ``pauses`` with their ids shifted by ``existing``. Raises ValueError
        for a model the rows do not fit."""
        if text_vocab != self.text_vocab:
            raise ValueError(f"text_vocab {text_vocab} != the rows' {self.text_vocab}")
        blocks = {"phonemes": self.phonemes, "start": self.start, "pauses": self.pauses}
        if all(block is None for block in blocks.values()):
            return spec
        if spec is None:
            raise ValueError("phoneme, start and pause rows need a SuperBPE text_tokenizer_spec")
        for key in blocks:
            if spec.get(key) is not None:
                raise ValueError(f"the model already has text_tokenizer_spec.{key}")
        out = {**spec}
        if self.phonemes is not None:
            out["phonemes"] = dict(
                self.phonemes,
                open_id=self.phonemes["open_id"] + existing,
                close_id=self.phonemes["close_id"] + existing,
            )
        if self.start is not None:
            out["start"] = dict(self.start, id=self.start["id"] + existing)
        if self.pauses is not None:
            out["pauses"] = dict(self.pauses, ids=[i + existing for i in self.pauses["ids"]])
        return out

    def append(self, table: torch.Tensor) -> torch.Tensor:
        """``table`` (the text embedder weight) with these rows after it, in its dtype and device."""
        if table.shape[1] != self.rows.shape[1]:
            raise ValueError(f"rows of width {self.rows.shape[1]} for a text table of width {table.shape[1]}")
        return torch.cat((table, self.rows.to(table)), dim=0).contiguous()
