"""TTS decode manager for handling running TTS requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from zonos3_sgl.core import TTSBatch, TTSReq


@dataclass
class TTSDecodeManager:
    """Manager for running TTS decode requests."""

    running_reqs: Set[TTSReq] = field(default_factory=set)

    def add_reqs(self, reqs: Iterable[TTSReq]) -> None:
        """Add requests that can still decode."""
        self.running_reqs.update(req for req in reqs if req.can_decode())

    def remove_req(self, req: TTSReq) -> None:
        """Remove a finished request."""
        self.running_reqs.discard(req)

    @property
    def inflight_tokens(self) -> int:
        """Total remaining tokens across all running requests."""
        return sum(req.remain_len for req in self.running_reqs)

    def schedule_next_batch(self) -> TTSBatch | None:
        """Schedule the next decode batch."""
        if not self.runnable:
            return None
        return TTSBatch(reqs=list(self.running_reqs), phase="decode")

    @property
    def runnable(self) -> bool:
        """Whether there are requests that can be scheduled."""
        return bool(self.running_reqs)
