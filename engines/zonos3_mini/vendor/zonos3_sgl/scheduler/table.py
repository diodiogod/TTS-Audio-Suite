"""TTS table manager for 2D token pool."""

from __future__ import annotations

import torch


class TTSTableManager:
    """Table manager for TTS with 3D token pool (max_reqs, max_seq_len, frame_width)."""

    def __init__(
        self,
        max_running_reqs: int,
        page_table: torch.Tensor,
        frame_width: int,
    ) -> None:
        """Initialize the TTS table manager.

        Args:
            max_running_reqs: Maximum number of concurrent requests
            page_table: Page table tensor for KV cache management
            frame_width: Number of elements per frame (n_codebooks + extras)
        """
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        self.frame_width = frame_width

        # 3D token pool: (max_reqs, max_seq_len, frame_width)
        # Initialized with zeros (valid padding)
        self.token_pool = torch.zeros(
            (page_table.shape[0], page_table.shape[1], frame_width),
            dtype=torch.int32,
            device=page_table.device,
        )

    @property
    def available_size(self) -> int:
        """Number of available request slots."""
        return len(self._free_slots)

    def allocate(self) -> int:
        """Allocate a table slot for a new request."""
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        """Free a table slot."""
        self._free_slots.append(slot)
