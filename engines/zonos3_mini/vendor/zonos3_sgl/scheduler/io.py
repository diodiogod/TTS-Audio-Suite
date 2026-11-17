from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch

from zonos3_sgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg
from zonos3_sgl.utils.logger import init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        self.tp_cpu_group: Final = tp_cpu_group
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        # TTS Audio Suite Patch: only the offline mode exists in this copy (no ZeroMQ transport, no tensor-parallel ranks).
        raise NotImplementedError("zonos3_sgl in TTS Audio Suite runs offline only (offline_mode=True)")

    def run_when_idle(self):
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: BatchTokenizerMsg) -> None:
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        pass
