from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from zonos3_sgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    # Stop rule: "any" codebook's EOA starts the stop countdown, or only
    # "codebook0"'s (the rule accent adapters are scored with).
    eoa_policy: str = "any"
    # LACI (zonos3.tts.laci): alignment-head spec JSON per language code ("en_us", "cmn", ...). A request with
    # TTSUserMsg.laci = one of these keys runs under LACI; it needs align_probe_max_text > 0 for graph decode.
    laci_heads: Dict[str, str] = field(default_factory=dict)

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/zonos3_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/zonos3_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/zonos3_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
