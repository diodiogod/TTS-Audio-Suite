from __future__ import annotations

"""
Stable request/response schema for isolated engine workers.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RuntimeJobRequest:
    engine_name: str
    action: str
    model_name: str
    device: str
    payload: Dict[str, Any] = field(default_factory=dict)
    runtime_profile: Optional[str] = None
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeJobResponse:
    ok: bool
    error: Optional[str] = None
    result: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
