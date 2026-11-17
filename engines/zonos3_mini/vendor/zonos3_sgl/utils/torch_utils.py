from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def make_device_tensor(data: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """``data`` as a tensor on ``device`` without blocking the host: built in pinned memory and copied non-blocking
    (``torch.tensor(data, device=cuda)`` and pageable sources synchronize with the stream)."""
    import torch  # real import when used

    return to_device(torch.tensor(data, dtype=dtype), device)


def to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """A CPU ``tensor`` on ``device`` without blocking the host (through pinned memory when bound for CUDA)."""
    import torch  # real import when used

    if torch.device(device).type != "cuda":
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
