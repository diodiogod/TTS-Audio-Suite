"""Small PyTorch helpers shared by offline autoencoder adapters."""

from __future__ import annotations

import os
from pathlib import Path


def default_device(torch) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if torch.cuda.device_count() == 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    return f"cuda:{local_rank}"


def materialize_weight_norm(torch, model) -> None:
    """Replace both legacy and parametrized weight normalization in eval models."""

    for module in model.modules():
        try:
            if torch.nn.utils.parametrize.is_parametrized(module, "weight"):
                torch.nn.utils.parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
                continue
        except (AttributeError, ValueError):
            pass
        try:
            torch.nn.utils.remove_weight_norm(module)
        except (AttributeError, ValueError):
            pass


def compile_fixed(
    torch,
    function,
    warmup,
    *,
    enabled: bool,
    cache_name: str,
    strategy: str = "inductor",
):
    """Optimize one fixed CUDA shape with Inductor or a CUDA Graph.

    CUDA Graphs retain normal PyTorch kernels when discrete output stability is
    important. Inductor is faster for continuous encoders, where small floating
    point differences are expected and measured.
    """

    warmup_args = warmup if isinstance(warmup, tuple) else (warmup,)
    device = warmup_args[0].device
    if strategy not in {"inductor", "cuda_graph"}:
        raise ValueError(f"unknown fixed-shape strategy: {strategy}")
    if device.type != "cuda":
        with torch.inference_mode():
            function(*warmup_args)
        return None

    with torch.inference_mode():
        function(*warmup_args)
        torch.cuda.synchronize(device)
        if not enabled:
            return None

        if strategy == "inductor":
            cache_root = Path(os.environ.get("TMPDIR", "/tmp"))
            cache_dir = cache_root / f"bingua2-{cache_name}-inductor-u{os.getuid()}"
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
            lock_path = cache_root / f"{cache_dir.name}.lock"
            import fcntl  # TTS Audio Suite Patch: imported here so the module loads on Windows

            with lock_path.open("a+b") as compile_lock:
                fcntl.flock(compile_lock, fcntl.LOCK_EX)
                optimized = torch.compile(function, mode="reduce-overhead", fullgraph=True)
                optimized(*warmup_args)
                # reduce-overhead may defer its CUDA-graph recording until the
                # next invocation; complete it before processing timers start.
                optimized(*warmup_args)
            torch.cuda.synchronize(device)
            return optimized

        for _ in range(2):
            output = function(*warmup_args)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = function(*warmup_args)
        torch.cuda.synchronize(device)

    def replay(*args):
        if len(args) != len(warmup_args):
            raise ValueError("fixed CUDA graph received the wrong argument count")
        for static, current in zip(warmup_args, args):
            if static.shape != current.shape or static.dtype != current.dtype:
                raise ValueError("fixed CUDA graph received an incompatible tensor")
            static.copy_(current)
        graph.replay()
        return output

    return replay


__all__ = ["compile_fixed", "default_device", "materialize_weight_norm"]
