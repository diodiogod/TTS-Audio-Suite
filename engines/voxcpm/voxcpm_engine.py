"""ComfyUI lifecycle wrapper for the official VoxCPM package."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import inspect
import os
import random
import re
import sys
from typing import Any, Iterator, Optional

from engines.voxcpm.voxcpm_downloader import VoxCPMDownloader


_CURRENT_DIR = os.path.dirname(__file__)
_ENGINES_DIR = os.path.dirname(_CURRENT_DIR)
_PROJECT_ROOT = os.path.dirname(_ENGINES_DIR)


class VoxCPMEngine:
    """Thin, local-files-only wrapper around official ``voxcpm.VoxCPM``."""

    REQUIRED_RUNTIME_VERSION = "2.0.3"
    SAMPLE_RATE = 48000

    def __init__(
        self,
        model_name: str = VoxCPMDownloader.DEFAULT_MODEL,
        device: str = "auto",
        optimize: bool = False,
        model_dir: Optional[str] = None,
    ):
        self.model_name = str(model_name or VoxCPMDownloader.DEFAULT_MODEL)
        self.device = str(device or "auto")
        self.optimize = bool(optimize)
        self.downloader = VoxCPMDownloader()
        self.model_dir = os.path.abspath(model_dir) if model_dir else None
        self._model_spec = self._inspect_model_spec(self.model_dir)
        self._runtime = None
        self._runtime_was_optimized = False

    def _inspect_model_spec(
        self,
        resolved_path: Optional[str],
    ):
        expected = self.downloader.get_model_spec(
            self.model_name,
            resolved_path=resolved_path,
        )
        if resolved_path is None:
            return expected

        # A local finetune may use either supported weight serialization.
        # Preserve the expected generation metadata while validating against
        # the files the official runtime will actually open.
        local = self.downloader.get_model_spec(
            f"local:{os.path.basename(resolved_path)}",
            resolved_path=resolved_path,
        )
        local["canonical"] = expected["canonical"]
        local["repo_id"] = expected.get("repo_id")
        return local

    @staticmethod
    def _version_tuple(version: str) -> tuple[int, ...]:
        values = re.findall(r"\d+", str(version))
        return tuple(int(value) for value in values[:3])

    def _import_official_package(self):
        """Import PyPI VoxCPM without this engine folder shadowing it."""
        try:
            installed_version = importlib.metadata.version("voxcpm")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportError(
                "VoxCPM is not installed in the active ComfyUI Python "
                f"environment ({sys.executable}). Install voxcpm=="
                f"{self.REQUIRED_RUNTIME_VERSION}."
            ) from exc

        if self._version_tuple(installed_version) < self._version_tuple(
            self.REQUIRED_RUNTIME_VERSION
        ):
            raise ImportError(
                f"VoxCPM {installed_version} is too old. Install voxcpm=="
                f"{self.REQUIRED_RUNTIME_VERSION} in {sys.executable}."
            )

        importlib.invalidate_caches()
        nodes_dir = os.path.join(_PROJECT_ROOT, "nodes")
        blocked_paths = {
            os.path.abspath(_ENGINES_DIR),
            os.path.abspath(nodes_dir),
        }
        original_sys_path = list(sys.path)
        sys.path[:] = [
            path
            for path in sys.path
            if os.path.abspath(path or os.getcwd()) not in blocked_paths
        ]

        stale_modules = []
        for module_name, module in list(sys.modules.items()):
            if not (
                module_name == "voxcpm"
                or module_name.startswith("voxcpm.")
            ):
                continue
            module_file = getattr(module, "__file__", "") or ""
            if module_file and os.path.abspath(module_file).startswith(
                os.path.abspath(_PROJECT_ROOT)
            ):
                stale_modules.append((module_name, module))
                del sys.modules[module_name]

        try:
            official_module = importlib.import_module("voxcpm")
            VoxCPM = official_module.VoxCPM
        except Exception as exc:
            for module_name, module in stale_modules:
                sys.modules.setdefault(module_name, module)
            raise ImportError(
                "Failed to import the official `voxcpm` package in the active "
                "ComfyUI Python environment. This is usually a missing "
                "dependency or an import-path collision. "
                f"Active Python: {sys.executable}"
            ) from exc
        finally:
            sys.path[:] = original_sys_path

        return VoxCPM

    def _ensure_runtime_loaded(self) -> None:
        if self._runtime is not None:
            return

        if self.model_dir is None:
            self.model_dir = self.downloader.resolve_model_path(self.model_name)
        self.model_dir = os.path.abspath(self.model_dir)
        self._model_spec = self._inspect_model_spec(self.model_dir)
        if not self.downloader.is_model_complete(
            self.model_dir,
            self._model_spec,
        ):
            raise RuntimeError(
                f"Refusing to load incomplete VoxCPM model: {self.model_dir}"
            )

        from utils.device import resolve_torch_device

        resolved_device = resolve_torch_device(self.device)
        effective_optimize = self.optimize and str(resolved_device).startswith(
            "cuda"
        )
        if self.optimize and not effective_optimize:
            print(
                "⚠️ VoxCPM torch.compile optimization is CUDA-only; "
                f"loading without it on {resolved_device}"
            )

        VoxCPM = self._import_official_package()
        print(f"🔄 Loading {self._model_spec['canonical']} via official VoxCPM")
        print(f"   Path: {self.model_dir}")
        print(
            f"   Architecture: {self.architecture} | "
            f"Device: {resolved_device} | Optimize: {effective_optimize}"
        )

        # Passing an already-validated absolute directory plus local_files_only
        # guarantees the official wrapper cannot create a second HF cache copy.
        runtime = VoxCPM.from_pretrained(
            hf_model_id=self.model_dir,
            load_denoiser=False,
            local_files_only=True,
            optimize=effective_optimize,
            device=resolved_device,
        )
        self._runtime = runtime
        self._runtime_was_optimized = effective_optimize
        self.device = str(resolved_device)
        print("✅ VoxCPM runtime ready")

    @property
    def architecture(self) -> str:
        return str(self._model_spec["architecture"])

    @property
    def sample_rate(self) -> int:
        if self._runtime is not None:
            tts_model = getattr(self._runtime, "tts_model", None)
            runtime_rate = getattr(tts_model, "sample_rate", None)
            if runtime_rate is not None:
                return int(runtime_rate)
        return int(self._model_spec["sample_rate"])

    @staticmethod
    def _seed_global_rngs(seed: int) -> None:
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _runtime_accepts_seed(runtime: Any) -> bool:
        # VoxCPM.generate itself is a *args/**kwargs forwarding method. Inspect
        # the concrete implementation so PyPI 2.0.3 is not sent an unsupported
        # seed keyword, while newer official releases can receive it directly.
        target = getattr(runtime, "_generate", None)
        if target is None:
            target = getattr(runtime, "generate")
        try:
            return "seed" in inspect.signature(target).parameters
        except (TypeError, ValueError):
            return False

    def generate(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        reference_wav_path: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        normalize: bool = False,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        seed: Optional[int] = None,
    ):
        """Call the official non-streaming API and return mono ``[1, T]`` audio."""
        self._ensure_runtime_loaded()

        kwargs = {
            "text": text,
            "prompt_wav_path": prompt_wav_path or None,
            "prompt_text": prompt_text if prompt_text is not None else None,
            "reference_wav_path": reference_wav_path or None,
            "cfg_value": float(cfg_value),
            "inference_timesteps": int(inference_timesteps),
            "min_len": int(min_len),
            "max_len": int(max_len),
            "normalize": bool(normalize),
            "denoise": False,
            "retry_badcase": bool(retry_badcase),
            "retry_badcase_max_times": int(retry_badcase_max_times),
            "retry_badcase_ratio_threshold": float(
                retry_badcase_ratio_threshold
            ),
        }

        if seed is not None:
            seed_value = int(seed)
            if self._runtime_accepts_seed(self._runtime):
                kwargs["seed"] = seed_value
            else:
                self._seed_global_rngs(seed_value)

        waveform = self._runtime.generate(**kwargs)

        import torch

        if isinstance(waveform, torch.Tensor):
            audio = waveform.detach().float().cpu()
        else:
            audio = torch.as_tensor(waveform, dtype=torch.float32).cpu()
        return audio.reshape(1, -1)

    def parameters(self) -> Iterator[Any]:
        """Expose official model parameters for ComfyUI memory accounting."""
        if self._runtime is None:
            return
        tts_model = getattr(self._runtime, "tts_model", None)
        if tts_model is not None and hasattr(tts_model, "parameters"):
            yield from tts_model.parameters()

    @staticmethod
    def _clear_model_caches(tts_model: Any) -> None:
        if tts_model is None:
            return
        for module_name in ("base_lm", "residual_lm"):
            module = getattr(tts_model, module_name, None)
            cache = getattr(module, "kv_cache", None)
            if hasattr(cache, "clear"):
                try:
                    cache.clear()
                except Exception:
                    pass
            if module is not None:
                try:
                    module.kv_cache = None
                except Exception:
                    pass

    def unload_runtime(self) -> None:
        """Destroy the official runtime instead of copying it into system RAM."""
        runtime = self._runtime
        self._runtime = None
        was_optimized = self._runtime_was_optimized
        self._runtime_was_optimized = False

        if runtime is not None:
            tts_model = getattr(runtime, "tts_model", None)
            self._clear_model_caches(tts_model)
            for attr_name in ("tts_model", "text_normalizer", "denoiser"):
                try:
                    setattr(runtime, attr_name, None)
                except Exception:
                    pass
            del tts_model
            del runtime

        torch = sys.modules.get("torch")
        if was_optimized and torch is not None:
            try:
                import torch._dynamo as torch_dynamo

                torch_dynamo.reset()
            except Exception:
                pass

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    def to(self, device):
        """ComfyUI offload hook; the next generation reloads on demand."""
        target = str(device)
        if self._runtime is not None and (
            target.startswith("cpu") or target != self.device
        ):
            self.unload_runtime()
        self.device = target
        return self

    def unload(self) -> None:
        self.unload_runtime()
