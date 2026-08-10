"""ComfyUI lifecycle wrapper around the official DramaBox warm server."""

import gc
import importlib.util
import os
import sys
import tempfile
from typing import Any, Dict, Iterator, Optional

import torch
import torchaudio

from engines.dramabox.dramabox_downloader import DramaBoxDownloader
from utils.device import resolve_torch_device


class DramaBoxEngine:
    """Load official DramaBox inference and tear it down cleanly on VRAM clear."""

    SAMPLE_RATE = 48000

    def __init__(
        self,
        model_name: str = DramaBoxDownloader.MODEL_NAME,
        device: str = "auto",
        precision: str = "auto",
        model_paths: Optional[Dict[str, str]] = None,
        memory_mode: str = "fast",
        transformer_quantization: str = "none",
        compile_model: bool = False,
        lora_path: str = "",
        lora_strength: float = 1.0,
    ):
        self.model_name = model_name
        self.device = resolve_torch_device(device)
        self.precision = self._resolve_precision(precision)
        self.model_paths = model_paths
        self.memory_mode = str(memory_mode)
        self.transformer_quantization = str(transformer_quantization)
        self.compile_model = bool(compile_model)
        self.lora_path = str(lora_path or "").strip()
        self.lora_strength = float(lora_strength)
        self._server = None
        self._server_module = None

    def _resolve_precision(self, precision: str) -> str:
        value = str(precision or "auto").lower()
        if value in {"float16", "fp16"}:
            return "fp16"
        if value in {"bfloat16", "bf16"}:
            return "bf16"
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            return "bf16"
        return "fp16"

    @staticmethod
    def _vendor_paths():
        vendor_dir = os.path.join(os.path.dirname(__file__), "vendor")
        return (
            os.path.join(vendor_dir, "src"),
            os.path.join(vendor_dir, "ltx2"),
        )

    def _import_server(self):
        if self._server_module is not None:
            return self._server_module

        src_dir, ltx_dir = self._vendor_paths()
        for path in (src_dir, ltx_dir):
            if path not in sys.path:
                sys.path.insert(0, path)

        module_path = os.path.join(src_dir, "inference_server.py")
        spec = importlib.util.spec_from_file_location(
            "tts_audio_suite_dramabox_inference_server", module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load bundled DramaBox server: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._server_module = module
        return module

    def _ensure_runtime_loaded(self):
        if self._server is not None:
            return
        if not str(self.device).startswith("cuda"):
            raise RuntimeError(
                "DramaBox requires an NVIDIA CUDA GPU. Try the experimental "
                "staged or sequential mode with fp8_cast on lower-memory cards."
            )
        if self.model_paths is None:
            self.model_paths = DramaBoxDownloader().resolve_model_path(self.model_name)

        module = self._import_server()
        print(
            f"🔄 Loading DramaBox on {self.device} "
            f"({self.precision}, official 4-bit Gemma encoder)"
        )
        self._server = module.TTSServer(
            checkpoint=self.model_paths["transformer"],
            full_checkpoint=self.model_paths["audio_components"],
            gemma_root=self.model_paths["gemma_root"],
            device=self.device,
            dtype=self.precision,
            compile_model=self.compile_model,
            bnb_4bit=True,
            memory_mode=self.memory_mode,
            transformer_quantization=self.transformer_quantization,
            lora_path=self.lora_path,
            lora_strength=self.lora_strength,
        )
        print("✅ DramaBox runtime ready")

    @staticmethod
    def _check_interrupt():
        import comfy.model_management as model_management

        if model_management.interrupt_processing:
            raise InterruptedError("DramaBox generation interrupted by user")

    def generate(
        self,
        prompt: str,
        voice_ref_path: Optional[str] = None,
        cfg_scale: float = 2.5,
        stg_scale: float = 1.5,
        duration_multiplier: float = 1.1,
        gen_duration: float = 0.0,
        ref_duration: float = 10.0,
        rescale_scale: Any = "auto",
        watermark: bool = False,
        negative_prompt: str = "",
        seed: int = 42,
    ) -> Dict[str, Any]:
        self._ensure_runtime_loaded()
        self._check_interrupt()

        def progress_callback(_index: int, _total: int, _estimated_seconds: float):
            self._check_interrupt()

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="tts_suite_dramabox_"
        )
        temp_path = temp_file.name
        temp_file.close()
        try:
            self._server.generate_to_file(
                prompt=prompt,
                output=temp_path,
                voice_ref=voice_ref_path,
                cfg_scale=float(cfg_scale),
                stg_scale=float(stg_scale),
                duration_multiplier=float(duration_multiplier),
                gen_duration=float(gen_duration),
                ref_duration=float(ref_duration),
                rescale_scale=rescale_scale,
                negative_prompt=str(negative_prompt or ""),
                seed=int(seed),
                denoise_ref=False,
                watermark=bool(watermark),
                progress_callback=progress_callback,
            )
            waveform, sample_rate = torchaudio.load(temp_path)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        self._check_interrupt()
        return {
            "audio": waveform.detach().float().cpu(),
            "sample_rate": int(sample_rate),
        }

    def set_lora(self, lora_path: str = "", strength: float = 1.0, revision: str = ""):
        """Update the live adapter without rebuilding the base DramaBox runtime."""
        self.lora_path = str(lora_path or "").strip()
        self.lora_strength = float(strength)
        if self._server is not None:
            self._server.configure_lora(
                self.lora_path,
                self.lora_strength,
                revision=str(revision or ""),
            )

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Expose loaded submodule parameters for ComfyUI memory accounting."""
        if self._server is None:
            return
        seen = set()
        stack = list(vars(self._server).values())
        while stack:
            value = stack.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, torch.nn.Module):
                yield from value.parameters()
            elif hasattr(value, "__dict__"):
                stack.extend(vars(value).values())

    def unload_runtime(self):
        """Drop the quantized Gemma and LTX runtime instead of copying it to RAM."""
        server = self._server
        if server is not None:
            for name in (
                "_ref_denoise_cache",
                "_prompt_encoder",
                "_velocity_model",
                "_audio_conditioner",
                "_audio_decoder",
                "_ref_denoiser",
            ):
                value = getattr(server, name, None)
                if hasattr(value, "clear"):
                    try:
                        value.clear()
                    except Exception:
                        pass
                try:
                    setattr(server, name, None)
                except Exception:
                    pass
        self._server = None
        del server
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    def to(self, device):
        target = str(device) if isinstance(device, str) else str(torch.device(device))
        if target.startswith("cpu"):
            self.unload_runtime()
        self.device = target
        return self

    def unload(self):
        self.unload_runtime()
