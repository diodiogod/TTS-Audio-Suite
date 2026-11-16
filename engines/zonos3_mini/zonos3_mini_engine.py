"""
ZONOS3-Mini engine wrapper.

Runs the vendored official plain-PyTorch reference implementation (`zonos3.reference`) on the native
Zyphra/ZONOS3-Mini weights, with ComfyUI-friendly loading, offloading, progress and interruption.
"""

import gc
import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.zonos3_mini import ASSETS_DIR, ensure_vendor_path
from engines.zonos3_mini.voices import PRESETS
from engines.zonos3_mini.zonos3_mini_downloader import Zonos3MiniDownloader
from utils.device import resolve_torch_device

LM_FOLDER = os.path.join("lm", "zonos3_mini")
TEXT_ROWS = os.path.join(LM_FOLDER, "text_rows", "pause_tokens_s5000.pt")
FRAME_RATE = 15


class Zonos3MiniProgress:
    """Print generated frames, frames per second and elapsed time while the LM runs."""

    def __init__(self, calls_per_frame: int):
        self.calls_per_frame = max(1, int(calls_per_frame))
        self.calls = 0
        self.start_time = time.time()
        self.last_print_time = self.start_time
        self.last_frame = 0
        self.rendered = False

    @property
    def frames(self) -> int:
        return self.calls // self.calls_per_frame

    def step(self):
        self.calls += 1
        now = time.time()
        if now - self.last_print_time < 0.5:
            return
        frames = self.frames
        rate = (frames - self.last_frame) / max(now - self.last_print_time, 1e-6)
        print(
            f"\r   Progress: {frames} frames ({frames / FRAME_RATE:.1f}s audio) | {rate:.1f} frames/s | "
            f"{now - self.start_time:.0f}s      ",
            end="",
            flush=True,
        )
        self.last_print_time = now
        self.last_frame = frames
        self.rendered = True

    def end(self):
        total = max(time.time() - self.start_time, 1e-6)
        if self.rendered:
            print(
                f"\r   Complete: {self.frames} frames in {total:.1f}s (avg {self.frames / total:.1f} frames/s)"
                + " " * 30
            )

    def abort(self):
        if self.rendered:
            print()


class Zonos3MiniEngine:
    """Thin wrapper around `zonos3.reference.synth.synthesize`."""

    SAMPLE_RATE = 44100

    def __init__(
        self,
        model_name: str = "ZONOS3-Mini",
        device: str = "auto",
        dtype: str = "auto",
        model_dir: Optional[str] = None,
    ):
        self.model_name = model_name or "ZONOS3-Mini"
        resolved = str(resolve_torch_device(device))
        if not (resolved.startswith("cuda") or resolved == "cpu"):
            print(f"⚠️ ZONOS3-Mini supports CUDA and CPU only; '{resolved}' was requested, using CPU")
            resolved = "cpu"
        self._target_device = resolved
        self.device = resolved
        self.dtype = self._resolve_dtype(dtype, resolved)
        self.model_dir = model_dir
        self.downloader = Zonos3MiniDownloader()
        self.root: Optional[str] = None
        self.lm = None
        self.assets = None
        self._decoders_ready = set()

    @staticmethod
    def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
        normalized = str(dtype or "auto").lower()
        if normalized == "bfloat16":
            return torch.bfloat16
        if normalized == "float32":
            return torch.float32
        if normalized != "auto":
            raise ValueError(f"ZONOS3-Mini dtype must be auto, bfloat16 or float32, got '{dtype}'")
        return torch.bfloat16 if device.startswith("cuda") else torch.float32

    # ------------------------------------------------------------------ loading

    def _ensure_files(self, features) -> str:
        if self.model_dir:
            absent = self.downloader.missing(self.model_dir, features)
            if absent:
                raise FileNotFoundError(f"ZONOS3-Mini model folder {self.model_dir} is missing: {', '.join(absent)}")
            return self.model_dir
        return self.downloader.ensure(self.model_name, features)

    def _ensure_model_loaded(self, decoder: str = "keep2_plain"):
        decoder = decoder or "keep2_plain"
        if self.lm is not None:
            if decoder not in self._decoders_ready:
                self._ensure_files([decoder])
                self._decoders_ready.add(decoder)
            return

        self.root = self._ensure_files(["base", decoder])
        self._decoders_ready.add(decoder)
        ensure_vendor_path()
        try:
            from zonos3.models.text_extra import TextExtraRows
            from zonos3.reference.lm import Zonos3LM
            from zonos3.reference.synth import ReferenceAssets
            from zonos3.tts.laci import discover_laci_heads
            from zonos3.utils import release
        except ModuleNotFoundError as e:
            if e.name and e.name.split(".")[0] in ("dac", "audiotools"):
                raise ImportError(
                    "ZONOS3-Mini needs descript-audio-codec and descript-audiotools (run the suite install.py)"
                ) from e
            raise

        print("🔄 Loading ZONOS3-Mini (official reference implementation)")
        print(f"   Path: {self.root}")
        print(f"   Device: {self._target_device} | Dtype: {self.dtype}")
        release.set_local_root(self.root)
        lm = Zonos3LM.load(os.path.join(self.root, LM_FOLDER), device=self._target_device, dtype=torch.float32)
        lm.append_text_rows(TextExtraRows.load(os.path.join(self.root, TEXT_ROWS)))
        if self.dtype != torch.float32:
            lm.cast(self.dtype)
        lm.eval()
        self.assets = ReferenceAssets.load(
            steer_bundle=os.path.join(ASSETS_DIR, "steering", "studio_bundle_mect60k_x10.json"),
            laci_heads=discover_laci_heads(os.path.join(ASSETS_DIR, "laci_heads")),
        )
        self.lm = lm
        self.device = self._target_device
        print("✅ ZONOS3-Mini model loaded successfully")

    def _restore_device(self):
        if self.device != self._target_device:
            self.to(self._target_device)

    # ------------------------------------------------------------------ voices

    def embed_speaker(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Reference clip ``[channels, samples]`` -> the 192-D speaker embedding (CPU float32)."""
        self._ensure_model_loaded()
        self._restore_device()
        config = self.lm.config
        encoder = self.assets.speaker_encoder(
            config.speaker_embedding_codec, config.speaker_embedding_dim, torch.device(self.device)
        )
        with torch.inference_mode():
            output = encoder(waveform.detach().float(), int(sample_rate))
        for candidate in output if isinstance(output, tuple) else (output,):
            candidate = candidate.squeeze(0).to(dtype=torch.float32, device="cpu")
            if candidate.numel() == config.speaker_embedding_dim:
                return candidate.clone()
        raise ValueError(
            f"ZONOS3-Mini speaker encoder did not return a {config.speaker_embedding_dim}-dimensional embedding"
        )

    def encode_reference(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Reference clip ``[channels, samples]`` -> codec codes int32 ``[frames, 10]`` (CUDA only)."""
        self._ensure_model_loaded()
        self._restore_device()
        if not str(self.device).startswith("cuda"):
            raise ValueError(
                "ZONOS3-Mini reference voice mode needs a CUDA device; use voice_mode 'embedding' on CPU"
            )
        self._ensure_files(["teacher"])
        from zonos3.tokenizer.dual_reference import encode_dual_reference

        device = torch.device(self.device)
        index = device.index if device.index is not None else torch.cuda.current_device()
        # the upstream codec adapter sets these process-wide; put them back afterwards
        threads = torch.get_num_threads()
        tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        tf32_cudnn = torch.backends.cudnn.allow_tf32
        try:
            return encode_dual_reference(
                waveform.detach().float().cpu(),
                int(sample_rate),
                max_seconds=self.lm.config.speaker_reference_max_seconds,
                device=f"cuda:{index}",
            )
        finally:
            torch.set_num_threads(threads)
            torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
            torch.backends.cudnn.allow_tf32 = tf32_cudnn

    @staticmethod
    def preset(name: str) -> Dict[str, np.ndarray]:
        if name not in PRESETS:
            raise ValueError(f"ZONOS3-Mini has no voice preset '{name}'")
        stem = os.path.join(ASSETS_DIR, "default_voices", *PRESETS[name].split("/"))
        return {
            "codes": np.load(stem + ".codes.npy", allow_pickle=False),
            "embedding": np.load(stem + ".embedding.npy", allow_pickle=False),
        }

    # ------------------------------------------------------------------ generation

    def generate(self, request: dict) -> Tuple[torch.Tensor, Optional[dict]]:
        """One native GenerateRequest -> (CPU float32 ``[1, samples]`` at 44.1 kHz, LACI record or None)."""
        self._ensure_model_loaded(request.get("decoder") or "keep2_plain")
        self._restore_device()
        from zonos3.reference.synth import synthesize

        import comfy.model_management as model_management

        guided = (
            float(request.get("speaker_cfg_scale", 1.0)) != 1.0
            or float(request.get("controls_cfg_scale", 1.0)) != 1.0
        )
        progress = Zonos3MiniProgress(2 if guided else 1)

        def before_forward(_module, _args):
            if model_management.interrupt_processing:
                raise InterruptedError("ZONOS3-Mini generation interrupted by user")
            progress.step()

        handle = self.lm.register_forward_pre_hook(before_forward)
        try:
            result = synthesize(
                self.lm,
                {**request, "stream": False, "stream_decode": "whole"},
                assets=self.assets,
                device=self.device,
            )
        except BaseException:
            progress.abort()
            raise
        finally:
            handle.remove()
        progress.end()

        pcm = np.asarray(result.pcm, dtype=np.float32).reshape(-1)
        if pcm.size == 0 or not np.isfinite(pcm).all():
            raise RuntimeError(f"ZONOS3-Mini returned no audio for: '{str(request.get('text', ''))[:40]}'")
        return torch.from_numpy(pcm.copy())[None], result.laci

    # ------------------------------------------------------------------ device management

    def to(self, device):
        device = str(device)
        self.device = device
        if self.lm is None:
            return self
        self.lm.to(device)
        if device == "cpu":
            # GPU-owning helper caches of the reference implementation; they reload lazily on the next request
            from zonos3.models.mect_speaker import MECTFbankSpeakerEmbedding
            from zonos3.tokenizer import dual_reference, matched_codec

            self.assets._decoders.clear()
            MECTFbankSpeakerEmbedding._get_resampler.cache_clear()
            self.assets._speaker_encoders.clear()
            matched_codec._GRAPHS.clear()
            matched_codec._CACHE.clear()
            matched_codec._INFO.clear()
            dual_reference._ENCODERS.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return self
