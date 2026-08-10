#!/usr/bin/env python3
"""Warm validation runner — loads base dev + LoRA + all aux models ONCE,
then iterates every speaker in val_config generating each output.

Matches the same generation path as inference.py but keeps Gemma / audio VAE
/ velocity model / audio decoder resident across entries. Inference
settings default to the Gradio warm-server values (cfg=2.5, stg=1.5,
modality=1.0, rescale=0, 30 steps, fps=25) — use --inference-params to
override.
"""
import argparse
import logging
import os
import sys
import time
import traceback

import torch
import torchaudio

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = REPO_DIR
sys.path.insert(0, os.path.join(REPO_DIR, "ltx2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV_FULL_CKPT = os.environ.get(
    "LTX_FULL_CHECKPOINT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ltx-2.3-22b-dev.safetensors"),
)
# TTS Audio Suite patch: organized DramaBox installs keep the transformer and
# audio components in separate checkpoints, unlike the upstream full checkpoint.
DRAMABOX_TRANSFORMER_CKPT = os.environ.get("LTX_CHECKPOINT", DEV_FULL_CKPT)
GEMMA_ROOT = os.environ.get(
    "GEMMA_ROOT",
    os.path.expanduser("~/.cache/dramabox/gemma-3-12b-it-bnb-4bit"),
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val-config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--lora-rank", type=int, default=128)
    p.add_argument("--checkpoint", default=DRAMABOX_TRANSFORMER_CKPT)
    p.add_argument("--full-checkpoint", default=DEV_FULL_CKPT)
    p.add_argument("--gemma-root", default=GEMMA_ROOT)
    p.add_argument("--cfg-scale", type=float, default=2.5)
    p.add_argument("--stg-scale", type=float, default=1.5)
    p.add_argument("--rescale-scale", type=float, default=0.0)
    p.add_argument("--modality-scale", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--stg-block", type=int, default=29)
    p.add_argument("--cfg-clamp", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duration-multiplier", type=float, default=1.1)
    # Match Gradio / inference_server.py DEFAULT_NEG exactly
    p.add_argument("--negative-prompt", default=(
        "worst quality, inconsistent, robotic, distorted, noise, static, "
        "muffled, unclear, unnatural, monotone"
    ))
    return p.parse_args()


def estimate_speech_duration(prompt: str, speed: float = 1.0) -> float:
    import re
    quoted = re.findall(r'"([^"]*)"', prompt) or re.findall(r"'([^']*)'", prompt)
    text = " ".join(quoted) if quoted else prompt
    duration = len(text) * 0.065 / max(speed, 0.1) + 1.5
    return max(3.0, round(duration, 1))


class WarmValidator:
    def __init__(self, checkpoint, full_checkpoint, gemma_root, lora_path=None, lora_rank=128,
                 device="cuda", dtype=torch.bfloat16):
        from audio_conditioning import AudioConditionByReferenceLatent  # noqa: F401 (imported by inference.py)
        from ltx_core.components.patchifiers import AudioPatchifier
        from ltx_pipelines.utils.blocks import PromptEncoder, AudioConditioner, AudioDecoder

        self.device = torch.device(device)
        self.dtype = dtype
        self.full_checkpoint = full_checkpoint
        self.gemma_root = gemma_root
        self.patchifier = AudioPatchifier(patch_size=1)

        logging.info("Loading PromptEncoder (Gemma + embeddings_processor)...")
        t0 = time.time()
        self.prompt_encoder = PromptEncoder(
            checkpoint_path=full_checkpoint, gemma_root=gemma_root,
            dtype=dtype, device=self.device, warm=True, audio_only=True,
        )
        logging.info(f"  PromptEncoder ready in {time.time()-t0:.1f}s")

        logging.info("Loading AudioConditioner (audio VAE encoder)...")
        t0 = time.time()
        self.audio_conditioner = AudioConditioner(
            checkpoint_path=full_checkpoint, dtype=dtype, device=self.device, warm=True,
        )
        logging.info(f"  AudioConditioner ready in {time.time()-t0:.1f}s")

        logging.info("Loading AudioDecoder...")
        t0 = time.time()
        self.audio_decoder = AudioDecoder(
            checkpoint_path=full_checkpoint, dtype=dtype, device=self.device, warm=True,
        )
        logging.info(f"  AudioDecoder ready in {time.time()-t0:.1f}s")

        logging.info("Building velocity model (audio-only from base dev)...")
        t0 = time.time()
        # TTS Audio Suite patch: build the DiT from the dedicated transformer
        # checkpoint while keeping the audio connector/VAE/decoder checkpoint separate.
        self.velocity_model = self._build_velocity_model(checkpoint, lora_path, lora_rank)
        logging.info(f"  Velocity model ready in {time.time()-t0:.1f}s "
                     f"({sum(p.numel() for p in self.velocity_model.parameters()) / 1e9:.1f}B params)")

    def _build_velocity_model(self, checkpoint_path, lora_path, lora_rank):
        from ltx_core.loader.registry import DummyRegistry
        from ltx_core.loader.sd_ops import SDOps
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
        from ltx_core.model.model_protocol import ModelConfigurator
        from ltx_core.model.transformer.attention import AttentionFunction
        from ltx_core.model.transformer.model import LTXModel, LTXModelType
        from ltx_core.model.transformer.rope import LTXRopeType

        sd_ops = (
            SDOps("AO")
            .with_matching(prefix="model.diffusion_model.")
            .with_replacement("model.diffusion_model.", "")
        )

        class Cfg(ModelConfigurator[LTXModel]):
            @classmethod
            def from_config(cls, config):
                t = config.get("transformer", {})
                cp = None
                if not t.get("caption_proj_before_connector", False):
                    from ltx_core.model.transformer.text_projection import create_caption_projection
                    with torch.device("meta"):
                        cp = create_caption_projection(t, audio=True)
                return LTXModel(
                    model_type=LTXModelType.AudioOnly,
                    audio_num_attention_heads=t.get("audio_num_attention_heads", 32),
                    audio_attention_head_dim=t.get("audio_attention_head_dim", 64),
                    audio_in_channels=t.get("audio_in_channels", 128),
                    audio_out_channels=t.get("audio_out_channels", 128),
                    num_layers=t.get("num_layers", 48),
                    audio_cross_attention_dim=t.get("audio_cross_attention_dim", 2048),
                    norm_eps=t.get("norm_eps", 1e-6),
                    attention_type=AttentionFunction(t.get("attention_type", "default")),
                    positional_embedding_theta=10000.0,
                    audio_positional_embedding_max_pos=[20.0],
                    timestep_scale_multiplier=t.get("timestep_scale_multiplier", 1000),
                    use_middle_indices_grid=t.get("use_middle_indices_grid", True),
                    rope_type=LTXRopeType(t.get("rope_type", "interleaved")),
                    double_precision_rope=t.get("frequencies_precision", False) == "float64",
                    apply_gated_attention=t.get("apply_gated_attention", False),
                    audio_caption_projection=cp,
                    cross_attention_adaln=t.get("cross_attention_adaln", False),
                )

        builder = Builder(
            model_path=checkpoint_path, model_class_configurator=Cfg,
            model_sd_ops=sd_ops, registry=DummyRegistry(),
        )
        velocity = builder.build(device=self.device, dtype=self.dtype).to(self.device).eval()

        if lora_path and os.path.exists(lora_path):
            from peft import LoraConfig, get_peft_model
            from safetensors.torch import load_file as st_load
            logging.info(f"Attaching LoRA: {lora_path}")
            lora_sd = st_load(lora_path)
            is_peft = any("base_model.model." in k for k in lora_sd.keys())
            is_iclora = any("diffusion_model." in k for k in lora_sd.keys())
            cfg = LoraConfig(
                r=lora_rank, lora_alpha=lora_rank, lora_dropout=0.0, bias="none",
                target_modules=[
                    "audio_attn1.to_k", "audio_attn1.to_q",
                    "audio_attn1.to_v", "audio_attn1.to_out.0",
                    "audio_attn2.to_k", "audio_attn2.to_q",
                    "audio_attn2.to_v", "audio_attn2.to_out.0",
                    "audio_ff.net.0.proj", "audio_ff.net.2",
                ],
            )
            velocity = get_peft_model(velocity, cfg)

            if is_peft:
                mapped = {}
                for k, v in lora_sd.items():
                    nk = k
                    if ".lora_A.weight" in k and ".lora_A.default.weight" not in k:
                        nk = k.replace(".lora_A.weight", ".lora_A.default.weight")
                    if ".lora_B.weight" in k and ".lora_B.default.weight" not in k:
                        nk = k.replace(".lora_B.weight", ".lora_B.default.weight")
                    mapped[nk] = v
                _, unexpected = velocity.load_state_dict(mapped, strict=False)
                logging.info(f"  Loaded {len(mapped) - len(unexpected)} LoRA weights (peft)")
            elif is_iclora:
                audio_keys = {k: v for k, v in lora_sd.items()
                              if "audio_attn1" in k or "audio_attn2" in k or "audio_ff" in k}
                mapped = {}
                for k, v in audio_keys.items():
                    nk = k.replace("diffusion_model.", "base_model.model.")
                    nk = nk.replace(".lora_A.weight", ".lora_A.default.weight")
                    nk = nk.replace(".lora_B.weight", ".lora_B.default.weight")
                    mapped[nk] = v
                _, unexpected = velocity.load_state_dict(mapped, strict=False)
                logging.info(f"  Loaded {len(mapped) - len(unexpected)} LoRA weights (iclora)")

            velocity = velocity.merge_and_unload()
            logging.info("  Merged LoRA into base weights")

        return velocity

    @torch.inference_mode()
    def generate(self, prompt, output_path, voice_ref=None, args=None):
        from audio_conditioning import AudioConditionByReferenceLatent
        from ltx_core.batch_split import BatchSplitAdapter
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.schedulers import LTX2Scheduler
        from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
        from ltx_core.model.transformer.model import X0Model
        from ltx_core.tools import AudioLatentTools
        from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
        from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
        from ltx_pipelines.utils.gpu_model import gpu_model
        from ltx_pipelines.utils.media_io import decode_audio_from_file
        from ltx_pipelines.utils.samplers import euler_denoising_loop

        t_total = time.time()

        # ---- Duration + shape ----
        gen_dur = estimate_speech_duration(prompt) * args.duration_multiplier
        raw_frames = int(round(gen_dur * args.fps)) + 1
        num_frames = ((raw_frames - 1 + 4) // 8) * 8 + 1
        pixel_shape = VideoPixelShape(batch=1, frames=num_frames, height=64, width=64, fps=args.fps)
        tgt_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        audio_tools = AudioLatentTools(patchifier=self.patchifier, target_shape=tgt_shape)

        state = audio_tools.create_initial_state(self.device, self.dtype)

        # ---- Voice reference ----
        if voice_ref and os.path.exists(voice_ref):
            voice = decode_audio_from_file(voice_ref, self.device, 0.0, 10.0)
            if voice is not None:
                w = voice.waveform
                if w.dim() == 2:
                    if w.shape[0] == 1:
                        w = w.repeat(2, 1)
                    w = w.unsqueeze(0)
                elif w.dim() == 3 and w.shape[1] == 1:
                    w = w.repeat(1, 2, 1)
                target_samples = int(10.0 * voice.sampling_rate)
                if w.shape[-1] < target_samples:
                    w = w.repeat(1, 1, (target_samples // w.shape[-1]) + 1)
                w = w[..., :target_samples]
                peak = w.abs().max()
                if peak > 0:
                    w = w * (10 ** (-4.0 / 20) / peak)
                voice = Audio(waveform=w, sampling_rate=voice.sampling_rate)
                ref_latent = self.audio_conditioner(lambda enc: vae_encode_audio(voice, enc, None))
                cond = AudioConditionByReferenceLatent(
                    latent=ref_latent.to(self.device, self.dtype), strength=1.0,
                )
                state = cond.apply_to(latent_state=state, latent_tools=audio_tools)

        # ---- Noise ----
        gen = torch.Generator(device=self.device).manual_seed(args.seed)
        noiser = GaussianNoiser(generator=gen)
        state = noiser(state, noise_scale=1.0)

        # ---- Prompt encode ----
        use_cfg = args.cfg_scale > 1.0
        prompts = [prompt, args.negative_prompt] if use_cfg else [prompt]
        ctx = self.prompt_encoder(prompts, streaming_prefetch_count=None)
        a_ctx = ctx[0].audio_encoding
        a_ctx_neg = ctx[1].audio_encoding if use_cfg else None

        # ---- Denoiser ----
        needs_guidance = args.cfg_scale > 1.0 or args.stg_scale > 0.0 or args.modality_scale > 1.0
        if needs_guidance:
            guider = MultiModalGuider(
                params=MultiModalGuiderParams(
                    cfg_scale=args.cfg_scale, stg_scale=args.stg_scale,
                    stg_blocks=[args.stg_block] if args.stg_scale > 0 else [],
                    rescale_scale=args.rescale_scale,
                    modality_scale=args.modality_scale,
                    cfg_clamp_scale=args.cfg_clamp,
                ),
                negative_context=a_ctx_neg,
            )
            denoiser = GuidedDenoiser(
                v_context=None, a_context=a_ctx,
                video_guider=None, audio_guider=guider,
            )
        else:
            denoiser = SimpleDenoiser(v_context=None, a_context=a_ctx)

        sigmas = LTX2Scheduler().execute(steps=args.steps, latent=state.latent).to(self.device)

        # ---- Denoise ----
        # NOTE: don't wrap in gpu_model() — that context manager moves the
        # model back off GPU on exit, which breaks subsequent iterations of
        # our warm validator. We keep the velocity model resident.
        x0 = X0Model(self.velocity_model)
        batched = BatchSplitAdapter(x0, max_batch_size=1)
        _, audio_state = euler_denoising_loop(
            sigmas=sigmas, video_state=None, audio_state=state,
            stepper=EulerDiffusionStep(), transformer=batched, denoiser=denoiser,
        )

        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)
        decoded = self.audio_decoder(audio_state.latent)

        wav = decoded.waveform
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torchaudio.save(output_path, wav.float().cpu(), decoded.sampling_rate)
        logging.info(f"  -> {output_path} ({wav.shape[-1]/decoded.sampling_rate:.1f}s, "
                     f"{time.time()-t_total:.1f}s)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    import yaml
    with open(args.val_config) as f:
        val_cfg = yaml.safe_load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    # Build validator once (models warm for all entries).
    validator = WarmValidator(
        checkpoint=args.checkpoint,
        full_checkpoint=args.full_checkpoint,
        gemma_root=args.gemma_root,
        lora_path=args.lora,
        lora_rank=args.lora_rank,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
    )

    n_ok = n_fail = 0
    t0 = time.time()
    for entry in val_cfg.get("speakers", []):
        name = entry["name"]
        out_path = os.path.join(args.output_dir, f"{name}.wav")
        try:
            validator.generate(
                prompt=entry["prompt"],
                output_path=out_path,
                voice_ref=entry.get("reference"),
                args=args,
            )
            n_ok += 1
            logging.info(f"  [{name}] OK")
        except Exception as e:
            n_fail += 1
            logging.warning(f"  [{name}] FAILED: {e}")
            traceback.print_exc()

    logging.info(f"Validation done: ok={n_ok} fail={n_fail} in {(time.time()-t0)/60:.1f}min "
                 f"at {args.output_dir}")


if __name__ == "__main__":
    main()
