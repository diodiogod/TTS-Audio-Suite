#!/usr/bin/env python3
"""
Audio-Only IC-LoRA Training for Voice Cloning on LTX-2.3.

Uses the IC-LoRA pattern: reference audio tokens are APPENDED to the end of
the target sequence using AudioConditionByReferenceLatent.  Loss is computed
only on target tokens; reference tokens remain clean (denoise_mask=0).

This follows the official video-to-video IC-LoRA strategy closely, but adapted
for the audio-only modality path.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python train_audio_iclora.py --data-dir ... --speaker-index ...

Usage (multi-GPU with accelerate):
    CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 train_audio_iclora.py ...
"""

import argparse
import logging
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ltx2"))
# ltx-pipelines already on path via ltx2/

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import audio conditioning item from our module
sys.path.insert(0, MODEL_DIR)
from audio_conditioning import AudioConditionByReferenceLatent


# ─── Timestep Sampling ───

class DistilledTimestepSampler:
    """Sample timesteps from the distilled sigma schedule.

    The distilled model was trained to denoise at these specific sigma values.
    We sample uniformly from the intervals between consecutive sigmas,
    matching the distribution the model actually operates on.
    """

    # Distilled 8-step sigma values (boundaries of denoising intervals)
    SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

    def __init__(self, jitter: float = 0.02):
        self.jitter = jitter

    def sample(self, batch_size: int, seq_length: int = None, device: torch.device = None) -> torch.Tensor:
        n_intervals = len(self.SIGMAS) - 1
        interval_idx = torch.randint(0, n_intervals, (batch_size,), device=device)
        t = torch.rand(batch_size, device=device)
        sigma_high = torch.tensor([self.SIGMAS[i] for i in interval_idx], device=device)
        sigma_low = torch.tensor([self.SIGMAS[i + 1] for i in interval_idx], device=device)
        sigma = sigma_low + t * (sigma_high - sigma_low)
        return sigma.clamp(0.01, 0.99)


class ShiftedLogitNormalTimestepSampler:
    """Shifted logit-normal distribution, shift depends on sequence length."""

    def __init__(self, std: float = 1.0, eps: float = 1e-3, uniform_prob: float = 0.1):
        self.std = std
        self.eps = eps
        self.uniform_prob = uniform_prob
        self.normal_999_percentile = 3.0902 * std
        self.normal_005_percentile = -2.5758 * std

    def sample(self, batch_size: int, seq_length: int, device: torch.device = None) -> torch.Tensor:
        mu = self._get_shift(seq_length)
        normal = torch.randn(batch_size, device=device) * self.std + mu
        logitnormal = torch.sigmoid(normal)

        p999 = torch.sigmoid(torch.tensor(mu + self.normal_999_percentile, device=device))
        p005 = torch.sigmoid(torch.tensor(mu + self.normal_005_percentile, device=device))
        stretched = (logitnormal - p005) / (p999 - p005)
        stretched = torch.where(stretched >= self.eps, stretched, 2 * self.eps - stretched)
        stretched = stretched.clamp(0, 1)

        uniform = (1 - self.eps) * torch.rand(batch_size, device=device) + self.eps
        prob = torch.rand(batch_size, device=device)
        return torch.where(prob > self.uniform_prob, stretched, uniform)

    @staticmethod
    def _get_shift(seq_length, min_tok=1024, max_tok=4096, min_s=0.95, max_s=2.05):
        m = (max_s - min_s) / (max_tok - min_tok)
        return m * seq_length + (min_s - m * min_tok)


# ─── Dataset ───

def build_speaker_map(index_paths, data_dirs):
    """Map speaker → [(data_dir, sample_idx)] from index file(s).

    The sample index comes from field 0 of the `~`-delimited row when it
    parses as int (allows subset indexes that keep original sample numbers),
    otherwise we fall back to the row's line number (legacy behaviour for
    string-keyed indexes like tts_training_data_podcast).
    """
    speaker_to_samples = defaultdict(list)
    for index_path, data_dir in zip(index_paths, data_dirs):
        with open(index_path) as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split("~")
                if len(parts) < 7:
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    idx = line_num
                speaker_id = parts[1]
                speaker_to_samples[speaker_id].append((data_dir, idx))
    return {k: v for k, v in speaker_to_samples.items() if len(v) >= 2}


class IDLoRADataset(Dataset):
    # Silence-latent reference loaded once, used to detect and strip any
    # leading silence frames baked into the preprocessed audio_latents. The
    # training loop ALREADY prepends 0-25 random silence frames, so we don't
    # want accidental silence in the source data compounding on top.
    _silence_ref = None

    @classmethod
    def _load_silence_ref(cls):
        if cls._silence_ref is None:
            p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "assets", "silence_latent_frame.pt")
            if os.path.exists(p):
                cls._silence_ref = torch.load(p, weights_only=True).float().squeeze()  # [C, F]
        return cls._silence_ref

    def __init__(self, speaker_map):
        self.samples = []
        self.speaker_map = {}
        for speaker, entries in speaker_map.items():
            valid = []
            for data_dir, idx in entries:
                audio_path = Path(data_dir) / "audio_latents" / f"sample_{idx:06d}.pt"
                cond_path = Path(data_dir) / "conditions" / f"sample_{idx:06d}.pt"
                if audio_path.exists() and cond_path.exists():
                    valid.append((data_dir, idx))
            if len(valid) >= 2:
                self.speaker_map[speaker] = valid
        for speaker, entries in self.speaker_map.items():
            for entry in entries:
                self.samples.append((entry, speaker))
        IDLoRADataset._load_silence_ref()

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, data_dir, idx):
        base = Path(data_dir)
        audio = torch.load(base / "audio_latents" / f"sample_{idx:06d}.pt", weights_only=False)
        # Prefer prefix-stripped text embeddings if they exist (re-encoded with
        # just the quoted dialogue, dropping the "A woman says, " / "A man
        # speaks with X accent, " scene-description prefix).
        stripped = base / "conditions_stripped" / f"sample_{idx:06d}.pt"
        cond_path = stripped if stripped.exists() else base / "conditions" / f"sample_{idx:06d}.pt"
        cond = torch.load(cond_path, weights_only=False)
        if isinstance(audio, dict):
            audio = audio.get("audio_latent", audio.get("latent", list(audio.values())[0]))
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        audio_feats = cond.get("audio_prompt_embeds", cond.get("prompt_embeds"))
        attn_mask = cond.get("prompt_attention_mask")
        # The audio_connector has num_learnable_registers=128 and asserts the
        # input sequence length is divisible by 128. Our new preprocessing
        # saved trimmed conditions (dropping left-padding to save disk), which
        # produces short/irregular sequence lengths. Left-pad back to the next
        # multiple of 128 with zeros (matching the tokenizer's left-padding
        # convention) so this assertion holds.
        REG = 128
        L = audio_feats.shape[0]
        target_L = ((L + REG - 1) // REG) * REG
        if target_L != L:
            pad_len = target_L - L
            pad_emb = torch.zeros(pad_len, audio_feats.shape[1],
                                  dtype=audio_feats.dtype)
            pad_mask = torch.zeros(pad_len, dtype=attn_mask.dtype)
            audio_feats = torch.cat([pad_emb, audio_feats], dim=0)
            attn_mask = torch.cat([pad_mask, attn_mask], dim=0)
        return audio, audio_feats, attn_mask

    def __getitem__(self, idx):
        (data_dir, tgt_idx), speaker = self.samples[idx]
        tgt_latent, audio_feats, attn_mask = self._load_sample(data_dir, tgt_idx)

        # Drop the reference entirely for non-voice-cloning categories:
        #   - SFX samples (speaker starts with "sfx_"): descriptive sound events,
        #     no speaker identity to clone.
        #   - Song/music samples (suno dataset): prompts describe the music style,
        #     reference audio doesn't transfer anything useful.
        # Return a zero-length ref so the model trains target-only for these.
        drop_ref = speaker.startswith("sfx_") or "preprocessed_ltx_suno" in str(data_dir)
        if drop_ref:
            C, F_dim = tgt_latent.shape[0], tgt_latent.shape[2]
            ref_latent = torch.zeros(C, 0, F_dim, dtype=tgt_latent.dtype)
        else:
            entries = self.speaker_map[speaker]
            ref_entry = random.choice([e for e in entries if e[1] != tgt_idx])
            ref_latent, _, _ = self._load_sample(*ref_entry)

        return {
            "tgt_latent": tgt_latent,
            "ref_latent": ref_latent,
            "audio_features": audio_feats,
            "attention_mask": attn_mask,
        }


# ─── Model building ───

def build_audio_only_model(checkpoint_path, device, dtype):
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.loader.sd_ops import SDOps
    from ltx_core.model.transformer.model import LTXModel, LTXModelType
    from ltx_core.model.model_protocol import ModelConfigurator
    from ltx_core.model.transformer.attention import AttentionFunction
    from ltx_core.model.transformer.rope import LTXRopeType

    sd_ops = SDOps("AO").with_matching(prefix="model.diffusion_model.").with_replacement("model.diffusion_model.", "")

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
                positional_embedding_theta=t.get("positional_embedding_theta", 10000.0),
                audio_positional_embedding_max_pos=t.get("audio_positional_embedding_max_pos", [20]),
                timestep_scale_multiplier=t.get("timestep_scale_multiplier", 1000),
                use_middle_indices_grid=t.get("use_middle_indices_grid", True),
                rope_type=LTXRopeType(t.get("rope_type", "interleaved")),
                double_precision_rope=t.get("frequencies_precision", False) == "float64",
                apply_gated_attention=t.get("apply_gated_attention", False),
                audio_caption_projection=cp,
                cross_attention_adaln=t.get("cross_attention_adaln", False),
            )

    builder = Builder(model_path=checkpoint_path, model_class_configurator=Cfg,
                      model_sd_ops=sd_ops, registry=DummyRegistry())
    return builder.build(device=device, dtype=dtype)


def load_audio_connector(checkpoint_path, device, dtype):
    # ltx-trainer already on path via ltx2/
    from ltx_trainer.model_loader import load_embeddings_processor
    emb_proc = load_embeddings_processor(checkpoint_path, device=device, dtype=dtype)
    connector = emb_proc.audio_connector
    del emb_proc
    return connector


def apply_lora(model, rank, alpha, dropout=0.0):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=[
            # Self-attention over audio tokens (voice-transfer pathway via ref).
            "audio_attn1.to_k", "audio_attn1.to_q", "audio_attn1.to_v", "audio_attn1.to_out.0",
            # Cross-attention (audio ↔ text context) NOT adapted — keep base
            # model's prompt→audio behaviour intact and rely on dataset balance
            # to drive expressiveness. (v15c tried this with adaLN unfreeze,
            # that proved too destructive; v16 tries it adaLN-frozen.)
            # FFN — non-linear capacity for style/phonetic adaptation.
            "audio_ff.net.0.proj", "audio_ff.net.2",
        ],
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")
    return model


@torch.no_grad()
def prepare_audio_context(audio_connector, audio_features, attention_mask, device, dtype):
    from ltx_core.text_encoders.gemma.embeddings_processor import convert_to_additive_mask
    audio_features = audio_features.to(device=device, dtype=dtype)
    attention_mask = attention_mask.to(device=device)
    if audio_features.shape[0] > 1:
        results = []
        for i in range(audio_features.shape[0]):
            feat_i = audio_features[i:i+1]
            mask_i = attention_mask[i:i+1]
            additive = convert_to_additive_mask(mask_i, feat_i.dtype)
            enc_i, _ = audio_connector(feat_i, additive)
            results.append(enc_i)
        return torch.cat(results, dim=0)
    additive_mask = convert_to_additive_mask(attention_mask, audio_features.dtype)
    audio_encoded, _ = audio_connector(audio_features, additive_mask)
    return audio_encoded


# ─── Validation ───

def _unwrap_model_safe(model):
    """Strip DDP / peft wrappers without going through accelerate.unwrap_model,
    which imports deepspeed — broken in our env (torch API drift)."""
    while hasattr(model, "module"):
        model = model.module
    return model


def run_validation(lora_path, val_config_path, output_dir, step, lora_rank=128):
    """Call validate.py in a subprocess. It loads TTSServer (the same stack
    the warm server / Gradio app uses), attaches our LoRA, then iterates every
    entry in val_config with the same inference settings the user tests with.
    Single subprocess amortises the model-load cost across all val entries.

    Forces validation onto VAL_GPU (default "0") because training already
    occupies the rest. Override via TRAIN_VAL_GPU env var.
    """
    import subprocess
    val_dir = os.path.join(output_dir, "validation", f"step_{step:05d}")
    os.makedirs(val_dir, exist_ok=True)
    script = os.path.join(os.path.dirname(__file__), "validate.py")
    cmd = [
        sys.executable, script,
        "--val-config", val_config_path,
        "--output-dir", val_dir,
        "--lora", lora_path,
        "--lora-rank", str(lora_rank),
        # Use raw estimator output (no +10% buffer) so we can hear
        # whether the model needs more/less duration at current quality.
        "--duration-multiplier", "1.0",
    ]
    log_path = os.path.join(val_dir, "validate.log")
    env = os.environ.copy()
    # Validation needs its OWN GPU (training fills the others).
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("TRAIN_VAL_GPU", "0")
    try:
        with open(log_path, "w") as logf:
            result = subprocess.run(
                cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=1800, env=env,
            )
        if result.returncode == 0:
            logging.info(f"  Validation step {step}: OK → {val_dir}")
        else:
            logging.warning(f"  Validation step {step} FAILED (see {log_path})")
    except subprocess.TimeoutExpired:
        logging.warning(f"  Validation step {step} TIMEOUT (>30min)")


# ─── Args ───

def collate_audio_batch(batch):
    """Pad variable-length audio and track real lengths for loss masking."""
    # TTS Audio Suite patch: keep this callable at module scope so Windows
    # spawn-based DataLoader workers can pickle it.
    max_tgt_T = max(item["tgt_latent"].shape[1] for item in batch)
    max_ref_T = max(item["ref_latent"].shape[1] for item in batch)
    channels = batch[0]["tgt_latent"].shape[0]
    feature_dim = batch[0]["tgt_latent"].shape[2]

    tgt_list, ref_list, feat_list, mask_list = [], [], [], []
    tgt_lengths, ref_lengths = [], []
    for item in batch:
        tgt = item["tgt_latent"]
        ref = item["ref_latent"]
        tgt_lengths.append(tgt.shape[1])
        ref_lengths.append(ref.shape[1])

        if tgt.shape[1] < max_tgt_T:
            pad = torch.zeros(
                channels,
                max_tgt_T - tgt.shape[1],
                feature_dim,
                dtype=tgt.dtype,
            )
            tgt = torch.cat([tgt, pad], dim=1)
        tgt_list.append(tgt)

        if ref.shape[1] < max_ref_T:
            pad = torch.zeros(
                channels,
                max_ref_T - ref.shape[1],
                feature_dim,
                dtype=ref.dtype,
            )
            ref = torch.cat([ref, pad], dim=1)
        ref_list.append(ref)
        feat_list.append(item["audio_features"])
        mask_list.append(item["attention_mask"])

    return {
        "tgt_latent": torch.stack(tgt_list),
        "ref_latent": torch.stack(ref_list),
        "audio_features": torch.stack(feat_list),
        "attention_mask": torch.stack(mask_list),
        "tgt_lengths": torch.tensor(tgt_lengths),
        "ref_lengths": torch.tensor(ref_lengths),
    }


def parse_args():
    # First pass: pull out --config so its values can become argparse defaults.
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("--config", default=None,
                            help="YAML file with default values for any of the flags below. "
                                 "Explicit CLI flags still override the YAML.")
    cfg_args, remaining = cfg_parser.parse_known_args()
    yaml_defaults: dict = {}
    if cfg_args.config:
        import yaml as _yaml
        with open(cfg_args.config) as f:
            yaml_defaults = _yaml.safe_load(f) or {}
        # YAML keys are dashes-or-underscores → normalize to argparse dest (underscore).
        yaml_defaults = {k.replace("-", "_"): v for k, v in yaml_defaults.items()}

    def _yaml(name, fallback):
        return yaml_defaults.get(name, fallback)

    p = argparse.ArgumentParser(
        parents=[cfg_parser],
        description="Audio-Only IC-LoRA Training for Voice Cloning",
    )
    p.add_argument("--data-dir", required="data_dir" not in yaml_defaults,
                   nargs="+", default=_yaml("data_dir", None))
    p.add_argument("--speaker-index", required="speaker_index" not in yaml_defaults,
                   nargs="+", default=_yaml("speaker_index", None))
    p.add_argument("--output-dir", default=_yaml("output_dir", os.path.join(MODEL_DIR, "tts_iclora_v1")))
    p.add_argument("--checkpoint", default=_yaml("checkpoint", os.path.join(MODEL_DIR, "dramabox-dit-v1.safetensors")))
    p.add_argument("--full-checkpoint", default=_yaml("full_checkpoint", os.path.join(MODEL_DIR, "dramabox-audio-components.safetensors")))
    p.add_argument("--base-model", choices=["distilled", "dev"], default=_yaml("base_model", "dev"),
                   help="Base model type: distilled uses DistilledTimestepSampler, dev uses ShiftedLogitNormal")
    p.add_argument("--lora-rank", type=int, default=_yaml("lora_rank", 128))
    p.add_argument("--lora-alpha", type=int, default=_yaml("lora_alpha", 128))
    p.add_argument("--lora-dropout", type=float, default=_yaml("lora_dropout", 0.0),
                   help="Dropout applied to LoRA A/B matrices during training. "
                        "Recommended ~0.1 for small datasets to regularize.")
    p.add_argument("--resume-lora", default=_yaml("resume_lora", None))
    p.add_argument("--resume-step-offset", type=int, default=_yaml("resume_step_offset", None),
                   help="Step to add when naming saved checkpoints. If None, inferred "
                        "from --resume-lora filename (e.g. lora_step_10000.safetensors → 10000). "
                        "Set to 0 to start numbering at 0 regardless.")
    p.add_argument("--ref-ratio", type=float, default=_yaml("ref_ratio", 0.3),
                   help="Fraction of target length to use as reference (default 0.3)")
    p.add_argument("--max-ref-tokens", type=int, default=_yaml("max_ref_tokens", 200),
                   help="Maximum reference tokens after patchification (default 200)")
    p.add_argument("--text-dropout", type=float, default=_yaml("text_dropout", 0.0),
                   help="Probability of dropping text conditioning (forces reliance on voice ref)")
    p.add_argument("--steps", type=int, default=_yaml("steps", 30000))
    p.add_argument("--lr", type=float, default=_yaml("lr", 3e-5))
    p.add_argument("--lr-scheduler", choices=["cosine", "linear", "constant"], default=_yaml("lr_scheduler", "cosine"))
    p.add_argument("--batch-size", type=int, default=_yaml("batch_size", 1))
    p.add_argument("--grad-accum", type=int, default=_yaml("grad_accum", 4))
    p.add_argument("--max-grad-norm", type=float, default=_yaml("max_grad_norm", 1.0))
    p.add_argument("--save-every", type=int, default=_yaml("save_every", 1000))
    p.add_argument("--log-every", type=int, default=_yaml("log_every", 50))
    p.add_argument("--seed", type=int, default=_yaml("seed", 42))
    p.add_argument("--warmup-steps", type=int, default=_yaml("warmup_steps", 100))
    p.add_argument("--val-config", default=_yaml("val_config", None))
    return p.parse_args(remaining)


# ─── Main ───

def main():
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
    )

    is_main = accelerator.is_main_process
    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    set_seed(args.seed)
    device = accelerator.device
    dtype = torch.bfloat16

    os.makedirs(args.output_dir, exist_ok=True)

    # Save training args
    if is_main:
        import yaml
        args_dict = vars(args).copy()
        args_dict["_meta"] = {
            "world_size": accelerator.num_processes,
            "dtype": str(dtype),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": "train_audio_iclora.py",
            "pattern": "IC-LoRA (ref appended to end)",
        }
        with open(os.path.join(args.output_dir, "training_args.yaml"), "w") as f:
            yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False)

    from ltx_core.components.patchifiers import AudioPatchifier
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig
    from ltx_core.tools import AudioLatentTools
    from ltx_core.types import AudioLatentShape, LatentState
    from ltx_pipelines.utils.helpers import modality_from_latent_state, timesteps_from_mask

    # Build speaker map
    if is_main:
        logging.info("Building speaker map...")
    speaker_map = build_speaker_map(args.speaker_index, args.data_dir)
    if is_main:
        logging.info(f"Speaker map: {len(speaker_map)} speakers, "
                     f"{sum(len(v) for v in speaker_map.values())} samples")

    # Load model
    if is_main:
        logging.info("Loading audio-only model...")
    model = build_audio_only_model(args.checkpoint, device, dtype)

    if is_main:
        logging.info("Loading audio connector...")
    audio_connector = load_audio_connector(args.full_checkpoint, device, dtype)
    audio_connector.eval()
    for p in audio_connector.parameters():
        p.requires_grad = False

    if is_main:
        logging.info(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    model = apply_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)

    # Resume from checkpoint
    if args.resume_lora:
        from safetensors.torch import load_file as st_load
        if is_main:
            logging.info(f"Resuming from: {args.resume_lora}")
        lora_sd = st_load(args.resume_lora)
        mapped = {}
        for k, v in lora_sd.items():
            nk = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                ".lora_B.weight", ".lora_B.default.weight")
            mapped[nk] = v
        model.load_state_dict(mapped, strict=False)

    # Determine step offset for save filenames. Without this, resuming a run
    # restarts step numbering at 0 and would overwrite earlier phase-1
    # checkpoints with the same save_every cadence.
    if args.resume_step_offset is None:
        resume_offset = 0
        if args.resume_lora:
            import re as _re
            m = _re.search(r"lora_step_(\d+)", os.path.basename(args.resume_lora))
            if m:
                resume_offset = int(m.group(1))
        args.resume_step_offset = resume_offset
    if is_main and args.resume_step_offset:
        logging.info(f"Save-step offset: +{args.resume_step_offset}")

    model.train()
    model.base_model.model.set_gradient_checkpointing(True)

    # Dataset & DataLoader
    dataset = IDLoRADataset(speaker_map)
    if is_main:
        logging.info(f"Dataset: {len(dataset)} samples, {len(dataset.speaker_map)} speakers")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                            pin_memory=True, drop_last=True, collate_fn=collate_audio_batch)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01,
    )

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_steps)
    remaining = args.steps - args.warmup_steps
    if args.lr_scheduler == "cosine":
        # Warmup -> constant hold (20% of remaining) -> cosine decay
        hold_steps = max(remaining // 5, 0)
        decay_steps = max(remaining - hold_steps, 1)
        hold_sched = ConstantLR(optimizer, factor=1.0, total_iters=hold_steps)
        decay_sched = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)
        scheduler = SequentialLR(
            optimizer,
            [warmup, hold_sched, decay_sched],
            milestones=[args.warmup_steps, args.warmup_steps + hold_steps],
        )
    elif args.lr_scheduler == "linear":
        main_sched = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=max(remaining, 1))
        scheduler = SequentialLR(optimizer, [warmup, main_sched], milestones=[args.warmup_steps])
    else:
        main_sched = ConstantLR(optimizer, factor=1.0, total_iters=max(remaining, 1))
        scheduler = SequentialLR(optimizer, [warmup, main_sched], milestones=[args.warmup_steps])

    # Prepare with Accelerate — but NOT the scheduler. AcceleratedScheduler
    # calls the underlying scheduler.step() `num_processes` times per sync,
    # which silently scales down our warmup/cosine spans by that factor.
    # We call scheduler.step() ourselves, gated on sync_gradients → exactly
    # one advance per optimizer step, as the yaml spec intends.
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    patchifier = AudioPatchifier(patch_size=1)

    # Select timestep sampler based on base model type
    if args.base_model == "distilled":
        timestep_sampler = DistilledTimestepSampler()
        if is_main:
            logging.info("Using DistilledTimestepSampler (matching distilled model sigmas)")
    else:
        timestep_sampler = ShiftedLogitNormalTimestepSampler()
        if is_main:
            logging.info("Using ShiftedLogitNormalTimestepSampler (dev model)")

    # Training loop
    if is_main:
        logging.info(f"Training: {args.steps} steps, lr={args.lr}, scheduler={args.lr_scheduler}, "
                     f"batch={args.batch_size}, grad_accum={args.grad_accum}, "
                     f"world_size={accelerator.num_processes}, "
                     f"ref_ratio={args.ref_ratio}, max_ref_tokens={args.max_ref_tokens}")
        logging.info("IC-LoRA pattern: ref tokens APPENDED to target, loss on target only")

    data_iter = iter(dataloader)
    step = 0
    accum_loss = 0.0
    best_loss = float("inf")
    best_step = 0
    t0 = time.time()

    total_micro_steps = args.steps * args.grad_accum

    for micro_step in range(total_micro_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        is_opt_step = (micro_step + 1) % args.grad_accum == 0
        if is_opt_step:
            step += 1
            if is_main:
                # TTS Audio Suite patch: provide lightweight per-step telemetry
                # to the parent process even when human logs use log_every=50.
                print(
                    f"TTS_SUITE_PROGRESS step={step} total={args.steps}",
                    flush=True,
                )

        with accelerator.accumulate(model):
            tgt_latent = batch["tgt_latent"].to(dtype=dtype)  # [B, C, max_tgt_T, F]
            ref_latent = batch["ref_latent"].to(dtype=dtype)  # [B, C, max_ref_T, F]
            tgt_lengths = batch["tgt_lengths"].to(device=device)  # [B]
            B = tgt_latent.shape[0]

            # ── Random silence padding (0-1s) ── ltx_audio_tts baseline.
            # User observed reference-audio leak at end of generations when this
            # was reduced to 5 (v14) or 10 frames (v16/v17) — the model seemed
            # to use the extra target budget to regurgitate ref content. Full
            # 25 frames (0-1s avg 500ms) was apparently load-bearing for
            # regularising the boundary and reducing hallucinations.
            # Uses the real silence latent (not zeros) so the VAE decodes it as
            # true silence instead of static noise.
            max_pad_frames = 25  # ~1s at 25 latent frames/sec
            pad_frames = random.randint(0, max_pad_frames)
            if pad_frames > 0:
                C, F_dim = tgt_latent.shape[1], tgt_latent.shape[3]
                if not hasattr(args, '_silence_frame') or args._silence_frame is None:
                    _sf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "silence_latent_frame.pt")
                    if os.path.exists(_sf_path):
                        args._silence_frame = torch.load(_sf_path, weights_only=True)  # [C, 1, F]
                        if is_main:
                            logging.info(f"Loaded silence latent from {_sf_path}")
                    else:
                        args._silence_frame = False  # fallback to zeros
                        if is_main:
                            logging.warning(f"silence_latent_frame.pt not found, using zeros")
                if args._silence_frame is not False:
                    sf = args._silence_frame.to(dtype=dtype, device=device)  # [C, 1, F]
                    silence_pad = sf.unsqueeze(0).expand(B, -1, pad_frames, -1)  # [B, C, pad, F]
                else:
                    silence_pad = torch.zeros(B, C, pad_frames, F_dim, dtype=dtype, device=device)
                tgt_latent = torch.cat([silence_pad, tgt_latent], dim=2)

            # Cap reference to max_ref_tokens (in latent frames, before patchification)
            # After patchification, ref_T tokens = ref frames (patch_size=1)
            ref_T_frames = min(ref_latent.shape[2], args.max_ref_tokens)
            ref_latent = ref_latent[:, :, :ref_T_frames, :]

            tgt_T_frames = tgt_latent.shape[2]  # max (padded) target frames

            # ── Step 1: Create target AudioLatentShape and AudioLatentTools ──
            tgt_shape = AudioLatentShape(
                batch=B,
                channels=tgt_latent.shape[1],  # 8
                frames=tgt_T_frames,
                mel_bins=tgt_latent.shape[3],   # 16
            )

            audio_tools = AudioLatentTools(
                patchifier=patchifier,
                target_shape=tgt_shape,
            )

            # ── Step 2: Create initial state from target latent ──
            # create_initial_state patchifies: [B, C, T, F] -> [B, T, C*F]
            # Also creates denoise_mask=1 (all target tokens will be denoised)
            # and computes temporal positions
            state = audio_tools.create_initial_state(
                device=device,
                dtype=dtype,
                initial_latent=tgt_latent,
            )
            # state.latent: [B, tgt_T, 128], state.denoise_mask: [B, tgt_T, 1]
            # state.positions: [B, 1, tgt_T, 2]

            tgt_T = audio_tools.target_shape.token_count()  # = tgt_T_frames

            # ── Step 3: Apply flow-matching noise to target BEFORE appending ref ──
            # Sample sigma
            total_tokens = tgt_T + ref_T_frames
            sigma = timestep_sampler.sample(B, total_tokens, device=device)
            sigma_exp = sigma.view(-1, 1, 1)  # [B, 1, 1]

            noise = torch.randn_like(state.latent)  # [B, tgt_T, 128]
            noisy_tgt = (1 - sigma_exp) * state.latent + sigma_exp * noise

            # Replace the latent in state with the noisy version
            # (clean_latent stays clean for post_process_latent pattern)
            state = LatentState(
                latent=noisy_tgt,
                denoise_mask=state.denoise_mask,
                positions=state.positions,
                clean_latent=state.clean_latent,
                attention_mask=state.attention_mask,
            )

            # ── Step 4: Append reference tokens using AudioConditionByReferenceLatent ──
            # This appends ref tokens to the END with denoise_mask=0 (frozen/clean)
            # Skip entirely when ref_T=0 (SFX / song samples): the model trains
            # target-only for those categories since there's no voice to clone.
            if ref_T_frames > 0:
                ref_conditioning = AudioConditionByReferenceLatent(
                    latent=ref_latent,
                    strength=1.0,  # 1.0 = ref fully clean (denoise_mask=0)
                )
                state = ref_conditioning.apply_to(
                    latent_state=state,
                    latent_tools=audio_tools,
                )
            # state.latent: [B, tgt_T + ref_T, 128]
            # state.denoise_mask: [B, tgt_T + ref_T, 1]
            #   target tokens: 1.0 (denoise), ref tokens: 0.0 (frozen)
            # state.positions: [B, 1, tgt_T + ref_T, 2]

            # ── Step 5: Build loss mask for target tokens (excluding padding) ──
            # loss_mask: 1 for real target tokens, 0 for padding and ref tokens
            loss_mask = torch.zeros(B, tgt_T, device=device)
            for b_idx in range(B):
                real_len = min(tgt_lengths[b_idx].item(), tgt_T)
                loss_mask[b_idx, :real_len] = 1.0

            # ── Step 6: Prepare text context ──
            # Text conditioning dropout: randomly zero out text context to force
            # the model to rely on the voice reference for identity/style.
            with torch.no_grad():
                audio_context = prepare_audio_context(
                    audio_connector, batch["audio_features"],
                    batch["attention_mask"], device, dtype)
                if args.text_dropout > 0 and random.random() < args.text_dropout:
                    audio_context = torch.zeros_like(audio_context)

            # ── Step 7: Build Modality using modality_from_latent_state ──
            # timesteps = sigma * denoise_mask (ref gets 0, target gets sigma)
            audio_mod = modality_from_latent_state(
                state=state,
                context=audio_context,
                sigma=sigma,
                enabled=True,
            )

            # ── Step 8: Forward pass ──
            perturbations = BatchedPerturbationConfig.empty(B)
            with torch.autocast(device_type="cuda", dtype=dtype):
                _, velocity_pred = model(video=None, audio=audio_mod, perturbations=perturbations)

            # ── Step 9: Compute loss (IC-LoRA pattern) ──
            # Target is at the FRONT (indices 0..tgt_T), ref at the END
            # velocity target = noise - clean
            tgt_patchified = audio_tools.patchifier.patchify(tgt_latent)  # [B, tgt_T, 128]
            target_velocity = noise - tgt_patchified

            # Extract target portion of prediction
            pred_tgt = velocity_pred[:, :tgt_T]  # [B, tgt_T, 128]

            # MSE loss with mask: only on real target tokens (not padding or ref)
            per_token_mse = (pred_tgt - target_velocity).pow(2).mean(dim=-1)  # [B, tgt_T]
            loss = per_token_mse.mul(loss_mask).div(loss_mask.mean().clamp(min=1e-6)).mean()

            accelerator.backward(loss)

            if accelerator.sync_gradients and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            # Only advance the LR scheduler once per OPTIMIZER step (not per
            # micro-step). Mirrors AcceleratedOptimizer.step() which is
            # internally gated on sync_gradients.
            if accelerator.sync_gradients:
                scheduler.step()

        accum_loss += loss.item()

        # Logging & saving on optimization steps only
        if is_opt_step and step % args.log_every == 0 and is_main:
            avg_loss = accum_loss / (args.log_every * args.grad_accum)
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            sps = step / elapsed if elapsed > 0 else 0
            eta = (args.steps - step) / sps if sps > 0 else 0
            logging.info(
                f"Step {step}/{args.steps} | loss={avg_loss:.4f} | lr={lr:.2e} | "
                f"tgt_T={tgt_T} ref_T={ref_T_frames} total={tgt_T + ref_T_frames} | "
                f"{sps:.1f} steps/s | ETA {eta/60:.0f}min"
            )

            # Save best whenever loss improves — no warmup gate, so we can
            # observe best checkpoints during warmup too.
            if avg_loss < best_loss:
                best_loss = avg_loss
                old_best = os.path.join(args.output_dir, f"best_step_{best_step:05d}.safetensors")
                best_step = step + args.resume_step_offset
                new_best = os.path.join(args.output_dir, f"best_step_{best_step:05d}.safetensors")
                unwrapped = _unwrap_model_safe(model)
                unwrapped.save_pretrained(args.output_dir)
                adapter = os.path.join(args.output_dir, "adapter_model.safetensors")
                if os.path.exists(adapter):
                    shutil.copy(adapter, new_best)
                if old_best != new_best and os.path.exists(old_best):
                    os.remove(old_best)
                logging.info(f"New best: loss={best_loss:.4f} at step {best_step}")

            accum_loss = 0.0

        if is_opt_step and step % args.save_every == 0 and is_main:
            global_step = step + args.resume_step_offset
            save_path = os.path.join(args.output_dir, f"lora_step_{global_step:05d}.safetensors")
            logging.info(f"Saving: {save_path}")
            unwrapped = _unwrap_model_safe(model)
            unwrapped.save_pretrained(args.output_dir)
            adapter = os.path.join(args.output_dir, "adapter_model.safetensors")
            if os.path.exists(adapter):
                shutil.copy(adapter, save_path)

            if args.val_config:
                logging.info(f"Running validation at step {global_step}...")
                model.eval()
                run_validation(save_path, args.val_config, args.output_dir, global_step,
                               lora_rank=args.lora_rank)
                model.train()

    # Final save
    if is_main:
        unwrapped = _unwrap_model_safe(model)
        unwrapped.save_pretrained(args.output_dir)
        adapter = os.path.join(args.output_dir, "adapter_model.safetensors")
        global_step = step + args.resume_step_offset
        save_path = os.path.join(args.output_dir, f"lora_step_{global_step:05d}.safetensors")
        if os.path.exists(adapter):
            shutil.copy(adapter, save_path)
        logging.info(f"Training complete! {step} steps in {time.time()-t0:.0f}s")
        logging.info(f"Best loss: {best_loss:.4f} at step {best_step}")


if __name__ == "__main__":
    main()
