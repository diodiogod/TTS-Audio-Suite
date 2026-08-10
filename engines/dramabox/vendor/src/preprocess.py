#!/usr/bin/env python3
"""
Preprocess TTS datasets for LTX-2.3 audio-only LoRA fine-tuning.

Takes paired (audio, transcript) data and produces the format expected by
the LTX trainer:
    .precomputed/
    ├── latents/sample_N.pt         # Dummy video latents (minimal)
    ├── conditions/sample_N.pt      # Text embeddings from Gemma
    └── audio_latents/sample_N.pt   # Audio VAE-encoded latents

Supports multiple dataset formats:
  - gemini_synthetic: index.txt with ~-separated fields (id~speaker~lang~sr~samples~dur~phonemes~text)
  - libriheavy: index_ft.txt with ~-separated fields (id~speaker~lang~samples~dur~phonemes~text)
  - manifest: JSON/JSONL with {"audio_filepath": ..., "text": ...}
  - tsv: TSV file with audio_path<TAB>text columns

Usage:
    python preprocess_tts_data.py \
        --dataset-type gemini_synthetic \
        --index /path/to/dataset/index.txt \
        --audio-dir /path/to/dataset/wavs \
        --output-dir /path/to/output/tts_training_data \
        --max-samples 10000 \
        --max-duration 20.0 \
        --min-duration 3.0
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torchaudio

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ltx2"))
# ltx-pipelines on path via ltx2/

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEMMA_DIR = os.environ.get("GEMMA_DIR", "gemma-3-12b-it-qat-q4_0-unquantized")


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess TTS data for LTX-2.3 fine-tuning")
    p.add_argument("--dataset-type", required=True,
                   choices=["gemini_synthetic", "libriheavy", "manifest", "tsv"],
                   help="Dataset format type")
    p.add_argument("--index", required=True, help="Path to index/manifest file")
    p.add_argument("--audio-dir", default=None,
                   help="Base directory for audio files (if paths in index are relative)")
    p.add_argument("--output-dir", required=True, help="Output directory for preprocessed data")
    p.add_argument("--checkpoint", default=os.path.join(MODEL_DIR, "ltx-2.3-22b-distilled.safetensors"))
    p.add_argument("--gemma-root", default=GEMMA_DIR)
    p.add_argument("--max-samples", type=int, default=0, help="Max samples to process (0=all)")
    p.add_argument("--max-duration", type=float, default=20.0, help="Max audio duration in seconds")
    p.add_argument("--min-duration", type=float, default=2.0, help="Min audio duration in seconds")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for text encoding")
    p.add_argument("--skip-existing", action="store_true", help="Skip already processed samples")
    p.add_argument("--audio-only-ckpt", default=None,
                   help="Audio-only checkpoint for VAE encoding (optional, uses full ckpt if not set)")
    p.add_argument("--shard", type=int, default=0, help="Shard index (for parallel processing)")
    p.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    p.add_argument("--gpu", type=int, default=None, help="GPU device index to use")
    return p.parse_args()


def parse_gemini_synthetic(index_path: str, audio_dir: str | None) -> list[dict]:
    """Parse gemini_synthetic format: id~speaker~lang~sr~samples~dur~phonemes~text"""
    samples = []
    with open(index_path) as f:
        for line in f:
            parts = line.strip().split("~")
            if len(parts) < 7:
                continue
            file_id = parts[0]
            text = parts[-1]  # Last field is always the text
            sr = int(parts[3])
            n_samples = int(parts[4])
            duration = n_samples / sr

            # Find audio file
            if audio_dir:
                # Try common extensions
                for ext in [".flac", ".wav", ".mp3"]:
                    audio_path = os.path.join(audio_dir, file_id + ext)
                    if os.path.exists(audio_path):
                        break
                else:
                    continue
            else:
                audio_path = file_id

            samples.append({
                "id": file_id,
                "audio_path": audio_path,
                "text": text,
                "duration": duration,
            })
    return samples


def parse_libriheavy(index_path: str, audio_dir: str | None) -> list[dict]:
    """Parse libriheavy format: id~speaker~lang~samples~dur~phonemes~text"""
    samples = []
    with open(index_path) as f:
        for line in f:
            parts = line.strip().split("~")
            if len(parts) < 7:
                continue
            file_id = parts[0]
            text = parts[-1]
            n_samples = int(parts[3])
            duration = int(parts[4]) / 1000.0  # milliseconds to seconds

            if audio_dir:
                for ext in [".flac", ".wav", ".mp3"]:
                    audio_path = os.path.join(audio_dir, file_id + ext)
                    if os.path.exists(audio_path):
                        break
                else:
                    continue
            else:
                audio_path = file_id

            samples.append({
                "id": file_id,
                "audio_path": audio_path,
                "text": text,
                "duration": duration,
            })
    return samples


def parse_manifest(index_path: str, audio_dir: str | None) -> list[dict]:
    """Parse JSON/JSONL manifest with audio_filepath and text fields."""
    samples = []
    with open(index_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            audio_path = entry.get("audio_filepath", entry.get("audio_path", ""))
            text = entry.get("text", entry.get("transcript", ""))
            duration = entry.get("duration", 0.0)

            if audio_dir and not os.path.isabs(audio_path):
                audio_path = os.path.join(audio_dir, audio_path)

            if os.path.exists(audio_path) and text:
                samples.append({
                    "id": Path(audio_path).stem,
                    "audio_path": audio_path,
                    "text": text,
                    "duration": duration,
                })
    return samples


def parse_tsv(index_path: str, audio_dir: str | None) -> list[dict]:
    """Parse TSV file with audio_path<TAB>text."""
    samples = []
    with open(index_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            audio_path, text = parts[0], parts[1]
            if audio_dir and not os.path.isabs(audio_path):
                audio_path = os.path.join(audio_dir, audio_path)
            if os.path.exists(audio_path):
                samples.append({
                    "id": Path(audio_path).stem,
                    "audio_path": audio_path,
                    "text": text,
                    "duration": 0.0,
                })
    return samples


PARSERS = {
    "gemini_synthetic": parse_gemini_synthetic,
    "libriheavy": parse_libriheavy,
    "manifest": parse_manifest,
    "tsv": parse_tsv,
}


@torch.inference_mode()
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
    from ltx_core.types import Audio
    from ltx_pipelines.utils.blocks import AudioConditioner, PromptEncoder
    from ltx_pipelines.utils.media_io import decode_audio_from_file
    from ltx_trainer.model_loader import load_text_encoder, load_embeddings_processor

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Create output directories
    out = Path(args.output_dir)
    (out / "latents").mkdir(parents=True, exist_ok=True)
    (out / "conditions").mkdir(parents=True, exist_ok=True)
    (out / "audio_latents").mkdir(parents=True, exist_ok=True)

    # Parse dataset
    logging.info(f"Parsing {args.dataset_type} dataset from {args.index}...")
    samples = PARSERS[args.dataset_type](args.index, args.audio_dir)
    logging.info(f"Found {len(samples)} samples")

    # Filter by duration
    before = len(samples)
    samples = [s for s in samples if args.min_duration <= s["duration"] <= args.max_duration]
    logging.info(f"After duration filter [{args.min_duration}s, {args.max_duration}s]: {len(samples)} (dropped {before - len(samples)})")

    if args.max_samples > 0:
        samples = samples[:args.max_samples]
        logging.info(f"Limiting to {len(samples)} samples")

    # Assign global indices before sharding
    for i, s in enumerate(samples):
        s["global_idx"] = i

    # Shard the data for parallel processing
    if args.num_shards > 1:
        total = len(samples)
        samples = samples[args.shard::args.num_shards]
        logging.info(f"Shard {args.shard}/{args.num_shards}: {len(samples)} samples (of {total} total)")

    # ── Step 1: Encode text with Gemma (Blocks 1+2 only) ──
    # The trainer runs Block 3 (embeddings processor/connectors) during training,
    # so we only precompute Blocks 1+2 here (Gemma LLM + feature extractor).
    logging.info("Loading text encoder (Gemma + feature extractor)...")
    gemma_config_path = Path(args.gemma_root) / "config.json"
    gemma_config = {}
    if gemma_config_path.is_file():
        try:
            gemma_config = json.loads(gemma_config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    prompt_encoder = None
    if "quantization_config" in gemma_config:
        # TTS Audio Suite patch: the suite distributes a pre-quantized BNB
        # Gemma checkpoint. The official tensor builder treats its packed
        # weights as dense matrices, producing thousands of shape mismatches.
        # Reuse the inference loader that already understands this format.
        logging.info("Loading pre-quantized Gemma through the BNB prompt encoder...")
        prompt_encoder = PromptEncoder(
            checkpoint_path=args.checkpoint,
            gemma_root=args.gemma_root,
            dtype=dtype,
            device=device,
            warm=True,
            use_bnb_4bit=True,
            audio_only=True,
        )
        text_encoder = prompt_encoder._warm_text_encoder
        embeddings_processor = prompt_encoder._warm_embeddings_processor
        text_encoder.feature_extractor = embeddings_processor.feature_extractor
        prompt_encoder._warm_text_encoder = None
        prompt_encoder._warm_embeddings_processor = None
    else:
        text_encoder = load_text_encoder(args.gemma_root, device=device, dtype=dtype)

        # Load feature extractor on CPU first to save GPU memory, then move to device
        logging.info("Loading feature extractor (on CPU first to save GPU memory)...")
        emb_proc = load_embeddings_processor(args.checkpoint, device="cpu", dtype=dtype)
        text_encoder.feature_extractor = emb_proc.feature_extractor.to(device)
        del emb_proc
    torch.cuda.empty_cache()

    logging.info("Encoding text prompts (Blocks 1+2: Gemma + feature extractor)...")
    for i, sample in enumerate(samples):
        gidx = sample["global_idx"]
        cond_path = out / "conditions" / f"sample_{gidx:06d}.pt"
        if args.skip_existing and cond_path.exists():
            continue

        text = sample["text"]
        # Run Blocks 1+2: Gemma LLM → feature extractor
        hidden_states, attention_mask = text_encoder.encode(text)
        video_feats, audio_feats = text_encoder.feature_extractor(
            hidden_states, attention_mask, "left"
        )

        torch.save({
            "video_prompt_embeds": video_feats.squeeze(0).cpu(),
            "audio_prompt_embeds": audio_feats.squeeze(0).cpu() if audio_feats is not None else video_feats.squeeze(0).cpu(),
            "prompt_attention_mask": attention_mask.squeeze(0).bool().cpu(),
        }, cond_path)

        if i % 100 == 0:
            logging.info(f"  Text encoding: {i}/{len(samples)}")

    del text_encoder
    if prompt_encoder is not None:
        del embeddings_processor
        del prompt_encoder
    torch.cuda.empty_cache()

    # ── Step 2: Encode audio with Audio VAE ──
    ckpt_for_vae = args.audio_only_ckpt or args.checkpoint
    logging.info(f"Loading audio VAE from {ckpt_for_vae}...")

    ac = AudioConditioner(checkpoint_path=ckpt_for_vae, dtype=dtype, device=device)

    logging.info("Encoding audio samples...")
    for idx, sample in enumerate(samples):
        gidx = sample["global_idx"]
        audio_path = out / "audio_latents" / f"sample_{gidx:06d}.pt"
        if args.skip_existing and audio_path.exists():
            continue

        try:
            # Load audio
            voice = decode_audio_from_file(sample["audio_path"], device, 0.0, args.max_duration)
            if voice is None:
                logging.warning(f"  Skipping {sample['id']}: no audio")
                continue

            w = voice.waveform
            if w.dim() == 2:
                if w.shape[0] == 1:
                    w = w.repeat(2, 1)
                w = w.unsqueeze(0)
            elif w.dim() == 3 and w.shape[1] == 1:
                w = w.repeat(1, 2, 1)
            voice = Audio(waveform=w, sampling_rate=voice.sampling_rate)

            # Encode through Audio VAE
            audio_latent = ac(lambda enc: vae_encode_audio(voice, enc, None))

            # Save audio latent
            torch.save({
                "latents": audio_latent.squeeze(0).cpu(),  # [C=8, T, F=16]
                "sample_rate": 16000,
            }, audio_path)

        except Exception as e:
            logging.warning(f"  Skipping {sample['id']}: {e}")
            continue

        if idx % 100 == 0:
            logging.info(f"  Audio encoding: {idx}/{len(samples)}")

    del ac
    torch.cuda.empty_cache()

    # ── Step 3: Create dummy video latents ──
    logging.info("Creating dummy video latents...")
    # Minimal video: 1 frame, 64x64 = 2x2 in latent space
    dummy_video = {
        "latents": torch.zeros(128, 1, 2, 2),
        "num_frames": 1,
        "height": 2,
        "width": 2,
        "fps": 24.0,
    }
    for idx, sample in enumerate(samples):
        gidx = sample["global_idx"]
        latent_path = out / "latents" / f"sample_{gidx:06d}.pt"
        if args.skip_existing and latent_path.exists():
            continue
        torch.save(dummy_video, latent_path)

    # ── Summary ──
    n_audio = len(list((out / "audio_latents").glob("*.pt")))
    n_cond = len(list((out / "conditions").glob("*.pt")))
    n_lat = len(list((out / "latents").glob("*.pt")))
    logging.info(f"\nDone! Output: {args.output_dir}")
    logging.info(f"  audio_latents: {n_audio} files")
    logging.info(f"  conditions:    {n_cond} files")
    logging.info(f"  latents:       {n_lat} files")


if __name__ == "__main__":
    main()
