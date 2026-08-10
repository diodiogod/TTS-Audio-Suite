# DramaBox LoRA training

TTS Audio Suite exposes the official DramaBox audio-branch IC-LoRA trainer
through the unified `🎓 Model Training` flow. The bundled scripts are pinned to
the same upstream DramaBox revision as the inference implementation.

See the official DramaBox
[LoRA training guide](https://github.com/resemble-ai/DramaBox#training-a-lora-on-top-of-dramabox)
for the upstream dataset format and training behavior.

## Workflow

1. Build a `⚙️ DramaBox Engine`.
2. Create the dataset either externally or entirely inside ComfyUI:
   `🎞️ Training Clip Staging` → `🧾 DramaBox Dataset Rows`.
3. Connect the resulting manifest to `📦 DramaBox Dataset Prep` and keep
   `dataset_type` set to `manifest`.
4. Provide at least two clips per speaker.
5. Connect the dataset to `🎛️ DramaBox Training Config` and then to `🎓 Model Training`.
6. Select the resulting adapter in the DramaBox engine, or enter its path in the
   advanced LoRA override field.

The dataset node accepts:

- JSONL/JSON manifests with `audio_filepath` (or `audio_path`) and `text` (or
  `transcript`)
- TSV rows with audio path and text
- the official `gemini_synthetic` and `libriheavy` index formats

Manifest rows may include `speaker`, `speaker_id`, `language`, and `duration`.
If `speaker` is omitted, rows are grouped as `speaker_1`. Duration and audio
metadata are measured without loading the waveform into the GPU. The suite
converts all accepted formats into the `~`-delimited speaker index required by
the upstream training loop. Clips are restricted to 2–20 seconds by default.

For an all-ComfyUI dataset, connect one or more `AUDIO` sources to
`🎞️ Training Clip Staging`, then enter one transcript per clip in
`🧾 DramaBox Dataset Rows`. Speaker and language lines are optional; shared
defaults are used when those lines are blank.

### Transcripts and scene descriptions

The official trainer accepts either plain spoken transcripts or the same
scene-style prompt format used for inference. For example, both of these are
valid training text:

```text
This is the spoken sentence.
A woman speaks warmly, "This is the spoken sentence."
```

Use scene descriptions only when they accurately describe the clip. Plain
transcripts remain valid and are the safer choice when no reliable style or
scene annotation is available.

## What training does

The first preprocessing pass uses Gemma and the DramaBox audio VAE to create
cached conditions and audio latents. The training process then attaches a LoRA
to the audio transformer branch. It saves periodic checkpoints and exports the
selected adapter to:

```text
ComfyUI/models/TTS/dramabox/loras/<adapter_name>/
```

The job directory, normalized index, preprocessing cache, progress file, and
logs are stored under:

```text
ComfyUI/output/tts_audio_suite_training/dramabox/
```

`continue_from` is a warm start from an existing LoRA checkpoint; it is not an
exact optimizer-state resume. Use saved checkpoints to compare quality rather
than assuming the last step is best. Optional upstream validation can be
enabled with a `val_config` YAML path, but it launches full DramaBox inference
at each save step. It requires a second GPU: set `validation_gpu` to that
physical CUDA device index. The suite rejects validation on the training GPU
instead of allowing both full model processes to compete for the same VRAM.

DramaBox LoRA inference supports normal transformer precision, `fp8_cast`, and
the optional `torch.compile` path. With normal precision the live adapter is
reversibly merged for fast inference. With FP8 storage the BF16 adapter remains
unmerged above the immutable FP8 base weights, avoiding unsafe mixed-dtype
weight fusion while retaining the main FP8 memory saving.

The base DramaBox runtime is reused when the selected adapter or LoRA strength
changes. Strength updates are applied directly to the live PEFT adapter, while
the generated-audio cache still treats adapter path, file revision, and strength
as distinct generation settings. Replacing an adapter with a different rank may
retrace compiled transformer blocks, but does not reload the base checkpoint.

## CPU-safe preflight

Training and Gemma/VAE preprocessing are GPU workloads. For development or
validation without touching CUDA, enable `dry_run` in the training config and
the dataset node's `dry_run`/`preprocess_now` controls. This writes the
normalized index and official command/config without loading DramaBox weights.
