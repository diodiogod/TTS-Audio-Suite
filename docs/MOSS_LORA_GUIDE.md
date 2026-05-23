# MOSS LoRA Guide

This guide defines the local adapter layout and loading behavior for **MOSS-TTS LoRA adapters** in TTS Audio Suite.

## Local Folder Layout

MOSS LoRA adapters should live under:

```text
ComfyUI/models/TTS/moss_tts/loras/
```

Each adapter must have its **own folder**:

```text
ComfyUI/models/TTS/moss_tts/loras/
  MyNorwegianLoRA/
    adapter_config.json
    adapter_model.safetensors
```

Recommended structure:

```text
<adapter_name>/
  adapter_config.json
  adapter_model.safetensors
```

Also accepted:

```text
<adapter_name>/
  adapter_config.json
  adapter_model.bin
```

## Required Files

Each adapter folder must contain:

- `adapter_config.json`
- one adapter weights file:
  - `adapter_model.safetensors` preferred
  - or `adapter_model.bin`

Extra files are allowed, for example:

- `README.md`
- training metadata
- tokenizer notes

## What Is Not Supported

Do not place loose adapter files directly inside:

```text
ComfyUI/models/TTS/moss_tts/loras/
```

This is **not** a supported layout:

```text
loras/
  adapter_config.json
  some_weights.safetensors
```

This is also **not** supported:

```text
loras/
  MyLoRA.safetensors
```

One adapter must always be one folder.

## Engine Node Behavior

The MOSS engine node supports two adapter paths:

1. **Local LoRA dropdown**
   - lists valid adapter folders discovered under `models/TTS/moss_tts/loras`

2. **Advanced override field**
   - accepts:
     - a local adapter folder path
     - a Hugging Face repo id

If you enter a Hugging Face repo id, TTS Audio Suite installs it into:

```text
ComfyUI/models/TTS/moss_tts/loras/<owner__repo>/
```

Then it loads the adapter from that local folder.

## Why This Layout Exists

TTS Audio Suite does **not** use Hugging Face cache as the final managed storage location for MOSS LoRAs.

This layout keeps adapters:

- visible in the MOSS dropdown
- stored inside the normal TTS model tree
- ready for future training output
- separate from ComfyUI diffusion LoRA folders

## Training Output Standard

Future MOSS training should write adapters in exactly this format:

```text
ComfyUI/models/TTS/moss_tts/loras/<run_name>/
  adapter_config.json
  adapter_model.safetensors
```

That keeps training, manual installs, and inference all using the same contract.
