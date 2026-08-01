# TADA Engine Guide

TADA is Hume AI's zero-shot text-to-speech model. TTS Audio Suite exposes it through the unified **Text to Speech** and **SRT** nodes, including character switching, per-segment language and generation parameters, pause tags, caching, interruption, and Clear VRAM.

This integration deliberately describes TADA's expressive capability as **reference-conditioned expression transfer**. The model can carry voice, delivery, timing, and emotion present in the prompt audio into generated speech, but it does not provide a native emotion label, intensity slider, embedding, or proven text-instruction emotion control. It is not direct emotion control.

## Supported models and languages

| Model | Languages | Model weights | Intended use |
|---|---|---:|---|
| `HumeAI/tada-1b` | English | ~3.9GB | Lighter English-only option |
| `HumeAI/tada-3b-ml` | English, Arabic, Chinese, German, Spanish, French, Italian, Japanese, Polish, Portuguese | ~8.9GB | Multilingual voice cloning |

The upstream language code for Chinese is `ch`; the suite presents it as Chinese/`zh` and maps it internally. TADA generates mono audio at 24kHz. Reference audio is resampled to 24kHz when needed.

## First-use requirements

TADA runs only in the suite's **shared Transformers 4 runtime**. The integration installs `hume-tada==0.1.9` there and does not modify the main ComfyUI Transformers 5 environment. TADA's current configuration code is incompatible with that main environment.

Before first use, select TADA in an **⚙️ TADA Engine** node and run a workflow. The suite downloads the selected TADA model, shared codec encoder/decoder, the aligner for the selected language, and tokenizer files into `ComfyUI/models/TTS/tada/`.

Downloads are selective. Choosing the 1B model does not download the 3B model, and changing language downloads only that language's codec aligner. A selected codec set is approximately 2.15GB in addition to the model checkpoint.

### Automatic download

No Hugging Face login is required for the normal automatic path. The suite downloads the three tokenizer files from the ungated `onnx-community/Llama-3.2-1B` redistribution, plus its `LICENSE.txt` and `USE_POLICY.md`. Those runtime files are byte-identical to Meta's official `meta-llama/Llama-3.2-1B` tokenizer and are checked against the official Git blob hashes before use.

The tokenizer remains governed by the Llama 3.2 Community License and Acceptable Use Policy. TADA's use of this component is **Built with Llama**. The full Llama model weights are not downloaded.

### Manual/offline installation

Users accustomed to managing ComfyUI models manually can copy the exact tokenizer and model files into the following layout. Only these files are required at runtime:

```text
ComfyUI/models/TTS/tada/
├── TADA-1B/
│   ├── config.json
│   ├── generation_config.json
│   └── model.safetensors
├── TADA-3B-ML/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors.index.json
│   ├── model-00001-of-00002.safetensors
│   └── model-00002-of-00002.safetensors
├── tada-codec/
│   ├── encoder/config.json
│   ├── encoder/model.safetensors
│   ├── decoder/config.json
│   ├── decoder/model.safetensors
│   ├── aligner/config.json                 # English
│   ├── aligner/model.safetensors
│   ├── aligner-<language>/config.json      # selected non-English language
│   ├── aligner-<language>/model.safetensors
│   └── wav2vec2-large/config.json
└── llama-3.2-1b-tokenizer/
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

Install only the selected model directory. For non-English generation, `<language>` is one of `ar`, `ch`, `de`, `es`, `fr`, `it`, `ja`, `pl`, or `pt`. Existing complete files are detected and are not downloaded again. The full Llama model weights and Wav2Vec weights are not required.

## Reference voice requirements

Every generation in this integration requires:

- a clean reference speech clip; and
- an exact transcript of the speech in that clip.

The transcript must match what is spoken, including word order. A mismatched transcript weakens alignment and cloning quality. TADA's official notebook can invoke an English Parakeet ASR model when text is omitted, but the suite does not do that: it would add a hidden ~4.25GB English-only dependency and would be unsuitable for multilingual references. Supply the exact text explicitly or save it beside a character voice as `.reference.txt`/`.txt`.

Expression comes from the reference recording. Use a prompt that actually contains the desired energy, pace, emotion, and speaking style. There is no emotion dropdown because adding one would be a fake control unsupported by the model API.

## Engine controls

Choose the model variant, device, language, and dtype on the engine node. `auto` is the default for device and dtype, and English is the default language. The engine node and unified TTS nodes expose these generation controls:

| Control | Suite default | Effect |
|---|---:|---|
| `seed` | unified-node value | Reproducible stochastic generation |
| `acoustic_cfg_scale` | `1.6` | Acoustic classifier-free guidance strength |
| `duration_cfg_scale` | `1.0` | Duration guidance strength |
| `cfg_schedule` | `cosine` | Guidance schedule: `constant`, `linear`, or `cosine` |
| `num_flow_matching_steps` | `10` | More ODE steps trade speed for refinement |
| `noise_temperature` | `0.9` | Initial flow-matching noise scale |
| `time_schedule` | `logsnr` | ODE schedule: `uniform`, `cosine`, or `logsnr` |
| `negative_condition_source` | `negative_step_output` | Source used for negative CFG conditioning |
| `speed_up_factor` | `0.0` (disabled) | Native two-pass duration scaling; values above 0 add a second generation pass, and values above 1 speak faster |
| `num_transition_steps` | `5` | Prompt-to-generated-speech transition overlap |

Defaults follow the official implementation. Change one parameter at a time: CFG, noise, flow steps, and transition handling interact, and higher values are not automatically better. `speed_up_factor` is not a cheap playback-rate effect; it reruns generation with scaled predicted durations.

Per-segment tags can override supported parameters and language in unified Text or SRT prompts. The 1B model rejects non-English language selection instead of silently generating with the wrong model.

## Long text and SRT

TADA has no native long-form or streaming mode in this integration. The unified Text node uses the suite's sentence-aware chunking and combines the resulting 24kHz clips. The SRT processor generates each subtitle independently and uses the suite's timing modes and overlap fallback.

The official model was trained on short clips (up to roughly 30 seconds), so long uninterrupted generations can drift. Prefer normal chunking or subtitle-sized segments. Native batch inference, streaming, speech continuation, random-voice generation, text-sampling controls, multiple-candidate scoring, and speaker verification are not exposed in the first integration. Text sampling is irrelevant to normal suite TTS because the exact target text is supplied.

## Model layout

```text
ComfyUI/models/TTS/tada/
├── TADA-1B/                  # downloaded only if selected
├── TADA-3B-ML/               # downloaded only if selected
├── tada-codec/
│   ├── encoder/
│   ├── decoder/
│   ├── aligner[-<language>]/ # selected language only
│   └── wav2vec2-large/       # aligner config only, not model weights
└── llama-3.2-1b-tokenizer/   # tokenizer files only
```

## Licensing and upstream sources

- Official implementation: [HumeAI/tada](https://github.com/HumeAI/tada) — MIT code license.
- Model checkpoints: [HumeAI/tada-1b](https://huggingface.co/HumeAI/tada-1b), [HumeAI/tada-3b-ml](https://huggingface.co/HumeAI/tada-3b-ml), and [HumeAI/tada-codec](https://huggingface.co/HumeAI/tada-codec).
- Model weights and tokenizer are subject to the Meta Llama 3.2 Community License. Commercial use is conditional on those terms. Applications distributing or presenting the model must follow the license's attribution requirements, including the required **Built with Llama** notice where applicable.

TTS Audio Suite does not replace or relax upstream license terms. Review them before redistributing models or generated products.
