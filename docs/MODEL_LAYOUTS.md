# Model Folder Layouts

Use this document for folder structures and placement notes.
For repository/source URLs, use [MODEL_DOWNLOAD_SOURCES.md](MODEL_DOWNLOAD_SOURCES.md).

## ChatterBox

Recommended:

```text
ComfyUI/models/TTS/chatterbox/
```

Legacy (still supported):

```text
ComfyUI/models/chatterbox/
```

Required base files:

- `conds.pt`
- `s3gen.pt`
- `t3_cfg.pt`
- `tokenizer.json`
- `ve.pt`

Optional multilingual subfolders:

```text
ComfyUI/models/TTS/chatterbox/
├── English/
├── German/
└── Norwegian/
```

## ChatterBox Official 23-Lang

```text
ComfyUI/models/TTS/chatterbox_official_23lang/
├── ChatterBox Official 23-Lang/
│   ├── t3_23lang.safetensors
│   ├── t3_mtl23ls_v2.safetensors
│   ├── grapheme_mtl_merged_expanded_v1.json
│   ├── s3gen.pt
│   ├── ve.pt
│   ├── mtl_tokenizer.json
│   └── conds.pt
└── russian_text_stresser/
    ├── russian_dict.db
    └── simple_cases.pkl
```

Notes:

- v1 and v2 coexist in one directory.
- Vietnamese (Viterbox) is available as a community finetune option.
- `russian_text_stresser/` is auxiliary Russian-only data for Official 23-Lang stress labeling and downloads on demand.

## F5-TTS

Recommended:

```text
ComfyUI/models/TTS/F5-TTS/
```

Legacy (still supported):

```text
ComfyUI/models/F5-TTS/
```

Typical structure:

```text
ComfyUI/models/TTS/F5-TTS/
├── F5TTS_Base/
│   ├── model_1200000.safetensors
│   └── vocab.txt
├── F5TTS_v1_Base/
│   ├── model_1250000.safetensors
│   └── vocab.txt
└── vocos/
    ├── config.yaml
    └── pytorch_model.bin
```

Notes:

- F5-TTS models auto-download on first use.
- `vocab.txt` is required per model.
- Vocos is optional and also auto-downloadable.

## F5-TTS Voice References

```text
ComfyUI/models/voices/
├── character1.wav
├── character1.reference.txt
├── narrator.wav
└── narrator.txt
```

Requirements:

- WAV, clean speech, 5-30s (24kHz recommended).
- Text must match spoken audio.
- Naming: `name.wav` + `name.reference.txt` (preferred) or `name.txt`.

Inference guidelines:

1. Keep reference audio under ~12s with a little trailing silence.
2. Use uppercase letter-by-letter only when desired.
3. Use spaces/punctuation for explicit pauses.
4. Keep a space after sentence-ending punctuation.
5. For Chinese output, preprocess numbers if needed.

## Higgs Audio 2

```text
ComfyUI/models/TTS/HiggsAudio/
└── higgs-audio-v2-3B/
    ├── generation/
    ├── tokenizer/
    └── voices/
```

Notes:

- Both generation model and tokenizer auto-download.
- Place reference audio/transcription in `voices/`.

## VibeVoice

```text
ComfyUI/models/TTS/VibeVoice/
├── vibevoice-1.5B/
└── vibevoice-7B/
```

Notes:

- Both variants auto-download on first use.
- `vibevoice-7B` uses a community mirror in downloader config.

## RVC

Recommended:

```text
ComfyUI/models/TTS/RVC/
├── *.pth
├── content-vec-best.safetensors
├── rmvpe.pt
├── hubert/
├── pretrained_v2/
│   ├── f0G32k.pth
│   ├── f0D32k.pth
│   ├── f0G40k.pth
│   ├── f0D40k.pth
│   ├── f0G48k.pth
│   └── f0D48k.pth
└── .index/
```

Legacy (still supported):

```text
ComfyUI/models/RVC/
```

Notes:

- Base models auto-download.
- Character `.pth` models can be auto-downloaded defaults or user-provided.
- `pretrained_v2/` is used by the integrated RVC trainer and auto-downloads on first training run.
- Training datasets, logs, progress snapshots, and resumable checkpoints are stored under `ComfyUI/output/tts_audio_suite_training/rvc/`, not inside the custom node repo.
- UVR models are downloaded under `ComfyUI/models/TTS/UVR/` (or legacy `ComfyUI/models/UVR/`).

## IndexTTS-2

```text
ComfyUI/models/TTS/IndexTTS/
├── IndexTTS-2/
└── w2v-bert-2.0/
```

Notes:

- Emotion components and semantic feature models auto-download.

## Step Audio EditX

```text
ComfyUI/models/TTS/step_audio_editx/
├── Step-Audio-EditX/
│   └── CosyVoice-300M-25Hz/
└── FunASR-Paraformer/
```

Notes:

- Main model, tokenizer assets, and speech stack auto-download.

## Higgs Audio v3

```text
ComfyUI/models/TTS/higgs_audio_v3/
└── higgs-audio-v3-tts-4b/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── chat_template.jinja
    ├── model.safetensors
    ├── model.safetensors.index.json
    └── LICENSE
```

Notes:

- Downloads into its own `higgs_audio_v3` folder.
- Requires the main Transformers 5 environment.
- Reference transcript `.txt` files are optional but improve cloning quality.

## CosyVoice3

```text
ComfyUI/models/TTS/CosyVoice/
└── Fun-CosyVoice3-0.5B/
    ├── llm.pt
    ├── llm.rl.pt
    └── shared model files...
```

Notes:

- First selected variant downloads first.
- Shared files are reused across variants.

## Qwen3-TTS and Qwen3-ASR

```text
ComfyUI/models/TTS/qwen3_tts/
├── Qwen3-TTS-12Hz-0.6B-CustomVoice/
├── Qwen3-TTS-12Hz-1.7B-CustomVoice/
├── Qwen3-TTS-12Hz-1.7B-VoiceDesign/
├── Qwen3-TTS-12Hz-0.6B-Base/
├── Qwen3-TTS-12Hz-1.7B-Base/
├── qwen2-audio-encoder/
└── asr/
    ├── Qwen3-ASR-1.7B/
    └── Qwen3-ForcedAligner-0.6B/
```

Notes:

- Only selected variants download.
- Shared tokenizer assets are reused.

## MOSS-TTS

```text
ComfyUI/models/TTS/moss_tts/
├── MOSS-TTS-Local-Transformer/
├── MOSS-TTS-v1.5/
├── MOSS-TTS/
├── MOSS-VoiceGenerator/
├── MOSS-SoundEffect/
├── MOSS-TTSD-v1.0/
├── MOSS-Audio-Tokenizer/
└── loras/
    └── <adapter_name>/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

Notes:

- `MOSS-Audio-Tokenizer` is required by the official TTS and TTSD variants.
- `MOSS-TTS-Local-Transformer` is the smaller 1.7B model.
- `MOSS-TTS-v1.5` is the current 8B delay model with 31-language support.
- `MOSS-TTS` is the legacy official 8B delay model.
- `MOSS-VoiceGenerator` is the 1.7B voice-design provider used by Voice Designer.
- `MOSS-SoundEffect` is the v1 sound-effect checkpoint used through the MOSS-TTS engine and 🌩️ Sound Effects.
- `MOSS-TTSD-v1.0` is the official 8B native multi-speaker dialogue model.
- Integrated training currently exports LoRA adapters into `moss_tts/loras/<adapter_name>/`.
- Training jobs, temporary manifests, and checkpoints are stored under `ComfyUI/output/tts_audio_suite_training/moss_tts/`.

## MOSS-SoundEffect v2

```text
ComfyUI/models/TTS/moss_soundeffect_v2/
└── MOSS-SoundEffect-v2.0/
    ├── model_index.json
    ├── scheduler/
    ├── text_encoder/
    ├── tokenizer/
    ├── transformer/
    └── vae/
```

Notes:

- This is a separate v2 diffusion family, not a MOSS-TTS checkpoint variant.
- It runs in the configured ComfyUI environment; the official Apache-2.0 inference package is bundled without modifying its dependencies.
- The 🌩️ Sound Effects node limits generation to the official 30-second maximum.

## Granite ASR

```text
ComfyUI/models/TTS/granite_asr/
└── granite-4.0-1b-speech/
    ├── config.json
    ├── chat_template.jinja
    ├── model-00001-of-00003.safetensors
    ├── model-00002-of-00003.safetensors
    ├── model-00003-of-00003.safetensors
    └── tokenizer / processor files...
```

Notes:

- Granite downloads into its own `granite_asr` folder.
- `granite-speech-4.1-2b-plus` adds native speaker diarization and native word-level timestamps, but drops Japanese.
- If Granite word timestamps are enabled, it lazily reuses `Qwen3-ForcedAligner-0.6B` from the Qwen ASR folder instead of duplicating that model.

## Echo-TTS

```text
ComfyUI/models/TTS/
├── echo-tts-base/
│   ├── pytorch_model.safetensors
│   └── pca_state.safetensors
└── fish-s1-dac-min/
    └── pytorch_model.safetensors
```

Notes:

- Both components are required and auto-downloaded on first use.
- License: CC-BY-NC-SA (non-commercial).

## Fish Audio S2 Pro

```text
ComfyUI/models/TTS/fish_audio_s2_pro/
├── codec.pth
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
└── tokenizer.json
```

Optional FP8 variant:

```text
ComfyUI/models/TTS/fish_audio_s2_pro_fp8/
├── codec.pth
├── config.json
├── model.safetensors
├── quantization_info.json
└── tokenizer.json
```

The complete official repository metadata and tokenizer files are downloaded alongside these files. License: Fish Audio Research License (non-commercial without a separate commercial license).

The `s2-pro-bnb-int8` and `s2-pro-bnb-nf4` options reuse `fish_audio_s2_pro/` and quantize its official checkpoint while loading. They do not download another model copy and require `bitsandbytes`.

## Dots TTS

```text
ComfyUI/models/TTS/dots_tts/
├── dots.tts-base/
├── dots.tts-soar/
└── dots.tts-mf/
```

Typical checkpoint contents:

```text
ComfyUI/models/TTS/dots_tts/dots.tts-soar/
├── added_tokens.json
├── chat_template.jinja
├── config.json
├── latent_stats.pt
├── llm_config.json
├── merges.txt
├── model.safetensors
├── speaker_encoder.safetensors
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
└── vocoder.safetensors
```

Notes:

- Downloads into a dedicated `dots_tts/` folder.
- Native sample rate is 48kHz.
- Main-environment support works on Transformers 5; on Windows, `normalize_text` falls back to no-op if `WeTextProcessing` is unavailable.

## OmniVoice

```text
ComfyUI/models/TTS/omnivoice/
└── OmniVoice/
```

Main model contents:

```text
ComfyUI/models/TTS/omnivoice/OmniVoice/
├── chat_template.jinja
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── audio_tokenizer/
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
```

Notes:

- Downloads into a dedicated `omnivoice/` folder.
- Native sample rate is 24kHz.
- The main OmniVoice model stays in the main Transformers 5 environment.
- Voice cloning requires reference audio plus explicit reference text in this suite.
