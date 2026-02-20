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
└── ChatterBox Official 23-Lang/
    ├── t3_23lang.safetensors
    ├── t3_mtl23ls_v2.safetensors
    ├── grapheme_mtl_merged_expanded_v1.json
    ├── s3gen.pt
    ├── ve.pt
    ├── mtl_tokenizer.json
    └── conds.pt
```

Notes:

- v1 and v2 coexist in one directory.
- Vietnamese (Viterbox) is available as a community finetune option.

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
└── .index/
```

Legacy (still supported):

```text
ComfyUI/models/RVC/
```

Notes:

- Base models auto-download.
- Character `.pth` models can be auto-downloaded defaults or user-provided.
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
