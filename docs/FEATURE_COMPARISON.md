# Feature Comparison Matrix

## Feature Comparison Matrix

| Feature                      | F5-TTS | ChatterBox | ChatterBox 23L | VibeVoice | Higgs Audio 2 | Higgs Audio v3 | IndexTTS-2 | CosyVoice3 | Qwen3-TTS | Granite ASR | Step Audio EditX | Echo-TTS | Fish Audio S2 Pro | Dots TTS | OmniVoice | MOSS-TTS | RVC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **TTS**                      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **SRT**                      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Voice Conversion**         | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **ASR (Transcribe)**         | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Training**                 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Voice Cloning**            | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (Base model) | ❌ | ✅ | ✅ | ✅ Reference audio plus exact transcript | ✅ | ✅ | ✅ | ⚠️ (needs training) |
| **Reference Transcript**     | **Required** | Not used | Not used | Not used | Optional | Optional | Not used | Conditional | Conditional | N/A | **Required** | Not used | **Required** | Optional | **Required** | Conditional | N/A |
| **Native Multi-Speaker**     | ❌ | ❌ | ❌ | ✅ (Base only, Kugel uses fallback) | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ (Plus variant speaker attribution / diarization) | ❌ | ❌ | ✅ Native dialogue is default; optional custom mode generates each [Character] segment independently as local speaker 0 | ❌ | ❌ | ✅ (TTSD v1.0; 1-5 speakers) | ❌ |
| **Emotion Control**          | ❌ | ❌ | ⚠️ (v2 tags - doesn't work) | ❌ | ⚠️ (via prompt) | ✅ (native inline tags) | ✅ (8 emotions) | ⚠️ (via instruct) | ⚠️ (via instruct) | ❌ | ✅ (14 emotions) | ❌ | ✅ Free-form inline natural-language tags | ❌ | ⚠️ (voice-design instruct + inline non-verbal tags) | ❌ | ❌ |
| **Native Long-form**         | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Configurable 4K-32K native context; suite text chunking is bypassed | ❌ | ✅ (uses upstream audio_chunk_duration / audio_chunk_threshold orchestration; bypasses suite char-based chunk splitting) | ✅ (TTSD/Delay long-form; use chunk orchestration for very long inputs) | N/A |
| **Community Finetunes**      | ✅ | ✅ | ✅ | ✅ KugelAudio, Hindi | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (LoRA adapter inference supported; initial integrated LoRA training support added for MOSS-TTS Delay 8B) | ✅ |
| **VRAM Efficient**           | ✅ | ✅ | ✅ | ⚠️ (5-18GB) | ⚠️ (9GB) | ⚠️ (~8-10GB) | ⚠️ (9-12GB) | ✅ (5.4GB) | ✅ (3-6GB) | ✅ (~4.6GB) | ⚠️ (7GB) | ⚠️ (~7GB total) | ⚠️ 8K context measured at ~15.2GB BF16, ~11.2GB FP8, ~11.2GB BNB INT8, or ~8.9GB BNB NF4; BF16 codec and activations; BNB is a load-time option for the official checkpoint | ⚠️ (main env works; 2B-class model is not lightweight) | ⚠️ (main env works; 2B-class model is not lightweight) | ⚠️ (Local 1.7B smaller; tokenizer is large) | ✅ |
| **Speed/Performance**        | ✅ Very Fast | ✅ Fast | ✅ Fast | ⚠️ | ⚠️ | ⚠️ CUDA recommended | ⚠️ | ✅ Fast | ⚠️ | ⚠️ Moderate | ⚠️ | ✅ Fast (diffusion, realtime-capable) | ✅ Main-environment subprocess with reliable teardown; local compile measurements: ~40 it/s BF16, ~11.8 it/s NF4 at ~8.9GB VRAM, and ~3.7 it/s INT8 at ~11.2GB VRAM; quality comparison pending | ⚠️ Moderate; mf variant is faster | ✅ Fast; upstream reports sub-realtime RTF | ✅ Fast with CUDA/FlashAttention | ✅ Fast |
| **No Narrator Required**     | ❌ | ✅ (default speaker) | ✅ (default speaker) | ✅ (zero-shot / default speaker) | ✅ (basic TTS if no narrator/reference is provided) | ✅ (zero-shot) | ❌ | ✅ (cross-lingual or instruct mode) | ✅ (Base default voice or CustomVoice presets) | N/A | ❌ | ❌ | ✅ Reference audio is optional | ✅ (default speaker) | ✅ (native default voice; instruct also works without narrator) | ✅ (direct TTS and prompt-only generation) | N/A |

**Reference transcript notes:**

- **Required** means the suite requires an exact transcript when cloning from reference audio.
- **Conditional** — CosyVoice3: required for zero-shot mode; cross-lingual and instruct modes do not use it.
- **Conditional** — Qwen3-TTS: required for full Base-model voice cloning; x-vector-only and preset voices do not use it.
- **Conditional** — MOSS-TTS: required for cloned speakers in MOSS-TTSD dialogue; regular MOSS-TTS reference audio does not use it.
- **Optional** means the engine accepts audio-only cloning but can use matching reference text when provided.