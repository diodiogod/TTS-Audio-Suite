# Feature Comparison Matrix

## Feature Comparison Matrix

| Feature                      | F5-TTS | ChatterBox | ChatterBox 23L | VibeVoice | Higgs Audio 2 | Higgs Audio v3 | IndexTTS-2 | CosyVoice3 | Qwen3-TTS | Granite ASR | Step Audio EditX | Echo-TTS | MOSS-TTS | RVC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **TTS**                      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **SRT**                      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Voice Conversion**         | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **ASR (Transcribe)**         | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Training**                 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Voice Cloning**            | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (Base model) | ❌ | ✅ | ✅ | ✅ | ⚠️ (needs training) |
| **Native Multi-Speaker**     | ❌ | ❌ | ❌ | ✅ (Base only, Kugel uses fallback) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (TTSD v1.0; 1-5 speakers) | ❌ |
| **Emotion Control**          | ❌ | ❌ | ⚠️ (v2 tags - doesn't work) | ❌ | ⚠️ (via prompt) | ✅ (native inline tags) | ✅ (8 emotions) | ⚠️ (via instruct) | ⚠️ (via instruct) | ❌ | ✅ (14 emotions) | ❌ | ❌ | ❌ |
| **Native Long-form**         | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (TTSD/Delay long-form; use chunk orchestration for very long inputs) | N/A |
| **Community Finetunes**      | ✅ | ✅ | ✅ | ✅ KugelAudio, Hindi | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (LoRA adapter inference supported; initial integrated LoRA training support added for MOSS-TTS Delay 8B) | ✅ |
| **VRAM Efficient**           | ✅ | ✅ | ✅ | ⚠️ (5-18GB) | ⚠️ (9GB) | ⚠️ (~8-10GB) | ⚠️ (9-12GB) | ✅ (5.4GB) | ✅ (3-6GB) | ✅ (~4.6GB) | ⚠️ (7GB) | ⚠️ (~7GB total) | ⚠️ (Local 1.7B smaller; tokenizer is large) | ✅ |
| **Speed/Performance**        | ✅ Very Fast | ✅ Fast | ✅ Fast | ⚠️ | ⚠️ | ⚠️ CUDA recommended | ⚠️ | ✅ Fast | ⚠️ | ⚠️ Moderate | ⚠️ | ✅ Fast (diffusion, realtime-capable) | ✅ Fast with CUDA/FlashAttention | ✅ Fast |