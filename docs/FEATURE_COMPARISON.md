# Feature Comparison Matrix

## Feature Comparison Matrix

| Feature                      | F5-TTS | ChatterBox | ChatterBox 23L | VibeVoice | Higgs Audio 2 | IndexTTS-2 | CosyVoice3 | Qwen3-TTS | Step Audio EditX | RVC |
|---|---|---|---|---|---|---|---|---|---|---|
| **Voice Cloning**            | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (Base model) | ✅ | ⚠️ (needs training) |
| **Native Multi-Speaker**     | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Voice Conversion**         | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |
| **ASR (Transcribe)**         | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Emotion Control**          | ❌ | ❌ | ⚠️ (v2 tags - doesn't work) | ❌ | ⚠️ (via prompt) | ✅ (8 emotions) | ⚠️ (via instruct) | ⚠️ (via instruct) | ✅ (14 emotions) | ❌ |
| **Native Long-form (90min)** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | N/A |
| **Community Finetunes**      | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **VRAM Efficient**           | ✅ | ✅ | ✅ | ⚠️ (5-18GB) | ⚠️ (9GB) | ⚠️ (9-12GB) | ✅ (5.4GB) | ✅ (3-6GB) | ⚠️ (7GB) | ✅ |
| **Speed/Performance**        | ✅ Very Fast | ✅ Fast | ✅ Fast | ⚠️ | ⚠️ | ⚠️ | ✅ Fast | ⚠️ | ⚠️ | ✅ Fast |