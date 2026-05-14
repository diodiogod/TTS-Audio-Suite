# 01 - First Ask The LLM To Research The Official Model

Do this before implementation.

The official model implementation is the source of truth. Not this suite, not another ComfyUI node pack, not a random demo script. The LLM must understand what the model actually supports before it writes code.

## What The User Should Ask

Copy this to the LLM:

```text
Do not write code yet.

Research the official implementation for this model/engine:

[PASTE OFFICIAL REPO, MODEL CARD, PAPER, OR PACKAGE LINK HERE]

Produce a capability report for TTS Audio Suite. Read the official repo, model card, examples, docs, paper if useful, license, dependency files, and inference scripts.

The report must answer:

1. What task does this model actually support?
   - TTS (in this suite, TTS means both Unified TTS Text and Unified SRT TTS)
   - Voice Changer / voice conversion
   - ASR
   - Audio editing
   - Other special feature

2. What native parameters does the official implementation expose?

3. What features are real native model capabilities?

4. What features are not supported natively?

5. What model files are required, and where do they come from?

6. What voice/reference input mode does it require?
   - Reference audio only
   - Reference audio plus exact transcript/reference text
   - Either reference-audio-only or reference-audio-plus-text modes
   - No reference audio

7. What sample rate, channel layout, and audio tensor/file format does it expect?

8. What dependencies and versions does it require?

9. Does it conflict with common packages already used by TTS Audio Suite?

10. What license applies to the code and weights?

11. Does it need only unified nodes, or does it need a special node?

Do not start implementation until this report is complete.
```

## What The Capability Report Must Decide

The report must decide the actual scope:

- TTS integration, which includes both Unified TTS Text and Unified SRT TTS.
- Voice Changer / voice conversion.
- ASR / transcription.
- Special node for a native unique feature.
- No integration, if the model is not suitable.

Do not ask whether SRT is natively supported by the model. For TTS engines, SRT support is implemented by TTS Audio Suite by generating subtitle text segments and assembling timing with the suite timing systems.

If the model has a native feature that does not fit existing unified nodes, say so. Do not hide it inside a generic node if it needs a special UI.

## Native Features Come First

Implement native official features first.

Good native examples:

- Official speaker/reference audio support.
- Official language parameter.
- Official emotion/style controls.
- Official voice conversion mode.
- Official ASR timestamps.

Bad unsupported examples unless discussed first:

- Fake speed control by time-stretching output.
- Fake emotion by post-processing audio.
- Fake language dropdown when the model is English-only.
- Fake SRT timing controls that bypass the suite timing engine or imply unsupported model behavior.

Extra suite-added features can be useful, but they need maintainer discussion first.

## Minimum Research Checklist

The LLM should inspect:

- Official README.
- Model card.
- Inference examples.
- CLI or demo scripts.
- Python package metadata.
- Requirements files.
- Download instructions.
- License files.
- Hugging Face or ModelScope file layout.
- Open issues if the model has known broken behavior.

If any of these cannot be found, the report must say that clearly instead of guessing.

---

Navigation: [Guide Hub](README.md) | Next: [02 - Check Existing ComfyUI Implementations](02_CHECK_EXISTING_COMFYUI_IMPLEMENTATIONS.md)
