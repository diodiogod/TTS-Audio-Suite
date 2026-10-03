# Step Audio EditX checkpoints

Choose the checkpoint in the existing **model_path** dropdown of **⚙️ Step Audio EditX Engine**. No widget was added or reordered, so saved device, precision, quantization, generation, and runtime values keep their positions.

| Model choice | Checkpoint | Download revision |
|---|---|---|
| `Step-Audio-EditX-2026-01-23` | Updated weights and expanded sound tags; default for new engine nodes | `5fe2f8a05c2353301ad47d3c1747b262115da138` |
| `Step-Audio-EditX-2025-11-28` | Legacy weights | `7f3de603ae46c96dff6f06f47b1d5a45aabd34fe` |
| `Step-Audio-EditX` | Compatibility choice for an existing unversioned installation | Existing files are retained; a missing installation downloads the January checkpoint |
| `local:...` | Explicitly selected local model | Existing files are retained |

Dates identify the weight uploads. The January release was announced on January 29; its pinned revision also includes subsequent configuration fixes. These are versions of the same 3B model architecture, not official “v1/v2” names.

Sources: [official model](https://huggingface.co/stepfun-ai/Step-Audio-EditX), [November snapshot](https://huggingface.co/stepfun-ai/Step-Audio-EditX/tree/7f3de603ae46c96dff6f06f47b1d5a45aabd34fe), [upstream release notes](https://github.com/stepfun-ai/Step-Audio-EditX#readme).

## Existing workflows and storage

- Existing `Step-Audio-EditX` and `local:...` selections remain valid. The suite does not replace their existing weights, rename their folders, or infer their checkpoint. The existing compatibility normalization of `config.json` still applies.
- Dated models download lazily into separate folders under `ComfyUI/models/TTS/step_audio_editx/`, respecting `extra_model_paths.yaml`.
- Selecting both checkpoints stores both sets of model files. The second LLM checkpoint requires approximately another 7 GB.
- Model and edit-result caches distinguish the selected checkpoint.
- The shared Transformers 4 runtime remains recommended. Selecting the updated standard checkpoint does not switch to vLLM or to the separate upstream AWQ checkpoint. The suite's `int4` option remains bitsandbytes NF4.

## Sound tags

Use the **2026-01-23** checkpoint for expanded sound tags. The November checkpoint is available to preserve older behavior; selecting it does not add the new learned sounds.

Existing spellings remain supported:

`<Breathing>`, `<Laughter>`, `<Sigh>`, `<Uhm>`, `<Surprise-oh>`, `<Surprise-ah>`, `<Surprise-wa>`, `<Confirmation-en>`, `<Question-ei>`, `<Dissatisfaction-hnn>`.

The January vocabulary also exposes:

`<inhale>`, `<exhale>`, `<laugh>`, `<chuckle>`, `<clears throat>`, `<snort>`, `<giggle>`, `<cough>`, `<breath>`, `<Surprise-yo>`, `<Question-ah>`, `<Question-en>`, `<Question-yi>`, `<Question-oh>`.

Input is case-insensitive. `<clears_throat>` is an alias for `<clears throat>`; both become `[clears throat]` in the model instruction. Expanded tags support the existing iteration and pipe syntax, for example:

```text
[Alice] That was funny <giggle:2>.
[Bob] Let me explain <clears_throat:1|style:serious>.
```

For ChatterBox v2/v3, use an explicit iteration to request Step editing: `<giggle:1>`. Bare `<giggle>`, `<inhale>`, `<exhale>`, and `<cough>` retain their ChatterBox-native meaning. CosyVoice's native sound tags remain native as well.

## Audio Editor and other engines

Connect the dated **Step Audio EditX Engine** output to **🎨 Step Audio EditX - Audio Editor** to choose a checkpoint explicitly. In the Audio Editor's transcript, use bare sound tags such as `<giggle>` or `<clears throat>`; its **n_edit_iterations** input controls the number of passes.

Without a connected engine, the Audio Editor prefers an installed January checkpoint, then an existing unversioned installation. If neither exists, it downloads the dated January checkpoint. This same fallback applies to automatic Step editing of other engines' audio. An existing unversioned installation may contain older weights; use the manual Audio Editor with a dated engine for explicit control.

See [the inline tag guide](INLINE_EDIT_TAGS_USER_GUIDE.md) for position, iteration, restoration, language, and duration behavior.
