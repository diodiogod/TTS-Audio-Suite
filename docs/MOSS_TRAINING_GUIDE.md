# MOSS Training Guide

This guide explains the **practical dataset meaning** of the current MOSS training nodes in TTS Audio Suite.

Use this if `🧾 MOSS Dataset Rows` feels unclear.

## Current Training Scope

Current first training slice supports:

- **MOSS-TTS 8B v1.0 and v1.5 (Delay)**
- **LoRA adapter training**

The model selected on the connected MOSS engine is used for dataset preparation and training. Prepare the dataset again after switching between v1.0 and v1.5.

It does **not** currently support:

- Local 1.7B training
- TTSD training
- full-parameter training from the node path

## Normal Training Flow

Current ComfyUI flow:

1. `🎞️ MOSS Clip Staging`
2. `🧾 MOSS Dataset Rows`
3. `📦 MOSS Dataset Prep`
4. `🎛️ MOSS Training Config`
5. `🎓 Model Training`

## The Important Fields

### `text_lines`

This is the easiest one:

- one line per staged clip
- each line is the **transcript/text of that clip**

Example:

- `clip001` audio says: `Hello there.`
- `clip002` audio says: `How are you?`

Then:

```text
Hello there.
How are you?
```

That is the normal default dataset shape:

- `audio + transcript`

For many voice LoRAs, this is enough.

### `reference_clip_lines`

This is **advanced**. Most users should leave it blank at first.

It means:

- which other staged clip should be used as the **reference audio** for that row

That teaches the model a different pattern:

- `text` says **what to say**
- `reference audio` says **whose voice/style to follow**
- `target audio` is the **correct output**

Example:

- `clip001` = Bob saying: `My name is Bob.`
- `clip002` = Bob saying: `Welcome to the show.`

Row 2 can be:

- target audio = `clip002`
- text = `Welcome to the show.`
- reference = `clip001`

That teaches:

- “use `clip001` as the reference voice, then produce the target line from `clip002`”

## When To Ignore `reference_clip_lines`

Ignore it for:

- first MOSS LoRA tests
- single-speaker voice LoRAs
- fixed-voice adaptation
- plain language/domain adaptation

That means:

- just use `audio + transcript`
- maybe also use `instruction`

This is the best default.

## When To Use `reference_clip_lines`

Use it only if you deliberately want **reference-conditioned training**.

That means teaching the model:

- how to use a separate reference clip for voice cloning
- how to follow reference style/prosody
- how to pick speaker identity from reference audio

This makes more sense for:

- reference-conditioned voice cloning behavior
- multi-speaker conditioning datasets
- style-following datasets

## How A Model Learns Voice Cloning

Think of it this way:

- if a TTS model only sees `audio + transcript`, it learns how to do TTS for the voices/styles present in the data
- if a TTS model also sees `ref_audio`, it can learn a different pattern:
  - text says **what to say**
  - reference audio says **whose voice/style to imitate**
  - target audio is the correct output

That is how a model can learn **reference-conditioned voice cloning**, not just plain TTS.

So if your goal is to fine-tune that specific ability:

- “give the model a reference clip and make it speak in that voice/style”

then `ref_audio` is when this field makes sense.

If your goal is only:

- “make the model speak in this one fixed voice better”

then `ref_audio` usually does not matter much, and plain `audio + transcript` is the better starting point.

## `instruction` vs `ref_audio`

These are different tools.

### Use `instruction` for explicit text labels

Examples:

- `speaking angrily`
- `whispering`
- `calm narration`

This is usually clearer when you want to label the desired behavior directly.

### Use `quality` for recording/presentation quality

Examples:

- `telephone call quality`
- `studio recording`
- `noisy room recording`

This is better when the label is about how the audio sounds as a recording or presentation format, not about acting/performance.

### Use `ref_audio` for example audio conditioning

This is for:

- “sound like this other clip”
- “follow this speaker/style from audio”

So for many datasets:

- `instruction` is the simpler control
- `ref_audio` is the more advanced control

## Practical Recommendations

### Best first test

Use:

- staged clips
- correct transcripts
- no reference clips
- optional `instruction` only if clearly useful

In other words:

- `text_lines`: yes
- `reference_clip_lines`: leave blank

### If training a single person's voice

Start with:

- `audio + transcript`

Do **not** add `ref_audio` unless you specifically want to teach reference-conditioned cloning behavior.

### If training language adaptation like the Norwegian LoRA

Start with:

- `audio + transcript`

That is the normal base case.

You can still use reference audio later at inference time, because MOSS supports optional reference-based cloning, but the training dataset does not need to be built around `ref_audio`.

## Validation In Plain Terms

`📦 MOSS Dataset Prep` also mentions validation.

That means:

- a small holdout part of the dataset used only to check training progress
- not used to update weights

If you do not have a separate validation manifest:

- leave `validation_source` blank
- let the node auto-split the main dataset

## Short Version

If you want the least confusing starting point:

- use `🎞️ MOSS Clip Staging`
- use `🧾 MOSS Dataset Rows`
- fill only `text_lines`
- leave `reference_clip_lines` blank
- train the LoRA

That is the correct first path for most users.
