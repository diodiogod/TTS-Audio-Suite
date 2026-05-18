# MOSS-TTS Prompt Fields Guide

## Overview

MOSS-TTS exposes official whole-segment prompt fields through the `⚙️ MOSS-TTS Engine` node.

These are not Step Audio EditX effects and they are not positional insertions. They are passed to the official MOSS prompt structure for the entire generated segment.

Supported official fields:

- `instruction`
- `quality`
- `sound_event`
- `ambient_sound`
- `language`
- `duration_tokens`

## Stability Note

`ambient_sound` should currently be treated as experimental on base `MOSS-TTS` inside this suite.

Reason:

- the official prompt schema exposes `ambient_sound`
- but the clearest public OpenMOSS example we found is in the `MOSS-SoundEffect` documentation path, not as a validated short-utterance base-TTS control

So on `MOSS-TTS` you may get:

- weak or missing ambience
- longer-than-expected generation
- unstable tails if the model is pushed to the token cap

## Important Limitation

MOSS does **not** treat these like exact inline insertions inside the sentence.

For example:

```text
[Alice|sound_event:Laughter] Hello there.
```

That means the whole segment is conditioned with the event. It is **not** positional the way Step Audio EditX paralinguistic tags are.

## Engine Fields

You can set official fields directly on the engine node:

- `instruction`: whole-segment speaking instruction
- `quality`: whole-segment quality/style hint
- `sound_event`: whole-segment event hint such as `Laughter`, `Sigh`, `Breathing`
- `ambient_sound`: whole-segment ambience hint such as `Rain`, `Crowd`, `Forest`

These apply to every generated segment that uses that engine config.

## Per-Segment Override Syntax

For per-segment overrides, use `[]` parameter switching.

Supported forms:

```text
[Narrator|instruction:Speak softly and calmly] Hello there.
[Narrator|quality:Studio recording] Hello there.
[Narrator|sound_event:Laughter] Hello there.
[Narrator|ambient_sound:Rain on window] Hello there.
```

Inside normal character tags, combine them like this:

```text
[Alice|instruction:Speak softly and calmly] Hello there.
[Bob|quality:Telephone call quality|ambient_sound:Office room tone] Hi.
[Alice|sound_event:Laughter] That's funny.
```

These are whole-segment overrides, not exact insertion points.

## Why Not `<>`

MOSS does not have true native inline positioning for these controls.

So `<>` should stay free for real inline post-processing tags such as Step Audio EditX effects. Using `<>` for MOSS prompt fields would blur two different systems and create confusion.

## Examples

### Direct engine fields

Set on the engine node:

- `sound_event = Laughter`
- `quality = Studio recording`

Text:

```text
[Alice] Hello there!
```

Result: the entire Alice segment is conditioned with `Laughter` and `Studio recording`.

### Per-segment `[]` syntax

```text
[Alice|sound_event:Laughter] Hello there!
[Bob|ambient_sound:Rain] Nice to meet you.
```

Result:

- Alice segment uses `sound_event = Laughter`
- Bob segment uses `ambient_sound = Rain`

## Conflicts and Precedence

- If you set an engine field and also override it in `[]`, the `[]` value wins for that segment.
- These MOSS prompt fields are not supposed to use `<>`.

## Native TTSD Dialogue Note

`MOSS-TTSD-v1.0` generates one native dialogue request across the formatted `[S1]...[S5]` conversation.

So these official fields still apply to the whole TTSD request, not to one precise speaker turn inside that request.

## Native TTSD Compatibility Rules

Native TTSD mode is strict in this suite.

If any of these are detected, generation **fails with an explicit error** instead of auto-switching models:

- `pause tags`
- inline Step Audio EditX tags
- per-segment `[]` parameter changes
- more than 5 speakers

This applies to both normal TTS Text and SRT workflows.

Required action when that happens:

- switch to `Custom Character Switching`
- choose a standard MOSS model (`MOSS-TTS-Local-Transformer` or `MOSS-TTS`)

## When To Use Something Else

If you need exact placement of laughter/breathing/cough inside the sentence, use:

- Step Audio EditX inline tags and audio editor workflow
- an engine with true native positional tag support, such as CosyVoice3

Use MOSS prompt fields when you want official MOSS conditioning, not precise insertion control.

If you specifically want generated environmental audio like rain, traffic, or scene beds, `MOSS-SoundEffect` is the more correct OpenMOSS model family target than base `MOSS-TTS`.
