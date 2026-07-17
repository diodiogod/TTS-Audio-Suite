# 🌩️ Sound Effects Guide

The `🌩️ Sound Effects` node generates non-speech audio from a written description. It works with any connected engine that advertises sound-effect support.

## Engines

| Model | Engine node | Notes |
|---|---|---|
| MOSS-SoundEffect v1 | `⚙️ MOSS-TTS Engine` | Autoregressive MOSS 8B model |
| MOSS-SoundEffect v2 | `⚙️ MOSS SoundEffect v2 Engine` | 48 kHz diffusion model; up to 30 seconds per native generation |

Connecting a speech-only engine stops with a user-facing compatibility error.

## Basic workflow

1. Select a sound-effect model on its engine node.
2. Connect the engine to `🌩️ Sound Effects`.
3. Describe the sound rather than words to be spoken.
4. Choose the duration and seed, then queue the workflow.

Example:

```text
Heavy rain hitting a metal rooftop, distant rolling thunder, occasional wind gusts.
```

Descriptions generally work best when they state the source, environment, distance, texture, and progression of the sound.

## Timeline segments

Separate descriptions with parameter tags to generate multiple segments and concatenate them:

```text
[seconds:4|seed:42] Bright application startup chime. [seconds:2|cfg:5] Low, dark shutdown tone.
```

`duration_seconds` is the default duration for every generated segment. `[seconds:X]` overrides it for the following segment.

Newlines also create segments, as they do in TTS. They are optional because a parameter tag can start another segment on the same line.

## Pauses

Use a standalone pause tag to insert exact silence:

```text
[seconds:4] Startup chime. [pause:1.2] [seconds:2] Shutdown tone.
```

The aliases `[wait:X]` and `[stop:X]` are also accepted. Durations may use seconds or milliseconds:

```text
[wait:500ms]
```

Keep pauses separate from parameter tags:

```text
[pause:1.2] [cfg:7.5] Thunder crack.
```

Do not combine them as `[pause:1.2|cfg:7.5]`.

## Crossfade and long sounds

`crossfade_seconds` overlaps adjacent generated segments to soften their join. Set it to `0` for a hard join.

A pause creates an exact silent boundary, so crossfade is not applied across that pause.

MOSS-SoundEffect v2 has a native 30-second generation limit. Longer requested segments are generated as overlapping chunks, joined with the selected crossfade, and trimmed to the requested duration.

## Per-segment parameters

Common parameters:

| Tag | Engines | Purpose |
|---|---|---|
| `seed` | v1, v2 | Generated variation |
| `seconds` / `duration_seconds` | v1, v2 | Segment duration |
| `temperature` | v1 | Sampling randomness |
| `top_p`, `top_k` | v1 | Sampling limits |
| `repetition_penalty` | v1 | Discourage repetition |
| `duration_tokens` | v1 | Native duration-token control |
| `max_new_tokens` | v1 | Generation token limit |
| `steps` / `inference_steps` | v2 | Diffusion steps |
| `cfg` | v2 | Prompt guidance strength |
| `sigma_shift` | v2 | Flow-matching schedule shift |
| `negative_prompt`, `negative`, `neg` | v2 | Sounds or qualities to discourage |

Example using a negative prompt:

```text
[seconds:8|cfg:5|neg:speech, music] Dense forest ambience with insects and distant birds.
```

Unsupported parameters are ignored with a warning rather than being sent blindly to the engine.

## Seed and cache behavior

- `seed: 0` chooses a random seed.
- Reusing a positive seed with identical settings makes the request repeatable.
- With audio caching enabled, an identical segment and configuration can reuse its generated audio.
- Changing the description, seed, duration, engine configuration, or inline parameters invalidates that cached result.

## MOSS-SoundEffect v2 first-run compilation

The v2 DiT uses `torch.compile`. Its first generation may spend several minutes compiling before progress begins. Compatible compilation artifacts are cached and can be reused across later ComfyUI sessions.

This compile delay is separate from model downloading and normal generation time.

## Related guides

- [Per-Segment Parameter Switching](PARAMETER_SWITCHING_GUIDE.md)
- [Multiline TTS Tag Editor](MULTILINE_TTS_TAG_EDITOR_GUIDE.md)
