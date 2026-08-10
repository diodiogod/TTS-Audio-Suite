# DramaBox Prompting Guide

DramaBox is an English expressive TTS engine. It accepts ordinary narration,
dialogue in quotation marks, and natural-language stage directions in one
prompt.

## Basic prompts

Use quoted text for speech and surrounding prose for delivery:

```text
A tired detective speaks quietly in a rain-soaked office. "I knew this case would find me again."
```

`prompt_template` defaults to `"{seg}"`. `{seg}` is replaced by the current
plain fragment, so the default marks the whole fragment as literal spoken
dialogue. For example, `Hello.` becomes `"Hello."`. Clear the template field to
send plain text unchanged.

Customize the template to add delivery context, for example
`A man speaks warmly, "{seg}"`. Quote-only input is normalized without adding
another pair of quotes. Complete scene prompts with directions outside their
quotation marks remain unchanged. Every non-empty template must contain `{seg}`.
If it is omitted accidentally, TTS Audio Suite warns once and appends
`"{seg}"` automatically instead of failing generation.

For a one-segment override, `prompt_template` (or its `template` alias)
automatically enables templating for that segment and then reverts to the node
setting:

```text
[Narrator|template:A woman whispers, "{seg}"] This line uses a custom wrapper.
```

DramaBox can render non-verbal and delivery cues when they are described
naturally:

```text
She tries to stay serious, then breaks into a short laugh. "That is the worst excuse I have ever heard." She sighs and continues more gently. "But I believe you."
```

Write one continuous scene paragraph. DramaBox does not require newlines as
prompt syntax. In TTS Audio Suite, an untagged newline starts another generated
segment, so use prose action directions and quoted dialogue in the same
paragraph when they should remain one coherent DramaBox scene.

Do not use ChatterBox V2 special tokens such as `[giggle]`. DramaBox was
trained for prose-style scene direction, not that token vocabulary.

## Voice references

Reference audio is optional. Connect narrator audio or use a character voice
file to clone its speaker and delivery. Upstream uses the first 10 seconds, so
a clean single-speaker clip is the useful input; a transcript is not required.

Without a reference, DramaBox uses its built-in voice behavior.

DramaBox can occasionally produce a near-silent sample for a particular
combination of reference audio, reference duration, generation duration, and
seed. TTS Audio Suite checks the decoded waveform and prints a warning when both
its RMS and peak levels are conservatively near silence. The audio is preserved;
the suite does not retry or change parameters automatically. Try another
generation duration, reference duration/audio, guidance setting, or seed for
the affected segment. A different seed can help some combinations but is not a
guaranteed fix.

The warning is also propagated to node outputs. TTS Text includes affected
segments in `generation_info`. TTS SRT marks affected subtitle numbers in
`timing_report`, including the parameters that may be worth testing for that
segment.

## Character and pause tags

TTS Audio Suite character tags still work. Each tagged character is generated
as a separate DramaBox segment:

```text
[Alice] "We should leave now."
[Bob] He answers without looking up. "Give me one minute."
[pause:0.8]
[Alice] "You said that five minutes ago."
```

Suite pause tags create exact silence outside the model. Natural pauses inside
a spoken scene are better expressed in the prose prompt.

## Engine controls

- `cfg_scale`: text/prompt guidance. Official default: `2.5`.
- `stg_scale`: skip-token guidance. Official default: `1.5`.
- `duration_multiplier`: scales the estimated speaking duration. Official
  default: `1.1`.
- `gen_duration`: explicit generated-audio duration from `0` to `60` seconds.
  `0` keeps automatic prompt-based estimation.
- `ref_duration`: uses the first `3` to `30` seconds of a voice reference.
  The default is `10`; audio later in the source file is ignored.
- `rescale_scale`: CFG latent rescaling. Use `auto` or a fixed value from
  `0` to `1`.
- `watermark`: enables the optional official Perth output watermark. It is off
  by default and requires Perth.
- `seed`: supplied by the unified TTS Text or SRT node.

Segment overrides support `seed`, `cfg_scale`, `stg_scale`, and
`duration_multiplier`, `gen_duration`, `ref_duration`, and `rescale_scale`.
Watermarking remains a whole-engine setting rather than a segment override.

DramaBox performs its own duration-aware long-form chunking. The suite does
not split a DramaBox scene by character count before passing it to the model.
Automatically estimated scenes above 45 seconds use text chunking. A nonzero
`gen_duration` remains one native generation so its explicit 0–60 second
target is preserved.

The unified SRT node's **Native Duration Targeting** option passes each
subtitle's duration to DramaBox before final timing assembly. For subtitles
containing multiple character or pause-separated fragments, the available
speech time is allocated proportionally after explicit pause durations and
inline `gen_duration` overrides are accounted for. The selected SRT timing
mode still performs its normal final correction.

## Negative Prompt and Segment Switching

DramaBox uses CFG and exposes its negative prompt in the engine node. The
default discourages robotic, distorted, noisy, muffled, unclear, and monotone
speech. Override it for one character segment with:

```text
[Alice|negative:robotic, muffled] "Keep this line clean and intimate."
[Bob|neg:noise, static] "This line uses a different negative prompt."
```

The segment override ends at the next character tag.

## Memory and Performance

- `fast` keeps all components on CUDA for the fastest repeated generation.
- `staged` is an experimental strategy for lowering peak VRAM. It loads and
  releases Gemma, the voice encoder, and audio decoder by stage, at the cost of
  reloading them for each generated segment or long-form chunk.
- `sequential` is a more aggressive experimental strategy for lowering peak
  VRAM. It additionally keeps the diffusion transformer in system RAM while
  another major stage uses CUDA. It transfers the transformer for every
  generated segment or long-form chunk and is therefore substantially slower.
  Actual peak usage varies with the environment, generation settings, and
  other loaded components; no minimum GPU size is guaranteed.
  System RAM must hold the offloaded transformer (about 3.4GB with FP8 or
  6.6GB without it).
- `fp8_cast` uses the official LTX FP8 transformer weight-storage policy and
  upcasts linear weights during inference. It can lower VRAM and may be slower.
- `compile_model` compiles the diffusion transformer blocks with DramaBox's
  bundled LTX compilation path. The first generation can take substantially
  longer while kernels compile; later denoising may be faster.

## Requirements and license

The full download is approximately 16.4GB and the official runtime requires
an NVIDIA CUDA GPU. Fast mode targets roughly 24GB VRAM; the experimental
staged modes can run with less memory at a speed cost. Output is 48kHz stereo.
The optional official Perth watermark is applied only when enabled and the
dependency is available.

DramaBox uses the LTX-2 Community License. Entities with at least USD 10
million in annual revenue require a separate paid commercial license. Review
the bundled license before production use.
