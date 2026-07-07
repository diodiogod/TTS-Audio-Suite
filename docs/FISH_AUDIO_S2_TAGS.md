# Fish Audio S2 Pro Inline Tags

Use the suite's public angle-bracket syntax for Fish's free-form instructions:

```text
<whisper in small voice>Hello. <professional broadcast tone>Good evening.
```

The Fish adapter converts these to native `[...]` instructions only when text reaches the Fish engine. Do not write Fish tags with square brackets in suite text because `[Character]` is reserved for character switching.

Fish also supports normal suite tags such as `[pause:500ms]`, character switching, and per-segment parameters. Those are processed by the suite before Fish inference.

## Language Prompting

Fish has no native language dropdown or parameter. The engine can instead prepend a natural inline instruction from the suite's resolved segment language:

- `language_prompting = Auto Inline Tag`: non-English resolved languages become tags such as `<German>` or `<French>`
- `language_prompting = Off`: no automatic language instruction is added

English is only added when the user explicitly requested it with a language tag such as `[en:Bob]` or `[English:Bob]`. Implicit/default English stays untagged.

## Character Switching Modes

The engine defaults to `Native Multi-Speaker`. All `[Character]` turns in a generated block are sent through one Fish dialogue request, preserving native multi-turn context and long-form behavior.

Select `Custom Character Switching` to generate every parsed character segment independently. Each call uses only that character's reference and is remapped to Fish speaker 0. This can reduce speaker leakage, but it requires more calls and does not preserve context between character turns. SRT subtitle boundaries and timing are preserved in both modes.

The engine UI separates checkpoint choice from load-time quantization:

- `model_variant`: `s2-pro` or the separate `s2-pro-fp8` checkpoint
- `quantization`: `none`, `bnb_int8`, or `bnb_nf4` for on-the-fly quantization of the official `s2-pro` checkpoint

The BNB options reuse the official files and do not download a second model copy.

The S2 Pro weights use the Fish Audio Research License: research and non-commercial use are allowed; commercial use requires a separate Fish Audio license.
