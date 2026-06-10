# Higgs Audio v3 Inline Tags

Higgs Audio v3 uses native inline control tokens in the text prompt. Type these tags directly in **TTS Text** or **TTS SRT** when using the Higgs Audio v3 engine.

## Syntax

Canonical native tags use:

```text
<|category:value|>
```

TTS Audio Suite also accepts a convenience alias form without the bars:

```text
<category:value>
```

Aliases are normalized internally to the canonical Higgs syntax. The editor inserts the canonical `<|...|>` form on purpose.

Examples:

```text
<|emotion:amusement|>Wait, that was actually funny.
<|style:whispering|>Keep your voice down.
I need a second.<|prosody:pause|> Okay, continue.
That was perfect. <|sfx:laughter|>Haha, absolutely perfect.
```

## Supported Categories

Emotion:

```text
elation, amusement, enthusiasm, determination, pride, contentment, affection,
relief, contemplation, confusion, surprise, awe, longing, arousal, anger, fear,
disgust, bitterness, sadness, shame, helplessness
```

Style:

```text
singing, shouting, whispering
```

Prosody:

```text
speed_very_slow, speed_slow, speed_fast, speed_very_fast,
pitch_low, pitch_high, pause, long_pause,
expressive_high, expressive_low
```

Sound effects:

```text
cough, laughter, crying, screaming, burping, humming, sigh, sniff, sneeze
```

## Placement Rules

- Put delivery-wide tags like emotion, style, speed, pitch, and expressiveness at the start of the sentence or turn.
- Put positional tags like `<|prosody:pause|>`, `<|prosody:long_pause|>`, and `<|sfx:...|>` exactly where they should happen.
- Pair SFX tags with written sound text, for example `<|sfx:laughter|>Haha` or `<|sfx:sigh|>Ahh`.
- Canonical Higgs syntax is `<|...|>`. Alias input like `<emotion:amusement>` is accepted by TTS Audio Suite for convenience, but the official form stays `<|emotion:amusement|>`.
- Higgs Audio v3 does not use Step Audio EditX second-pass inline processing here. Use Higgs native tags instead of Step semantics when targeting Higgs.

## Character Voices

Higgs Audio v3 supports reference audio cloning. A transcript of the reference audio strongly improves cloning quality. Character voice `.txt` files are passed as reference transcripts when available.

## Multiline Editor

The multiline TTS tag editor has an `Inline Tags` panel with a dedicated `Higgs Audio v3` mode. That panel inserts canonical native Higgs tags directly.

## License

Higgs Audio v3 is released under the Boson Higgs Audio v3 Research and Non-Commercial License. Production, hosted API, or revenue-generating use requires a separate commercial license from Boson AI.
