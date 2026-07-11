# ⚙️ Per-Segment Parameter Switching Guide

## Overview

The TTS Audio Suite supports **per-segment parameter control** through bracket tags like `[Alice|seed:42|temp:0.5]`. This allows you to override TTS generation parameters (seed, temperature, cfg, speed, etc.) on a per-segment basis, providing fine-grained control over individual audio segments without modifying node-level defaults.

Parameters are applied **only to the current segment** and automatically revert to node defaults for subsequent segments.

---

## ✨ Key Features

- **⚙️ Per-Segment Control** - Override parameters for individual text segments or SRT subtitle entries
- **🔀 Order-Independent** - Specify parameters in any order: `[Alice|seed:42|temp:0.5]` or `[seed:42|Alice|temp:0.5]`
- **📝 Alias Support** - Use shortcuts like `temp` (temperature), `cfg_weight` (cfg), `exag` (exaggeration)
- **🔤 Case-Insensitive** - `[SEED:42]`, `[Seed:42]`, `[seed:42]` all work identically
- **🚨 Smart Filtering** - Unsupported parameters are silently ignored with warnings (e.g., `[speed:2.0]` on ChatterBox)
- **📺 SRT Support** - Apply parameters to individual SRT subtitle lines
- **🔙 Backward Compatible** - Works alongside character switching: `[de:Alice|seed:42|temp:0.7]`

---

## 🚀 Quick Start

### Basic Parameter Syntax

**Format:** `[character|parameter:value]` or `[parameter:value|character]`

```
[Alice|seed:42] Hi there with a specific seed!
[Bob|temperature:0.5] Bob speaks more deterministically.
[Alice|seed:42|temperature:0.7] Alice with both seed and temperature.
[seed:42|Alice] Order doesn't matter!
```

### Real-World Examples

#### Consistent Character Voice
```
[Alice|seed:42] This is the first segment of Alice.
[Alice|seed:42] This is the second segment, sounding identical.
[Alice|seed:42] Third segment with same seed.
```

#### Varied Voice Characteristics
```
[Alice|temperature:0.3] Alice speaks carefully and precisely.
[Alice|temperature:0.8] Alice speaks more creatively and varied.
[Alice|temperature:0.3] Alice is precise again.
```

#### SRT Example with Parameters
```srt
1
00:00:01,000 --> 00:00:04,000
[Alice|seed:42] Hello! I'm Alice and this line has a specific seed.

2
00:00:04,500 --> 00:00:09,500
[Bob|seed:100] And I'm Bob with my own unique seed.

3
00:00:10,000 --> 00:00:14,000
[Alice|seed:42] Alice again with the same seed as segment 1.
```

---

## 📋 Supported Parameters

### Universal Parameters (All Engines)

| Parameter | Alias | Type | Range | Description |
|-----------|-------|------|-------|-------------|
| `seed` | — | int | 0-4294967295 | Random seed for reproducible generation |
| `temperature` | `temp` | float | 0.1-2.0 | Randomness/creativity (lower=more deterministic) |

### Engine-Specific Parameters

#### MOSS-TTS
| Parameter | Alias | Type | Range | Description |
|-----------|-------|------|-------|-------------|
| `top_p` | `topp` | float | 0.0-1.0 | Nucleus sampling probability |
| `top_k` | `topk` | int | 1-100 | Top-k sampling |
| `audio_temperature` | `audio_temp` | float | 0.1-2.5 | MOSS audio sampling temperature |
| `audio_top_p` | — | float | 0.0-1.0 | MOSS audio nucleus sampling |
| `audio_top_k` | — | int | 1-200 | MOSS audio top-k sampling |
| `repetition_penalty` | `rep_penalty` | float | 0.5-3.0 | MOSS repetition penalty |
| `audio_repetition_penalty` | `audio_rep_penalty` | float | 0.5-3.0 | MOSS audio repetition penalty |
| `duration_tokens` | `tokens` | int | 0-8192 | MOSS target output length hint in audio tokens |
| `max_new_tokens` | — | int | 64-16384 | Hard generation token limit |
| `n_vq_for_inference` | `n_vq` | int | 0-32 | Local 1.7B only: RVQ layers/codebooks for inference |
| `instruction` | — | string | text | Whole-segment speaking instruction |
| `quality` | — | string | text | Whole-segment recording/presentation quality hint |
| `sound_event` | — | string | text | Whole-segment sound event hint |
| `ambient_sound` | — | string | text | Whole-segment ambient sound hint |

#### ChatterBox & ChatterBox Official 23-Lang
| Parameter | Alias | Type | Range | Description |
|-----------|-------|------|-------|-------------|
| `cfg` | `cfg_weight` | float | 0.0-1.0 | CFG weight for speaker guidance |
| `exaggeration` | `exag` | float | 0.0-2.0 | Voice emotion exaggeration |

#### F5-TTS
| Parameter | Alias | Type | Range | Description |
|-----------|-------|------|-------|-------------|
| `cfg` | — | float | 0.0-20.0 | Classifier-free guidance strength |
| `speed` | — | float | 0.5-2.0 | Speech speed multiplier |

#### Higgs Audio 2
| Parameter | Alias | Type | Range | Description |
|-----------|-------|------|-------|-------------|
| `cfg` | — | float | 0.0-20.0 | CFG strength |
| `top_p` | `topp` | float | 0.0-1.0 | Nucleus sampling probability |
| `top_k` | `topk` | int | 1-100 | Top-k sampling |

#### VibeVoice
| Parameter | Alias | Type | Range | Description |
|-----------|-------|------|-------|-------------|
| `cfg` | — | float | 0.0-20.0 | CFG strength |
| `top_p` | `topp` | float | 0.0-1.0 | Nucleus sampling probability |
| `inference_steps` | `steps` | int | 1-100 | Number of inference steps |

#### IndexTTS-2
| Parameter | Alias | Type | Range | Description |
|-----------|-------|------|-------|-------------|
| `cfg` | — | float | 0.0-20.0 | CFG strength |
| `top_p` | `topp` | float | 0.0-1.0 | Nucleus sampling probability |
| `top_k` | `topk` | int | 1-100 | Top-k sampling |
| `emotion_alpha` | — | float | 0.0-2.0 | Shared audio/vector/text emotion intensity |

IndexTTS-2 also supports inline emotion controls. Named unsigned values replace
that dimension; explicitly signed values adjust the connected vector:

```text
[sad:0.7|calm:0.2] Absolute values for this segment.
[sad:+0.3|calm:-0.2] Adjust the connected vector for this segment.
```

All eight values can be supplied in the official order `happy, angry, sad,
afraid, disgusted, melancholic, surprised, calm`:

```text
[vector:0,0,0.7,0,0,0.4,0,0.2] Absolute replacement.
[vector:+0,+0,+0.3,+0,+0,+0,+0,-0.2] Relative adjustment.
```

Full relative vectors require an explicit sign on every value. Results are
clamped to IndexTTS-2's supported range and revert to the connected vector at
the next segment.

Text emotion can use a saved preset or quoted text. `{seg}` is expanded with
the current segment before QwenEmotion analysis:

```text
[emotion:restrained_anger] A saved preset.
[emotion:"Quiet grief masking frustration"] A direct description.
[emotion:"Infer nervous anticipation from this line: {seg}"] Dynamic analysis.
```

Click a numeric emotion tag in the TTS Tag Editor to open a contextual radar
directly beside that tag. The editor also creates and manages text
presets in `models/TTS/IndexTTS/emotion_presets.json`.

IndexTTS-2 has separate engine inputs for these sources: connect vector or text
emotion to `emotion_control` and audio emotion references to `emotion_audio`.
Both may be connected simultaneously; IndexTTS-2 blends them during emotion
conditioning. Inline vector/text controls override the connected vector/text
values for their segment, while `[Character:emotion_ref]` selects a
segment-local audio reference that can still blend with vector/text emotion.
When an emotion control is inserted with the caret inside a character/audio tag,
the editor appends or updates it as another pipe parameter, for example
`[Bob:br_ivan_raiva3|sad:+0.25]`.

---

## 🎯 Advanced Usage

### Combining with Character Switching

Parameters work seamlessly with character and language tags:

```
[de:Alice|seed:42] German Alice with specific seed.
[fr:Bob|temperature:0.7] French Bob with higher temperature.
[seed:123|de:Alice] Order-independent syntax.
```

### MOSS Whole-Segment Prompt Fields

MOSS official prompt fields are better treated as per-segment parameters, not inline `<>` tags:

```
[Alice|instruction:Speak softly and calmly] Hello there.
[Bob|quality:Telephone call quality|ambient_sound:Office room tone] Can you hear me?
[Alice|sound_event:Laughter] That's actually funny.
```

Important:

- These are whole-segment controls
- They are not positional inline effects
- Keep `<>` free for true inline post-processing tags like Step Audio EditX

### Per-Segment Fine-Tuning in SRT

Create dynamic SRT content with per-segment control:

```srt
1
00:00:00,500 --> 00:00:03,000
[Narrator|temp:0.4] The important introduction needs precision.

2
00:00:03,100 --> 00:00:06,500
[Narrator|temp:0.4] Continuing with careful, consistent delivery.

3
00:00:06,600 --> 00:00:10,000
[Narrator|temp:0.7] Now for the creative narrative section!

4
00:00:10,100 --> 00:00:14,000
[Narrator|temp:0.4] Back to precise delivery for the conclusion.
```

### Alias Examples

Use shorter aliases for convenience:

```
[Alice|temp:0.5]           # temperature
[Alice|cfg_weight:0.7]     # Same as cfg:0.7
[Alice|exag:1.5]           # exaggeration
[Alice|steps:50]           # inference_steps (VibeVoice)
[Alice|topk:20]            # top_k
[Alice|topp:0.95]          # top_p
```

### Semantic Parameters for Control

#### Temperature Examples
```
[Alice|temperature:0.2] This is spoken very carefully, word for word.
[Alice|temperature:0.5] This is natural but somewhat consistent.
[Alice|temperature:0.9] This is creative with varied expression.
```

#### Seed Examples
```
# Same seed = identical audio output
[Alice|seed:42] First attempt.
[Alice|seed:42] Identical to first attempt.

# Different seeds = different variations
[Alice|seed:42] Variation 1.
[Alice|seed:100] Variation 2.
[Alice|seed:999] Variation 3.
```

---

## ⚠️ Important Notes

### Parameter Validation

- **Invalid values** are logged with warnings but don't cause errors
- **Unsupported parameters** for your engine are silently ignored
- **Out-of-range values** are clamped to valid ranges
- **Type mismatches** (e.g., `temperature:abc`) are logged and rejected

### Engine Compatibility

Different engines support different parameters. The system automatically:
- ✅ Applies supported parameters
- ⚠️ Warns about unsupported parameters
- ✅ Continues processing without errors

Example:
```
[Alice|speed:0.8|temperature:0.5]  # ChatterBox: ignores speed, applies temperature
[Alice|speed:0.8|temperature:0.5]  # F5-TTS: applies both
```

### Performance Considerations

- Parameter extraction happens during text parsing (minimal overhead)
- Parameters only affect the specific segment
- No impact on caching or model loading

---

## 🔧 Implementation Notes

### For Node Developers

If you're adding parameter support to a new engine:

1. Import the parameter system:
   ```python
   from utils.text.segment_parameters import apply_segment_parameters
   ```

2. When processing segments with parameters:
   ```python
   if segment.parameters:
       # Apply segment parameters to override config
       segment_config = apply_segment_parameters(base_config, segment.parameters, engine_type)
   else:
       segment_config = base_config
   ```

3. Generate audio with `segment_config` instead of `base_config`

### Parameter Storage

Parameters are stored in the `CharacterSegment.parameters` dictionary:
```python
segment.parameters = {
    'seed': 42,
    'temperature': 0.5,
    'cfg': 0.7
}
```

---

## 📚 See Also

- [Character Switching Guide](CHARACTER_SWITCHING_GUIDE.md) - Multi-character and language switching
- [ChatterBox Special Tokens Guide](CHATTERBOX_V2_SPECIAL_TOKENS.md) - Emotion and sound tokens
- README.md - Complete feature overview

---

## 🐛 Troubleshooting

### Parameters Not Being Applied

1. **Check engine compatibility** - Parameter might not be supported by your engine
2. **Verify syntax** - Use `|` to separate parameters: `[Alice|seed:42|temp:0.5]`
3. **Check for typos** - Parameter names are case-insensitive but must match known parameters
4. **Look at console** - Warnings are logged when parameters are unsupported or invalid

### "Parameter not supported" Warning

This is normal! Different engines support different parameters. For example:
- ChatterBox doesn't support `speed` (F5-TTS-only parameter)
- F5-TTS doesn't support `exaggeration` (ChatterBox-only parameter)

The system silently ignores unsupported parameters and continues processing.

### Values Not Taking Effect

- **Check range** - Values outside the valid range are clamped to min/max
- **Verify engine** - Make sure you're using an engine that supports that parameter
- **Check node defaults** - Parameters only override node-level defaults, they don't create new capabilities
