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

#### MOSS Sound Effects

MOSS-SoundEffect v1 uses the applicable MOSS-TTS parameters above. Both sound-effect engines also support a duration override:

| Parameter | Alias | Engines | Type | Range | Description |
|-----------|-------|---------|------|-------|-------------|
| `duration_seconds` | `seconds` | v1, v2 | float | 0.5-300 | Duration of the sound segment |
| `inference_steps` | `steps` | v2 | int | 1-150 | Diffusion steps |
| `cfg` | — | v2 | float | 0.0-20.0 | Prompt guidance strength |
| `sigma_shift` | — | v2 | float | 0.0-10.0 | Flow-matching schedule shift |
| `negative_prompt` | `negative`, `neg` | v2 | string | text | Sounds or qualities to discourage |

See the [Sound Effects Guide](SOUND_EFFECTS_GUIDE.md) for pauses, crossfades, long-duration chunking, and complete examples.

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

#### ZONOS3-Mini
| Parameter | Alias | Type | Range | Description |
|-----------|-------|------|-------|-------------|
| `top_p` | `topp` | float | 0.0-1.0 | Nucleus sampling probability |
| `top_k` | `topk` | int | 1-4096 | Top-k sampling |
| `min_p` | — | float | 0.0-1.0 | Minimum probability relative to the top token |
| `repetition_penalty` | `rep_penalty` | float | 0.5-3.0 | Audio repetition penalty |
| `speaker_cfg_scale` | `speaker_cfg` | float | 0.0-5.0 | Speaker guidance (1 = off) |
| `loudness_bucket` | — | int | 0-10 | Loudness conditioning (10 = unknown) |
| `fmax_bucket` | — | int | 0-10 | Bandwidth conditioning (10 = unknown) |
| `speaking_rate_bucket` | `rate_bucket` | int | 0-30 | Speaking-rate conditioning (30 = unknown) |

**Steering** is under **Inline Tags → ZONOS3-Mini**. Use **Character → Add Parameter** for the generation settings above.

| Control | Tags and ranges | Notes |
|---------|-----------------|-------|
| Named emotion | `happy`, `angry` 0–6 · `confused` 0–4 · `sad`, `calm` 0–8 · `scared` 0–14 | `happy:2` sets the strength. `happy:+2` and `happy:-2` adjust it. |
| Emotion vector | `zonos_vector` in order `happy, angry, confused, sad, calm, scared` | Sign all six values to adjust the connected settings. Unsigned values replace them. Named tags apply after the vector. |
| Delivery and mood | `projection` −3 to 4 · `pace` −3 to 3 · `chatty` −5 to 5 · `hesitant` −8 to 20 · `lively` −16 to 16 · `valence`, `arousal` −6 to 6 | Signed values set the direction and strength. |
| Voice characteristics | `gender`, `age`, `pitch`, `pitch_range` −3 to 3 · `deep`, `bright`, `husky`, `raspy`, `soft`, `authority` 0–6 · `room` −4 to 6 | Embedding voice mode. |
| Accents | `*_accent` 0–2 · `mandarin_accent` 0–1.5 | Pick the accent in Voice Characteristics. |
| Additional effects | Native IDs such as `emotions.proud` −50 to 50 | See the Additional Effects selector for the full list. |
| Effect timing | `full_strength`, `reduction` 0–600 s · `remaining_strength` 0–100% · `control_guidance` 0–5 | Pace and whisper keep their strength throughout the segment. |

```text
[Alice|happy:+2|pace:-1] Increase happiness and slow the delivery.
[Alice|zonos_vector:2,0,0,0,1,0] Set the six emotion strengths.
```

Extra control guidance requires an active delivery effect, embedding voice mode, LACI off, and speaker guidance at 1.

#### IndexTTS 2 / 2.5

| Parameter | Alias | Type | Range | Description |
|-----------|-------|------|-------|-------------|
| `top_p` | `topp` | float | 0–1 | Nucleus sampling probability |
| `top_k` | `topk` | int | 1–4096 | Top-k sampling |
| `emotion_alpha` | — | float | 0–1 | Audio, vector, and text emotion intensity |
| `duration_factor` | `dur_factor` | float | 0.5–2.0 | IndexTTS-2.5 feature duration multiplier |

| Emotion tag | Example | Behavior |
|-------------|---------|----------|
| Named strength | `[sad:0.7\|calm:-0.2]` | Unsigned values set the strength. Signed values adjust it. Results stay in 0–1.2. |
| Full vector | `[vector:0,0,0.7,0,0,0.4,0,0.2]` | Order is `happy, angry, sad, afraid, disgusted, melancholic, surprised, calm`. Sign every value for a relative adjustment. |
| Text or preset | `[emotion:"Quiet grief"]` or `[emotion:restrained_anger]` | Use `{seg}` in quoted text to include the current segment in emotion analysis. |
| Audio reference | `[Bob:emotion_ref\|sad:+0.25]` | The audio reference can blend with vector or text emotion. |

Use one of named values, a full vector, or text emotion per segment. The editor’s **Inline Tags → IndexTTS-2** panel includes the radar, press-and-drag picker, and preset tools.

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

### DramaBox Prompt Templates

`prompt_template` (alias `template`) applies a `{seg}` wrapper and enables
templating for that segment automatically:

```text
[Narrator|template:A woman whispers, "{seg}"] This line is whispered.
[Narrator] This line returns to the DramaBox engine-node settings.
```

The template should include `{seg}`. If it is omitted, DramaBox warns once and
appends `"{seg}"` automatically. A separate inline enable parameter is not
required.

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
