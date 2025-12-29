# CosyVoice3 Paralinguistic Tags Guide

## Overview

CosyVoice3 has **native paralinguistic tag support** built into the model. Unlike Step Audio EditX post-processing, these tags are processed **during TTS generation** for natural integration.

**⚠️ IMPORTANT**: Step Audio EditX post-processing is **DISABLED** for CosyVoice3. EditX syntax like `<Laughter:2>`, `<emotion:happy>`, `<style:whisper>` will be stripped but **not processed**. Use CosyVoice3 native tags instead.

---

## Supported Tags

### Single Tags (Sound Insertion)

| Tag | Effect | Example |
|-----|--------|---------|
| `<breath>` | Breathing sound | `I'm tired <breath> let's rest` |
| `<quick_breath>` | Quick breath | `Running <quick_breath> almost there` |
| `<laughter>` | Laughter | `That's hilarious <laughter>!` |
| `<cough>` | Cough | `Excuse me <cough> sorry` |
| `<sigh>` | Sigh | `Fine <sigh> I'll do it` |
| `<gasp>` | Gasp | `Oh no <gasp> what happened?` |
| `<noise>` | Background noise | `Walking <noise> through forest` |
| `<hissing>` | Hissing sound | `The snake <hissing> away` |
| `<vocalized-noise>` | Vocalized noise | `Hmm <vocalized-noise> interesting` |
| `<lipsmack>` | Lip smack | `Delicious <lipsmack> food` |
| `<mn>` | "mn" sound | `I think <mn> maybe` |
| `<clucking>` | Clucking sound | `Disapproving <clucking>` |
| `<accent>` | Accent emphasis | `Very <accent> important` |

**Position matters**: Tags insert sounds where you place them.

### Wrapper Tags (Text Emphasis)

| Tag | Effect | Example |
|-----|--------|---------|
| `<laughing>text</laughing>` | Apply laughter to text | `<laughing>so funny</laughing>!` |
| `<strong>text</strong>` | Emphasize text | `<strong>very important</strong>` |

**Note**: Use `<laughing>` for wrapper tags (auto-converted to `<laughter>` internally).

### Language Switching

CosyVoice3 supports 4 native languages (EN, ZH, JA, KO) and 5 cross-lingual languages (DE, ES, FR, IT, RU).

**TTS Audio Suite bracket syntax** (auto-converted):
```
[en:Alice] Hello world
[zh:Bob] 你好世界
[ja:Alice] こんにちは
[ko:Bob] 안녕하세요

Or without character name:
[en:] Hello world
[zh:] 你好世界
```

**CosyVoice3 native syntax** (manual):
```
<|en|>Hello world
<|zh|>你好世界
<|ja|>こんにちは
<|ko|>안녕하세요
```

Both work. Bracket syntax is recommended for consistency with other engines.

---

## Tag Syntax

### User-Friendly Format (Recommended)

**You write**: `<breath>`, `<laughter>`, `<cough>`

**Auto-converted to**: `[breath]`, `[laughter]`, `[cough]`

This prevents conflicts with character switching `[CharacterName]` brackets.

### Direct Format (Advanced)

You can also use native CosyVoice3 format: `[breath]`, `[laughter]`, `[cough]`

Both formats work identically.

---

## Usage Examples

### Basic

```
[Alice] Hello there <breath> nice to meet you!
```

### Multiple Effects

```
[Bob] I can't believe it <gasp> that's <laughter> amazing!
```

### Wrapper Tags

```
[Alice] This is <laughing>hilarious</laughing>!
[Narrator] The news was <strong>absolutely shocking</strong>.
```

### Natural Conversation

```
[Alice] Did you hear the news? <breath>
[Bob] Yes <gasp> I couldn't believe it!
[Alice] I know right <laughter> it's crazy!
```

### SRT Subtitles

```
1
00:00:00,000 --> 00:00:03,000
[Alice] Welcome to the show <breath> everyone!

2
00:00:03,500 --> 00:00:06,000
[Bob] Thanks for having me <laughter>
```

---

## Best Practices

### 1. Position Tags Naturally

```
✅ GOOD: "I'm so tired <sigh> let's take a break"
❌ AWKWARD: "<sigh> <breath> I'm tired"
```

### 2. Don't Overuse

```
✅ GOOD: "Hello <breath> nice to meet you"
❌ BAD: "Hello <breath> there <laughter> nice <sigh> to you"
```

**Recommended**: 1-2 tags per sentence, 3-4 maximum per segment.

### 3. Match Context

```
# Tired
[Alice] I can't go on <sigh> I need rest <breath>

# Happy
[Bob] That was hilarious <laughter> tell me more!

# Surprised
[Alice] Wait what? <gasp> Are you serious?
```

---

## CosyVoice3 Tags vs Step Audio EditX

| Feature | CosyVoice3 Native | Step Audio EditX |
|---------|------------------|-----------------|
| **Syntax** | `<breath>`, `<laughter>` | `<Laughter:2>`, `<emotion:happy>` |
| **Processing** | During TTS | Post-processing (disabled for CosyVoice3) |
| **Effects** | 13 paralinguistic sounds | 13 paralinguistic + 14 emotions + 32 styles + speed + restore |
| **Iterations** | Single pass | 1-30 customizable |
| **Engine Support** | CosyVoice3 only | All other engines |

**When to use CosyVoice3 tags**: Generating with CosyVoice3 engine, need paralinguistic effects

**When to use EditX tags**: Need emotion/style/speed control, OR using other engines (ChatterBox, F5-TTS, etc.)

---

## Limitations

### ⚠️ No Step Audio EditX Support

```
❌ WON'T WORK: <Laughter:2>        (EditX - will be stripped)
❌ WON'T WORK: <emotion:happy>     (EditX - will be stripped)
❌ WON'T WORK: <style:whisper>     (EditX - will be stripped)

✅ USE INSTEAD: <laughter>         (CosyVoice native)
✅ USE INSTEAD: <strong>text</strong>
```

For emotion/style control, use ChatterBox, IndexTTS-2, or other engines with EditX support.

### Tag Case

Tags are **case-insensitive**: `<breath>` = `<BREATH>` = `<Breath>`

---

## FAQ

**Q: Can I use Step Audio EditX tags with CosyVoice3?**
A: No. EditX syntax (`<Laughter:2>`, `<emotion:happy>`) will be stripped but not processed.

**Q: Can I control effect strength like EditX iterations?**
A: No. CosyVoice3 tags are single-pass. Effect strength is determined by the model.

**Q: Which languages support tags best?**
A: All supported languages (EN, ZH, JA, KO, DE, ES, FR, IT, RU) work with tags.

**Q: Can I use tags in SRT mode?**
A: Yes! Include tags in subtitle text.

---

## Related Guides

- **[Inline Edit Tags Guide](INLINE_EDIT_TAGS_USER_GUIDE.md)** - Step Audio EditX tags for other engines
- **[Parameter Switching Guide](PARAMETER_SWITCHING_GUIDE.md)** - Per-segment parameter control

---

**Last Updated**: 2025-12-29
**Compatible with**: TTS Audio Suite v4.16.0+
