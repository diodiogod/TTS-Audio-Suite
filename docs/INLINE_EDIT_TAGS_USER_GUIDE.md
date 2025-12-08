# Inline Edit Tags - User Guide

## 💡 Inline vs Manual Node Workflow

**ComfyUI is modular** - you can manually chain **TTS → 🎨 Step Audio EditX - Audio Editor** nodes for full control.

**Why use inline tags instead?**
- Apply edits to **specific segments** of text without multiple TTS→Edit→TTS chains
- Convenient for **SRT files** with selective editing per subtitle
- Automatic batch processing for efficiency

**For maximum control and quality**, build manual workflows. **For convenience and selective segment editing**, use inline tags.

---

## Overview

Inline edit tags allow you to apply **Step Audio EditX** post-processing effects to TTS-generated audio from ANY engine (F5-TTS, ChatterBox, Higgs Audio, VibeVoice, etc.). These tags are automatically stripped before TTS generation and applied as a second processing pass.

**Example:**
```
[Alice] This is hilarious <Laughter:3>, I can't believe it!
```

**What happens:**
1. TTS generates: "This is hilarious, I can't believe it!" (tag removed)
2. Step Audio EditX edits the audio with 3 iterations of Laughter effect
3. Final audio has natural laughter inserted

---

## Supported Edit Types

### 1. Paralinguistic Tags (Sound Insertion)

Insert non-verbal sounds at specific positions in the audio:

| Tag | Effect | Example |
|-----|--------|---------|
| `<Laughter>` or `<Laughter:2>` | Insert laughter | `Hello <Laughter> world` |
| `<Breathing>` | Insert breathing | `I'm tired <Breathing> let's rest` |
| `<Sigh>` | Insert sigh | `Okay <Sigh> fine` |
| `<Uhm>` | Insert "uhm" hesitation | `Well <Uhm> maybe` |
| `<Surprise-oh>` | Surprised "oh!" | `<Surprise-oh> I didn't know!` |
| `<Surprise-ah>` | Surprised "ah!" | `<Surprise-ah> That's amazing!` |
| `<Surprise-wa>` | Surprised "wa!" | `<Surprise-wa> Incredible!` |
| `<Confirmation-en>` | Confirmation sound | `<Confirmation-en> Yes` |
| `<Question-ei>` | Question sound | `<Question-ei> Really?` |
| `<Dissatisfaction-hnn>` | Dissatisfied "hnn" | `<Dissatisfaction-hnn> Not good` |

**Position matters:** The sound is inserted where you place the tag.

**💡 Pro Tip - Stronger Effects:** For more reliable and pronounced effects, include relevant text alongside the tag:
```
# Weaker effect
I'm happy <Laughter>

# Stronger, more guaranteed effect
I'm laughing haha <Laughter:2>
```

The TTS-generated "haha" combined with the post-processing laughter creates a more natural and pronounced effect.

### 2. Emotion Tags (Whole Segment)

Apply emotional tone to the entire audio segment:

`<emotion:VALUE>` or `<emotion:VALUE:ITERATIONS>`

**Available emotions:** happy, sad, angry, excited, calm, fearful, surprised, disgusted, confusion, empathy, embarrass, depressed, coldness, admiration

**Examples:**
```
[Alice] I'm so happy to see you! <emotion:happy>
[Bob] This is terrible news. <emotion:sad:2>
```

**Position doesn't matter:** Tag can be anywhere in the text.

### 3. Style Tags (Whole Segment)

Apply speaking style to the entire audio segment:

`<style:VALUE>` or `<style:VALUE:ITERATIONS>`

**Available styles:** whisper, serious, child, older, girl, pure, sister, sweet, exaggerated, ethereal, generous, recite, act_coy, warm, shy, comfort, authority, chat, radio, soulful, gentle, story, vivid, program, news, advertising, roar, murmur, shout, deeply, loudly, arrogant, friendly

**Examples:**
```
[Alice] Come closer, I have a secret <style:whisper>
[Bob] Listen carefully to this. <style:serious:2>
```

**Position doesn't matter:** Tag can be anywhere in the text.

### 4. Speed Tags (Whole Segment)

Adjust speaking speed for the entire audio segment:

`<speed:VALUE>` or `<speed:VALUE:ITERATIONS>`

**Available speeds:** faster, slower, more_faster, more_slower

**Examples:**
```
[Alice] Quick! We need to hurry! <speed:faster>
[Bob] Let me think about this... <speed:slower:2>
```

**Position doesn't matter:** Tag can be anywhere in the text.

---

## Syntax Guide

### Basic Syntax

**Default (1 iteration):**
```
<Laughter>
<emotion:happy>
<style:whisper>
<speed:faster>
```

**Custom iterations:**
```
<Laughter:3>           # 3 iterations of laughter
<emotion:happy:2>      # 2 iterations of happy emotion
<style:whisper:4>      # 4 iterations of whisper style
```

### Multiple Tags

**Option 1 - Pipe-separated (single tag):**
```
[Alice] Hello world! <Laughter:2|style:whisper|emotion:happy>
```

**Option 2 - Separate tags:**
```
[Alice] Hello world! <Laughter:2><style:whisper><emotion:happy>
```

**Option 3 - Mixed (both work):**
```
[Alice] Hello <Laughter:2|emotion:happy> world! <style:whisper>
```

### Processing Order

When multiple tags are present:

1. **First:** Non-paralinguistic tags in order: emotion → style → speed
2. **Last:** Paralinguistic tags (preserves position for sound insertion)

**Example:**
```
[Alice] Hello <Laughter:2> world <style:whisper:1>
```

Execution:
1. Generate TTS: "Hello world"
2. Apply whisper style (1 iteration)
3. Insert Laughter at position (2 iterations)

---

## Usage Examples

### Simple Effects

**Laughter:**
```
[Alice] That's the funniest thing I've heard <Laughter>!
```

**Multiple iterations:**
```
[Bob] I can't stop laughing <Laughter:3>, this is amazing!
```

### Combined Effects

**Emotion + Paralinguistic:**
```
[Alice] I'm so excited <Laughter:2|emotion:happy> to see you!
```

**Style + Speed:**
```
[Bob] Listen carefully <style:whisper|speed:slower>, this is important.
```

### Full Example with Character Switching

```
[Narrator] The story begins on a dark night.

[Alice|seed:42] Oh no <Surprise-oh>, what was that sound? <emotion:fearful>

[Bob] Don't worry <style:comfort>, I'll check it out. <Breathing>

[Alice] Please be careful! <style:whisper|emotion:worried>
```

### SRT Mode

```
1
00:00:00,000 --> 00:00:03,000
[Alice] Hello there <Laughter> my friend!

2
00:00:03,500 --> 00:00:06,000
[Bob|seed:123] Hi! <style:excited|emotion:happy>

3
00:00:06,500 --> 00:00:09,000
[Alice] Let me tell you a secret <style:whisper:2>
```

---

## ChatterBox 23-Lang v2 Users: Native vs Post-Processing Tags

### Understanding the Difference

**ChatterBox Official 23-Lang v2** has **native special tokens** built into the model (e.g., `<laughter>`, `<giggle>`, `<whisper>`). Our **Step Audio EditX post-processing system** uses similar-looking tags but works differently.

### Syntax Distinction

| Syntax | System | When to Use |
|--------|--------|-------------|
| `<laughter>` | **ChatterBox v2 Native** | Model generates laughter during TTS (experimental, may not work well) |
| `<laughter:1>` | **Step Audio EditX Post-Processing** | Applied AFTER TTS generation (more reliable) |

**The colon (`:`) makes the difference!**

### Examples

```
# Native ChatterBox v2 tag (model-generated during TTS)
[Alice] I'm laughing <laughter> so hard!

# Step Audio EditX tag (post-processing after TTS)
[Alice] I'm laughing <Laughter:2> so hard!
```

### Which Should You Use?

**For ChatterBox 23-Lang users:**

✅ **Use Step Audio EditX tags** (`<Laughter:2>`, etc.) - More reliable, works consistently

⚠️ **ChatterBox v2 native tags** (`<laughter>`, `<giggle>`) - Experimental, limited functionality:
- Tokens exist in vocabulary but may produce minimal/no effect
- No official documentation from ResembleAI
- Results are inconsistent
- See [ResembleAI/chatterbox #186](https://github.com/resemble-ai/chatterbox/issues/186)

**Console Warning:**

If you use ChatterBox v2 native tags without the colon, you may see:
```
⚠️ Using ChatterBox v2 native tag '<laughter>' - experimental feature with limited effect.
   For more reliable results, use Step Audio EditX: '<Laughter:2>'
```

### Compatibility Note

Step Audio EditX tags work with **ALL engines** (F5-TTS, ChatterBox, Higgs Audio, VibeVoice), not just ChatterBox. Native ChatterBox tags only work with ChatterBox 23-Lang v2.

---

## Important Limitations

### ⚠️ Language Support Warning

**Step Audio EditX supports ONLY:**
- **Mandarin Chinese** (primary)
- **English**
- **Sichuanese** (Chinese dialect)
- **Cantonese** (Chinese dialect)
- **Japanese**
- **Korean**

**If using other languages** (French, German, Spanish, etc.):

❌ **PROBLEM:** Step Audio EditX will process non-supported languages as English or Chinese, which will:
- **Lose the original accent** - French audio may lose French pronunciation
- **Distort the audio** - The model tries to "fix" unknown languages
- **Generate poor quality** - Editing may sound unnatural or broken

✅ **SOLUTION:**
1. **Don't use edit tags** with unsupported languages
2. **Use only supported languages** when edit tags are needed
3. **Test carefully** if mixing languages - some distortion may be acceptable

**Example - BAD:**
```
[Alice:fr] Bonjour mon ami <Laughter>     # French + edit tag = accent loss/distortion
```

**Example - GOOD:**
```
[Alice:fr] Bonjour mon ami                 # French without edit tag = preserves accent
[Alice:en] Hello my friend <Laughter>      # English + edit tag = works perfectly
```

### Duration Limits

**Minimum:** 0.5 seconds - Segments shorter than this will skip editing
**Maximum:** 30 seconds - Segments longer than this will skip editing

**Warning messages:**
```
⚠️ Segment too short (0.3s < 0.5s). Edit tags will be skipped.
⚠️ Segment too long (35.2s > 30s). Edit tags will be skipped. Split your text into shorter segments.
```

**Solution for long segments:**
```
# BAD - Too long
[Alice] [Very long paragraph that exceeds 30 seconds...] <Laughter>

# GOOD - Split into multiple segments
[Alice] First part of the paragraph. <Laughter>
[Alice] Second part continues here.
[Alice] Final part with the conclusion. <emotion:happy>
```

### Batch Processing (Automatic)

Edit tags are **automatically batched** for efficiency:

✅ **What happens:**
1. ALL TTS audio is generated first (for all segments/subtitles)
2. Step Audio EditX loads ONCE
3. ALL segments with edit tags are processed together
4. Engine unloads

✅ **Benefits:**
- Faster generation (engine loads once, not per segment)
- Less VRAM thrashing
- Better performance

**You don't need to do anything** - batching happens automatically in both regular and SRT modes.

---

## Performance Tips

### 1. Use Iterations Wisely

More iterations = stronger effect BUT **can alter voice quality and resemblance**:

```
<Laughter:1>     # Subtle effect, minimal voice change (recommended)
<Laughter:2>     # Moderate effect, slight voice alteration (acceptable)
<Laughter:3>     # Strong effect, noticeable voice change (risky)
<Laughter:4>     # Very strong effect, significant voice distortion (not recommended)
```

⚠️ **Voice Resemblance Warning:**
- **1-2 iterations:** Generally safe, minimal voice alteration
- **3+ iterations:** High risk of voice distortion and quality degradation
- Each iteration processes the audio again, which can degrade the original voice characteristics

**Recommendation:** Start with 1 iteration, only increase to 2 if needed. Avoid 3+ unless you're willing to accept voice changes.

**🔬 Planned Feature:** A voice restoration pass using ChatterBox Voice Changer will be added to restore original voice resemblance after editing (not yet implemented).

### 2. Combine Tags Efficiently

**Less efficient (multiple passes):**
```
[Alice] Text here <emotion:happy>
[Alice] More text <emotion:happy>
[Alice] Even more <emotion:happy>
```

**More efficient (apply once):**
```
[Alice|emotion:happy] Text here. More text. Even more.
```

Though with inline tags, segment-level parameters are cleaner for consistent effects across multiple sentences.

### 3. Position Paralinguistic Tags Carefully

**Position matters for sound insertion:**
```
Hello <Laughter> world    # Laughter after "Hello"
Hello world <Laughter>    # Laughter at end
<Laughter> Hello world    # Laughter at start
```

Choose the position that sounds most natural for your use case.

---

## Troubleshooting

### Problem: Edit tags not applied

**Possible causes:**
1. **Segment too short/long** - Check duration (0.5s - 30s range)
2. **Syntax error** - Verify tag format: `<type:value:iterations>`
3. **Model not downloaded** - First use triggers download (be patient)

**Check console output:**
```
🎨 Applying Step Audio EditX processing to 3 segments...
📝 Segment 1 - Applying edit tags:
  🎨 Applying paralinguistic: Laughter:2
✅ Edit post-processing complete for 3 segments
```

### Problem: Audio quality degraded

**Possible causes:**
1. **Too many iterations** - Reduce iterations (try 1-2 instead of 3-4)
2. **Incompatible tags** - Some emotion+style combinations may clash
3. **Unsupported language** - Check language support (see warning above)

**Solution:**
- Use fewer iterations
- Test individual tags separately
- Use only supported languages with edit tags

### Problem: "Segment too long" warning

**Solution:** Split text into shorter segments:

```
# Instead of:
[Alice] [300-word monologue that lasts 45 seconds] <Laughter>

# Do this:
[Alice] First paragraph of about 2-3 sentences.
[Alice] Second paragraph continues. <Laughter>
[Alice] Third paragraph concludes the thought. <emotion:happy>
```

---

## Advanced Usage

### Combining with Segment Parameters

Edit tags work with ALL segment parameters:

```
[Alice|seed:42|temperature:0.8|speed:1.2] Hello <Laughter:2> friend! <emotion:happy>
```

Processing:
1. TTS uses seed:42, temp:0.8, speed:1.2
2. Edit tags applied after generation

### Using with Language Switching

**Supported languages ONLY:**
```
[Alice:en] Hello my friend <Laughter>              # English - works ✓
[Bob:zh] 你好朋友 <Laughter>                         # Chinese - works ✓
[Alice:ja] こんにちは友達 <Laughter>                  # Japanese - works ✓
[Bob:ko] 안녕 친구 <Laughter>                        # Korean - works ✓
```

**Unsupported languages - AVOID:**
```
[Alice:fr] Bonjour mon ami <Laughter>              # French - will distort! ✗
[Bob:de] Hallo mein Freund <Laughter>              # German - will distort! ✗
[Alice:es] Hola mi amigo <Laughter>                # Spanish - will distort! ✗
```

### Creating Natural Conversations

```
[Narrator] The interview begins.

[Host] Welcome to the show! <emotion:excited>

[Guest] Thank you for having me. <style:warm>

[Host] So tell us <style:chat>, what's your secret? <Laughter>

[Guest] Well <Uhm> it's quite simple actually. <style:story>

[Host] That's amazing <Surprise-oh>! <emotion:surprised>
```

---

## Best Practices

### 1. Start Simple

Begin with single tags before combining:

```
# Step 1: Test basic tag
[Alice] Hello <Laughter>

# Step 2: Add iterations
[Alice] Hello <Laughter:2>

# Step 3: Combine effects
[Alice] Hello <Laughter:2|emotion:happy>
```

### 2. Match Tags to Context

Choose appropriate effects for the situation:

```
# Scary scene
[Alice] What was that? <Surprise-oh|emotion:fearful>

# Happy scene
[Bob] We did it! <Laughter:2|emotion:excited>

# Sad scene
[Alice] I can't believe it's over. <Sigh|emotion:sad>

# Secret conversation
[Bob] Come closer. <style:whisper|emotion:calm>
```

### 3. Test and Iterate

1. Generate with basic tag
2. Listen to result
3. Adjust iterations if needed
4. Try different tag combinations
5. Find what works best for your use case

### 4. Use Language Hints Correctly

**Always match edit language to TTS language:**

```
# GOOD - Language consistency
[Alice:en] English text here <Laughter>           # Both English
[Bob:zh] 中文文本 <Laughter>                        # Both Chinese

# BAD - Language mismatch
[Alice:fr] French text <Laughter>                 # French TTS + English edit model = distortion
```

---

## Reference

### All Paralinguistic Tags
`<Breathing>`, `<Laughter>`, `<Sigh>`, `<Uhm>`, `<Surprise-oh>`, `<Surprise-ah>`, `<Surprise-wa>`, `<Confirmation-en>`, `<Question-ei>`, `<Dissatisfaction-hnn>`

### All Emotions
`<emotion:happy>`, `<emotion:sad>`, `<emotion:angry>`, `<emotion:excited>`, `<emotion:calm>`, `<emotion:fearful>`, `<emotion:surprised>`, `<emotion:disgusted>`, `<emotion:confusion>`, `<emotion:empathy>`, `<emotion:embarrass>`, `<emotion:depressed>`, `<emotion:coldness>`, `<emotion:admiration>`

### All Styles
`<style:whisper>`, `<style:serious>`, `<style:child>`, `<style:older>`, `<style:girl>`, `<style:pure>`, `<style:sister>`, `<style:sweet>`, `<style:exaggerated>`, `<style:ethereal>`, `<style:generous>`, `<style:recite>`, `<style:act_coy>`, `<style:warm>`, `<style:shy>`, `<style:comfort>`, `<style:authority>`, `<style:chat>`, `<style:radio>`, `<style:soulful>`, `<style:gentle>`, `<style:story>`, `<style:vivid>`, `<style:program>`, `<style:news>`, `<style:advertising>`, `<style:roar>`, `<style:murmur>`, `<style:shout>`, `<style:deeply>`, `<style:loudly>`, `<style:arrogant>`, `<style:friendly>`

### All Speeds
`<speed:faster>`, `<speed:slower>`, `<speed:more_faster>`, `<speed:more_slower>`

---

## FAQ

**Q: Can I use edit tags with any TTS engine?**
A: Yes! Edit tags work with F5-TTS, ChatterBox, Higgs Audio, VibeVoice, and any future engines.

**Q: Do edit tags slow down generation?**
A: Slightly. The TTS generates normally, then Step Audio EditX loads once and processes all tagged segments in batch. Total time depends on number of segments and iterations.

**Q: Can I use edit tags in SRT mode?**
A: Yes! Batch processing is automatically applied - all subtitles are generated first, then all edits are applied at once.

**Q: What if I make a typo in a tag?**
A: Unknown tags are ignored and stripped from the text. Check console for warnings.

**Q: Can I nest tags?**
A: No. Use pipe syntax or separate tags: `<tag1|tag2>` or `<tag1><tag2>`

**Q: Do edit tags work with ChatterBox special tokens?**
A: Yes, but ChatterBox v2 uses different syntax (`<giggle>`, `<whisper>` built into model). Our edit tags are post-processing, not model-native.

**Q: Will edit tags affect my voice cloning?**
A: Yes, they can. TTS generates with your voice reference normally, but each edit iteration can alter voice quality and resemblance:
- **1 iteration:** Minimal impact, usually safe
- **2 iterations:** Slight alteration, generally acceptable
- **3+ iterations:** Noticeable voice change, may lose original voice characteristics

Use the minimum iterations needed for your desired effect. A voice restoration feature using ChatterBox Voice Changer is planned but not yet implemented.

**Q: Can I disable edit tags temporarily?**
A: Just remove them from your text. No special setting needed.

---

## Support

For issues, feature requests, or questions:
- GitHub Issues: https://github.com/diodiogod/TTS-Audio-Suite/issues
- Check console output for detailed processing information
- Test with simple examples first before complex combinations

---

**Last Updated:** 2025-12-08
**Compatible with:** TTS Audio Suite v4.4.0+
