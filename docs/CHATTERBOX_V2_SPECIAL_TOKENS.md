# ChatterBox Official 23-Lang v2 Special Tokens

## 🚧 EXPERIMENTAL FEATURE - LIMITED FUNCTIONALITY

**IMPORTANT: These special tokens are present in the v2 vocabulary but may not work as expected!**

Based on community testing and the lack of official documentation from ResembleAI, these emotion/sound tokens appear to be an **incomplete implementation**:

- ✅ **Tokens exist** in the v2 tokenizer vocabulary
- ⚠️ **Limited effect** - tokens may produce minimal or no audible changes
- ❌ **No official documentation** - ResembleAI has not published usage guidelines
- 🔬 **Experimental** - model may not be fully trained to respond to these tokens

**What works (partially):**
- Some users report `<laughter> hahaha` produces slight effects
- Results are inconsistent and unreliable

**Community Discussion:**
- See [ResembleAI/chatterbox Issue #186](https://github.com/resemble-ai/chatterbox/issues/186) - "Use of Emotional Tags Like [laughter] During Generation"
- No official response from ResembleAI team on proper usage

**Our implementation is ready for when/if ResembleAI improves this feature. Feel free to experiment, but don't expect production-ready results.**

---

## Overview

ChatterBox v2 vocabulary includes special tokens for emotions, sounds, and vocal effects. While the tokenizer supports these tags, the model's response to them is limited and undocumented.

## ⚠️ IMPORTANT: Use Angle Brackets `<tag>`

To avoid conflicts with the character switching system `[CharacterName]`, **use angle brackets `<>` for v2 special tokens**:

- ✅ **Correct**: `<giggle>`, `<sigh>`, `<whisper>`
- ❌ **Wrong**: `[giggle]`, `[sigh]`, `[whisper]` (conflicts with character names)

The system will automatically convert `<emotion>` → `[emotion]` internally for ChatterBox v2.

## New Special Tokens

### Emotional Expressions
- `<giggle>` - Light laughter
- `<laughter>` - Full laughter
- `<guffaw>` - Loud, boisterous laugh
- `<sigh>` - Sighing sound
- `<cry>` - Crying sound
- `<gasp>` - Gasping sound
- `<groan>` - Groaning sound

### Breathing & Speech Modifiers
- `<inhale>` - Inhaling/breath in
- `<exhale>` - Exhaling/breath out
- `<whisper>` - Whispered speech
- `<mumble>` - Mumbled speech
- `<UH>` - Hesitation sound (uh)
- `<UM>` - Hesitation sound (um)

### Vocal Performances
- `<singing>` - Singing voice
- `<music>` - Musical sounds
- `<humming>` - Humming sound
- `<whistle>` - Whistling sound

### Body Sounds
- `<cough>` - Coughing sound
- `<sneeze>` - Sneezing sound
- `<sniff>` - Sniffing sound
- `<snore>` - Snoring sound
- `<clear_throat>` - Throat clearing
- `<chew>` - Chewing sound
- `<sip>` - Sipping/drinking sound
- `<kiss>` - Kissing sound

### Animal Sounds
- `<bark>` - Dog barking
- `<howl>` - Howling sound
- `<meow>` - Cat meowing

### Other
- `<shhh>` - Shushing sound
- `<gibberish>` - Nonsensical speech

## Usage Examples

```
Hello there! <giggle> I'm so happy to see you.
Wait... <UM> I think I forgot something. <sigh>
<whisper> This is a secret message.
Look at that! <gasp> It's amazing!
<singing> La la la la la!
```

## Important Notes

### ✅ No Conflicts - Safe to Use!

The angle bracket syntax `<emotion>` is specifically designed to avoid conflicts:

- **Character switching** uses `[CharacterName]` - no conflict
- **Pause tags** use `[pause:2]`, `[wait:1.5]` - no conflict
- **v2 special tokens** use `<giggle>`, `<sigh>` - no conflict

### Combining with Other Systems

You can freely mix all three systems:

```
[Alice] Hello! <giggle> Nice to meet you. [pause:0.5] How are you? <whisper> I have a secret.
[Bob] <gasp> Really? Tell me more! [wait:1]
```

**Processing order:**
1. Character tags `[CharacterName]` are extracted first
2. Pause tags `[pause:XX]` are processed second
3. v2 special tags `<emotion>` are converted last (to `[emotion]` for the engine)

This ensures everything works together seamlessly!

## Model Version Selection

To use these special tokens:

1. Set **model_version** to **v2** in the ChatterBox Official 23-Lang Engine node
2. The v2 model will automatically download the enhanced tokenizer files
3. Special tokens will be processed during generation

## Technical Details

### New v2 Model Files
- `t3_mtl23ls_v2.safetensors` - Enhanced T3 model with improved tokenization
- `grapheme_mtl_merged_expanded_v1.json` - Enhanced grapheme/phoneme mappings with 118 special tokens
- Improved Russian stress handling
- Enhanced multilingual tokenizer fixes

### Cache Invalidation

The TTS Audio Suite automatically manages cache keys to prevent conflicts:
- v1 and v2 generations are cached separately
- Switching between versions will regenerate audio with the correct model
- Cache keys include `model_version` to ensure proper invalidation
