# 🎭 Character & Narrator Switching Guide

## Overview

ChatterBox Voice now supports seamless character and narrator switching with **language-aware processing** for both **F5TTS** and **ChatterBox TTS** engines. Use `[CharacterName]` or `[language:CharacterName]` tags in your text to automatically switch between different voices and languages, creating dynamic multilingual multi-character audio content.

---

## ✨ Key Features

- **🎭 Universal Character Parsing** - Works with both F5TTS and ChatterBox engines
- **🌍 Language-Aware Switching** - Switch both character AND language with `[lang:character]` syntax
- **⚙️ Per-Segment Parameters** - Control seed, temperature, cfg, and other parameters per segment with `[Alice|seed:42|temperature:0.5]` syntax (see [Parameter Switching Guide](PARAMETER_SWITCHING_GUIDE.md))
- **🔄 Robust Fallback System** - No errors when characters or languages not found
- **📁 Voice Folder Integration** - Organized character voice management
- **📺 SRT Support** - Character and language switching in subtitle timing
- **🗂️ Default Language Settings** - Set default languages per character in alias files
- **⚡ Performance Optimized** - Preserves all existing caching systems
- **🔙 Backward Compatible** - Existing workflows work unchanged

---

## 🚀 Quick Start

### 1. Text Format
Use `[CharacterName]` tags to switch voices, with optional language specification:

#### Basic Character Switching
```
Hello! This is the narrator speaking.
[Alice] Hi there! I'm Alice, nice to meet you.
[Bob] And I'm Bob! Great to meet you both.
Back to the narrator for the conclusion.
```

#### Language-Aware Character Switching (NEW!)
```
Hello! This is English narrator.
[de:Alice] Hallo! Ich bin Alice und spreche Deutsch.
[fr:Bob] Bonjour! Je suis Bob et je parle français.
[Alice] Alice now speaks in her default language.
Back to the narrator in default language.
```

**Language Tag Format:** `[language:character]`
- `[de:Alice]` - Alice speaks in German
- `[fr:Bob]` - Bob speaks in French  
- `[en:narrator]` - Narrator speaks in English
- `[Alice]` - Alice uses her default language (from alias settings or global default)

### 2. Supported Languages

#### F5-TTS Language Models
- **English** (`en`) - F5TTS_Base, F5TTS_v1_Base, E2TTS_Base
- **German** (`de`) - F5-DE
- **Spanish** (`es`) - F5-ES
- **French** (`fr`) - F5-FR
- **Japanese** (`jp`, `ja`) - F5-JP
- **Italian** (`it`) - F5-IT
- **Thai** (`th`) - F5-TH
- **Portuguese Brazilian** (`pt`, `pt-br`) - F5-PT-BR

#### ChatterBox Language Models
- **English** (`en`) - English
- **German** (`de`) - German
- **Norwegian** (`no`, `nb`, `nn`) - Norwegian

**Note:** If a language model is not available, the system will fallback to the default model with a warning.

### 3. Voice File Structure
Organize character voices using filenames in `models/voices` (preferred) or in the custom_node TTS Audio Suite folder under `voices_examples/`:

```
voices_examples/
├── narrator.wav (or .mp3, .flac, .ogg, .m4a, .aac)
├── narrator.reference.txt (for F5TTS only)
├── alice.mp3
├── alice.reference.txt (for F5TTS only)
├── bob.flac
├── bob.reference.txt (for F5TTS only)
└── characters/          (folders for organization)
    ├── female_01.wav
    ├── female_01.reference.txt
    ├── male_01.ogg
    └── male_01.reference.txt
```

**Character names are determined by the audio filename, not folder names. Folders are for organization only.**

**Supported audio formats:** `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`

### 3. Engine Differences
- **F5TTS**: Requires both **audio files** and `.reference.txt` files
- **ChatterBox**: Only needs **audio files** (simpler setup)

---

## 📖 Detailed Usage

### Character Tag Syntax
- **Format**: `[CharacterName]Text content here`
- **Case**: Character names are case-insensitive (`[Alice]` = `[alice]`)
- **Punctuation**: Automatically cleaned (`[Alice:]` → `alice`)
- **Fallback**: Unknown characters use narrator voice automatically

### Example Multi-Character Script
```
Welcome to our story! This is the narrator.

[Alice] Hello everyone! I'm excited to be here.

[Bob] Nice to meet you, Alice. I'm Bob.

[Alice] Great to meet you too, Bob!

[Wizard] I am the ancient wizard...

And the narrator concludes the tale.
```

### SRT Subtitle Example
```srt
1
00:00:01,000 --> 00:00:04,000
Hello! This is F5-TTS SRT with character switching.

2
00:00:04,500 --> 00:00:09,500
[Alice] Hi there! I'm Alice speaking with precise timing.

3
00:00:10,000 --> 00:00:14,000
[Bob] And I'm Bob! The audio matches these exact SRT timings.
```

---

## 🛠️ Setup Instructions

### For F5TTS Nodes

1. **Add Character Voice Files**:
   ```
   voices_examples/alice.wav (or .mp3, .flac, etc.)
   voices_examples/alice.reference.txt
   voices_examples/bob.mp3
   voices_examples/bob.reference.txt
   ```

2. **Voice File Requirements**:
   - `alice.wav` - Audio sample of Alice's voice (5-15 seconds recommended)
   - `alice.reference.txt` - Transcript of what Alice says in the audio
   - **Supported formats:** `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`

3. **⚠️ F5-TTS Best Practices**: Follow [F5-TTS inference guidelines](#f5-tts-inference-guidelines) to avoid generation failures

4. **Reference Text Example**:
   ```
   Hello, this is Alice speaking clearly and naturally.
   ```

### For ChatterBox Nodes

1. **Add Audio Files Only**:
   ```
   voices_examples/alice.wav (or .mp3, .flac, etc.)
   voices_examples/bob.ogg
   ```
   - No text files needed!
   - **Supported formats:** `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`

2. **Flexible Organization**:
   ```
   voices_examples/
   ├── main_characters/
   │   ├── alice.wav
   │   └── bob.mp3
   └── background_voices/
       ├── shopkeeper.flac
       └── guard.ogg
   ```

### Alternative Voice Sources
- **ComfyUI Models**: `models/voices/` directory (same filename-based system)
- **Flexible Organization**: Any subfolder structure supported for organization

### 🏷️ Character Aliases with Language Defaults (Optional)

Create a `#character_alias_map.txt` file to use friendly names and set default languages:

```
# Character Alias Map
# Supported formats: 
# Alias = Character_Name
# Alias = Character_Name, default_language
# Alias[TAB]Character_Name[TAB]default_language

# Main Characters with language defaults:
Alice		female_01		de
Bob		male_01			fr
Narrator	david_attenborough_cc3

# Supporting Cast with language defaults:
Girl = female_01, en
Woman = female_02, en

# Supporting Cast:
Girl     = female_01
Woman    = female_02
Cowboy   = clint_eastwood_cc3_enhanced2

# Background Voices:
Old Man		male_01
Shopkeeper	male_02
```

**Benefits:**
- Use `[Alice]` instead of `[female_01]` in your text
- Multiple aliases can point to the same voice file
- Flexible format: supports both `=` and tab separation
- Comments and empty lines are ignored

---

## 🔄 Fallback System

The system gracefully handles missing characters:

1. **Character Found**: Uses character-specific voice
2. **Character Not Found**: 
   - ⚠️ Warning message: `Using main voice for character 'Unknown' (not found in voice folders)`
   - 🔄 Automatically uses narrator/main reference voice
   - ✅ **No errors or workflow interruption**

### Example Fallback Behavior
```
[Alice] This uses Alice's voice (if available)
[UnknownCharacter] This falls back to narrator voice
[Bob] This uses Bob's voice (if available)
```
---

## 💡 Tips & Best Practices

### Voice Recording
- **Duration**: 5-15 seconds per character voice but it's not mandatory
- **Quality**: Clear, noise-free recordings
- **Content**: Natural speech that represents the character
- **Format**: WAV, MP3, FLAC supported

### Character Naming
- **Consistency**: Use the same character names throughout
- **Simplicity**: Avoid special characters in names
- **Organization**: Group related characters in subfolders

### Reference Text (F5TTS)
- **Accuracy**: Must match the audio exactly
- **Clarity**: Write exactly what is spoken
- **Length**: Should match audio duration

### Performance Optimization
- **Caching**: Character voices are cached automatically
- **Chunking**: Long character segments are chunked intelligently
- **Reuse**: Same character voices used across multiple generations

---

## 🐛 Troubleshooting

### Common Issues

#### "Character not found" warnings
- **Cause**: Character audio file missing or incorrectly named
- **Solution**: Check that audio filename matches character name used in `[Character]` tags
- **Note**: Supports `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac` formats
- **Result**: Uses fallback voice (no workflow break)

#### F5TTS missing reference text
- **Cause**: `.reference.txt` file missing for character
- **Solution**: Add reference text file matching audio
- **Alternative**: Use ChatterBox engine (no text required)

#### Audio quality inconsistent
- **Cause**: Different recording conditions per character
- **Solution**: Record all characters with similar setups
- **Tip**: Use consistent volume and background noise levels

### Debugging
Detailed logging to see character and language detection:
- Character switching mode messages: `🎭 F5-TTS: Character switching mode`
- Language switching mode messages: `🌍 F5-TTS: Language switching mode`
- Voice loading messages: `🎭 Using character voice for 'Alice'`
- Model switching messages: `🌍 Loading F5-TTS model 'F5-DE' for language 'de'`

---

## 📈 Advanced Usage

### Nested Character Organization
```
voices_examples/
├── story1/
│   ├── hero.wav
│   ├── hero.reference.txt
│   ├── villain.mp3
│   └── villain.reference.txt
├── story2/
│   ├── alice.flac
│   ├── alice.reference.txt
│   ├── bob.ogg
│   └── bob.reference.txt
└── narrator.m4a
└── narrator.reference.txt
```

### Language Priority System

The language selection follows this priority order:
1. **Explicit language tag**: `[de:Alice]` (highest priority)
2. **Character default language**: Set in alias file (medium priority)
3. **Global default language**: From node settings (lowest priority)

Example:
```
# If Alice has default language 'de' in alias file:
[Alice] Uses German (from alias default)
[en:Alice] Uses English (explicit override)
[fr:Alice] Uses French (explicit override)
```

### Mixed Character Scenes
```
[Narrator] The scene opens in a busy marketplace.
[Merchant] Fresh apples! Get your fresh apples here!
[Customer] How much for a dozen?
[Merchant] Two coins, good sir!
[Narrator] The customer nodded and made the purchase.
```

### Multilingual Character Scenes (NEW!)
```
[en:Narrator] Welcome to our international marketplace!
[de:Merchant] Frische Äpfel! Holt euch frische Äpfel!
[fr:Customer] Combien pour une douzaine?
[de:Merchant] Zwei Münzen, mein Herr!
[en:Narrator] The transaction concluded successfully.
```

### Dynamic Character Assignment
- Characters are detected automatically from text
- No pre-configuration needed
- Add new characters by adding audio files with matching names
- Remove characters by deleting audio files
- Character name = audio filename (without extension)
- **Supports all audio formats:** `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`

---

## 🎯 Integration Examples

### Story Narration
Perfect for audiobooks, stories, and educational content with multiple speakers.

### Dialogue Systems
Ideal for game dialogue, chatbots, and interactive applications.

### Educational Content
Great for language learning with different character voices.

### Accessibility
Helps distinguish speakers in audio content for better comprehension.

---

## 🔗 Related Features

- **[Voice Discovery System](../core/voice_discovery.py)**: Automatic character voice detection
- **[Audio Processing](../core/audio_processing.py)**: Smart audio chunking and combining
- **[SRT Integration](../chatterbox_srt/)**: Subtitle timing with character voices
- **[Caching System](../core/)**: Performance optimizations for character voices

---

## 📋 F5-TTS Inference Guidelines

To avoid possible inference failures when using F5-TTS character voices, make sure you follow these optimization guidelines:

1. **Reference Audio Duration**: Use reference audio <12s and leave proper silence space (e.g. 1s) at the end. Otherwise there is a risk of truncating in the middle of word, leading to suboptimal generation.

2. **Letter Case Handling**: Uppercased letters (best with form like K.F.C.) will be uttered letter by letter, and lowercased letters used for common words.

3. **Pause Control**: Add some spaces (blank: " ") or punctuations (e.g. "," ".") to explicitly introduce some pauses.

4. **Punctuation Spacing**: If English punctuation marks the end of a sentence, make sure there is a space " " after it. Otherwise not regarded as sentence chunk.

5. **Number Processing**: Preprocess numbers to Chinese letters if you want to have them read in Chinese, otherwise they will be read in English.

These guidelines help ensure optimal F5-TTS generation quality and prevent common audio artifacts.

---

## 📝 Version History

- **v3.0.13**: Initial character switching implementation
- **Future**: Enhanced character management UI, voice cloning improvements

---

*For technical support or feature requests, please check the main README or create an issue on GitHub.*

