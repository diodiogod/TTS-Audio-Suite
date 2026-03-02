<a id="readme-top"></a>

[![](https://dcbadge.limes.pink/api/server/EwKE8KBDqD)](https://discord.gg/EwKE8KBDqD)
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Forks][forks-shield]][forks-url]
[![Dynamic TOML Badge][version-shield]][version-url]
[![Ko-Fi](https://img.shields.io/badge/Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/diogogo)

# TTS Audio Suite v4.22.1

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/diogogo)

*Universal multi-engine TTS extension for ComfyUI - evolved from the original [ChatterBox Voice project](https://github.com/diodiogod/ComfyUI_ChatterBox_SRT_Voice).*

<div align="center">
  <img src="images/AllNodesShowcase.jpg" alt="TTS Audio Suite Nodes Showcase" />
</div>

A comprehensive ComfyUI extension providing unified Text-to-Speech, Voice Conversion, and Audio Editing capabilities through multiple engines including ChatterboxTTS, F5-TTS, Higgs Audio 2, Step Audio EditX, and RVC (Real-time Voice Conversion), with modular architecture designed for extensibility and future engine integrations.

<!-- ENGINE_COMPARISON_START -->

## Quick Engine Comparison — 11 Engines

| Engine | Languages | Size | Key Features |
|--------|-----------|------|--------------|
| **F5-TTS** | 🇺🇸​🇩🇪​🇪🇸​🇫🇷​🇮🇹​🇯🇵 +4 | ~1.2GB each | Targeted Word/Speech Editing, Speed control |
| **ChatterBox** | 🇺🇸​🇩🇪​🇫🇷​🇮🇹​🇯🇵​🇰🇷 +4 | ~4.3GB | Expressiveness slider |
| **ChatterBox 23L** | 🌐 24 languages | ~4.3GB | 24 languages in single model, emotion tokens (v2 - doesn't work) |
| **VibeVoice** | 🇺🇸​🇨🇳​🇩🇪​🇪🇸​🇫🇷​🇮🇹 +21 | 5.4GB / 18GB | 90-min long-form, Native 4-speaker (Base models) |
| **Higgs Audio 2** | 🇺🇸​🇨🇳​🇩🇪​🇪🇸​🇰🇷 | ~9GB | 3 multi-speaker, CUDA graphs (55+ tokens/sec) |
| **IndexTTS-2** | 🇺🇸​🇨🇳​🇯🇵 | ~4.7GB | Emotion Control: 8 vectors, Text as reference |
| **CosyVoice3** | 🇺🇸​🇨🇳​🇯🇵​🇰🇷 | ~5.4GB | Paralinguistic tags |
| **Qwen3-TTS** | 🇺🇸​🇨🇳​🇩🇪​🇪🇸​🇫🇷​🇮🇹 +4 | ~3-6GB | Voice design, ASR (Automatic Speech Recognition) |
| **Step Audio EditX** | 🇺🇸​🇨🇳​🇯🇵​🇰🇷 | ~7GB | Second Pass Speech Editing Node: 14 emotions, 32 speaking styles |
| **Echo-TTS** | 🇺🇸 | ~5.3GB + ~1.8GB | Diffusion-based (~30s best), Force Speaker KV (speaker drift control) |
| **RVC** | 🌐 Any | 100-300MB | Real-time VC, Pitch shift (±14) |

📊 **[Full comparison tables →](docs/ENGINE_COMPARISON.md)** | **[Language matrix →](docs/LANGUAGE_SUPPORT.md)** | **[Feature matrix →](docs/FEATURE_COMPARISON.md)** | **[Model download sources →](docs/MODEL_DOWNLOAD_SOURCES.md)** | **[Model folder layouts →](docs/MODEL_LAYOUTS.md)**

*Note: These tables are generated automatically from source: [tts_audio_suite_engines.yaml](docs/Dev%20reports/tts_audio_suite_engines.yaml)*

<!-- ENGINE_COMPARISON_END -->

## 🚀 Project Evolution Timeline

```
🎭 ChatterBox Voice Era                     🌟 Multi-Engine Era
|                                                      |        
v1.0 ───────────► v1.1 ────────► v2.0 ──────────► v3.0 ─────────┐
Jun 25            Jun 25         Jun 25           Jul 25        │
│                 │              │                │             │
Foundation        SRT            Modular          F5-TTS +      │
ChatterBox        Subtitles      Structure        Audio         │
Voice Cloning     Timing Node    Refactor         Analyzer      │
                                                                ▼
v3.4 ◄──────────────── v3.2 ◄──────────────── v3.1 ◄────────────┘
Jul 25                 Jul 25                 Jul 25             
│                      │                      │                  
Language               Pause                  Character          
Switching              Tags                   Switching          
[German:Bob]           [pause:1s]             [Alice]            
│                                                  
│         ⚙️ TTS Audio Suite Era                                
▼         |                                   
v4.0 ──────────► v4.3 ──────────► v4.4 ────────► v4.5 ──────────┐
Aug 25           Aug 25           Aug 25         Aug 25         │
│                │                │              │              │
BREAKING!        RVC +            Silent         Higgs Audio 2  │
Project          Voice            Speech         New TTS Engine │
Renamed          Conversion       Analyzer       Voice Cloning  │
TTS Audio Suite  + Streaming                                    │
                                                                ▼
v4.9 ◄─────────────── v4.8 ◄────────────────── v4.6 ◄───────────┘
Sep 25                Sep 25                    Aug 25
│                     │                         │
IndexTTS-2            Chatterbox                VibeVoice
Emotion               Multilingual              New TTS Engine
Control               Official (23-lang)        90min Generation
│
│             🎨 Inline Editor Tags Era
▼                            |
v4.12 ──────────────► v4.15 ────────────► v4.16 ───────────────┐
Oct 25                Dez 25              Dez 25               │
│                     │                   │                    │
Per-Seg Parameter     Step Audio EditX    CosyVoice3           │
Switching [seed:24]   Inline Edit tags    TTS + VC             │
                      <laughter:2>                             │
                                                               ▼
      v4.22 ◄──────────────────── v4.19 ◄──────────────────────┘
      Mar 26                      Jan 26
      │                           │
      Echo-TTS                    Qwen3-TTS
      English TTS                 TTS + ASR
                                  VoiceDesign
```

<details>
<summary><h2>📋 Table of Contents</h2></summary>

- [🎥 Demo Videos](#-demo-videos)
- [Features](#features)
- [🆕 What's New in my Project?](#-whats-new-in-my-project)
  - [SRT Timing and TTS Node](#srt-timing-and-tts-node)
  - [🆕 F5-TTS Integration and 🆕 Audio Analyzer](#-f5-tts-integration-and--audio-analyzer)
  - [🗣️ Silent Speech Analyzer](#️-silent-speech-analyzer)
  - [🎭 Character & Narrator Switching](#-character--narrator-switching)
  - [🌍 Language Switching with Bracket Syntax](#-language-switching-with-bracket-syntax)
  - [🔄 Iterative Voice Conversion](#-iterative-voice-conversion)
  - [🎵 RVC Voice Conversion Integration](#-rvc-voice-conversion-integration)
  - [⏸️ Pause Tags System](#️-pause-tags-system)
  - [🌍 Multi-language ChatterBox Community Models](#-multi-language-chatterbox-community-models)
  - [🌐 Chatterbox Multilingual TTS (Official 23-Lang)](#-chatterbox-multilingual-tts-official-23-lang)
  - [⚙️ Universal Streaming Architecture](#️-universal-streaming-architecture)
  - [🎙️ Higgs Audio 2 Voice Cloning](#️-higgs-audio-2-voice-cloning)
  - [🎵 VibeVoice Long-Form Generation](#-vibevoice-long-form-generation)
  - [ IndexTTS-2 With Emotion Control](#-indextts-2-with-emotion-control)
  - [🎨 Step Audio EditX - LLM Audio Editing](#-step-audio-editx---llm-audio-editing)
  - [🗣️ CosyVoice3 Multilingual Voice Cloning](#️-cosyvoice3-multilingual-voice-cloning)
  - [🎤 Qwen3-TTS - 3 Model Types with Text-to-Voice Design](#-qwen3-tts---3-model-types-with-text-to-voice-design)
  - [🎧 Echo-TTS Voice Cloning](#-echo-tts-voice-cloning)
  - [📝 Phoneme Text Normalizer](#-phoneme-text-normalizer)
  - [🏷️ Multiline TTS Tag Editor & Per-Segment Parameter Switching](#️-multiline-tts-tag-editor--per-segment-parameter-switching)
- [🚀 Quick Start](#-quick-start)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installation Methods](#installation-methods)
  - [Troubleshooting Dependency Issues](#troubleshooting-dependency-issues)
  - [Updating the Node](#updating-the-node)
- [Enhanced Features](#enhanced-features)
- [Usage](#usage)
  - [Voice Recording](#voice-recording)
  - [Enhanced Text-to-Speech](#enhanced-text-to-speech)
  - [F5-TTS Voice Synthesis](#f5-tts-voice-synthesis)
  - [Voice Conversion with Iterative Refinement](#voice-conversion-with-iterative-refinement)
- [📁 Example Workflows](#-example-workflows)
- [Settings Guide](#settings-guide)
- [Text Processing Capabilities](#text-processing-capabilities)
- [License](#license)
- [Credits](#credits)
- [🔗 Links](#-links)

</details>

## 🎥 Demo Videos

<div align="center">
  <a href="https://youtu.be/aHz1mQ2bvEY">
    <img src="https://img.youtube.com/vi/aHz1mQ2bvEY/maxresdefault.jpg" width="400" alt="ChatterBox SRT Voice v3.2 - F5-TTS Integration & Features Overview">
  </a>
  <br>
  <strong><a href="https://youtu.be/aHz1mQ2bvEY">▶️ v3.2 Features Overview (20min) - F5-TTS Integration, Speech Editor & More!</a></strong>
</div>

<br>

<div align="center">
  <a href="https://youtu.be/VyOawMrCB1g?si=7BubljRhsudGqG3s">
    <img src="https://img.youtube.com/vi/VyOawMrCB1g/maxresdefault.jpg" width="400" alt="ChatterBox SRT Voice Demo">
  </a>
  <br>
  <strong><a href="https://youtu.be/VyOawMrCB1g?si=7BubljRhsudGqG3s">▶️ Original Demo - SRT Timing & Basic Features</a></strong>
</div>

<details>
<summary><h3>📜 Original ShmuelRonen ChatterBox TTS Nodes</h3></summary>

<div align="center">
  <img src="https://github.com/user-attachments/assets/4197818c-8093-4da4-abd5-577943ac902c" width="45%" alt="ChatterBox TTS Nodes" />
  <img src="https://github.com/user-attachments/assets/701c219b-12ff-4567-b414-e58560594ffe" width="45%" alt="ChatterBox Voice Capture" />
</div>

* **Voice Recording**: Smart silence detection for voice capture
* **Enhanced Chunking**: Intelligent text splitting with multiple combination methods
* **Unlimited Text Length**: No character limits with smart processing

**Original creator:** [ShmuelRonen](https://github.com/ShmuelRonen/ComfyUI_ChatterBox_Voice)

</details>

<div align="right"><a href="#-table-of-contents">Back to top</a></div>

## Features

- 🎤 **Multi-Engine TTS** - ChatterBox TTS, **Chatterbox Multilingual TTS**, F5-TTS, Higgs Audio 2, VibeVoice, **IndexTTS-2**, **CosyVoice3**, **Qwen3-TTS**, and **Echo-TTS** with voice cloning, reference audio synthesis, and production-grade quality
- ✏️ **ASR Transcription** - Qwen3-ASR via the ✏️ ASR Transcribe node (more engines planned)
- 🎨 **Audio Post-Processing** - **Step Audio EditX** LLM-based audio editing with paralinguistic effects (laughter, breathing, sigh), emotion control (14 emotions), speaking styles (32 styles), speed adjustment, and voice restoration → **[📖 Inline Edit Tags Guide](docs/INLINE_EDIT_TAGS_USER_GUIDE.md)**
- 🔄 **Voice Conversion** - ChatterBox VC with iterative refinement + RVC real-time conversion using .pth character models
- 🎙️ **Voice Capture & Recording** - Smart silence detection and voice input recording
- 🎭 **Character & Language Switching** - Multi-character TTS with `[CharacterName]` tags, alias system, and `[language:character]` syntax for seamless model switching
- 🏷️ **Multiline TTS Tag Editor & Per-Segment Parameter Switching** - Override generation parameters (seed, temperature, CFG, speed, etc.) on a per-segment basis using a new multiline string editor node that makes building complex tags easier and more visual, with character/language/parameter dropdowns for quick selection, preset management, and tag validation → **[📖 Complete Guide](docs/PARAMETER_SWITCHING_GUIDE.md)**
- 🌍 **Multi-language Support** - **Chatterbox Multilingual TTS (Arabic, Danish, German, Greek, English, Spanish, Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, Chinese)** + ChatterBox community models (English, German, Italian, French, Russian, Armenian, Georgian, Japanese, Korean, Norwegian) + F5-TTS (English, German, Spanish, French, Japanese, Hindi, and more)
- 📝 **Multilingual Text Processing** - Universal Phoneme Text Normalizer with IPA phonemization, Unicode decomposition, and character mapping for improved pronunciation quality across languages (Experimental)
- 😤 **Emotion Control** - ChatterBox exaggeration parameter for expressive speech + IndexTTS-2 advanced emotion control with dynamic text analysis, character tags, and 8-emotion vectors → **[📖 IndexTTS-2 Guide](docs/IndexTTS2_Emotion_Control_Guide.md)**
- 📝 **Enhanced Chunking** - Intelligent text splitting for long content with multiple combination methods
- 🎵 **Advanced Audio Processing** - Optional FFmpeg support for premium audio quality with graceful fallback
- 🤐 **Vocal/Noise Removal** - AI-powered vocal separation, noise reduction, and echo removal with GPU acceleration → **[📖 Complete Guide](docs/VOCAL_REMOVAL_GUIDE.md)**
- 🌊 **Audio Wave Analyzer** - Interactive waveform visualization and precise timing extraction for F5-TTS workflows → **[📖 Complete Guide](docs/🌊_Audio_Wave_Analyzer-Complete_User_Guide.md)**
- 🗣️ **Silent Speech Analyzer** - Video analysis with experimental viseme detection, mouth movement tracking, and base SRT timing generation from silent video using MediaPipe
- ⚙️ **Parallel Processing** - Configurable worker-based processing via `batch_size` parameter (Note: sequential processing with `batch_size=0` remains optimal for performance)
- ⚡ **Performance Optimizations** - Qwen3-TTS supports torch.compile for ~1.7x speedup (requires PyTorch 2.10+ and triton-windows 3.6+) → **[📖 Optimization Guide](docs/qwen3_tts_optimizations.md)**

<div align="right"><a href="#-table-of-contents">Back to top</a></div>

<summary><h2>🆕 What's New in my Project? (click to expand)</h2></summary>

<details>
<summary><h3>📺 SRT Timing and TTS Node</h3></summary>

<img title="" src="images/srt.png" alt="SRT Node Screenshot" width="500" data-align="center">

The **"ChatterBox SRT Voice TTS"** node allows TTS generation by processing SRT content (SubRip Subtitle) files, ensuring precise timing and synchronization with your audio.

**Key SRT Features:**

* **SRT style Processing**: Uses SRT style to generate TTS, aligning audio with subtitle timings
* **`smart_natural` Timing Mode**: Intelligent shifting logic that prevents overlaps and ensures natural speech flow
* **`Adjusted_SRT` Output**: Provides actual timings for generated audio for accurate post-processing
* **Segment-Level Caching**: Only regenerates modified segments, significantly speeding up workflows

For comprehensive technical information, refer to the [SRT_IMPLEMENTATION.md](docs/Dev%20reports/SRT_IMPLEMENTATION.md) file.

</details>

<details>
<summary><h3>🆕 F5-TTS Integration and 🆕 Audio Analyzer</h3></summary>

<img title="" src="images/waveanalgif.gif" alt="Audio Wave gif" width="500" data-align="center">

* **F5-TTS Voice Synthesis**: High-quality voice cloning with reference audio + text
* **Audio Wave Analyzer**: Interactive waveform visualization for precise timing extraction
* **Multi-language Support**: English, German, Spanish, French, Japanese models
* **Speech Editing Workflows**: Advanced F5-TTS editing capabilities

</details>

<details>
<summary><h3>🗣️ Silent Speech Analyzer</h3></summary>

**NEW in v4.4.0**: Video analysis and mouth movement detection for silent video processing!

* **Mouth Movement Analysis**: Real-time detection of mouth shapes and movements from video
* **Experimental Viseme Classification**: Approximate detection of vowels (A, E, I, O, U) and consonants (B, F, M, etc.) - results are experimental approximations, not precise
* **3-Level Analysis System**:
  - Frame-level mouth movement detection
  - Syllable grouping with temporal analysis  
  - Word prediction using CMU Pronouncing Dictionary (135K+ words)
* **Base SRT Generation**: Creates timing-focused SRT files with start/end speech timing as foundation for user editing
* **MediaPipe Integration**: Production-ready analysis using Google's MediaPipe framework
* **Visual Feedback**: Preview videos with overlaid detection results
* **Automatic Phonetic Placeholders**: Word predictions provide phonetically-sensible placeholders, but phrases require user editing for meaningful content
* **TTS Integration**: SRT output designed for use with TTS SRT nodes after manual content editing

**Perfect for:**

- Creating base timing templates from silent video footage
- Animation and VFX reference timing
- Foundation for manual subtitle creation

**Important Notes**: 

- OpenSeeFace provider is experimental and not recommended for production use - MediaPipe is the stable solution
- Viseme detection is experimental approximation - expect to manually edit both timing and content
- Generated text placeholders are phonetic suggestions, not meaningful sentences

</details>

<details>
<summary><h3>🎙️ Higgs Audio 2 Voice Cloning</h3></summary>

**NEW in v4.5.0**: State-of-the-art voice cloning technology with advanced neural voice replication!

* **High-Quality Voice Cloning**: Clone any voice from 30+ second reference audio with exceptional fidelity
* **Multi-Speaker Conversations**: Native support for character switching within conversations
* **Real-Time Processing**: Generate speech in cloned voices with minimal latency
* **Universal Integration**: Works seamlessly with existing TTS Text and TTS SRT nodes

**Key Capabilities:**

- **Voice Cloning from Reference Audio**: Upload any 30+ second audio file for voice replication
- **Multi-Language Support**: English (tested), with potential support for Chinese, Korean, German, and Spanish (based on model training data)
- **Character Switching**: Use `[CharacterName]` syntax for multi-speaker dialogues
- **Advanced Generation Control**: Fine-tune temperature, top-p, top-k, and token limits
- **Smart Chunking**: Automatic handling of unlimited text length with seamless audio combination
- **Intelligent Caching**: Instant regeneration of previously processed content

**Technical Features:**

- **Modular Architecture**: Clean integration with unified TTS system
- **Automatic Model Management**: Downloads and organizes models in `ComfyUI/models/TTS/HiggsAudio/` structure
- **Progress Tracking**: Real-time generation feedback with tqdm progress bars
- **Voice Reference Discovery**: Flexible voice file management system

**Quick Start:**

1. Add `Higgs Audio Engine` node to configure voice cloning parameters
2. Connect to `TTS Text` or `TTS SRT` node for generation
3. Specify reference audio file or use voice discovery system
4. Generate high-quality cloned speech with automatic optimization

**Perfect for:**

- Voice acting and character dialogue creation
- Audiobook narration with consistent voice characteristics
- Multi-speaker content with distinct voice personalities
- Professional voice replication for content creation

</details>

<details>
<summary><h3>🎵 VibeVoice Long-Form Generation</h3></summary>

- **Custom Character Switching**: Use `[Alice]`, `[Bob]` character tags with voice files from the voices folder - supports unlimited characters with pause tags and per-character control
- **Native Multi-Speaker**: Efficient single-pass generation supporting both `[Character]` tag auto-conversion and manual "Speaker 1: Hello" format for up to 4 speakers  
- **Voice File Integration**: Seamless compatibility with existing voice folder structure and Character Voices node
- **Smart Chunking**: Automatic text chunking with configurable time-based limits for memory efficiency
- **Priority System**: Connected speaker2/3/4_voice inputs override character aliases with intelligent warnings

**Technical Features:**

- **Model Support**: Microsoft vibevoice-1.5B/7B (official models for English/Chinese) + vibevoice-hindi-1.5B/7B (community Hindi finetunes) + **KugelAudio-0-Open (multilingual 7B variant)**
- **Language Detection**: VibeVoice/KugelAudio automatically detect language from input text and reference audio - **no language parameters are used**
- **Intelligent Caching**: Advanced caching system with mode-aware invalidation for instant regeneration
- **Memory Optimization**: Configurable chunking system balances quality with memory usage
- **Unified Architecture**: Seamless integration with existing TTS Text and TTS SRT nodes

**Quick Start:**

1. Add `⚙️ VibeVoice Engine` node to configure model and multi-speaker mode  
2. Connect to `TTS Text` or `TTS SRT` node for generation
3. Choose between Custom Character Switching (recommended) or Native Multi-Speaker mode
4. Generate long-form content with automatic voice cloning from your voices folder

**Perfect for:**

- Long-form audiobooks and narration with consistent voice quality
- Multi-character dialogue and conversations with distinct speaker voices  
- Extended podcast-style content with natural speech patterns
- Educational content requiring extended generation without quality degradation

</details>

<details>
<summary><h3>🎭 Character & Narrator Switching</h3></summary>

**NEW in v3.1.0**: Seamless character switching for both F5TTS and ChatterBox engines!

* **Multi-Character Support**: Use `[CharacterName]` tags to switch between different voices
* **Voice Folder Integration**: Organized character voice management system
* **🏷️ Character Aliases**: User-friendly alias system - use `[Alice]` instead of `[female_01]` with `#character_alias_map.txt`
* **Robust Fallback**: Graceful handling when characters not found (no errors!)
* **Universal Compatibility**: Works with both F5TTS and ChatterBox TTS engines
* **SRT Integration**: Character switching within subtitle timing
* **Backward Compatible**: Existing workflows work unchanged

**📖 [Complete Character Switching Guide](docs/CHARACTER_SWITCHING_GUIDE.md)**

Example usage:

```
Hello! This is the narrator speaking.
[Alice] Hi there! I'm Alice, nice to meet you.
[Bob] And I'm Bob! Great to meet you both.
Back to the narrator for the conclusion.
```

</details>

<details>
<summary><h3>🌍 Language Switching with Bracket Syntax</h3></summary>

**NEW in v3.4.0**: Seamless language switching using simple bracket notation!

* **Language Code Syntax**: Use `[language:character]` tags to switch languages and models automatically
* **Smart Model Loading**: Automatically loads correct language models (F5-DE, F5-FR, German, Norwegian, etc.)
* **Flexible Aliases** *(v3.4.3)*: Support for `[German:Alice]`, `[Brazil:Bob]`, `[USA:]`, `[Portugal:]` - no need to remember language codes!
* **Standard Format**: Also supports traditional `[fr:Alice]`, `[de:Bob]`, or `[es:]` (language only) patterns
* **Character Integration**: Combines perfectly with character switching and alias system
* **Performance Optimized**: Language groups processed efficiently to minimize model switching
* **Alias Support**: Language defaults work with character alias system

**Supported Languages:**

* **F5-TTS**: English (en), German (de), Spanish (es), French (fr), Italian (it), Japanese (jp), Thai (th), Portuguese (pt), Hindi (hi)
* **ChatterBox**: English (en), German (de, de-best, de-expressive), Italian (it), French (fr), Russian (ru), Armenian (hy), Georgian (ka), Japanese (ja), Korean (ko), Norwegian (no/nb/nn)
* **ChatterBox 23-Lang** & **Qwen3-TTS**: Use explicit language parameters - language tags directly control output
* **VibeVoice/KugelAudio**: Do NOT use language parameters - auto-detect from text (language tags have no effect on these models)

Example usage:

```
Hello! This is English text with the default model.
[de:Alice] Hallo! Ich spreche Deutsch mit Alice's Stimme.
[fr:] Bonjour! Je parle français avec la voix du narrateur.
[es:Bob] ¡Hola! Soy Bob hablando en español.
Back to English with the original model.
```

**Advanced SRT Integration:**

```srt
1
00:00:01,000 --> 00:00:04,000
Hello! Welcome to our multilingual show.

2
00:00:04,500 --> 00:00:08,000
[de:female_01] Willkommen zu unserer mehrsprachigen Show!

3
00:00:08,500 --> 00:00:12,000
[fr:] Bienvenue à notre émission multilingue!
```

</details>

<details>
<summary><h3>🔄 Iterative Voice Conversion</h3></summary>

**NEW**: Progressive voice refinement with intelligent caching for instant experimentation!

* **Refinement Passes**: Multiple conversion iterations (1-30, recommended 1-5)
* **Smart Caching**: Results cached up to 5 iterations - change from 5→3→4 passes instantly
* **Progressive Quality**: Each pass refines output to sound more like target voice

</details>

<details>
<summary><h3>🎵 RVC Voice Conversion Integration</h3></summary>

**NEW in v4.1.0**: Professional-grade Real-time Voice Conversion with .pth character models!

* **RVC Character Models**: Load .pth voice models with 🎭 Load RVC Character Model node
* **Unified Voice Changer**: Full RVC integration in the Voice Changer node
* **Iterative Refinement**: 1-30 passes with smart caching (like ChatterBox)
* **Enhanced Quality**: Automatic .index file loading for improved voice similarity
* **Auto-Download**: Required models download from official sources automatically
* **Cache Intelligence**: Skip recomputation - change 5→3→4 passes instantly
* **Neural Network Quality**: High-quality voice conversion using trained RVC models

📖 **See [Model folder layouts](docs/MODEL_LAYOUTS.md#rvc) for detailed setup paths**

**How it works:**

1. Load your .pth RVC model with 🎭 Load RVC Character Model
2. Connect to 🔄 Voice Changer, select "RVC" engine
3. Process with iterative refinement for progressive quality improvement
4. Results cached for instant experimentation with different pass counts

</details>

<details>
<summary><h3>⏸️ Pause Tags System</h3></summary>

**NEW**: Intelligent pause insertion for natural speech timing control!

* **Smart Pause Syntax**: Use pause tags anywhere in your text with multiple aliases
* **Flexible Duration Formats**: 
  - Seconds: `[pause:1.5]`, `[wait:2s]`, `[stop:3]`
  - Milliseconds: `[pause:500ms]`, `[wait:1200ms]`, `[stop:800ms]`
  - Supported aliases: `pause`, `wait`, `stop` (all work identically)
* **Character Integration**: Pause tags work seamlessly with character switching
* **Intelligent Caching**: Changing pause durations won't regenerate unchanged text segments
* **Universal Support**: Works across all TTS nodes (ChatterBox, F5-TTS, SRT)
* **Automatic Processing**: No additional parameters needed - just add tags to your text

Example usage:

```
Welcome to our show! [pause:1s] Today we'll discuss exciting topics.
[Alice] I'm really excited! [wait:500ms] This will be great.
[stop:2] Let's get started with the main content.
```

</details>

<details>
<summary><h3>🌍 Multi-language ChatterBox Community Models</h3></summary>

**NEW in v4.6.29**: ChatterBox TTS now supports 11 languages with community-finetuned models and automatic model management!

**Supported Languages:**

- 🇺🇸 **English**: Original ResembleAI model (default)
- 🇩🇪 **German**: Three variants available:
  - Standard German (stlohrey/chatterbox_de)
  - German Best (havok2) - Multi-speaker hybrid, best quality
  - German Expressive (SebastianBodza) - Emotion control with `<haha>`, `<wow>` tags
- 🇮🇹 **Italian**: Bilingual Italian/English model with `[it]` prefix for Italian text
- 🇫🇷 **French**: 1,400 hours Emilia dataset with zero-shot voice cloning
- 🇷🇺 **Russian**: Complete model with training artifacts
- 🇦🇲 **Armenian**: Complete model with unique architecture
- 🇬🇪 **Georgian**: Complete model with specialized features
- 🇯🇵 **Japanese**: Uses shared English components with Japanese text processing
- 🇰🇷 **Korean**: Uses shared English components with Korean text processing
- 🇳🇴 **Norwegian**: Norwegian ChatterBox model (akhbar/chatterbox-tts-norwegian)

**Key Features:**

* **Language Dropdown**: Simple language selection in all ChatterBox nodes
* **Auto-Download**: Models download automatically on first use (~1GB per language)
* **Local Priority**: Prefers locally installed models over downloads for offline use
* **Safetensors Support**: Modern format support for newer language models
* **Seamless Integration**: Works with existing workflows - just select your language

**Usage**: Select language from dropdown → First generation downloads model → Subsequent generations use cached model

</details>

<details>
<summary><h3>🌐 Chatterbox Multilingual TTS (Official 23-Lang)</h3></summary>

**NEW in v4.8.0**: Official ResembleAI Chatterbox Multilingual TTS model with native support for 23 languages!

The **Chatterbox Multilingual TTS** (referred to internally as "ChatterBox Official 23-Lang" to distinguish from community models) is ResembleAI's first production-grade open-source TTS model supporting 23 languages out of the box. This is the official successor to the original ChatterBox model with enhanced multilingual capabilities.

**🎯 Key Advantages over Community Models:**

* **Single Unified Model**: One model handles all 23 languages - no model switching required
* **Language Parameter Switching**: Changes language via parameter, not model loading (faster)
* **Zero-Shot Voice Cloning**: Clone any voice with just a few seconds of reference audio across all languages
* **Production-Grade Quality**: Benchmarked against leading closed-source systems like ElevenLabs
* **MIT Licensed**: Fully open-source with commercial usage rights
* **Perth Watermarking**: Built-in responsible AI usage (disabled by default for compatibility)

**🌍 Supported Languages (23 total + Vietnamese community finetune):**

Arabic (ar), Danish (da), German (de), Greek (el), English (en), Spanish (es), Finnish (fi), French (fr), Hebrew (he), Hindi (hi), Italian (it), Japanese (ja), Korean (ko), Malay (ms), Dutch (nl), Norwegian (no), Polish (pl), Portuguese (pt), Russian (ru), Swedish (sv), Swahili (sw), Turkish (tr), Chinese (zh)

**🇻🇳 Vietnamese (Viterbox)**: Community finetune by Dolly AI 23 with expanded Vietnamese tokenization (dolly-vn/viterbox) - select from model version dropdown

**🔧 Fully Integrated Features:**

* ✅ **Character Switching**: Full `[CharacterName]` support with per-character voice references
* ✅ **Language Switching**: `[language:character]` syntax with intelligent parameter switching
* ✅ **Pause Tags**: Complete `[pause:Ns]` support with character voice inheritance  
* ✅ **SRT Processing**: Advanced subtitle timing with overlapping modes and audio assembly
* ✅ **Voice Conversion**: Built-in VC engine supporting all 23 languages
* ✅ **Emotion Control**: Unique exaggeration parameter for expressive speech
* ✅ **Advanced Parameters**: Full control over repetition_penalty, min_p, top_p for fine-tuning
* ✅ **Cache Invalidation**: Proper parameter-based caching for responsive generation

**🆚 vs Community Models:**

| Feature               | Chatterbox Multilingual TTS       | Community Models            |
| --------------------- | --------------------------------- | --------------------------- |
| Languages             | 23 native languages               | 11 finetuned variants       |
| Model Loading         | Single model, parameter switching | Separate model per language |
| Voice Cloning         | Zero-shot across all languages    | Per-model training          |
| Official Support      | ✅ ResembleAI official             | Community maintained        |
| Character Integration | ✅ Full integration                | ✅ Full integration          |
| SRT Support           | ✅ Advanced timing modes           | ✅ Advanced timing modes     |
| Performance           | Optimized single-model            | Multiple model overhead     |

**🎭 Character Example:**

```
[En:Alice] Hello everyone! [De:Hans] Guten Tag! [Es:Maria] ¡Hola! [pause:2s] [En:Alice] That was amazing multilingual switching!
```

This creates seamless multilingual character switching with proper voice inheritance and pause support - all within a single model.

**🎭 NEW: v2 Special Emotion & Sound Tokens** 🚧 *Experimental*

ChatterBox v2 vocabulary includes 30+ special tokens for emotions, sounds, and vocal effects. **Note: These are experimental - tokens may produce minimal or no audible effects.** ResembleAI has not officially documented their usage ([see issue #186](https://github.com/resemble-ai/chatterbox/issues/186)).

Try angle brackets `<emotion>` to experiment:

```
[Alice] Hello! <laughter> hahaha. [pause:0.5] <whisper> This might work slightly.
```

**Available v2 Tokens:**

- **Emotions**: `<giggle>`, `<laughter>`, `<sigh>`, `<cry>`, `<gasp>`, `<groan>`
- **Speech Modifiers**: `<whisper>`, `<mumble>`, `<singing>`, `<humming>`
- **Sounds**: `<cough>`, `<sneeze>`, `<sniff>`, `<inhale>`, `<exhale>`
- **And more!** See the **[📖 Complete v2 Special Tokens Guide](docs/CHATTERBOX_V2_SPECIAL_TOKENS.md)** for all 30+ tokens

**Model Version Selection:**

- **v2** (default): Enhanced tokenization with experimental emotion/sound tokens
- **v1**: Original model without special tokens

Both versions fully support character switching, language switching, and pause tags. The v2 special tokens are **experimental with limited effectiveness** - our implementation is ready for when/if ResembleAI improves this feature. The angle bracket syntax `<emotion>` avoids conflicts with character tags `[Name]` and pause tags `[pause:1s]`.

</details>

<details>
<summary><h3>⚙️ Universal Streaming Architecture</h3></summary>

**NEW in v4.3.0**: Complete architectural overhaul implementing universal streaming system with parallel processing capabilities!

**Key Features:**

* **Universal Streaming Infrastructure**: Unified processing system eliminating engine-specific code complexity
* **Parallel Processing**: Configurable worker-based processing via `batch_size` parameter
* **Thread-Safe Design**: Stateless wrapper architecture eliminates shared state corruption
* **Future-Proof**: New engines require only adapter implementation

**Performance Notes:**

* **Sequential Recommended**: Use `batch_size=0` for optimal performance (sequential processing)
* **Parallel Available**: `batch_size > 1` enables parallel workers but typically slower due to GPU inference characteristics
* **Memory Efficiency**: Improved model sharing prevents memory exhaustion when switching modes

→ **[📖 Read Technical Details](docs/Dev%20reports/POST_V4.2.3_DEVELOPMENT_REVIEW.md)**

</details>

<details>
<summary><h3>🌈 IndexTTS-2 With Emotion Control</h3></summary>

**NEW in v4.9.0**: Revolutionary IndexTTS-2 engine with advanced emotion control and unified emotion architecture!

* **Unified Emotion Control**: Single `emotion_control` input supporting multiple emotion methods with intelligent priority system
* **Dynamic Text Emotion**: AI-powered QwenEmotion analysis with dynamic `{seg}` template processing for contextual per-segment emotions
* **Direct Audio Reference**: Use any audio file as emotion reference for natural emotional expression
* **Character Voices Integration**: Use Character Voices `opt_narrator` output as emotion reference with automatic detection
* **8-Emotion Vector Control**: Manual precision control over Happy, Angry, Sad, Surprised, Afraid, Disgusted, Calm, and Melancholic emotions
* **Character Tag Emotions**: Per-character emotion control using `[Character:emotion_ref]` syntax (highest priority)
* **Emotion Alpha Control**: Fine-tune emotion intensity from 0.0 (neutral) to 2.0 (maximum dramatic expression)

**Key Features:**

- **Emotion Priority System**: Character tags > Global emotion control with intelligent override handling
- **Dynamic Templates**: Use `{seg}` placeholder for contextual emotion analysis (e.g., "Worried parent speaking: {seg}")
- **Universal Compatibility**: Works with existing TTS Text and TTS SRT nodes seamlessly
- **Advanced Caching**: Stable audio content hashing for reliable cache hits across sessions
- **QwenEmotion Integration**: State-of-the-art text emotion analysis with configurable model selection

**Example Usage:**

```text
Welcome to our show! [Alice:happy_sarah] I'm so excited to be here!
[Bob:angry_narrator] That's completely unacceptable behavior.
```

**Perfect for:**

- Multi-character dialogue with individual emotional expressions
- Dynamic storytelling with contextual emotion adaptation
- Professional voice acting with precise emotional control
- Content creation requiring sophisticated emotional nuance

**📖 [Complete IndexTTS-2 Emotion Control Guide](docs/IndexTTS2_Emotion_Control_Guide.md)**

</details>

<details>
<summary><h3>🎨 Step Audio EditX - LLM Audio Editing</h3></summary>

**NEW in v4.15**: Revolutionary LLM-based audio post-processing with emotion, style, and paralinguistic control!

* **🎨 Step Audio EditX - Audio Editor Node**: Post-process ANY TTS audio with advanced editing capabilities
* **🗣️ Paralinguistic Effects**: Insert natural sounds - Laughter, Breathing, Sigh, Uhm, Surprise (oh/ah/wa), Confirmation, Question, Dissatisfaction
* **😊 14 Emotion Controls**: happy, sad, angry, excited, calm, fearful, surprised, disgusted, confusion, empathy, embarrass, depressed, coldness, admiration
* **🎭 32 Speaking Styles**: whisper, serious, child, older, girl, pure, sister, sweet, exaggerated, ethereal, generous, recite, act_coy, warm, shy, comfort, authority, chat, radio, soulful, gentle, story, vivid, program, news, advertising, roar, murmur, shout, deeply, loudly, arrogant, friendly
* **⚡ Speed Control**: faster, slower, more_faster, more_slower with multi-iteration support
* **🔊 Voice Restoration**: ChatterBox VC integration to restore original voice resemblance after editing
* **🏷️ Inline Edit Tags**: Apply effects directly in text using `<Laughter:2>`, `<emotion:happy>`, `<style:whisper>` tags
* **🎛️ Multiline Tag Editor Integration**: Tabbed interface with dropdowns for all edit types, iteration sliders, and pipe-separator support

**Key Features:**

- **LLM-Based Processing**: 3B parameter Step-1 model for high-quality audio editing
- **Multi-Language Support**: Mandarin Chinese (primary), English, Sichuanese, Cantonese, Japanese, Korean
- **Iteration Control**: 1-5 iterations per effect for strength adjustment (WARNING: 3+ iterations risk voice degradation)
- **Batch Processing**: Automatically processes all tagged segments efficiently
- **Position-Aware**: Paralinguistic tags insert sounds at exact cursor positions
- **Pipe-Separator Tags**: Combine multiple effects `<Laughter:2|emotion:happy|style:whisper>`
- **Universal Compatibility**: Works with ALL TTS engines - F5-TTS, ChatterBox, Higgs Audio 2, VibeVoice, IndexTTS-2

**Example Usage:**

```text
[Alice] Hello there <Laughter:2> my friend! <emotion:happy>
[Bob] Listen carefully <style:whisper|speed:slower>, this is important.
[Alice] I'm laughing so hard <Laughter:3> <restore>
```

**Perfect for:**

- Adding natural expressiveness to robotic TTS output
- Creating dynamic conversations with varied emotional tones
- Professional voice-over work requiring subtle emotional nuances
- Storytelling with immersive paralinguistic effects

**⚠️ Important Notes:**

- **Language Limitation**: Only supports Mandarin, English, Sichuanese, Cantonese, Japanese, Korean - using other languages will lose accents and distort audio
- **Voice Quality**: 3+ iterations can degrade voice resemblance - use `<restore>` tag to recover original voice character
- **Duration Limits**: Segments must be 0.5s - 30s (split longer segments)

**📖 [Complete Inline Edit Tags Guide](docs/INLINE_EDIT_TAGS_USER_GUIDE.md)**

</details>

<details>
<summary><h3>🗣️ CosyVoice3 Multilingual Voice Cloning</h3></summary>

**NEW in v4.16**: Alibaba's fast multilingual voice cloning with native paralinguistic tags, instruct mode, and zero-shot voice conversion!

* **⚡ Ultra-Fast Generation**: ~0.05 RTF (20x faster than real-time) - 10s audio in ~0.5s
* **🎵 Native Paralinguistic Tags**: Built-in `<breath>`, `<laughter>`, `<cough>`, `<sigh>`, `<laughing>text</laughing>` - processed during generation → **[📖 Tags Guide](docs/COSYVOICE3_TAGS_GUIDE.md)**
* **🎭 Instruct Mode**: Natural language control - "Speak with a joyful tone" or "Use Cantonese dialect with excitement"
* **🔄 Zero-Shot Voice Conversion**: Built-in VC with iterative refinement (10 passes) and chunking via Voice Changer node
* **🌍 4 Core Languages**: English, Chinese, Japanese, Korean with native language tag support
* **🎤 Zero-Shot Voice Cloning**: Clone any voice from 3-30s reference audio
* **~5.4GB Model**: Efficient 0.5B parameter model with production quality

**Key Features:**

- **Native Tag Support**: 13 paralinguistic effects - `<breath>`, `<laughter>`, `<cough>`, `<sigh>`, `<gasp>`, `<laughing>text</laughing>`, `<strong>text</strong>`
- **Instruct Mode**: Control emotions, dialects, styles via text instructions - no tag syntax needed
- **Fast Voice Conversion**: CosyVoice VC with 10-pass iterative refinement and smart/fixed chunking
- **Character Switching**: Full `[CharacterName]` support with per-character voice references
- **Language Switching**: `[en:]`, `[zh:]`, `[ja:]`, `[ko:]` bracket syntax or native `<|en|>` tags
- **Pause Tags**: Complete `[pause:Ns]` support for natural speech timing
- **SRT Processing**: Advanced subtitle timing with all timing modes
- **Speed Control**: 0.5x to 2.0x speech speed adjustment

**Example Usage:**

```
[Alice] Hello everyone! This is zero-shot voice cloning.
[Bob] 你好！我说普通话。[pause:1s] 还可以说方言。
```

**Instruct Mode Examples:**

```
# Speak with Cantonese dialect
Instruct: 请用广东话表达。

# Speak with excitement
Instruct: 用兴奋的语气说话。
```

**Perfect for:**

- Multilingual content with Chinese dialect support
- Voice cloning with fine emotional control via instructions
- Multi-character conversations across languages
- Professional localization with native accent preservation

</details>

<details>
<summary><h3>🎤 Qwen3-TTS - 4 Model Types with Text-to-Voice Design</h3></summary>

**NEW in v4.19**: Alibaba's Qwen3-TTS with 3 distinct TTS model types - CustomVoice presets, unique text-to-voice design, and zero-shot voice cloning! A **single engine** automatically selects and downloads the correct model based on your settings — no manual model management needed.
**NEW**: ✏️ Qwen3-ASR transcription via the Unified ASR Transcribe node (Qwen3 engine)

**Model Types:**

* **🎭 CustomVoice Model** (0.6B / 1.7B): 9 preset multilingual speakers (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee)
  - ✅ Supports style instructions ("Speak cheerfully", "Sound professional")
  - Character switching auto-maps to different preset speakers

* **✍️ VoiceDesign Model** (1.7B only): **UNIQUE** - Create voices from text descriptions
  - Input: "A cheerful young woman with a bright, energetic tone"
  - Output: Instant voice generation matching the description
  - ✅ Supports style instructions alongside the voice description
  - Smart disk caching for reuse across sessions

* **🎤 Base Model** (0.6B / 1.7B): Zero-shot voice cloning from 3-30s reference audio
  - **ICL mode** (default, best quality): requires reference audio **+ reference transcript** — use the 🎭 Character Voices node which always includes both
  - **X-Vector mode**: uses only the audio to extract a speaker embedding, no transcript needed — faster but lower quality
  - ⚠️ **Does NOT support style instructions** — instruction field is ignored in this mode

* **🔤 ASR**: Transcription via the ✏️ Unified ASR Transcribe node

> **⚠️ Style instructions only work with CustomVoice and VoiceDesign.** Voice cloning (Base) ignores the instruction field entirely.

**Technical Specs:**

* **🌍 10 Languages**: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
* **📏 2 Model Sizes**: 0.6B and 1.7B parameter variants (VoiceDesign is 1.7B only)
* **⚡ Attention Options**: eager, flash_attention_2, sdpa, sage_attn for performance tuning
* **🔧 Generation Control**: Temperature, top_k, top_p, repetition_penalty parameters
* **⚡ torch.compile Optimizations**: Optional 1.7x speedup (PyTorch 2.10+ required) → [📖 Setup Guide](docs/qwen3_tts_optimizations.md)

**Unified Features Support:**
- Works with all project features: character switching, language switching, pause tags, SRT timing, Step Audio EditX post-processing
- **ASR Transcription**: Qwen3 engine now supports ASR via the ✏️ ASR Transcribe node

**Voice Designer Node:**

Unique text-to-voice generation node that creates voices from descriptions and outputs unified NARRATOR_VOICE format for use with any TTS node.

```
Description: "A deep, authoritative male voice with clear articulation"
→ Voice generated and cached → Use in TTS Text/SRT nodes
```

**Perfect for:**

- Quick multilingual content with preset speakers (CustomVoice)
- **Creative voice design from text descriptions** (VoiceDesign) - **unique to Qwen3-TTS**
- High-quality voice cloning with reference audio (Base)
- Content requiring specific vocal characteristics defined by text

</details>

<details>
<summary><h3>🎧 Echo-TTS Voice Cloning</h3></summary>

**NEW in v4.22**: Echo-TTS DiT-based voice cloning with reference audio support.

* **⏱️ Best at ≤30s per generation** — Echo-TTS performs best at ~30 seconds or less per chunk
* **🧩 Long-form (best-effort)** — Longer text is handled via the unified chunking system
* **⚡ CUDA recommended** — CPU works but is very slow for real workloads
* **🔑 Force Speaker KV** — Controls speaker identity drift across chunks
* **🪪 License** — Echo-TTS weights are non-commercial (CC-BY-NC-SA)

**Usage:**
1. Add `⚙️ Echo-TTS Engine` node
2. Connect to `🎤 TTS Text` or `📺 TTS SRT`
3. First run auto-downloads the model (~7.1GB total) into `ComfyUI/models/TTS/echo-tts-base/`

📖 **See [Model folder layouts](docs/MODEL_LAYOUTS.md#echo-tts) for detailed setup paths**

</details>

<details>
<summary><h3>📝 Phoneme Text Normalizer</h3></summary>

**NEW in v4.10.0**: Universal multilingual text preprocessing node for improved TTS pronunciation quality across languages!

* **📝 Phoneme Text Normalizer Node**: Standalone text processing node with multiple normalization methods
* **🌍 Universal Language Support**: Handles special characters for Polish, German, French, Spanish, Czech, Nordic languages, and more
* **🔄 Multiple Processing Methods**:
  - **Pass-through**: No processing (original text)
  - **Unicode Decomposition**: Converts special characters to base + diacritical marks
  - **IPA Phonemization**: Full International Phonetic Alphabet conversion using espeak
  - **Character Mapping**: ASCII fallback for maximum compatibility
* **🧠 Auto-Language Detection**: Automatically detects input language based on character patterns
* **🖥️ Cross-Platform Support**: Works on Windows (espeak-phonemizer-windows), Linux/Mac (phonemizer + system espeak)
* **🔗 Engine Integration**: Compatible with all TTS engines - F5-TTS, ChatterBox, Higgs Audio 2, VibeVoice, IndexTTS-2

**Perfect for:**

- Improving Polish, German, French pronunciation in F5-TTS
- Processing multilingual content with special characters
- Standardizing text input across different TTS engines
- Converting text to phonemic representation for better synthesis

> **⚠️ Experimental Feature**: This is a new experimental feature - user feedback needed to validate pronunciation improvements across different languages. Please test with your target languages and report results!

**📖 Try the workflow**: [F5 TTS integration + 📝 Phoneme Text Normalizer](example_workflows/F5%20TTS%20integration%20+%20📝%20Phoneme%20Text%20Normalizer.json)

</details>

<details>
<summary><h3>🏷️ Multiline TTS Tag Editor & Per-Segment Parameter Switching</h3></summary>

**NEW in v4.12.0**: Fine-grained per-segment control over TTS generation parameters across all engines with an interactive tag editor!

Beyond character switching and language control, you can now override generation parameters (seed, temperature, CFG, speed, etc.) on a per-segment basis using inline tags. The new **🏷️ Multiline TTS Tag Editor** node makes building complex tags easier and more visual with:
- **Rich Text Editor**: Multiline editor with resizable font sizes (2-120px), multiple font families, and customizable UI scaling
- **Visual Tag Management**: Character/language/parameter dropdowns for quick selection, inline tag validation with syntax checking
- **Preset System**: Save and load up to 3 preset configurations for rapid tag reuse
- **Keyboard Shortcuts**: Alt+L/C/P for tag insertion, Alt+1/2/3 for preset loading
- **History & Undo/Redo**: Full edit history with Alt+Z for undo (Alt+Shift+Z for redo)
- **Tag Formatting**: Auto-format button for proper spacing and structure

This enables dynamic control over individual audio segments without modifying node defaults.

**🎯 Key Features:**

* **⚙️ Per-Segment Control** - Override parameters for individual text segments or SRT subtitle entries
* **🔀 Order-Independent Syntax** - Specify parameters in any order: `[Alice|seed:42|temp:0.5]` or `[seed:42|Alice|temp:0.5]`
* **📝 Alias Support** - Use shortcuts like `temp` (temperature), `cfg_weight` (cfg), `exag` (exaggeration)
* **🔤 Case-Insensitive** - `[SEED:42]`, `[Seed:42]`, `[seed:42]` all work identically
* **📺 SRT Support** - Apply parameters to individual SRT subtitle lines
* **🔙 Backward Compatible** - Works alongside character and language switching: `[de:Alice|seed:42|temp:0.7]`
* **🚨 Smart Filtering** - Unsupported parameters are silently ignored; engine-specific parameters work automatically

**💡 Real-World Examples:**

```
[Alice|seed:42] This is the first segment of Alice.
[Alice|seed:42] This is the second segment, sounding identical.

[Bob|temperature:0.3] Bob speaks carefully and precisely.
[Bob|temperature:0.8] Bob speaks more creatively and varied.

[Narrator|temp:0.4] Important introduction - precise delivery.
[Narrator|temp:0.7] Creative narrative section - more varied!
```

**🔧 Supported Parameters by Engine:**

- **ChatterBox & Official 23-Lang**: seed, temperature, cfg, exaggeration
- **F5-TTS**: seed, temperature, cfg, speed
- **Higgs Audio 2**: seed, temperature, cfg, top_p, top_k
- **VibeVoice**: seed, temperature, cfg, top_p, top_k, inference_steps
- **IndexTTS-2**: seed, temperature, cfg, top_p, top_k, emotion_alpha

**📖 [Complete Per-Segment Parameter Switching Guide](docs/PARAMETER_SWITCHING_GUIDE.md)**

Perfect for:

- Creating consistent character voices with fixed seeds across segments
- Varying emotional intensity within a single narration
- Fine-tuning generation quality per-segment without regenerating everything
- Dynamic control in SRT workflows with per-line customization

</details>

## 🚀 Quick Start

> [!TIP]
> You might want to create a clean portable installation just for TTS-Audio-Suite, instead of adding it to your regular ComfyUI setup. This can make the installation process smoother and may reduce initialization time.

### Option 1: ComfyUI Manager (Recommended) ✨

**One-click installation with intelligent dependency management:**

1. Use ComfyUI Manager to install **"TTS Audio Suite"**
2. **That's it!** ComfyUI Manager automatically runs our install.py script which handles:
   - ✅ **Python 3.13 compatibility** (MediaPipe → OpenSeeFace fallback)
   - ✅ **Dependency conflicts** (NumPy, librosa, etc.)
   - ✅ **All bundled engines** (ChatterBox, F5-TTS, Higgs Audio)
   - ✅ **RVC voice conversion** dependencies
   - ✅ **Intelligent conflict resolution** with --no-deps handling

**Python 3.13 Support:**

- 🟢 **All TTS engines**: ChatterBox, F5-TTS, Higgs Audio ✅ Working
- 🟢 **RVC voice conversion**: ✅ Working  
- 🟢 **OpenSeeFace mouth movement**: ✅ Working (experimental)
- 🔴 **MediaPipe mouth movement**: ❌ Incompatible (use OpenSeeFace)

### Option 2: Manual Installation

**Same intelligent installer, manual setup:**

1. **Clone the repository**
   
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/diodiogod/TTS-Audio-Suite.git
   cd TTS-Audio-Suite
   ```

2. **Run the intelligent installer:**
   
   **ComfyUI Portable:**
   
   ```bash
   # Windows:
   ..\..\..\python_embeded\python.exe install.py
   
   # Linux/Mac:
   ../../../python_embeded/python.exe install.py
   ```
   
   **ComfyUI with venv/conda:**
   
   ```bash
   # First activate your ComfyUI environment, then:
   python install.py
   ```
   
   The installer automatically handles all dependency conflicts and Python version compatibility.

3. **Run your first workflow**
   
   - Models auto-download on first use for all engines.
   - If you need offline/manual setup, use:
     - [Model Download Sources](docs/MODEL_DOWNLOAD_SOURCES.md)
     - [Model Folder Layouts](docs/MODEL_LAYOUTS.md)

4. **Try a Workflow**
   
   - Download: [ChatterBox Integration Workflow](example_workflows/Chatterbox%20integration.json)
   - Drag into ComfyUI and start generating!

5. **Restart ComfyUI** and look for 🎤 TTS Audio Suite nodes

> **🧪 Python 3.13 Users**: Installation is fully supported! The system automatically uses OpenSeeFace for mouth movement analysis when MediaPipe is unavailable.

> **Need offline/manual setup?** Use [docs/MODEL_DOWNLOAD_SOURCES.md](docs/MODEL_DOWNLOAD_SOURCES.md) and [docs/MODEL_LAYOUTS.md](docs/MODEL_LAYOUTS.md)

<div align="right"><a href="#-table-of-contents">Back to top</a></div>

## Installation

<details>
<summary>📋 Detailed Installation Guide (Click to expand if you're having dependency issues)</summary>

This section provides a detailed guide for installing TTS Audio Suite, covering different ComfyUI installation methods.

### Prerequisites

* ComfyUI installation (Portable, Direct with venv, or through Manager)

* Python 3.12 or higher

* **System libraries** (Linux only):
  
  ```bash
  # Ubuntu/Debian - Required for audio processing
  sudo apt-get install portaudio19-dev libsamplerate0-dev
  
  # Fedora/RHEL
  sudo dnf install portaudio-devel libsamplerate-devel
  ```
  
  > **📋 Why needed?** `libsamplerate0-dev` provides audio resampling libraries for packages like `resampy` and `soxr`. `portaudio19-dev` enables voice recording features.

* **macOS dependencies**:
  
  ```bash
  brew install portaudio
  ```

* **Windows**: No additional system dependencies needed (libraries come pre-compiled)

### Installation Methods

#### 1. Portable Installation

For portable installations, follow these steps:

1. Clone the repository into the `ComfyUI/custom_nodes` folder:
   
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/diodiogod/TTS-Audio-Suite.git
   ```

2. Navigate to the cloned directory:
   
   ```bash
   cd TTS-Audio-Suite
   ```

3. Run the install script to automatically handle all dependencies. **Important:** Use the `python.exe` executable located in your ComfyUI portable installation.
   
   ```bash
   ../../../python_embeded/python.exe install.py
   ```
   
   The script will automatically install all required Python packages and detect any missing system dependencies.

#### 2. Direct Installation with venv

If you have a direct installation with a virtual environment (venv), follow these steps:

1. Clone the repository into the `ComfyUI/custom_nodes` folder:
   
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/diodiogod/TTS-Audio-Suite.git
   ```

2. Activate your ComfyUI virtual environment.  This is crucial to ensure dependencies are installed in the correct environment. The method to activate the venv may vary depending on your setup.  Here's a common example:
   
   ```bash
   cd ComfyUI
   . ./venv/bin/activate
   ```
   
   or on Windows:
   
   ```bash
   ComfyUI\venv\Scripts\activate
   ```

3. Navigate to the cloned directory:
   
   ```bash
   cd custom_nodes/TTS-Audio-Suite
   ```

4. Run the install script to automatically handle all dependencies:
   
   ```bash
   python install.py
   ```
   
   The script will automatically install all required Python packages and detect any missing system dependencies.

#### 3. Installation through the ComfyUI Manager

1. Install the ComfyUI Manager if you haven't already.

2. Use the Manager to install the "TTS Audio Suite" node.

3. The manager might handle dependencies automatically, but it's still recommended to verify the installation.  Navigate to the node's directory:
   
   ```bash
   cd ComfyUI/custom_nodes/TTS-Audio-Suite
   ```

4. Activate your ComfyUI virtual environment (see instructions in "Direct Installation with venv").

5. If you encounter issues, run the install script to manually install dependencies:
   
   ```bash
   python install.py
   ```

### Troubleshooting Dependency Issues

#### System Dependencies (Linux)

**Our install script automatically detects missing system libraries** and will display helpful error messages like:

```
[!] Missing system dependencies detected!
============================================================
SYSTEM DEPENDENCIES REQUIRED
============================================================
• libsamplerate0-dev (for audio resampling)  
• portaudio19-dev (for voice recording)

Please install with:
# Ubuntu/Debian:
sudo apt-get install libsamplerate0-dev portaudio19-dev

# Fedora/RHEL:
sudo dnf install libsamplerate-devel portaudio-devel
============================================================
Then run this install script again.
```

#### Python Environment Issues

A common problem is installing dependencies in the wrong Python environment. Always ensure you are installing dependencies within your ComfyUI's Python environment.

* **Verify your Python environment:** After activating your venv or navigating to your portable ComfyUI installation, check the Python executable being used:
  
  ```bash
  which python
  ```
  
  This should point to the Python executable within your ComfyUI installation (e.g., `ComfyUI/python_embeded/python.exe` or `ComfyUI/venv/bin/python`).

* **If `s3tokenizer` fails to install:** This dependency can be problematic. Try upgrading your pip and setuptools:
  
  ```bash
  python -m pip install --upgrade pip setuptools wheel
  ```
  
  Then, try installing the requirements again.

* **If you cloned the node manually (without the Manager):** Make sure you install the requirements.txt file.

### Updating the Node

To update the node to the latest version:

1. Navigate to the node's directory:
   
   ```bash
   cd ComfyUI/custom_nodes/TTS-Audio-Suite
   ```

2. Pull the latest changes from the repository:
   
   ```bash
   git pull
   ```

3. Reinstall the dependencies (in case they have been updated):
   
   ```bash
   python install.py
   ```

</details>

### 1. Clone Repository

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/diodiogod/TTS-Audio-Suite.git
```

### 2. Install Dependencies

Some dependencies, particularly `s3tokenizer`, can occasionally cause installation issues on certain Python setups (e.g., Python 3.10, sometimes used by tools like Stability Matrix).

To minimize potential problems, it's highly recommended to first ensure your core packaging tools are up-to-date in your ComfyUI's virtual environment:

```bash
python -m pip install --upgrade pip setuptools wheel
```

After running the command above, install the node's specific requirements:

```bash
python install.py
```

### 3. Optional: Install FFmpeg for Enhanced Audio Processing

ChatterBox Voice now supports FFmpeg for high-quality audio stretching. While not required, it's recommended for the best audio quality:

**Windows:**

```bash
winget install FFmpeg
# or with Chocolatey
choco install ffmpeg
```

**macOS:**

```bash
brew install ffmpeg
```

**Linux:**

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

If FFmpeg is not available, ChatterBox will automatically fall back to using the built-in phase vocoder method for audio stretching - your workflows will continue to work without interruption.

### 4. Models — All Auto-Download

**No manual downloads needed.** All engines download required models automatically on first use.

For offline/manual setup:

- **All model source links (single source of truth):** [docs/MODEL_DOWNLOAD_SOURCES.md](docs/MODEL_DOWNLOAD_SOURCES.md)
- **All folder trees and placement notes:** [docs/MODEL_LAYOUTS.md](docs/MODEL_LAYOUTS.md)

| Engine | Primary model path | Auto-download | Notes |
|---|---|---|---|
| ChatterBox | `ComfyUI/models/TTS/chatterbox/` | ✅ | Legacy `ComfyUI/models/chatterbox/` still works |
| ChatterBox 23-Lang | `ComfyUI/models/TTS/chatterbox_official_23lang/` | ✅ | v1/v2 coexist in same folder |
| F5-TTS | `ComfyUI/models/TTS/F5-TTS/` | ✅ | Optional Vocos and voice refs |
| Higgs Audio 2 | `ComfyUI/models/TTS/HiggsAudio/` | ✅ | Generation + tokenizer |
| VibeVoice | `ComfyUI/models/TTS/VibeVoice/` | ✅ | 1.5B and 7B variants |
| RVC | `ComfyUI/models/TTS/RVC/` | ✅* | Base models auto; character `.pth` can be user-provided |
| IndexTTS-2 | `ComfyUI/models/TTS/IndexTTS/` | ✅ | Emotion components included |
| Step Audio EditX | `ComfyUI/models/TTS/step_audio_editx/` | ✅ | Main model + tokenizer stack |
| CosyVoice3 | `ComfyUI/models/TTS/CosyVoice/` | ✅ | Variant-specific lazy downloads |
| Qwen3-TTS / ASR | `ComfyUI/models/TTS/qwen3_tts/` | ✅ | Per-variant download + shared tokenizer |
| Echo-TTS | `ComfyUI/models/TTS/echo-tts-base/` | ✅ | ~7.1GB total (base + dac); CC-BY-NC-SA |

### 5. Restart ComfyUI

<div align="right"><a href="#-table-of-contents">Back to top</a></div>

## 💖 Support This Project

If TTS Audio Suite has been helpful for your projects, consider supporting its development:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/diogogo)

Your support helps maintain and improve this project for the entire community!

<div align="right"><a href="#-table-of-contents">Back to top</a></div>

## Enhanced Features

### 📝 Intelligent Text Chunking (NEW!)

**Long text support with smart processing:**

- **Character-based limits** (100-1000 chars per chunk)
- **Sentence boundary preservation** - won't cut mid-sentence
- **Multiple combination methods**:
  - `auto` - Smart selection based on text length
  - `concatenate` - Simple joining
  - `silence_padding` - Add configurable silence between chunks
  - `crossfade` - Smooth audio blending
- **Comma-based splitting** for very long sentences
- **Backward compatible** - works with existing workflows

**Chunking Controls (all optional):**

- `enable_chunking` - Enable/disable smart chunking (default: True)
- `max_chars_per_chunk` - Chunk size limit (default: 400)
- `chunk_combination_method` - How to join audio (default: auto)
- `silence_between_chunks_ms` - Silence duration (default: 100ms)

**Auto-selection logic:**

- **Text > 1000 chars** → silence_padding (natural pauses)
- **Text > 500 chars** → crossfade (smooth blending)
- **Text < 500 chars** → concatenate (simple joining)

### 📦 Smart Model Loading

**Priority-based model detection:**

1. **Bundled models** in node folder (self-contained)
2. **ComfyUI models** in standard location
3. **HuggingFace download** with authentication

**Console output shows source:**

```
📦 Using BUNDLED ChatterBox (self-contained)
📦 Loading from bundled models: ./models/chatterbox
✅ ChatterboxTTS model loaded from bundled!
```

<div align="right"><a href="#-table-of-contents">Back to top</a></div>

## Usage

### Voice Recording

1. Add **"🎤 ChatterBox Voice Capture"** node
2. Select your microphone from the dropdown
3. Adjust recording settings:
   - **Silence Threshold**: How quiet to consider "silence" (0.001-0.1)
   - **Silence Duration**: How long to wait before stopping (0.5-5.0 seconds)
   - **Sample Rate**: Audio quality (8000-96000 Hz, default 44100)
4. Change the **Trigger** value to start a new recording
5. Connect output to TTS (for voice cloning) or VC nodes

### Enhanced Text-to-Speech

1. Add **"🎤 ChatterBox Voice TTS"** node
2. Enter your text (any length - automatic chunking)
3. Optionally connect reference audio for voice cloning
4. Adjust TTS settings:
   - **Exaggeration**: Emotion intensity (0.25-2.0)
   - **Temperature**: Randomness (0.05-5.0)
   - **CFG Weight**: Guidance strength (0.0-1.0)

### F5-TTS Voice Synthesis

1. Add **"🎤 F5-TTS Voice Generation"** node
2. Enter your target text (any length - automatic chunking)
3. **Required**: Connect reference audio for voice cloning
4. **Required**: Enter reference text that matches the reference audio exactly

<details>
<summary>📖 Voice Reference Setup Options</summary>

**Two ways to provide voice references:**

1. **Easy Method**: Select voice from `reference_audio_file` dropdown → text auto-detected from companion `.txt` file
2. **Manual Method**: Set `reference_audio_file` to "none" → connect `opt_reference_audio` + `opt_reference_text` inputs

</details>

5. Select F5-TTS model:
   - **F5TTS_Base**: English base model (recommended)
   - **F5TTS_v1_Base**: English v1 model
   - **E2TTS_Base**: E2-TTS model
   - **F5-DE**: German model
   - **F5-ES**: Spanish model
   - **F5-FR**: French model
   - **F5-JP**: Japanese model
6. Adjust F5-TTS settings:
   - **Temperature**: Voice variation (0.1-2.0, default: 0.8)
   - **Speed**: Speech speed (0.5-2.0, default: 1.0)
   - **CFG Strength**: Guidance strength (0.0-10.0, default: 2.0)
   - **NFE Step**: Quality vs speed (1-100, default: 32)

### Voice Conversion with Iterative Refinement

1. Add **"🔄 ChatterBox Voice Conversion"** node
2. Connect source audio (voice to convert)
3. Connect target audio (voice style to copy)
4. Configure refinement settings:
   - **Refinement Passes**: Number of conversion iterations (1-30, recommended 1-5)
   - Each pass refines the output to sound more like the target
   - **Smart Caching**: Results cached up to 5 iterations for instant experimentation

**🧠 Intelligent Caching Examples:**

- Run **3 passes** → caches iterations 1, 2, 3
- Change to **5 passes** → resumes from cached 3, runs 4, 5  
- Change to **2 passes** → returns cached iteration 2 instantly
- Change to **4 passes** → resumes from cached 3, runs 4

**💡 Pro Tip**: Start with 1 pass, then experiment with 2-5 passes to find the sweet spot for your audio. Each iteration can improves voice similarity!

<div align="right"><a href="#-table-of-contents">Back to top</a></div>

## 📁 Example Workflows

**Ready-to-use ComfyUI workflows** - Download and drag into ComfyUI:

### 🆕 Unified Workflows (v4.5+)

| Workflow                     | Description                                   | Features                                                                                                             | Status                 | Files                                                                                       |
| ---------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------- |
| **Unified 📺 TTS SRT**       | Universal SRT processing with all TTS engines | • ChatterBox/F5-TTS/Higgs Audio 2<br>• Multiple timing modes<br>• Multi-character switching<br>• Overlap SRT support | ✅ **New in v4.5**      | [📁 JSON](example_workflows/Unified%20📺%20TTS%20SRT.json)                                  |
| **Unified 🔄 Voice Changer** | Modern voice conversion with multiple engines | • RVC + ChatterBox VC<br>• Iterative refinement<br>• Real-time conversion                                            | ✅ **Updated for v4.3** | [📁 JSON](example_workflows/Unified%20🔄%20Voice%20Changer%20-%20RVC%20X%20ChatterBox.json) |

### Specific Workflows

| Workflow                                       | Description                                                | Status               | Files                                                                                                               |
| ---------------------------------------------- | ---------------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **🤐 Voice Cleaning**                          | Audio restoration & cleanup with dual tool pipeline        | ✅ **New in v4.13**   | [📁 JSON](example_workflows/Voice%20Cleaning%20-%20🤐%20Noise%20or%20Vocal%20Removal%20+%20🤐%20Voice%20Fixer.json) |
| **🎨 Step Audio EditX - Audio Editor**         | Step Audio EditX audio editing with inline edit tags       | ✅ **New in v4.14**   | [📁 JSON](example_workflows/🎨%20Step%20Audio%20EditX%20-%20Audio%20Editor%20+%20Inline%20Edit%20Tags.json)        |
| **⚙️ Step Audio EditX Integration**            | Step Audio EditX TTS engine with zero-shot voice cloning   | ✅ **New in v4.14**   | [📁 JSON](example_workflows/Step%20Audio%20EditX%20Integration.json)                                                |
| **🌈 IndexTTS-2 Integration**                  | IndexTTS-2 engine with advanced emotion control            | ✅ **New in v4.9**    | [📁 JSON](example_workflows/🌈%20IndexTTS-2%20integration.json)                                                     |
| **📝 F5 TTS + Text Normalizer**                | F5-TTS with multilingual text processing and phonemization | ✅ **New in v4.10.0** | [📁 JSON](example_workflows/F5%20TTS%20integration%20+%20📝%20Phoneme%20Text%20Normalizer.json)                     |
| **Qwen3 integration + ASR**                    | Qwen3-TTS voice generation with ASR transcription          | ✅ **New in v4.21**   | [📁 JSON](example_workflows/Qwen3%20integration%20+%20ASR.json)                                                     |
| **VibeVoice Integration**                      | VibeVoice long-form TTS with multi-speaker support         | ✅ **Compatible**     | [📁 JSON](example_workflows/VibeVoice%20integration.json)                                                           |
| **ChatterBox Integration**                     | General ChatterBox TTS and Voice Conversion                | ✅ **Compatible**     | [📁 JSON](example_workflows/Chatterbox%20integration.json)                                                          |
| **F5-TTS Speech Editor**                       | Interactive waveform analysis for F5-TTS editing           | ✅ **Updated for v4** | [📁 JSON](example_workflows/👄%20F5-TTS%20Speech%20Editor%20Workflow.json)                                          |

> **💡 Recommended:** Use the new **Unified 📺 TTS SRT** workflow which showcases all engines and features in one comprehensive workflow. It demonstrates SRT processing, timing modes, multi-character switching, and supports ChatterBox, F5-TTS, and Higgs Audio 2 engines.
> 
> **📥 Usage:** Download the `.json` files and drag them directly into your ComfyUI interface. The workflows will automatically load with proper node connections.

<div align="right"><a href="#-table-of-contents">Back to top</a></div>

## Settings Guide

### Enhanced Chunking Settings

**For Long Articles/Books:**

- `max_chars_per_chunk=600`, `combination_method=silence_padding`, `silence_between_chunks_ms=200`

**For Natural Speech:**

- `max_chars_per_chunk=400`, `combination_method=auto` (default - works well)

**For Fast Processing:**

- `max_chars_per_chunk=800`, `combination_method=concatenate`

**For Smooth Audio:**

- `max_chars_per_chunk=300`, `combination_method=crossfade`

### Voice Recording Settings

**General Recording:**

- `silence_threshold=0.01`, `silence_duration=2.0` (default settings)

**Noisy Environment:**

- Higher `silence_threshold` (~0.05) to ignore background noise
- Longer `silence_duration` (~3.0) to avoid cutting off speech

**Quiet Environment:**

- Lower `silence_threshold` (~0.005) for sensitive detection
- Shorter `silence_duration` (~1.0) for quick stopping

### TTS Settings

**General Use:**

- `exaggeration=0.5`, `cfg_weight=0.5` (default settings work well)

**Expressive Speech:**

- Lower `cfg_weight` (~0.3) + higher `exaggeration` (~0.7)
- Higher exaggeration speeds up speech; lower CFG slows it down

<div align="right"><a href="#-table-of-contents">Back to top</a></div>

## Text Processing Capabilities

### 📚 No Hard Text Limits!

Unlike many TTS systems:

- **OpenAI TTS**: 4096 character limit
- **ElevenLabs**: 2500 character limit
- **ChatterBox**: No documented limits + intelligent chunking

### 🧠 Smart Text Splitting

**Sentence Boundary Detection:**

- Splits on `.!?` with proper spacing
- Preserves sentence integrity
- Handles abbreviations and edge cases

**Long Sentence Handling:**

- Splits on commas when sentences are too long
- Maintains natural speech patterns
- Falls back to character limits only when necessary

<div align="right"><a href="#-table-of-contents">Back to top</a></div>

## License

MIT License - Same as ChatterboxTTS

<div align="right"><a href="#-table-of-contents">Back to top</a></div>

## Credits

- **ResembleAI** for ChatterboxTTS
- **ComfyUI** team for the amazing framework
- **sounddevice** library for audio recording functionality
- **[ShmuelRonen](https://github.com/ShmuelRonen/ComfyUI_ChatterBox_Voice)** for the Original ChatteBox Voice TTS node
- **[Diogod](https://github.com/diodiogod/TTS-Audio-Suite)** for the TTS Audio Suite universal multi-engine implementation

<div align="right"><a href="#-table-of-contents">Back to top</a></div>

## 🔗 Links

- [Resemble AI ChatterBox](https://github.com/resemble-ai/chatterbox)
- [Model Download Sources](docs/MODEL_DOWNLOAD_SOURCES.md)
- [Model Folder Layouts](docs/MODEL_LAYOUTS.md)
- [ChatterBox Demo](https://resemble-ai.github.io/chatterbox_demopage/)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Resemble AI Official Site](https://www.resemble.ai/chatterbox/)

---

**Note**: The original ChatterBox model includes Resemble AI's Perth watermarking system for responsible AI usage. This ComfyUI integration includes the Perth dependency but has watermarking disabled by default to ensure maximum compatibility. Users can re-enable watermarking by modifying the code if needed, while maintaining the full quality and capabilities of the underlying TTS model.

<!-- MARKDOWN LINKS & IMAGES -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/diodiogod/TTS-Audio-Suite.svg?style=for-the-badge
[contributors-url]: https://github.com/diodiogod/TTS-Audio-Suite/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/diodiogod/TTS-Audio-Suite.svg?style=for-the-badge
[forks-url]: https://github.com/diodiogod/TTS-Audio-Suite/network/members
[stars-shield]: https://img.shields.io/github/stars/diodiogod/TTS-Audio-Suite.svg?style=for-the-badge
[stars-url]: https://github.com/diodiogod/TTS-Audio-Suite/stargazers
[issues-shield]: https://img.shields.io/github/issues/diodiogod/TTS-Audio-Suite.svg?style=for-the-badge
[issues-url]: https://github.com/diodiogod/TTS-Audio-Suite/issues
[license-shield]: https://img.shields.io/github/license/diodiogod/TTS-Audio-Suite.svg?style=for-the-badge
[license-url]: https://github.com/diodiogod/TTS-Audio-Suite/blob/master/LICENSE.txt

[version-shield]: https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fdiodiogod%2FTTS-Audio-Suite%2Fmain%2Fpyproject.toml&query=%24.project.version&label=Version&color=red&style=for-the-badge
[version-url]: pyproject.toml
