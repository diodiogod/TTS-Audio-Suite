# Character and Narrator Switching

Use `[Alice]` and `[Bob]` tags in **TTS Text** or **TTS SRT** to select character voices. This is shared TTS Audio Suite syntax; reference requirements, languages, and native multi-speaker modes depend on the selected engine.

## Add character voices

Save your voices under `ComfyUI/models/voices/`:

```text
ComfyUI/models/voices/
├── alice.wav
├── alice.reference.txt
├── bob.wav
└── bob.reference.txt
```

- The audio filename gives the character its name: `alice.wav` becomes `[Alice]`. Names are case-insensitive. Subfolders are for organization; use unique filenames across your library.
- For engines that use a transcript, put the exact spoken words in `name.reference.txt`. A matching `name.txt` is also accepted when no reference transcript is found. Keep metadata in a separate file from the transcript.
- Transcript requirements vary by engine and mode; see **Reference Transcript** in the [Feature Comparison](FEATURE_COMPARISON.md).

The suite also searches `models/TTS/voices/`, bundled `voices_examples/`, and voice paths configured in `extra_model_paths.yaml`. Supported audio formats include WAV, MP3, FLAC, OGG, M4A, and AAC.

You can create library voices with **Save Character Voice**, using the `opt_narrator` output from **Character Voices** or **Voice Designer**.

## Set the narrator and write your script

Connect an engine to **TTS Text** or **TTS SRT**. For reference-based voice cloning, select the engine's cloning mode when it offers multiple modes. Choose the default voice with `narrator_voice`, or connect **Character Voices** to `opt_narrator`.

```text
Welcome to our story. This line uses the narrator.
[Alice] Hi, Bob!
[Bob] Hello, Alice.
[Alice] Shall we begin? [narrator] And so their journey started.
The next line uses the narrator again.
```

**Each new line starts with the narrator.** A character tag applies until the next tag on that line. Repeat `[Alice]` at the start of each line Alice should speak; use `[narrator]` to switch back within a line.

Use the same tags inside SRT subtitle text. Subtitle timing is controlled separately by the TTS SRT node's timing mode.

## Languages and segment settings

| Tag | Meaning |
| --- | --- |
| `[Alice]` | Select Alice's voice. |
| `[narrator]` | Select the narrator. |
| `[de:Alice]` | Select Alice and request German. |
| `[fr:]` | Request French with the narrator. |
| `[Alice\|seed:42]` | Select Alice and override the segment's seed. |

Language tags do not translate text or add languages to a model. Some engines infer language from the text; others accept an explicit language or load a language-specific model. See [Language Support](LANGUAGE_SUPPORT.md) for engine coverage and the [Parameter Switching Guide](PARAMETER_SWITCHING_GUIDE.md) for supported segment settings.

Language priority is: explicit tag → character's alias default → node/model default.

## Aliases (optional)

**Character Alias Manager demo (v5.5.3):**

https://github.com/user-attachments/assets/960a27d8-c0d8-45e6-b343-7d18bf85d1d6

Open the manager from **Character Voices** or the **Multiline TTS Tag Editor**, or use the **Open Character Alias Manager** command. Assign friendly names, choose voice files, and set optional default languages. Save changes to apply your mappings.

Alternatively, place `#character_alias_map.txt` in a voice-library root. For example, to address `female_01.wav` as Alice:

```text
Alice = female_01, de
Bob = male_01, en
```

`[Alice]` then selects `female_01` with German as its default language; `[en:Alice]` overrides that language. The language field is optional. Alias Manager user overrides take priority over mappings loaded from files.

## If the wrong voice speaks

- Check the audio filename or alias target. Unknown character names fall back to the narrator and produce a console warning.
- Check the transcript requirements for the engine and mode. If you trim a reference, update its transcript to match the retained speech.
- After adding or changing library files, use **Refresh Voice Cache** to rescan them.

For engine-specific audio preparation, see [F5-TTS Reference and Inference Guidelines](F5_TTS_REFERENCE_GUIDE.md). For visual tag insertion, see the [Multiline TTS Tag Editor Guide](MULTILINE_TTS_TAG_EDITOR_GUIDE.md).

