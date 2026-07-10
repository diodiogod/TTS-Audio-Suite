# OmniVoice Native Tags Guide

OmniVoice has its own official inline square-bracket control tokens upstream, but **this suite does not expose `[]` for OmniVoice tags**.

Inside TTS Audio Suite, you can write the suite-default angle-tag aliases in text:

```text
<laughter>
<sigh>
<question-ei>
```

The OmniVoice processor converts those aliases internally to the official OmniVoice native form before generation:

```text
[laughter]
[sigh]
[question-ei]
```

Do **not** type OmniVoice non-verbal tags in `[]` form in suite text. In this suite, `[]` belongs to character, language, parameter, and pause syntax.

## Supported Non-Verbal Tags

Official OmniVoice non-verbal meanings exposed by this suite through `<>` aliases:

```text
laughter
sigh
confirmation-en
question-en
question-ah
question-oh
question-ei
question-yi
surprise-ah
surprise-oh
surprise-wa
surprise-yo
dissatisfaction-hnn
```

Examples:

```text
[Alice] <laughter> You really got me there.
[Bob] <sigh> Fine, let's try again.
[Narrator] <question-ei> Really?
[Narrator] <surprise-oh> I didn't expect that.
```

## Important Behavior

- OmniVoice uses native generation tags here, not Step Audio EditX inline post-processing.
- User-facing suite syntax stays in `<>` form for OmniVoice non-verbal tags.
- `[]` is reserved for suite structural syntax like `[Alice]`, `[en:Alice]`, `[pause:1s]`, and parameter switching.
- Do not rely on Step-style `<Laughter:2>` or `<emotion:happy>` semantics in the OmniVoice text path.
- If you want Step Audio EditX as a second pass on OmniVoice output, use the separate `🎨 Audio Editor` node manually after generation.

## Multiline Tag Editor

The `🏷️ Multiline TTS Tag Editor` has a dedicated `OmniVoice` mode in the `Inline Tags` panel.

- The editor inserts suite-default angle-tag aliases like `<laughter>`
- The processor converts them internally to official OmniVoice square tags during generation
- The editor does not encourage raw OmniVoice `[]` input because `[]` is suite syntax

## Sources

This behavior follows the official OmniVoice documentation for non-verbal symbols and pronunciation control:

- [OmniVoice GitHub README](https://github.com/k2-fsa/OmniVoice)
- [OmniVoice Hugging Face model card](https://huggingface.co/k2-fsa/OmniVoice)
