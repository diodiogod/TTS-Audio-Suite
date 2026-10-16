# F5-TTS Reference and Inference Guidelines

These recommendations apply to F5-TTS generation, whether you use a narrator reference or saved character voices. For switching between voices, see the [Character Switching Guide](CHARACTER_SWITCHING_GUIDE.md).

## Prepare the reference

- Use clean speech from one speaker, with the delivery you want to reproduce.
- Keep the reference **shorter than 12 seconds** and leave a little silence at the end, roughly one second. It does not need to be exactly 10 or 12 seconds, or have one second of silence at both ends.
- Supply an exact transcript of the speech in that clip. When trimming audio, update the transcript too.
- Save library references as `name.wav` with `name.reference.txt`. A matching `name.txt` can serve as the transcript when no reference transcript is found. For connected audio, use **Character Voices** to provide both audio and text through `opt_narrator`.

F5-TTS works at **24,000 Hz** and resamples reference audio automatically. Manual conversion to 22,050 Hz is unnecessary. If converting a file yourself, resample it normally; changing only its declared sample rate changes playback speed and pitch.

## If generated speech is too fast

F5 estimates generation duration from the reference audio's duration and transcript length, adjusted by the engine's `speed` setting. Reference delivery and transcript accuracy both affect pacing.

**Long references can create a hidden mismatch:** F5's preprocessing automatically shortens audio to around 12 seconds, but retains a supplied transcript. Text that matched the original recording may no longer match the processed clip, causing rushed or unstable generation. Prepare a short reference and its matching transcript before passing them to F5.

Start with `speed = 1.0`. After checking the reference, lower it for slower speech or raise it for faster speech. Large changes can affect generation quality. Include the reference audio, transcript, and generated output when reporting a pacing problem.

## Text preparation

- Use punctuation and spaces to suggest pauses. Put a space after English sentence-ending punctuation so F5 can recognize sentence boundaries for chunking.
- Use ordinary lowercase words for normal reading; uppercase forms such as `K.F.C.` are intended for letter-by-letter pronunciation.
- For Chinese speech, write numbers in Chinese words when needed to avoid an unintended English reading.

These points follow the [official F5-TTS inference guidance](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/README.md). Reference clipping and duration estimation are implemented in the [upstream inference code](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/utils_infer.py).
