"""Shared selection of an audio range without modifying the source clip."""


def trim_audio(audio, trim_start=0.0, trim_end=0.0, *, component="Audio"):
    if not isinstance(audio, dict):
        return audio, False
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0) or 0)
    if waveform is None or sample_rate <= 0:
        return audio, False

    samples = int(waveform.shape[-1])
    duration = samples / sample_rate
    start = max(0.0, min(float(trim_start or 0.0), duration))
    requested_end = float(trim_end or 0.0)
    end = duration if requested_end <= 0.0 else max(0.0, min(requested_end, duration))
    trimmed = start > 1e-6 or end < duration - 1e-6
    if not trimmed:
        return audio, False
    if end <= start:
        raise ValueError(f"Invalid {component} trim range: start {start:.2f}s must be before end {end:.2f}s")

    first = min(samples, max(0, round(start * sample_rate)))
    last = min(samples, max(first + 1, round(end * sample_rate)))
    return {"waveform": waveform[..., first:last].contiguous(), "sample_rate": sample_rate}, True
