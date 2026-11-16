"""Torchaudio sinc kernels with integer-exact output length for long files."""

from torch.nn import functional as F


def output_length(samples, source_rate, target_rate):
    return (int(samples) * int(target_rate) + int(source_rate) - 1) // int(source_rate)


def resample_exact(waveform, transform):
    if transform.orig_freq == transform.new_freq:
        return waveform
    original = transform.orig_freq // transform.gcd
    new = transform.new_freq // transform.gcd
    shape = waveform.shape
    packed = waveform.reshape(-1, shape[-1])
    padded = F.pad(packed, (transform.width, transform.width + original))
    result = F.conv1d(padded[:, None], transform.kernel, stride=original)
    result = result.transpose(1, 2).reshape(packed.shape[0], -1)
    target = output_length(shape[-1], original, new)
    assert result.shape[-1] >= target
    return result[..., :target].reshape(shape[:-1] + (target,))
