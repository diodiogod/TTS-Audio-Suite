"""Streaming iSTFT head + resampler on the codec's device: the torch twins of `zonos3.codec.stream_host`.

Same algorithm and float64 arithmetic as the NumPy ports (overlap-add iSTFT of the two head branches, the FIR mixer,
the 44160 -> 44100 Hz sinc resampler), on the tensors the trunk leaves on its device, so a chunk needs no host round
trip before its final samples. Positions and lengths are host integers (they follow from the frame counts), so the
only device sync is the caller's copy of the output. Outputs are float32 like the NumPy ports' (the head's output is
rounded to float32 before the resampler, as there); the overlap-add sums in another order, so the samples match the
NumPy path to float32 rounding, not bit for bit.
"""

import torch
from torch.nn import functional as F


class IstftHeadStream:
    """Two-branch centered iSTFT (shared hop) followed by the linear FIR mixer, fed head frames incrementally."""

    def __init__(self, n_ffts, hop, windows, mixer, max_log_mag, device):
        self.n_ffts = [int(n) for n in n_ffts]
        self.hop = int(hop)
        self.device = torch.device(device)
        f64 = dict(dtype=torch.float64, device=self.device)
        self.windows = [torch.as_tensor(w).to(**f64) for w in windows]
        self.mixer = torch.as_tensor(mixer).to(**f64)
        if self.mixer.ndim != 2 or self.mixer.shape[0] != 2 or self.mixer.shape[1] % 2 != 1:
            raise ValueError("istft: mixer kernel must be odd")
        for n, w in zip(self.n_ffts, self.windows):
            if len(w) != n or n % self.hop:
                raise ValueError(f"istft: window of {len(w)} taps for n_fft {n}, hop {self.hop}")
        self.max_log_mag = float(max_log_mag)
        self.K = self.mixer.shape[1]
        self.h = (self.K - 1) // 2
        self._f64 = f64
        self.ola = [torch.zeros(0, **f64), torch.zeros(0, **f64)]
        self.env = [torch.zeros(0, **f64), torch.zeros(0, **f64)]
        self.ola_base = [0, 0]
        self.mix_in = [torch.zeros(self.h, **f64), torch.zeros(self.h, **f64)]  # index 0 sits at absolute mix_base
        self.mix_base = self.mix_next = 0
        self.frames = 0
        self.done = False
        self._env: dict[tuple[int, int], torch.Tensor] = {}  # (branch, cols) -> overlap-added squared window

    def copy(self) -> "IstftHeadStream":
        """An independent twin (the overlap-add buffers are updated in place; everything else is replaced)."""
        twin = object.__new__(IstftHeadStream)
        twin.__dict__.update(self.__dict__)
        twin.ola, twin.env = [t.clone() for t in self.ola], [t.clone() for t in self.env]
        twin.ola_base, twin.mix_in = list(self.ola_base), list(self.mix_in)
        return twin

    def _overlap_add(self, frames: torch.Tensor, n: int) -> torch.Tensor:
        """Frames [cols, n] placed every hop samples and summed -> [(cols - 1) * hop + n]."""
        cols = frames.shape[0]
        span = (cols - 1) * self.hop + n
        return F.fold(
            frames.T.unsqueeze(0), output_size=(1, span), kernel_size=(1, n), stride=(1, self.hop)
        ).reshape(span)

    def push(self, y0: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
        """y_i [cols, n_i + 2]: log-magnitude for n_i//2+1 bins, then phase. Returns the 44160 Hz samples now final."""
        if self.done:
            raise RuntimeError("istft: push after flush")
        cols = None
        for i, y in enumerate((y0, y1)):
            n = self.n_ffts[i]
            bins = n // 2 + 1
            y = y.to(**self._f64)
            if y.ndim != 2 or y.shape[1] != n + 2 or (cols is not None and y.shape[0] != cols):
                raise ValueError(f"istft: branch {i} has shape {tuple(y.shape)}, expected [cols, {n + 2}]")
            cols = y.shape[0]
            if cols == 0:
                return torch.zeros(0, dtype=torch.float32, device=self.device)
            mag = torch.exp(torch.clamp(y[:, :bins], max=self.max_log_mag))
            ph = y[:, bins:]
            imag = mag * torch.sin(ph)
            imag[:, 0] = 0.0
            imag[:, -1] = 0.0
            win = self.windows[i]
            frames = torch.fft.irfft(torch.complex(mag * torch.cos(ph), imag), n, dim=-1) * win
            add = self._overlap_add(frames, n)
            env = self._env.get((i, cols))
            if env is None:
                env = self._env[(i, cols)] = self._overlap_add((win * win).expand(cols, n), n)
            span = add.shape[0]
            o = self.frames * self.hop - self.ola_base[i]
            need = o + span
            if self.ola[i].shape[0] < need:
                grow = torch.zeros(need - self.ola[i].shape[0], **self._f64)
                self.ola[i] = torch.cat([self.ola[i], grow])
                self.env[i] = torch.cat([self.env[i], grow])
            self.ola[i][o:need] += add
            self.env[i][o:need] += env
        self.frames += cols
        for i in range(2):
            self._finalize(i, self.frames * self.hop)
        return self._mix_emit()

    def flush(self) -> torch.Tensor:
        """End of stream: every sample m < frames*hop is final; the mixer sees h zeros after the last sample."""
        if self.done or self.frames == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        self.done = True
        L = self.frames * self.hop
        for i in range(2):
            self._finalize(i, self.ola_base[i] + self.ola[i].shape[0])
            keep = self.h + L - self.mix_base
            m = self.mix_in[i]
            m = m[:keep] if m.shape[0] >= keep else torch.cat([m, torch.zeros(keep - m.shape[0], **self._f64)])
            self.mix_in[i] = torch.cat([m, torch.zeros(self.h, **self._f64)])
        return self._mix_emit()

    def _finalize(self, i, pf):
        """Moves overlap-add positions [ola_base, pf) of branch i into the mixer input, dropping the centre padding."""
        n = pf - self.ola_base[i]
        if n <= 0:
            return
        skip = min(n, max(0, self.n_ffts[i] // 2 - self.ola_base[i]))
        self.mix_in[i] = torch.cat([self.mix_in[i], self.ola[i][skip:n] / self.env[i][skip:n]])
        self.ola[i] = self.ola[i][n:]
        self.env[i] = self.env[i][n:]
        self.ola_base[i] = pf

    def _mix_emit(self) -> torch.Tensor:
        """y[m] = sum_c sum_k mixer[c][k] * mix_in_c[m + k] for every m whose taps are available."""
        K = self.K
        m1 = min(m.shape[0] for m in self.mix_in) + self.mix_base - K + 1
        if m1 <= self.mix_next:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        take = m1 - self.mix_base + K - 1
        x = torch.stack([m[:take] for m in self.mix_in]).unsqueeze(0)  # [1, 2, take]
        y = F.conv1d(x, self.mixer.unsqueeze(0)).reshape(-1)  # cross-correlation summed over both channels
        drop = m1 - self.mix_base
        self.mix_in = [m[drop:] for m in self.mix_in]
        self.mix_base = self.mix_next = m1
        return y.to(torch.float32)


class ResamplerStream:
    """Streaming torchaudio sinc resampler: out[n*new + i] = sum_k kernel[i][k] * x[n*orig + k - width]."""

    def __init__(self, kernel, orig, new, width, device):
        self.orig, self.new, self.width = int(orig), int(new), int(width)
        self.kl = 2 * self.width + self.orig
        self.device = torch.device(device)
        self._f64 = dict(dtype=torch.float64, device=self.device)
        self.kernel = torch.as_tensor(kernel).to(**self._f64)
        if tuple(self.kernel.shape) != (self.new, self.kl):
            raise ValueError(f"resampler: kernel shape {tuple(self.kernel.shape)}, expected {(self.new, self.kl)}")
        self._weight = self.kernel.unsqueeze(1)  # [new, 1, kl]
        self.buf = torch.zeros(0, **self._f64)
        self.base = 0  # absolute index of buf[0]
        self.n_in = 0  # samples pushed
        self.next_block = 0

    def copy(self) -> "ResamplerStream":
        """An independent twin (the buffer is only ever replaced, never written in place)."""
        twin = object.__new__(ResamplerStream)
        twin.__dict__.update(self.__dict__)
        return twin

    def push(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(**self._f64).reshape(-1)
        self.buf = torch.cat([self.buf, x])
        self.n_in += x.shape[0]
        return self._run(False)

    def flush(self) -> torch.Tensor:
        return self._run(True)

    def _run(self, flush):
        O, N, W, KL = self.orig, self.new, self.width, self.kl
        if flush:
            total = -(-N * self.n_in // O)
            end = -(-total // N)
        else:
            reach = self.n_in + W - KL
            end = reach // O + 1 if reach >= 0 else 0
        out = torch.zeros(0, dtype=torch.float32, device=self.device)
        if end > self.next_block:
            lo = self.next_block * O - W
            hi = (end - 1) * O - W + KL
            a, b = max(lo, 0), min(hi, self.n_in)
            if b > a:
                seg = F.pad(self.buf[a - self.base : b - self.base], (a - lo, hi - b))
            else:
                seg = torch.zeros(hi - lo, **self._f64)
            # windows of KL samples every O samples, each dotted with the N kernel rows: [N, blocks] -> block-major
            y = F.conv1d(seg.view(1, 1, -1), self._weight, stride=O)[0].T.reshape(-1)
            if flush:
                y = y[: total - self.next_block * N]
            out = y.to(torch.float32)
            self.next_block = end
        keep_from = min(self.n_in, max(0, self.next_block * O - W))
        if keep_from > self.base:
            self.buf = self.buf[keep_from - self.base :]
            self.base = keep_from
        return out
