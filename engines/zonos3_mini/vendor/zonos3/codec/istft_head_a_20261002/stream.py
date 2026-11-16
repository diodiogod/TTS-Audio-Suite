"""Stateful incremental decode of an `ISTFTCodec` with the multi-resolution iSTFT head (the torch twin of zac_stream).

`ISTFTStream.decode(codes)` takes the next undelayed code frames and returns the 44100 Hz samples that are final;
`flush()` returns the rest. The causal trunk keeps its state between calls: one depthwise-conv history per ConvNeXt
block (semantic head and iSTFT head) and a sliding-window KV cache per transformer layer. The rest (overlap-add iSTFT,
FIR mixer, resampler) runs on the same device, `zonos3.codec.stream_device` (the torch twins of the NumPy ports in
`zonos3.codec.stream_host`), so a chunk's only host copy is its final samples. Concatenated outputs + flush equal
`model(codes)` over the whole sequence: after F frames exactly 2940*F - 1470 samples are out, flush brings the total
to 2940*F.
"""

import copy
import math

import numpy as np
import torch
from torch.nn import functional as F

from zonos3.codec.hybrid10_dual_encoding.base_model import MossStage, Patch
from zonos3.codec.hybrid10_dual_encoding.model import SemanticConvNeXt
from zonos3.codec.istft_head_a_20261002.model import ISTFTCodec, MultiResISTFTHead
from zonos3.codec.stream_device import IstftHeadStream, ResamplerStream

_CODEBOOK_SIZE = 4096


def _convnext(block, x, hist):
    """SemanticConvNeXt.forward with the left zero padding replaced by the previous call's last K-1 inputs."""
    k = block.dwconv.kernel_size[0]
    if hist is None:
        hist = x.new_zeros(x.shape[0], x.shape[1], k - 1)
    xf = torch.cat((hist.to(x.dtype), x), -1)
    h = block.dwconv(xf).transpose(1, 2)
    h = block.pwconv2(F.gelu(block.pwconv1(block.norm(h))))
    return x + h.transpose(1, 2), xf[..., -(k - 1) :]


def _clone(state):
    """A copy of a (nested lists/tuples of) tensor state; None entries (no history yet) stay None."""
    if state is None:
        return None
    if isinstance(state, (list, tuple)):
        return type(state)(_clone(s) for s in state)
    return state.clone()


def _tensors(state):
    """The tensors of a nested state, in order."""
    if state is None:
        return []
    if isinstance(state, (list, tuple)):
        return [t for s in state for t in _tensors(s)]
    return [state]


def _moss_layer(layer, x, pos0, cache):
    """MossLayer.forward on positions [pos0, pos0+t) (pos0: an int64 scalar tensor) attending to the cached
    keys/values of the previous positions."""
    b, t, d = x.shape
    q, k, v = layer.qkv(layer.norm1(x)).reshape(b, t, 3, layer.heads, layer.head_dim).unbind(2)
    frequency = torch.exp(
        torch.arange(layer.head_dim // 2, device=x.device, dtype=torch.float32)
        * (-math.log(10000.0) * 2 / layer.head_dim)
    )
    steps = torch.arange(t, device=x.device)
    phase = (pos0 + steps).to(torch.float32)[:, None] * frequency[None, :]
    cos, sin = phase.cos()[None, :, None, :], phase.sin()[None, :, None, :]

    def rope(z):
        zf = z.float().reshape(b, t, layer.heads, layer.head_dim // 2, 2)
        r, i = zf.unbind(-1)
        return torch.stack((r * cos - i * sin, r * sin + i * cos), -1).flatten(-2).to(z.dtype)

    q, k = rope(q), rope(k)
    q, k, v = (z.transpose(1, 2) for z in (q, k, v))
    if cache is not None:
        k = torch.cat((cache[0].to(k.dtype), k), 2)
        v = torch.cat((cache[1].to(v.dtype), v), 2)
    width = k.shape[2]
    # query i sits at pos0 + i, key j at pos0 - (width - t) + j
    delta = (steps + (width - t))[:, None] - torch.arange(width, device=x.device)[None, :]
    valid = (delta >= 0) & (delta < layer.context)
    bias = x.new_full((t, (width + 7) // 8 * 8), float("-inf"))
    bias[:, :width].masked_fill_(valid, 0)
    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[:, :width])
    keep = layer.context - 1
    cache = (k[:, :, -keep:], v[:, :, -keep:]) if keep > 0 else None
    x = x + layer.attention_scale * layer.attention_out(attn.transpose(1, 2).reshape(b, t, d))
    return x + layer.ffn_scale * layer.ffn(layer.norm2(x)), cache


class ChunkGraphs:
    """CUDA graphs of `ISTFTStream._trunk` shared by the streams of one model, one per chunk shape: (chunk frames,
    stream position up to `ISTFTStream.full_frames`, past which every KV cache is full and the shapes stop changing;
    the position itself is a graph input). Shapes are captured while `capturing` (`capture`: a warm-up decode);
    afterwards streams replay the graph of a captured shape and run any other shape eagerly. A replay runs the eager
    kernels on copies of the same inputs, so its outputs are bit-identical to the eager decode."""

    def __init__(self):
        self.graphs: dict[tuple[int, int], _ChunkGraph] = {}
        self.capturing = False
        self.pool = None

    def capture(self, model: ISTFTCodec, chunks) -> None:
        """Capture the shapes of one stream decoding ``chunks`` frames per call from the start (zero codes)."""
        stream = ISTFTStream(model, graphs=self)
        self.capturing = True
        try:
            for n in chunks:
                stream.decode(np.zeros((int(n), stream.n_codebooks), np.int64))
        finally:
            self.capturing = False

    def run(self, stream, key, packed, frames, state):
        graph = self.graphs.get(key)
        if graph is None:
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            graph = self.graphs[key] = _ChunkGraph(stream, packed, frames, state, self.pool)
        return graph.run(packed, frames, state)


class _ChunkGraph:
    """One captured `_trunk` shape with static inputs (codes, position, state before) and outputs."""

    def __init__(self, stream, packed, frames, state, pool):
        device = stream.device
        self.packed = packed.to(device)
        self.frames = torch.full((), frames, dtype=torch.int64, device=device)
        self.state = _clone(state)
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):  # lazy initialization (cuBLAS handles, workspaces) outside the capture
            stream._trunk(self.packed, self.frames, *self.state)
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.ys, self.out = stream._trunk(self.packed, self.frames, *self.state)

    def run(self, packed, frames, state):
        """Replay on ``state`` -> the head outputs (graph buffers, valid until the next replay) and a copy of the state
        after."""
        self.packed.copy_(packed)
        self.frames.fill_(frames)
        for dst, src in zip(_tensors(self.state), _tensors(state), strict=True):
            dst.copy_(src)
        self.graph.replay()
        return self.ys, _clone(self.out)


class ISTFTStream:
    """Incremental decoder over one utterance; feed undelayed code frames [n, codebooks] in order."""

    def __init__(self, model: ISTFTCodec, max_chunk: int = 32, graphs: ChunkGraphs | None = None):
        if not isinstance(model.head, MultiResISTFTHead):
            raise ValueError("ISTFTStream needs the multi-resolution iSTFT head")
        self.model = model
        self.max_chunk = int(max_chunk)
        self.n_codebooks = 1 + model.cfg["residual_codebooks"]
        self.frames = 0
        blocks = list(model.semantic_head)
        if not all(isinstance(b, SemanticConvNeXt) for b in blocks[:-1]):
            raise ValueError("ISTFTStream: unexpected semantic head layout")
        self.sem_blocks, self.sem_out = blocks[:-1], blocks[-1]
        self.sem_hist = [None] * len(self.sem_blocks)
        head = model.head
        self.head_hist = [None] * len(head.blocks)
        self.kv = [[None] * len(s.layers) if isinstance(s, MossStage) else None for s in model.stages]
        # frames after which every KV cache holds its full window (the state shapes stop changing)
        self.full_frames, cpf = 0, 1
        for stage in model.stages:
            if isinstance(stage, Patch):
                cpf *= stage.size
            elif isinstance(stage, MossStage):
                keep = max(layer.context for layer in stage.layers) - 1
                self.full_frames = max(self.full_frames, -(-keep // cpf))
        self.device = next(model.parameters()).device
        windows = [getattr(head, f"window_{i}").detach() for i in range(len(head.n_ffts))]
        self.head = IstftHeadStream(
            head.n_ffts, head.hop, windows, head.mixer.weight[0].detach(), head.max_log_magnitude, self.device
        )
        rs = model.to_external
        self.resampler = ResamplerStream(
            rs.kernel[:, 0].detach(), rs.orig_freq // rs.gcd, rs.new_freq // rs.gcd, rs.width, self.device
        )
        self.graphs = graphs if self.device.type == "cuda" else None

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """The next code frames [n, codebooks] -> the samples now final. A residual code (codebook >= 1) of -1 is
        missing: that codebook's contribution is left out of the frame, a coarser RVQ decode (the codec trained with
        quantizer dropout)."""
        codes = np.asarray(codes)
        if (
            codes.ndim != 2
            or codes.shape[1] != self.n_codebooks
            or ((codes < -1) | (codes >= _CODEBOOK_SIZE)).any()
            or (codes[:, 0] < 0).any()
        ):
            raise ValueError("Invalid matched codec tokens")
        if len(codes) == 0:
            return np.zeros(0, np.float32)
        prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with torch.inference_mode():
                out = [self._chunk(codes[i : i + self.max_chunk]) for i in range(0, len(codes), self.max_chunk)]
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
        return np.concatenate(out)

    def flush(self) -> np.ndarray:
        with torch.inference_mode():
            out = torch.cat([self.resampler.push(self.head.flush()), self.resampler.flush()])
        return out.cpu().numpy()

    def fork(self) -> "ISTFTStream":
        """An independent stream continuing from this one's state. The trunk state tensors are never written in place
        (every chunk makes new ones), so the twins share them; the head's overlap-add buffers are copied."""
        twin = copy.copy(self)
        twin.sem_hist, twin.head_hist = list(self.sem_hist), list(self.head_hist)
        twin.kv = [None if layers is None else list(layers) for layers in self.kv]
        twin.head, twin.resampler = self.head.copy(), self.resampler.copy()
        return twin

    def _chunk(self, codes):
        n = len(codes)
        packed = torch.as_tensor(np.ascontiguousarray(codes.T[None]), dtype=torch.int64)
        state = (self.sem_hist, self.kv, self.head_hist)
        key = (n, min(self.frames, self.full_frames))
        if self.graphs is not None and (self.graphs.capturing or key in self.graphs.graphs):
            ys, state = self.graphs.run(self, key, packed, self.frames, state)
        else:
            frames = torch.full((), self.frames, dtype=torch.int64, device=self.device)
            ys, state = self._trunk(packed.to(self.device), frames, *state)
        self.sem_hist, self.kv, self.head_hist = state
        out = self.resampler.push(self.head.push(*ys))  # before the next replay overwrites a graph's static ys
        self.frames += n
        return out.cpu().numpy()

    def _trunk(self, packed, frames, sem_hist, kv, head_hist):
        """The GPU half of a chunk, free of side effects (CUDA-graph capturable): codes [1, codebooks, n] int64, the
        stream position (int64 scalar tensor) and the state before -> the iSTFT head's float32 outputs [n, bins] and the
        state after."""
        m, dev = self.model, self.device.type
        sem_hist, head_hist = list(sem_hist), list(head_hist)
        kv = [None if layers is None else list(layers) for layers in kv]
        q = m.quantizer
        with torch.autocast(dev, enabled=False):
            x = q.semantic_quantizer.from_codes(packed[:, :1])[0]
            for i, block in enumerate(self.sem_blocks):
                x, sem_hist[i] = _convnext(block, x, sem_hist[i])
            z = self.sem_out(x).float()
            for i, vq in enumerate(q.quantizer.quantizers):
                code = packed[:, 1 + i]
                part = vq.out_proj(vq.decode_code(code.clamp(min=0)))
                z = z + torch.where(code[:, None, :] >= 0, part, 0.0)  # exactly ``part`` for present codes
        x = z
        cpf = 1
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            for si, stage in enumerate(m.stages):
                if isinstance(stage, Patch):
                    x = stage(x)
                    cpf *= stage.size
                elif isinstance(stage, MossStage):
                    x = stage.input_proj(x.transpose(1, 2))
                    pos0 = frames * cpf
                    for li, layer in enumerate(stage.layers):
                        x, kv[si][li] = _moss_layer(layer, x, pos0, kv[si][li])
                    x = stage.output_proj(x).transpose(1, 2)
                else:
                    raise ValueError(f"ISTFTStream: unsupported stage {type(stage).__name__}")
        head = m.head
        with torch.autocast(dev, enabled=False):
            h = x
            for i, block in enumerate(head.blocks):
                h, head_hist[i] = _convnext(block, h, head_hist[i])
            h = head.norm(h.transpose(1, 2))
            ys = [lin(h)[0].float() for lin in head.outs]
        return ys, (sem_hist, kv, head_hist)
