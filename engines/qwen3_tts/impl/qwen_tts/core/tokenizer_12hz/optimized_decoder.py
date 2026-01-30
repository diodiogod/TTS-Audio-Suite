# coding=utf-8
# Optimized decoder with torch.compile and CUDA graphs support
"""
Provides optimized decoding for streaming TTS with:
1. torch.compile for the decoder forward pass
2. CUDA graphs for static-shape decode operations
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class CUDAGraphDecoder:
    """
    Wraps a decoder with CUDA graph capture for static-shape inference.

    CUDA graphs capture a sequence of GPU operations and replay them
    without CPU overhead, giving significant speedup for repeated
    fixed-size operations like streaming decode windows.
    """

    def __init__(
        self,
        decoder: nn.Module,
        static_window_size: int = 80,
        num_quantizers: int = 8,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            decoder: The Qwen3TTSTokenizerV2Decoder module
            static_window_size: Fixed window size for CUDA graph capture
            num_quantizers: Number of codec quantizers (usually 8)
            device: CUDA device
            dtype: Data type for the decoder
        """
        self.decoder = decoder
        self.static_window_size = static_window_size
        self.num_quantizers = num_quantizers
        self.device = device or next(decoder.parameters()).device
        self.dtype = dtype

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_output: Optional[torch.Tensor] = None
        self._is_captured = False

    def warmup_and_capture(self, warmup_runs: int = 3):
        """
        Warm up the decoder and capture CUDA graph.

        Call this once before streaming starts for optimal performance.
        """
        if not torch.cuda.is_available():
            print("[CUDAGraphDecoder] CUDA not available, skipping graph capture")
            return

        # Create static input tensor (will be reused for all captures)
        self._static_input = torch.zeros(
            1, self.num_quantizers, self.static_window_size,
            dtype=torch.long,
            device=self.device
        )

        # Warmup runs to stabilize CUDA state
        print(f"[CUDAGraphDecoder] Warming up with {warmup_runs} runs...")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for i in range(warmup_runs):
                _ = self.decoder(self._static_input)

        torch.cuda.current_stream().wait_stream(s)

        # Capture the graph
        print(f"[CUDAGraphDecoder] Capturing CUDA graph for window_size={self.static_window_size}...")
        self._graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._graph):
            self._static_output = self.decoder(self._static_input)

        self._is_captured = True
        print(f"[CUDAGraphDecoder] Graph captured successfully")

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode codes to audio waveform.

        If codes match the static window size and graph is captured,
        uses CUDA graph replay for faster execution.
        Otherwise falls back to regular forward pass.

        Args:
            codes: [B, num_quantizers, T] tensor of codec indices

        Returns:
            Waveform tensor [B, 1, samples]
        """
        B, Q, T = codes.shape

        # Check if we can use the captured graph
        if (self._is_captured
            and B == 1
            and Q == self.num_quantizers
            and T == self.static_window_size):
            # Copy input to static buffer and replay graph
            self._static_input.copy_(codes)
            self._graph.replay()
            return self._static_output.clone()
        else:
            # Fallback to regular forward
            return self.decoder(codes)


def compile_decoder(
    decoder: nn.Module,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = True,
) -> nn.Module:
    """
    Apply torch.compile to the decoder for optimized execution.

    Args:
        decoder: The Qwen3TTSTokenizerV2Decoder module
        mode: Compilation mode:
            - "default": Good balance of compile time and speedup
            - "reduce-overhead": Minimize framework overhead (good for streaming)
            - "max-autotune": Maximum optimization (longer compile time)
        fullgraph: If True, requires the entire model to be traceable as single graph.
                   Set False if there are dynamic control flows.
        dynamic: If True, enables dynamic shape support (slower but more flexible)

    Returns:
        Compiled decoder module
    """
    if not hasattr(torch, 'compile'):
        print("[compile_decoder] torch.compile not available (requires PyTorch 2.0+)")
        return decoder

    print(f"[compile_decoder] Compiling decoder with mode={mode}, fullgraph={fullgraph}, dynamic={dynamic}")

    compiled = torch.compile(
        decoder,
        mode=mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )

    return compiled


class OptimizedStreamingDecoder:
    """
    Combines torch.compile and CUDA graphs for optimal streaming decode performance.

    Usage:
        # During model initialization:
        opt_decoder = OptimizedStreamingDecoder(
            decoder=model.speech_tokenizer.model.decoder,
            static_window_size=80,  # matches decode_window_frames
        )
        opt_decoder.warmup()

        # During streaming:
        wav = opt_decoder.decode(codes)
    """

    def __init__(
        self,
        decoder: nn.Module,
        static_window_size: int = 80,
        num_quantizers: int = 8,
        use_compile: bool = True,
        use_cuda_graphs: bool = True,
        compile_mode: str = "reduce-overhead",
    ):
        self.original_decoder = decoder
        self.static_window_size = static_window_size
        self.num_quantizers = num_quantizers
        self.use_compile = use_compile
        self.use_cuda_graphs = use_cuda_graphs and torch.cuda.is_available()
        self.compile_mode = compile_mode

        self.device = next(decoder.parameters()).device
        self.dtype = next(decoder.parameters()).dtype

        self._compiled_decoder: Optional[nn.Module] = None
        self._cuda_graph_decoder: Optional[CUDAGraphDecoder] = None
        self._is_warmed_up = False

    def warmup(self, warmup_runs: int = 3):
        """
        Initialize optimizations. Call once before streaming starts.
        """
        if self._is_warmed_up:
            return

        print(f"[OptimizedStreamingDecoder] Starting warmup...")

        # Step 1: Apply torch.compile
        if self.use_compile:
            self._compiled_decoder = compile_decoder(
                self.original_decoder,
                mode=self.compile_mode,
                fullgraph=False,  # Decoder has some dynamic ops
                dynamic=False,    # We use static shapes for streaming
            )
        else:
            self._compiled_decoder = self.original_decoder

        # Step 2: Setup CUDA graphs
        if self.use_cuda_graphs:
            self._cuda_graph_decoder = CUDAGraphDecoder(
                decoder=self._compiled_decoder,
                static_window_size=self.static_window_size,
                num_quantizers=self.num_quantizers,
                device=self.device,
                dtype=self.dtype,
            )
            self._cuda_graph_decoder.warmup_and_capture(warmup_runs=warmup_runs)

        self._is_warmed_up = True
        print(f"[OptimizedStreamingDecoder] Warmup complete")

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode codes to waveform with optimizations.

        Args:
            codes: [B, num_quantizers, T] tensor

        Returns:
            Waveform tensor
        """
        if not self._is_warmed_up:
            # Auto-warmup on first call
            self.warmup()

        if self._cuda_graph_decoder is not None:
            return self._cuda_graph_decoder.decode(codes)
        elif self._compiled_decoder is not None:
            return self._compiled_decoder(codes)
        else:
            return self.original_decoder(codes)


def create_optimized_tokenizer_decode(tokenizer, static_window_size: int = 80):
    """
    Patch a Qwen3TTSTokenizer instance to use optimized decoding.

    Args:
        tokenizer: Qwen3TTSTokenizer instance
        static_window_size: Window size for CUDA graph optimization

    Returns:
        The patched tokenizer (same instance, modified in place)
    """
    decoder = tokenizer.model.decoder
    num_quantizers = tokenizer.config.decoder_config.num_quantizers

    opt = OptimizedStreamingDecoder(
        decoder=decoder,
        static_window_size=static_window_size,
        num_quantizers=num_quantizers,
        use_compile=True,
        use_cuda_graphs=True,
    )

    # Store reference to prevent garbage collection
    tokenizer._optimized_decoder = opt

    # Warmup
    opt.warmup()

    return tokenizer
