"""
Qwen3-TTS Progress Callback for ComfyUI Integration

Provides real-time progress tracking during generation by hooking into
the transformers model's generation loop via the Streamer interface.
"""

from typing import Optional
import torch
import time


class Qwen3TTSProgressStreamer:
    """
    Streamer for tracking Qwen3-TTS generation progress with ComfyUI progress bars.

    Implements transformers' BaseStreamer interface to receive tokens during generation.
    Provides console output with it/s (iterations per second) matching Step Audio EditX style.
    """

    def __init__(self, max_new_tokens: int, progress_bar=None, text_input: str = ""):
        """
        Initialize progress streamer.

        Args:
            max_new_tokens: Maximum tokens to generate (for progress bar total)
            progress_bar: ComfyUI progress bar instance (with .update() method)
            text_input: Input text for estimating total tokens (optional)
        """
        self.max_new_tokens = max_new_tokens
        self.progress_bar = progress_bar
        self.generated_tokens = 0
        self.start_time = time.time()
        self.last_print_time = time.time()
        self.last_print_tokens = 0

        # Estimate total tokens based on input text length
        # Rough heuristic: ~1.5 tokens per character for TTS (conservative estimate)
        if text_input:
            estimated_tokens = int(len(text_input) * 1.5)
            # Cap estimate to max_new_tokens
            self.estimated_total = min(estimated_tokens, max_new_tokens)
        else:
            # No text provided, use max_new_tokens as fallback
            self.estimated_total = max_new_tokens

    def put(self, value: torch.Tensor):
        """
        Called by transformers during generation with new token IDs.

        Args:
            value: Tensor of token IDs (shape: [batch_size, sequence_length])
        """
        # Check for interruption (ComfyUI interrupt signal)
        import comfy.model_management as model_management
        if model_management.interrupt_processing:
            raise InterruptedError("Qwen3-TTS generation interrupted by user")

        # Count new tokens generated (last dimension is sequence length)
        if value.ndim >= 2:
            new_tokens = value.shape[-1]
        else:
            new_tokens = 1

        self.generated_tokens += new_tokens

        # Update GUI progress bar
        if self.progress_bar is not None:
            try:
                self.progress_bar.update(new_tokens)
            except Exception:
                pass  # Ignore progress bar errors

        # Console progress output (update every 0.5 seconds to avoid spam)
        current_time = time.time()
        if current_time - self.last_print_time >= 0.5:
            tokens_since_print = self.generated_tokens - self.last_print_tokens
            time_since_print = current_time - self.last_print_time

            # Calculate it/s (tokens per second)
            if time_since_print > 0:
                its = tokens_since_print / time_since_print
            else:
                its = 0

            # Create visual progress bar (12 chars wide like Step Audio EditX)
            # Use estimated_total for more accurate progress indication
            bar_width = 12
            filled = int(bar_width * self.generated_tokens / self.estimated_total) if self.estimated_total > 0 else 0
            filled = min(filled, bar_width)  # Don't overflow bar
            progress_bar_str = f"[{'█' * filled}{'░' * (bar_width - filled)}] {self.generated_tokens}/{self.estimated_total}"

            # Get elapsed and remaining time from ComfyUI progress bar if available (like Step Audio EditX)
            job_remaining = None
            job_elapsed = None
            if self.progress_bar:
                if hasattr(self.progress_bar, 'get_job_remaining_str'):
                    job_remaining = self.progress_bar.get_job_remaining_str()
                if hasattr(self.progress_bar, 'get_job_elapsed'):
                    job_elapsed = self.progress_bar.get_job_elapsed()

            # Use job elapsed if available, otherwise calculate from start
            elapsed = job_elapsed if job_elapsed else (current_time - self.start_time)

            # Print progress (use \r to overwrite same line, match Step Audio EditX style)
            # Format: Progress: [███░░░] current/total | it/s | elapsed | ETA (if available)
            if job_remaining:
                print(f"\r   Progress: {progress_bar_str} | {its:.1f} it/s | {elapsed:.0f}s | {job_remaining}      ", end='', flush=True)
            else:
                print(f"\r   Progress: {progress_bar_str} | {its:.1f} it/s | {elapsed:.0f}s      ", end='', flush=True)

            self.last_print_time = current_time
            self.last_print_tokens = self.generated_tokens

    def end(self):
        """Called when generation completes."""
        # Clear progress line and print final stats
        total_time = time.time() - self.start_time
        avg_its = self.generated_tokens / total_time if total_time > 0 else 0
        print(f"\r   Complete: {self.generated_tokens} tokens in {total_time:.1f}s (avg {avg_its:.1f} it/s)" + " " * 30)
