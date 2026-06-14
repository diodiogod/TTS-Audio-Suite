"""
Dots TTS progress callback.

Provides suite-owned console progress output during streaming generation.
"""

import time


class DotsTTSProgressTracker:
    """Track Dots streaming progress with live it/s and RTF reporting."""

    def __init__(self, max_generate_length: int, sample_rate: int):
        self.max_generate_length = max(1, int(max_generate_length or 1))
        self.sample_rate = max(1, int(sample_rate or 1))
        self.generated_chunks = 0
        self.emitted_samples = 0
        self.start_time = time.time()
        self.last_print_time = self.start_time
        self.last_print_chunks = 0

    def update(self, chunk_samples: int) -> None:
        self.generated_chunks += 1
        self.emitted_samples += max(0, int(chunk_samples or 0))

        current_time = time.time()
        should_print = (
            self.generated_chunks == 1 or
            current_time - self.last_print_time >= 0.5
        )
        if not should_print:
            return

        chunk_delta = self.generated_chunks - self.last_print_chunks
        time_delta = current_time - self.last_print_time
        its = chunk_delta / time_delta if time_delta > 0 else 0.0
        elapsed = current_time - self.start_time
        audio_seconds = self.emitted_samples / self.sample_rate
        rtf = elapsed / audio_seconds if audio_seconds > 0 else 0.0

        bar_width = 12
        display_chunks = min(self.generated_chunks, self.max_generate_length)
        filled = int(bar_width * display_chunks / self.max_generate_length)
        filled = min(filled, bar_width)
        progress_bar = f"[{'█' * filled}{'░' * (bar_width - filled)}] {display_chunks}/{self.max_generate_length}"

        print(
            f"\r   Progress: {progress_bar} | {its:.1f} it/s | {elapsed:.0f}s | "
            f"{audio_seconds:.1f}s audio | RTF {rtf:.2f}      ",
            end="",
            flush=True,
        )

        self.last_print_time = current_time
        self.last_print_chunks = self.generated_chunks

    def end(self) -> None:
        total_time = time.time() - self.start_time
        avg_its = self.generated_chunks / total_time if total_time > 0 else 0.0
        audio_seconds = self.emitted_samples / self.sample_rate
        rtf = total_time / audio_seconds if audio_seconds > 0 else 0.0
        print(
            f"\r   Complete: {self.generated_chunks} chunks in {total_time:.1f}s "
            f"(avg {avg_its:.1f} it/s, {audio_seconds:.1f}s audio, RTF {rtf:.2f})"
            + " " * 20
        )
