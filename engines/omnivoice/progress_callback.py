"""
OmniVoice progress callback.

Tracks real iterative decoding steps per second for OmniVoice generation.
"""

import time


class OmniVoiceGenerationProgress:
    """Render real iterative decode progress with it/s."""

    def __init__(self, total_steps: int):
        self.total_steps = max(int(total_steps or 0), 0)
        self.start_time = time.time()
        self.last_print_time = self.start_time
        self.last_step = 0
        self.rendered = False

    def update(self, current_step: int, force: bool = False):
        current_step = max(0, min(int(current_step), self.total_steps))
        current_time = time.time()
        if not force and current_time - self.last_print_time < 0.25:
            return

        delta_steps = current_step - self.last_step
        delta_time = max(current_time - self.last_print_time, 1e-6)
        its = delta_steps / delta_time
        elapsed = current_time - self.start_time
        remaining_steps = max(self.total_steps - current_step, 0)
        eta = remaining_steps / its if its > 0 else 0.0

        bar_width = 12
        filled = int(bar_width * current_step / self.total_steps) if self.total_steps > 0 else 0
        progress_bar = f"[{'█' * filled}{'░' * (bar_width - filled)}] {current_step}/{self.total_steps}"
        print(
            f"\r   Progress: {progress_bar} | {its:.1f} it/s | {elapsed:.0f}s | ETA {eta:.0f}s      ",
            end="",
            flush=True,
        )
        self.last_print_time = current_time
        self.last_step = current_step
        self.rendered = True

    def end(self, final_step: int):
        final_step = max(0, int(final_step))
        total_time = max(time.time() - self.start_time, 1e-6)
        avg_its = final_step / total_time
        if self.rendered:
            print(
                f"\r   Complete: {final_step} steps in {total_time:.1f}s (avg {avg_its:.1f} it/s)"
                + " " * 30
            )

    def abort(self):
        if self.rendered:
            print()
