"""
ASR Progress Callback for ComfyUI Integration
Token-level progress tracking with it/s and ETA.
"""

import time


class ASRProgressStreamer:
    def __init__(self, max_new_tokens: int, progress_bar=None, label: str = "ASR"):
        self.max_new_tokens = max_new_tokens
        self.progress_bar = progress_bar
        self.label = label
        self.generated_tokens = 0
        self.start_time = time.time()
        self.last_update = 0.0

    def put(self, value):
        # value is a batch of token ids; count tokens
        try:
            new_tokens = len(value)
        except Exception:
            new_tokens = 1

        self.generated_tokens += new_tokens

        if self.progress_bar is not None:
            try:
                self.progress_bar.update(new_tokens)
            except Exception:
                pass

        now = time.time()
        if now - self.last_update >= 0.5:
            elapsed = now - self.start_time
            its = self.generated_tokens / elapsed if elapsed > 0 else 0.0
            bar_width = 12
            total = self.max_new_tokens if self.max_new_tokens > 0 else 1
            filled = int(min(1.0, self.generated_tokens / total) * bar_width)
            bar = f"[{'█' * filled}{'░' * (bar_width - filled)}]"
            print(f"\r   {self.label} Progress: {bar} {self.generated_tokens}/{total} | {its:.1f} it/s", end="", flush=True)
            self.last_update = now

    def end(self):
        elapsed = time.time() - self.start_time
        avg = self.generated_tokens / elapsed if elapsed > 0 else 0.0
        print(f"\r   {self.label} Complete: {self.generated_tokens} tokens in {elapsed:.1f}s (avg {avg:.1f} it/s)" + " " * 20)
