"""Native generation progress and interruption support for Audio8 TTS."""

from __future__ import annotations

import time
from typing import Any, Optional

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


def _make_comfy_progress_bar(total_steps: int) -> Optional[Any]:
    try:
        from comfy.utils import ProgressBar

        return ProgressBar(total_steps)
    except Exception:
        return None


class Audio8TTSStoppingCriteria(StoppingCriteria):
    """Track native codec-frame generation and honor ComfyUI interruption."""

    def __init__(self, max_steps: int, progress_bar: Optional[Any] = None):
        self.max_steps = max(1, int(max_steps))
        self.progress_bar = progress_bar
        self.steps = 0
        self.start_time = time.monotonic()
        self.last_print_time = self.start_time
        self.last_print_step = 0
        self.rendered = False

    @staticmethod
    def _check_interrupted() -> None:
        try:
            import comfy.model_management as model_management
        except Exception:
            return

        checker = getattr(
            model_management,
            "throw_exception_if_processing_interrupted",
            None,
        )
        if callable(checker):
            checker()
        elif getattr(model_management, "interrupt_processing", False):
            raise InterruptedError("Audio8 TTS generation interrupted by user")

    def __call__(self, input_ids, scores, **kwargs):
        del scores, kwargs
        self._check_interrupted()
        self.steps = min(self.steps + 1, self.max_steps)
        if self.progress_bar is not None:
            try:
                self.progress_bar.update(1)
            except Exception:
                pass

        now = time.monotonic()
        if (
            self.steps == 1
            or self.steps >= self.max_steps
            or now - self.last_print_time >= 0.5
        ):
            delta_time = max(now - self.last_print_time, 1e-6)
            delta_steps = self.steps - self.last_print_step
            rate = delta_steps / delta_time
            elapsed = now - self.start_time
            width = 12
            filled = min(
                width,
                int(width * self.steps / self.max_steps),
            )
            bar = "█" * filled + "░" * (width - filled)
            print(
                f"\r   Audio8: [{bar}] {self.steps}/{self.max_steps} | "
                f"{rate:.1f} frames/s | {elapsed:.1f}s      ",
                end="",
                flush=True,
            )
            self.last_print_time = now
            self.last_print_step = self.steps
            self.rendered = True

        # The official model handles EOS itself. Interruption raises above.
        batch_size = int(input_ids.shape[0]) if input_ids.ndim else 1
        return torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=input_ids.device,
        )

    def close(self) -> None:
        elapsed = max(time.monotonic() - self.start_time, 1e-6)
        average = self.steps / elapsed
        if self.rendered:
            print(
                f"\r   Audio8 complete: {self.steps} frames in "
                f"{elapsed:.1f}s ({average:.1f} frames/s)" + " " * 20
            )

    def abort(self) -> None:
        if self.rendered:
            print()


def build_audio8_stopping_criteria(
    max_steps: int,
) -> tuple[StoppingCriteriaList, Audio8TTSStoppingCriteria]:
    """Create the native stopping-criteria list and its progress tracker."""
    tracker = Audio8TTSStoppingCriteria(
        max_steps=max_steps,
        progress_bar=_make_comfy_progress_bar(max(1, int(max_steps))),
    )
    return StoppingCriteriaList([tracker]), tracker
