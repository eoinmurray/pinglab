"""Timeline sampling and backend encoding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation


@dataclass(frozen=True)
class FrameTimeline:
    steps: np.ndarray
    dt_ms: float

    @classmethod
    def sample(
        cls, steps: int, *, frames: int, dt_ms: float
    ) -> "FrameTimeline":
        if steps <= 0 or frames <= 0 or frames > steps:
            raise ValueError("require 0 < frames <= steps")
        return cls(np.linspace(0, steps - 1, frames, dtype=int), dt_ms)

    def time_ms(self, frame: int) -> float:
        return float(self.steps[frame] * self.dt_ms)


def save_animation(
    figure,
    update: Callable[[int], object],
    output: str | Path,
    *,
    frames: int,
    fps: int = 25,
    bitrate: int = 3800,
) -> FuncAnimation:
    animation = FuncAnimation(
        figure, update, frames=frames, interval=1000 / fps, blit=False
    )
    animation.save(
        Path(output), writer=FFMpegWriter(fps=fps, bitrate=bitrate)
    )
    return animation
