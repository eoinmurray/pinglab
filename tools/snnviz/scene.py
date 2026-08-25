"""An intentionally thin composition layer over Matplotlib."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import matplotlib.pyplot as plt

from .contracts import Recording


class Panel(Protocol):
    def draw(self, ax: Any, recording: Recording) -> Any: ...

    def update(self, frame: int) -> object: ...


@dataclass
class Scene:
    """Compose reusable and bespoke panels without hiding Matplotlib."""

    recording: Recording
    figure: Any = None
    panels: list[Panel] = field(default_factory=list)
    callbacks: list[Callable[[int, "Scene"], object]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if self.figure is None:
            self.figure = plt.figure()

    def add(self, panel: Panel, *, axis: Any = None) -> Any:
        """Draw a panel on a caller-owned axis or a new default axis."""

        ax = axis if axis is not None else self.figure.add_subplot(1, 1, 1)
        panel.draw(ax, self.recording)
        self.panels.append(panel)
        return ax

    def on_frame(self, callback: Callable[[int, "Scene"], object]) -> None:
        self.callbacks.append(callback)

    def update(self, frame: int) -> tuple[object, ...]:
        artists = [panel.update(frame) for panel in self.panels]
        artists.extend(
            callback(frame, self) for callback in self.callbacks
        )
        return tuple(artists)
