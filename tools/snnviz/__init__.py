"""Composable still-image and animation tools for SNN recordings."""

from .animation import FrameTimeline, save_animation
from .contracts import Recording, RecordingError
from .layouts import grid_layout
from .loaders import load_snnsim_recording
from .scene import Panel, Scene
from .transforms import exponential_trace, projection_activity, representative_frame

__all__ = [
    "FrameTimeline", "Panel", "Recording", "RecordingError", "Scene",
    "exponential_trace", "grid_layout", "load_snnsim_recording",
    "projection_activity", "representative_frame", "save_animation",
]
