"""Matplotlib with the headless Agg backend pre-selected.

Import ``plt`` (and ``FFMpegWriter``) from here instead of importing
``matplotlib.pyplot`` directly, so the non-interactive backend is chosen in
exactly one place — before pyplot is first imported anywhere in the package.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

__all__ = ["matplotlib", "plt", "FFMpegWriter"]
