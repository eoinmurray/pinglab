"""A Manim scene whose single object is controlled by a JSON file."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from manim import BLUE, Circle, GREEN, RED, Square, Star, Triangle, VGroup
from manim import Scene


STATE_PATH = Path(__file__).with_name("state.json")
COLORS = {
    "blue": BLUE,
    "green": GREEN,
    "red": RED,
}


def read_state() -> dict[str, Any]:
    """Read and minimally validate the live state."""
    data = json.loads(STATE_PATH.read_text())
    shape = str(data.get("shape", "circle")).lower()
    if shape not in {"circle", "square", "triangle", "star"}:
        raise ValueError(f"Unsupported shape: {shape}")

    size = float(data.get("size", 1.5))
    if not 0.1 <= size <= 5:
        raise ValueError("size must be between 0.1 and 5")

    color_name = str(data.get("color", "blue")).lower()
    if color_name not in COLORS:
        raise ValueError(f"Unsupported color: {color_name}")

    return {"shape": shape, "size": size, "color": color_name}


def make_shape(state: dict[str, Any]):
    """Build the object described by state."""
    constructors = {
        "circle": Circle,
        "square": Square,
        "triangle": Triangle,
        "star": Star,
    }
    object_ = constructors[state["shape"]]()
    object_.set(width=state["size"] * 2)
    object_.set_fill(COLORS[state["color"]], opacity=0.65)
    object_.set_stroke(COLORS[state["color"]], width=6)
    return object_


class LiveShape(Scene):
    """Keep a Manim object synchronized with ``state.json``."""

    def construct(self):
        holder = VGroup(make_shape(read_state()))
        last_seen = {"mtime": STATE_PATH.stat().st_mtime_ns}

        def reload_if_changed(group, dt):
            del dt
            try:
                mtime = STATE_PATH.stat().st_mtime_ns
                if mtime == last_seen["mtime"]:
                    return
                last_seen["mtime"] = mtime
                state = read_state()
                group.become(VGroup(make_shape(state)))
                print(f"Updated live object: {state}")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                # Keep the last valid object alive while a bad edit is repaired.
                print(f"Ignoring invalid state: {error}")

        holder.add_updater(reload_if_changed)
        self.add(holder)

        if os.environ.get("MANIM_LIVE_SMOKE_TEST") == "1":
            self.wait(0.1)
        else:
            self.interactive_embed()
