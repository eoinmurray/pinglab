"""Atomically update the state watched by the live Manim scene."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


STATE_PATH = Path(__file__).with_name("state.json")


def main() -> None:
    current = json.loads(STATE_PATH.read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=("circle", "square", "triangle", "star"))
    parser.add_argument("--color", choices=("blue", "green", "red"))
    parser.add_argument("--size", type=float)
    args = parser.parse_args()

    updates = {key: value for key, value in vars(args).items() if value is not None}
    current.update(updates)

    with tempfile.NamedTemporaryFile(
        "w", dir=STATE_PATH.parent, delete=False, prefix="state.", suffix=".json"
    ) as temporary:
        json.dump(current, temporary, indent=2)
        temporary.write("\n")
        temporary_path = temporary.name
    os.replace(temporary_path, STATE_PATH)
    print(json.dumps(current))


if __name__ == "__main__":
    main()
