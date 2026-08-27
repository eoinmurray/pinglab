"""Compatibility launcher: exp022 execution now means compute only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.exp022.compute import main

if __name__ == "__main__":
    main()
