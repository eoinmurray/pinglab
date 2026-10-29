"""Durable Modal dispatch, status checks and collection for Pinglab experiments."""

import sys
from pathlib import Path

# Match experiment scripts' resolution of the repository-local Pingstore package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
