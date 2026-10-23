"""Run the independent present stage for exp022.

Invoke this CLI from the repository root. The experiment-local implementation
owns argument validation, explicit source pins and Pingstore stage completion.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp022.custom.presentation import main


if __name__ == "__main__":
    main()
