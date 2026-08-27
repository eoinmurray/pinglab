"""Retired combined runner: invoke an independent exp041 stage explicitly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    raise SystemExit(
        "exp041 now has independent stages: experiments/exp041/compute.py --source BANK; "
        "experiments/exp041/analyse.py --source COMPUTE; "
        "experiments/exp041/present.py --source ANALYSE. No combined execution."
    )
