"""Explicitly assemble completed paired compute phases into one full result."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp126.compute_parts.assembly import assemble

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="explicit completed phase run ID; repeat for both phases",
    )
    parser.add_argument("--run-id", help="optional reserved full compute identity")
    args = parser.parse_args()
    assemble(REPO, args.source, args.run_id)
