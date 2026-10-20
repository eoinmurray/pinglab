"""Measure one explicit shared-sweep compute run without upstream execution.

Run analyse.py --source COMPUTE_RUN with optional --workers and --reuse-analysis.
The custom estimators retain trajectories, dynamics, costs and decision coverage
in a new Pingstore analysis run; inputs are validated by identity and digest.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp123 import recipe
from experiments.exp123.custom.shared_analysis import analyse_shared
from pingstore.stages import source_run


def analyse(identity, workers=1, reuse_identity=None):
    if workers < 1:
        raise ValueError("workers must be positive")
    source = source_run(
        REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    if source.record["execution"]["configuration"].get("schema") != "exp123.compute/v3":
        raise ValueError("expected the uniform shared-sweep compute protocol")
    reuse = (
        source_run(
            REPO / ".pingstore", reuse_identity, stage="analyse", experiment=recipe.SLUG
        )
        if reuse_identity
        else None
    )
    return analyse_shared(source, workers=workers, reuse=reuse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reuse-analysis")
    args = parser.parse_args()
    print(analyse(args.source, args.workers, args.reuse_analysis))
