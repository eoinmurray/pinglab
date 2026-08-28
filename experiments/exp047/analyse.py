"""Aggregate explicit completed computation; never simulate, draw or publish."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp047 import evidence, inputs, measurements, recipe
from pingstore.contracts import PingstoreError, write_json_atomic


def analyse(identity, *, run_id=None):
    compute = inputs.source(REPO, identity, "compute")
    cfg = evidence.compute_contract(compute)
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": compute},
        run_id=run_id,
        configuration=recipe.MEASUREMENT,
    ) as run:
        rows = evidence.rows(compute.export, compute.directory / "provenance", cfg)
        result = measurements.analyse_rows(rows, cfg)
        evidence.analysis(result, cfg)
        write_json_atomic(run.export / "results.json", result)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="explicit completed exp047 compute ID"
    )
    parser.add_argument("--run-id", help="unused v3 identity reserved before dispatch")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp047 analyse: {exc}\n")


if __name__ == "__main__":
    main()
