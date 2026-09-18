"""Measure explicit full recordings; never simulate or publish."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp054 import evidence, inputs, measurements, recipe
from pingstore.contracts import PingstoreError, write_json_atomic


def analyse(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "compute")
    cfg = evidence.compute_contract(source)
    if source.record["inputs"]:
        raise PingstoreError("native exp054 compute must not have upstream inputs")
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": source},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        coords = measurements.recordings(source, cfg)
        numbers = measurements.summary(coords, cfg)
        coords.update(schema="exp054.analysis/v2", recipe=cfg)
        write_json_atomic(run.export / "results.json", numbers)
        evidence.write(run.export, coords)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp054 compute run")
    parser.add_argument("--run-id", help="fresh v4 identity reserved before dispatch")
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
