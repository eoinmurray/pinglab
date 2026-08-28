"""Measure explicit numerical evidence and exp041 frequencies; never simulate."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp033 import evidence, historical, inputs, measurements
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def analyse(identity, frequency_source, *, run_id=None):
    compute = inputs.source(REPO, identity, "compute")
    inputs.configuration(compute)
    imported = compute.record["execution"]["operation"] == "historical-import"
    if compute.record["inputs"] and not imported:
        raise PingstoreError("initial exp033 computation must not have upstream inputs")
    frequencies = inputs.source(REPO, frequency_source, "analyse", experiment="exp041")
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": compute, "frequencies": frequencies},
        run_id=run_id,
    ) as run:
        if imported:
            numbers, coordinates, provenance = historical.analyse(compute, frequencies)
            run.record["historical_analysis"] = provenance
        else:
            numbers, coordinates = measurements.analyse(
                evidence.read(compute.export),
                load_json(frequencies.export / "results.json"),
            )
        write_json_atomic(run.export / "results.json", numbers)
        evidence.write(run.export, coordinates)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp033 v3 compute run"
    )
    parser.add_argument(
        "--frequency-source", required=True, help="completed exp041 v3 analysis run"
    )
    parser.add_argument("--run-id", help="unused v3 identity reserved before dispatch")
    args = parser.parse_args()
    analyse(args.source, args.frequency_source, run_id=args.run_id)


if __name__ == "__main__":
    main()
