"""Validate an explicit compute run and measure the three protocol windows.

Run analyse.py --source COMPUTE_ID. Firing rates and hold-window population-cycle
estimates are saved in a new analysis run, preserving the compute ancestry.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp148.custom.metrics import measure
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def analyse(*, source, run_id=None):
    computed = source_run(
        REPO / ".pingstore", source, experiment="exp148", stage="compute"
    )
    cfg = computed.record["execution"]["configuration"]
    audit = load_json(computed.file("verification.json"))
    if not audit["unchanged_other_weights"] or not audit["final_gain_verified"]:
        raise ValueError("compute run did not verify the scheduled weights")
    with np.load(computed.file("events.npz")) as events:
        numbers = measure(events, cfg)
    with stage_run(
        REPO,
        "exp148",
        "analyse",
        inputs={"compute": computed},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        write_json_atomic(run.export / "results.json", numbers)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    analyse(**vars(parser.parse_args()))
