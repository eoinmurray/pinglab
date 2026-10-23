"""Validate one explicit HCO compute run and measure its spike ordering.

Run analyse.py --source COMPUTE_ID. Check full float64 trajectories against an
independent gate-first Euler calculation and retain measured numbers separately
from the immutable source. Toolbox reference agreement remains untested.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp147.custom.validation import validate_and_measure
from experiments.exp147.recipe import validate_configuration
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def analyse(*, source, run_id=None):
    computed = source_run(
        REPO / ".pingstore", source, experiment="exp147", stage="compute"
    )
    cfg = computed.record["execution"]["configuration"]
    validate_configuration(cfg)
    with np.load(computed.file("traces.npz")) as data:
        numbers = validate_and_measure(data, cfg)
    numbers["timing"] = load_json(computed.file("timing.json"))
    with stage_run(
        REPO,
        "exp147",
        "analyse",
        inputs={"compute": computed},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        write_json_atomic(run.export / "results.json", numbers)
        shutil.copyfile(computed.file("traces.npz"), run.export / "traces.npz")
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    analyse(**vars(parser.parse_args()))
