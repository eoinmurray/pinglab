"""Validate an explicit Task 1 compute run and measure its voltage trajectory.

Run analyse.py --source COMPUTE_ID. Saved traces are checked against input,
gate and crossing contracts and an independent scalar Euler implementation.
The absent toolbox reference is reported explicitly, never counted as a pass.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp146.custom.validation import compare_reference, validate_traces
from experiments.exp146.recipe import validate_configuration
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def analyse(*, source, run_id=None):
    computed = source_run(
        REPO / ".pingstore", source, experiment="exp146", stage="compute"
    )
    cfg = computed.record["execution"]["configuration"]
    validate_configuration(cfg)
    with np.load(computed.file("traces.npz")) as data:
        validate_traces(data, cfg)
        numbers = compare_reference(data, cfg)
    numbers["timing"] = load_json(computed.file("timing.json"))
    numbers["duration_ms"] = cfg["duration_ms"]
    numbers["dt_ms"] = cfg["dt_ms"]
    with stage_run(
        REPO,
        "exp146",
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
