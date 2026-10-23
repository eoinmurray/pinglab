"""Render the transition raster and controls from explicit analysis ancestry.

Run present.py --source ANALYSE_ID. PNG, SVG and PDF exports use the same saved
spikes and recorded settings; a native circuit diagram accompanies the report.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp148.custom.figures import transition_figure
from experiments.exp148.custom.metrics import validate_events
from pingstore.stages import source_run, stage_run


def present(*, source, run_id=None):
    analysed = source_run(
        REPO / ".pingstore", source, experiment="exp148", stage="analyse"
    )
    ref = analysed.record["inputs"]["compute"]
    computed = source_run(
        REPO / ".pingstore",
        ref["run_id"],
        experiment="exp148",
        stage="compute",
        reference=ref,
    )
    cfg = computed.record["execution"]["configuration"]
    if cfg != analysed.record["execution"]["configuration"]:
        raise ValueError("analysis and compute settings differ")
    with stage_run(
        REPO,
        "exp148",
        "present",
        inputs={"analyse": analysed, "compute": computed},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        with np.load(computed.file("events.npz")) as saved:
            validate_events(saved, cfg)
            events = {
                p: (saved[p + "_time_ms"], saved[p + "_cell"]) for p in ("E", "I")
            }
            fig = transition_figure(events, cfg)
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(run.export / f"transition.{suffix}")
        plt.close(fig)
        shutil.copyfile(computed.file("circuit.svg"), run.export / "circuit.svg")
        shutil.copyfile(analysed.file("results.json"), run.export / "numbers.json")
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    present(**vars(parser.parse_args()))
