"""Render the frozen-digit comparison using its local physical-time figures.

Run present.py --source ANALYSE_ID to read validated measurements and saved input
events. Save figures and their numerical summaries in a separate Pingstore run;
this stage does not simulate networks or select an upstream run automatically."""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp142.custom.metrics import arrays
from experiments.exp142.custom.physical_time_figures import plot_all
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def present(identity, *, run_id=None):
    source = source_run(
        REPO / ".pingstore", identity, experiment="exp142", stage="analyse"
    )
    numbers = load_json(source.file("numbers.json"))
    summary = numbers["plots"]
    if (
        summary["schema"] != "exp142.physical-time-figures/v1"
        or summary["readout_policy"] != "matched-spiking"
    ):
        raise ValueError("requires validated matched-step COBA-LIF output evidence")
    saved, inputs = (
        arrays(source.file("plot-comparison.npz")),
        arrays(source.file("input.npz")),
    )
    with stage_run(
        REPO,
        "exp142",
        "present",
        inputs={"analysis": source},
        configuration={
            "schema": "exp142.figures/v2",
            "layout": "exp142 local physical-time renderer",
            "output": "matching discrete timestep / continuous output roots",
        },
        run_id=run_id,
    ) as run:
        plot_all(
            summary,
            saved,
            inputs["pixels"].reshape(8, 8),
            inputs,
            run.export,
            label=int(inputs["label"]),
            image_index=numbers["input"]["sample_index"],
        )
        for mode in numbers["configuration"]["models"]:
            shutil.copyfile(
                source.file(f"{mode}-circuit.svg"), run.export / f"{mode}-circuit.svg"
            )
        write_json_atomic(
            run.export / "numbers.json",
            numbers,
        )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)
