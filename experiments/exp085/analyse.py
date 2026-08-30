"""Measure explicitly retained exp085 compute evidence; never simulate or plot."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp085 import evidence, inputs, measurements, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def measure(root, acquisition):
    jobs = {job["id"]: job for job in acquisition["jobs"]}

    def read(name):
        return evidence.recording(root, jobs[name])

    # The original uncoupled analysis explicitly converted its spike recordings.
    uncoupled = measurements.analyse_uncoupled(
        {k: v.astype(np.uint8) for k, v in read("uncoupled").items()}
    )
    baseline = read("prc-baseline")
    left, right = recipe.reference_cycle(baseline)
    if acquisition["reference_cycle"] != {"left_step": left, "next_step": right}:
        raise PingstoreError(
            "retained exp085 reference cycle differs from the baseline"
        )
    phase_response, examples = measurements.analyse_phase_response(baseline, read)
    conditions, traces, recordings = [], {}, {}
    for name, label, ee, ei in recipe.PATHWAYS:
        data = read(f"pathway-{name}")
        record, trace = measurements.analyse_pathway_branch(data)
        conditions.append(
            {**record, "id": name, "label": label, "K_EE": ee, "K_EI": ei}
        )
        traces[name] = trace
        if name in ("none", "e_to_e"):
            recordings[name] = data
    pathways: dict[str, object] = {
        "coupling_onset_ms": recipe.COUPLING_ONSET_MS,
        "shared_delay_ms": recipe.COUPLING_DELAY_MS,
        "classification": {
            "final_window_ms": 500.0,
            "maximum_absolute_drift_cycles_per_s": 0.25,
            "minimum_phase_concentration": 0.95,
        },
        "conditions": conditions,
    }
    mechanism, mechanism_traces = measurements.analyse_event_aligned_mechanism(
        recordings
    )
    results = measurements.experiment_record(
        uncoupled, phase_response, pathways, mechanism
    )
    plot_data = {
        "uncoupled": uncoupled,
        "phase_response_examples": examples,
        "pathway_traces": traces,
        "mechanism_traces": mechanism_traces,
    }
    return results, plot_data


def analyse(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "compute")
    if source.record["inputs"]:
        raise PingstoreError("standalone exp085 compute must have no upstream inputs")
    acquisition = evidence.compute_export(source.export, recipe.configuration())
    with inputs.execution(
        REPO, "analyse", sources={"compute": source}, run_id=run_id
    ) as run:
        results, data = measure(source.export, acquisition)
        evidence.save_plot_data(run.export, data)
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp085.analysis/v1",
                "recipe": recipe.configuration(),
                "results": results,
            },
        )
        # Retain the network definition for rendering, not an upstream simulation.
        write_json_atomic(
            run.export / "network.json", load_json(source.file("graphs", "both.json"))
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp085 v3 compute ID"
    )
    parser.add_argument("--run-id", help="unused source-neutral v3 reservation")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"exp085 analyse: {exc}\n")


if __name__ == "__main__":
    main()
