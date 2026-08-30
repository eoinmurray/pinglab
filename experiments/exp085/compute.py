"""Acquire the committed exp085 recordings; never analyse, plot or publish."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

import numpy as np
from execution import ExecutionSpec, save_runtime_state, simulate
from experiments.exp085 import evidence, inputs, recipe
from pingstore.contracts import PingstoreError, write_json_atomic
from pingstore.stages import utc_now


def numpy_tensors(values):
    return {key: value.detach().cpu().numpy() for key, value in values.items()}


def acquire(run):
    cfg = run.record["execution"]["configuration"]
    graphs = recipe.graphs()
    for name, graph in graphs.items():
        write_json_atomic(run.export / "graphs" / f"{name}.json", graph)
    completed = []

    def execute(job, drive, *, state=None):
        directory = run.export / "jobs" / job["id"]
        directory.mkdir(parents=True)
        request = {
            **job,
            "seed": recipe.NETWORK_SEED,
            "recording": "full",
            "recording_fields": list(evidence.recording_sizes(job)),
            "graph_sha256": cfg["graph_hashes"][job["graph"]],
            "kind": "simulate",
            "executor": "graph",
        }
        write_json_atomic(directory / "request.json", request)
        np.savez_compressed(directory / "inputs.npz", **numpy_tensors(drive))
        started = utc_now()
        result = simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graphs[job["graph"]],
                inputs=drive,
                seed=recipe.NETWORK_SEED,
                recording_fields=request["recording_fields"],
            ),
            runtime_state=state,
        )
        if result.recordings:
            np.savez_compressed(
                directory / "recordings.npz", **numpy_tensors(result.recordings)
            )
        np.savez_compressed(
            directory / "parameters.npz", **numpy_tensors(result.parameters)
        )
        write_json_atomic(
            run.scratch / "simulations" / f"{job['id']}.json",
            {
                "request": request,
                "started_at": started,
                "completed_at": utc_now(),
                "metrics": result.metrics,
            },
        )
        completed.append(job)
        return result

    preliminary = evidence.jobs([])
    drive = recipe.make_uncoupled_inputs()
    execute(preliminary[0], drive)
    baseline = execute(preliminary[1], recipe.make_phase_response_inputs())
    left, right = recipe.reference_cycle(numpy_tensors(baseline.recordings))
    del baseline
    schedule = recipe.probe_schedule(left, right)
    job_list = evidence.jobs(schedule)
    by_name = {job["id"]: job for job in job_list}
    for probe in schedule:
        execute(
            by_name[probe["id"]],
            recipe.make_phase_response_inputs(
                target=probe["target"], arrival_step=probe["arrival_step"]
            ),
        )
    onset = round(recipe.COUPLING_ONSET_MS / recipe.DT_MS)
    prefix = execute(
        by_name["prefix"], {name: value[:onset] for name, value in drive.items()}
    )
    if prefix.runtime_state is None:
        raise RuntimeError("the uncoupled prefix did not return runtime state")
    save_runtime_state(run.export / "prefix-state", prefix.runtime_state)
    suffix = {name: value[onset:] for name, value in drive.items()}
    for name, *_ in recipe.PATHWAYS:
        execute(
            by_name[f"pathway-{name}"], suffix, state=prefix.runtime_state.detached()
        )
    write_json_atomic(
        run.export / "evidence.json",
        {
            "schema": "exp085.compute/v1",
            "recipe": cfg,
            "reference_cycle": {"left_step": left, "next_step": right},
            "probes": schedule,
            "jobs": completed,
        },
    )
    evidence.compute_export(run.export, cfg)


def compute(*, run_id=None):
    with inputs.execution(REPO, "compute", run_id=run_id) as run:
        acquire(run)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused source-neutral v3 reservation")
    args = parser.parse_args()
    try:
        compute(run_id=args.run_id)
    except (PingstoreError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"exp085 compute: {exc}\n")


if __name__ == "__main__":
    main()
