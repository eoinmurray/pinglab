"""Measure retained exp037 perturbations; never simulate or publish."""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp037 import evidence, inputs, measurements, recipe
from experiments.helpers.frontier import summarize_frontier
from pingstore.contracts import PingstoreError, load_json, write_json_atomic

MEASUREMENT = {
    "schema": "exp037.measurement/v1",
    "accuracy": "across-seed mean and sample SD; raw per-seed rows retained",
    "frontier": "retained final history acc/rate_e, across-seed mean and SEM",
    "add_normalization": "add Hz divided by arithmetic mean of final baseline history rate_e across seeds",
    "raster": "full-population E rate, RNG(0), sorted sample of 200 E then 64 I cells; original array dtype sum",
}


def analyse(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "compute")
    cfg, bank, contract = inputs.compute_evidence(REPO, source)
    histories = evidence.histories(bank.export, contract)
    rows = measurements.baseline_rows(histories)
    perturb = []
    rasters = []
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": source, "bank": bank},
        run_id=run_id,
        configuration=MEASUREMENT,
    ) as run:
        for job in recipe.jobs(cfg):
            train = contract["configs"][job["cell_name"]]
            evidence.inference_config(
                load_json(
                    source.directory
                    / "provenance/simulations"
                    / job["path"]
                    / "config.json"
                ),
                train,
                job,
            )
            directory = source.export / job["path"]
            m = evidence.recordings(directory, train, job)
            if job["kind"] == "sweep":
                perturb.append(measurements.perturbation_row(m, job))
            else:
                data = measurements.raster(directory, train, job)
                filename = f"raster-{len(rasters):03d}.npz"
                np.savez_compressed(run.export / filename, **data)
                rasters.append({"job": job, "file": filename})
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp037.analysis/v1",
                "recipe": cfg,
                "measurement": MEASUREMENT,
                "checkpoint_policy": recipe.CHECKPOINT_POLICY,
                "checkpoint_provenance": contract["checkpoints"],
                "git_sha_train": contract["configs"][
                    recipe.cell_name(recipe.MODELS[0], None, recipe.SEEDS_BASELINE[0])
                ].get("git_sha"),
                "config": {
                    "dataset": "mnist",
                    "models": recipe.MODELS,
                    "rate_target_grid_hz": [
                        t for t in recipe.RATE_TARGET_GRID_HZ if t is not None
                    ],
                    "max_samples": 7000,
                    "evaluation_pool_samples": cfg["evaluation_samples"],
                    "epochs": 50,
                    "t_ms": 200.0,
                    "dt": 0.1,
                    "frontier_seeds": recipe.SEEDS_BASELINE,
                    "quantitative_inference_seeds": recipe.SEEDS_BASELINE,
                    "illustrative_raster_seed": recipe.SEEDS_BASELINE[0],
                    "evaluation_samples_per_seed": sorted(
                        {r["n_total"] for r in perturb}
                    ),
                    "fr_strength_upper": recipe.FR_STRENGTH_UPPER,
                },
                "baseline_results": rows,
                "frontier_summary": summarize_frontier(rows),
                "perturbation": perturb,
                "perturbation_summary": measurements.summarize_perturbation_rows(
                    perturb
                ),
                "plot_data": measurements.plot_data(rows, perturb),
                "rasters": rasters,
            },
        )
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    a = p.parse_args()
    try:
        analyse(a.source, run_id=a.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        p.exit(1, f"exp037 analyse: {exc}\n")


if __name__ == "__main__":
    main()
