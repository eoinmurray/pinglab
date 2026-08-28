"""Analyse retained exp038 probes and histories; never simulate or publish."""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp038 import evidence, inputs, measurements, recipe
from experiments.helpers.frontier import summarize_frontier
from pingstore.contracts import PingstoreError, load_json, write_json_atomic

MEASUREMENT = {
    "schema": "exp038.measurement/v1",
    "baseline": "retained best validation accuracy, final epoch validation accuracy and final epoch rate_e; mean and SEM across three seeds",
    "ei": "official-test accuracy and E/I mean rates; mean and sample SD across seeds 42-44",
    "uniform": "saved population mean rates across uniform-input trials, seed 42; unweighted E+I sum for the retained overlay",
    "raster": "full-trial rate from all cells; independent RNG(0) sample of 200 E and 64 I cells for display",
}


def analyse(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "compute")
    cfg, bank, contract = inputs.compute_evidence(REPO, source)
    histories = evidence.histories(bank.export, contract)
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": source, "bank": bank},
        run_id=run_id,
        configuration=MEASUREMENT,
    ) as run:
        baseline, ei, uniform, rasters = [], [], [], []
        for cell in contract["cells"]:
            name = cell["cell_name"]
            m = histories[name]
            last = m["epochs"][-1]
            baseline.append(
                {
                    "cell_name": name,
                    "model": cell["model"],
                    "seed": cell["seed"],
                    "rate_target_hz": cell["rate_target_hz"],
                    "rate_target_display": recipe.rate_target_display(
                        cell["rate_target_hz"]
                    ),
                    "best_acc": float(m["best_acc"]),
                    "best_epoch": int(m["best_epoch"]),
                    "final_acc": float(last["acc"]),
                    "rate_e": float(last.get("rate_e") or 0.0),
                }
            )
        for job in recipe.jobs(cfg):
            directory = source.export / job["path"]
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
            m = evidence.recordings(directory, train, job)
            if "sample_index" in job:
                data = measurements.raster(directory, train, job)
                filename = f"raster-{len(rasters):03d}.npz"
                np.savez_compressed(run.export / filename, **data)
                rasters.append({"job": job, "file": filename})
            elif job["kind"] == "fi_uniform":
                uniform.append(
                    {
                        "model": job["model"],
                        "input_rate_hz": job["input_rate"],
                        "e_rate_hz": float(m["rate_e_hz"]),
                        "i_rate_hz": float(m["rate_i_hz"]),
                    }
                )
            else:
                ei.append(
                    {
                        "seed": job["seed"],
                        "ei_strength": job["ei_strength"],
                        "acc": float(m["best_acc"]),
                        "n_total": m["n_total"],
                        "hid_rate_hz": evidence.population_rate(m, "hid"),
                        "inh_rate_hz": evidence.population_rate(m, "inh"),
                    }
                )
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp038.analysis/v1",
                "recipe": cfg,
                "measurement": MEASUREMENT,
                "checkpoint_policy": recipe.CHECKPOINT_POLICY,
                "checkpoint_provenance": contract["checkpoints"],
                "git_sha_train": contract["configs"][
                    recipe.cell_name("coba", None, 42)
                ].get("git_sha"),
                "config": {
                    "dataset": "mnist",
                    "models": recipe.MODELS,
                    "rate_target_grid_hz": [
                        t for t in recipe.RATE_TARGET_GRID_HZ if t is not None
                    ],
                    "max_samples": 7000,
                    "evaluation_pool_samples": 10000,
                    "epochs": 50,
                    "t_ms": 200.0,
                    "dt": 0.1,
                    "frontier_seeds": recipe.SEEDS_BASELINE,
                    "quantitative_inference_seeds": recipe.SEEDS_BASELINE,
                    "illustrative_raster_seed": 42,
                    "evaluation_samples_per_seed": sorted({p["n_total"] for p in ei}),
                    "fr_strength_upper": recipe.FR_STRENGTH_UPPER,
                },
                "baseline_results": baseline,
                "frontier_summary": summarize_frontier(baseline),
                "ei_sweep": ei,
                "ei_sweep_summary": measurements.summarize_ei_points(ei),
                "fi_sweep_uniform": uniform,
                "plot_data": {
                    "uniform": [
                        {**row, "e_plus_i_rate_hz": row["e_rate_hz"] + row["i_rate_hz"]}
                        for row in uniform
                    ],
                },
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
        p.exit(1, f"exp038 analyse: {exc}\n")


if __name__ == "__main__":
    main()
