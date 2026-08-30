"""Prepare measurements, seed summaries and raster samples from retained evidence."""

import argparse
import sys
from pathlib import Path
from statistics import fmean, stdev

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp044 import evidence, inputs, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic

MEASUREMENT = {
    "schema": "exp044.measurement/v1",
    "aggregation": "mean_across_training_seeds",
    "uncertainty": "sample_standard_deviation_divided_by_sqrt_n",
    "history_partition": "validation",
    "gamma_period_estimator": None,
}


def summarize(rows: list[dict], dts: list[float]) -> list[dict]:
    summaries = []
    for dt in dts:
        group = [row for row in rows if row["dt_ms"] == dt]
        summary = {"dt_ms": dt, "n": len(group)}
        for key in ("acc", "e_rate_hz", "i_rate_hz"):
            values = [row[key] for row in group]
            summary[key] = {
                "mean": fmean(values),
                "sem": stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0,
            }
        summaries.append(summary)
    return summaries


def analyse(identity: str, *, run_id: str | None = None) -> str:
    compute = inputs.source(REPO, identity, "compute")
    cfg = inputs.configuration(compute)
    ref = compute.record["inputs"]["bank"]
    bank = inputs.source(
        REPO, ref["run_id"], "compute", experiment="exp022", reference=ref
    )
    retained = load_json(compute.export / "evidence.json")
    contract = evidence.training_contract(bank.export)
    checkpoints = evidence.checkpoints(bank.export, contract)
    if (
        retained.get("schema") != "exp044.compute/v1"
        or retained.get("config") != cfg
        or retained.get("training_contract") != contract
        or retained.get("checkpoint_provenance") != checkpoints
    ):
        raise PingstoreError(
            "compute evidence disagrees with its pinned bank or recipe"
        )
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": compute, "bank": bank},
        run_id=run_id,
        configuration=MEASUREMENT,
    ) as run:
        rows = [
            evidence.measurement(
                compute.file("infer", cell["cell_name"], "metrics.json"),
                cell,
                contract["common"],
                cfg["evaluation_samples"],
            )
            for cell in contract["cells"]
        ]
        curves = evidence.histories(bank.export, contract)
        raster_rows, arrays = [], {}
        raster = cfg["raster"]
        for dt in cfg["dt_sweep_ms"]:
            name = recipe.cell_name(dt, raster["seed"])
            snap = evidence.snapshot(
                compute.file("snapshot", name, "snapshot.npz"),
                dt,
                contract["common"],
            )
            e_full, i_full = snap["spk_e"], snap["spk_i"]
            duration = contract["common"]["t_ms"] / 1000.0
            rng = np.random.default_rng(raster["selection_seed"])
            e_idx = np.sort(
                rng.choice(e_full.shape[1], raster["n_e_plot"], replace=False)
            )
            i_idx = np.sort(
                rng.choice(i_full.shape[1], raster["n_i_plot"], replace=False)
            )
            arrays[name + "__e"] = e_full[:, e_idx].astype(bool)
            arrays[name + "__i"] = i_full[:, i_idx].astype(bool)
            raster_rows.append(
                {
                    "cell_name": name,
                    "dt_ms": dt,
                    "seed": raster["seed"],
                    "t_ms": contract["common"]["t_ms"],
                    "sample_index": raster["sample_index"],
                    "e_indices": e_idx.tolist(),
                    "i_indices": i_idx.tolist(),
                    "e_rate_hz": float(e_full.sum() / (e_full.shape[1] * duration)),
                    "i_rate_hz": float(i_full.sum() / (i_full.shape[1] * duration)),
                }
            )
        np.savez_compressed(run.export / "rasters.npz", **arrays)
        aggregate = summarize(rows, cfg["dt_sweep_ms"])
        e_means = [row["e_rate_hz"]["mean"] for row in aggregate]
        a_means = [row["acc"]["mean"] for row in aggregate]
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp044.analysis/v1",
                "git_sha_train": load_json(
                    bank.file(contract["cells"][0]["cell_name"], "config.json")
                ).get("git_sha"),
                "recipe": cfg,
                "measurement": MEASUREMENT,
                "config": {
                    "dataset": "mnist",
                    "dt_sweep_ms": cfg["dt_sweep_ms"],
                    "seeds": cfg["seeds"],
                    "max_samples": contract["common"]["max_samples"],
                    "epochs": contract["common"]["epochs"],
                    "t_ms": contract["common"]["t_ms"],
                    "evaluation_samples": cfg["evaluation_samples"],
                    "training_contract": contract,
                },
                "checkpoint_policy": cfg["checkpoint_policy"],
                "checkpoint_provenance": checkpoints,
                "results": rows,
                "aggregate": aggregate,
                "curves": curves,
                "rasters": raster_rows,
                "summary": {
                    "e_rate_min_hz": min(e_means),
                    "e_rate_max_hz": max(e_means),
                    "acc_min_pct": min(a_means),
                    "acc_max_pct": max(a_means),
                    "acc_span_pp": max(a_means) - min(a_means),
                },
            },
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp044 compute ID")
    parser.add_argument("--run-id", help="unused v3 reservation")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id)
    except PingstoreError as exc:
        parser.exit(1, f"exp044 analyse: {exc}\n")


if __name__ == "__main__":
    main()
