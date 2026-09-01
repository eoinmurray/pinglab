"""Measure retained exp042 responses and prepare compact plot inputs; never simulate."""

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp042 import inputs, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic

MEASUREMENT = {
    "schema": "exp042.measurement/v2",
    "aggregation": "mean_across_training_seeds",
    "uncertainty": "sample_standard_deviation_divided_by_sqrt_n",
    "rate_population": "last_hidden_and_inhibitory_layers",
    "raster_rate_population": "all_cells_before_display_subsampling",
    "override_invariant": "exact_per_trial_per_cell_spike_count",
}


def finite(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise PingstoreError(f"missing or nonfinite {label}")
    return float(value)


def measurement(metrics, job, cfg):
    rates = metrics.get("rates_hz", {})
    hid = max((k for k in rates if k.startswith("hid")), default=None)
    inh = max((k for k in rates if k.startswith("inh")), default=None)
    if (
        hid is None
        or inh is None
        or metrics.get("n_total") != cfg["evaluation_samples"]
    ):
        raise PingstoreError(
            "missing population rates or wrong evaluation sample count"
        )
    result = {
        "condition": job["condition"],
        "seed": job["seed"],
        "acc": finite(metrics.get("best_acc"), "accuracy"),
        "e_rate_hz": finite(rates[hid], "E rate"),
        "i_rate_hz": finite(rates[inh], "I rate"),
        "n_total": metrics["n_total"],
    }
    transform = metrics.get("override_transform")
    if (
        not isinstance(transform, dict)
        or transform.get("schema") != "exp042.override/v2"
        or transform.get("boundary_policy") != cfg["jitter_policy"]["boundary"]
        or transform.get("collision_policy") != cfg["jitter_policy"]["collision"]
        or transform.get("per_trial_cell_count_invariant") is not True
        or transform.get("input_spikes") != transform.get("output_spikes")
        or transform.get("trials_checked") != cfg["evaluation_samples"]
        or not isinstance(transform.get("cells_checked_per_trial"), int)
        or transform["cells_checked_per_trial"] <= 0
    ):
        raise PingstoreError("missing or invalid override spike-count invariant")
    for key in (
        "input_spikes",
        "output_spikes",
        "boundary_reflected_spikes",
        "collision_resolved_spikes",
        "max_collision_resolution_steps",
    ):
        if (
            isinstance(transform.get(key), bool)
            or not isinstance(transform.get(key), int)
            or transform[key] < 0
        ):
            raise PingstoreError("invalid override transform diagnostics")
    result["override_transform"] = transform
    if (
        not 0 <= result["acc"] <= 100
        or min(result["e_rate_hz"], result["i_rate_hz"]) < 0
    ):
        raise PingstoreError("invalid accuracy or firing rate")
    if job["sigma_ms"] is not None:
        result["sigma_ms"] = job["sigma_ms"]
    return result


def summarize(rows):
    result = []
    for sigma in sorted({row["sigma_ms"] for row in rows}):
        group = [row for row in rows if row["sigma_ms"] == sigma]
        record = {"sigma_ms": sigma, "n": len(group)}
        for key in ("e_rate_hz", "i_rate_hz", "acc"):
            values = [row[key] for row in group]
            record[key] = {
                "mean": float(np.mean(values)),
                "sem": float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0,
            }
        result.append(record)
    return result


def raster_sample(path, training, cfg):
    with np.load(path, allow_pickle=False) as data:
        e, i = np.array(data["spk_e"]), np.array(data["spk_i"])
        label = int(data["label"])
    for value, cells in ((e, training["n_hidden"]), (i, training["n_inh"])):
        expected_steps = int(round(training["t_ms"] / training["dt"]))
        if (
            value.ndim not in (2, 3)
            or value.shape[0] != expected_steps
            or value.shape[-1] != cells
            or (value.ndim == 3 and value.shape[1] != 1)
            or not np.isfinite(value).all()
            or not np.isin(value, [0, 1]).all()
        ):
            raise PingstoreError("invalid retained single-trial spikes")
    if e.ndim == 3:
        e = e[:, 0, :]
    if i.ndim == 3:
        i = i[:, 0, :]
    raster = cfg["raster"]
    rng = np.random.default_rng(raster["selection_seed"])
    e_idx = np.sort(rng.choice(e.shape[1], raster["n_e_plot"], replace=False))
    i_idx = np.sort(rng.choice(i.shape[1], raster["n_i_plot"], replace=False))
    seconds = training["t_ms"] / 1000.0
    row = {
        "label": label,
        "dt": training["dt"],
        "t_ms": training["t_ms"],
        "seed": raster["seed"],
        "sample_index": raster["sample_index"],
        "sigma_ms": raster["sigma_ms"],
        "e_indices": e_idx.tolist(),
        "i_indices": i_idx.tolist(),
        "e_rate_hz": float(e.sum() / (e.shape[1] * seconds)),
        "i_rate_hz": float(i.sum() / (i.shape[1] * seconds)),
    }
    return row, e[:, e_idx].astype(bool), i[:, i_idx].astype(bool)


def analyse(identity, *, run_id=None):
    compute = inputs.source(REPO, identity, "compute")
    cfg = inputs.configuration(compute)
    ref = compute.record["inputs"]["bank"]
    bank = inputs.source(
        REPO, ref["run_id"], "compute", experiment="exp022", reference=ref
    )
    bank_evidence = inputs.bank_evidence(bank)
    retained = load_json(compute.export / "evidence.json")
    if retained != {
        "schema": "exp042.compute/v4",
        "recipe": cfg,
        "bank_evidence": bank_evidence,
        "jobs": recipe.jobs(cfg),
        "recordings": ["cycle.npz", "cell.npz"],
    }:
        raise PingstoreError("compute payload disagrees with bank or recipe")
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": compute, "bank": bank},
        run_id=run_id,
        configuration=MEASUREMENT,
    ) as run:
        groups = {key: [] for key in ("jitter_sweep", "cell_jitter_sweep")}
        for job in recipe.jobs(cfg):
            data = load_json(compute.export / "jobs" / (job["id"] + ".json"))
            if data.get("job") != job:
                raise PingstoreError("retained condition does not match recipe")
            groups[job["group"]].append(measurement(data["metrics"], job, cfg))
        training = bank_evidence["configurations"][
            recipe.cell_name(cfg["raster"]["seed"])
        ]
        rasters, arrays = {}, {}
        for name in ("cycle", "cell"):
            row, e, i = raster_sample(compute.export / f"{name}.npz", training, cfg)
            rasters[name] = row
            arrays[name + "__e"], arrays[name + "__i"] = e, i
        np.savez_compressed(run.export / "rasters.npz", **arrays)
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp042.analysis/v4",
                "recipe": cfg,
                "measurement": MEASUREMENT,
                "checkpoint_policy": cfg["checkpoint_policy"],
                "checkpoint_provenance": bank_evidence["checkpoints"],
                "config": {
                    "evaluation_samples_per_condition": cfg["evaluation_samples"],
                    "seeds": cfg["seeds"],
                    "jitter_sigmas_ms": cfg["jitter_sigmas_ms"],
                    "cell_jitter_sigmas_ms": cfg["cell_jitter_sigmas_ms"],
                    "f_gamma_reference_hz": cfg["f_gamma_reference_hz"],
                    "raster_sample_idx": cfg["raster"]["sample_index"],
                },
                **groups,
                "rasters": rasters,
                "aggregate": {
                    key: summarize(groups[key])
                    for key in ("jitter_sweep", "cell_jitter_sweep")
                },
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp042 compute ID")
    parser.add_argument("--run-id", help="unused v4 reservation")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, ValueError) as exc:
        parser.exit(1, f"exp042 analyse: {exc}\n")


if __name__ == "__main__":
    main()
