"""Measure retained exp025 evidence and prepare plot data; never simulate."""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp025 import evidence, inputs, measurements, recipe
from pingstore.contracts import PingstoreError, write_json_atomic

MEASUREMENT = {
    "schema": "exp025.measurement/v1",
    "frontier": "final-epoch official-test accuracy and mean E rate; three independent seeds, mean and SEM",
    "pfg": "representative seed 42; Welch E-population frequency; per-trial frequency anchors inhibitory peak cycles; active cell-cycle fraction",
    "penalty": "0.041 times mean squared positive overshoot of per-sample E rate above 1 Hz",
    "low_w_in": "retained validation histories; original fallback rules and seed aggregation",
    "crossing": "midpoint of first adjacent scale pair crossing I rate 0.05 Hz",
}


def low_curve(histories):
    def series(key, fallback=None):
        return np.asarray(
            [
                [
                    float(
                        e.get(key)
                        if e.get(key) is not None
                        else (e.get(fallback) or 0.0)
                        if fallback
                        else 0.0
                    )
                    for e in history
                ]
                for history in histories
            ],
            dtype=float,
        )

    result = {"epochs": list(range(1, len(histories[0]) + 1))}
    for name, key, fallback in (
        ("acc", "acc", None),
        ("rate_e", "test_rate_e", "rate_e"),
        ("rate_i", "test_rate_i", "rate_i"),
    ):
        values = series(key, fallback)
        result[name + "_mean"] = values.mean(axis=0).tolist()
        result[name + "_sem"] = (
            values.std(axis=0, ddof=1) / np.sqrt(len(histories))
            if len(histories) > 1
            else np.zeros(values.shape[1])
        ).tolist()
    result["rate_max"] = float(
        max(
            series("test_rate_e", "rate_e").max(), series("test_rate_i", "rate_i").max()
        )
    )
    return result


def analyse(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "compute")
    cfg, bank, contract = inputs.compute_evidence(REPO, source)
    history = evidence.histories(bank.export, contract)
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": source, "bank": bank},
        run_id=run_id,
        configuration=MEASUREMENT,
    ) as run:
        frontier = []
        pfg = []
        scales = []
        by_name = {c["cell_name"]: c for c in contract["cells"]}
        for job in recipe.jobs(cfg):
            name = job["cell_name"]
            cell = by_name[name]
            train = contract["configs"][name]
            directory = source.unit(job["path"])
            evidence.recordings(directory, train, job)
            if job["kind"] == "snapshot":
                with np.load(directory / "snapshot.npz", allow_pickle=False) as raw:
                    et, ec = np.where(raw["spk_e"])
                    it, ic = np.where(raw["spk_i"])
                    np.savez_compressed(
                        run.export / f"raster__{job['model']}.npz",
                        dt=raw["dt"],
                        T=raw["spk_e"].shape[0],
                        n_e=raw["spk_e"].shape[1],
                        n_i=raw["spk_i"].shape[1],
                        e_t=et,
                        e_cell=ec,
                        i_t=it,
                        i_cell=ic,
                    )
            elif job["kind"] == "frontier":
                m = evidence.metric(directory / "metrics.json", train, job)
                frontier.append(
                    {
                        "cell_name": name,
                        "model": cell["model"],
                        "rate_target_hz": cell["rate_target_hz"],
                        "rate_target_display": recipe.rate_target_display(
                            cell["rate_target_hz"]
                        ),
                        "seed": cell["seed"],
                        "best_acc": float(history[name]["best_acc"]),
                        "best_epoch": int(history[name]["best_epoch"]),
                        "final_acc": float(m["best_acc"]),
                        "rate_e": float(m["rates_hz"]["hid"]),
                        "evaluation_partition": "official_mnist_test",
                        "evaluation_samples": job["samples"],
                        "checkpoint_role": recipe.CHECKPOINT_ROLE,
                    }
                )
            elif job["kind"] == "pfg":
                m = measurements.measure_p_fgamma(
                    directory, train["dt"], job["is_ping"]
                )
                pfg.append(
                    {
                        "model": cell["model"],
                        "rate_target_hz": cell["rate_target_hz"],
                        "seed": 42,
                        "selection": "representative_seed",
                        **m,
                        "p_times_f_gamma": m["p"] * m["f_gamma"]
                        if m["p"] is not None and m["f_gamma"] is not None
                        else None,
                    }
                )
            else:
                acc, loss, penalty, er, ir = measurements.scaled_metrics(
                    directory, 1.0, recipe.FR_STRENGTH_UPPER
                )
                scales.append(
                    {
                        "cell": job["label"],
                        "scale": job["scale"],
                        "loss": loss,
                        "penalty": penalty,
                        "total_loss": loss + penalty,
                        "acc": acc,
                        "rate_e": er,
                        "rate_i": ir,
                    }
                )
        low = []
        low_curves = {}
        baseline_curves = {}
        for w in recipe.LOW_W_IN_VALUES:
            per_seed = []
            hist = []
            for s in cfg["low_w_in_seeds"]:
                m = history[recipe.low_w_in_cell_name(w, s)]
                last = m["epochs"][-1]
                hist.append(m["epochs"])
                per_seed.append(
                    {
                        "seed": s,
                        "best_acc": float(m["best_acc"]),
                        "best_epoch": int(m["best_epoch"]),
                        "final_acc": float(last["acc"]),
                        "rate_e": float(
                            last.get("test_rate_e") or last.get("rate_e") or 0.0
                        ),
                        "rate_i": float(
                            last.get("test_rate_i") or last.get("rate_i") or 0.0
                        ),
                    }
                )
            low.append(measurements.aggregate_low_w_in_seed_rows(w, per_seed))
            low_curves[f"{w:g}"] = low_curve(hist)
        for model in recipe.MODELS:
            histories = [
                history[recipe.cell_name(model, None, s)]["epochs"]
                for s in recipe.SEEDS
            ]
            baseline_curves[model] = {
                "epochs": [e["ep"] for e in histories[0]],
                "acc_mean": np.asarray([[e["acc"] for e in h] for h in histories])
                .mean(0)
                .tolist(),
            }
        ping = sorted(
            [r for r in scales if r["cell"] == "ping@rt1hz"], key=lambda r: r["scale"]
        )
        crossing = next(
            (
                0.5 * (a["scale"] + b["scale"])
                for a, b in zip(ping, ping[1:])
                if a["rate_i"] < 0.05 <= b["rate_i"]
            ),
            None,
        )
        training_sources = {}
        for group, registry in (
            ("shared_tr02", "TR-02"),
            ("low_w_in_controls", "TR-07"),
        ):
            names = [
                c["cell_name"]
                for c in contract["cells"]
                if c["group"] == group
                and (group == "shared_tr02" or c["seed"] in cfg["low_w_in_seeds"])
            ]
            training_sources[group] = {
                "owner": f"exp022/{registry}",
                "max_samples": 7000,
                "epochs": 50,
                "seeds": recipe.SEEDS
                if group == "shared_tr02"
                else cfg["low_w_in_seeds"],
                "checkpoint_role": recipe.CHECKPOINT_ROLE,
                "checkpoints": [
                    c for c in contract["checkpoints"] if c["training_cell"] in names
                ],
            }
        result = {
            "schema": "exp025.analysis/v1",
            "recipe": cfg,
            "measurement": MEASUREMENT,
            "checkpoint_policy": recipe.CHECKPOINT_POLICY,
            "checkpoint_provenance": training_sources["shared_tr02"]["checkpoints"],
            "training_sources": training_sources,
            "git_sha_train": contract["configs"][
                recipe.cell_name(recipe.MODELS[0], None, 42)
            ].get("git_sha"),
            "config": {
                "dataset": "mnist",
                "models": recipe.MODELS,
                "rate_target_grid_hz": [
                    t for t in recipe.RATE_TARGET_GRID_HZ if t is not None
                ],
                "max_samples": 7000,
                "evaluation_samples": cfg["evaluation_samples"],
                "epochs": 50,
                "t_ms": 200.0,
                "dt": 0.1,
                "frontier_seeds": recipe.SEEDS,
                "representative_seed": 42,
                "fr_strength_upper": recipe.FR_STRENGTH_UPPER,
            },
            "results": frontier,
            "frontier_statistics": measurements.aggregate_frontier(frontier),
            "rate_target_p_fgamma": pfg,
            "low_w_in_sweep": low,
            "w_in_scale_sweep": scales,
            "plot_data": {
                "low_w_in": low_curves,
                "baseline": baseline_curves,
                "scale_crossing": crossing,
            },
        }
        write_json_atomic(run.export / "results.json", result)
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    a = p.parse_args()
    try:
        analyse(a.source, run_id=a.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        p.exit(1, f"exp025 analyse: {exc}\n")


if __name__ == "__main__":
    main()
