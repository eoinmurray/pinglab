"""Analyse explicit exp049 recordings and pinned TR-05 histories."""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp049 import evidence, inputs, measurements, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic

MEASUREMENT = {
    "schema": "exp049.measurement/v1",
    "endpoint": "final-epoch checkpoint; official MNIST test accuracy and exact hid/inh rate keys",
    "psd": "per-trial full-length Welch density; demean; detrend=False; skip constant trials; average PSD; maximum raw bin in inclusive 5-150 Hz band",
    "weights": "native-dtype means; <=0 zero fraction; positive-only mean; pool seeds before 49-bin histograms",
    "cards": "raw training rate_e/rate_i; per-seed trajectories and nanmean; mean endpoint PSD and scalars",
    "trajectories": "prefer test_rate_e/test_rate_i keys even if null; exclude incomplete curves; mean and min/max then five-epoch edge-padded moving average",
    "phase": "unsmoothed seed means; segment contrast averages adjacent endpoints",
    "rhythmicity": "frozen final mean; trainable first available contrast mean and final minimum/maximum",
    "raster": "seed 42, image index 0; RNG(42) samples up to 200 E then 50 I cells without replacement",
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
        for job in recipe.jobs(cfg):
            train = contract["configs"][job["cell_name"]]
            evidence.recordings(source.unit(job["path"]), train, job)
        summary, curves, cards, weights, rasters, attractor = [], {}, {}, {}, [], {}
        for cond in recipe.COND_ORDER:
            endpoints, matrices, metrics = [], [], []
            for seed in recipe.SEEDS:
                name = recipe.cell_name(cond, seed)
                train = contract["configs"][name]
                directory = source.unit("infer", name)
                with np.load(
                    source.file("weights_dump", name, "weights_dump.npz"),
                    allow_pickle=False,
                ) as raw:
                    arrays = tuple(raw[k] for k in recipe.WEIGHT_ARRAYS)
                matrices.append(arrays)
                with np.load(directory / "pop_traces.npz", allow_pickle=False) as raw:
                    result = measurements.endpoint(
                        train,
                        load_json(directory / "metrics.json"),
                        raw["pop_e"],
                        arrays,
                    )
                endpoints.append(result)
                summary.append(
                    {
                        "condition": cond,
                        "seed": seed,
                        **{
                            k: v
                            for k, v in result.items()
                            if k not in ("psd", "freqs_hz")
                        },
                    }
                )
                metrics.append(histories[name])
                curves[name] = measurements.epoch_curve(histories[name]["epochs"], cond)
            cards[cond] = measurements.card(metrics, endpoints)
            weights[cond] = {
                "seeds": recipe.SEEDS,
                "weights": measurements.weight_distributions(matrices),
            }
            attractor[cond] = {
                "e": [r["e_rate_hz"] for r in endpoints],
                "i": [r["i_rate_hz"] for r in endpoints],
                "acc": float(np.nanmean([r["acc"] for r in endpoints])),
            }
            name = recipe.cell_name(cond, cfg["snapshot_seed"])
            filename = f"raster-{cond}.npz"
            np.savez_compressed(
                run.export / filename,
                **measurements.raster(
                    source.unit("snapshot", name), contract["configs"][name]
                ),
            )
            rasters.append({"condition": cond, "file": filename})
        write_json_atomic(
            run.export / "results.json",
            measurements.clean(
                {
                    "schema": "exp049.analysis/v1",
                    "recipe": cfg,
                    "measurement": MEASUREMENT,
                    "checkpoint_policy": recipe.CHECKPOINT_POLICY,
                    "checkpoint_provenance": contract["checkpoints"],
                    "config": {
                        "epochs": recipe.EPOCHS,
                        "max_samples": recipe.MAX_SAMPLES,
                        "evaluation_samples": cfg["evaluation_samples"],
                        "seeds": recipe.SEEDS,
                        "conditions": recipe.CONDITIONS,
                        "common_recipe": recipe.COMMON_RECIPE,
                    },
                    "summary": summary,
                    "rhythmicity": measurements.rhythmicity(curves),
                    "epoch_curves": curves,
                    "plot_data": {
                        "cards": cards,
                        "weights": weights,
                        "attractor": attractor,
                        "trajectories": measurements.trajectories(curves),
                        "last_epoch": max(max(c["ep"]) for c in curves.values()),
                    },
                    "rasters": rasters,
                }
            ),
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp049 analyse: {exc}\n")


if __name__ == "__main__":
    main()
