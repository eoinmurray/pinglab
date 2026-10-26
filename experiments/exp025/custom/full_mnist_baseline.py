"""Measure full-MNIST histories and recorded rasters; retain the frontier exactly.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

import copy

import numpy as np
from experiments.exp025 import recipe
from experiments.exp025.custom import inputs
from experiments.exp025.custom.source_exp022_custom_checkpoints import (
    public_provenance,
    resolve_checkpoint,
)
from experiments.exp025.custom.source_exp044_custom_evidence import snapshot
from pingstore.contracts import PingstoreError, load_json, write_json_atomic

MEASUREMENT = {
    "schema": "exp025.full-mnist-baseline/v1",
    "baseline": "full official training pool; 54000 optimization and 6000 validation images; mean validation accuracy across seeds 42-44 at each epoch",
    "rasters": "retained final-epoch canonical seed-42 digit-0/sample-0 recordings; 200 ms",
    "frontier": "unchanged measurements from the explicitly pinned endpoint analysis",
}


def canonical_evidence(bank):
    curves, recordings, checkpoints = {}, {}, []
    for model in recipe.MODELS:
        histories = []
        for seed in recipe.SEEDS:
            name = f"{model}__canonical__seed{seed}"
            unit = bank.unit(name)
            cfg = load_json(unit / "config.json")
            expected = {
                "dataset": "mnist",
                "max_samples": 60000,
                "epochs": 50,
                "seed": seed,
                "dt": 0.1,
                "t_ms": 200.0,
                "ei_strength": float(model == "ping"),
                "v_grad_dampen": 1000.0,
                "fr_reg_upper_strength": 0.0,
            }
            if any(cfg.get(k) != v for k, v in expected.items()):
                raise PingstoreError(f"{name}: full-MNIST training contract differs")
            split = cfg.get("dataset_split", {})
            if any(
                split.get(k) != v
                for k, v in {
                    "optimizer_train_samples": 54000,
                    "validation_samples": 6000,
                    "split_seed": 42,
                    "source_train_partition": "official_mnist_train",
                    "source_test_partition": "official_mnist_test",
                    "official_test_used_during_training": False,
                    "checkpoint_selection_partition": "validation",
                }.items()
            ):
                raise PingstoreError(f"{name}: full-MNIST data split differs")
            rows = load_json(unit / "metrics.json").get("epochs", [])
            if len(rows) != 50 or [r.get("ep") for r in rows] != list(range(1, 51)):
                raise PingstoreError(f"{name}: incomplete training history")
            acc = np.asarray([r["acc"] for r in rows], dtype=float)
            if not np.isfinite(acc).all() or np.any((acc < 0) | (acc > 100)):
                raise PingstoreError(f"{name}: invalid validation accuracy")
            if any(r.get("samples") != 54000 for r in rows):
                raise PingstoreError(f"{name}: optimizer sample counts differ")
            checkpoint = public_provenance(resolve_checkpoint(unit, "final_epoch"))
            if checkpoint["epoch"] != 50 or checkpoint["training_cell"] != name:
                raise PingstoreError(f"{name}: final checkpoint differs")
            checkpoints.append(checkpoint)
            histories.append(acc)
            if seed == 42:
                diagnostic = (
                    bank.record.get("bank_reuse", {})
                    .get("diagnostic_provenance", {})
                    .get(name, {})
                )
                if diagnostic.get("checkpoint") != checkpoint:
                    raise PingstoreError(f"{name}: diagnostic checkpoint differs")
                recorded_cfg = diagnostic.get("execution_records", {}).get(
                    "config.json", {}
                )
                if any(
                    recorded_cfg.get(k) != v
                    for k, v in {
                        "digit": 0,
                        "sample": 0,
                        "t_ms": 200.0,
                        "dt": 0.1,
                        "seed": 42,
                        "infer": True,
                    }.items()
                ):
                    raise PingstoreError(f"{name}: diagnostic selection differs")
                path = bank.file(f"snapshots/{name}", "recording.npz")
                recordings[model] = snapshot(path, cfg["dt"], cfg)
        curves[model] = {
            "epochs": list(range(1, 51)),
            "acc_mean": np.mean(histories, axis=0).tolist(),
        }
    return {
        "training_pool_samples": 60000,
        "optimizer_train_samples": 54000,
        "validation_samples": 6000,
        "seeds": list(recipe.SEEDS),
        "snapshot_duration_ms": 200.0,
        "checkpoint_role": "final_epoch",
        "checkpoint_provenance": checkpoints,
        "curves": curves,
    }, recordings


def analyse(repo, analysis_id, bank_id, *, run_id=None):
    previous = inputs.source(repo, analysis_id, "analyse")
    retained = inputs.analysis_evidence(repo, previous)
    bank = inputs.source(repo, bank_id, "compute", experiment="exp022")
    baseline, recordings = canonical_evidence(bank)
    result = copy.deepcopy(retained)
    result["schema"] = "exp025.analysis/v3"
    result["baseline"] = baseline
    result["plot_data"]["baseline"] = baseline["curves"]
    with inputs.execution(
        repo,
        "analyse",
        sources={
            "frontier_analysis": previous,
            "baseline_bank": bank,
        },
        run_id=run_id,
        configuration=MEASUREMENT,
    ) as run:
        for model, raw in recordings.items():
            et, ec = np.where(raw["spk_e"])
            it, ic = np.where(raw["spk_i"])
            np.savez_compressed(
                run.export / f"raster__{model}.npz",
                dt=0.1,
                T=raw["spk_e"].shape[0],
                n_e=raw["spk_e"].shape[1],
                n_i=raw["spk_i"].shape[1],
                e_t=et,
                e_cell=ec,
                i_t=it,
                i_cell=ic,
            )
        write_json_atomic(run.export / "results.json", result)
    return run.run_id
