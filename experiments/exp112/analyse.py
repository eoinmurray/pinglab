"""Compare the four explicit exp112 compute runs without training or inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp112 import recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import SourceRun, source_run, stage_run


def _condition(source: SourceRun) -> dict:
    cfg = source.record.get("execution", {}).get("configuration")
    if not isinstance(cfg, dict) or cfg.get("schema") != recipe.SCHEMA:
        raise PingstoreError(f"{source.record['run_id']} has no exp112 recipe")
    case = cfg.get("condition")
    if case not in recipe.CASES or cfg != recipe.configuration(case):
        raise PingstoreError(
            f"{source.record['run_id']} recipe differs from committed exp112"
        )
    return case


def _load_sources(identities: list[str]) -> dict[str, SourceRun]:
    if len(identities) != len(recipe.CASES) or len(set(identities)) != len(identities):
        raise PingstoreError("analysis requires four distinct compute run IDs")
    result = {}
    dataset_identity = None
    paired_initialization = None
    for identity in identities:
        source = source_run(
            REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
        )
        case = _condition(source)
        if case["id"] in result:
            raise PingstoreError(f"duplicate exp112 condition: {case['id']}")
        observed = source.record.get("execution", {}).get("dataset_identity")
        if not isinstance(observed, dict):
            raise PingstoreError(f"{identity} lacks the executed dataset identity")
        if dataset_identity is None:
            dataset_identity = observed
        elif observed != dataset_identity:
            raise PingstoreError(
                "exp112 conditions did not use identical MNIST data and splits"
            )
        resolved = source.record.get("execution", {}).get(
            "resolved_training_config", {}
        )
        initialization = resolved.get("weight_initialization", {})
        shared_initialization = {
            role: initialization.get(role) for role in ("W_in", "W_out")
        }
        if any(value is None for value in shared_initialization.values()):
            raise PingstoreError(f"{identity} lacks input/readout initialization")
        if paired_initialization is None:
            paired_initialization = shared_initialization
        elif shared_initialization != paired_initialization:
            raise PingstoreError(
                "exp112 conditions did not share input/readout initialization"
            )
        result[case["id"]] = source
    expected = {case["id"] for case in recipe.CASES}
    if set(result) != expected:
        raise PingstoreError(
            f"analysis condition set differs: expected {sorted(expected)}"
        )
    return result


def analyse(identities: list[str], *, run_id: str | None = None) -> str:
    sources = _load_sources(identities)
    rows = []
    histories = {}
    raster_arrays = {}
    raster_metadata = {}
    for case in recipe.CASES:
        case_id = case["id"]
        source = sources[case_id]
        training = load_json(source.export / "training_metrics.json")
        testing = load_json(source.export / "test_metrics.json")
        history = training.get("epochs", [])
        if (
            len(history) != recipe.EPOCHS
            or testing.get("n_total") != recipe.MNIST_TEST_SAMPLES
        ):
            raise PingstoreError(f"incomplete scientific output for {case_id}")
        histories[case_id] = [
            {
                key: epoch.get(key)
                for key in (
                    "ep",
                    "loss",
                    "test_loss",
                    "acc",
                    "grad_norm",
                    "grad_norm_max",
                    "skipped_steps",
                    "nan_forward_batches",
                    "rate_e",
                    "rate_i",
                    "test_rate_e",
                    "test_rate_i",
                )
            }
            for epoch in history
        ]
        with np.load(source.export / "spikes.npz", allow_pickle=False) as recording:
            e = recording["spk_e"]
            i = recording["spk_i"]
            if e.ndim == 3:
                e = e[:, 0, :]
            if i.ndim == 3:
                i = i[:, 0, :]
            if (
                e.shape
                != (round(recipe.PRESENTATION_MS / recipe.DT_MS), recipe.N_EXCITATORY)
                or i.shape
                != (round(recipe.PRESENTATION_MS / recipe.DT_MS), recipe.N_INHIBITORY)
                or int(recording["n_e"]) != recipe.N_EXCITATORY
                or int(recording["n_i"]) != recipe.N_INHIBITORY
                or not np.isclose(float(recording["dt"]), recipe.DT_MS)
                or int(recording["label"]) != recipe.RASTER_DIGIT
            ):
                raise PingstoreError(f"illustrative raster differs for {case_id}")
            e_t, e_cell = np.nonzero(e)
            i_t, i_cell = np.nonzero(i)
            raster_arrays[f"{case_id}__e_t"] = e_t.astype(np.int32)
            raster_arrays[f"{case_id}__e_cell"] = e_cell.astype(np.int32)
            raster_arrays[f"{case_id}__i_t"] = i_t.astype(np.int32)
            raster_arrays[f"{case_id}__i_cell"] = i_cell.astype(np.int32)
            raster_metadata[case_id] = {
                "digit": int(recording["label"]),
                "dt_ms": float(recording["dt"]),
                "duration_ms": recipe.PRESENTATION_MS,
                "n_e": int(recording["n_e"]),
                "n_i": int(recording["n_i"]),
                "e_spikes": int(len(e_t)),
                "i_spikes": int(len(i_t)),
            }
        rows.append(
            {
                **case,
                "compute_run": source.reference,
                "selected_epoch": training.get("best_epoch"),
                "best_validation_accuracy_pct": training.get("best_acc"),
                "best_validation_loss": training.get("best_validation_loss"),
                "final_validation_accuracy_pct": history[-1].get("acc"),
                "official_test_correct": testing.get("n_correct"),
                "official_test_samples": testing.get("n_total"),
                "official_test_accuracy_pct": 100.0
                * testing["n_correct"]
                / testing["n_total"],
                "official_test_cross_entropy": testing.get("ce_loss"),
                "training_elapsed_seconds": training.get("total_elapsed_s"),
                "total_skipped_steps": sum(
                    int(epoch.get("skipped_steps") or 0) for epoch in history
                ),
                "total_nan_forward_batches": sum(
                    int(epoch.get("nan_forward_batches") or 0) for epoch in history
                ),
            }
        )

    ordered_sources = {case["id"]: sources[case["id"]] for case in recipe.CASES}
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs=ordered_sources,
        run_id=run_id,
        configuration={
            "schema": "exp112.analysis/v1",
            "comparison": "complete 2x2 architecture by voltage-gradient-damping design",
            "pairing_required": True,
        },
    ) as run:
        np.savez_compressed(run.export / "rasters.npz", **raster_arrays)
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp112.results/v1",
                "design": recipe.configuration(recipe.CASES[0])["design"],
                "dataset": recipe.configuration(recipe.CASES[0])["dataset"],
                "dataset_identity": next(iter(sources.values())).record["execution"][
                    "dataset_identity"
                ],
                "conditions": rows,
                "histories": histories,
                "rasters": raster_metadata,
            },
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="repeat once for each compute run",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id)
    except (OSError, KeyError, TypeError, ValueError, PingstoreError) as exc:
        parser.exit(1, f"exp112 analyse: {exc}\n")


if __name__ == "__main__":
    main()
