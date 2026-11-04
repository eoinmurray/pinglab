"""Validate the frozen control and measure the three retained raster conditions."""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
import torch
from experiments.exp169 import recipe
from pingstore.library import file_sha256, source_run, stage_run
from scipy.signal import lfilter
from snnlab.sim import NetworkSnapshot, load_dataset


def metrics(g, target):
    corr = np.corrcoef(g.T)
    return dict(
        mean=float(g.mean()),
        std=float(g.std()),
        per_cell_mean_relative_error_max=float(
            np.max(np.abs(g.mean(0) / target.mean(0) - 1))
        ),
        per_cell_std_relative_error_max=float(
            np.max(np.abs(g.std(0) / target.std(0) - 1))
        ),
        pairwise_correlation_mean=float(corr[np.triu_indices(g.shape[1], 1)].mean()),
        population_mean_temporal_std=float(g.mean(1).std()),
    )


def read_evidence(computed):
    pin = computed.record["inputs"]["bank"]
    bank = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        stage="compute",
        experiment="exp022",
        reference=pin,
    )
    cfg = computed.record["execution"]["configuration"]
    rows = []
    inputs = []
    for name, cell in [("COBA", cfg["coba_cell"]), ("PING", cfg["ping_cell"])]:
        with np.load(bank.file("snapshots", cell, "recording.npz")) as z:
            rows.append(dict(name=name, e=z["spk_e"].copy(), i=z["spk_i"].copy()))
            inputs.append(z["input_spikes"].copy())
            if name == "PING":
                target = z["gi_e_1"].astype(float)
    if not np.array_equal(*inputs):
        raise ValueError("Sensory inputs differ")
    with np.load(computed.file("artificial.npz")) as z:
        e = z["E_spikes"][:, 0, :].copy()
        conductance = z["g_artificial"][:, 0, :].astype(float)
    with np.load(computed.file("calibration.npz")) as z:
        events, amplitudes = z["events"].copy(), z["amplitudes_us"].copy()
    rows.append(dict(name="COBA + asynchronous GABA", e=e, i=events))
    return cfg, bank, rows, target, conductance, amplitudes


def validate(computed):
    cfg, bank, rows, target, conductance, amplitudes = read_evidence(computed)
    snapshots = {
        name: NetworkSnapshot.read(computed.unit(name, "start"), parameters=True)
        for name in ("baseline", "artificial")
    }
    original = torch.load(
        bank.file(cfg["coba_cell"], cfg["checkpoint"]["filename"]),
        map_location="cpu",
        weights_only=True,
    )
    if (
        file_sha256(bank.file(cfg["coba_cell"], cfg["checkpoint"]["filename"]))
        != cfg["checkpoint"]["sha256"]
    ):
        raise ValueError("Checkpoint digest mismatch")
    mapping = {
        "input_to_E.weight": "W_ff.0",
        "E_to_I.weight": "W_ei.1",
        "I_to_E.weight": "W_ie.1",
        "readout_projection.weight": "W_ff.1",
    }
    for snapshot in snapshots.values():
        parameters = snapshot.load_parameters()
        for name, key in mapping.items():
            np.testing.assert_array_equal(
                np.asarray(parameters[name]), original[key].clamp(min=0).numpy()
            )
        np.testing.assert_array_equal(
            np.asarray(parameters["artificial_to_E.weight"]), np.diag(amplitudes)
        )
    events = rows[-1]["i"]
    for row in rows:
        for spikes in (row["e"], row["i"]):
            if not np.isin(spikes, [0, 1]).all():
                raise ValueError("Nonbinary spikes")
        if row["e"].shape != (2000, 1024):
            raise ValueError("Wrong E dimensions")
    delayed = np.vstack([np.zeros((1, 1024)), events[:-1]])
    predicted = (
        lfilter(
            [1.0], [1.0, -np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])], delayed, axis=0
        )
        * amplitudes
    )
    filter_error = float(np.max(np.abs(predicted - conductance)))
    if filter_error > 1e-5:
        raise ValueError("Conductance differs from delayed GABA filter")
    matched = metrics(conductance, target)
    if (
        matched["per_cell_mean_relative_error_max"] > cfg["mean_relative_tolerance"]
        or matched["per_cell_std_relative_error_max"] > cfg["std_relative_tolerance"]
    ):
        raise ValueError("Conductance moments do not match")
    if abs(matched["pairwise_correlation_mean"]) >= 0.1:
        raise ValueError("Excess shared fluctuation")
    with np.load(computed.file("baseline.npz")) as z:
        baseline = z["E_spikes"][:, 0, :]
    np.testing.assert_array_equal(baseline.sum(0), rows[0]["e"].sum(0))
    saved = json.loads(computed.file("summary.json").read_text())
    seconds = cfg["duration_ms"] / 1000
    reference_config = json.loads(
        bank.file(cfg["coba_cell"], "config.json").read_text()
    )
    return dict(
        schema="exp169.analysis/v1",
        configuration=cfg,
        reference_input_rate_hz=reference_config["input_rate"],
        conditions=[
            dict(
                name=row["name"],
                rate_e_hz=float(row["e"].sum() / 1024 / seconds),
                rate_i_or_input_hz=float(row["i"].sum() / row["i"].shape[1] / seconds),
                inhibitory_channels=row["i"].shape[1],
            )
            for row in rows
        ],
        matched=matched,
        ping=metrics(target, target),
        checks=dict(
            frozen_checkpoint_parameters_exact=True,
            private_diagonal_parameters_exact=True,
            native_baseline_counts_exact=True,
            sensory_input_identical=True,
            gaba_filter_max_error_us=filter_error,
            legacy_baseline_mismatched_spike_bins=int(
                np.count_nonzero(baseline != rows[0]["e"])
            ),
        ),
        original_simulation_seconds=saved["simulation_seconds"],
        limitations=saved["limitations"],
    )


def classification_summary(rows, selection):
    """Recompute decisions and paired accuracy from retained scores."""
    conditions = ["COBA", "PING", "COBA + artificial inhibition"]
    if not selection or len({item["index"] for item in selection}) != len(selection):
        raise ValueError("Duplicate or missing selected images")
    matrices = {name: np.zeros((10, 10), dtype=int) for name in conditions}
    decisions = {}
    for row in rows:
        index, label = int(row["index"]), int(row["label"])
        condition = row["condition"]
        scores = np.array([float(row[f"score_{i}"]) for i in range(10)])
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite classifier scores")
        prediction = int(scores.argmax())
        tied = int(np.count_nonzero(scores == scores.max()) > 1)
        if (
            prediction != int(row["prediction"])
            or int(row["correct"]) != int(prediction == label)
            or tied != int(row["top_score_tied"])
        ):
            raise ValueError("Saved decision disagrees with scores")
        key = (index, condition)
        if key in decisions or condition not in matrices:
            raise ValueError("Duplicate or unknown image-condition row")
        decisions[key] = row
        matrices[condition][label, prediction] += 1
    if len(rows) != 3 * len(selection):
        raise ValueError("Incomplete paired classification")
    for item in selection:
        group = [decisions[(item["index"], name)] for name in conditions]
        if any(int(row["label"]) != item["label"] for row in group):
            raise ValueError("Image labels disagree")
        if len({row["input_sha256"] for row in group}) != 1:
            raise ValueError("Sensory encodings differ")
    baseline = [
        int(decisions[(s["index"], conditions[0])]["correct"]) for s in selection
    ]
    artificial = [
        int(decisions[(s["index"], conditions[2])]["correct"]) for s in selection
    ]
    return dict(
        sample_count=len(selection),
        conditions=[
            dict(
                name=name,
                correct=int(matrix.trace()),
                total=len(selection),
                accuracy=float(matrix.trace() / len(selection)),
                per_digit_correct=matrix.diagonal().tolist(),
                confusion_matrix=matrix.tolist(),
            )
            for name, matrix in matrices.items()
        ],
        artificial_fixes_vs_coba=sum(
            not a and c for a, c in zip(baseline, artificial, strict=True)
        ),
        artificial_breaks_vs_coba=sum(
            a and not c for a, c in zip(baseline, artificial, strict=True)
        ),
        tied_scores=sum(int(row["top_score_tied"]) for row in rows),
    )


def validate_classification(computed):
    """Check all paired predictions, calibration moments and native start weights."""
    cfg = computed.record["execution"]["configuration"]
    pin = computed.record["inputs"]["bank"]
    bank = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        stage="compute",
        experiment="exp022",
        reference=pin,
    )
    selection = json.loads(computed.file("selection.json").read_text())
    with computed.file("predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    results = classification_summary(rows, selection)
    if (
        np.bincount([s["label"] for s in selection], minlength=10).tolist()
        != [cfg["classification_images_per_digit"]] * 10
    ):
        raise ValueError("Classification sample is not balanced")
    _, _, _, test_labels = load_dataset(
        "mnist", split=True, evaluation_only=True, evaluation_split="test"
    )
    rng = np.random.default_rng(cfg["classification_selection_seed"])
    expected_indices = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(test_labels == digit),
                cfg["classification_images_per_digit"],
                replace=False,
            )
            for digit in range(10)
        ]
    )
    if expected_indices.tolist() != [s["index"] for s in selection]:
        raise ValueError("Selection does not match recorded sampling protocol")
    for item in selection:
        if (
            int(test_labels[item["index"]]) != item["label"]
            or item["encoding_seed"]
            != cfg["classification_encoding_seed"] + item["index"]
            or item["artificial_seed"] != cfg["artificial_seed"] + item["index"]
        ):
            raise ValueError("Selected label or seed differs from protocol")
    with np.load(computed.file("calibration-moments.npz")) as z:
        moments = z["moments"].copy()
    if (
        moments.shape != (len(selection), 6, cfg["n_e"])
        or not np.isfinite(moments).all()
        or not (moments > 0).all()
    ):
        raise ValueError("Invalid calibration moments")
    mean_errors = np.max(np.abs(moments[:, 2] / moments[:, 0] - 1), axis=1)
    sd_errors = np.max(np.abs(moments[:, 3] / moments[:, 1] - 1), axis=1)
    if (mean_errors >= cfg["mean_relative_tolerance"]).any() or (
        sd_errors >= cfg["std_relative_tolerance"]
    ).any():
        raise ValueError("Classification conductance moments do not match")
    checks = json.loads(computed.file("calibration.json").read_text())
    if len(checks) != len(selection) or not all(check["passed"] for check in checks):
        raise ValueError("Calibration checks incomplete or failed")
    np.testing.assert_allclose(
        [c["max_mean_relative_error"] for c in checks], mean_errors
    )
    np.testing.assert_allclose([c["max_sd_relative_error"] for c in checks], sd_errors)
    originals = {}
    for kind, checkpoint in cfg["checkpoint_pins"].items():
        path = bank.file(cfg[kind + "_cell"], checkpoint["filename"])
        if (
            checkpoint["role"] != cfg["checkpoint_role"]
            or file_sha256(path) != checkpoint["sha256"]
        ):
            raise ValueError("Classification checkpoint mismatch")
        originals[kind] = torch.load(path, map_location="cpu", weights_only=True)
    mapping = {
        "input_to_E.weight": "W_ff.0",
        "E_to_I.weight": "W_ei.1",
        "I_to_E.weight": "W_ie.1",
        "readout_projection.weight": "W_ff.1",
    }
    for number in range(len(selection)):
        for suffix in ("coba", "ping", "artificial"):
            snapshot = NetworkSnapshot.read(
                computed.unit("snapshots", f"{number:03d}-{suffix}", "start"),
                parameters=True,
            )
            weights = snapshot.load_parameters()
            original = originals["ping" if suffix == "ping" else "coba"]
            for name, key in mapping.items():
                np.testing.assert_array_equal(
                    np.asarray(weights[name]), original[key].clamp(min=0).numpy()
                )
            expected = (
                np.diag(moments[number, 4].astype(np.float32))
                if suffix == "artificial"
                else np.zeros((cfg["n_e"], cfg["n_e"]))
            )
            np.testing.assert_array_equal(
                np.asarray(weights["artificial_to_E.weight"]), expected
            )
    if not all(
        r["per_neuron_counts_exact"] and r["spike_bin_agreement"] > 0.9999
        for r in cfg["replay"].values()
    ):
        raise ValueError("Native classification replay failed")
    results.update(
        configuration=cfg,
        checks=dict(
            frozen_checkpoint_parameters_exact=True,
            paired_sensory_input_identical=True,
            calibration_passes=len(checks),
            snapshots_checked=3 * len(selection),
            max_mean_relative_error=float(mean_errors.max()),
            max_sd_relative_error=float(sd_errors.max()),
        ),
        timing={
            key: computed.record["execution"][key]
            for key in (
                "elapsed_seconds",
                "simulation_seconds",
                "calibration_seconds",
                "inference_seconds",
            )
        },
        limitations="One seed and one encoder draw per image; no burn-in or retraining. Artificial inhibition uses same-image PING conductance targets, not a fixed standalone generator. This comparison does not isolate synchrony.",
    )
    bank.check_unchanged()
    return results


def analyse(identity, *, classification_source=None, run_id=None):
    started = perf_counter()
    computed = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="compute"
    )
    results = validate(computed)
    inputs = {"compute": computed}
    if classification_source:
        classified = source_run(
            ROOT / ".pingstore",
            classification_source,
            experiment=recipe.SLUG,
            stage="compute",
            phase="classification",
        )
        if classified.record["inputs"]["bank"] != computed.record["inputs"]["bank"]:
            raise ValueError("Raster and classification bank pins differ")
        results["classification"] = validate_classification(classified)
        inputs["classification"] = classified
    with stage_run(
        ROOT,
        recipe.SLUG,
        "analyse",
        inputs=inputs,
        configuration={
            "measurement": "full 200-ms per-cell moments and raw spike events"
        },
        run_id=run_id,
    ) as run:
        (run.export / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        run.record["execution"].update(
            no_simulation=True, elapsed_seconds=perf_counter() - started
        )
        for source in inputs.values():
            source.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--classification-source",
        help="Explicit classification compute run to combine with the raster evidence",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    analyse(
        args.source,
        classification_source=args.classification_source,
        run_id=args.run_id,
    )
