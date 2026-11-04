"""Validate the paired adaptation evidence without simulation."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
import torch
import torch.nn.functional as F
from experiments.exp171 import recipe
from experiments.exp171.compute import artificial, filtered, write_json
from pingstore.library import file_sha256, source_run, stage_run
from snnlab.sim import NetworkSnapshot, load_runtime_state


def summarise(scores, labels, rates):
    if not np.isfinite(scores).all() or not np.isfinite(rates).all():
        raise ValueError("Nonfinite validation evidence")
    losses = (
        F.cross_entropy(
            torch.from_numpy(scores.reshape(-1, scores.shape[-1])),
            torch.from_numpy(np.tile(labels, scores.shape[0])),
            reduction="none",
        )
        .numpy()
        .reshape(scores.shape[:2])
    )
    return dict(
        cross_entropy=float(losses.mean()),
        accuracy_pct=float(100 * (scores.argmax(-1) == labels).mean()),
        e_rate_hz=float(rates.mean()),
    ), losses


def validate_dataset(computed, cfg):
    with np.load(computed.file("dataset.npz")) as z:
        indices, labels = z["indices"], z["labels"]
        nt = cfg["train_per_digit"] * 10
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Repeated sample identities")
        for selected, partition, _count in [
            (indices[:nt], z["source_train"], cfg["train_per_digit"]),
            (indices[nt:], z["source_validation"], cfg["validation_per_digit"]),
        ]:
            if not np.isin(selected, partition).all():
                raise ValueError("Sample partition mismatch")
        if np.intersect1d(z["source_train"], z["source_validation"]).size:
            raise ValueError("Source partitions overlap")
        for yy, count in [
            (labels[:nt], cfg["train_per_digit"]),
            (labels[nt:], cfg["validation_per_digit"]),
        ]:
            np.testing.assert_array_equal(
                np.bincount(yy, minlength=10), np.full(10, count)
            )
    return indices, labels, nt


def validate_parameters(computed, bank, cfg, metrics):
    snapshots = {
        name: NetworkSnapshot.read(
            computed.unit(name, stage), parameters=True
        ).load_parameters()
        for name, stage in [
            ("initial", "start"),
            ("final-training", "end"),
            ("best-validation", "end"),
            ("teacher", "start"),
        ]
    }
    snapshots = {
        stage: {name: np.asarray(value) for name, value in parameters.items()}
        for stage, parameters in snapshots.items()
    }
    changes = {}
    for kind, snap in [("coba", "initial"), ("ping", "teacher")]:
        cp = cfg["source_checkpoints"][kind]
        path = bank.file(cfg[kind + "_cell"], cp["filename"])
        if file_sha256(path) != cp["sha256"]:
            raise ValueError("Checkpoint digest mismatch")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        for name, key in recipe.MAPPING.items():
            np.testing.assert_array_equal(
                snapshots[snap][name], raw[key].clamp(min=0).numpy()
            )
    for name, initial in snapshots["initial"].items():
        final = snapshots["final-training"][name]
        if not np.isfinite(final).all() or np.any(final < 0):
            raise ValueError("Invalid final parameter")
        changes[name] = float(np.max(np.abs(final - initial)))
        if name in recipe.TRAINABLE:
            if changes[name] == 0:
                raise ValueError("Trainable matrix unchanged")
        else:
            for stage in ("final-training", "best-validation"):
                np.testing.assert_array_equal(initial, snapshots[stage][name])
        np.testing.assert_allclose(changes[name], metrics["parameter_changes"][name])
    np.testing.assert_array_equal(
        snapshots["initial"]["artificial_to_E.weight"], np.eye(cfg["n_e"])
    )
    return changes


def validate_inhibition_history(computed, cfg, indices, metrics):
    """Authenticate a reproducible example of fresh-cell inhibitory initialization."""
    steps = cfg.get("inhibition_prehistory_steps", 0)
    if not steps:
        return
    if steps != round(cfg["inhibition_prehistory_ms"] / cfg["dt_ms"]):
        raise ValueError("Inhibitory prehistory length disagrees with configuration")
    if metrics["inhibition_replay"] != dict(
        original_index=int(indices[0]),
        domain="validation",
        draw=0,
        prehistory_steps=steps,
        scored_steps=cfg["steps"],
    ):
        raise ValueError("Inhibitory initialization replay metadata mismatch")
    with np.load(computed.file("calibration.npz")) as z:
        ev = artificial(
            z["probability"][:1], z["amplitude_us"][:1], indices[:1], cfg, draw=0
        )
    state = load_runtime_state(computed.unit("initial-inhibition-state"))
    if (
        state.completed_steps
        or state.reduction_state
        or state.reduction_tensors
        or state.segment_state
        or state.segment_counts
        or state.custom_state
    ):
        raise ValueError("Prehistory leaked into scored neuronal trajectory")
    for name, value in state.voltages.items():
        if not torch.all(value == (0 if name == "readout" else -65)):
            raise ValueError("Neuronal voltages did not start fresh")
    for group in (state.refractory, state.population_histories, state.currents):
        if any(torch.count_nonzero(v) for v in group.values()):
            raise ValueError("Neuronal or readout history was retained")
    for name, value in state.conductances.items():
        if name != "artificial_to_E" and torch.count_nonzero(value):
            raise ValueError("Non-private synaptic history was retained")
    np.testing.assert_allclose(
        state.conductances["artificial_to_E"].numpy(),
        filtered(ev[:steps], cfg)[-1],
        rtol=1e-6,
        atol=1e-7,
    )
    if set(state.input_histories) != {"artificial"}:
        raise ValueError("Unexpected queued input history")
    np.testing.assert_array_equal(
        state.input_histories["artificial"].numpy(), ev[steps - 1 : steps]
    )


def validate(computed):
    cfg = computed.record["execution"]["configuration"]
    pin = computed.record["inputs"]["bank"]
    if (
        set(computed.record["inputs"]) != {"bank"}
        or pin["run_id"] != "exp022-r015-compute"
        or pin["payload_digest"] != recipe.configuration()["source_digest"]
    ):
        raise ValueError("Scientific dependency boundary violated")
    bank = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        experiment="exp022",
        stage="compute",
        reference=pin,
    )
    indices, labels, nt = validate_dataset(computed, cfg)
    metrics = json.loads(computed.file("metrics.json").read_text())
    validate_inhibition_history(computed, cfg, indices, metrics)
    timings = json.loads(computed.file("timings.json").read_text())
    with np.load(computed.file("validation.npz")) as z:
        scores, rates, saved_losses = z["scores"], z["e_rates"], z["losses"]
        if scores.shape != (
            cfg["epochs"] + 1,
            cfg["validation_draws"],
            10 * cfg["validation_per_digit"],
            cfg["n_output"],
        ):
            raise ValueError("Incomplete validation grid")
        history = []
        for epoch, row in enumerate(metrics["history"]):
            measured, losses = summarise(scores[epoch], labels[nt:], rates[epoch])
            np.testing.assert_allclose(
                losses, saved_losses[epoch], rtol=1e-6, atol=1e-7
            )
            for key in measured:
                np.testing.assert_allclose(
                    row[key], measured[key], rtol=1e-6, atol=1e-7
                )
            if row["epoch"] != epoch or row["invalid_updates"]:
                raise ValueError("Invalid epoch history")
            history.append(dict(row, **measured))
        paired = (scores[-1].argmax(-1) == labels[nt:]).mean(0) - (
            scores[0].argmax(-1) == labels[nt:]
        ).mean(0)
        val_g, train_g = (
            z["validation_conductance_moments"],
            z["training_conductance_moments"],
        )
    with np.load(computed.file("calibration.npz")) as z:
        if not np.all((z["probability"] > 0) & (z["probability"] < 1)) or not np.all(
            z["amplitude_us"] > 0
        ):
            raise ValueError("Invalid frozen generator")
        mean_error = float(
            np.max(np.abs(z["fitted_mean_us"] / z["target_mean_us"] - 1))
        )
        sd_error = float(np.max(np.abs(z["fitted_sd_us"] / z["target_sd_us"] - 1)))
        if mean_error > 1e-6 or sd_error >= cfg["calibration_std_tolerance"]:
            raise ValueError("Calibration outside tolerance")
        vm = np.tile(z["target_mean_us"][nt:], (cfg["validation_draws"], 1))
        vs = np.tile(z["target_sd_us"][nt:], (cfg["validation_draws"], 1))
        # Training moments follow recorded shuffled batches; map back by identity.
        pos = {int(index): n for n, index in enumerate(indices)}
        order = [
            pos[index]
            for update in metrics["updates"]
            for index in update["original_indices"]
        ]
        tm, ts = z["target_mean_us"][order], z["target_sd_us"][order]
        fresh = dict(
            validation_mean_relative_error_median=float(
                np.median(np.abs(val_g[:, 0] / vm - 1))
            ),
            validation_sd_relative_error_median=float(
                np.median(np.abs(val_g[:, 1] / vs - 1))
            ),
            training_mean_relative_error_median=float(
                np.median(np.abs(train_g[:, 0] / tm - 1))
            ),
            training_sd_relative_error_median=float(
                np.median(np.abs(train_g[:, 1] / ts - 1))
            ),
        )
    expected_updates = cfg["epochs"] * int(np.ceil(nt / cfg["batch_size"]))
    if (
        len(metrics["updates"]) != expected_updates
        or len(set(metrics["training_input_hashes"])) != expected_updates
    ):
        raise ValueError("Missing updates or reused training draws")
    for epoch in range(1, cfg["epochs"] + 1):
        seen = [
            index
            for update in metrics["updates"]
            if update["epoch"] == epoch
            for index in update["original_indices"]
        ]
        np.testing.assert_array_equal(np.sort(seen), np.sort(indices[:nt]))
    if not all(
        np.isfinite(row["grad_norm"]) and np.isfinite(row["loss"])
        for row in metrics["updates"]
    ):
        raise ValueError("Invalid update metrics")
    changes = validate_parameters(computed, bank, cfg, metrics)
    best_epoch = min(
        range(len(history)),
        key=lambda ep: (history[ep]["cross_entropy"], -history[ep]["accuracy_pct"], ep),
    )
    if (
        metrics["best_epoch"] != best_epoch
        or not metrics["epoch_zero_replay_exact"]
        or metrics["native_filter_max_abs_error_us"] > 2e-5
    ):
        raise ValueError("Replay or selection evidence invalid")
    first, last = history[0], history[-1]
    bank.check_unchanged()
    return dict(
        history=history,
        best_epoch=best_epoch,
        update_count=expected_updates,
        accuracy_change_pp=last["accuracy_pct"] - first["accuracy_pct"],
        cross_entropy_change=last["cross_entropy"] - first["cross_entropy"],
        promising=last["cross_entropy"] < first["cross_entropy"]
        and last["accuracy_pct"] > first["accuracy_pct"],
        improved_images=int((paired > 0).sum()),
        worsened_images=int((paired < 0).sum()),
        unchanged_images=int((paired == 0).sum()),
        calibration_mean_error_max=mean_error,
        calibration_sd_error_max=sd_error,
        fresh_draw_moments=fresh,
        parameter_changes=changes,
        timings=timings,
        train_images=nt,
        validation_images=len(indices) - nt,
        validation_draws=cfg["validation_draws"],
        limitations=[
            cfg["validation_limitation"],
            "Fine-tuning only; single seed; image-conditioned frozen PING teacher; no synchrony isolation",
            "Fresh draws have fluctuating conductance moments; negative short pilot is inconclusive",
        ],
    )


def validate_raster(recorded, training):
    if recorded.record["inputs"]["training"] != training.reference:
        raise ValueError("Raster and training evidence pins differ")
    cfg = recorded.record["execution"]["configuration"]
    summary = json.loads(recorded.file("raster.json").read_text())
    metrics = json.loads(training.file("metrics.json").read_text())
    if summary["epoch"] != metrics["best_epoch"]:
        raise ValueError("Raster checkpoint selection mismatch")
    begin = cfg["train_per_digit"] * 10
    end = begin + summary["replayed_batch_size"]
    history = cfg.get("inhibition_prehistory_steps", 0)
    with (
        np.load(recorded.file("raster.npz")) as z,
        np.load(training.file("dataset.npz")) as data,
        np.load(training.file("calibration.npz")) as cal,
        np.load(training.file("validation.npz")) as val,
    ):
        indices = data["indices"][begin:end]
        np.testing.assert_array_equal(z["batch_indices"], indices)
        np.testing.assert_array_equal(z["batch_labels"], data["labels"][begin:end])
        np.testing.assert_array_equal(
            z["sensory_packed"], data["sensory_packed"][:, begin]
        )
        ev = artificial(
            cal["probability"][begin:end],
            cal["amplitude_us"][begin:end],
            indices,
            cfg,
            draw=summary["validation_draw"],
        )
        np.testing.assert_array_equal(z["artificial_events"], ev[history:, 0] > 0)
        np.testing.assert_array_equal(z["prehistory_events"], ev[:history, 0] > 0)
        np.testing.assert_allclose(
            z["g_artificial"],
            filtered(ev[:, :1], cfg)[history:, 0],
            rtol=1e-5,
            atol=2e-5,
        )
        expected = val["scores"][
            summary["epoch"], summary["validation_draw"], : len(indices)
        ]
        np.testing.assert_allclose(z["batch_scores"], expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            z["batch_rates_hz"],
            val["e_rates"][
                summary["epoch"], summary["validation_draw"], : len(indices)
            ],
            rtol=1e-6,
            atol=1e-5,
        )
        if not np.isin(z["E_spikes"], [0, 1]).all() or z["I_spikes"].any():
            raise ValueError("Invalid raster spikes")
        rate = float(z["E_spikes"].sum() / cfg["n_e"] / (cfg["duration_ms"] / 1000))
        np.testing.assert_allclose(rate, summary["e_rate_hz"], rtol=1e-6)
        if (summary["original_index"], summary["label"], summary["predicted"]) != (
            int(indices[0]),
            int(z["batch_labels"][0]),
            int(z["batch_scores"][0].argmax()),
        ):
            raise ValueError("Raster digit identity or prediction mismatch")
    actual = NetworkSnapshot.read(
        recorded.unit("execution", "start"), parameters=True
    ).load_parameters()
    expected = NetworkSnapshot.read(
        training.unit("best-validation", "end"), parameters=True
    ).load_parameters()
    for name, value in expected.items():
        np.testing.assert_array_equal(np.asarray(actual[name]), np.asarray(value))
    recorded.check_unchanged()
    return summary


def analyse(identity, run_id=None, *, raster_source=None):
    start = time.perf_counter()
    computed = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="compute"
    )
    results = validate(computed)
    inputs = {"compute": computed}
    if raster_source:
        recorded = source_run(
            ROOT / ".pingstore",
            raster_source,
            experiment=recipe.SLUG,
            stage="compute",
            phase="raster",
        )
        results["raster"] = validate_raster(recorded, computed)
        inputs["raster"] = recorded
    with stage_run(
        ROOT,
        recipe.SLUG,
        "analyse",
        inputs=inputs,
        configuration=computed.record["execution"]["configuration"],
        run_id=run_id,
    ) as run:
        results["timings"]["analysis_seconds"] = time.perf_counter() - start
        write_json(run.export / "results.json", results)
        run.record["execution"].update(
            no_simulation=True, elapsed_seconds=time.perf_counter() - start
        )
        computed.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--raster-source")
    args = parser.parse_args()
    analyse(args.source, args.run_id, raster_source=args.raster_source)
