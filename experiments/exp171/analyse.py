"""Validate the paired adaptation evidence without simulation."""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import numpy as np
import torch
import torch.nn.functional as F
from experiments.exp171 import recipe
from experiments.exp171.compute import (
    artificial,
    check_calibration_configuration,
    dataset_event_counts,
    digest,
    event_counts,
    filtered,
    first_correct_position,
    generator_key,
    moment_keys,
    poisson_expected_moments,
    relative_moment_error,
    source_bank,
    teacher_event_counts,
    validate_poisson_parameters,
    validate_source_configuration,
    write_json,
)
from pingstore.library import file_sha256, source_run, stage_run
from snnlab import sim
from snnlab.sim import NetworkSnapshot, load_runtime_state, load_training_checkpoint


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
            z[generator_key(cfg)][:1], z["amplitude_us"][:1], indices[:1], cfg, draw=0
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


def validate_generator(z, cfg, indices):
    parameter = z[generator_key(cfg)]
    upper = 2 if cfg.get("inhibition_method") == "count_matched_timing_v2" else 1
    valid = (
        (parameter >= 0) & (parameter <= upper)
        if generator_key(cfg) == "timing_interpolation"
        else (parameter > 0) & (parameter < 1)
    )
    if (
        not np.all(valid)
        or not np.all(z["amplitude_us"] > 0)
        or not np.isfinite(z["amplitude_us"]).all()
    ):
        raise ValueError("Invalid frozen generator")
    if generator_key(cfg) == "timing_interpolation":
        for coordinate in (
            {"epoch": 1},
            *({"draw": d} for d in range(cfg["validation_draws"])),
        ):
            for start in range(0, len(indices), cfg["batch_size"]):
                stop = start + cfg["batch_size"]
                events = artificial(
                    parameter[start:stop],
                    z["amplitude_us"][start:stop],
                    indices[start:stop],
                    cfg,
                    **coordinate,
                )
                if (
                    not np.all(
                        (events[cfg.get("inhibition_prehistory_steps", 0) :] > 0).sum(0)
                        == cfg["target_events_per_source"]
                    )
                    or events[-1].any()
                ):
                    raise ValueError("Invalid private event counts or delivery bins")


def native_history(metrics, *, penalised=False):
    """Use the trainer's fixed-weight split measurements, without re-evaluation."""
    history = []
    for row in metrics["epochs"]:
        components = row.get("validation_cross_entropies", {})
        if penalised and "objective[0]" not in components:
            raise ValueError(
                "Penalised validation requires separate cross-entropy evidence"
            )
        history.append(
            dict(
                epoch=row["epoch"],
                training_loss=row.get("train_loss"),
                cross_entropy=components.get("objective[0]", row["validation_loss"]),
                accuracy_pct=100 * row["validation_accuracy"],
                e_rate_hz=row["observations"]["validation"]["population_rates_hz"]["E"],
            )
        )
    if not all(
        np.isfinite(row[key])
        for row in history
        for key in ("cross_entropy", "accuracy_pct", "e_rate_hz")
    ):
        raise ValueError("Nonfinite native validation metrics")
    return history


def validate_teacher_activity(calibration, cfg):
    counts = calibration["teacher_i_spike_count"]
    rates = calibration["teacher_i_rate_hz"]
    expected = (cfg["train_images"] + cfg["validation_images"], cfg["n_i"])
    if (
        counts.shape != expected
        or rates.shape != expected
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0)
        or np.any(counts > cfg["steps"])
        or not np.isfinite(rates).all()
    ):
        raise ValueError("Invalid teacher inhibitory activity")
    np.testing.assert_allclose(
        rates,
        counts.astype(np.float32) / (cfg["duration_ms"] / 1000),
        rtol=1e-6,
        atol=0,
    )


def event_count_summary(computed, cfg):
    if cfg.get("inhibition_method") != recipe.TEACHER_COUNT_METHOD:
        return None
    with np.load(computed.file("calibration.npz")) as cal:
        counts = cal["event_count"]
        teacher_rate = cal["teacher_i_rate_hz"].mean(axis=1, dtype=np.float64)
        duration_s = cfg["duration_ms"] / 1000
        splits = {
            "train": slice(None, cfg["train_images"]),
            "validation": slice(cfg["train_images"], None),
        }
        return {
            name: dict(
                min=int(counts[part].min()),
                max=int(counts[part].max()),
                mean=float(counts[part].mean()),
                teacher_mean_i_rate_hz=float(teacher_rate[part].mean()),
                max_rate_rounding_error_hz=float(
                    np.max(np.abs(counts[part] / duration_s - teacher_rate[part]))
                ),
            )
            for name, part in splits.items()
        }


def poisson_rate_summary(computed, cfg):
    if cfg.get("inhibition_method") != recipe.POISSON_METHOD:
        return None
    with np.load(computed.file("dataset.npz")) as z:
        rates = z["poisson_rate_hz"]
        return {
            name: dict(
                mean=float(rates[part].mean()),
                min=float(rates[part].min()),
                max=float(rates[part].max()),
            )
            for name, part in {
                "train": slice(None, cfg["train_images"]),
                "validation": slice(cfg["train_images"], None),
            }.items()
        }


def validate_poisson_calibration(data, calibration, cfg):
    rates, amplitude = data["poisson_rate_hz"], data["amplitude"]
    validate_poisson_parameters(rates, amplitude)
    mean, sd = poisson_expected_moments(rates, amplitude, cfg)
    for actual, expected in (
        (mean, calibration["target_mean"]),
        (sd, calibration["target_sd"]),
        (mean, calibration["expected_mean"]),
        (sd, calibration["expected_sd"]),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)


def validate_native_dataset(computed, cfg):
    with (
        np.load(computed.file("dataset.npz")) as z,
        np.load(computed.file("calibration.npz")) as cal,
    ):
        ids = z["ids"]
        nt = cfg["train_images"]
        if len(ids) != nt + cfg["validation_images"] or len(np.unique(ids)) != len(ids):
            raise ValueError("Invalid native dataset identities")
        if (
            np.intersect1d(z["source_train"], z["source_validation"]).size
            or not np.isin(ids[:nt], z["source_train"]).all()
            or not np.isin(ids[nt:], z["source_validation"]).all()
        ):
            raise ValueError("Native dataset partition mismatch")
        np.testing.assert_array_equal(ids, cal["ids"])
        if cfg.get("teacher_statistics") == "per_image_conductance_and_i_rate_v1":
            validate_teacher_activity(cal, cfg)
        poisson = cfg.get("inhibition_method") == recipe.POISSON_METHOD
        if poisson:
            validate_poisson_calibration(z, cal, cfg)
        if cfg.get("inhibition_method") == recipe.TEACHER_COUNT_METHOD:
            expected_counts = teacher_event_counts(cal["teacher_i_spike_count"], cfg)
            event_counts(cfg, len(ids), z["event_count"])
            np.testing.assert_array_equal(z["event_count"], expected_counts)
            np.testing.assert_array_equal(cal["event_count"], expected_counts)
            if np.any((z["amplitude"] == 0) != (expected_counts[:, None] == 0)):
                raise ValueError(
                    "Zero-count images must have zero artificial amplitude"
                )
            if np.any(cal["target_mean"][expected_counts == 0] != 0) or np.any(
                cal["target_sd"][expected_counts == 0] != 0
            ):
                raise ValueError(
                    "Zero-count images cannot match nonzero teacher conductance"
                )
        alpha, amplitude = z[generator_key(cfg)], z["amplitude"]
        if (
            alpha.shape != (len(ids), cfg["n_e"])
            or amplitude.shape != alpha.shape
            or not np.isfinite(amplitude).all()
            or not np.all(amplitude >= 0)
            or (not poisson and not np.all((alpha >= 0) & (alpha <= 2)))
        ):
            raise ValueError("Invalid calibrated native generator")
        if cfg.get("inhibition_method") not in (
            recipe.TEACHER_COUNT_METHOD,
            recipe.POISSON_METHOD,
        ) and np.any(amplitude == 0):
            raise ValueError("Fixed-count calibration requires positive amplitudes")
        mean_error = float(
            np.max(relative_moment_error(cal[moment_keys(cfg)[0]], cal["target_mean"]))
        )
        sd_error = float(
            np.max(relative_moment_error(cal[moment_keys(cfg)[1]], cal["target_sd"]))
        )
        if (
            not np.isfinite([mean_error, sd_error]).all()
            or mean_error > 1e-6
            or sd_error >= cfg["calibration_std_tolerance"]
            or np.any(cal["target_mean"] < 0)
            or np.any(cal["target_sd"] < 0)
        ):
            raise ValueError("Native calibration outside tolerance")
    return mean_error, sd_error


def native_replay_events(data, raster, cfg, start, stop):
    """Decode retained device draws without requiring an accelerator for analysis."""
    history = cfg["inhibition_prehistory_steps"]
    ids = data["ids"][start:stop]
    if cfg.get("inhibition_method") == recipe.POISSON_METHOD:
        counts = raster["batch_artificial_counts"]
        if (
            counts.shape != (history + cfg["steps"], len(ids), cfg["n_e"])
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0)
        ):
            raise ValueError("Invalid retained Poisson event counts")
        if np.any(counts[:, data["poisson_rate_hz"][start:stop] == 0] != 0):
            raise ValueError("Silent Poisson channels contain events")
        return (counts.astype(np.float32) * data["amplitude"][None, start:stop]).astype(
            np.float32
        )
    if cfg.get("encoding_method") == "torch_device_v1":
        packed = raster["batch_artificial_packed"]
        expected = ((history + cfg["steps"] + 7) // 8, len(ids), cfg["n_e"])
        if packed.shape != expected or packed.dtype != np.uint8:
            raise ValueError("Invalid retained device-generated events")
        flags = np.unpackbits(packed, axis=0, count=history + cfg["steps"])
        if (
            not np.all(
                flags[history:].sum(0)
                == event_counts(
                    cfg, len(ids), dataset_event_counts(data, cfg, slice(start, stop))
                )[:, None]
            )
            or flags[-1].any()
        ):
            raise ValueError("Invalid device-generated event counts or delivery bins")
        events = flags * data["amplitude"][None, start:stop]
    else:
        events = artificial(
            data[generator_key(cfg)][start:stop],
            data["amplitude"][start:stop],
            ids,
            cfg,
            draw=0,
            counts=dataset_event_counts(data, cfg, slice(start, stop)),
        )
    return events


def validate_native_raster(computed, cfg, parameters, *, dataset_source=None):
    summary = json.loads(computed.file("raster.json").read_text())
    if (
        "precision_policy" in summary
        and summary["precision_policy"] != cfg["precision"]
    ):
        raise ValueError("Selected replay precision differs from validation")
    position = summary.get("batch_position", 0)
    dataset_source = dataset_source or computed
    start = cfg["train_images"]
    stop = start + min(
        cfg.get("replay_batch_size") or cfg["microbatch_size"], cfg["validation_images"]
    )
    history = cfg["inhibition_prehistory_steps"]
    with (
        np.load(dataset_source.file("dataset.npz")) as data,
        np.load(computed.file("raster.npz")) as z,
    ):
        ids, labels = data["ids"][start:stop], data["labels"][start:stop]
        np.testing.assert_array_equal(z["batch_indices"], ids)
        np.testing.assert_array_equal(z["batch_labels"], labels)
        if type(position) is not int or not 0 <= position < len(ids):
            raise ValueError("Invalid raster batch position")
        events = native_replay_events(data, z, cfg, start, stop)
        flags = (
            z["batch_artificial_counts"][:, position]
            if cfg.get("inhibition_method") == recipe.POISSON_METHOD
            else events[:, position] > 0
        )
        np.testing.assert_array_equal(z["artificial_events"], flags[history:])
        np.testing.assert_array_equal(z["prehistory_events"], flags[:history])
        np.testing.assert_allclose(
            z["g_artificial"],
            filtered(events[:, position : position + 1], cfg)[history:, 0],
            rtol=1e-5,
            atol=2e-5,
        )
        if (
            not np.isin(z["E_spikes"], [0, 1]).all()
            or z["I_spikes"].any()
            or not np.isfinite(z["batch_scores"]).all()
        ):
            raise ValueError("Invalid selected replay")
        rate = float(z["E_spikes"].sum() / cfg["n_e"] / (cfg["duration_ms"] / 1000))
        np.testing.assert_allclose(rate, summary["e_rate_hz"], rtol=1e-6)
        if (summary["original_index"], summary["label"], summary["predicted"]) != (
            int(ids[position]),
            int(labels[position]),
            int(z["batch_scores"][position].argmax()),
        ):
            raise ValueError("Selected replay digit mismatch")
    saved = NetworkSnapshot.read(
        computed.unit("best-validation", "end"), parameters=True
    ).load_parameters()
    for name, value in parameters.items():
        np.testing.assert_array_equal(saved[name], np.asarray(value))
    return summary


def validate_cached_calibration(computed, cfg):
    if "calibration" in computed.record["inputs"]:
        cached_pin = computed.record["inputs"]["calibration"]
        cached = source_run(
            ROOT / ".pingstore",
            cached_pin["run_id"],
            experiment=recipe.SLUG,
            stage="compute",
            reference=cached_pin,
        )
        check_calibration_configuration(
            cached.record["execution"]["configuration"], cfg
        )
        # The retained scientific calibration must be exactly the pinned one.
        for filename in ("dataset.npz", "calibration.npz"):
            with (
                np.load(computed.file(filename)) as actual,
                np.load(cached.file(filename)) as expected,
            ):
                if set(actual.files) != set(expected.files):
                    raise ValueError("Cached calibration arrays differ")
                for key in actual.files:
                    np.testing.assert_array_equal(actual[key], expected[key])
        cached.check_unchanged()


def validate_final_training_metrics(metrics, cfg):
    if cfg.get("evaluation_schedule") == "validation_each_epoch_train_final_only":
        measured = metrics["final_training"]
        if (
            measured["checkpoint"] != "final.checkpoint"
            or measured["epoch"] != cfg["epochs"]
            or measured["examples"] != cfg["train_images"]
            or measured["draw"] != cfg["final_training_draw"]
            or not np.isfinite(
                [
                    measured[key]
                    for key in ("cross_entropy", "accuracy_pct", "e_rate_hz", "seconds")
                ]
            ).all()
            or any(row.get("train_loss") is not None for row in metrics["epochs"])
        ):
            raise ValueError("Invalid final-only training evaluation")


def validate_native(computed):
    cfg = computed.record["execution"]["configuration"]
    pin = computed.record["inputs"]["bank"]
    if (
        set(computed.record["inputs"]) not in ({"bank"}, {"bank", "calibration"})
        or pin["run_id"] != cfg["source_run"]
        or pin["payload_digest"] != cfg["source_digest"]
    ):
        raise ValueError("Scientific dependency boundary violated")
    if pin != {"run_id": cfg["source_run"], "payload_digest": cfg["source_digest"]}:
        raise ValueError("Scientific dependency boundary violated")
    bank, _ = source_bank(cfg, root=ROOT)
    actual_spec = json.loads(
        computed.file("network.bundle", "training.json").read_text()
    )
    expected_spec = recipe.author_network(cfg).training
    if actual_spec["regularizers"] != expected_spec["regularizers"]:
        raise ValueError("Native activity penalty differs from the recorded protocol")
    validate_cached_calibration(computed, cfg)
    metrics = json.loads(computed.file("native-metrics.json").read_text())
    history = native_history(metrics, penalised=bool(cfg["rate_penalty"]))
    if [row["epoch"] for row in history] != list(range(cfg["epochs"] + 1)):
        raise ValueError("Incomplete native epoch history")
    expected_updates = cfg["epochs"] * int(
        np.ceil(cfg["train_images"] / cfg["batch_size"])
    )
    if (
        metrics["batch_counts"]["optimizer_updates"] != expected_updates
        or metrics["batch_counts"]["examples"] != cfg["epochs"] * cfg["train_images"]
    ):
        raise ValueError("Incomplete native training schedule")
    validate_final_training_metrics(metrics, cfg)
    mean_error, sd_error = validate_native_dataset(computed, cfg)
    initial = NetworkSnapshot.read(
        computed.unit("training", "start"), parameters=True
    ).load_parameters()
    teacher = NetworkSnapshot.read(
        computed.unit("teacher", "start"), parameters=True
    ).load_parameters()
    for kind, values in (("coba", initial), ("ping", teacher)):
        validate_source_configuration(
            json.loads(bank.file(cfg[kind + "_cell"], "config.json").read_text()), cfg
        )
        cp = cfg["source_checkpoints"][kind]
        path = bank.file(cfg[kind + "_cell"], cp["filename"])
        if file_sha256(path) != cp["sha256"]:
            raise ValueError("Source checkpoint digest mismatch")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        for name, key in recipe.MAPPING.items():
            np.testing.assert_array_equal(values[name], raw[key].clamp(min=0).numpy())
    with tempfile.TemporaryDirectory() as temporary:
        final = load_training_checkpoint(
            computed.restore_bundle("final.checkpoint", Path(temporary) / "final")
        )
        selected = load_training_checkpoint(
            computed.restore_bundle("best.checkpoint", Path(temporary) / "best")
        )
    if final.completed_updates != expected_updates or final.data_state != {
        "epoch": cfg["epochs"],
        "batch": 0,
    }:
        raise ValueError("Final native checkpoint has incorrect coordinates")
    changes = {}
    for name, before in initial.items():
        after = final.parameters[name].numpy()
        if not np.isfinite(after).all() or np.any(after < 0):
            raise ValueError("Invalid trained weights")
        changes[name] = float(np.max(np.abs(after - np.asarray(before))))
        np.testing.assert_allclose(changes[name], metrics["parameter_changes"][name])
        if name not in recipe.TRAINABLE:
            np.testing.assert_array_equal(before, after)
            np.testing.assert_array_equal(before, selected.parameters[name].numpy())
    np.testing.assert_array_equal(
        initial["artificial_to_E.weight"], np.ones((1, cfg["n_e"]))
    )
    best_epoch = min(
        range(len(history)),
        key=lambda ep: (history[ep]["cross_entropy"], -history[ep]["accuracy_pct"], ep),
    )
    if selected.data_state["epoch"] != best_epoch:
        raise ValueError("Native selected checkpoint disagrees with validation history")
    replay = validate_native_raster(computed, cfg, selected.parameters)
    if replay["epoch"] != best_epoch:
        raise ValueError("Selected replay has the wrong epoch")
    bank.check_unchanged()
    first, last = history[0], history[-1]
    return dict(
        execution_backend="snnlab.sim.train",
        event_count_summary=event_count_summary(computed, cfg),
        poisson_rate_summary=poisson_rate_summary(computed, cfg),
        final_training=metrics.get("final_training"),
        history=history,
        best_epoch=best_epoch,
        update_count=expected_updates,
        train_images=cfg["train_images"],
        validation_images=cfg["validation_images"],
        validation_draws=cfg["validation_draws"],
        accuracy_change_pp=last["accuracy_pct"] - first["accuracy_pct"],
        cross_entropy_change=last["cross_entropy"] - first["cross_entropy"],
        promising=last["cross_entropy"] < first["cross_entropy"]
        and last["accuracy_pct"] > first["accuracy_pct"],
        calibration_mean_error_max=mean_error,
        calibration_sd_error_max=sd_error,
        parameter_changes=changes,
        timings=json.loads(computed.file("timings.json").read_text()),
        raster=replay,
        limitations=[
            cfg["validation_limitation"],
            "Single seed; image-conditioned PING teacher; no synchrony isolation",
            "Fresh event draws need not match calibration moments exactly",
        ],
    )


def validate(computed):
    if (
        computed.record["execution"]["configuration"].get("execution_backend")
        == "snnlab.sim.train"
    ):
        return validate_native(computed)
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
        validate_generator(z, cfg, indices)
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
            cal[generator_key(cfg)][begin:end],
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


def validate_correct_native_raster(recorded, training):
    if recorded.record["inputs"] != {"training": training.reference}:
        raise ValueError("Raster and training evidence pins differ")
    cfg = recorded.record["execution"]["configuration"]
    original = training.record["execution"]["configuration"]
    for key in original:
        if key not in {"device", "threads"} and cfg.get(key) != original[key]:
            raise ValueError("Raster changed its training configuration")
    summary = json.loads(recorded.file("raster.json").read_text())
    with (
        np.load(training.file("raster.npz")) as saved,
        np.load(recorded.file("raster.npz")) as replay,
    ):
        position = first_correct_position(saved["batch_labels"], saved["batch_scores"])
        if (
            summary.get("selection_policy")
            != "first_correct_in_retained_validation_batch"
            or summary.get("batch_position") != position
        ):
            raise ValueError("Raster does not use the first correct retained image")
        for key in ["batch_indices", "batch_labels"]:
            np.testing.assert_array_equal(replay[key], saved[key])
        np.testing.assert_array_equal(
            replay["batch_scores"].argmax(axis=1), saved["batch_scores"].argmax(axis=1)
        )
        if int(replay["batch_scores"][position].argmax()) != int(
            replay["batch_labels"][position]
        ):
            raise ValueError("Raster image is not correctly classified")
    original_summary = json.loads(training.file("raster.json").read_text())
    if summary["epoch"] != original_summary["epoch"]:
        raise ValueError("Raster checkpoint selection mismatch")
    parameters = NetworkSnapshot.read(
        training.unit("best-validation", "end"), parameters=True
    ).load_parameters()
    result = validate_native_raster(recorded, cfg, parameters, dataset_source=training)
    recorded.check_unchanged()
    return result


def validate_sweep_configurations(configurations):
    targets = [cfg.get("rate_target_hz") for cfg in configurations]
    if len(targets) != len(recipe.RATE_TARGETS_HZ) or set(targets) != set(
        recipe.RATE_TARGETS_HZ
    ):
        raise ValueError(
            "Sweep requires exactly one completed run for each of the six ceilings"
        )
    first = configurations[0]
    varying = {"rate_target_hz", "rate_penalty", "coba_cell", "ping_cell"}
    source_keys = {"source_run", "source_digest"}
    distinct_sources = {
        (cfg["source_run"], cfg["source_digest"]) for cfg in configurations
    }
    if len(distinct_sources) > 1:
        original_banks = [source_bank(cfg, root=ROOT)[1] for cfg in configurations]
        if any(pin != original_banks[0] for pin in original_banks):
            raise ValueError(
                "Sweep compact sources do not share the same original bank"
            )
    keys = set(recipe.configuration()) - varying - source_keys
    for cfg in configurations:
        if any(cfg.get(key) != first.get(key) for key in keys):
            raise ValueError(
                "Sweep conditions differ outside the matched ceiling intervention"
            )
        expected = recipe.configuration(
            seed=cfg["seed"], rate_target_hz=cfg.get("rate_target_hz")
        )
        if any(cfg[key] != expected[key] for key in varying - {"rate_target_hz"}):
            raise ValueError("Sweep source cells and penalty do not match the ceiling")


def source_controls(computed):
    cfg = computed.record["execution"]["configuration"]
    pin = computed.record["inputs"]["bank"]
    if pin != {"run_id": cfg["source_run"], "payload_digest": cfg["source_digest"]}:
        raise ValueError("Scientific dependency boundary violated")
    bank, _ = source_bank(cfg, root=ROOT)
    controls = {}
    for kind in ("coba", "ping"):
        metrics = json.loads(bank.file(cfg[kind + "_cell"], "metrics.json").read_text())
        conf, final = metrics["config"], metrics["epochs"][-1]
        validate_source_configuration(conf, cfg)
        if (
            metrics["training_run_id"] != "TR-02"
            or final["ep"] != cfg["source_checkpoints"][kind]["epoch"]
            or final["ep"] != conf["epochs"]
            or conf["dataset_split"]["validation_samples"] != cfg["validation_images"]
            or conf["validation_encoder_draws"]["count"] != cfg["validation_draws"]
        ):
            raise ValueError("Incompatible TR02 final-epoch control")
        controls[kind] = dict(
            epoch=final["ep"],
            accuracy_pct=final["acc"],
            e_rate_hz=final["test_rate_e"],
            i_rate_hz=final["test_rate_i"],
        )
        if not all(np.isfinite(v) for v in controls[kind].values()):
            raise ValueError("Nonfinite TR02 control")
    bank.check_unchanged()
    return controls


def prepare_test_inputs(identities, baseline_id, run_id=None):
    """Retain compact native endpoints and existing matched TR02 test metrics."""
    started = time.perf_counter()
    sources = [
        source_run(ROOT / ".pingstore", i, experiment=recipe.SLUG, stage="compute")
        for i in identities
    ]
    configs = [s.record["execution"]["configuration"] for s in sources]
    validate_sweep_configurations(configs)
    baseline = source_run(
        ROOT / ".pingstore", baseline_id, experiment="exp025", stage="analyse"
    )
    original = source_bank(configs[0], root=ROOT)[1]
    if baseline.record["inputs"].get("bank") != original:
        raise ValueError("Test baselines do not pin the training bank")
    controls = json.loads(baseline.file("results.json").read_text())["results"]
    inputs = {
        "baseline": baseline,
        **{
            "training_" + recipe.target_label(c["rate_target_hz"]): s
            for s, c in zip(sources, configs, strict=True)
        },
    }
    cases = []
    with stage_run(
        ROOT,
        recipe.SLUG,
        "analyse",
        inputs=inputs,
        run_id=run_id,
        operation="prepare-test-inputs",
        configuration=recipe.test_configuration(),
    ) as run:
        snapshots = {}
        for training, cfg in zip(sources, configs, strict=True):
            label = recipe.target_label(cfg["rate_target_hz"])
            selected = {}
            for kind in ("coba", "ping"):
                rows = [
                    r
                    for r in controls
                    if r["cell_name"] == cfg[kind + "_cell"]
                    and r["seed"] == cfg["seed"]
                ]
                if len(rows) != 1:
                    raise ValueError("Missing unique matched TR02 test control")
                row = rows[0]
                if (
                    row["evaluation_partition"] != "official_mnist_test"
                    or row["evaluation_samples"] != 1000
                    or row["checkpoint_role"] != "final_epoch"
                ):
                    raise ValueError(
                        "Baseline is not the final-checkpoint 1000-image test evaluation"
                    )
                selected[kind] = dict(
                    accuracy_pct=row["final_acc"],
                    e_rate_hz=row["rate_e"],
                    seed=row["seed"],
                    samples=row["evaluation_samples"],
                    cell_name=row["cell_name"],
                )
            for prefix in (
                "final.checkpoint",
                "teacher/start",
                "teacher.bundle",
                "network.bundle",
            ):
                training.restore_bundle(prefix, run.export / label / prefix)
            snapshots[label + "/teacher"] = ["start"]
            cases.append(
                dict(
                    label=label,
                    training=training.reference,
                    configuration=cfg,
                    controls=selected,
                )
            )
        write_json(
            run.export / "test-inputs.json",
            dict(
                protocol=recipe.test_configuration(),
                cases=cases,
                baseline=baseline.reference,
                original_bank=original,
            ),
        )
        run.record["scientific_artifacts"] = {"snapshot_stages": snapshots}
        run.record["execution"].update(
            no_simulation=True, elapsed_seconds=time.perf_counter() - started
        )
    return run.run_id


def validate_test_evaluation(evaluated, computations):
    if evaluated.record["execution"]["operation"] != "test-final-checkpoints":
        raise ValueError("Expected final-checkpoint test evaluation")
    package = source_run(
        ROOT / ".pingstore",
        evaluated.record["inputs"]["test_inputs"]["run_id"],
        reference=evaluated.record["inputs"]["test_inputs"],
        experiment=recipe.SLUG,
        stage="analyse",
    )
    spec = json.loads(package.file("test-inputs.json").read_text())
    results = json.loads(evaluated.file("test-results.json").read_text())
    protocol = recipe.test_configuration()
    if results["protocol"] != protocol or spec["protocol"] != protocol:
        raise ValueError("Test protocol changed")
    expected = {
        recipe.target_label(c.record["execution"]["configuration"]["rate_target_hz"]): c
        for c in computations
    }
    if len(results["cases"]) != 6 or {c["label"] for c in results["cases"]} != set(
        expected
    ):
        raise ValueError("Incomplete test grid")
    with np.load(evaluated.file("test-subset.npz")) as data:
        ids = data["indices"]
        labels = data["labels"]
        features = data["features"]
        np.testing.assert_array_equal(
            ids, np.random.RandomState(42).choice(10000, 1000, replace=False)
        )
        if labels.shape != (1000,) or features.shape != (1000, 784):
            raise ValueError("Invalid test dataset")
    packages = {c["label"]: c for c in spec["cases"]}
    cases = []
    reference_hashes = None
    for row in results["cases"]:
        label = row["label"]
        training = expected[label]
        cfg = training.record["execution"]["configuration"]
        entry = packages[label]
        if (
            row["training"] != training.reference
            or entry["training"] != training.reference
            or row["checkpoint_epoch"] != cfg["epochs"]
            or row["target_hz"] != cfg["rate_target_hz"]
        ):
            raise ValueError("Test checkpoint provenance changed")
        with np.load(evaluated.file(label, "measurements.npz")) as data:
            scores = data["scores"]
            counts = data["e_counts"]
        if (
            scores.shape != (1000, 10)
            or counts.shape != (1000, cfg["n_e"])
            or not np.isfinite(scores).all()
            or not np.isfinite(counts).all()
            or np.any(counts < 0)
        ):
            raise ValueError("Invalid test measurements")
        correct = int((scores.argmax(-1) == labels).sum())
        accuracy = 100 * correct / 1000
        rate = float(counts.mean() / (cfg["duration_ms"] / 1000))
        if correct != row["correct"] or row["samples"] != 1000:
            raise ValueError("Test sample accounting changed")
        np.testing.assert_allclose(
            [accuracy, rate], [row["accuracy_pct"], row["e_rate_hz"]], rtol=1e-6
        )
        with (
            np.load(evaluated.file(label, "dataset.npz")) as data,
            np.load(evaluated.file(label, "calibration.npz")) as cal,
        ):
            np.testing.assert_array_equal(data["ids"], ids)
            np.testing.assert_array_equal(data["labels"], labels)
            np.testing.assert_array_equal(data["features"], features)
            mean, sd = poisson_expected_moments(
                data["poisson_rate_hz"], data["amplitude"], cfg
            )
            np.testing.assert_allclose(mean, cal["target_mean"], rtol=1e-6, atol=1e-10)
            np.testing.assert_allclose(sd, cal["target_sd"], rtol=1e-6, atol=1e-10)
            hashes = []
            rng = torch.Generator().manual_seed(protocol["encoder_seed"])
            for start in range(0, 1000, protocol["batch_size"]):
                np.testing.assert_array_equal(
                    data["test_rng_states"][start // 64], rng.get_state().numpy()
                )
                images = torch.from_numpy(features[start : start + 64]).float() / 255
                drive = sim.encode_images_poisson(
                    images,
                    cfg["steps"],
                    cfg["dt_ms"],
                    cfg["input_rate_hz"],
                    generator=rng,
                )
                hashes.append(digest(drive.numpy()))
            if hashes != row["sensory_sha256"]:
                raise ValueError("Test sensory inputs changed")
            if reference_hashes is not None and hashes != reference_hashes:
                raise ValueError("Unpaired test sensory inputs")
            reference_hashes = hashes
        cases.append(
            dict(
                target_hz=row["target_hz"],
                test=dict(accuracy_pct=accuracy, e_rate_hz=rate),
                controls=entry["controls"],
                training=training.reference,
            )
        )
    evaluated.check_unchanged()
    package.check_unchanged()
    return sorted(cases, key=lambda c: recipe.RATE_TARGETS_HZ.index(c["target_hz"]))


def analyse(identity, run_id=None, *, raster_source=None, test_source=None):
    start = time.perf_counter()
    identities = [identity] if isinstance(identity, str) else list(identity)
    computations = [
        source_run(ROOT / ".pingstore", value, experiment=recipe.SLUG, stage="compute")
        for value in identities
    ]
    if len(computations) > 1:
        validate_sweep_configurations(
            [c.record["execution"]["configuration"] for c in computations]
        )
        computations.sort(
            key=lambda c: recipe.RATE_TARGETS_HZ.index(
                c.record["execution"]["configuration"].get("rate_target_hz")
            )
        )
    computed = computations[0]
    results = validate(computed)
    inputs = {"compute": computed}
    if len(computations) > 1:
        cases = []
        with np.load(computed.file("dataset.npz")) as primary:
            for index, cell in enumerate(computations):
                cfg = cell.record["execution"]["configuration"]
                measured = results if index == 0 else validate(cell)
                role = (
                    "compute"
                    if index == 0
                    else "cell_" + recipe.target_label(cfg["rate_target_hz"])
                )
                inputs[role] = cell
                with np.load(cell.file("dataset.npz")) as data:
                    for key in (
                        "ids",
                        "features",
                        "labels",
                        "source_train",
                        "source_validation",
                    ):
                        np.testing.assert_array_equal(data[key], primary[key])
                cases.append(
                    dict(
                        **measured,
                        target_hz=cfg.get("rate_target_hz"),
                        compute_role=role,
                        controls=source_controls(cell),
                    )
                )
        results["sweep"] = cases
    if test_source:
        evaluated = source_run(
            ROOT / ".pingstore", test_source, experiment=recipe.SLUG, stage="compute"
        )
        results["test_frontier"] = validate_test_evaluation(evaluated, computations)
        results["test_protocol"] = recipe.test_configuration()
        inputs["test_evaluation"] = evaluated
    if raster_source:
        recorded = source_run(
            ROOT / ".pingstore",
            raster_source,
            experiment=recipe.SLUG,
            stage="compute",
            phase="raster",
        )
        results["raster"] = (
            validate_correct_native_raster(recorded, computed)
            if results.get("execution_backend") == "snnlab.sim.train"
            else validate_raster(recorded, computed)
        )
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
        for cell in computations:
            cell.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        action="append",
        help="Compute input; repeat for all six ceilings to analyse a sweep",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--raster-source")
    parser.add_argument("--test-source")
    parser.add_argument("--prepare-test-inputs", action="store_true")
    parser.add_argument(
        "--baseline-source",
        help="retained TR02 test analysis; no new baseline inference",
    )
    args = parser.parse_args()
    if args.prepare_test_inputs:
        if not args.baseline_source or args.test_source or args.raster_source:
            parser.error(
                "Preparing test inputs requires only sources and --baseline-source"
            )
        prepare_test_inputs(args.source, args.baseline_source, args.run_id)
    else:
        analyse(
            args.source,
            args.run_id,
            raster_source=args.raster_source,
            test_source=args.test_source,
        )
