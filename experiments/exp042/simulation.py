"""Native checkpoint-backed inference; shared caches exist only in writer scratch."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from experiments.helpers import ping
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from snnlab.sim import encoders, execution, extensions, interventions, models, streaming
from snnlab.sim.datasets import load_dataset
from snnlab.sim.execution import GraphExecutor, plan_graph
from snnlab.sim.interventions import ReplaySpikes, SparseSpikeReplay
from snnlab.sim.streaming import RecordingSpec, SignalRecording
from snnlab.sim.timing import duration_steps

from . import inputs, recipe, transforms


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _array_digest(*values):
    digest = hashlib.sha256()
    for value in values:
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([value.dtype.str, list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def execution_settings():
    """Resolve runtime identity before any cache or completed-shard reuse."""
    device = torch.device(
        os.environ.get("PINGLAB_DEVICE")
        or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    return {
        "device": str(device),
        "requested_device": os.environ.get("PINGLAB_DEVICE", "auto"),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else device.type,
        "snnlab_version": version("snnlab"),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda,
        "threads": torch.get_num_threads(),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "snnlab_source_sha256": {
            str(path.relative_to(Path(execution.__file__).parents[1])): file_sha256(
                path
            )
            for path in sorted(Path(execution.__file__).parents[1].rglob("*"))
            if path.is_file() and path.suffix in (".py", ".json")
        },
        "source_sha256": {
            module.__name__: file_sha256(Path(module.__file__))
            for module in (
                recipe,
                inputs,
                transforms,
                ping,
                encoders,
                execution,
                extensions,
                interventions,
                models,
                streaming,
            )
        },
        "inference_source_sha256": file_sha256(Path(__file__)),
        "compute_source_sha256": file_sha256(Path(__file__).with_name("compute.py")),
    }


def load_evaluation(cfg):
    """Freeze test pixels, labels and ordered subset; preserve the encoder stream."""
    _, pixels, _, labels = load_dataset(
        "mnist",
        split=True,
        evaluation_split="test",
        evaluation_only=True,
    )
    if (
        pixels.shape != (10000, 784)
        or pixels.dtype != np.float32
        or labels.shape != (10000,)
        or labels.dtype != np.int64
        or not np.isfinite(pixels).all()
        or (pixels < 0).any()
        or (pixels > 1).any()
        or (labels < 0).any()
        or (labels >= 10).any()
    ):
        raise PingstoreError("invalid normalized official MNIST test partition")
    samples = cfg["evaluation_samples"]
    if not 0 < samples <= len(labels):
        raise PingstoreError("invalid evaluation sample count")
    selected = np.random.RandomState(cfg["evaluation_subset_seed"]).choice(
        len(labels),
        samples,
        replace=False,
    )
    return {
        "pixels": pixels,
        "labels": labels,
        "selected": selected,
        "protocol": {
            "runtime": execution_settings(),
            "dataset_sha256": _array_digest(pixels, labels, selected),
        },
    }


def load_model(train_dir, cfg, protocol, bank_reference):
    """Bind four validated tensors once per training cell; no simulator adapter."""
    training = inputs.bank_configuration(train_dir)
    path, checkpoint = inputs.final_checkpoint(train_dir, training)
    state = inputs.checkpoint_tensors(path, training)
    bundle = recipe.author_network(cfg, training)
    model = GraphExecutor(
        plan_graph(bundle.graph),
        seed=training["seed"],
        surrogate_slope=training["surrogate_slope"],
    )
    parameters = model.parameter_map()
    if set(parameters) != set(recipe.CHECKPOINT_PARAMETERS):
        raise PingstoreError("graph parameter roles disagree with exp042 recipe")
    with torch.no_grad():
        for name, source in recipe.CHECKPOINT_PARAMETERS.items():
            value = (
                state[source].clamp(min=0)
                if source.startswith("W_ff")
                else state[source]
            )
            if parameters[name].shape != value.shape:
                raise PingstoreError(f"checkpoint/graph shape mismatch: {name}")
            parameters[name].copy_(value)
    identity = {
        "schema": "exp042.native-request/v2",
        "bank": dict(bank_reference),
        "recipe": cfg,
        "training": training,
        "checkpoint": checkpoint,
        "training_config_sha256": file_sha256(train_dir / "config.json"),
        "graph_digest": bundle.manifest["graph_digest"],
        **protocol,
    }
    return model.to(protocol["runtime"]["device"]).eval(), training, identity


def request(identity, *, recording, sample=None, replay=None, job=None):
    return {
        **identity,
        "recording": recording,
        "reset_policy": "fresh complete state per independent presentation; no continuation",
        "sample_index": sample,
        "encoder_seed": identity["training"]["seed"]
        if sample is not None
        else identity["recipe"]["evaluation_seed"],
        "job": job,
        "replay_sha256": replay,
    }


def replay_batch(rasters, dt_ms, start, end):
    """Native sparse validation and replay; never allocate a dense batch override."""
    trial, time, cell = (rasters[key] for key in ("i_trial", "i_t", "i_cell"))
    mask = (trial >= start) & (trial < end)
    trial, time, cell = trial[mask] - start, time[mask], cell[mask]
    order = np.lexsort((cell, trial, time))
    return SparseSpikeReplay.from_events(
        steps=torch.from_numpy(time[order]),
        batches=torch.from_numpy(trial[order]),
        cells=torch.from_numpy(cell[order]),
        steps_count=int(rasters["T"]),
        batch_size=end - start,
        cells_count=int(rasters["n_i"]),
        dt_ms=dt_ms,
    )


def raster_bounds(rasters, dt_ms):
    steps, cells, trials = (int(rasters[key]) for key in ("T", "n_i", "n_trials"))
    for key in ("T", "n_i", "n_trials"):
        value = np.asarray(rasters[key])
        if (
            value.ndim != 0
            or not np.issubdtype(value.dtype, np.integer)
            or int(value) <= 0
        ):
            raise PingstoreError(
                "inhibitory replay dimensions must be positive integers"
            )
    trial = rasters["i_trial"]
    if trial.size > 1 and (trial[1:] < trial[:-1]).any():
        raise PingstoreError("inhibitory raster must be ordered by trial")
    # Do not mask malformed coordinates away during native validation.
    if trial.size and ((trial < 0).any() or (trial >= trials).any()):
        raise PingstoreError("invalid baseline inhibitory raster trial")
    try:
        replay_batch(rasters, dt_ms, 0, trials)
    except (ValueError, TypeError) as exc:
        raise PingstoreError(str(exc)) from exc
    return steps, cells, trials, np.searchsorted(trial, np.arange(trials + 1))


def infer_batches(model, training, cfg, data, *, recording, rasters=None, sample=None):
    """Only encoding and measurement aggregation remain experiment-owned."""
    pixels, labels, selected = (data[key] for key in ("pixels", "labels", "selected"))
    steps = duration_steps(training["t_ms"], training["dt"])
    indices = selected if sample is None else np.array([sample], dtype=np.int64)
    if sample is not None and not 0 <= sample < len(labels):
        raise PingstoreError("snapshot index outside the official test partition")
    generator = torch.Generator().manual_seed(
        training["seed"] if sample is not None else cfg["evaluation_seed"]
    )
    batch_size = cfg["evaluation_batch_size"] if sample is None else 1
    if rasters is not None and tuple(
        int(rasters[k]) for k in ("T", "n_i", "n_trials")
    ) != (steps, training["n_inh"], len(indices)):
        raise PingstoreError("inhibitory replay coverage mismatch")
    correct, ce_sum, counts, events = 0, 0.0, {"e": 0, "i": 0}, []
    sparse_recording = (
        recipe.baseline_recording()
        if recording == "inhibitory" and sample is None
        else None
    )
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            drive = encoders.encode_images_poisson(
                torch.from_numpy(pixels[batch]),
                steps,
                training["dt"],
                training["input_rate"],
                generator=generator,
            )
            intervention = (
                ()
                if rasters is None
                else (
                    ReplaySpikes(
                        "I",
                        replay_batch(
                            rasters, training["dt"], start, start + len(batch)
                        ),
                    ),
                )
            )
            # Each call resets all state. Native recording/replay operates on
            # emitted spikes, including their recurrent transmission.
            result = model(
                {"drive": drive.to(next(model.parameters()).device)},
                diagnostics=sample is not None and recording == "spikes",
                recording=(
                    RecordingSpec((SignalRecording("I.spikes"),))
                    if sample is not None and recording == "inhibitory"
                    else sparse_recording
                ),
                interventions=intervention,
                batch_offset=start,
            )
            if sample is not None:
                values = (
                    result.numpy(batch=0).diagnostics
                    if recording == "spikes"
                    else {"spk_i": result.recorded_signals["I.spikes"].numpy()[:, 0]}
                )
                return {
                    key: values[key].astype(bool)
                    for key in ("spk_e", "spk_i")
                    if key in values
                } | {"label": np.int64(labels[sample])}
            scores = result.outputs["class_scores"]
            if (
                scores.shape != (len(batch), training["n_out"])
                or not torch.isfinite(scores).all()
            ):
                raise PingstoreError("invalid mean pre-reset-voltage class scores")
            targets = torch.from_numpy(labels[batch]).to(scores.device)
            correct += int((scores.argmax(1) == targets).sum())
            ce_sum += float(F.cross_entropy(scores, targets, reduction="sum"))
            for label, size in (("e", training["n_hidden"]), ("i", training["n_inh"])):
                value = result.outputs[f"spk_{label}_count"]
                if (
                    value.shape != (len(batch), size)
                    or not torch.isfinite(value).all()
                    or (value < 0).any()
                    or (value > steps).any()
                    or not torch.equal(value, value.floor())
                ):
                    raise PingstoreError("invalid population spike-count reduction")
                counts[label] += int(value.sum())
            if sparse_recording is not None:
                events.append(result.recorded_signals["I.spikes"].numpy())
    seconds = steps * training["dt"] / 1000
    metrics = {
        "best_acc": 100 * correct / len(indices),
        "ce_loss": ce_sum / len(indices),
        "n_correct": correct,
        "n_total": len(indices),
        "rates_hz": {
            "hid": counts["e"] / (len(indices) * training["n_hidden"] * seconds),
            "inh": counts["i"] / (len(indices) * training["n_inh"] * seconds),
        },
    }
    coordinates = np.concatenate(events) if events else np.empty((0, 3), dtype=np.int64)
    coordinates = coordinates[np.argsort(coordinates[:, 1], kind="stable")]
    return metrics, {
        "T": np.int32(steps),
        "n_i": np.int32(training["n_inh"]),
        "n_trials": np.int32(len(indices)),
        **{
            name: coordinates[:, axis].astype(np.int32)
            for name, axis in (("i_t", 0), ("i_trial", 1), ("i_cell", 2))
        },
    }


def baseline(model, training, cfg, data, identity, root):
    folder = root / "baseline" / training["training_cell_name"] / _digest(identity)
    folder.mkdir(parents=True, exist_ok=True)
    marker, path = folder / "complete.json", folder / "rasters.npz"
    with (folder / "cache.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            record = load_json(marker)
            if (
                record.get("request") != identity
                or record.get("rasters_sha256") != file_sha256(path)
                or record.get("metrics_sha256") != _digest(record["metrics"])
            ):
                raise PingstoreError("baseline scratch identity or payload changed")
            with np.load(path, allow_pickle=False) as archive:
                rasters = {key: np.array(archive[key]) for key in archive.files}
        else:
            metrics, rasters = infer_batches(
                model, training, cfg, data, recording="inhibitory"
            )
            np.savez_compressed(path, **rasters)
            write_json_atomic(
                marker,
                {
                    "request": identity,
                    "metrics": metrics,
                    "metrics_sha256": _digest(metrics),
                    "rasters_sha256": file_sha256(path),
                },
            )
        steps, cells, trials, _ = raster_bounds(rasters, training["dt"])
        if (steps, cells, trials) != (
            duration_steps(training["t_ms"], training["dt"]),
            training["n_inh"],
            cfg["evaluation_samples"],
        ):
            raise PingstoreError("baseline scratch coverage mismatch")
        return rasters


def override_rasters(rasters, condition, generator, dt_ms, cfg):
    steps, cells, trials, bounds = raster_bounds(rasters, dt_ms)
    output = {key: [] for key in ("i_trial", "i_t", "i_cell")}
    diagnostics = {
        "schema": "exp042.override/v2",
        "boundary_policy": cfg["jitter_policy"]["boundary"],
        "collision_policy": cfg["jitter_policy"]["collision"],
        "input_spikes": 0,
        "output_spikes": 0,
        "boundary_reflected_spikes": 0,
        "collision_resolved_spikes": 0,
        "max_collision_resolution_steps": 0,
        "trials_checked": trials,
        "cells_checked_per_trial": cells,
        "per_trial_cell_count_invariant": True,
    }
    for trial in range(trials):
        lo, hi = bounds[trial : trial + 2]
        value = torch.zeros((steps, 1, cells))
        value[rasters["i_t"][lo:hi], 0, rasters["i_cell"][lo:hi]] = 1
        override, row = transforms._build_override(
            value, condition, generator, dt_ms=dt_ms, return_diagnostics=True
        )
        for key in (
            "input_spikes",
            "output_spikes",
            "boundary_reflected_spikes",
            "collision_resolved_spikes",
        ):
            diagnostics[key] += row[key]
        diagnostics["max_collision_resolution_steps"] = max(
            diagnostics["max_collision_resolution_steps"],
            row["max_collision_resolution_steps"],
        )
        ti, ci = override.numpy()[:, 0].nonzero()
        output["i_trial"].append(np.full(ti.size, trial, dtype=np.int32))
        output["i_t"].append(ti.astype(np.int32))
        output["i_cell"].append(ci.astype(np.int32))
    result = {
        "T": np.int32(steps),
        "n_i": np.int32(cells),
        "n_trials": np.int32(trials),
        **{key: np.concatenate(rows) for key, rows in output.items()},
    }
    if (
        not diagnostics["input_spikes"]
        == diagnostics["output_spikes"]
        == result["i_t"].size
        == rasters["i_t"].size
    ):
        raise PingstoreError("override serialization changed inhibitory spike count")
    return result, diagnostics


def zero_replay(model, training, cfg, data, override, diagnostics, identity, root):
    folder = root / "zero-replay" / training["training_cell_name"] / _digest(identity)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "result.json"
    with (folder / "cache.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            record = load_json(path)
            if record.get("request") != identity or record.get(
                "metrics_sha256"
            ) != _digest(record["metrics"]):
                raise PingstoreError("zero-replay scratch identity or payload changed")
            return record["metrics"]
        metrics, _ = infer_batches(
            model, training, cfg, data, recording="none", rasters=override
        )
        metrics = {**metrics, "override_transform": diagnostics}
        write_json_atomic(
            path,
            {
                "request": identity,
                "metrics": metrics,
                "metrics_sha256": _digest(metrics),
            },
        )
        return metrics


def evaluate_jobs(bank, cfg, data, root, jobs, requests):
    """One model and baseline per cell; only the two canonical zero arms reuse metrics."""
    for cell in dict.fromkeys(job["cell"] for job in jobs):
        model, training, identity = load_model(
            bank.unit(cell), cfg, data["protocol"], bank.reference
        )
        base_request = request(identity, recording="inhibitory_events")
        requests.append(base_request)
        rasters = baseline(model, training, cfg, data, base_request, root)
        for job in (job for job in jobs if job["cell"] == cell):
            canonical = recipe.replay_job(job)
            generator = torch.Generator().manual_seed(
                cfg["evaluation_seed"] + 17 + canonical["seed_offset"]
            )
            override, diagnostics = override_rasters(
                rasters, canonical["condition"], generator, training["dt"], cfg
            )
            expected = request(
                identity,
                recording="none",
                job=canonical,
                replay=_array_digest(*(override[k] for k in sorted(override))),
            )
            requests.append(expected)
            if canonical["condition"] == "jitter_sigma_0":
                metrics = zero_replay(
                    model, training, cfg, data, override, diagnostics, expected, root
                )
            else:
                metrics, _ = infer_batches(
                    model, training, cfg, data, recording="none", rasters=override
                )
                metrics["override_transform"] = diagnostics
            yield job, metrics


def recordings(train_dir, cfg, data, bank_reference, requests):
    model, training, identity = load_model(
        train_dir, cfg, data["protocol"], bank_reference
    )
    sample = cfg["raster"]["sample_index"]
    requests.append(request(identity, recording="inhibitory", sample=sample))
    base = infer_batches(
        model, training, cfg, data, recording="inhibitory", sample=sample
    )
    for name, condition, offset in recipe.recording_jobs(cfg):
        generator = torch.Generator().manual_seed(cfg["evaluation_seed"] + 17 + offset)
        override = transforms._build_override(
            torch.from_numpy(base["spk_i"][:, None].astype(np.float32)),
            condition,
            generator,
            dt_ms=training["dt"],
        ).numpy()[:, 0]
        ti, ci = override.nonzero()
        rasters = {
            "T": np.int32(override.shape[0]),
            "n_i": np.int32(override.shape[1]),
            "n_trials": np.int32(1),
            "i_trial": np.zeros(ti.size, dtype=np.int32),
            "i_t": ti.astype(np.int32),
            "i_cell": ci.astype(np.int32),
        }
        requests.append(
            request(
                identity,
                recording="spikes",
                sample=sample,
                replay=_array_digest(override),
                job={"condition": condition, "seed_offset": offset},
            )
        )
        yield (
            name,
            infer_batches(
                model,
                training,
                cfg,
                data,
                recording="spikes",
                rasters=rasters,
                sample=sample,
            ),
        )
