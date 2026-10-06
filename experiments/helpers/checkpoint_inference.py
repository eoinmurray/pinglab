"""Native inference on explicit checkpoint tensors and recipe-owned requests.

There is no cache: every call computes fresh evidence. The request is retained
in execution scratch for shard authentication, never as a simulator config.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from experiments.exp022.checkpoints import resolve_checkpoint
from experiments.helpers import checkpoint_graph
from pingstore.contracts import PingstoreError, file_sha256, write_json_atomic
from snnlab.sim.datasets import load_dataset
from snnlab.sim.encoders import encode_images_poisson
from snnlab.sim.execution import resolve_device
from snnlab.sim.streaming import RecordingSpec, SignalRecording
from snnlab.sim.timing import duration_steps


def run_inference(
    train, output, attachments, training, request, author, *, forward=None
):
    """Each batch begins with fresh state; input and checkpoint order are explicit."""
    checkpoint = resolve_checkpoint(train, request["checkpoint_role"])
    state = checkpoint_graph.checkpoint_tensors(checkpoint["path"], training)
    device = torch.device(
        resolve_device(request.get("device", os.environ.get("PINGLAB_DEVICE", "auto")))
    )
    cursor, fresh = checkpoint_graph.initial_draws(
        training, ei_strength=request.get("ei_strength")
    )
    bundle = author(training, observables=request["observables"])
    model = checkpoint_graph.bind_model(
        bundle,
        training,
        state,
        scale=request.get("scale", 1),
        fresh_loop=fresh if "ei_strength" in request else None,
        device=device,
    )
    output.mkdir(parents=True, exist_ok=False)
    attachments.mkdir(parents=True, exist_ok=True)
    steps = duration_steps(request["t_ms"], training["dt"])
    snapshot = request["input"] == "snapshot"
    synthetic = request["input"] == "synthetic"
    if synthetic:
        pixels, labels = None, None
        selected = np.arange(request["trials"])
        generator = torch.Generator().manual_seed(training["seed"] + 1)
    else:
        _, pixels, _, labels = load_dataset(
            "mnist", split=True, evaluation_only=True, evaluation_split="test"
        )
        if (
            pixels.shape != (10000, training["n_in"])
            or labels.shape != (10000,)
            or not np.isfinite(pixels).all()
            or (pixels < 0).any()
            or (pixels > 1).any()
        ):
            raise PingstoreError("invalid official MNIST test partition")
        if snapshot:
            index = request.get("sample_index")
            if index is None:
                indices = np.flatnonzero(labels == request["digit"])
                index = int(indices[request["sample"]])
            if not 0 <= index < len(labels):
                raise PingstoreError("snapshot index out of range")
            selected = np.array([index])
            # Historical snapshots encode on the device stream after CPU
            # initialization; non-CPU initialization consumes no device draws.
            generator = (
                cursor
                if device.type == "cpu"
                else torch.Generator(device=device).manual_seed(training["seed"])
            )
        else:
            samples = request["samples"]
            selected = (
                np.random.RandomState(request["subset_seed"]).choice(
                    len(labels), samples, replace=False
                )
                if samples < len(labels)
                else np.arange(len(labels))
            )
            generator = torch.Generator().manual_seed(request["encoder_seed"])
    recording = (
        RecordingSpec(
            tuple(
                SignalRecording(f"{label}.spikes", kind="spike_events")
                for label in ("E", "I")
            )
        )
        if any(k in request["products"] for k in ("rasters", "population"))
        else None
    )
    counts = {"e": [], "i": []}
    events = {"e": [], "i": []}
    correct = 0
    ce_sum = 0.0
    perturb = {}
    with torch.inference_mode():
        for start in range(0, len(selected), request["batch_size"]):
            indices = selected[start : start + request["batch_size"]]
            if synthetic:
                probability = request["input_rate_hz"] * training["dt"] / 1000
                drive = (
                    (
                        torch.rand(
                            steps, len(indices), training["n_in"], generator=generator
                        )
                        < probability
                    )
                    .float()
                    .to(device)
                )
            else:
                images = (
                    torch.from_numpy(pixels[indices]).to(device)
                    if snapshot
                    else torch.from_numpy(pixels[indices])
                )
                drive = encode_images_poisson(
                    images,
                    steps,
                    training["dt"],
                    request["input_rate_hz"],
                    generator=generator,
                ).to(device)
            if forward:
                result, accounting = forward(model, drive, request)
                for key, row in accounting.items():
                    saved = perturb.setdefault(key, {field: 0 for field in row})
                    for field, value in row.items():
                        saved[field] += value
            else:
                result = model(
                    {"drive": drive},
                    diagnostics=snapshot,
                    recording=recording,
                    batch_offset=start,
                )
            for label in ("e", "i"):
                value = result.outputs[f"spk_{label}_count"]
                population = training["n_hidden" if label == "e" else "n_inh"]
                if (
                    value.shape != (len(indices), population)
                    or not torch.isfinite(value).all()
                    or (value < 0).any()
                    or (value > steps).any()
                    or not torch.equal(value, value.floor())
                ):
                    raise PingstoreError("invalid population count reduction")
                counts[label].append(
                    result.outputs[f"spk_{label}_count"].detach().cpu().numpy()
                )
                if recording:
                    events[label].append(
                        result.recorded_signals[f"{label.upper()}.spikes"].cpu().numpy()
                    )
            if snapshot:
                raw = {
                    k: v.detach().cpu().numpy()[:, 0]
                    for k, v in result.diagnostics.items()
                }
                np.savez_compressed(
                    output / "recording.npz",
                    dt=np.float32(training["dt"]),
                    n_e=np.int32(training["n_hidden"]),
                    n_i=np.int32(training["n_inh"]),
                    label=np.int64(labels[indices[0]]),
                    **raw,
                )
            elif not synthetic:
                scores = result.outputs["class_scores"]
                if (
                    scores.shape != (len(indices), training["n_out"])
                    or not torch.isfinite(scores).all()
                ):
                    raise PingstoreError("invalid class score reduction")
                targets = torch.from_numpy(labels[indices]).to(device)
                correct += int((scores.argmax(1) == targets).sum())
                ce_sum += float(F.cross_entropy(scores, targets, reduction="sum"))
    count = {k: np.concatenate(v) for k, v in counts.items()}
    seconds = steps * training["dt"] / 1000
    rates = {
        key: float(count[label].sum() / (len(selected) * training[size] * seconds))
        for label, key, size in (("e", "hid", "n_hidden"), ("i", "inh", "n_inh"))
    }
    config = {
        k: training[k]
        for k in (
            "dt",
            "t_ms",
            "seed",
            "tau_gaba_ms",
            "n_in",
            "n_hidden",
            "n_inh",
            "ei_strength",
            "ei_ratio",
        )
    }
    config.update(
        t_ms=request["t_ms"],
        load_weights=str(checkpoint["path"]),
        ei_strength=request.get("ei_strength", training["ei_strength"]),
    )
    if synthetic:
        config.update(input_rate_hz=request["input_rate_hz"], n_batch=len(selected))
    else:
        config.update(
            dataset="mnist",
            evaluation_partition="official_mnist_test",
            evaluation_samples=len(selected),
        )
    metrics = {
        "config": config,
        "rates_hz": rates,
        "rate_e_hz": rates["hid"],
        "rate_i_hz": rates["inh"],
    }
    if not synthetic:
        metrics.update(
            best_acc=100 * correct / len(selected),
            n_correct=correct,
            n_total=len(selected),
            ce_loss=ce_sum / len(selected),
        )
    if perturb:
        metrics["perturbation"] = {
            "mode": request["perturb_mode"],
            "level": request["perturb_level"],
            "dt_ms": training["dt"],
            "populations": perturb,
        }
    if not snapshot:
        write_json_atomic(output / "metrics.json", metrics)
    if recording:
        raster = {
            "dt": np.float32(training["dt"]),
            "T": np.int32(steps),
            "n_trials": np.int32(len(selected)),
            "n_e": np.int32(training["n_hidden"]),
            "n_i": np.int32(training["n_inh"]),
        }
        for label in ("e", "i"):
            coordinates = np.concatenate(events[label])
            for suffix, axis in (("t", 0), ("trial", 1), ("cell", 2)):
                raster[f"{label}_{suffix}"] = coordinates[:, axis].astype(np.int32)
        if "rasters" in request["products"]:
            np.savez_compressed(output / "rasters.npz", **raster)
        if "population" in request["products"]:
            linear = raster["e_trial"].astype(np.int64) * steps + raster["e_t"]
            pop = (
                np.bincount(linear, minlength=len(selected) * steps)
                .reshape(len(selected), steps)
                .astype(np.float32)
                / training["n_hidden"]
            )
            np.savez_compressed(
                output / "pop_traces.npz", pop_e=pop, dt=np.float32(training["dt"])
            )
    if "rates" in request["products"]:
        np.savez_compressed(
            output / "per_cell_rates.npz",
            rate_e_per_sample=count["e"].mean(1) / seconds,
            rate_e_per_cell=count["e"].mean(0) / seconds,
        )
    write_json_atomic(
        attachments / "request.json",
        {
            "request": request,
            "training": training,
            "checkpoint": {k: v for k, v in checkpoint.items() if k != "path"},
            "graph_digest": bundle.manifest["graph_digest"],
            "selected_indices": selected.tolist(),
            "input_content_sha256": None
            if synthetic
            else hashlib.sha256(pixels.tobytes() + labels.tobytes()).hexdigest(),
            "device": str(device),
        },
    )
    if file_sha256(checkpoint["path"]) != checkpoint["sha256"]:
        raise PingstoreError("checkpoint changed during inference")
    return metrics


def shard_identity(cfg, bank, recipe_module, *, dataset=None):
    """Freeze expected scientific/execution content before shard reuse.

    ``auto`` remains an explicit device-selection request; actual devices are
    recorded separately in worker provenance and each inference request.
    This identity does not rely on an artifact validator's inferred coverage.
    """
    import hashlib
    import os
    from importlib.metadata import version

    import snnlab
    from experiments.helpers import datasets, ping

    if dataset is None:
        _, pixels, _, labels = load_dataset(
            "mnist", split=True, evaluation_only=True, evaluation_split="test"
        )
    else:
        pixels, labels = dataset
    source = Path(snnlab.__file__).parent
    return {
        "recipe": cfg,
        "bank": bank.reference,
        "dataset": {
            "pixels": {
                "dtype": str(pixels.dtype),
                "shape": list(pixels.shape),
                "sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            },
            "labels": {
                "dtype": str(labels.dtype),
                "shape": list(labels.shape),
                "sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            },
        },
        "runtime": {
            "snnlab": version("snnlab"),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "requested_device": os.environ.get("PINGLAB_DEVICE", "auto"),
            "cuda": torch.version.cuda,
            "default_dtype": str(torch.get_default_dtype()),
            "threads": torch.get_num_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        },
        "snnlab_source": {
            str(p.relative_to(source)): file_sha256(p)
            for p in sorted(source.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".json")
        },
        "experiment_source": {
            str(p): file_sha256(p)
            for p in sorted(Path(recipe_module.__file__).parent.glob("*.py"))
        },
        "helper_source": {
            module.__name__: file_sha256(Path(module.__file__))
            for module in (ping, checkpoint_graph, datasets)
        },
        "inference_source": file_sha256(Path(__file__)),
    }
