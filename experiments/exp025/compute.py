"""Run explicit exp025 inference from an exp022 bank; never analyse or publish.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from experiments.exp025 import recipe
from experiments.exp025.custom import evidence, inputs
from experiments.exp025.custom.network_artifacts import model_execution
from pingstore.library import PingstoreError, file_sha256, write_json_atomic
from snnlab.lang import duration_steps
from snnlab.sim import (
    GraphExecutor,
    RecordingSpec,
    SignalRecording,
    encode_images_poisson,
    load_dataset,
    plan_graph,
    resolve_device,
)


def compute(identity, *, run_id=None):
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    contract = evidence.training_contract(bank.export)
    evidence.histories(bank.export, contract)
    with inputs.execution(
        REPO, "compute", sources={"bank": bank}, run_id=run_id, configuration=cfg
    ) as run:
        run.record["execution"]["environment"] = {
            "PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0"
        }
        for job in recipe.jobs(cfg):
            name = job["cell_name"]
            output = run.export / job["path"]
            training = contract["configs"][name]
            run_inference(
                train=bank.unit(name),
                output=output,
                attachments=run.scratch / "simulations" / job["path"],
                training=training,
                request=recipe.inference_request(training, job),
                author=recipe.author_network,
            )
            evidence.recordings(output, training, job)
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    a = p.parse_args()
    try:
        compute(a.source, run_id=a.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        p.exit(1, f"exp025 compute: {exc}\n")


ROLES = {
    "best_validation": "weights.pth",
    "final_epoch": "weights_final.pth",
}


def resolve_checkpoint(train_dir: Path, role: str) -> dict:
    """Return a verified checkpoint record for ``role`` or fail closed."""
    train_dir = Path(train_dir).resolve()
    if role not in ROLES:
        raise ValueError(
            f"unknown checkpoint role {role!r}; expected one of {sorted(ROLES)}"
        )
    metrics_path = train_dir / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read checkpoint metadata from {metrics_path}"
        ) from exc
    recorded = metrics.get("checkpoints", {}).get(role)
    if not isinstance(recorded, dict):
        raise RuntimeError(f"{metrics_path} does not register checkpoint role {role!r}")
    filename = recorded.get("filename")
    if filename != ROLES[role]:
        raise RuntimeError(
            f"{metrics_path} maps {role!r} to {filename!r}, expected {ROLES[role]!r}"
        )
    path = train_dir / filename
    if not path.is_file():
        raise RuntimeError(f"missing {role} checkpoint: {path}")
    digest = file_sha256(path)
    if recorded.get("sha256") != digest:
        raise RuntimeError(f"checkpoint hash mismatch for {path}")
    expected_epoch = (
        metrics.get("best_epoch")
        if role == "best_validation"
        else metrics.get("config", {}).get("epochs")
    )
    if int(recorded.get("epoch", -1)) != int(expected_epoch):
        raise RuntimeError(
            f"checkpoint epoch mismatch for {path}: {recorded.get('epoch')} != {expected_epoch}"
        )
    return {
        "training_cell": metrics.get("training_cell_name", train_dir.name),
        "role": role,
        "filename": filename,
        "epoch": int(recorded["epoch"]),
        "sha256": digest,
        "path": path,
    }


PARAMETERS = {
    "input_to_E.weight": "W_ff.0",
    "readout_projection.weight": "W_ff.1",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
}


def checkpoint_tensors(path, training):
    n_in, n_e, n_i, n_out = (
        training[k] for k in ("n_in", "n_hidden", "n_inh", "n_out")
    )
    for key in (
        "adaptive_threshold",
        "train_leak",
        "signed_readout",
        "readout_bias",
        "state_clamp",
        "trainable_w_ee",
        "trainable_w_ii",
    ):
        if training.get(key, False) is not False:
            raise PingstoreError(f"unsupported checkpoint dynamics: {key}")
    if training.get("dales_law") is not True:
        raise PingstoreError("requires Dale-constrained bank weights")
    if training.get("hidden_sizes", [n_e]) != [n_e] or training["readout_mode"] not in (
        "mem-mean",
        "spike-count",
    ):
        raise PingstoreError(
            "requires one E/I layer with mean pre-reset voltage readout"
        )
    shapes = {
        "W_ff.0": (n_in, n_e),
        "W_ff.1": (n_e, n_out),
        "W_ee.1": (n_e, n_e),
        "W_ei.1": (n_e, n_i),
        "W_ie.1": (n_i, n_e),
        "W_ii.1": (n_i, n_i),
    }
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise PingstoreError("checkpoint must contain exactly six matrices")
    for key, shape in shapes.items():
        value = state[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or tuple(value.shape) != shape
            or not torch.isfinite(value).all()
        ):
            raise PingstoreError(f"invalid checkpoint tensor: {key}")
        if key in ("W_ee.1", "W_ii.1") and torch.count_nonzero(value):
            raise PingstoreError(f"unsupported same-population recurrence: {key}")
        if key in ("W_ei.1", "W_ie.1") and (value < 0).any():
            raise PingstoreError(f"negative recurrent weight: {key}")
    return state


def initial_draws(training, *, ei_strength=None):
    """Ordered CPU weight draws, including unused zero recurrence.

    Return the generator cursor needed for CPU snapshots and the fresh loop
    needed for transfer probes. No model is rebuilt to recover these draws.
    """
    generator = torch.Generator().manual_seed(training["seed"])
    n_in, n_e, n_i, n_out = (
        training[k] for k in ("n_in", "n_hidden", "n_inh", "n_out")
    )

    def normal(shape, mean, sd, zero=0, scale=True):
        value = (
            torch.randn(*shape, generator=generator).mul_(sd).add_(mean).clamp_(min=0)
        )
        if zero:
            value *= (torch.rand(*shape, generator=generator) > zero).float()
            value /= 1 - zero
        return value / shape[0] if scale else value

    mean, sd = training["w_in"]
    external = normal((n_in, n_e), mean, sd, training["w_in_initial_zero_fraction"])
    if training.get("readout_w_init_mean") is not None:
        readout = normal(
            (n_e, n_out),
            training["readout_w_init_mean"],
            training["readout_w_init_std"],
            scale=False,
        )
    else:
        readout = normal((n_e, n_out), 5.1, 3.8) * training.get(
            "readout_w_out_scale", 225.0
        )
    normal((n_e, n_e), 0, 0)
    strength = training["ei_strength"] if ei_strength is None else ei_strength
    ei = normal((n_e, n_i), strength, strength * 0.1)
    ie_strength = strength * training["ei_ratio"]
    ie = normal((n_i, n_e), ie_strength, ie_strength * 0.1)
    normal((n_i, n_i), 0, 0)
    return generator, {
        "input_to_E.weight": external,
        "readout_projection.weight": readout,
        "E_to_I.weight": ei,
        "I_to_E.weight": ie,
    }


def bind_model(bundle, training, state, *, scale=1.0, fresh_loop=None, device=None):
    model = GraphExecutor(
        plan_graph(bundle.graph),
        seed=training["seed"],
        surrogate_slope=training["surrogate_slope"],
    )
    parameters = model.parameter_map()
    if set(parameters) != set(PARAMETERS):
        raise PingstoreError("graph parameter roles differ from checkpoint contract")
    with torch.no_grad():
        for name, key in PARAMETERS.items():
            value = state[key].clamp(min=0) if key.startswith("W_ff") else state[key]
            if fresh_loop is not None and name in ("E_to_I.weight", "I_to_E.weight"):
                value = fresh_loop[name]
            if name == "input_to_E.weight":
                value = value * scale
            if parameters[name].shape != value.shape:
                raise PingstoreError(f"graph/checkpoint shape mismatch: {name}")
            parameters[name].copy_(value)
    return model.to(device or resolve_device("auto")).eval()


def _run_inference_batches(
    *,
    model,
    training,
    request,
    device,
    selected,
    generator,
    synthetic,
    snapshot,
    steps,
    pixels,
    labels,
    recording,
    forward,
    output,
):
    """Run fresh-state batches and collect population counts and task metrics."""
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
                            steps,
                            len(indices),
                            training["n_in"],
                            generator=generator,
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
                    saved = perturb.setdefault(key, dict.fromkeys(row, 0))
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
    return counts, events, correct, ce_sum, perturb


def _select_inference_inputs(*, training, request, cursor, device):
    """Resolve input selection and random streams before executing batches."""
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
    return steps, snapshot, synthetic, pixels, labels, selected, generator


def run_inference(
    *, train, output, attachments, training, request, author, forward=None
):
    """Each batch begins with fresh state; input and checkpoint order are explicit."""
    checkpoint = resolve_checkpoint(train, request["checkpoint_role"])
    state = checkpoint_tensors(checkpoint["path"], training)
    device = torch.device(
        resolve_device(request.get("device", os.environ.get("PINGLAB_DEVICE", "auto")))
    )
    cursor, fresh = initial_draws(training, ei_strength=request.get("ei_strength"))
    bundle = author(training, observables=request["observables"])
    model = bind_model(
        bundle,
        training,
        state,
        scale=request.get("scale", 1),
        fresh_loop=fresh if "ei_strength" in request else None,
        device=device,
    )
    output.mkdir(parents=True, exist_ok=False)
    attachments.mkdir(parents=True, exist_ok=True)
    with model_execution(
        model,
        bundle,
        output / "inference",
        kind="infer",
        seed=training["seed"],
        settings=request,
    ):
        steps, snapshot, synthetic, pixels, labels, selected, generator = (
            _select_inference_inputs(
                training=training, request=request, cursor=cursor, device=device
            )
        )
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
        counts, events, correct, ce_sum, perturb = _run_inference_batches(
            model=model,
            training=training,
            request=request,
            device=device,
            selected=selected,
            generator=generator,
            synthetic=synthetic,
            snapshot=snapshot,
            steps=steps,
            pixels=pixels,
            labels=labels,
            recording=recording,
            forward=forward,
            output=output,
        )
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


if __name__ == "__main__":
    main()
