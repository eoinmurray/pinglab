"""Evaluate the explicit TR-05 final checkpoints through native graphs."""

import argparse
import hashlib
import os
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import torch
from experiments.exp049 import evidence, inputs, recipe
from pingstore.contracts import PingstoreError, file_sha256, write_json_atomic
from snnlab.sim.datasets import load_dataset
from snnlab.sim.encoders import encode_images_poisson
from snnlab.sim.execution import GraphExecutor, plan_graph
from snnlab.sim.timing import duration_steps


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def evaluation(cfg):
    _, pixels, _, labels = load_dataset(
        "mnist", split=True, evaluation_split="test", evaluation_only=True
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
        raise PingstoreError("invalid official MNIST test pixels/labels")
    selected = np.random.RandomState(cfg["evaluation_subset_seed"]).choice(
        len(labels), cfg["evaluation_samples"], replace=False
    )
    return pixels, labels, selected


def snapshot(
    model, train, pixels, labels, job, destination, device, snapshot_generator
):
    """Encode the one reference image and retain its full-window E/I raster."""
    steps = duration_steps(train["t_ms"], train["dt"])
    stream = hashlib.sha256()
    indices = np.asarray([job["sample_index"]], dtype=np.int64)
    # Construction draws occurred on CPU. Device encoding used that CPU
    # state on CPU, and the untouched device stream on CUDA/MPS.
    with torch.inference_mode():
        if device.type == "mps":
            # MPS has a device-global RNG rather than a Generator.
            previous_rng = torch.mps.get_rng_state()
            try:
                torch.mps.manual_seed(train["seed"])
                drive = encode_images_poisson(
                    torch.from_numpy(pixels[indices]).to(device),
                    steps,
                    train["dt"],
                    train["input_rate"],
                )
            finally:
                torch.mps.set_rng_state(previous_rng)
        else:
            generator = (
                snapshot_generator
                if device.type == "cpu"
                else torch.Generator(device=device).manual_seed(train["seed"])
            )
            drive = encode_images_poisson(
                torch.from_numpy(pixels[indices]).to(device),
                steps,
                train["dt"],
                train["input_rate"],
                generator=generator,
            )
        stream.update(drive.cpu().numpy().tobytes())
        result = model(
            {"drive": drive},
            diagnostics=False,
            recording=recipe.recording(snapshot=True),
        )
    arrays = {
        "dt": np.float32(train["dt"]),
        "n_e": np.int32(train["n_hidden"]),
        "n_i": np.int32(train["n_inh"]),
        "label": np.int32(labels[indices[0]]),
    }
    for label, size in (("e", train["n_hidden"]), ("i", train["n_inh"])):
        coordinates = result.recorded_signals[f"{label.upper()}.spikes"].cpu().numpy()
        raster = np.zeros((steps, size), dtype=bool)
        raster[coordinates[:, 0], coordinates[:, 2]] = True
        arrays[f"spk_{label}"] = raster
    np.savez_compressed(
        destination / "recording.npz",
        dt=arrays["dt"],
        n_e=arrays["n_e"],
        n_i=arrays["n_i"],
        label=arrays["label"],
        spk_e=arrays["spk_e"],
        spk_i=arrays["spk_i"],
    )
    return indices, stream.hexdigest()


def inference(model, train, cfg, data, job, checkpoint, destination, device):
    """Preserve minibatch order, encoder stream, population means and rate weighting."""
    pixels, labels, selected = data
    steps = duration_steps(train["t_ms"], train["dt"])
    stream = hashlib.sha256()
    indices = selected
    generator = torch.Generator(device="cpu").manual_seed(
        cfg["evaluation_encoder_seed"]
    )
    correct, ce_sum, traces = 0, 0.0, []
    rate_sums = {"hid": 0.0, "inh": 0.0}
    with torch.inference_mode():
        for start in range(0, len(indices), cfg["evaluation_batch_size"]):
            batch = indices[start : start + cfg["evaluation_batch_size"]]
            drive = encode_images_poisson(
                torch.from_numpy(pixels[batch]).to(device),
                steps,
                train["dt"],
                train["input_rate"],
                generator=generator,
            )
            stream.update(drive.cpu().numpy().tobytes())
            result = model(
                {"drive": drive},
                diagnostics=False,
                recording=recipe.recording(),
            )
            scores = result.outputs["class_scores"]
            if (
                scores.shape != (len(batch), train["n_out"])
                or not torch.isfinite(scores).all()
            ):
                raise PingstoreError("invalid graph class scores")
            targets = torch.from_numpy(labels[batch]).to(device)
            correct += int((scores.argmax(1) == targets).sum())
            ce_sum += float(
                torch.nn.functional.cross_entropy(scores, targets, reduction="sum")
            )
            for label, key, size in (
                ("e", "hid", train["n_hidden"]),
                ("i", "inh", train["n_inh"]),
            ):
                counts = result.outputs[f"spk_{label}_count"]
                if (
                    counts.shape != (len(batch), size)
                    or not torch.isfinite(counts).all()
                    or (counts < 0).any()
                    or (counts > steps).any()
                    or not torch.equal(counts, counts.floor())
                ):
                    raise PingstoreError("invalid online spike counts")
                rate = float(counts.sum()) / (
                    len(batch) * size * steps * train["dt"] / 1000.0
                )
                rate_sums[key] += rate * len(batch)
            coordinates = result.recorded_signals["E.spikes"].cpu().numpy()
            population = np.zeros((steps, len(batch)), dtype=np.float32)
            np.add.at(
                population,
                (coordinates[:, 0], coordinates[:, 1]),
                np.float32(1),
            )
            population /= np.float32(train["n_hidden"])
            traces.extend(population.T)
    np.savez_compressed(
        destination / "pop_traces.npz",
        dt=np.float32(train["dt"]),
        pop_e=np.stack(traces),
    )
    write_json_atomic(
        destination / "metrics.json",
        {
            "config": {
                **train,
                "load_weights": f"{job['cell_name']}/{checkpoint['filename']}",
                "evaluation_partition": "official_mnist_test",
                "evaluation_samples": len(indices),
            },
            "best_acc": 100.0 * correct / len(indices),
            "n_correct": correct,
            "n_total": len(indices),
            "cross_entropy": ce_sum / len(indices),
            "rates_hz": {key: value / len(indices) for key, value in rate_sums.items()},
        },
    )
    return indices, stream.hexdigest()


def evaluate_cell(train_dir, train, checkpoint, cfg, data, export, device):
    """Bind source/target tensors directly; reset the graph on every call."""
    path = train_dir / checkpoint["filename"]
    if file_sha256(path) != checkpoint["sha256"]:
        raise PingstoreError("checkpoint content changed before binding")
    state = evidence.checkpoint_tensors(path, train)
    initial, snapshot_generator = recipe.initial_recurrence(train)
    bundle = recipe.author_network(cfg, train)
    model = GraphExecutor(
        plan_graph(bundle.graph),
        seed=train["seed"],
        surrogate_slope=train["surrogate_slope"],
    )
    parameters = model.parameter_map()
    if set(parameters) != set(recipe.CHECKPOINT_PARAMETERS):
        raise PingstoreError("exp049 graph parameter roles differ")
    with torch.no_grad():
        for name, source in recipe.CHECKPOINT_PARAMETERS.items():
            value = (
                state[source].clamp(min=0)
                if source.startswith("W_ff")
                else state[source]
            )
            if parameters[name].shape != value.shape:
                raise PingstoreError(f"checkpoint/graph shape differs: {name}")
            parameters[name].copy_(value)
    model.to(device).eval()
    pixels, labels, _ = data
    trials = []
    for job in (j for j in recipe.jobs(cfg) if j["cell_name"] == train_dir.name):
        destination = export / job["path"]
        destination.mkdir(parents=True, exist_ok=False)
        started = perf_counter()
        encoded_digest = None
        if job["kind"] == "weights_dump":
            np.savez_compressed(
                destination / "weights_dump.npz",
                **initial,
                W_ei_1_trained=state["W_ei.1"].numpy(),
                W_ie_1_trained=state["W_ie.1"].numpy(),
            )
            indices = np.empty(0, dtype=np.int64)
        elif job["kind"] == "snapshot":
            indices, encoded_digest = snapshot(
                model,
                train,
                pixels,
                labels,
                job,
                destination,
                device,
                snapshot_generator,
            )
        else:
            indices, encoded_digest = inference(
                model, train, cfg, data, job, checkpoint, destination, device
            )
        evidence.recordings(destination, train, job)
        trials.append(
            {
                "job": job,
                "checkpoint": checkpoint,
                "graph_digest": bundle.manifest["graph_digest"],
                "input_sha256": array_digest(pixels[indices], labels[indices], indices),
                "encoded_spikes_sha256": encoded_digest,
                "duration_seconds": perf_counter() - started,
            }
        )
        print(f"[{job['kind']}] {job['cell_name']}", flush=True)
    if file_sha256(path) != checkpoint["sha256"]:
        raise PingstoreError("checkpoint content changed during inference")
    return trials


def compute(identity, *, run_id=None):
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    contract = evidence.training_contract(bank.export)
    evidence.histories(bank.export, contract)
    data = evaluation(cfg)
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
    with inputs.execution(
        REPO, "compute", sources={"bank": bank}, run_id=run_id, configuration=cfg
    ) as run:
        run.record["execution"].update(
            environment={
                "PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0",
                "PINGLAB_DEVICE": str(device),
            },
            executor=cfg["executor"],
            device=str(device),
            snnlab_version=version("snnlab"),
            torch_version=str(torch.__version__),
            numpy_version=np.__version__,
            dataset_sha256=array_digest(*data),
            parameter_binding=recipe.CHECKPOINT_PARAMETERS,
            default_dtype=str(torch.get_default_dtype()),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_version=torch.version.cuda,
            cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            reuse="none: every cell and job is computed afresh; completed runs are never resumed",
            trials=[],
        )
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp049.compute/v1",
                "recipe": cfg,
                "training_contract": contract,
                "jobs": recipe.jobs(cfg),
            },
        )
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for cell, checkpoint in zip(
                contract["cells"], contract["checkpoints"], strict=True
            ):
                name = cell["cell_name"]
                run.record["execution"]["trials"].extend(
                    evaluate_cell(
                        bank.unit(name),
                        contract["configs"][name],
                        checkpoint,
                        cfg,
                        data,
                        run.export,
                        device,
                    )
                )
        finally:
            torch.set_num_threads(previous_threads)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        compute(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp049 compute: {exc}\n")


if __name__ == "__main__":
    main()
