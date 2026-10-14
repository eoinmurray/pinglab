"""Train on 6,000 MNIST images and benchmark complete held-out inference."""

import argparse
import hashlib
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import snntorch as snn
import torch
import torchvision
from experiments.exp126.recipe import CONFIG, SLUG
from pingstore.contracts import write_json_atomic
from pingstore.stages import stage_run
from snntorch import surrogate
from torch import nn
from torchvision.datasets import MNIST


class Classifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(cfg["inputs"], cfg["hidden"])
        self.output = nn.Linear(cfg["hidden"], cfg["outputs"])
        options = dict(
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            threshold=cfg["threshold"],
            spike_grad=surrogate.fast_sigmoid(slope=cfg["surrogate_slope"]),
            reset_mechanism="subtract",
            reset_delay=True,
        )
        self.hidden_lif = snn.Synaptic(**options)
        self.output_lif = snn.Synaptic(**options)
        nn.init.constant_(self.input.bias, 0.1)
        nn.init.constant_(self.output.bias, 0.1)

    def forward(self, images, generator):
        hidden_syn = images.new_zeros((len(images), self.cfg["hidden"]))
        hidden_mem = torch.zeros_like(hidden_syn)
        output_syn = images.new_zeros((len(images), self.cfg["outputs"]))
        output_mem = torch.zeros_like(output_syn)
        counts = torch.zeros_like(output_syn)
        probability = images * (
            self.cfg["peak_input_rate_hz"] * self.cfg["dt_ms"] / 1000
        )
        for _ in range(self.cfg["steps"]):
            events = (
                torch.rand(images.shape, device=images.device, generator=generator)
                < probability
            ).to(images.dtype)
            spikes, hidden_syn, hidden_mem = self.hidden_lif(
                self.input(events), hidden_syn, hidden_mem
            )
            output_spikes, output_syn, output_mem = self.output_lif(
                self.output(spikes), output_syn, output_mem
            )
            counts = counts + output_spikes
        return counts


def partition(labels, cfg):
    rng = np.random.default_rng(cfg["seed"])
    train, validation = [], []
    for label in range(cfg["outputs"]):
        selected = rng.permutation(np.flatnonzero(labels == label))
        stop = cfg["train_per_class"] + cfg["validation_per_class"]
        if len(selected) < stop:
            raise ValueError("insufficient examples for balanced partitions")
        train.extend(selected[: cfg["train_per_class"]])
        validation.extend(selected[cfg["train_per_class"] : stop])
    return np.sort(train), np.sort(validation)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def weights(model):
    return {
        name: value.detach().cpu().numpy().copy()
        for name, value in model.state_dict().items()
    }


@torch.inference_mode()
def evaluate(model, images, labels, seed, cfg):
    model.eval()
    generator = torch.Generator(device=images.device).manual_seed(seed)
    total_loss = images.new_zeros(())
    predictions = []
    for start in range(0, len(images), cfg["evaluation_batch_size"]):
        stop = start + cfg["evaluation_batch_size"]
        scores = model(images[start:stop], generator)
        total_loss += nn.functional.cross_entropy(
            scores, labels[start:stop], reduction="sum"
        )
        predictions.append(scores.argmax(1))
    predicted = torch.cat(predictions)
    return {
        "loss": float((total_loss / len(images)).cpu()),
        "accuracy": float((predicted == labels).double().mean().cpu()),
        "samples": len(images),
    }, predicted.cpu().numpy()


def compute(device_name, cpu_threads, *, run_id=None):
    cfg = dict(CONFIG)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    # Use the same floating-point precision for the local and GPU protocols.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with stage_run(REPO, SLUG, "compute", run_id=run_id, configuration=cfg) as run:
        setup_started = time.perf_counter()
        cache = Path(os.environ.get("PINGLAB_DATA_ROOT", "/tmp/pinglab-mnist"))
        official_train = MNIST(cache, train=True, download=True)
        official_test = MNIST(cache, train=False, download=True)
        labels = official_train.targets.numpy()
        train, validation = partition(labels, cfg)
        if len(labels) != 60000 or len(official_test) != 10000:
            raise ValueError("unexpected official MNIST split sizes")
        np.savez_compressed(
            run.export / "dataset.npz",
            train_indices=train,
            validation_indices=validation,
            official_train_labels=labels,
            official_test_labels=official_test.targets.numpy(),
            train_pixels=official_train.data[train].numpy(),
            validation_pixels=official_train.data[validation].numpy(),
            test_pixels=official_test.data.numpy(),
        )
        tensors = {}
        for name, raw, target in (
            ("train", official_train.data[train], official_train.targets[train]),
            (
                "validation",
                official_train.data[validation],
                official_train.targets[validation],
            ),
            ("test", official_test.data, official_test.targets),
        ):
            tensors[name] = (
                raw.flatten(1).to(device=device, dtype=torch.float32) / 255,
                target.to(device),
            )
        model = Classifier(cfg).to(device)
        np.savez_compressed(run.export / "initial_weights.npz", **weights(model))
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        order_rng = torch.Generator().manual_seed(cfg["seed"] + 2)
        encoding_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 3)
        sync(device)
        setup_seconds = time.perf_counter() - setup_started
        history, epoch_timings = [], []
        best_loss, selected_epoch = float("inf"), None
        protocol_started = time.perf_counter()
        for epoch in range(cfg["epochs"] + 1):
            sync(device)
            optimize_started = time.perf_counter()
            if epoch:
                model.train()
                for positions in torch.randperm(len(train), generator=order_rng).split(
                    cfg["batch_size"]
                ):
                    indices = positions.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    scores = model(tensors["train"][0][indices], encoding_rng)
                    loss = nn.functional.cross_entropy(
                        scores, tensors["train"][1][indices]
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg["gradient_clip"],
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
            sync(device)
            optimize_seconds = time.perf_counter() - optimize_started if epoch else 0.0
            evaluate_started = time.perf_counter()
            row = {"epoch": epoch}
            for split, seed in (
                ("train", cfg["seed"] + 10),
                ("validation", cfg["seed"] + 11),
            ):
                row[split], _ = evaluate(model, *tensors[split], seed, cfg)
                if not np.isfinite(row[split]["loss"]):
                    raise ValueError("nonfinite evaluation loss")
            sync(device)
            evaluation_seconds = time.perf_counter() - evaluate_started
            history.append(row)
            save_started = time.perf_counter()
            if epoch and row["validation"]["loss"] < best_loss:
                best_loss, selected_epoch = row["validation"]["loss"], epoch
                np.savez_compressed(
                    run.export / "selected_weights.npz", **weights(model)
                )
            sync(device)
            epoch_timings.append(
                dict(
                    epoch=epoch,
                    optimization_seconds=optimize_seconds,
                    evaluation_seconds=evaluation_seconds,
                    checkpoint_seconds=time.perf_counter() - save_started,
                )
            )
            print(
                f"epoch {epoch:02d}: train loss={row['train']['loss']:.4f} "
                f"accuracy={row['train']['accuracy']:.3f}; "
                f"validation loss={row['validation']['loss']:.4f} "
                f"accuracy={row['validation']['accuracy']:.3f}",
                flush=True,
            )
        sync(device)
        training_protocol_seconds = time.perf_counter() - protocol_started
        np.savez_compressed(run.export / "final_weights.npz", **weights(model))
        with np.load(run.export / "selected_weights.npz", allow_pickle=False) as saved:
            model.load_state_dict(
                {
                    name: torch.from_numpy(saved[name].copy()).to(device)
                    for name in saved.files
                }
            )
        # Warm kernels using validation images: test data never selects the model.
        warmup_started = time.perf_counter()
        warm = tuple(
            value[: cfg["evaluation_batch_size"]] for value in tensors["validation"]
        )
        for _ in range(cfg["warmup_batches"]):
            evaluate(model, *warm, cfg["seed"] + 20, cfg)
        sync(device)
        warmup_seconds = time.perf_counter() - warmup_started
        inference_seconds = []
        first_predictions = None
        for _ in range(cfg["inference_repeats"]):
            sync(device)
            started = time.perf_counter()
            metrics, predictions = evaluate(
                model, *tensors["test"], cfg["seed"] + 12, cfg
            )
            sync(device)
            inference_seconds.append(time.perf_counter() - started)
            if first_predictions is None:
                first_predictions, test_metrics = predictions, metrics
            elif not np.array_equal(first_predictions, predictions):
                raise ValueError(
                    "repeated inference changed predictions at fixed encoding seed"
                )
        np.savez_compressed(
            run.export / "test_predictions.npz",
            predictions=first_predictions,
            labels=official_test.targets.numpy(),
            indices=np.arange(10000),
        )
        write_json_atomic(
            run.export / "learning.json",
            dict(epochs=history, selected_epoch=selected_epoch, test=test_metrics),
        )
        dataset_hash = hashlib.sha256()
        for raw in (
            official_train.data,
            official_train.targets,
            official_test.data,
            official_test.targets,
        ):
            dataset_hash.update(raw.numpy().tobytes())
        run.record["execution"].update(
            {
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor() or platform.machine(),
                "cpu_threads": torch.get_num_threads(),
                "machine": platform.machine(),
                "environment": {
                    key: os.environ.get(key)
                    for key in (
                        "PINGLAB_DEVICE",
                        "PINGLAB_CPU_THREADS",
                        "PINGLAB_MODAL_CPU_CORES",
                    )
                },
                "versions": dict(
                    torch=torch.__version__,
                    snntorch=snn.__version__,
                    torchvision=torchvision.__version__,
                    numpy=np.__version__,
                ),
                "dataset": {
                    "source": "torchvision.datasets.MNIST",
                    "pixel_label_sha256": dataset_hash.hexdigest(),
                },
                "timing": dict(
                    setup_seconds=setup_seconds,
                    epoch_timings=epoch_timings,
                    optimization_seconds=sum(
                        row["optimization_seconds"] for row in epoch_timings
                    ),
                    training_protocol_seconds=training_protocol_seconds,
                    warmup_seconds=warmup_seconds,
                    inference_seconds=inference_seconds,
                    inference_scope="device-resident images; encoding, forward, decoding, loss, CPU prediction transfer; excludes loading, host-to-device upload and warmup",
                ),
            }
        )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="reserved v4 compute identity")
    parser.add_argument(
        "--phase",
        "--implementation",
        dest="implementation",
        choices=("snntorch", "snnlab"),
        help="equation-matched comparison condition; omitted preserves the original baseline",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-current", action="store_true")
    parser.add_argument("--source", help="explicit full comparison input for profiling")
    parser.add_argument(
        "--profile", action="store_true", help="diagnostic profiling only; no training"
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=os.environ.get("PINGLAB_DEVICE", "cpu"),
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=int(os.environ.get("PINGLAB_CPU_THREADS", "4")),
    )
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.profile_current:
        if not args.source or args.profile or args.implementation:
            parser.error(
                "--profile-current requires --source and excludes --profile/--phase"
            )
        from experiments.exp126.compute_parts.current_profiling import (
            compute as current_profile_compute,
        )

        current_profile_compute(args.source, args.device, args.cpu_threads, args.run_id)
    elif args.profile:
        if args.implementation:
            parser.error("--profile and --phase are mutually exclusive")
        if not args.source:
            parser.error("--profile requires --source")
        from experiments.exp126.compute_parts.profiling import (
            compute as profile_compute,
        )

        profile_compute(args.source, args.device, args.cpu_threads, args.run_id)
    elif args.implementation:
        from experiments.exp126.compute_parts.comparison_compute import (
            compute as matched_compute,
        )

        matched_compute(
            args.implementation, args.seed, args.device, args.cpu_threads, args.run_id
        )
    else:
        compute(args.device, args.cpu_threads, run_id=args.run_id)
