"""Simulate features, train diagnostic decoders and retain held-out evidence only."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp080 import recipe
from experiments.exp080.recipe import (
    BATCH_SIZE,
    DT_MS,
    HIDDEN_UNITS,
    LEARNING_RATE,
    N_TIMESTEPS,
    PARAMETERS,
    PROBE_US,
    RATES_HZ,
    SEEDS,
    stable_seed,
)
from pingstore.contracts import file_sha256 as sha256_file
from pingstore.contracts import write_json_atomic
from pingstore.stages import stage_run


def torch_device() -> Any:
    import torch

    requested = os.environ.get("EXP080_DEVICE", "auto")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mnist_training() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from torchvision.datasets import MNIST

    dataset = MNIST(
        root=Path(os.environ.get("PINGLAB_DATA_ROOT", "/tmp/mnist")),
        train=True,
        download=True,
    )
    images = dataset.data.numpy().astype(np.uint8, copy=False)
    labels = dataset.targets.numpy().astype(np.int64, copy=False)
    if images.shape != (60_000, 28, 28) or labels.shape != (60_000,):
        raise RuntimeError(f"unexpected MNIST training contract: {images.shape}")
    return (
        images,
        labels,
        {
            "source": "torchvision.datasets.MNIST official training partition",
            "image_shape": list(images.shape),
            "label_shape": list(labels.shape),
            "raw_sha256": {
                path.name: sha256_file(path)
                for path in sorted(Path(dataset.raw_folder).glob("train-*-ubyte"))
            },
        },
    )


def load_mnist_test(cfg: dict) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from torchvision.datasets import MNIST

    dataset = MNIST(
        root=Path(os.environ.get("PINGLAB_DATA_ROOT", "/tmp/mnist")),
        train=False,
        download=True,
    )
    images = dataset.data.numpy().astype(np.uint8, copy=False)[: cfg["test_count"]]
    labels = dataset.targets.numpy().astype(np.int64, copy=False)[: cfg["test_count"]]
    return (
        images,
        labels,
        {
            "source": f"first {cfg['test_count']} images of the official MNIST test partition",
            "image_shape": list(images.shape),
            "label_shape": list(labels.shape),
            "raw_sha256": {
                path.name: sha256_file(path)
                for path in sorted(Path(dataset.raw_folder).glob("t10k-*-ubyte"))
            },
        },
    )


def direct_features(
    images_uint8: Any,
    rates_hz: Any,
    generator: Any,
) -> Any:
    """Direct decay-then-add AMPA and exponential membrane simulation."""
    import torch

    images = images_uint8.to(dtype=torch.float32).reshape(-1, 784) / 255.0
    rates = rates_hz.to(device=images.device, dtype=torch.float32).reshape(-1, 1)
    probability = images * rates * DT_MS / 1000.0
    if bool(torch.any(probability > 1.0)):
        raise ValueError("Bernoulli event probability exceeds one")
    conductance = torch.zeros_like(images)
    voltage = torch.full_like(images, PARAMETERS["E_L_mV"])
    feature_sum = torch.zeros_like(images)
    ampa_decay = math.exp(-DT_MS / PARAMETERS["tau_ampa_ms"])
    for _ in range(N_TIMESTEPS):
        events = (
            torch.rand(
                images.shape,
                device=images.device,
                dtype=images.dtype,
                generator=generator,
            )
            < probability
        )
        conductance = conductance * ampa_decay + PROBE_US * events
        total_g = PARAMETERS["g_L_uS"] + conductance
        equilibrium = (
            PARAMETERS["g_L_uS"] * PARAMETERS["E_L_mV"]
            + conductance * PARAMETERS["E_e_mV"]
        ) / total_g
        voltage = equilibrium + (voltage - equilibrium) * torch.exp(
            -DT_MS * total_g / PARAMETERS["C_m_nF"]
        )
        feature_sum += voltage - PARAMETERS["E_L_mV"]
    return feature_sum / N_TIMESTEPS


def make_model(device: Any, seed: int) -> Any:
    import torch

    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(784, HIDDEN_UNITS),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_UNITS, 10),
    )
    return model.to(device)


def batches(indices: np.ndarray, seed: int, shuffle: bool) -> list[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    return [
        order[start : start + BATCH_SIZE] for start in range(0, len(order), BATCH_SIZE)
    ]


def train_seed(
    images: np.ndarray,
    labels: np.ndarray,
    seed: int,
    output: Path,
    cfg: dict,
    *,
    state_cache=None,
) -> dict[str, Any]:
    import torch

    device = torch_device()
    model = make_model(device, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator(device=device)
    train_indices = np.arange(cfg["train_count"])
    validation_indices = np.arange(
        cfg["train_count"], cfg["train_count"] + cfg["validation_count"]
    )
    history: list[dict[str, Any]] = []
    best_accuracy = -math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    started = time.perf_counter()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_correct = 0
        train_seen = 0
        rate_rng = np.random.default_rng(stable_seed(seed, epoch, 1))
        generator.manual_seed(stable_seed(seed, epoch, 2))
        for indices in batches(train_indices, stable_seed(seed, epoch, 3), True):
            rates = np.asarray(RATES_HZ)[
                rate_rng.integers(0, len(RATES_HZ), len(indices))
            ]
            batch_images = torch.as_tensor(images[indices], device=device)
            batch_labels = torch.as_tensor(labels[indices], device=device)
            features = direct_features(
                batch_images,
                torch.as_tensor(rates, device=device, dtype=torch.float32),
                generator,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_correct += int((logits.argmax(1) == batch_labels).sum().item())
            train_seen += len(indices)

        model.eval()
        validation_correct = 0
        validation_seen = 0
        rate_rng = np.random.default_rng(stable_seed(seed, epoch, 4))
        generator.manual_seed(stable_seed(seed, epoch, 5))
        with torch.no_grad():
            for indices in batches(validation_indices, 0, False):
                rates = np.asarray(RATES_HZ)[
                    rate_rng.integers(0, len(RATES_HZ), len(indices))
                ]
                features = direct_features(
                    torch.as_tensor(images[indices], device=device),
                    torch.as_tensor(rates, device=device, dtype=torch.float32),
                    generator,
                )
                batch_labels = torch.as_tensor(labels[indices], device=device)
                validation_correct += int(
                    (model(features).argmax(1) == batch_labels).sum().item()
                )
                validation_seen += len(indices)
        validation_accuracy = validation_correct / validation_seen
        history.append(
            {
                "epoch": epoch,
                "train_accuracy": train_correct / train_seen,
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"exp080 seed={seed} epoch={epoch}/{cfg['epochs']} val={validation_accuracy:.4f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    directory = output / "models" / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "decoder.pt"
    if state_cache is None:
        torch.save({"state_dict": best_state, "seed": seed}, checkpoint)
    else:
        state_cache[seed] = best_state
    record = {
        "seed": seed,
        "device": str(device),
        "runtime_s": time.perf_counter() - started,
        "selected_epoch": best_epoch,
        "selected_validation_accuracy": best_accuracy,
        "history": history,
    }
    if state_cache is None:
        record.update(
            checkpoint=str(checkpoint.relative_to(output)),
            checkpoint_sha256=sha256_file(checkpoint),
        )
    else:
        record["checkpoint_retention"] = "memory_only"
    (directory / "training.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def load_models(
    records: list[dict[str, Any]], device: Any, output: Path, *, state_cache=None
) -> list[Any]:
    import torch

    models = []
    for item in records:
        seed = int(item["seed"])
        if state_cache is None:
            path = output / item["checkpoint"]
            if sha256_file(path) != item["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {path}")
            state = torch.load(path, map_location=device, weights_only=True)[
                "state_dict"
            ]
        else:
            state = state_cache[seed]
        model = make_model(device, seed)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def evaluate(
    records: list[dict[str, Any]], output: Path, cfg: dict, *, state_cache=None
) -> tuple[dict[str, Any], np.ndarray]:
    import torch

    images, labels, dataset = load_mnist_test(cfg)
    device = torch_device()
    models = load_models(records, device, output, state_cache=state_cache)
    correctness = np.empty(
        (len(RATES_HZ), len(SEEDS), cfg["test_count"]), dtype=np.bool_
    )
    labels_tensor = torch.as_tensor(labels, device=device)
    started = time.perf_counter()
    for rate_index, rate in enumerate(RATES_HZ):
        for start in range(0, cfg["test_count"], BATCH_SIZE):
            stop = min(start + BATCH_SIZE, cfg["test_count"])
            generator = torch.Generator(device=device).manual_seed(
                stable_seed(9, rate_index, start)
            )
            features = direct_features(
                torch.as_tensor(images[start:stop], device=device),
                torch.full((stop - start,), rate, device=device),
                generator,
            )
            with torch.no_grad():
                for seed_index, model in enumerate(models):
                    correctness[rate_index, seed_index, start:stop] = (
                        (model(features).argmax(1) == labels_tensor[start:stop])
                        .cpu()
                        .numpy()
                    )
        print(f"exp080 held-out rate={rate:g} Hz", flush=True)
    arrays_path = output / "held_out_correctness.npz"
    np.savez_compressed(
        arrays_path,
        correctness=correctness,
        rates_hz=np.asarray(RATES_HZ),
        seeds=np.asarray(SEEDS),
        labels=labels,
    )
    return {
        "device": str(device),
        "runtime_s": time.perf_counter() - started,
        "dataset": dataset,
        "arrays_sha256": sha256_file(arrays_path),
    }, correctness


def illustrative_features(images: np.ndarray, output: Path) -> None:
    import torch

    device = torch_device()
    rates = (0.5, 5.0, 25.0)
    generator = torch.Generator(device=device).manual_seed(stable_seed(11))
    selected = torch.as_tensor(
        np.repeat(images[[0]], len(rates), axis=0), device=device
    )
    features = (
        direct_features(
            selected,
            torch.as_tensor(rates, device=device, dtype=torch.float32),
            generator,
        )
        .cpu()
        .numpy()
    )
    np.savez_compressed(
        output / "feature_samples.npz",
        image=images[0],
        features_mV=features,
        rates_hz=np.asarray(rates),
    )


def compute(*, run_id: str | None = None) -> str:
    import torch

    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    with stage_run(
        REPO, recipe.SLUG, "compute", run_id=run_id, configuration=cfg
    ) as run:
        started = time.perf_counter()
        environment = {
            "PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0",
            "EXP080_DEVICE": str(torch_device()),
            "PINGLAB_DATA_ROOT": str(
                Path(os.environ.get("PINGLAB_DATA_ROOT", "/tmp/mnist")).absolute()
            ),
        }
        run.record["execution"]["environment"] = environment
        validation = recipe.validate_simulator()
        images, labels, dataset = load_mnist_training()
        illustrative_features(images, run.export)
        state_cache = {}
        records = [
            train_seed(images, labels, seed, run.export, cfg, state_cache=state_cache)
            for seed in SEEDS
        ]
        evaluation, _ = evaluate(records, run.export, cfg, state_cache=state_cache)
        state_cache.clear()
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp080.compute/v1",
                "recipe": cfg,
                "simulator_validation": validation,
                "training_dataset": dataset,
                "training": records,
                "evaluation": evaluation,
                "runtime_s": time.perf_counter() - started,
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                    "device": str(torch_device()),
                },
                "illustration": {"kind": "samples", "path": "feature_samples.npz"},
            },
        )
        from experiments.exp080.evidence import validate

        validate(run.export, cfg)
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", help="source-neutral v3 identity reserved before dispatch"
    )
    args = parser.parse_args()
    compute(run_id=args.run_id)


if __name__ == "__main__":
    main()
