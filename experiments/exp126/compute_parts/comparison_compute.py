"""One independently executable, equation-matched implementation replicate."""

import hashlib
import os
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from experiments.exp126.compute import REPO, partition, sync
from experiments.exp126.compute_parts.matched_models import (
    initial_parameters,
    make_model,
)
from experiments.exp126.recipe import (
    SLUG,
    comparison_configuration,
    implementation_source_digest,
)
from pingstore.contracts import write_json_atomic
from pingstore.stages import stage_run
from snnlab.sim.execution import GraphExecutor
from torch import nn
from torchvision.datasets import MNIST


def arrays(model, gradients=False):
    return {
        key: (value.grad if gradients else value).detach().cpu().numpy().copy()
        for key, value in model.canonical().items()
    }


def restore(model, values):
    with torch.no_grad():
        for key, value in values.items():
            model.canonical()[key].copy_(
                torch.as_tensor(value, device=model.canonical()[key].device)
            )


def encode(images, seed, cfg):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probability = images * cfg["peak_input_rate_hz"] * cfg["dt_ms"] / 1000
    return (
        torch.rand((cfg["steps"], *images.shape), device="cpu", generator=generator).to(
            images.device
        )
        < probability
    ).float()


def timed(device, function):
    sync(device)
    started = time.perf_counter()
    value = function()
    sync(device)
    return value, time.perf_counter() - started


def optimizer(model, cfg):
    return torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )


def step(model, events, labels, opt, cfg):
    model.train()
    opt.zero_grad(set_to_none=True)
    scores, _ = model(events)
    loss = nn.functional.cross_entropy(scores, labels)
    loss.backward()
    nn.utils.clip_grad_norm_(
        model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
    )
    opt.step()


def benchmark(model, images, labels, cfg, device, initial):
    events = encode(images, cfg["seed"] + 600, cfg)
    measurements = {
        key: []
        for key in ("encoding", "forward", "forward_backward", "optimization_step")
    }
    peaks = {}
    for operation in measurements:
        for repeat in range(cfg["warmup_batches"] + cfg["timing_repeats"]):
            restore(model, initial)
            model.zero_grad(set_to_none=True)
            opt = optimizer(model, cfg)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            def call():
                if operation == "encoding":
                    return encode(images, cfg["seed"] + 600, cfg)
                if operation == "forward":
                    with torch.inference_mode():
                        return model(events)[0]
                if operation == "optimization_step":
                    return step(model, events, labels, opt, cfg)
                scores, _ = model(events)
                nn.functional.cross_entropy(scores, labels).backward()

            value, elapsed = timed(device, call)
            if repeat >= cfg["warmup_batches"]:
                measurements[operation].append(elapsed)
                if device.type == "cuda":
                    peaks[operation] = max(
                        peaks.get(operation, 0), torch.cuda.max_memory_allocated(device)
                    )
            del value
    restore(model, initial)
    model.zero_grad(set_to_none=True)
    return dict(
        seconds=measurements, peak_allocated_bytes=peaks, batch_size=len(images)
    )


def compute(implementation, seed, device_name, cpu_threads, run_id=None):
    cfg = comparison_configuration(seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.set_num_threads(cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    initial = initial_parameters(cfg, seed)
    with stage_run(
        REPO, SLUG, "compute", run_id=run_id, configuration=cfg, phase=implementation
    ) as run:
        run.record["implementation"] = implementation
        run.record["implementation_source_sha256"] = implementation_source_digest(REPO)
        train_data = MNIST(
            Path(os.environ.get("PINGLAB_DATA_ROOT", "/tmp/pinglab-mnist")),
            train=True,
            download=True,
        )
        test_data = MNIST(
            Path(os.environ.get("PINGLAB_DATA_ROOT", "/tmp/pinglab-mnist")),
            train=False,
            download=True,
        )
        # Partition is fixed across replicate seeds; only model/order/encoding streams vary.
        train_indices, validation_indices = partition(
            train_data.targets.numpy(), {**cfg, "seed": 42}
        )
        np.savez_compressed(
            run.export / "dataset.npz",
            train_indices=train_indices,
            validation_indices=validation_indices,
            official_train_labels=train_data.targets.numpy(),
            official_test_labels=test_data.targets.numpy(),
        )
        tensors = {}
        dataset_hash = hashlib.sha256()
        for raw in (
            train_data.data,
            train_data.targets,
            test_data.data,
            test_data.targets,
        ):
            dataset_hash.update(raw.numpy().tobytes())
        for split, raw, labels in (
            (
                "train",
                train_data.data[train_indices],
                train_data.targets[train_indices],
            ),
            (
                "validation",
                train_data.data[validation_indices],
                train_data.targets[validation_indices],
            ),
            ("test", test_data.data, test_data.targets),
        ):
            tensors[split] = (
                raw.flatten(1).to(device=device, dtype=torch.float32) / 255,
                labels.to(device),
            )
        model = make_model(implementation, cfg, initial, device)
        np.savez_compressed(run.export / "initial_weights.npz", **arrays(model))
        probe_images, probe_labels = (
            value[: cfg["probe_examples"]] for value in tensors["validation"]
        )
        probe = encode(probe_images, seed + 500, cfg)
        scores, traces = model(probe, record=True)
        loss = nn.functional.cross_entropy(scores, probe_labels)
        loss.backward()
        probe_arrays = {
            key: value.detach().cpu().numpy() for key, value in traces.items()
        }
        np.savez_compressed(
            run.export / "agreement.npz",
            **probe_arrays,
            scores=scores.detach().cpu().numpy(),
            events=probe.cpu().numpy().astype(np.uint8),
            labels=probe_labels.cpu().numpy(),
            loss=float(loss.detach().cpu()),
        )
        np.savez_compressed(
            run.export / "gradients.npz", **arrays(model, gradients=True)
        )
        del scores, traces, loss, probe_arrays, probe
        micro = benchmark(
            model,
            tensors["validation"][0][: cfg["batch_size"]],
            tensors["validation"][1][: cfg["batch_size"]],
            cfg,
            device,
            initial,
        )
        strict_micro, validation_agreement, validation_profiles = None, None, {}
        if implementation == "snnlab" and cfg.get("strict_validation_microbenchmark"):
            structural_executor = model.executor
            strict_executor = GraphExecutor(
                structural_executor.plan,
                trainable_parameters=tuple(initial),
                surrogate=structural_executor.surrogate,
                extension_validation="strict",
            ).to(device)
            with torch.no_grad():
                for key, value in initial.items():
                    strict_executor.weights[key].copy_(value.to(device))
            events = encode(
                tensors["validation"][0][: cfg["batch_size"]], seed + 600, cfg
            )
            targets = tensors["validation"][1][: cfg["batch_size"]]
            checks = {}
            try:
                for policy, executor in (
                    ("structural", structural_executor),
                    ("strict", strict_executor),
                ):
                    model.executor = executor
                    model.zero_grad(set_to_none=True)
                    scores, _ = model(events)
                    nn.functional.cross_entropy(scores, targets).backward()
                    checks[policy] = (
                        scores.detach().clone(),
                        {
                            key: value.grad.detach().clone()
                            for key, value in model.canonical().items()
                        },
                    )
                    validation_profiles[policy] = {}
                    for backward in (False, True):
                        model.zero_grad(set_to_none=True)
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as profiler:
                            if backward:
                                measured, _ = model(events)
                                nn.functional.cross_entropy(
                                    measured, targets
                                ).backward()
                            else:
                                with torch.inference_mode():
                                    measured, _ = model(events)
                            sync(device)
                        validation_profiles[policy][
                            "forward_backward" if backward else "forward"
                        ] = {
                            row.key: row.count
                            for row in profiler.key_averages()
                            if row.key
                            in (
                                "aten::isfinite",
                                "aten::_local_scalar_dense",
                                "cudaStreamSynchronize",
                                "aten::any",
                            )
                        }
                validation_agreement = {
                    "scores_equal": torch.equal(
                        checks["structural"][0], checks["strict"][0]
                    ),
                    "gradients_equal": all(
                        torch.equal(
                            checks["structural"][1][key], checks["strict"][1][key]
                        )
                        for key in initial
                    ),
                }
                if not all(validation_agreement.values()):
                    raise ValueError(
                        "strict and structural initial computations differ"
                    )
                model.executor = strict_executor
                strict_micro = benchmark(
                    model,
                    tensors["validation"][0][: cfg["batch_size"]],
                    tensors["validation"][1][: cfg["batch_size"]],
                    cfg,
                    device,
                    initial,
                )
            finally:
                model.executor = structural_executor
                restore(model, initial)
                model.zero_grad(set_to_none=True)
            del strict_executor, events, targets, checks, measured, scores
        opt = optimizer(model, cfg)
        audit = hashlib.sha256()
        timing = dict(encoding_seconds=0.0, audit_seconds=0.0, epoch_timings=[])

        def draw(images, draw_seed):
            events, seconds = timed(device, lambda: encode(images, draw_seed, cfg))
            timing["encoding_seconds"] += seconds
            started = time.perf_counter()
            audit.update(events.to(dtype=torch.uint8).cpu().numpy().tobytes())
            timing["audit_seconds"] += time.perf_counter() - started
            return events

        def evaluate(split, draw_seed, audit_events=True):
            images, labels = tensors[split]
            loss_sum = images.new_zeros(())
            predictions = []
            model.eval()
            with torch.inference_mode():
                for start in range(0, len(images), cfg["evaluation_batch_size"]):
                    events = (
                        draw(
                            images[start : start + cfg["evaluation_batch_size"]],
                            draw_seed + start,
                        )
                        if audit_events
                        else encode(
                            images[start : start + cfg["evaluation_batch_size"]],
                            draw_seed + start,
                            cfg,
                        )
                    )
                    scores, _ = model(events)
                    target = labels[start : start + cfg["evaluation_batch_size"]]
                    loss_sum += nn.functional.cross_entropy(
                        scores, target, reduction="sum"
                    )
                    predictions.append(scores.argmax(1))
            predictions = torch.cat(predictions)
            return dict(
                loss=float((loss_sum / len(images)).cpu()),
                accuracy=float((predictions == labels).double().mean().cpu()),
                samples=len(images),
            ), predictions.cpu().numpy()

        history, selected_epoch, best_loss = [], None, float("inf")
        protocol_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(cfg["epochs"] + 1):
            optimization_seconds = 0.0
            if epoch:
                order = np.random.default_rng(seed + epoch + 1000).permutation(
                    len(train_indices)
                )
                for start in range(0, len(order), cfg["batch_size"]):
                    positions = torch.as_tensor(
                        order[start : start + cfg["batch_size"]], device=device
                    )
                    events = draw(
                        tensors["train"][0][positions], seed + epoch * 10000 + start
                    )
                    _, seconds = timed(
                        device,
                        lambda: step(
                            model, events, tensors["train"][1][positions], opt, cfg
                        ),
                    )
                    optimization_seconds += seconds
            row = dict(epoch=epoch)
            started = time.perf_counter()
            for split, offset in (("train", 100), ("validation", 200)):
                row[split], _ = evaluate(split, seed + offset)
                if not np.isfinite(row[split]["loss"]):
                    raise ValueError("nonfinite learning metrics")
            sync(device)
            evaluation_seconds = time.perf_counter() - started
            history.append(row)
            if epoch and row["validation"]["loss"] < best_loss:
                best_loss, selected_epoch = row["validation"]["loss"], epoch
                np.savez_compressed(
                    run.export / "selected_weights.npz", **arrays(model)
                )
            timing["epoch_timings"].append(
                dict(
                    epoch=epoch,
                    optimization_seconds=optimization_seconds,
                    evaluation_seconds=evaluation_seconds,
                )
            )
            print(
                f"{implementation} seed={seed} epoch={epoch}: validation loss={row['validation']['loss']:.4f} accuracy={row['validation']['accuracy']:.3f}",
                flush=True,
            )
        timing["training_protocol_seconds"] = time.perf_counter() - protocol_started
        timing["optimization_seconds"] = sum(
            row["optimization_seconds"] for row in timing["epoch_timings"]
        )
        np.savez_compressed(run.export / "final_weights.npz", **arrays(model))
        with np.load(run.export / "selected_weights.npz") as saved:
            restore(model, {key: saved[key].copy() for key in saved.files})
        for _ in range(cfg["warmup_batches"]):
            events = encode(
                tensors["validation"][0][: cfg["evaluation_batch_size"]],
                seed + 300,
                cfg,
            )
            with torch.inference_mode():
                model(events)
        timing["inference_seconds"] = []
        first = None
        for _ in range(cfg["inference_repeats"]):
            (metric, predictions), seconds = timed(
                device, lambda: evaluate("test", seed + 400, audit_events=False)
            )
            timing["inference_seconds"].append(seconds)
            if first is None:
                first, test_metric = predictions, metric
            elif not np.array_equal(first, predictions):
                raise ValueError("unstable repeated test predictions")
        np.savez_compressed(
            run.export / "test_predictions.npz",
            predictions=first,
            labels=test_data.targets.numpy(),
            indices=np.arange(len(test_data)),
        )
        timing["peak_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        )
        write_json_atomic(
            run.export / "learning.json",
            dict(epochs=history, selected_epoch=selected_epoch, test=test_metric),
        )
        run.record["execution"].update(
            device=str(device),
            device_name=torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
            cpu_threads=cpu_threads,
            timing=timing,
            microbenchmark=micro,
            extension_validation=cfg.get("snnlab_extension_validation")
            if implementation == "snnlab"
            else "not_applicable",
            strict_microbenchmark=strict_micro,
            validation_policy_agreement=validation_agreement,
            validation_policy_profiles=validation_profiles,
            versions={
                name: version(name)
                for name in ("torch", "snntorch", "snnlab", "torchvision")
            },
            dataset_sha256=dataset_hash.hexdigest(),
            encoding_sha256=audit.hexdigest(),
            batch_timing_scope="preencoded GPU-resident events; synchronization each boundary; extension validation included",
            inference_timing_scope="complete test split; GPU encoding, forward, decoding, loss and CPU predictions; excludes loading and warmup",
        )
    return run.run_id
