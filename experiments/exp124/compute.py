"""Train both SHD smoke conditions through the native graph executor."""

import argparse
import gzip
import hashlib
import os
import platform
import shutil
import sys
import time
import urllib.request
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import h5py
import numpy as np
import torch
from experiments.exp124 import recipe
from pingstore.contracts import write_json_atomic
from pingstore.stages import stage_run
from snnlab.sim.execution import GraphExecutor, plan_graph, resolve_device

BASE = "https://zenkelab.org/datasets/"


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def select_indices(all_labels, cfg):
    """Stratified held-out split; full mode retains every official train item."""
    rng = np.random.default_rng(cfg["seed"])
    train, validation = [], []
    for label in range(cfg["classes"]):
        available = np.flatnonzero(all_labels == label)
        if cfg.get("full_dataset", False):
            selected = rng.permutation(available)
            held_out = max(1, round(len(available) * cfg["validation_fraction"]))
            n_train = len(available) - held_out
            if n_train < 1:
                raise ValueError("each class needs training and validation examples")
        else:
            selected = rng.choice(available,
                                  cfg["train_per_class"] + cfg["validation_per_class"], replace=False)
            n_train = cfg["train_per_class"]
        train.extend(selected[:n_train])
        validation.extend(selected[n_train:])
    return train, validation


def load_inputs(cfg, cache):
    cache.mkdir(parents=True, exist_ok=True)
    filename = "shd_train.h5.gz"
    with urllib.request.urlopen(BASE + "md5sums.txt", timeout=60) as response:
        manifest = response.read().decode()
    expected = {line.split()[1].lstrip("*"): line.split()[0]
                for line in manifest.splitlines() if len(line.split()) == 2}[filename]
    archive = cache / filename
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        urllib.request.urlretrieve(BASE + filename, temporary)
        if digest(temporary, "md5") != expected:
            raise ValueError("SHD download checksum mismatch")
        temporary.rename(archive)
    if digest(archive, "md5") != expected:
        raise ValueError("cached SHD checksum mismatch")
    path = archive.with_suffix("")
    temporary = path.with_suffix(".unpack")
    with gzip.open(archive, "rb") as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
    temporary.replace(path)
    with h5py.File(path, "r") as handle:
        all_labels = np.asarray(handle["labels"], dtype=np.int64)
        train, validation = select_indices(all_labels, cfg)
        indices = np.asarray(train + validation, dtype=np.int64)
        events, rows, channels, times = [], [], [], []
        dropped, collisions = 0, 0
        for sample, index in enumerate(indices):
            t = np.asarray(handle["spikes/times"][index], dtype=np.float64) * 1000
            ch = np.asarray(handle["spikes/units"][index], dtype=np.int64)
            if not np.isfinite(t).all() or (t < 0).any() or (ch < 0).any() or (ch >= cfg["n_input"]).any():
                raise ValueError("invalid SHD events")
            keep = t < cfg["duration_ms"]
            dropped += int((~keep).sum())
            steps = np.floor(t[keep] / cfg["dt_ms"]).astype(np.int64)
            coordinates = np.unique(steps * cfg["n_input"] + ch[keep])
            collisions += int(keep.sum() - len(coordinates))
            events.append((coordinates // cfg["n_input"], coordinates % cfg["n_input"]))
            rows.append(np.full(int(keep.sum()), sample, dtype=np.int64))
            channels.append(ch[keep])
            times.append(t[keep])
        labels = all_labels[indices]
        speakers = np.asarray(handle["extra/speaker"])[indices]
    cfg.update(train_examples=len(train), validation_examples=len(validation),
               train_class_counts=np.bincount(labels[:len(train)], minlength=cfg["classes"]).tolist(),
               validation_class_counts=np.bincount(labels[len(train):], minlength=cfg["classes"]).tolist())
    if cfg.get("full_dataset", False) and len(indices) != len(all_labels):
        raise ValueError("full-data selection did not retain every official training sample")
    arrays = dict(event_sample=np.concatenate(rows), event_time_ms=np.concatenate(times),
                  event_channel=np.concatenate(channels), labels=labels,
                  official_train_indices=indices, speakers=speakers,
                  split=np.asarray([0] * len(train) + [1] * len(validation)))
    metadata = dict(url=BASE + filename, archive_md5=expected,
                    archive_sha256=digest(archive), hdf5_sha256=digest(path),
                    selection_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                    train_count=len(train), validation_count=len(validation),
                    official_train_count=len(all_labels),
                    noise_seeds={"training": cfg["seed"] + 2, "train_evaluation": cfg["seed"] + 3,
                                 "validation_evaluation": cfg["seed"] + 4},
                    evaluation_noise="same corruption each epoch; matched across models",
                    dropped_events=dropped, binary_collisions=collisions,
                    retained_events=int(sum(len(t) for t in times)), official_test_accessed=False)
    return events, labels, arrays, metadata


def batch(events, indices, cfg, device, rng=None):
    spikes = np.zeros((round(cfg["duration_ms"] / cfg["dt_ms"]), len(indices), cfg["n_input"]), dtype=np.float32)
    for position, index in enumerate(indices):
        steps, channels = events[int(index)]
        spikes[steps, position, channels] = 1
    if cfg["input_noise_hz"] or cfg["spike_deletion_probability"]:
        if rng is None:
            raise ValueError("noise requires an explicit seeded generator")
        noise = rng.random(spikes.shape, dtype=np.float32)
        occupied = spikes != 0
        spikes[occupied & (noise < cfg["spike_deletion_probability"])] = 0
        # Apply insertion to originally empty bins, independently of deletion.
        spikes[~occupied & (noise < cfg["input_noise_hz"] * cfg["dt_ms"] / 1000)] = 1
    return torch.from_numpy(spikes).to(device)


def initialize(model, cfg, condition):
    generator = torch.Generator().manual_seed(cfg["seed"])
    with torch.no_grad():
        for name, parameter in model.parameter_map().items():
            if name == "input_to_E1.weight":
                value = (torch.randn(parameter.shape, generator=generator) * cfg["input_weight_sd"] + cfg["input_weight_mean"]).clamp(min=0)
                value *= (torch.rand(parameter.shape, generator=generator) > cfg["input_initial_zero_fraction"])
                value /= cfg["n_input"] * (1 - cfg["input_initial_zero_fraction"])
            elif name == "E1_to_E2.weight":
                value = (torch.randn(parameter.shape, generator=generator) * cfg["interlayer_weight_sd"] + cfg["interlayer_weight_mean"]).clamp(min=0)
                zero_fraction = cfg["interlayer_initial_zero_fraction"]
                value *= (torch.rand(parameter.shape, generator=generator) > zero_fraction)
                value /= parameter.shape[0] * (1 - zero_fraction)
            elif name == "readout_projection.weight":
                value = torch.randn(parameter.shape, generator=generator) * cfg["readout_weight_sd"] + cfg["readout_weight_mean"]
                if cfg.get("center_readout_initialization", False):
                    value -= value.mean(dim=0, keepdim=True)
            elif name == "score_gain.weight":
                value = torch.eye(cfg["classes"]) * cfg["readout_gain"]
            else:
                strength = cfg["ei_summed_weight_us"] if name in ("E1_to_I1.weight", "E2_to_I2.weight") else cfg["ie_summed_weight_us"]
                value = torch.full(parameter.shape, strength / parameter.shape[0] if condition == "ping" else 0.0)
            parameter.copy_(value.to(parameter.device))
    return {name: value.detach().cpu().numpy().copy() for name, value in model.parameter_map().items()}


def evaluate(model, events, labels, indices, cfg, device, rng, condition):
    populations = ["e", "e1", "e2"]
    if condition == "ping" or cfg["schema"] == "exp124.recipe/v2":
        populations += ["i", "i1", "i2"]
    totals = torch.zeros(3 + len(populations), device=device, dtype=torch.float64)
    predictions = torch.zeros(cfg["classes"], device=device, dtype=torch.int64)
    finite = torch.ones((), device=device, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), cfg["batch_size"]):
            chosen = indices[start:start + cfg["batch_size"]]
            output = model({"drive": batch(events, chosen, cfg, device, rng)}, diagnostics=False,
                           **cfg["forward_retention"]).outputs
            scores = output["class_scores"]
            finite &= torch.isfinite(scores).all()
            target = torch.as_tensor(labels[chosen], device=device)
            predicted = scores.argmax(-1)
            totals[0] += torch.nn.functional.cross_entropy(scores, target) * len(chosen)
            totals[1] += (predicted == target).sum()
            totals[2] += scores.std(dim=-1).sum()
            predictions.scatter_add_(0, predicted, torch.ones_like(predicted))
            for offset, population in enumerate(populations, 3):
                totals[offset] += output[f"{population}_counts"].sum()
    if not bool(finite):
        raise ValueError("nonfinite class scores at evaluation boundary")
    measured = totals.cpu().tolist()
    return {"loss": measured[0] / len(indices), "accuracy": measured[1] / len(indices),
            "mean_score_sd": measured[2] / len(indices), "prediction_counts": predictions.cpu().tolist(),
            **{f"{population}_rate_hz": measured[offset] / len(indices) / cfg[f"n_{population[0]}"] / (cfg["duration_ms"] / 1000)
               for offset, population in enumerate(populations, 3)}}


def training_batches(indices, labels, cfg, rng):
    if not cfg.get("balanced_batches", False):
        order = rng.permutation(indices)
        return [order[start:start + cfg["batch_size"]] for start in range(0, len(order), cfg["batch_size"])]
    quota = cfg["batch_size"] // cfg["classes"]
    classes = [rng.permutation(indices[labels[indices] == label]) for label in range(cfg["classes"])]
    return [rng.permutation(np.concatenate([values[start:start + quota] for values in classes]))
            for start in range(0, cfg["train_per_class"], quota)]


def train_condition(condition, run, cfg, events, labels, device):
    directory = run.export / condition
    directory.mkdir()
    trainable = recipe.TRAINABLE
    bundle = recipe.author_network(cfg, condition)
    write_json_atomic(directory / "graph.json", bundle.graph)
    write_json_atomic(directory / "training.json", bundle.training)
    model = GraphExecutor(plan_graph(bundle.graph), seed=cfg["seed"],
                          trainable_parameters=trainable,
                          surrogate_slope=cfg["surrogate_slope"],
                          surrogate=bundle.training["surrogate"],
                          projection_mode=cfg["projection_mode"],
                          extension_validation=cfg["extension_validation"],
                          compile_mode=cfg["compile_mode"]).to(device)
    initial = initialize(model, cfg, condition)
    np.savez_compressed(directory / "initial_weights.npz", **initial)
    parameters = model.parameter_map()
    optimizer_type = torch.optim.Adam if cfg["optimizer"] == "adam" else torch.optim.AdamW
    optimizer = optimizer_type([
        {"params": [parameters[name] for name in trainable[:-1]], "lr": cfg["learning_rate"]},
        {"params": [parameters[trainable[-1]]], "lr": cfg["readout_learning_rate"]}],
                                  lr=cfg["learning_rate"], betas=tuple(cfg["optimizer_betas"]),
                                  weight_decay=cfg["weight_decay"])
    train_indices = np.arange(cfg["train_examples"])
    validation_indices = np.arange(len(train_indices), len(labels))
    rng = np.random.default_rng(cfg["seed"] + 1)
    noise_rng = np.random.default_rng(cfg["seed"] + 2)
    history, timings = [], []
    online_training = cfg["training_measurement"] == "online_pre_update_minibatches"
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(cfg["epochs"] + 1):
        epoch_start = time.perf_counter()
        gradients = {name: [] for name in trainable}
        train_totals = torch.zeros(2, device=device, dtype=torch.float64)
        update_timings = []
        if epoch:
            model.train()
            for chosen in training_batches(train_indices, labels, cfg, rng):
                if device.type == "cuda":
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                else:
                    begin = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                scores = model({"drive": batch(events, chosen, cfg, device, noise_rng)}, diagnostics=False,
                               **cfg["forward_retention"]).outputs["class_scores"]
                loss = torch.nn.functional.cross_entropy(scores, torch.as_tensor(labels[chosen], device=device))
                if online_training:
                    with torch.no_grad():
                        train_totals[0] += loss.detach() * len(chosen)
                        train_totals[1] += (scores.detach().argmax(-1) == torch.as_tensor(labels[chosen], device=device)).sum()
                elif not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                if not online_training:
                    for name in trainable:
                        gradient = parameters[name].grad
                        if gradient is None or not torch.isfinite(gradient).all():
                            raise ValueError(f"missing or nonfinite gradient: {name}")
                        gradients[name].append(float(gradient.norm()))
                if cfg["gradient_clip"] is not None:
                    torch.nn.utils.clip_grad_norm_([parameters[name] for name in trainable], cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                model.enforce_constraints()
                if device.type == "cuda":
                    end.record()
                    update_timings.append((len(chosen), begin, end))
                else:
                    update_timings.append((len(chosen), time.perf_counter() - begin))
        if online_training:
            loss_sum, correct = train_totals.cpu().tolist()
            if epoch and not np.isfinite(loss_sum):
                raise ValueError("nonfinite training loss at epoch boundary")
            train_measurement = {"loss": loss_sum / len(train_indices) if epoch else None,
                                 "accuracy": correct / len(train_indices) if epoch else None,
                                 "measurement": "online_pre_update_minibatches"}
        else:
            train_measurement = evaluate(model, events, labels, train_indices, cfg, device,
                                         np.random.default_rng(cfg["seed"] + 3), condition)
        row = {"epoch": epoch,
               "train": train_measurement,
               "validation": evaluate(model, events, labels, validation_indices, cfg, device,
                                      np.random.default_rng(cfg["seed"] + 4), condition),
               "gradient_l2_mean": {name: float(np.mean(values)) if values else None for name, values in gradients.items()}}
        row["updates"] = [{"examples": value[0], "seconds": value[1].elapsed_time(value[2]) / 1000
                          if device.type == "cuda" else value[1]} for value in update_timings]
        history.append(row)
        timings.append(time.perf_counter() - epoch_start)
        train_text = (f"train loss={row['train']['loss']:.4f}, accuracy={row['train']['accuracy']:.1%}"
                      if row["train"]["loss"] is not None else "training metrics unavailable before first update")
        print(f"{condition} epoch {epoch}: {train_text}; validation loss={row['validation']['loss']:.4f}, accuracy={row['validation']['accuracy']:.1%}; {timings[-1]:.1f}s", flush=True)
    final = {name: value.detach().cpu().numpy().copy() for name, value in parameters.items()}
    if any(not np.isfinite(value).all() for value in final.values()):
        raise ValueError("nonfinite learned weights")
    np.savez_compressed(directory / "final_weights.npz", **final)
    deltas = {name: float(np.linalg.norm(final[name] - initial[name])) for name in trainable}
    write_json_atomic(directory / "learning.json", {"epochs": history, "weight_change_l2": deltas})
    run.record["execution"].setdefault("models", {})[condition] = {
        "duration_seconds": time.perf_counter() - started, "epoch_seconds": timings,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "all_gradients_finite": None if online_training else True, "hidden_weights_changed": deltas[trainable[0]] > 0,
    }


def compute(
    run_id=None, device=None, cache=None, cpu_threads=None,
    train_examples=200, validation_examples=100, epochs=5, variant="paper",
    seed=42, batch_size=None, learning_rate=None, readout_gain=None, damping=None,
    readout_learning_rate=None, balanced_batches=None, center_readout=None,
    interlayer_strength=None, normalize_readout=None, input_strength=None, models=None,
    full_dataset=False, validation_fraction=0.1,
):
    cfg = recipe.configuration(
        train_examples, validation_examples, epochs, variant=variant, seed=seed,
        batch_size=batch_size, learning_rate=learning_rate, readout_gain=readout_gain,
        damping=damping, readout_learning_rate=readout_learning_rate,
        balanced_batches=balanced_batches, center_readout=center_readout,
        interlayer_strength=interlayer_strength, normalize_readout=normalize_readout,
        input_strength=input_strength, models=models, full_dataset=full_dataset,
        validation_fraction=validation_fraction,
    )
    device = torch.device(resolve_device(device or os.environ.get("PINGLAB_DEVICE", "cpu")))
    cpu_threads = int(cpu_threads if cpu_threads is not None else os.environ.get("PINGLAB_CPU_THREADS", "4"))
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(cpu_threads)
    cache = Path(cache or os.environ.get("SHD_CACHE", "/tmp/shd"))
    events, labels, arrays, metadata = load_inputs(cfg, cache)
    with stage_run(REPO, recipe.SLUG, "compute", run_id=run_id, configuration=cfg) as run:
        np.savez_compressed(run.export / "spikes.npz", **arrays)
        run.record["execution"].update(dataset=metadata, snnlab_version=version("snnlab"),
                                       torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
                                       host=platform.node(), cpu_architecture=platform.machine(),
                                       modal_cpu_cores=os.environ.get("PINGLAB_MODAL_CPU_CORES"),
                                       executor="snnlab.sim.execution.GraphExecutor")
        if Path("/proc/cpuinfo").exists():
            run.record["execution"]["cpu_model"] = next(
                (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("model name")), "unknown")
        for condition in cfg["models"]:
            train_condition(condition, run, cfg, events, labels, device)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--cache")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--train-examples", type=int, default=200)
    parser.add_argument("--validation-examples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--variant", choices=("baseline", "diverse", "damping", "sum", "gain", "combined", "maximum", "tuned", "paper"), default="paper")
    parser.add_argument("--full-dataset", action="store_true", help="Use every official training utterance; hold out validation within each class")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--readout-gain", type=float)
    parser.add_argument("--damping", type=float)
    parser.add_argument("--readout-learning-rate", type=float)
    parser.add_argument("--balanced-batches", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--center-readout", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--interlayer-strength", type=float)
    parser.add_argument("--normalize-readout", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--input-strength", type=float)
    parser.add_argument("--models", nargs="+", choices=recipe.MODELS,
                        help="Optional conditions for a focused diagnostic; defaults to COBA then PING")
    args = parser.parse_args()
    if args.full_dataset and any(token.split("=", 1)[0] in ("--train-examples", "--validation-examples")
                                 for token in sys.argv[1:]):
        parser.error("--full-dataset cannot be combined with example-count overrides")
    compute(**vars(args))


if __name__ == "__main__":
    main()
