"""Train SHD with SNNLab Lang recipes and its native training/timing APIs."""

import argparse
import gzip
import hashlib
import os
import platform
import shutil
import sys
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
from pingstore.stages import source_run, stage_run
from snnlab import extensions
from snnlab.sim import (
    Batching,
    CheckpointSelection,
    DatasetEncoder,
    DatasetSnapshotBinding,
    EpochObservations,
    ExecutionSpec,
    GraphExecutor,
    TrainingMetrics,
    ValidationSpec,
    build,
    plan_graph,
    resolve_device,
    train,
)

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


def encode_shd(arrays, selected, *, dt_ms, channels, config, seed):
    """Bin ragged SHD events and apply the experiment's insertion/deletion noise."""
    spikes = np.zeros((round(config["duration_ms"] / dt_ms), len(selected), channels), dtype=np.float32)
    for column, local_index in enumerate(selected):
        index = int(local_index) + config["sample_offset"]
        start, end = arrays["event_offsets"][index:index + 2]
        steps = np.floor(arrays["event_time_ms"][start:end] / dt_ms).astype(np.int64)
        keep = steps < len(spikes)
        spikes[steps[keep], column, arrays["event_channel"][start:end][keep]] = 1
    noise = np.random.default_rng(seed).random(spikes.shape, dtype=np.float32)
    occupied = spikes != 0
    spikes[occupied & (noise < config["spike_deletion_probability"])] = 0
    spikes[~occupied & (noise < config["input_noise_hz"] * dt_ms / 1000)] = 1
    return torch.from_numpy(spikes)


extensions.register_encoder("exp124.shd_events/v1", encode_shd)


def train_condition(condition, run, cfg, device):
    """Delegate initialization, batching, optimization, observations and timing to SNNLab."""
    directory = run.export / condition
    directory.mkdir()
    bundle = recipe.author_network(cfg, condition)
    write_json_atomic(directory / "graph.json", bundle.graph)
    write_json_atomic(directory / "training.json", bundle.training)
    initial_result = build(ExecutionSpec(kind="build", graph=bundle.graph, training=bundle.training,
                                       seed=cfg["seed"], device="cpu"))
    initial = {name: value.detach().numpy().copy() for name, value in initial_result.parameters.items()}
    del initial_result
    np.savez_compressed(directory / "initial_weights.npz", **initial)

    def binding(split):
        return DatasetSnapshotBinding(
            path=run.export / "spikes.npz", input_id="drive", dataset_id="shd", split=split,
            target_id="class", label_key=f"{split}_labels",
            encoder=DatasetEncoder(kind="custom", definition="exp124.shd_events/v1",
                seed=cfg["seed"] + (2 if split == "train" else 4), config={
                    "duration_ms": cfg["duration_ms"], "sample_offset": 0 if split == "train" else cfg["train_examples"],
                    "input_noise_hz": cfg["input_noise_hz"],
                    "spike_deletion_probability": cfg["spike_deletion_probability"]}))

    online = cfg["training_measurement"] == "online_pre_update_minibatches"

    def progress(report):
        print(f"{condition} epoch {report.epoch}: train loss={next(iter(report.train.cross_entropies.values())):.4f}, accuracy={report.train.accuracy:.1%}; "
              f"validation loss={next(iter(report.validation.cross_entropies.values())):.4f}, accuracy={report.validation.accuracy:.1%}; "
              f"timing={dict(report.timing)}", flush=True)

    result = train(ExecutionSpec(
        kind="train", graph=bundle.graph, training=bundle.training,
        input_bindings=(binding("train"),), validation=ValidationSpec(input_bindings=(binding("validation"),)),
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], seed=cfg["seed"], device=str(device),
        metrics=TrainingMetrics(train="online" if online else "evaluate", validation="epoch",
                                initial=("validation",) if online else ("train", "validation"), update_history=False),
        batching=Batching(order="stratified" if cfg["class_stratified_batches"] or cfg["balanced_batches"] else "shuffle",
                         stratify_by="class" if cfg["class_stratified_batches"] or cfg["balanced_batches"] else None,
                         remainder="merge_small" if cfg["merge_short_training_tail"] else "keep",
                         min_batch_size=min(cfg["classes"], cfg["batch_size"]) if cfg["merge_short_training_tail"] else None),
        observations=EpochObservations(parameter_norms=recipe.TRAINABLE),
        checkpoint_selection=CheckpointSelection.disabled(), save_final_checkpoint=directory,
        diagnostics=False, compile_mode=cfg["compile_mode"], projection_mode=cfg["projection_mode"],
        extension_validation=cfg["extension_validation"], dataset_prefetch=1,
        training_timing=True, timing_mode="synchronized", on_epoch_end=progress))
    final = {name: value.detach().cpu().numpy().copy() for name, value in result.training_checkpoint.parameters.items()}
    if any(not np.isfinite(value).all() for value in final.values()):
        raise ValueError("nonfinite learned weights")
    np.savez_compressed(directory / "final_weights.npz", **final)
    deltas = {name: float(np.linalg.norm(final[name] - initial[name])) for name in recipe.TRAINABLE}
    history = []
    for row in result.metrics["epochs"]:
        converted = {"epoch": row["epoch"], "updates": [],
                     "gradient_l2_mean": {name: None for name in recipe.TRAINABLE}}
        for split in ("train", "validation"):
            rates = row.get("observations", {}).get(split, {}).get("population_rates_hz", {})
            converted[split] = {"loss": next(iter(row.get(f"{split}_cross_entropies", {}).values()), None),
                                "optimization_loss": row.get(f"{split}_loss"),
                                "accuracy": row.get(f"{split}_accuracy"),
                                "measurement": row.get(f"{split}_measurement"),
                                **{f"{name.lower()}_rate_hz": value for name, value in rates.items()}}
            for label in ("e", "i"):
                if f"{label}2_rate_hz" in converted[split]:
                    converted[split][f"{label}_rate_hz"] = converted[split][f"{label}2_rate_hz"]
        history.append(converted)
    write_json_atomic(directory / "learning.json", {"epochs": history, "weight_change_l2": deltas})
    write_json_atomic(directory / "native_metrics.json", result.metrics)
    timing = result.metrics["training_timing"]
    write_json_atomic(directory / "timing.json", timing)
    run.record["execution"].setdefault("models", {})[condition] = {
        "duration_seconds": timing["total_s"] - timing["setup_s"],
        "duration_scope": "native timing total less setup; includes evaluations, reporting and checkpoint persistence",
        "epoch_seconds": [sum(value for key, value in row.items() if key.endswith("_s"))
                          for row in timing["epochs"]],
        "native_timing": timing, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "all_gradients_finite": None, "hidden_weights_changed": deltas[recipe.TRAINABLE[0]] > 0,
    }


def calibrate(source_identity, device):
    """Evaluate initial operating points using one training utterance per class."""
    source = source_run(REPO / ".pingstore", source_identity, stage="compute", experiment=recipe.SLUG)
    cfg = recipe.configuration(20, 20, 1, variant="paper", compile_mode="eager")
    cfg["calibration_cases"] = list(recipe.CALIBRATION_CASES)
    cfg["phase_definition"] = recipe.PHASES["operating-point"]
    cfg["calibration_selection"] = "first official-training utterance per class in explicit source; validation not used"
    with np.load(source.file("spikes.npz"), allow_pickle=False) as data:
        labels, split = data["labels"], data["split"]
        selected = [int(np.flatnonzero((labels == label) & (split == 0))[0]) for label in range(20)]
        events = []
        for index in selected:
            mask = data["event_sample"] == index
            times, channels = data["event_time_ms"][mask], data["event_channel"][mask]
            coordinates = np.unique(np.floor(times / cfg["dt_ms"]).astype(np.int64) * cfg["n_input"] + channels)
            events.append((coordinates // cfg["n_input"], coordinates % cfg["n_input"]))
        selected_labels = labels[selected]
        official_indices = data["official_train_indices"][selected].tolist()
    rows = []
    with stage_run(REPO, recipe.SLUG, "compute", inputs={"training_examples": source},
                   phase="operating-point", configuration=cfg) as run:
        for case in recipe.CALIBRATION_CASES:
            candidate = dict(cfg, input_weight_mean=case["input"], input_weight_sd=case["input"] / 10,
                             interlayer_weight_mean=case["interlayer"], interlayer_weight_sd=case["interlayer"] / 10,
                             ei_summed_weight_us=case["ei"], ie_summed_weight_us=case["ie"])
            for condition in recipe.MODELS:
                bundle = recipe.author_network(candidate, condition)
                model = GraphExecutor(plan_graph(bundle.graph), seed=cfg["seed"], trainable_parameters=recipe.TRAINABLE,
                                      surrogate=bundle.training["surrogate"]).to(device)
                initialize(model, candidate, condition)
                measured = evaluate(model, events, selected_labels, np.arange(20), candidate, device,
                                    np.random.default_rng(cfg["seed"] + 3), condition)
                rows.append({"case": case, "model": condition, "measurements": measured})
                print(condition, case, {key: value for key, value in measured.items() if "rate_hz" in key}, flush=True)
        write_json_atomic(run.export / "operating_points.json", {"official_train_indices": official_indices, "cases": rows})
    return run.run_id


def compute(
    run_id=None, device=None, cache=None, cpu_threads=None,
    train_examples=200, validation_examples=100, epochs=5, variant="calibrated",
    seed=42, batch_size=None, learning_rate=None, readout_gain=None, damping=None,
    readout_learning_rate=None, balanced_batches=None, center_readout=None,
    interlayer_strength=None, normalize_readout=None, input_strength=None, models=None,
    full_dataset=False, validation_fraction=0.1,
    compile_mode=None,
    phase=None, source=None,
):
    if phase == "sample-raster":
        from experiments.exp124.rasters import record
        torch.set_num_threads(cpu_threads or 4)
        return record([source] if isinstance(source, str) else source, torch.device(device or "cpu"),
                      batch_encoder=batch)
    if isinstance(source, list):
        if len(source) != 1:
            raise ValueError("this phase requires exactly one source")
        source = source[0]
    if phase == "gamma-probe":
        if not source:
            raise ValueError("gamma-probe requires an explicit completed source")
        from experiments.exp124.gamma import record
        torch.set_num_threads(cpu_threads or 4)
        return record(source, torch.device(device or "cpu"))
    if phase == "operating-point":
        if not source:
            raise ValueError("operating-point phase requires an explicit source compute run")
        torch.set_num_threads(cpu_threads or 4)
        return calibrate(source, torch.device(device or "cpu"))
    if source:
        raise ValueError("source is supported only for operating-point calibration")
    cfg = recipe.configuration(
        train_examples, validation_examples, epochs, variant=variant, seed=seed,
        batch_size=batch_size, learning_rate=learning_rate, readout_gain=readout_gain,
        damping=damping, readout_learning_rate=readout_learning_rate,
        balanced_batches=balanced_batches, center_readout=center_readout,
        interlayer_strength=interlayer_strength, normalize_readout=normalize_readout,
        input_strength=input_strength, models=models, full_dataset=full_dataset,
        validation_fraction=validation_fraction,
        compile_mode=compile_mode,
    )
    device = torch.device(resolve_device(device or os.environ.get("PINGLAB_DEVICE", "cpu")))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda" and compile_mode is None:
        cfg["compile_mode"] = None
    cpu_threads = int(cpu_threads if cpu_threads is not None else os.environ.get("PINGLAB_CPU_THREADS", "4"))
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(cpu_threads)
    cache = Path(cache or os.environ.get("SHD_CACHE", "/tmp/shd"))
    events, labels, arrays, metadata = load_inputs(cfg, cache)
    with stage_run(REPO, recipe.SLUG, "compute", run_id=run_id, configuration=cfg) as run:
        arrays.update(train_labels=labels[:cfg["train_examples"]], validation_labels=labels[cfg["train_examples"]:],
                      event_offsets=np.concatenate(([0], np.cumsum(np.bincount(arrays["event_sample"], minlength=len(labels))))))
        metadata["encoding_seed_policy"] = cfg["encoding_seed_policy"]
        np.savez_compressed(run.export / "spikes.npz", **arrays)
        run.record["execution"].update(dataset=metadata, snnlab_version=version("snnlab"),
                                       torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
                                       host=platform.node(), cpu_architecture=platform.machine(),
                                       modal_cpu_cores=os.environ.get("PINGLAB_MODAL_CPU_CORES"),
                                       executor="snnlab.sim.train")
        if Path("/proc/cpuinfo").exists():
            run.record["execution"]["cpu_model"] = next(
                (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("model name")), "unknown")
        for condition in cfg["models"]:
            train_condition(condition, run, cfg, device)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--phase", choices=("operating-point", "gamma-probe", "sample-raster"))
    parser.add_argument("--source", action="append")
    parser.add_argument("--device")
    parser.add_argument("--cache")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--compile-mode", choices=("default", "eager"), help="Paper profile compiles CUDA neuron kernels by default; eager opts out")
    parser.add_argument("--train-examples", type=int, default=200)
    parser.add_argument("--validation-examples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--variant", choices=("baseline", "diverse", "damping", "sum", "gain", "combined", "maximum", "tuned", "paper", "calibrated"), default="calibrated")
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
