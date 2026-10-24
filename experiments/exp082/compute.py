"""Compute exp082 streaming inference from a pinned v4 bank, with six shards.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
import hashlib
import os

import numpy as np
import torch
from experiments.exp082 import recipe
from experiments.exp082.custom import evidence, inputs
from experiments.exp082.custom.datasets import load_mnist_split
from experiments.exp082.custom.network_artifacts import model_execution
from experiments.exp082.custom.sharding import shard_identity
from experiments.exp082.custom.source_exp022_custom_checkpoints import (
    resolve_checkpoint,
)
from experiments.exp082.recipe import DT_MS, N_CLASSES, N_INPUT
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from pingstore.stages import _capture_code, reserve_stage
from snnlab.sim import (
    BoundarySchedule,
    DecisionSegments,
    GraphExecutor,
    RecordingSpec,
    ResetVoltage,
    SignalRecording,
    plan_graph,
    resolve_device,
)
from tools.hpc import concurrent_compute


def _run_jobs(bank, directory, jobs, contract):
    cfg = recipe.environment_configuration()
    worker = Inference(bank, directory, cfg)
    for job in jobs:
        if (directory / "export" / job["path"]).exists() or (
            directory / ".scratch/simulations" / job["path"]
        ).exists():
            raise PingstoreError(
                "incomplete job exists; explicit recovery or a fresh run required"
            )
        worker.condition(job)
        write_json_atomic(
            directory / ".scratch/simulations" / job["path"] / "dataset.json",
            worker.dataset,
        )
        retained = evidence.counts(
            directory / "export" / job["path"] / "counts.npz", cfg
        )
        if not (retained["labels"] == worker.image_stream_labels).all():
            raise PingstoreError("condition labels differ from shared image bank")


def _job_inventory(directory, jobs, *, verify_device=True):
    files = {}
    for job in jobs:
        for prefix in ("export", ".scratch/simulations"):
            folder = directory / prefix / job["path"]
            if not folder.is_dir() or folder.is_symlink():
                raise PingstoreError("missing or linked job evidence")
            for path in folder.rglob("*"):
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    raise PingstoreError("unsupported job evidence entry")
                if path.is_file():
                    files[str(path.relative_to(directory))] = file_sha256(path)
    if verify_device:
        from snnlab.sim import resolve_device

        expected_device = resolve_device(os.environ.get("PINGLAB_DEVICE", "auto"))
        for job in jobs:
            folder = directory / ".scratch/simulations" / job["path"]
            requests = list(folder.rglob("request.json"))
            if not requests or any(
                load_json(path).get("device") != expected_device for path in requests
            ):
                raise PingstoreError(
                    "completed shard used a different resolved execution device"
                )
    return files


def _shard_paths(repo, run_id, index, count):
    return concurrent_compute.working_directory(
        repo,
        run_id,
        index,
        count,
        experiment=recipe.SLUG,
        expected_count=recipe.SHARDS,
    )


_compute_lock = concurrent_compute.compute_lock


def shard(identity, *, run_id, index, count=recipe.SHARDS):
    """Compute-only recovery: each shard owns an isolated lock and completion record."""
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    contract = evidence.training_contract(bank.export)
    cfg = recipe.environment_configuration()
    directory = _shard_paths(REPO, run_id, index, count)
    job_list = recipe.jobs(cfg)[index::count]

    def check_inputs():
        for ancestor in inputs.lineage(REPO, identity, bank.reference).values():
            ancestor.check_unchanged()

    return concurrent_compute.execute_shard(
        repo=REPO,
        experiment=recipe.SLUG,
        run_id=run_id,
        index=index,
        count=count,
        expected_count=recipe.SHARDS,
        inputs={"bank": bank.reference},
        configuration=shard_identity(cfg, bank, recipe),
        items=job_list,
        run_items=lambda: _run_jobs(bank, directory, job_list, contract),
        inventory=lambda: _job_inventory(directory, job_list),
        check_inputs=check_inputs,
        capture_code=_capture_code,
    )


def compute(identity, *, run_id=None, collect=False):
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    contract = evidence.training_contract(bank.export)
    cfg = recipe.environment_configuration()
    if collect and not run_id:
        raise PingstoreError("collection requires an explicit compute reservation")
    run_id = run_id or reserve_stage(REPO / ".pingstore", recipe.SLUG, "compute")
    with concurrent_compute.collect_shards(
        repo=REPO,
        experiment=recipe.SLUG,
        run_id=run_id,
        count=recipe.SHARDS,
        inputs={"bank": bank.reference},
        configuration=shard_identity(cfg, bank, recipe),
        items_for=lambda index: recipe.jobs(cfg)[index :: recipe.SHARDS],
        inventory_for=lambda index: _job_inventory(
            _shard_paths(REPO, run_id, index, recipe.SHARDS),
            recipe.jobs(cfg)[index :: recipe.SHARDS],
            verify_device=False,
        ),
        collect=collect,
    ) as (_directory, shard_records):
        with inputs.execution(
            REPO, "compute", sources={"bank": bank}, run_id=run_id, configuration=cfg
        ) as run:
            if collect:
                concurrent_compute.retain_worker_provenance(run, shard_records)
            environment = {
                "PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0",
                **{
                    k: os.environ[k]
                    for k in (
                        "PINGLAB_EXP082_STREAMS_PER_CELL",
                        "PINGLAB_EXP082_DIGITS_PER_STREAM",
                        "PINGLAB_EXP082_STREAM_BATCH_SIZE",
                    )
                    if k in os.environ
                },
            }
            run.record["execution"]["environment"] = environment
            if not collect:
                _run_jobs(bank, run.directory, recipe.jobs(cfg), contract)
            worker = Inference(bank, run.directory, cfg)
            for job in recipe.jobs(cfg):
                if (
                    load_json(
                        run.scratch / "simulations" / job["path"] / "dataset.json"
                    )
                    != worker.dataset
                ):
                    raise PingstoreError(
                        "inference workers used different dataset bytes"
                    )
            for name in ("matched", "variable"):
                worker.stream(name)
            write_json_atomic(run.scratch / "dataset.json", worker.dataset)
            from experiments.exp082.custom import illustrate

            showcase = illustrate.write_showcase(
                bank, run, contract, inference_configuration=cfg
            )
            run.record["showcase_configuration"] = showcase["configuration"]
            write_json_atomic(
                run.export / "evidence.json",
                {
                    "schema": "exp082.compute/v3",
                    "recipe": cfg,
                    "training_contract": contract,
                    "jobs": recipe.jobs(cfg),
                    "image_stream_bank": worker.image_stream_bank,
                    "showcase": showcase,
                },
            )
            evidence.validate_compute(run.export, cfg)
            evidence.validate_showcase(run.export)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="explicit completed exp022 compute ID"
    )
    parser.add_argument("--run-id", help="unused v4 reservation")
    parser.add_argument(
        "--shard-index", type=int, help="compute worker index (six shards)"
    )
    args = parser.parse_args()
    try:
        if args.shard_index is not None:
            if not args.run_id:
                raise PingstoreError("shard workers require --run-id")
            shard(args.source, run_id=args.run_id, index=args.shard_index)
        else:
            compute(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp082 compute: {exc}\n")


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


def encode_segment(
    pixels: np.ndarray,
    duration_ms: float,
    rate_hz: float,
    generator: torch.Generator,
) -> torch.Tensor:
    steps = int(round(duration_ms / DT_MS))
    images = torch.as_tensor(pixels, dtype=torch.float32).reshape(-1, N_INPUT)
    probability = images * rate_hz * DT_MS / 1000.0
    return (
        torch.rand(steps, len(images), N_INPUT, generator=generator)
        < probability.unsqueeze(0)
    ).to(torch.float32)


def encode_stream(
    pixels: np.ndarray,
    conditions: tuple[tuple[float, float], ...],
    generator: torch.Generator,
) -> torch.Tensor:
    if len(pixels) != len(conditions):
        raise ValueError("one (duration, rate) condition is required per digit")
    return torch.cat(
        [
            encode_segment(pixels[i : i + 1], duration, rate, generator)
            for i, (duration, rate) in enumerate(conditions)
        ],
        dim=0,
    )


def pick_digits(
    x_test: np.ndarray, y_test: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = rng.permutation(N_CLASSES)[:n]
    indices = [int(rng.choice(np.flatnonzero(y_test == label))) for label in classes]
    return x_test[indices], y_test[indices]


def shared_image_stream_indices(n_images: int, cfg: dict) -> np.ndarray:
    """Select one image-stream bank shared by every quantitative condition."""
    if n_images < cfg["digits_per_stream"]:
        raise ValueError("image bank is smaller than one exp082 stream")
    rng = np.random.default_rng(cfg["image_sampling_seed"])
    return np.stack(
        [
            rng.choice(n_images, cfg["digits_per_stream"], replace=False)
            for _ in range(cfg["streams_per_cell"])
        ]
    )


def array_record(array):
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }


class Inference:
    def __init__(self, bank, directory, cfg):
        self.bank, self.directory, self.cfg = bank, directory, cfg
        _, self.images, _, self.labels = load_mnist_split(max_samples=7000)
        self.image_stream_indices = shared_image_stream_indices(len(self.labels), cfg)
        self.image_stream_labels = self.labels[self.image_stream_indices]
        self.dataset = {
            "partition": "official_mnist_test",
            "images": array_record(self.images),
            "labels": array_record(self.labels),
        }
        self.image_stream_bank = {
            "policy": cfg["image_stream_policy"],
            "sampling_seed": cfg["image_sampling_seed"],
            "indices": self.image_stream_indices.tolist(),
            "labels": self.image_stream_labels.tolist(),
            "dataset": self.dataset,
        }

    def simulate(self, train, spikes, resets, attachments, output_kind):
        attachments.mkdir(parents=True)
        training = load_json(train / "config.json")
        checkpoint = resolve_checkpoint(train, recipe.CHECKPOINT_ROLE)
        state = checkpoint_tensors(checkpoint["path"], training)
        bundle = recipe.author_network(training)
        device = resolve_device(os.environ.get("PINGLAB_DEVICE", "auto"))
        model = bind_model(bundle, training, state, device=device)
        schedule = BoundarySchedule(tuple(map(int, resets)))
        segments = DecisionSegments(
            schedule,
            len(spikes),
            population_totals=("E", "I"),
            output_spike_counts=("readout",),
        )
        recording = (
            RecordingSpec(
                tuple(
                    SignalRecording(f"{label}.spikes", kind="spike_events")
                    for label in ("E", "I", "readout")
                )
            )
            if output_kind == "rasters"
            else None
        )
        with model_execution(
            model,
            bundle,
            self.directory
            / "export"
            / attachments.relative_to(self.directory / ".scratch/simulations")
            / "inference",
            kind="infer",
            seed=training["seed"],
            settings={
                "recipe": self.cfg,
                "reset_steps": list(map(int, resets)),
                "recording": output_kind,
                "input_array": array_record(spikes.cpu().numpy()),
            },
        ):
            with torch.inference_mode():
                result = model(
                    {"drive": spikes.to(device)},
                    diagnostics=False,
                    recording=recording,
                    resets=(ResetVoltage("readout", schedule),),
                    decisions=segments,
                )
        starts, stops = list(resets), [*list(resets)[1:], len(spikes)]
        raw = {
            "dt": np.float32(DT_MS),
            "T": np.int32(len(spikes)),
            "n_trials": np.int32(spikes.shape[1]),
            "n_e": np.int32(training["n_hidden"]),
            "n_i": np.int32(training["n_inh"]),
        }
        if output_kind == "spike_summary":
            raw.update(
                segment_starts=np.array(starts, dtype=np.int64),
                segment_stops=np.array(stops, dtype=np.int64),
                e_counts=np.zeros((spikes.shape[1], len(resets)), dtype=np.int64),
                i_counts=np.zeros((spikes.shape[1], len(resets)), dtype=np.int64),
                out_counts=np.zeros(
                    (spikes.shape[1], len(resets), N_CLASSES), dtype=np.int64
                ),
            )
            seen = set()
            for row in result.decisions:
                index = starts.index(row["start_step"])
                batch = row["batch"]
                if row["end_step"] != stops[index] or (batch, index) in seen:
                    raise PingstoreError(
                        "decision coverage differs from requested boundaries"
                    )
                seen.add((batch, index))
                raw["e_counts"][batch, index] = row["population_totals"]["E"]
                raw["i_counts"][batch, index] = row["population_totals"]["I"]
                raw["out_counts"][batch, index] = row["output_spike_counts"]["readout"]
            if len(seen) != spikes.shape[1] * len(resets):
                raise PingstoreError("missing decision segment")
        else:
            for prefix, label in (("e", "E"), ("i", "I"), ("out", "readout")):
                coordinates = result.recorded_signals[f"{label}.spikes"].cpu().numpy()
                for field, axis in (("t", 0), ("trial", 1), ("cell", 2)):
                    raw[f"{prefix}_{field}"] = coordinates[:, axis].astype(np.int32)
        write_json_atomic(
            attachments / "request.json",
            {
                "recipe": self.cfg,
                "training": training,
                "checkpoint": {k: v for k, v in checkpoint.items() if k != "path"},
                "input_array": array_record(spikes.cpu().numpy()),
                "readout_reset_steps": list(map(int, resets)),
                "reset_policy": "readout voltage only; preserve output synapse, hidden state and delays",
                "recording": output_kind,
                "dataset": self.dataset,
                "graph_digest": bundle.manifest["graph_digest"],
                "device": device,
            },
        )
        if file_sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise PingstoreError("checkpoint changed during streaming inference")
        return raw

    def condition(self, job):
        cfg = self.cfg
        conditions = tuple(
            (job["duration_ms"], job["rate_hz"])
            for _ in range(cfg["digits_per_stream"])
        )
        resets = tuple(
            i * int(round(job["duration_ms"] / DT_MS))
            for i in range(cfg["digits_per_stream"])
        )
        values = {k: [] for k in ("out_counts", "e_counts", "i_counts", "labels")}
        pixels, labels = [], []
        for index in range(cfg["streams_per_cell"]):
            ids = self.image_stream_indices[index]
            pixels.append(
                encode_stream(
                    self.images[ids],
                    conditions,
                    torch.Generator().manual_seed(recipe.encoding_seed(job, index)),
                )
            )
            labels.append(self.labels[ids])
            if (
                len(pixels) == cfg["stream_batch_size"]
                or index == cfg["streams_per_cell"] - 1
            ):
                raw = self.simulate(
                    self.bank.export / job["cell_name"],
                    torch.cat(pixels, dim=1),
                    resets,
                    self.directory
                    / ".scratch/simulations"
                    / job["path"]
                    / f"batch-{index + 1 - len(pixels):03d}",
                    "spike_summary",
                )
                for k in ("out_counts", "e_counts", "i_counts"):
                    values[k].append(np.asarray(raw[k], dtype=np.int64))
                values["labels"].append(np.stack(labels))
                pixels, labels = [], []
        output = self.directory / "export" / job["path"]
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / "counts.npz", **{k: np.concatenate(v) for k, v in values.items()}
        )

    def stream(self, name):
        conditions = (
            tuple((200.0, 5.0) for _ in range(5))
            if name == "matched"
            else recipe.VARIABLE_STREAM
        )
        seed = 82 if name == "matched" else 83
        pixels, labels = pick_digits(self.images, self.labels, len(conditions), seed)
        spikes = encode_stream(
            pixels, conditions, torch.Generator().manual_seed(seed + 1)
        )
        boundaries = np.cumsum(
            [0, *[int(round(d / DT_MS)) for d, _ in conditions]]
        ).tolist()
        train = self.bank.export / recipe.training_cell_name(recipe.SEEDS[0])
        raw = self.simulate(
            train,
            spikes,
            tuple(boundaries[:-1]),
            self.directory / ".scratch/simulations/streams" / name,
            "rasters",
        )
        arrays: dict = {"pixels": pixels}
        for prefix, key, width in (
            ("e", "spikes_e", 1024),
            ("i", "spikes_i", 256),
            ("out", "spikes_out", 10),
        ):
            keep = raw[prefix + "_trial"] == 0
            dense = np.zeros((len(spikes), width), dtype=np.int8)
            dense[raw[prefix + "_t"][keep], raw[prefix + "_cell"][keep]] = 1
            arrays[key] = dense
        output = self.directory / "export/streams" / name
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "recording.npz", **arrays)
        write_json_atomic(
            output / "stream.json",
            {
                "labels": labels.tolist(),
                "boundaries": boundaries,
                "conditions": [list(c) for c in conditions],
            },
        )


if __name__ == "__main__":
    main()
