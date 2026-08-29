"""Compute-only inference, retaining counts and exact illustrative recordings."""

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from experiments.helpers.datasets import load_mnist_split
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    write_json_atomic,
)

from . import recipe
from .recipe import DT_MS, N_CLASSES, N_INPUT


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
        self.dataset = {
            "partition": "official_mnist_test",
            "images": array_record(self.images),
            "labels": array_record(self.labels),
        }

    def simulate(self, train, spikes, resets, attachments, output_kind):
        attachments.mkdir(parents=True)
        with tempfile.TemporaryDirectory(
            prefix=".simulation-", dir=self.directory
        ) as tmp:
            scratch = Path(tmp)
            reset = np.zeros(len(spikes), dtype=np.bool_)
            reset[list(resets)] = True
            input_path = scratch / "input.npz"
            np.savez_compressed(
                input_path, input_spikes=spikes.cpu().numpy(), readout_reset=reset
            )
            output = scratch / "output"
            tool = Path(__file__).resolve().parents[2] / "tools/snnsim/tool.py"
            command = [
                sys.executable,
                str(tool),
                "sim",
                "--load-config",
                str(train / "config.json"),
                "--load-weights",
                str(train / "weights.pth"),
                "--device",
                "auto",
                "--n-in",
                str(N_INPUT),
                "--input-file",
                str(input_path),
                "--outputs",
                output_kind,
                "--out-dir",
                str(output),
            ]
            write_json_atomic(
                attachments / "command.json",
                {
                    "command": command,
                    "input_sha256": file_sha256(input_path),
                    "input_array": array_record(spikes.cpu().numpy()),
                    "readout_reset_steps": list(map(int, resets)),
                    "dataset": self.dataset,
                },
            )
            with (
                (attachments / "stdout.log").open("w") as stdout,
                (attachments / "stderr.log").open("w") as stderr,
            ):
                subprocess.run(
                    command,
                    cwd=tool.parents[2],
                    check=True,
                    stdout=stdout,
                    stderr=stderr,
                )
            filename = output_kind + ".npz"
            with np.load(output / filename, allow_pickle=False) as archive:
                raw = {k: archive[k].copy() for k in archive.files}
            if (
                not np.isclose(float(raw["dt"]), DT_MS)
                or int(raw["T"]) != len(spikes)
                or int(raw["n_trials"]) != spikes.shape[1]
            ):
                raise PingstoreError("simulator dimensions or timestep differ")
            if output_kind == "spike_summary":
                starts, stops = list(resets), [*list(resets)[1:], len(spikes)]
                if (
                    raw["segment_starts"].tolist() != starts
                    or raw["segment_stops"].tolist() != stops
                ):
                    raise PingstoreError(
                        "simulator did not preserve decision boundaries"
                    )
                shape = (spikes.shape[1], len(resets))
                for key in ("e_counts", "i_counts", "out_counts"):
                    expected = (*shape, 10) if key == "out_counts" else shape
                    if (
                        raw[key].shape != expected
                        or raw[key].dtype.kind not in "iu"
                        or np.any(raw[key] < 0)
                    ):
                        raise PingstoreError("invalid simulator counts")
            elif int(raw["n_e"]) != 1024 or int(raw["n_i"]) != 256:
                raise PingstoreError("simulator populations differ")
            for path in output.iterdir():
                if path.name != filename:
                    if not path.is_file() or path.is_symlink():
                        raise PingstoreError("unexpected simulator attachment")
                    shutil.copyfile(path, attachments / path.name)
            return raw

    def condition(self, job):
        cfg = self.cfg
        rng = np.random.default_rng(
            82_000
            + job["seed"]
            + int(job["duration_ms"] * 10)
            + int(job["rate_hz"] * 100)
        )
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
            ids = rng.choice(len(self.labels), cfg["digits_per_stream"], replace=False)
            pixels.append(
                encode_stream(
                    self.images[ids],
                    conditions,
                    torch.Generator().manual_seed(82_000 + job["seed"] * 100 + index),
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
                    / "export/evidence/simulations"
                    / job["path"]
                    / f"batch-{index + 1 - len(pixels):03d}",
                    "spike_summary",
                )
                for k in ("out_counts", "e_counts", "i_counts"):
                    values[k].append(np.asarray(raw[k], dtype=np.int64))
                values["labels"].append(np.stack(labels))
                pixels, labels = [], []
        output = self.directory / "export" / job["path"]
        output.mkdir(parents=True)
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
            self.directory / "export/evidence/simulations/streams" / name,
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
        output.mkdir(parents=True)
        np.savez_compressed(output / "recordings.npz", **arrays)
        write_json_atomic(
            output / "stream.json",
            {
                "labels": labels.tolist(),
                "boundaries": boundaries,
                "conditions": [list(c) for c in conditions],
            },
        )
