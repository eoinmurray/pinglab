"""Record explicit digit streams from a pinned bank; never analyse or publish."""

import argparse
import contextlib
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp048 import evidence, inputs, recipe, stimuli
from experiments.helpers.datasets import load_mnist_split
from experiments.helpers.run_cli import run_cli
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def invoke(args, attachments):
    attachments.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        attachments / "command.json", {"arguments": [str(a) for a in args]}
    )
    with (
        (attachments / "stdout.log").open("w") as stdout,
        (attachments / "stderr.log").open("w") as stderr,
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        run_cli(args, no_sync=True)


def select(job, x, y, rng):
    if job["kind"] in ("headline", "varying"):
        return stimuli.pick_diverse_digits(
            x, y, len(job["segments"]), job["sample_seed"]
        )
    idx = rng.choice(len(y), len(job["segments"]), replace=False)
    return x[idx], y[idx]


def compute(identity, *, run_id=None):
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    contract = evidence.training_contract(bank.export)
    with inputs.execution(
        REPO,
        "compute",
        sources={"bank": bank},
        run_id=run_id,
        configuration=recipe.configuration(),
    ) as run:
        datasets = {}
        for seed in recipe.SEEDS:
            name = recipe.cell_name(seed)
            train = bank.unit(name)
            cfg = contract["configs"][name]
            _, x, _, y = load_mnist_split(max_samples=int(cfg["max_samples"]))
            datasets[seed] = (x, y)
            attachments = run.scratch / "readouts" / name
            with tempfile.TemporaryDirectory(
                prefix=".readout-", dir=run.directory
            ) as temp:
                output = Path(temp) / "output"
                args = [
                    "dump-weights",
                    "--load-config",
                    str(train / "config.json"),
                    "--load-weights",
                    str(train / "weights.pth"),
                    "--out-dir",
                    str(output),
                ]
                invoke(args, attachments)
                dump = evidence.load_arrays(output / "weights_dump.npz")
                keys = sorted(
                    (
                        k
                        for k in dump
                        if k.startswith("W_ff_") and k.endswith("_trained")
                    ),
                    key=lambda k: int(k.split("_")[2]),
                )
                if not keys:
                    raise PingstoreError("missing trained readout matrix")
                w = dump[keys[-1]]
                evidence.readout(w)
                np.savez_compressed(run.export / f"readout-{seed}.npz", W_out=w)
                for path in output.iterdir():
                    if path.name != "weights_dump.npz":
                        shutil.copy2(path, attachments / path.name)
        rngs = {}
        for job in recipe.jobs():
            print(f"[record] {job['id']} {job['kind']} seed={job['seed']}", flush=True)
            rng = rngs.setdefault(
                job["sample_group"], np.random.default_rng(job["sample_seed"])
            )
            x, y = datasets[job["seed"]]
            train = bank.unit(recipe.cell_name(job["seed"]))
            for index in range(job["streams"]):
                destination = run.export / job["id"] / f"stream-{index:03d}"
                destination.mkdir(parents=True)
                attachments = run.scratch / job["id"] / f"stream-{index:03d}"
                pixels, labels = select(job, x, y, rng)
                generator = torch.Generator().manual_seed(job["poisson_seed"] + index)
                spike_input = stimuli.encode_varying_stream(
                    pixels, job["segments"], generator
                )
                arr = spike_input.numpy()
                tt, _, cc = np.nonzero(arr)
                np.savez_compressed(
                    destination / "stimulus.npz",
                    pixels=pixels,
                    labels=labels,
                    input_t=tt,
                    input_cell=cc,
                )
                evidence.stimulus(destination / "stimulus.npz", job)
                write_json_atomic(
                    destination / "stream.json",
                    {
                        "job": job,
                        "stream": index,
                        "poisson_seed": job["poisson_seed"] + index,
                        "input_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                        "input_dtype": str(arr.dtype),
                        "input_shape": list(arr.shape),
                    },
                )
                with tempfile.TemporaryDirectory(
                    prefix=".stream-", dir=run.directory
                ) as temp:
                    scratch = Path(temp)
                    np.savez(scratch / "input.npz", input_spikes=arr)
                    output = scratch / "output"
                    args = [
                        "sim",
                        "--load-config",
                        str(train / "config.json"),
                        "--load-weights",
                        str(train / "weights.pth"),
                        "--n-in",
                        str(recipe.N_IN),
                        "--input-file",
                        str(scratch / "input.npz"),
                        "--outputs",
                        "rasters",
                        "--out-dir",
                        str(output),
                    ]
                    invoke(args, attachments)
                    evidence.simulation_configuration(
                        load_json(output / "config.json"),
                        contract["configs"][recipe.cell_name(job["seed"])],
                        args,
                    )
                    data = evidence.recording(output / "rasters.npz", job)
                    np.savez_compressed(destination / "rasters.npz", **data)
                    for path in output.iterdir():
                        if path.name != "rasters.npz":
                            shutil.copy2(path, attachments / path.name)
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp048.compute/v1",
                "recipe": recipe.configuration(),
                "training_contract": contract,
                "jobs": recipe.jobs(),
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        compute(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp048 compute: {exc}\n")


if __name__ == "__main__":
    main()
