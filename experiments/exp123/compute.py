"""Simulate paired encodings for trajectories or the 500-image accuracy probe."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from importlib.metadata import version

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[2] / "tools"),
]
import numpy as np
import torch
from torchvision.datasets import MNIST
from snnlab.sim.encoders import encode_images_poisson
from snnlab.sim.execution import ExecutionSpec, build
from experiments.exp123 import recipe
from experiments.helpers.checkpoints import pinned_checkpoint
from pingstore.stages import source_run, stage_run


def graph_trial(tau, weights, encoded):
    bundle = recipe.network(tau)
    model = build(
        ExecutionSpec(kind="simulate", graph=bundle.graph, device="cpu", seed=0)
    ).model
    with torch.inference_mode():
        for name, target in model.parameter_map().items():
            value = weights[recipe.PARAMETERS[name]]
            if value.shape != target.shape or (value < 0).any():
                raise ValueError("checkpoint projection shape or polarity mismatch")
            target.copy_(value)
        result = model({"image": encoded}, diagnostics=True)
        for name, target in model.parameter_map().items():
            if not torch.equal(target, weights[recipe.PARAMETERS[name]]):
                raise ValueError("frozen weights changed")
    return {
        name: value.cpu().numpy().astype(bool)
        for name, value in result.diagnostics.items()
    }


def compute(identity, protocol="trajectories"):
    torch.set_num_threads(1)
    bank = source_run(
        recipe.REPO / ".pingstore", identity, stage="compute", experiment="exp022"
    )
    unit = bank.unit(recipe.UNIT)
    checkpoint = pinned_checkpoint(
        unit,
        role=recipe.CHECKPOINT_ROLE,
        filename=recipe.CHECKPOINT_FILE,
        sha256=recipe.CHECKPOINT_SHA256,
    )
    config = json.loads((unit / "config.json").read_text())
    expected = dict(
        dt=0.1,
        input_rate=25.0,
        t_ms=200.0,
        tau_gaba_ms=6.0,
        tau_ampa_ms=2.0,
        n_in=784,
        n_hidden=1024,
        n_inh=256,
        n_out=10,
        train_leak=False,
        adaptive_threshold=False,
        input_rate_sampling="fixed",
        v_grad_dampen=1000.0,
        surrogate_slope=1.0,
        state_clamp=False,
        readout_mode="mem-mean",
        signed_readout=False,
        readout_bias=False,
    )
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported checkpoint setting: {key}")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if weights["W_ee.1"].count_nonzero() or weights["W_ii.1"].count_nonzero():
        raise ValueError("same-population recurrence must be absent")
    dataset = MNIST(str(recipe.REPO / "data"), train=False, download=False)
    indices = (
        recipe.IMAGE_INDICES
        if protocol == "trajectories"
        else recipe.accuracy_images(dataset.targets.numpy())
    )
    cfg = recipe.configuration(protocol, indices)
    seeds = cfg["seeds"]
    with stage_run(
        recipe.REPO,
        recipe.SLUG,
        "compute",
        inputs={"trained_network": bank},
        configuration=cfg,
    ) as run:
        run.record["execution"]["software"] = {
            "snnlab": version("snnlab"),
            "torch": torch.__version__,
        }
        hashes = {}
        for rate in cfg["input_rates_hz"]:
            for start in range(0, len(indices), cfg["execution_batch_images"]):
                batch_indices = indices[start : start + cfg["execution_batch_images"]]
                encodings = []
                for image_index in batch_indices:
                    unit = run.export / recipe.recording_unit(
                        image_index, rate, protocol
                    )
                    unit.mkdir()
                    pixels = dataset.data[image_index].reshape(1, 784).float() / 255
                    label = int(dataset.targets[image_index])
                    generator_seeds = [1000 * image_index + seed for seed in seeds]
                    encoded = torch.cat(
                        [
                            encode_images_poisson(
                                pixels,
                                recipe.STEPS,
                                recipe.DT_MS,
                                rate,
                                generator=torch.Generator().manual_seed(seed),
                            )
                            for seed in generator_seeds
                        ],
                        dim=1,
                    )
                    encodings.append(encoded)
                    hashes[
                        f"{rate:g}:{image_index}"
                        if protocol == "pareto"
                        else str(image_index)
                    ] = {
                        str(seed): hashlib.sha256(
                            encoded[:, b].numpy().tobytes()
                        ).hexdigest()
                        for b, seed in enumerate(seeds)
                    }
                    np.savez_compressed(
                        unit / "input.npz",
                        pixels=pixels.numpy().reshape(28, 28),
                        label=label,
                        spikes=encoded.numpy().astype(bool),
                        seeds=seeds,
                        generator_seeds=generator_seeds,
                    )
                batch_encoded = torch.cat(encodings, dim=1)
                for tau in recipe.TAUS_MS:
                    signals = graph_trial(tau, weights, batch_encoded)
                    for j, image_index in enumerate(batch_indices):
                        unit = run.export / recipe.recording_unit(
                            image_index, rate, protocol
                        )
                        selection = slice(j * len(seeds), (j + 1) * len(seeds))
                        np.savez_compressed(
                            unit / f"tau{tau:g}--spikes.npz",
                            **{
                                key: value[:, selection]
                                for key, value in signals.items()
                            },
                            dt_ms=recipe.DT_MS,
                            tau_gaba_ms=tau,
                            image_index=image_index,
                            label=int(dataset.targets[image_index]),
                            seeds=seeds,
                        )
                print(
                    f"rate {rate:g} Hz: {start + len(batch_indices)}/{len(indices)} images completed "
                    f"({len(seeds) * len(recipe.TAUS_MS)} paired presentations per image)",
                    flush=True,
                )
        if recipe.sha256(checkpoint) != recipe.CHECKPOINT_SHA256:
            raise ValueError("checkpoint changed")
        run.record["execution"]["encoding_sha256"] = hashes
        run.record["execution"]["observations"] = {
            "presentations": len(indices)
            * len(seeds)
            * len(recipe.TAUS_MS)
            * len(cfg["input_rates_hz"]),
            "weights_frozen": True,
            "inputs_paired_across_tau": True,
        }
        result = run.run_id
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--protocol",
        choices=("trajectories", "accuracy", "pareto"),
        default="trajectories",
    )
    args = parser.parse_args()
    compute(args.source, args.protocol)
