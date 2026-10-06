"""Freeze trained weights and compute paired-input parameter sweeps."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[2] / "tools"),
]
import numpy as np
import torch
from torchvision.datasets import MNIST
from experiments.exp122 import recipe
from experiments.helpers.checkpoints import pinned_checkpoint
from pingstore.stages import stage_run, source_run
from snnlab.sim.encoders import encode_images_poisson
from snnlab.sim.execution import ExecutionSpec, build

PARAMETERS = {
    "input_to_E.weight": "W_ff.0",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
    "readout_projection.weight": "W_ff.1",
}


def graph_trial(tau, weights, encoded, leak_scale=1.0, capacitance_scale=1.0):
    bundle = recipe.network(tau, leak_scale, capacitance_scale)
    model = build(
        ExecutionSpec(kind="simulate", graph=bundle.graph, device="cpu", seed=0)
    ).model
    with torch.inference_mode():
        for name, target in model.parameter_map().items():
            value = weights[PARAMETERS[name]]
            if target.shape != value.shape:
                raise ValueError("projection shape mismatch")
            target.copy_(
                value.clamp(min=0) if PARAMETERS[name].startswith("W_ff") else value
            )
        result = model({"image": encoded}, diagnostics=True)
    for name, target in model.parameter_map().items():
        if not torch.equal(target, weights[PARAMETERS[name]]):
            raise ValueError("frozen weights changed")
    return result


def compute(source_id):
    torch.set_num_threads(1)
    source = source_run(
        recipe.REPO / ".pingstore", source_id, stage="compute", experiment="exp022"
    )
    unit = source.unit(recipe.UNIT)
    checkpoint = pinned_checkpoint(
        unit,
        role=recipe.CHECKPOINT_ROLE,
        filename=recipe.CHECKPOINT_FILE,
        sha256=recipe.CHECKPOINT_SHA256,
    )
    trained = json.loads((unit / "config.json").read_text())
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if weights["W_ee.1"].count_nonzero() or weights["W_ii.1"].count_nonzero():
        raise ValueError("helper circuit requires zero same-population recurrence")
    if (
        trained["train_leak"]
        or trained["adaptive_threshold"]
        or trained["input_rate_sampling"] != "fixed"
    ):
        raise ValueError("checkpoint is outside the fixed-neuron protocol")
    if trained["dt"] != recipe.DT_MS or trained["tau_ampa_ms"] != 2.0:
        raise ValueError("checkpoint timestep or AMPA mismatch")
    expected = dict(
        input_rate=25.0,
        t_ms=200.0,
        tau_gaba_ms=6.0,
        n_in=784,
        n_hidden=1024,
        n_inh=256,
        n_out=10,
        v_grad_dampen=1000.0,
        surrogate_slope=1.0,
        state_clamp=False,
        readout_mode="mem-mean",
        signed_readout=False,
        readout_bias=False,
    )
    for name, value in expected.items():
        if trained.get(name) != value:
            raise ValueError(f"working checkpoint setting mismatch: {name}")
    dataset = MNIST(str(recipe.REPO / "data"), train=False, download=False)
    pixels = dataset.data[0].reshape(1, 784).float() / 255
    label = int(dataset.targets[0])
    encoded = torch.cat(
        [
            encode_images_poisson(
                pixels,
                int(recipe.DURATION_MS / recipe.DT_MS),
                recipe.DT_MS,
                trained["input_rate"],
                generator=torch.Generator().manual_seed(seed),
            )
            for seed in recipe.SEEDS
        ],
        dim=1,
    )
    hashes = {
        str(seed): __import__("hashlib")
        .sha256(encoded[:, i].numpy().tobytes())
        .hexdigest()
        for i, seed in enumerate(recipe.SEEDS)
    }
    cfg = recipe.configuration()
    cfg.update(
        input_rate_hz=trained["input_rate"],
        training_tau_gaba_ms=trained["tau_gaba_ms"],
        checkpoint_epoch=50,
        image_label=label,
        encoding_sha256=hashes,
        source_reference=source.reference,
        output_lif=dict(
            tau_ms=2.0, threshold=1.0, initial_voltage=0.0, reset="subtract"
        ),
    )
    with stage_run(
        recipe.REPO,
        recipe.SLUG,
        "compute",
        inputs={"trained_network": source},
        configuration=cfg,
    ) as run:
        run.record["execution"]["software"] = dict(
            torch=torch.__version__,
            snnlab=__import__("importlib.metadata", fromlist=["version"]).version(
                "snnlab"
            ),
        )
        np.savez_compressed(
            run.export / "input.npz",
            pixels=pixels.numpy().reshape(28, 28),
            label=label,
            spikes=encoded.numpy().astype(bool),
            seeds=recipe.SEEDS,
        )
        for condition in cfg["conditions"]:
            tau = condition["tau_gaba_ms"]
            leak_scale = condition["leak_scale"]
            capacitance_scale = condition["capacitance_scale"]
            start = perf_counter()
            with torch.inference_mode():
                result = graph_trial(
                    tau, weights, encoded, leak_scale, capacitance_scale
                )
            signals = {
                name: value.cpu().numpy().astype(bool)
                for name, value in result.diagnostics.items()
            }
            e, i = signals["spk_e"], signals["spk_i"]
            out = signals["spk_out"]
            np.savez_compressed(
                run.export / f"{condition['condition_id']}--spikes.npz",
                spk_e=e,
                spk_i=i,
                spk_out=out,
                tau_ms=tau,
                leak_scale=leak_scale,
                capacitance_scale=capacitance_scale,
                dt_ms=recipe.DT_MS,
                seeds=recipe.SEEDS,
            )
            print(
                f"{condition['condition_id']}: tau={tau:.4f} ms, leak ×{leak_scale:.4f}: {perf_counter() - start:.2f}s; E/I/output spikes {e.sum()}/{i.sum()}/{out.sum()}",
                flush=True,
            )
        if recipe.sha256(checkpoint) != recipe.CHECKPOINT_SHA256:
            raise ValueError("checkpoint changed")
        run.record["execution"]["observations"] = dict(
            presentations=len(cfg["conditions"]) * len(cfg["seeds"]),
            encoding_reused_across_tau=True,
            helper_sha256=recipe.sha256(recipe.REPO / "experiments/helpers/ping.py"),
        )
        identity = run.run_id
    print(identity, flush=True)
    return identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    compute(parser.parse_args().source)
