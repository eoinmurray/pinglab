"""Freeze trained weights and compute paired-input parameter sweeps."""

import json
from time import perf_counter

import numpy as np
import torch
from experiments.exp123 import recipe
from experiments.helpers.checkpoints import pinned_checkpoint
from experiments.helpers.network_artifacts import model_execution
from pingstore.stages import source_run, stage_run
from snnlab.sim import ExecutionSpec, build, encode_images_poisson

PARAMETERS = {
    "input_to_E.weight": "W_ff.0",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
    "readout_projection.weight": "W_ff.1",
}


def graph_trial(tau, weights, encoded, artifacts):
    bundle = recipe.network(tau)
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
        with model_execution(model, bundle, artifacts, kind="infer", settings={"tau_gaba_ms": tau}):
            result = model({"image": encoded}, diagnostics=True)
    for name, target in model.parameter_map().items():
        if not torch.equal(target, weights[PARAMETERS[name]]):
            raise ValueError("frozen weights changed")
    return result


def compute_overview(source_id, image_source_id):
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
    image_source = source_run(
        recipe.REPO / ".pingstore", image_source_id, stage="compute", experiment=recipe.SLUG
    )
    image_unit = image_source.unit(recipe.recording_unit(0, recipe.RATE_HZ, "trajectories"))
    with np.load(image_unit / "input.npz", allow_pickle=False) as data:
        pixels = torch.from_numpy(data["pixels"].copy()).reshape(1, 784).float()
        label = int(data["label"])
    encoded = torch.cat(
        [
            encode_images_poisson(
                pixels,
                int(recipe.DURATION_MS / recipe.DT_MS),
                recipe.DT_MS,
                trained["input_rate"],
                generator=torch.Generator().manual_seed(seed),
            )
            for seed in recipe.OVERVIEW_SEEDS
        ],
        dim=1,
    )
    hashes = {
        str(seed): __import__("hashlib")
        .sha256(encoded[:, i].numpy().tobytes())
        .hexdigest()
        for i, seed in enumerate(recipe.OVERVIEW_SEEDS)
    }
    cfg = recipe.overview_configuration()
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
        inputs={"trained_network": source, "image": image_source},
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
            seeds=recipe.OVERVIEW_SEEDS,
        )
        for condition in cfg["conditions"]:
            tau = condition["tau_gaba_ms"]
            leak_scale = condition["leak_scale"]
            capacitance_scale = condition["capacitance_scale"]
            start = perf_counter()
            with torch.inference_mode():
                result = graph_trial(
                    tau, weights, encoded, run.export / f"{condition['condition_id']}--inference"
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
                seeds=recipe.OVERVIEW_SEEDS,
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

