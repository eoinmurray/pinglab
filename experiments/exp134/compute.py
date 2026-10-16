"""Generate paired frozen-network recordings for leak and capacitance sweeps."""

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
import torch
from experiments.exp134 import recipe
from experiments.helpers.checkpoints import pinned_checkpoint
from experiments.helpers.network_artifacts import model_execution
from pingstore.stages import source_run, stage_run
from snnlab.sim import ExecutionSpec, build, encode_images_poisson


def training_input(source):
    unit = source.unit(recipe.UNIT)
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
    return weights, checkpoint


def image_bank(source):
    definition = source.record["execution"]["configuration"]
    if definition.get("schema") != "exp123.compute/v3":
        raise ValueError("expected exp123's balanced shared sweep")
    cfg = definition["sweep"]
    if cfg["partition"] != "official_mnist_test" or cfg["image_selection_seed"] != 123:
        raise ValueError("incompatible image source")
    images = {}
    for index in cfg["image_indices"]:
        with np.load(
            source.unit(f"image{index:05d}-rate25") / "input.npz", allow_pickle=False
        ) as data:
            pixels, label = data["pixels"].copy(), int(data["label"])
        if (
            pixels.shape != (28, 28)
            or not np.isfinite(pixels).all()
            or pixels.min() < 0
            or pixels.max() > 1
            or not 0 <= label < 10
        ):
            raise ValueError("invalid normalized image")
        images[index] = (pixels, label)
    if len(images) != 500 or not np.array_equal(
        np.bincount([v[1] for v in images.values()], minlength=10), [50] * 10
    ):
        raise ValueError("expected 500 distinct images, 50 per digit")
    return images


def encode(pixels, index):
    pixels = torch.from_numpy(pixels).reshape(1, 784).float()
    return torch.cat(
        [
            encode_images_poisson(
                pixels,
                recipe.STEPS,
                recipe.DT_MS,
                recipe.RATE_HZ,
                generator=torch.Generator().manual_seed(1000 * index + seed),
            )
            for seed in recipe.SEEDS
        ],
        dim=1,
    )


def frozen_model(condition, weights):
    bundle = recipe.network(condition)
    model = build(
        ExecutionSpec(kind="simulate", graph=bundle.graph, device="cpu", seed=0)
    ).model
    with torch.inference_mode():
        for name, target in model.parameter_map().items():
            value = weights[recipe.PARAMETERS[name]]
            if value.shape != target.shape or (value < 0).any():
                raise ValueError("checkpoint shape or polarity mismatch")
            target.copy_(value)
    return bundle, model


def check_weights(model, weights):
    if any(
        not torch.equal(value, weights[recipe.PARAMETERS[name]])
        for name, value in model.parameter_map().items()
    ):
        raise ValueError("frozen weights changed")


def compute(identity, image_identity, *, run_id=None):
    torch.set_num_threads(1)
    bank = source_run(
        REPO / ".pingstore", identity, stage="compute", experiment="exp022"
    )
    image_source = source_run(
        REPO / ".pingstore", image_identity, stage="compute", experiment="exp123"
    )
    weights, checkpoint = training_input(bank)
    images = image_bank(image_source)
    cfg = recipe.configuration(images)
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        inputs={"trained_network": bank, "images": image_source},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        run.record["execution"]["software"] = dict(
            snnlab=version("snnlab"), torch=torch.__version__
        )
        encoded = {}
        hashes = {}
        for index in cfg["image_indices"]:
            pixels, label = images[index]
            spikes = encode(pixels, index)
            encoded[index] = spikes.numpy().astype(bool)
            hashes[str(index)] = {
                str(seed): hashlib.sha256(spikes[:, b].numpy().tobytes()).hexdigest()
                for b, seed in enumerate(recipe.SEEDS)
            }
            # Encodings are generated once, then reused for every intervention.
            unit = run.export / f"image{index:05d}"
            unit.mkdir()
            np.savez_compressed(
                unit / "input.npz",
                pixels=pixels,
                label=label,
                spikes=encoded[index],
                seeds=recipe.SEEDS,
                generator_seeds=[1000 * index + s for s in recipe.SEEDS],
            )
        timings = {}
        for condition in cfg["conditions"]:
            name = condition["condition_id"]
            started = perf_counter()
            bundle, model = frozen_model(condition, weights)
            bundle.write(run.export / f"{name}.bundle")
            # Parameter sweeps share one circuit topology.
            if name == "baseline":
                bundle.visualise(run.export / "network.svg", view="circuit")
            with (
                torch.inference_mode(),
                model_execution(
                    model,
                    bundle,
                    run.export / f"{name}-execution",
                    kind="infer",
                    settings=condition,
                ),
            ):
                for start in range(0, len(cfg["image_indices"]), recipe.BATCH_IMAGES):
                    indices = cfg["image_indices"][start : start + recipe.BATCH_IMAGES]
                    batch = torch.from_numpy(
                        np.concatenate([encoded[i] for i in indices], axis=1)
                    ).float()
                    # A fresh call resets all dynamic state; batch lanes are independent.
                    result = model({"image": batch}, diagnostics=True)
                    signals = {
                        key: value.cpu().numpy().astype(bool)
                        for key, value in result.diagnostics.items()
                    }
                    for j, index in enumerate(indices):
                        selection = slice(
                            j * len(recipe.SEEDS), (j + 1) * len(recipe.SEEDS)
                        )
                        np.savez_compressed(
                            run.export / f"image{index:05d}" / f"{name}--spikes.npz",
                            **{
                                key: value[:, selection]
                                for key, value in signals.items()
                            },
                            dt_ms=recipe.DT_MS,
                            image_index=index,
                            seeds=recipe.SEEDS,
                        )
                    del result, signals, batch
                check_weights(model, weights)
            timings[name] = perf_counter() - started
            print(
                f"{name}: {len(cfg['image_indices']) * len(recipe.SEEDS)} presentations in {timings[name]:.1f}s ({len(timings)}/{len(cfg['conditions'])} conditions)",
                flush=True,
            )
            del model, bundle
        if recipe.sha256(checkpoint) != recipe.CHECKPOINT_SHA256:
            raise ValueError("checkpoint changed")
        run.record["execution"].update(
            encoding_sha256=hashes,
            condition_seconds=timings,
            observations=dict(
                presentations=cfg["presentations"],
                weights_frozen=True,
                paired_across_both_sweeps=True,
            ),
        )
        result = run.run_id
    print(result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--image-source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    compute(args.source, args.image_source, run_id=args.run_id)
