"""Simulate the uniform paired inhibitory-decay sweep in one run.

Invoke this stage independently through its CLI; it records configuration,
provenance and scientific outputs in Pingstore. Importing the module does not
launch computation or another stage.
"""

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[2] / "tools"),
]
import numpy as np
import torch
from experiments.exp123 import recipe
from experiments.exp123.custom.network_artifacts import model_execution
from pingstore.contracts import file_sha256, load_json
from pingstore.stages import source_run, stage_run
from snnlab.sim import Performance, build, encode_images_poisson
from torchvision.datasets import MNIST


def graph_trial(tau, weights, encoded, artifacts, cfg, execution=None):
    bundle = recipe.author_network(tau, cfg)
    built = build(
        graph=bundle.graph,
        device=cfg["device"],
        seed=cfg["model_seed"],
        performance=Performance(**cfg["performance"]),
    )
    model = built.model
    if execution is not None:
        execution["performance"] = built.metrics["performance"]
    with torch.inference_mode():
        for name, target in model.parameter_map().items():
            value = weights[recipe.PARAMETERS[name]]
            if value.shape != target.shape or (value < 0).any():
                raise ValueError("checkpoint projection shape or polarity mismatch")
            target.copy_(value)
        with model_execution(
            model,
            bundle,
            artifacts,
            kind="simulate",
            seed=cfg["model_seed"],
            settings={**cfg, "tau_gaba_ms": tau},
        ):
            result = model({"image": encoded}, diagnostics=True)
        for name, target in model.parameter_map().items():
            if not torch.equal(target, weights[recipe.PARAMETERS[name]]):
                raise ValueError("frozen weights changed")
    return {
        name: value.cpu().numpy().astype(bool)
        for name, value in result.diagnostics.items()
    }


def image_bank(source):
    """Reuse only the explicitly selected raw image pixels, never old activity."""
    definition = source.record["execution"]["configuration"]
    if definition.get("schema") == "exp123.compute/v3":
        cfg = definition["sweep"]
    elif definition.get("schema") == "exp123.compute/v2":
        cfg = definition["protocols"]["benchmark"]
    else:
        raise ValueError("image source must contain the balanced exp123 benchmark")
    if (
        cfg["partition"] != "official_mnist_test"
        or cfg["image_selection_seed"] != recipe.ACCURACY_SELECTION_SEED
    ):
        raise ValueError("image source has an incompatible test selection")
    images = {}
    for index in cfg["image_indices"]:
        with np.load(
            source.unit(recipe.recording_unit(index, recipe.RATE_HZ, "benchmark"))
            / "input.npz",
            allow_pickle=False,
        ) as data:
            pixels, label = data["pixels"].copy(), int(data["label"])
        if (
            pixels.shape != (28, 28)
            or not np.isfinite(pixels).all()
            or pixels.min() < 0
            or pixels.max() > 1
            or not 0 <= label < 10
        ):
            raise ValueError("invalid normalized MNIST image")
        images[index] = (torch.from_numpy(pixels).reshape(1, 784).float(), label)
    if len(images) != 500 or not np.array_equal(
        np.bincount([v[1] for v in images.values()], minlength=10), [50] * 10
    ):
        raise ValueError("image source must have 50 distinct test images per digit")
    return images


def compute(identity, image_identity=None, *, run_id=None):
    torch.set_num_threads(1)
    bank = source_run(
        recipe.REPO / ".pingstore", identity, stage="compute", experiment="exp022"
    )
    unit = bank.unit(recipe.UNIT)
    record = (
        load_json(unit / "metrics.json")
        .get("checkpoints", {})
        .get(recipe.CHECKPOINT_ROLE)
    )
    filename = recipe.CHECKPOINT_FILE
    if not isinstance(record, dict) or record.get("filename") != filename:
        raise ValueError(
            f"Checkpoint role {recipe.CHECKPOINT_ROLE!r} does not name {filename!r}"
        )
    if Path(filename).name != filename:
        raise ValueError("Checkpoint filename must be a single path component")
    checkpoint = unit / filename
    if (
        record.get("sha256") != recipe.CHECKPOINT_SHA256
        or file_sha256(checkpoint) != recipe.CHECKPOINT_SHA256
    ):
        raise ValueError(f"Checkpoint identity mismatch: {checkpoint}")
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
    inputs = {"trained_network": bank}
    if image_identity is not None:
        images_source = source_run(
            recipe.REPO / ".pingstore",
            image_identity,
            stage="compute",
            experiment=recipe.SLUG,
        )
        images = image_bank(images_source)
        cfg = recipe.shared_configuration_for_images(sorted(images))
        inputs["images"] = images_source
    else:
        dataset = MNIST(str(recipe.REPO / "data"), train=False, download=False)
        cfg = recipe.shared_configuration(dataset.targets.numpy())
        images = {
            index: (
                dataset.data[index].reshape(1, 784).float() / 255,
                int(dataset.targets[index]),
            )
            for index in cfg["sweep"]["image_indices"]
        }
    with stage_run(
        recipe.REPO,
        recipe.SLUG,
        "compute",
        inputs=inputs,
        configuration=cfg,
        run_id=run_id,
    ) as run:
        run.record["execution"]["software"] = {
            "snnlab": version("snnlab"),
            "torch": torch.__version__,
        }
        run.record["scientific_artifacts"] = {"snapshot_stages": {}}
        recipe.author_network(cfg["sweep"]["tau_gaba_ms"][0], cfg["sweep"]).visualise(
            run.export / "network.svg", view="circuit"
        )
        hashes = {}
        for protocol, cfg in {"benchmark": cfg["sweep"]}.items():
            indices, seeds = cfg["image_indices"], cfg["seeds"]
            for rate in cfg["input_rates_hz"]:
                for start in range(0, len(indices), cfg["execution_batch_images"]):
                    batch_indices = indices[
                        start : start + cfg["execution_batch_images"]
                    ]
                    encodings = []
                    for image_index in batch_indices:
                        unit = run.export / recipe.recording_unit(
                            image_index, rate, protocol
                        )
                        unit.mkdir()
                        pixels, label = images[image_index]
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
                        hashes[f"{protocol}:{rate:g}:{image_index}"] = {
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
                    for tau in cfg["tau_gaba_ms"]:
                        execution_name = (
                            f"{protocol}--rate-{rate:g}--batch-{start}--tau-{tau:g}"
                        )
                        run.record["scientific_artifacts"]["snapshot_stages"][
                            execution_name
                        ] = ["start"]
                        signals = graph_trial(
                            tau,
                            weights,
                            batch_encoded,
                            run.export / execution_name,
                            cfg,
                            run.record["execution"],
                        )
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
                                label=images[image_index][1],
                                seeds=seeds,
                            )
                    print(
                        f"rate {rate:g} Hz: {start + len(batch_indices)}/{len(indices)} images completed "
                        f"({len(seeds) * len(cfg['tau_gaba_ms'])} paired presentations per image)",
                        flush=True,
                    )
        if recipe.sha256(checkpoint) != recipe.CHECKPOINT_SHA256:
            raise ValueError("checkpoint changed")
        run.record["execution"]["encoding_sha256"] = hashes
        run.record["execution"]["observations"] = {
            "presentations": run.record["execution"]["configuration"]["presentations"],
            "weights_frozen": True,
            "inputs_paired_across_tau": True,
        }
        result = run.run_id
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="identity reserved before remote dispatch")
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--image-source",
        help="Validated image bank, otherwise local official MNIST test set",
    )
    args = parser.parse_args()
    print(compute(args.source, args.image_source, run_id=args.run_id))
